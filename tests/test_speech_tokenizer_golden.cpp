#include "qwen3_tts.h"
#include "speech_tokenizer.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_all_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    CHECK(file.good());
    const std::streamsize size = file.tellg();
    CHECK(size >= 0);
    CHECK_EQ_I64(size % static_cast<std::streamsize>(sizeof(T)), 0);
    file.seekg(0);
    std::vector<T> values(static_cast<size_t>(size) / sizeof(T));
    file.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(size));
    CHECK(file.good());
    return values;
}

double cosine_similarity(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
    CHECK_EQ_I64(actual.size(), expected.size());
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        dot += static_cast<double>(actual[i]) * expected[i];
        actual_norm += static_cast<double>(actual[i]) * actual[i];
        expected_norm += static_cast<double>(expected[i]) * expected[i];
    }
    return dot / std::sqrt(actual_norm * expected_norm);
}

} // namespace

int main() {
    const std::string fixture_dir = QWEN3_TTS_CUSTOMVOICE_GOLDEN_DIR;
    const std::vector<int32_t> code_values =
        read_all_binary<int32_t>(fixture_dir + "/custom_voice_codes.bin");
    const std::vector<float> expected =
        read_all_binary<float>(fixture_dir + "/custom_voice_audio.bin");
    CHECK_EQ_I64(code_values.size() % 16, 0);

    qwen3_tts::SpeechCodeMatrix codes;
    codes.groups = 16;
    codes.frames = static_cast<int32_t>(code_values.size() / codes.groups);
    codes.data = code_values;
    CHECK(codes.frames > 0);

    qwen3_tts::Qwen3TTSContext context;
    qwen3_tts::Qwen3TTSContextParams params;
    params.device = "CPU";
    params.n_threads = 1;
    CHECK_STATUS(context.load(
        QWEN3_TTS_TEST_GGUF,
        QWEN3_TTS_TEST_SPEECH_GGUF,
        params));

    qwen3_tts::SpeechTokenizerGGUFConfig config;
    CHECK_STATUS(qwen3_tts::load_speech_config(
        context.speech_gguf(),
        config));
    qwen3_tts::SpeechTokenizer tokenizer;
    CHECK_STATUS(tokenizer.load(config.tokenizer, context));

    qwen3_tts::AudioBuffer audio;
    CHECK_STATUS(tokenizer.decode(context, codes, audio));
    CHECK_EQ_I64(audio.sample_rate, 24000);
    CHECK_EQ_I64(audio.channels, 1);
    CHECK_EQ_I64(audio.samples.size(), expected.size());

    float max_error = 0.0f;
    for (size_t i = 0; i < audio.samples.size(); ++i) {
        max_error = std::max(
            max_error,
            std::fabs(audio.samples[i] - expected[i]));
    }
    const double cosine = cosine_similarity(audio.samples, expected);
    std::cout << "SpeechTokenizer golden: frames=" << codes.frames
              << " cosine=" << cosine
              << " max_abs_error=" << max_error << '\n';
    CHECK(cosine >= 0.999);
    CHECK(max_error <= 1e-2f);
    return 0;
}
