#include "qwen3_tts.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<float> read_binary(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    CHECK(file.good());
    CHECK_EQ_I64(
        file.tellg(),
        static_cast<std::streamsize>(count * sizeof(float)));
    file.seekg(0);
    std::vector<float> values(count);
    file.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    CHECK(file.good());
    return values;
}

double cosine_similarity(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
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
    constexpr size_t hidden = 1024;
    constexpr size_t prefill_rows = 15;
    const std::string fixture_dir = QWEN3_TTS_CUSTOMVOICE_GOLDEN_DIR;
    const std::vector<float> expected_prefill = read_binary(
        fixture_dir + "/custom_voice_prefill.bin",
        prefill_rows * hidden);
    const std::vector<float> expected_pad = read_binary(
        fixture_dir + "/custom_voice_tts_pad.bin",
        hidden);

    qwen3_tts::Qwen3TTSContext context;
    qwen3_tts::Qwen3TTSContextParams params;
    params.device = "CPU";
    params.n_threads = 1;
    CHECK_STATUS(context.load(
        QWEN3_TTS_TEST_GGUF,
        QWEN3_TTS_TEST_SPEECH_GGUF,
        params));
    qwen3_tts::Qwen3TTSModel model;
    CHECK_STATUS(model.load(context));

    qwen3_tts::CustomVoiceOptions voice;
    voice.speaker = "serena";
    voice.language = "Chinese";
    std::vector<float> actual_prefill;
    std::vector<float> actual_pad;
    CHECK_STATUS(model.build_custom_voice_prompt(
        "你好，世界。",
        voice,
        actual_prefill,
        actual_pad));
    CHECK_EQ_I64(actual_prefill.size(), expected_prefill.size());
    CHECK_EQ_I64(actual_pad.size(), expected_pad.size());

    float max_prefill_error = 0.0f;
    for (size_t i = 0; i < actual_prefill.size(); ++i) {
        max_prefill_error = std::max(
            max_prefill_error,
            std::fabs(actual_prefill[i] - expected_prefill[i]));
    }
    const double prefill_cosine =
        cosine_similarity(actual_prefill, expected_prefill);
    const double pad_cosine = cosine_similarity(actual_pad, expected_pad);
    std::cout << "CustomVoice prompt golden: prefill_cosine="
              << prefill_cosine
              << " max_abs_error=" << max_prefill_error
              << " pad_cosine=" << pad_cosine << '\n';
    CHECK(prefill_cosine >= 0.999999);
    CHECK(max_prefill_error <= 2e-4f);
    CHECK(pad_cosine >= 0.999999);

    voice.speaker = "not-a-speaker";
    CHECK(!model.build_custom_voice_prompt(
        "你好。",
        voice,
        actual_prefill,
        actual_pad));
    voice.speaker = "serena";
    voice.language = "not-a-language";
    CHECK(!model.build_custom_voice_prompt(
        "你好。",
        voice,
        actual_prefill,
        actual_pad));
    return 0;
}
