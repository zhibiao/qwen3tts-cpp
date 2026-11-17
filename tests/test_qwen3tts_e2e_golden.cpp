#include "qwen3_tts.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

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
    std::ifstream manifest_file(QWEN3_TTS_E2E_GOLDEN_MANIFEST);
    CHECK(manifest_file.good());
    nlohmann::json manifest;
    manifest_file >> manifest;
    CHECK_EQ_STR(
        manifest.at("format").get<std::string>(),
        "qwen3-tts-end-to-end-golden-v1");

    const std::string golden_dir = QWEN3_TTS_E2E_GOLDEN_DIR;
    int passed = 0;
    const auto run_model = [&](const char* model_path, const char* marker) {
        qwen3_tts::Qwen3TTSContext context;
        qwen3_tts::Qwen3TTSContextParams params;
        params.device = "CPU";
        params.n_threads = 1;
        CHECK_STATUS(context.load(
            model_path,
            QWEN3_TTS_TEST_SPEECH_GGUF,
            params));
        qwen3_tts::Qwen3TTSModel model;
        CHECK_STATUS(model.load(context));

        for (const auto& item : manifest.at("cases")) {
            const std::string codes_file =
                item.at("codes_file").get<std::string>();
            if (codes_file.find(marker) == std::string::npos) {
                continue;
            }
            const std::string name = item.at("name").get<std::string>();
            const std::string text = item.at("text").get<std::string>();
            const std::vector<int32_t> expected_codes =
                read_all_binary<int32_t>(golden_dir + "/" + codes_file);
            const std::vector<float> expected_audio =
                read_all_binary<float>(
                    golden_dir + "/" +
                    item.at("audio_file").get<std::string>());
            CHECK_EQ_I64(expected_codes.size() % 16, 0);
            const int64_t expected_frames =
                static_cast<int64_t>(expected_codes.size() / 16);

            qwen3_tts::CustomVoiceOptions voice;
            voice.speaker = "serena";
            voice.language = "Chinese";
            qwen3_tts::CodeGenerationOptions options;
            options.max_new_tokens = static_cast<int32_t>(expected_frames);
            options.talker = qwen3_tts::SamplingOptions::greedy();
            options.code_predictor = qwen3_tts::SamplingOptions::greedy();

            qwen3_tts::SpeechCodeMatrix actual_codes;
            CHECK_STATUS(model.generate_custom_voice_codes(
                text,
                voice,
                options,
                actual_codes));
            CHECK_EQ_I64(actual_codes.frames, expected_frames);
            CHECK_EQ_I64(actual_codes.groups, 16);

            int32_t matching_codes = 0;
            for (size_t i = 0; i < expected_codes.size(); ++i) {
                matching_codes +=
                    actual_codes.data[i] == expected_codes[i] ? 1 : 0;
            }
            std::cout << marker << ' ' << name
                      << " codes=" << matching_codes << '/'
                      << expected_codes.size() << '\n';
            CHECK(
                matching_codes + 1 >=
                static_cast<int32_t>(expected_codes.size()));

            qwen3_tts::AudioBuffer actual_audio;
            CHECK_STATUS(model.decode_codes(actual_codes, actual_audio));
            CHECK_EQ_I64(actual_audio.sample_rate, 24000);
            CHECK_EQ_I64(actual_audio.channels, 1);
            CHECK_EQ_I64(actual_audio.samples.size(), expected_audio.size());
            float max_audio_error = 0.0f;
            for (size_t i = 0; i < actual_audio.samples.size(); ++i) {
                max_audio_error = std::max(
                    max_audio_error,
                    std::fabs(
                        actual_audio.samples[i] - expected_audio[i]));
            }
            const double audio_cosine =
                cosine_similarity(actual_audio.samples, expected_audio);
            std::cout << marker << ' ' << name
                      << " audio_cosine=" << audio_cosine
                      << " max_abs_error=" << max_audio_error << '\n';
            CHECK(max_audio_error <= 1e-4f);
            if (actual_audio.duration_seconds() >= 0.5) {
                CHECK(audio_cosine >= 0.999);
            }
            ++passed;
        }
    };

    run_model(QWEN3_TTS_TEST_GGUF, "0.6B");
#if defined(QWEN3_TTS_TEST_GGUF_1_7)
    run_model(QWEN3_TTS_TEST_GGUF_1_7, "1.7B");
#endif

    CHECK(passed >= 6);
    std::cout << "Qwen3TTS end-to-end golden cases passed: "
              << passed << '\n';
    return 0;
}
