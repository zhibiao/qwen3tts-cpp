#include "code_predictor.h"
#include "config.h"
#include "qwen3_tts.h"
#include "test_assert.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_binary(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    CHECK(file.good());
    CHECK_EQ_I64(file.tellg(), static_cast<std::streamsize>(count * sizeof(T)));
    file.seekg(0);
    std::vector<T> values(count);
    file.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
    CHECK(file.good());
    return values;
}

} // namespace

int main() {
    constexpr int32_t kFrames = 2;
    const std::string fixture_dir = QWEN3_TTS_1_7_GOLDEN_DIR;

    qwen3_tts::Qwen3TTSContext context;
    qwen3_tts::Qwen3TTSContextParams params;
    params.n_threads = 1;
    CHECK_STATUS(context.load(
        QWEN3_TTS_TEST_GGUF_F32,
        QWEN3_TTS_TEST_SPEECH_GGUF,
        params));

    qwen3_tts::TalkerGGUFConfig config;
    CHECK_STATUS(qwen3_tts::load_talker_config(
        context.talker_gguf(),
        config));
    CHECK_EQ_I64(config.decoder.embedding_length, 2048);
    CHECK_EQ_I64(config.code_predictor.hidden_size, 1024);
    CHECK(config.code_predictor.has_input_projection);

    const size_t hidden_size =
        static_cast<size_t>(config.code_predictor.input_embedding_size);
    const size_t groups =
        static_cast<size_t>(config.code_predictor.num_code_groups);
#if defined(QWEN3_TTS_EXPECT_BF16)
    const std::string hidden_path =
        fixture_dir + "/1.7b-reference-3-hidden.bin";
    const std::string codes_path =
        fixture_dir + "/1.7b-reference-3-codes.bin";
    const std::string logits_path =
        fixture_dir + "/1.7b-reference-3-logits.bin";
#else
    const std::string hidden_path =
        fixture_dir + "/1.7b-reference-f32-hidden.bin";
    const std::string codes_path =
        fixture_dir + "/1.7b-reference-f32-codes.bin";
    const std::string logits_path;
#endif
    const std::vector<float> hidden = read_binary<float>(
        hidden_path,
        kFrames * hidden_size);
    const std::vector<int32_t> expected = read_binary<int32_t>(
        codes_path,
        kFrames * groups);
    std::vector<float> expected_logits;
    if (!logits_path.empty()) {
        expected_logits = read_binary<float>(
            logits_path,
            static_cast<size_t>(kFrames) * (groups - 1) *
                config.code_predictor.vocab_size);
    }

    qwen3_tts::CodePredictor predictor;
    CHECK_STATUS(predictor.load(config.code_predictor, context));
    CHECK_STATUS(predictor.init_graphs(context));
    for (int32_t frame = 0; frame < kFrames; ++frame) {
        const std::vector<float> frame_hidden(
            hidden.begin() + static_cast<std::ptrdiff_t>(frame * hidden_size),
            hidden.begin() + static_cast<std::ptrdiff_t>((frame + 1) * hidden_size));
        std::vector<int32_t> actual;
        std::vector<float> actual_logits;
        CHECK_STATUS(predictor.predict_frame(
            context,
            frame_hidden,
            expected[static_cast<size_t>(frame) * groups],
            actual,
            qwen3_tts::SamplingOptions::greedy(),
            &actual_logits));
        if (!expected_logits.empty()) {
            const size_t step_size =
                static_cast<size_t>(config.code_predictor.vocab_size);
            const float* expected_frame =
                expected_logits.data() +
                static_cast<size_t>(frame) * (groups - 1) * step_size;
            float max_error = 0.0f;
            for (size_t i = 0; i < actual_logits.size(); ++i) {
                max_error = std::max(
                    max_error,
                    std::fabs(actual_logits[i] - expected_frame[i]));
            }
            std::cout << "frame " << frame
                      << " logits max_abs_error=" << max_error << '\n';
        }
        for (size_t group = 0; group < groups; ++group) {
            const int32_t value =
                expected[static_cast<size_t>(frame) * groups + group];
            if (actual[group] != value) {
                std::cout << "frame " << frame << " group " << group
                          << ": actual=" << actual[group]
                          << " expected=" << value << '\n';
            }
        }
        CHECK(actual == std::vector<int32_t>(
            expected.begin() + static_cast<std::ptrdiff_t>(frame * groups),
            expected.begin() + static_cast<std::ptrdiff_t>((frame + 1) * groups)));
    }

    return 0;
}
