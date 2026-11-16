#include "config.h"
#include "qwen3_tts.h"
#include "talker.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<float> read_all_f32(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    CHECK(file.good());
    const std::streamsize size = file.tellg();
    CHECK_EQ_I64(size % static_cast<std::streamsize>(sizeof(float)), 0);
    file.seekg(0);
    std::vector<float> values(static_cast<size_t>(size) / sizeof(float));
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
    const std::string fixture_dir = QWEN3_TTS_1_7_GOLDEN_DIR;
#if defined(QWEN3_TTS_EXPECT_BF16)
    const std::string input_path =
        fixture_dir + "/1.7b-reference-prefill.bin";
    const std::string expected_path =
        fixture_dir + "/1.7b-reference-prefill-hidden.bin";
#else
    const std::string input_path =
        fixture_dir + "/1.7b-reference-f32-prefill.bin";
    const std::string expected_path =
        fixture_dir + "/1.7b-reference-f32-prefill-hidden.bin";
#endif
    const std::vector<float> input =
        read_all_f32(input_path);
    const std::vector<float> expected =
        read_all_f32(expected_path);
    CHECK_EQ_I64(input.size(), expected.size());

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
    CHECK_EQ_I64(
        input.size() %
            static_cast<size_t>(config.decoder.embedding_length),
        0);
    const int64_t frames =
        static_cast<int64_t>(input.size() / config.decoder.embedding_length);

    qwen3_tts::Talker talker;
    CHECK_STATUS(talker.load_tensors(config.decoder, context));
    std::vector<float> actual;
    std::vector<float> logits;
    CHECK_STATUS(talker.prefill(context, input, frames, actual, logits));
    CHECK_EQ_I64(actual.size(), expected.size());

    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
    }
    const double cosine = cosine_similarity(actual, expected);
    std::cout << "Talker 1.7B prefill frames=" << frames
              << " cosine=" << cosine
              << " max_abs_error=" << max_error << '\n';
    for (int64_t frame = 0; frame < frames; ++frame) {
        const auto begin = actual.begin() +
            static_cast<std::ptrdiff_t>(
                frame * config.decoder.embedding_length);
        const auto end = begin + config.decoder.embedding_length;
        const std::vector<float> frame_actual(begin, end);
        const std::vector<float> frame_expected(
            expected.begin() +
                static_cast<std::ptrdiff_t>(
                    frame * config.decoder.embedding_length),
            expected.begin() +
                static_cast<std::ptrdiff_t>(
                    (frame + 1) * config.decoder.embedding_length));
        std::cout << "  frame " << frame
                  << " cosine="
                  << cosine_similarity(frame_actual, frame_expected)
                  << '\n';
    }
    CHECK(cosine >= 0.999f);
    CHECK(max_error <= 5e-1f);

    return 0;
}
