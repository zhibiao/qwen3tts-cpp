#include "config.h"
#include "qwen3_tts.h"
#include "talker.h"
#include "test_assert.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int64_t kFrames = 2;
constexpr int64_t kHidden = 1024;
constexpr int64_t kVocab = 3072;

template <typename T>
std::vector<T> read_binary(const std::string& path, size_t expected_count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    CHECK(file.good());
    const std::streamsize size = file.tellg();
    CHECK_EQ_I64(size, static_cast<std::streamsize>(expected_count * sizeof(T)));
    file.seekg(0);
    std::vector<T> data(expected_count);
    file.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(T)));
    CHECK(file.good());
    return data;
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

float max_abs_error(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
    CHECK_EQ_I64(actual.size(), expected.size());
    float result = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        result = std::max(result, std::fabs(actual[i] - expected[i]));
    }
    return result;
}

int32_t argmax(const std::vector<float>& values) {
    return static_cast<int32_t>(
        std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

std::array<int32_t, 5> top5(const std::vector<float>& values) {
    std::array<int32_t, 5> result{};
    std::vector<int32_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(),
        indices.begin() + result.size(),
        indices.end(),
        [&](int32_t lhs, int32_t rhs) {
            return values[static_cast<size_t>(lhs)] >
                values[static_cast<size_t>(rhs)];
        });
    std::copy_n(indices.begin(), result.size(), result.begin());
    return result;
}

void check_outputs(
    const std::string& label,
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    int64_t channels = 0,
    float max_error_limit = 3.0f) {
    const double cosine = cosine_similarity(actual, expected);
    const float error = max_abs_error(actual, expected);
    std::cout << label << ": cosine=" << cosine
              << " max_abs_error=" << error << '\n';
    if (channels > 0) {
        for (int64_t frame = 0; frame * channels < static_cast<int64_t>(actual.size()); ++frame) {
            const auto begin = actual.begin() + frame * channels;
            const auto expected_begin = expected.begin() + frame * channels;
            const std::vector<float> frame_actual(begin, begin + channels);
            const std::vector<float> frame_expected(
                expected_begin,
                expected_begin + channels);
            std::cout << "  frame " << frame
                      << ": cosine=" << cosine_similarity(frame_actual, frame_expected)
                      << " max_abs_error=" << max_abs_error(frame_actual, frame_expected)
                      << '\n';
        }
    }
    CHECK(cosine >= 0.997);
    CHECK(error <= max_error_limit);
}

} // namespace

int main() {
    const std::string fixture_dir = QWEN3_TTS_TALKER_GOLDEN_DIR;
    const std::vector<float> input =
        read_binary<float>(fixture_dir + "/talker_input.bin", kFrames * kHidden);
    const std::vector<float> expected_hidden =
        read_binary<float>(fixture_dir + "/talker_hidden.bin", kFrames * kHidden);
    const std::vector<float> expected_logits =
        read_binary<float>(fixture_dir + "/talker_logits.bin", kFrames * kVocab);
    const std::vector<float> step_input =
        read_binary<float>(fixture_dir + "/talker_decode_step_input.bin", kHidden);
    const std::vector<float> expected_step_hidden =
        read_binary<float>(fixture_dir + "/talker_decode_step_hidden.bin", kHidden);
    const std::vector<float> expected_step_logits =
        read_binary<float>(fixture_dir + "/talker_decode_step_logits.bin", kVocab);
    const std::vector<int32_t> expected_c0 =
        read_binary<int32_t>(fixture_dir + "/talker_c0.bin", 1);

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
    qwen3_tts::Talker talker;
    CHECK_STATUS(talker.load_tensors(config.decoder, context));

    std::vector<float> hidden;
    std::vector<float> logits;
    CHECK_STATUS(talker.prefill(context, input, kFrames, hidden, logits));
    check_outputs("prefill logits", logits, expected_logits, kVocab);
    check_outputs("prefill hidden", hidden, expected_hidden, kHidden);
    const std::vector<float> last_logits(
        logits.end() - kVocab,
        logits.end());
    const std::vector<float> expected_last_logits(
        expected_logits.end() - kVocab,
        expected_logits.end());
    CHECK_EQ_I64(argmax(last_logits), expected_c0[0]);

    std::vector<float> step_hidden;
    std::vector<float> step_logits;
    CHECK_STATUS(talker.decode_step(
        context,
        step_input,
        step_hidden,
        step_logits));
    check_outputs("decode_step hidden", step_hidden, expected_step_hidden);
    check_outputs("decode_step logits", step_logits, expected_step_logits);

    return 0;
}
