#include "code_predictor.h"
#include "qwen3_tts.h"
#include "talker.h"
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
std::vector<T> read_binary(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    CHECK(file.good());
    CHECK_EQ_I64(
        file.tellg(),
        static_cast<std::streamsize>(count * sizeof(T)));
    file.seekg(0);
    std::vector<T> values(count);
    file.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
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

float max_abs_error(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
    CHECK_EQ_I64(actual.size(), expected.size());
    float result = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        result = std::max(
            result,
            std::fabs(actual[i] - expected[i]));
    }
    return result;
}

void check_values(
    const std::string& label,
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    double min_cosine,
    float max_error) {
    const double cosine = cosine_similarity(actual, expected);
    const float error = max_abs_error(actual, expected);
    std::cout << label << ": cosine=" << cosine
              << " max_abs_error=" << error << '\n';
    CHECK(cosine >= min_cosine);
    CHECK(error <= max_error);
}

} // namespace

int main() {
    qwen3_tts::Qwen3TTSContext context;
    qwen3_tts::Qwen3TTSContextParams params;
    params.device = "CUDA0";
    params.n_threads = 1;
    const qwen3_tts::Status load_status = context.load(
        QWEN3_TTS_TEST_GGUF,
        QWEN3_TTS_TEST_SPEECH_GGUF,
        params);
    if (!load_status) {
        std::cout << "CUDA unavailable: " << load_status.message() << '\n';
        return 77;
    }

    qwen3_tts::TalkerGGUFConfig config;
    CHECK_STATUS(qwen3_tts::load_talker_config(
        context.talker_gguf(),
        config));

    constexpr size_t hidden = 1024;
    constexpr size_t vocab = 3072;
    const std::string talker_dir = QWEN3_TTS_TALKER_GOLDEN_DIR;
    const std::vector<float> talker_input =
        read_binary<float>(talker_dir + "/talker_input.bin", 2 * hidden);
    const std::vector<float> expected_hidden =
        read_binary<float>(talker_dir + "/talker_hidden.bin", 2 * hidden);
    const std::vector<float> expected_logits =
        read_binary<float>(talker_dir + "/talker_logits.bin", 2 * vocab);
    const std::vector<float> step_input =
        read_binary<float>(
            talker_dir + "/talker_decode_step_input.bin",
            hidden);
    const std::vector<float> expected_step_hidden =
        read_binary<float>(
            talker_dir + "/talker_decode_step_hidden.bin",
            hidden);
    const std::vector<float> expected_step_logits =
        read_binary<float>(
            talker_dir + "/talker_decode_step_logits.bin",
            vocab);

    qwen3_tts::Talker talker;
    CHECK_STATUS(talker.load_tensors(config.decoder, context));
    std::vector<float> hidden_out;
    std::vector<float> logits_out;
    CHECK_STATUS(talker.prefill(
        context,
        talker_input,
        2,
        hidden_out,
        logits_out));
    check_values(
        "CUDA Talker prefill hidden",
        hidden_out,
        expected_hidden,
        0.997,
        3.0f);
    check_values(
        "CUDA Talker prefill logits",
        logits_out,
        expected_logits,
        0.997,
        3.0f);

    CHECK_STATUS(talker.decode_step(
        context,
        step_input,
        hidden_out,
        logits_out));
    check_values(
        "CUDA Talker decode hidden",
        hidden_out,
        expected_step_hidden,
        0.997,
        3.0f);
    check_values(
        "CUDA Talker decode logits",
        logits_out,
        expected_step_logits,
        0.997,
        3.0f);

    const std::string predictor_dir = QWEN3_TTS_CODE_PREDICTOR_GOLDEN_DIR;
    const std::vector<float> predictor_input =
        read_binary<float>(
            predictor_dir + "/code_predictor_talker_hidden.bin",
            hidden);
    const std::vector<int32_t> expected_codes =
        read_binary<int32_t>(
            predictor_dir + "/code_predictor_frame.bin",
            16);
    const std::vector<float> expected_predictor_logits =
        read_binary<float>(
            predictor_dir + "/code_predictor_logits.bin",
            15 * 2048);

    qwen3_tts::CodePredictor predictor;
    CHECK_STATUS(predictor.load(config.code_predictor, context));
    CHECK_STATUS(predictor.init_graphs(context));
    std::vector<int32_t> actual_codes;
    std::vector<float> actual_predictor_logits;
    CHECK_STATUS(predictor.predict_frame(
        context,
        predictor_input,
        expected_codes[0],
        actual_codes,
        qwen3_tts::SamplingOptions::greedy(),
        &actual_predictor_logits));
    size_t matching = 0;
    for (size_t i = 0; i < actual_codes.size(); ++i) {
        matching += actual_codes[i] == expected_codes[i] ? 1 : 0;
    }
    std::cout << "CUDA CodePredictor codes=" << matching << "/16\n";
    CHECK(matching + 1 >= 16);
    check_values(
        "CUDA CodePredictor logits",
        actual_predictor_logits,
        expected_predictor_logits,
        0.9999,
        2.0e-1f);
    return 0;
}
