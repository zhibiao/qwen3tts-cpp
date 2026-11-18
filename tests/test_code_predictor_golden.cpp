#include "config.h"
#include "code_predictor.h"
#include "qwen3_tts.h"
#include "test_assert.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr size_t kHidden = 1024;
constexpr size_t kGroups = 16;
constexpr size_t kCodebookSize = 2048;

template <typename T>
std::vector<T> read_binary(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    CHECK(file.good());
    CHECK_EQ_I64(file.tellg(), static_cast<std::streamsize>(count * sizeof(T)));
    file.seekg(0);
    std::vector<T> data(count);
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

} // namespace

int main() {
    const std::string fixture_dir = QWEN3_TTS_CODE_PREDICTOR_GOLDEN_DIR;
    const std::vector<float> talker_hidden =
        read_binary<float>(fixture_dir + "/code_predictor_talker_hidden.bin", kHidden);
    const std::vector<int32_t> expected_frame =
        read_binary<int32_t>(fixture_dir + "/code_predictor_frame.bin", kGroups);
    const std::vector<float> expected_logits =
        read_binary<float>(
            fixture_dir + "/code_predictor_logits.bin",
            (kGroups - 1) * kCodebookSize);
    const std::vector<float> expected_embedding =
        read_binary<float>(
            fixture_dir + "/code_predictor_frame_embedding.bin",
            kHidden);

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
    qwen3_tts::CodePredictor predictor;
    CHECK_STATUS(predictor.load(config.code_predictor, context));
    CHECK_STATUS(predictor.init_graphs(context));

    std::vector<int32_t> actual_frame;
    std::vector<float> actual_logits;
    CHECK_STATUS(predictor.predict_frame(
        context,
        talker_hidden,
        expected_frame[0],
        actual_frame,
        qwen3_tts::SamplingOptions::greedy(),
        &actual_logits));
    CHECK_EQ_I64(actual_frame.size(), kGroups);
    size_t matching_codes = 0;
    for (size_t i = 0; i < kGroups; ++i) {
        std::cout << "code[" << i << "]: actual=" << actual_frame[i]
                  << " expected=" << expected_frame[i] << '\n';
        matching_codes += actual_frame[i] == expected_frame[i] ? 1 : 0;
    }
    CHECK(matching_codes + 1 >= kGroups);
    float max_logits_error = 0.0f;
    for (size_t i = 0; i < actual_logits.size(); ++i) {
        max_logits_error = std::max(
            max_logits_error,
            std::fabs(actual_logits[i] - expected_logits[i]));
    }
    std::cout << "CodePredictor logits: cosine="
              << cosine_similarity(actual_logits, expected_logits)
              << " max_abs_error=" << max_logits_error << '\n';
    CHECK(cosine_similarity(actual_logits, expected_logits) >= 0.9999);
    CHECK(max_logits_error <= 2.0e-1f);

    std::vector<float> actual_embedding;
    CHECK_STATUS(predictor.embed_codes(
        context,
        expected_frame,
        actual_embedding));
    CHECK_EQ_I64(actual_embedding.size(), kHidden);
    float max_embedding_error = 0.0f;
    for (size_t i = 0; i < actual_embedding.size(); ++i) {
        max_embedding_error = std::max(
            max_embedding_error,
            std::fabs(actual_embedding[i] - expected_embedding[i]));
    }
    std::cout << "CodePredictor frame embedding: cosine="
              << cosine_similarity(actual_embedding, expected_embedding)
              << " max_abs_error=" << max_embedding_error << '\n';
    CHECK(cosine_similarity(actual_embedding, expected_embedding) >= 0.9999);
    CHECK(max_embedding_error <= 1e-3f);

    std::vector<int32_t> repeated_frame;
    CHECK_STATUS(predictor.predict_frame(
        context,
        talker_hidden,
        expected_frame[0],
        repeated_frame));
    CHECK(actual_frame == repeated_frame);

    std::cout << "CodePredictor frame matches PyTorch golden\n";
    return 0;
}
