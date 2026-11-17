#include "sampler.h"
#include "test_assert.h"

#include <cstdint>
#include <limits>
#include <vector>

int main() {
    qwen3_tts::Sampler sampler;
    sampler.seed(1234);
    int32_t token = -1;

    std::vector<float> tie_logits = {1.0f, 3.0f, 3.0f, 2.0f};
    CHECK_STATUS(sampler.select(
        tie_logits,
        qwen3_tts::SamplingOptions::greedy(),
        {},
        {},
        token));
    CHECK_EQ_I64(token, 1);

    CHECK(!sampler.select(
        {},
        qwen3_tts::SamplingOptions::greedy(),
        {},
        {},
        token));
    CHECK(!sampler.select(
        tie_logits,
        qwen3_tts::SamplingOptions::greedy(),
        {},
        {1, 1},
        token));
    CHECK(!sampler.select(
        {std::numeric_limits<float>::quiet_NaN(),
         std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity()},
        qwen3_tts::SamplingOptions::greedy(),
        {},
        {},
        token));

    qwen3_tts::SamplingOptions invalid;
    invalid.do_sample = false;
    invalid.repetition_penalty = 0.5f;
    CHECK(!sampler.select(tie_logits, invalid, {}, {}, token));
    invalid.repetition_penalty = 1.0f;
    invalid.do_sample = true;
    invalid.temperature = 0.0f;
    CHECK(!sampler.select(tie_logits, invalid, {}, {}, token));
    invalid.temperature = 1.0f;

    invalid.top_k = 1;
    invalid.top_p = 0.0f;
    CHECK_STATUS(sampler.select(tie_logits, invalid, {}, {}, token));
    CHECK(token == 1 || token == 2);

    std::vector<float> repetition = {8.0f, 7.0f, 6.0f};
    qwen3_tts::SamplingOptions penalty =
        qwen3_tts::SamplingOptions::greedy();
    penalty.repetition_penalty = 100.0f;
    CHECK_STATUS(sampler.select(repetition, penalty, {0}, {}, token));
    CHECK_EQ_I64(token, 1);

    std::vector<float> negative_repetition = {-8.0f, -7.0f, -6.0f};
    CHECK_STATUS(sampler.select(
        negative_repetition,
        penalty,
        {2},
        {},
        token));
    CHECK_EQ_I64(token, 1);

    std::vector<uint8_t> only_one = {0, 0, 1};
    CHECK_STATUS(sampler.select(
        {100.0f, 50.0f, -10.0f},
        qwen3_tts::SamplingOptions::greedy(),
        {},
        only_one,
        token));
    CHECK_EQ_I64(token, 2);
    return 0;
}
