#include "sampler.h"
#include "test_assert.h"

#include <cstdint>
#include <vector>

int main() {
    qwen3_tts::Sampler sampler;
    sampler.seed(1234);

    std::vector<float> logits = {1.0f, 8.0f, 3.0f, 2.0f};
    std::vector<uint8_t> allowed = {1, 0, 1, 1};
    qwen3_tts::SamplingOptions options;
    options.do_sample = false;

    int32_t token = -1;
    CHECK_STATUS(sampler.select(logits, options, {}, allowed, token));
    CHECK_EQ_I64(token, 2);

    std::vector<float> repeated = {1.0f, 5.0f, 4.0f};
    options.repetition_penalty = 2.0f;
    CHECK_STATUS(sampler.select(repeated, options, {1}, {}, token));
    CHECK_EQ_I64(token, 2);

    options.do_sample = true;
    options.top_k = 2;
    options.top_p = 1.0f;
    options.temperature = 1.0f;
    options.repetition_penalty = 1.0f;
    sampler.seed(42);
    CHECK_STATUS(sampler.select(logits, options, {}, allowed, token));
    CHECK(token == 0 || token == 2 || token == 3);

    sampler.seed(42);
    int32_t first = -1;
    int32_t second = -1;
    CHECK_STATUS(sampler.select(logits, options, {}, allowed, first));
    sampler.seed(42);
    CHECK_STATUS(sampler.select(logits, options, {}, allowed, second));
    CHECK_EQ_I64(first, second);

    CHECK(!sampler.select(logits, options, {}, {0, 0, 0, 0}, token));
    return 0;
}
