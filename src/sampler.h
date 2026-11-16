#pragma once

#include "status.h"

#include <cstdint>
#include <random>
#include <vector>

namespace qwen3_tts {

struct SamplingOptions {
    bool do_sample = true;
    int32_t top_k = 50;
    float top_p = 1.0f;
    float temperature = 0.9f;
    float repetition_penalty = 1.05f;

    // 贪婪预设：不做随机采样，也不加重复惩罚。
    // 只显式关掉这两项，其余字段在 do_sample == false 时不会被读到。
    static SamplingOptions greedy();
};

class Sampler {
public:
    Sampler();

    void seed(uint64_t value);

    Status select(
        const std::vector<float>& logits,
        const SamplingOptions& options,
        const std::vector<int32_t>& history,
        const std::vector<uint8_t>& allowed_mask,
        int32_t& token);

private:
    std::mt19937_64 rng_;
};

} // namespace qwen3_tts
