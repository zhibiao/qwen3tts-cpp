#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>

namespace qwen3_tts {

SamplingOptions SamplingOptions::greedy() {
    SamplingOptions options;
    options.do_sample = false;
    options.repetition_penalty = 1.0f;
    return options;
}

Sampler::Sampler()
    : rng_(std::random_device{}()) {
}

void Sampler::seed(uint64_t value) {
    rng_.seed(value);
}

Status Sampler::select(
    const std::vector<float>& logits,
    const SamplingOptions& options,
    const std::vector<int32_t>& history,
    const std::vector<uint8_t>& allowed_mask,
    int32_t& token) {
    if (logits.empty()) {
        return Status::error("sampler logits are empty");
    }
    if (!allowed_mask.empty() && allowed_mask.size() != logits.size()) {
        return Status::error("sampler mask size does not match logits");
    }
    if (options.repetition_penalty < 1.0f) {
        return Status::error("repetition penalty must be at least 1");
    }
    if (options.do_sample &&
        (!std::isfinite(options.temperature) || options.temperature <= 0.0f)) {
        return Status::error("sampling temperature must be positive");
    }

    std::vector<float> values = logits;
    if (options.repetition_penalty > 1.0f) {
        for (int32_t previous : history) {
            if (previous < 0 || previous >= static_cast<int32_t>(values.size())) {
                continue;
            }
            float& value = values[static_cast<size_t>(previous)];
            value = value > 0.0f
                ? value / options.repetition_penalty
                : value * options.repetition_penalty;
        }
    }

    for (size_t i = 0; i < values.size(); ++i) {
        if ((!allowed_mask.empty() && allowed_mask[i] == 0) ||
            !std::isfinite(values[i])) {
            values[i] = -std::numeric_limits<float>::infinity();
        }
    }

    if (!options.do_sample) {
        const auto best = std::max_element(values.begin(), values.end());
        if (best == values.end() || !std::isfinite(*best)) {
            return Status::error("sampler has no allowed token");
        }
        token = static_cast<int32_t>(std::distance(values.begin(), best));
        return Status::ok();
    }

    for (float& value : values) {
        if (std::isfinite(value)) {
            value /= options.temperature;
        }
    }

    std::vector<int32_t> indices;
    indices.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        if (std::isfinite(values[i])) {
            indices.push_back(static_cast<int32_t>(i));
        }
    }
    if (indices.empty()) {
        return Status::error("sampler has no allowed token");
    }

    const size_t top_k = options.top_k > 0
        ? std::min<size_t>(static_cast<size_t>(options.top_k), indices.size())
        : indices.size();

    // 后续 softmax / top_p 只用到前 top_k 个，没必要给全部候选排序：
    // 先用 O(n) 的 nth_element 把前 top_k 个选出来，再只排这 top_k 个。
    // 平局按小下标优先，使结果与输入顺序无关、可复现。
    const auto by_value_desc = [&](int32_t lhs, int32_t rhs) {
        const float left = values[static_cast<size_t>(lhs)];
        const float right = values[static_cast<size_t>(rhs)];
        if (left != right) {
            return left > right;
        }
        return lhs < rhs;
    };
    std::nth_element(
        indices.begin(),
        indices.begin() + static_cast<std::ptrdiff_t>(top_k - 1),
        indices.end(),
        by_value_desc);
    indices.resize(top_k);
    std::sort(indices.begin(), indices.end(), by_value_desc);

    const float max_value = values[static_cast<size_t>(indices[0])];
    double sum = 0.0;
    std::vector<double> weights(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        weights[i] = std::exp(
            static_cast<double>(
                values[static_cast<size_t>(indices[i])] - max_value));
        sum += weights[i];
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        return Status::error("sampler softmax normalization failed");
    }
    for (double& weight : weights) {
        weight /= sum;
    }

    const float top_p = std::clamp(options.top_p, 0.0f, 1.0f);
    if (top_p < 1.0f) {
        double cumulative = 0.0;
        size_t keep = 0;
        for (; keep < weights.size(); ++keep) {
            cumulative += weights[keep];
            if (cumulative >= top_p) {
                ++keep;
                break;
            }
        }
        keep = std::max<size_t>(keep, 1);
        indices.resize(keep);
        weights.resize(keep);
    }

    std::discrete_distribution<size_t> distribution(
        weights.begin(),
        weights.end());
    token = indices[distribution(rng_)];
    return Status::ok();
}

} // namespace qwen3_tts
