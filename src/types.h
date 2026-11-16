#pragma once

#include <cstdint>
#include <vector>

namespace qwen3_tts {

struct AudioBuffer {
    std::vector<float> samples;
    int32_t sample_rate = 24000;
    int32_t channels = 1;

    // samples 是交错存储，多声道要除以 channels
    double duration_seconds() const {
        if (samples.empty() || sample_rate <= 0 || channels <= 0) {
            return 0.0;
        }
        return static_cast<double>(samples.size()) /
            (static_cast<double>(sample_rate) *
             static_cast<double>(channels));
    }
};

struct SpeechCodeMatrix {
    int32_t frames = 0;
    int32_t groups = 16;
    std::vector<int32_t> data;
};

} // namespace qwen3_tts
