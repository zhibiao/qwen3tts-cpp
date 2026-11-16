#pragma once

#include <cstdint>
#include <vector>

namespace qwen3_tts {

struct AudioBuffer {
    std::vector<float> samples;
    int32_t sample_rate = 24000;
    int32_t channels = 1;
};

struct SpeechCodeMatrix {
    int32_t frames = 0;
    int32_t groups = 16;
    std::vector<int32_t> data;
};

} // namespace qwen3_tts
