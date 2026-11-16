#include "wav_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>

namespace qwen3_tts {

namespace {

void append_u16(std::vector<uint8_t>& output, uint16_t value) {
    output.push_back(static_cast<uint8_t>(value));
    output.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>& output, uint32_t value) {
    output.push_back(static_cast<uint8_t>(value));
    output.push_back(static_cast<uint8_t>(value >> 8));
    output.push_back(static_cast<uint8_t>(value >> 16));
    output.push_back(static_cast<uint8_t>(value >> 24));
}

} // namespace

Status write_wav(
    const std::string& path,
    const AudioBuffer& audio) {
    if (path.empty()) {
        return Status::error("WAV output path is empty");
    }
    if (audio.samples.empty()) {
        return Status::error("WAV output has no samples");
    }
    if (audio.sample_rate <= 0) {
        return Status::error("WAV sample rate must be positive");
    }
    if (audio.channels <= 0 || audio.channels > 2) {
        return Status::error("WAV writer supports mono or stereo");
    }
    if (audio.samples.size() >
        (std::numeric_limits<uint32_t>::max() - 36) / 2) {
        return Status::error("WAV output is too large");
    }

    const uint16_t channels = static_cast<uint16_t>(audio.channels);
    const uint32_t data_size =
        static_cast<uint32_t>(audio.samples.size() * sizeof(int16_t));
    const uint16_t block_align = static_cast<uint16_t>(channels * 2);
    const uint32_t byte_rate =
        static_cast<uint32_t>(audio.sample_rate) * block_align;

    std::vector<uint8_t> header;
    header.reserve(44);
    header.insert(header.end(), {'R', 'I', 'F', 'F'});
    append_u32(header, 36 + data_size);
    header.insert(header.end(), {'W', 'A', 'V', 'E'});
    header.insert(header.end(), {'f', 'm', 't', ' '});
    append_u32(header, 16);
    append_u16(header, 1);
    append_u16(header, channels);
    append_u32(header, static_cast<uint32_t>(audio.sample_rate));
    append_u32(header, byte_rate);
    append_u16(header, block_align);
    append_u16(header, 16);
    header.insert(header.end(), {'d', 'a', 't', 'a'});
    append_u32(header, data_size);

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        return Status::error("failed to open WAV output: " + path);
    }
    file.write(
        reinterpret_cast<const char*>(header.data()),
        static_cast<std::streamsize>(header.size()));

    for (float sample : audio.samples) {
        if (!std::isfinite(sample)) {
            return Status::error("WAV samples contain non-finite values");
        }
        sample = std::clamp(sample, -1.0f, 1.0f);
        const int16_t pcm = sample >= 0.0f
            ? static_cast<int16_t>(sample * 32767.0f)
            : static_cast<int16_t>(sample * 32768.0f);
        const uint16_t raw = static_cast<uint16_t>(pcm);
        const char bytes[2] = {
            static_cast<char>(raw),
            static_cast<char>(raw >> 8),
        };
        file.write(bytes, 2);
    }
    if (!file) {
        return Status::error("failed to write WAV output: " + path);
    }
    return Status::ok();
}

} // namespace qwen3_tts
