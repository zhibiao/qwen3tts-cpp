#include "test_assert.h"
#include "wav_writer.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>

namespace {

uint16_t read_u16(const char* value) {
    return static_cast<uint16_t>(
        static_cast<uint8_t>(value[0]) |
        (static_cast<uint16_t>(static_cast<uint8_t>(value[1])) << 8));
}

uint32_t read_u32(const char* value) {
    return static_cast<uint32_t>(static_cast<uint8_t>(value[0])) |
        (static_cast<uint32_t>(static_cast<uint8_t>(value[1])) << 8) |
        (static_cast<uint32_t>(static_cast<uint8_t>(value[2])) << 16) |
        (static_cast<uint32_t>(static_cast<uint8_t>(value[3])) << 24);
}

} // namespace

int main() {
    const std::string path = "wav_writer_extended_test.wav";
    qwen3_tts::AudioBuffer audio;
    audio.sample_rate = 24000;
    audio.channels = 1;
    audio.samples = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    CHECK_STATUS(qwen3_tts::write_wav(path, audio));

    std::ifstream file(path, std::ios::binary);
    char header[44];
    file.read(header, sizeof(header));
    CHECK(file.good());
    CHECK_EQ_I64(read_u16(header + 20), 1);
    CHECK_EQ_I64(read_u16(header + 22), 1);
    CHECK_EQ_I64(read_u32(header + 24), 24000);
    CHECK_EQ_I64(read_u16(header + 34), 16);
    CHECK_EQ_I64(read_u32(header + 40), audio.samples.size() * 2);

    int16_t pcm[5] = {};
    file.read(reinterpret_cast<char*>(pcm), sizeof(pcm));
    CHECK(file.good());
    CHECK_EQ_I64(pcm[0], -32768);
    CHECK_EQ_I64(pcm[1], -32768);
    CHECK_EQ_I64(pcm[2], 0);
    CHECK_EQ_I64(pcm[3], 32767);
    CHECK_EQ_I64(pcm[4], 32767);
    file.close();
    std::remove(path.c_str());

    qwen3_tts::AudioBuffer invalid = audio;
    invalid.samples = {};
    CHECK(!qwen3_tts::write_wav(path, invalid));
    invalid = audio;
    invalid.sample_rate = 0;
    CHECK(!qwen3_tts::write_wav(path, invalid));
    invalid = audio;
    invalid.channels = 3;
    CHECK(!qwen3_tts::write_wav(path, invalid));
    invalid = audio;
    invalid.samples[2] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!qwen3_tts::write_wav(path, invalid));
    std::remove(path.c_str());
    CHECK(!qwen3_tts::write_wav("", audio));

    audio.channels = 2;
    audio.samples = {-1.0f, 1.0f, 0.5f, -0.5f};
    CHECK_STATUS(qwen3_tts::write_wav(path, audio));
    file.open(path, std::ios::binary);
    file.read(header, sizeof(header));
    CHECK(file.good());
    CHECK_EQ_I64(read_u16(header + 22), 2);
    CHECK_EQ_I64(read_u16(header + 32), 4);
    CHECK_EQ_I64(read_u32(header + 40), audio.samples.size() * 2);
    file.close();
    std::remove(path.c_str());
    return 0;
}
