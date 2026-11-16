#include "test_assert.h"
#include "wav_writer.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

uint32_t read_u32(const char* value) {
    return static_cast<uint32_t>(static_cast<uint8_t>(value[0])) |
        (static_cast<uint32_t>(static_cast<uint8_t>(value[1])) << 8) |
        (static_cast<uint32_t>(static_cast<uint8_t>(value[2])) << 16) |
        (static_cast<uint32_t>(static_cast<uint8_t>(value[3])) << 24);
}

} // namespace

int main() {
    qwen3_tts::AudioBuffer audio;
    audio.sample_rate = 24000;
    audio.channels = 1;
    audio.samples = {0.0f, 0.5f, -0.5f, 1.0f};
    const std::string path = "wav_writer_test.wav";
    CHECK_STATUS(qwen3_tts::write_wav(path, audio));

    std::ifstream file(path, std::ios::binary);
    CHECK(file.good());
    char header[44];
    file.read(header, sizeof(header));
    CHECK(file.good());
    CHECK_EQ_STR(std::string(header, 4), "RIFF");
    CHECK_EQ_STR(std::string(header + 8, 4), "WAVE");
    CHECK_EQ_I64(read_u32(header + 24), 24000);
    CHECK_EQ_I64(read_u32(header + 40), 8);
    file.close();
    std::remove(path.c_str());
    return 0;
}
