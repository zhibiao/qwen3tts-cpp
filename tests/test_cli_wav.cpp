#include "test_assert.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
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

int main(int argc, char** argv) {
    CHECK_EQ_I64(argc, 2);
    std::ifstream file(argv[1], std::ios::binary);
    CHECK(file.good());
    char header[44];
    file.read(header, sizeof(header));
    CHECK(file.good());
    CHECK_EQ_STR(std::string(header, 4), "RIFF");
    CHECK_EQ_STR(std::string(header + 8, 4), "WAVE");
    CHECK_EQ_I64(read_u16(header + 20), 1);
    CHECK_EQ_I64(read_u16(header + 22), 1);
    CHECK_EQ_I64(read_u32(header + 24), 24000);
    CHECK_EQ_I64(read_u16(header + 34), 16);
    const int64_t data_size = read_u32(header + 40);
    CHECK(data_size > 0);
    CHECK_EQ_I64(data_size % (1920 * 2), 0);

    file.seekg(0, std::ios::end);
    CHECK_EQ_I64(file.tellg(), 44 + data_size);
    std::cout << "CLI WAV valid: " << argv[1]
              << " samples=" << data_size / 2 << '\n';
    return 0;
}
