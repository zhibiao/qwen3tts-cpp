#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace qwen3_tts::test {

inline void fail(const char* expression, const char* file, int line, const std::string& message = {}) {
    std::cerr << file << ':' << line << ": check failed: " << expression;
    if (!message.empty()) {
        std::cerr << " (" << message << ')';
    }
    std::cerr << '\n';
    std::exit(EXIT_FAILURE);
}

inline void expect(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        fail(expression, file, line);
    }
}

inline void expect_eq_i64(int64_t actual, int64_t expected, const char* expression, const char* file, int line) {
    if (actual != expected) {
        fail(expression, file, line, "actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
    }
}

inline void expect_eq_str(const std::string& actual, const std::string& expected, const char* expression, const char* file, int line) {
    if (actual != expected) {
        fail(expression, file, line, "actual='" + actual + "' expected='" + expected + "'");
    }
}

inline void expect_near(float actual, float expected, float tolerance, const char* expression, const char* file, int line) {
    if (std::fabs(actual - expected) > tolerance) {
        fail(expression, file, line, "actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
    }
}

} // namespace qwen3_tts::test

#define CHECK(condition) ::qwen3_tts::test::expect((condition), #condition, __FILE__, __LINE__)
#define CHECK_STATUS(condition) ::qwen3_tts::test::expect(static_cast<bool>(condition), #condition, __FILE__, __LINE__)
#define CHECK_EQ_I64(actual, expected) ::qwen3_tts::test::expect_eq_i64((actual), (expected), #actual " == " #expected, __FILE__, __LINE__)
#define CHECK_EQ_STR(actual, expected) ::qwen3_tts::test::expect_eq_str((actual), (expected), #actual " == " #expected, __FILE__, __LINE__)
#define CHECK_NEAR(actual, expected, tolerance) ::qwen3_tts::test::expect_near((actual), (expected), (tolerance), #actual " ~= " #expected, __FILE__, __LINE__)
