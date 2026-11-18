#pragma once

#include "status.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace qwen3_tts::text_tokenizer_internal {

bool decode_utf8(
    std::string_view text,
    std::u32string& codepoints,
    std::string& error);

std::string utf32_slice_to_utf8(
    std::u32string_view text,
    size_t begin,
    size_t end);

Status normalize_nfc(
    std::string_view input,
    std::string& output,
    std::u32string& codepoints);

bool is_unicode_letter(char32_t value);
bool is_unicode_number(char32_t value);
bool is_whitespace(char32_t value);

std::vector<std::string> pretokenize(std::u32string_view text);

} // namespace qwen3_tts::text_tokenizer_internal
