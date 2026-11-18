#include "text_tokenizer_internal.h"

#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "uni-algo-1.2.0/conv.h"
#include "uni-algo-1.2.0/norm.h"
#include "uni-algo-1.2.0/prop.h"

namespace qwen3_tts::text_tokenizer_internal {

bool decode_utf8(
    std::string_view text,
    std::u32string& codepoints,
    std::string& error) {
    una::error decode_error;
    codepoints = una::strict::utf8to32u(text, decode_error);
    if (decode_error.get_code() == una::error::code::success) {
        error.clear();
        return true;
    }

    error = "invalid UTF-8";
    if (decode_error.has_pos()) {
        error += " at byte offset " + std::to_string(decode_error.pos());
    }
    return false;
}

std::string utf32_slice_to_utf8(
    std::u32string_view text,
    size_t begin,
    size_t end) {
    if (begin > end || end > text.size()) {
        return {};
    }
    return una::utf32to8(text.substr(begin, end - begin));
}

Status normalize_nfc(
    std::string_view input,
    std::string& output,
    std::u32string& codepoints) {
    std::string error;
    if (!decode_utf8(input, codepoints, error)) {
        return Status::error(error);
    }

    output = una::norm::to_nfc_utf8(input);
    if (output != input && !decode_utf8(output, codepoints, error)) {
        return Status::error(error);
    }
    return Status::ok();
}

bool is_unicode_letter(char32_t value) {
    return una::codepoint::prop{value}.General_Category_L();
}

bool is_unicode_number(char32_t value) {
    return una::codepoint::prop{value}.General_Category_N();
}

bool is_whitespace(char32_t value) {
    return una::codepoint::is_whitespace(value);
}

bool is_contraction_at(
    std::u32string_view codepoints,
    size_t position,
    std::initializer_list<char32_t> suffix) {
    if (position >= codepoints.size() || codepoints[position] != U'\'') {
        return false;
    }
    if (position + suffix.size() >= codepoints.size()) {
        return false;
    }

    size_t offset = 1;
    for (char32_t expected : suffix) {
        char32_t actual = codepoints[position + offset];
        if (actual >= U'A' && actual <= U'Z') {
            actual = actual - U'A' + U'a';
        }
        if (actual != expected) {
            return false;
        }
        ++offset;
    }
    return true;
}

size_t consume_letters(std::u32string_view codepoints, size_t begin) {
    size_t end = begin;
    while (end < codepoints.size() &&
           is_unicode_letter(codepoints[end])) {
        ++end;
    }
    return end;
}

size_t consume_symbols(std::u32string_view codepoints, size_t begin) {
    size_t end = begin;
    while (end < codepoints.size()) {
        const char32_t value = codepoints[end];
        if (is_whitespace(value) ||
            is_unicode_letter(value) ||
            is_unicode_number(value)) {
            break;
        }
        ++end;
    }
    return end;
}

size_t consume_newlines(std::u32string_view codepoints, size_t begin) {
    size_t end = begin;
    while (end < codepoints.size()) {
        const char32_t value = codepoints[end];
        if (value != U'\r' && value != U'\n') {
            break;
        }
        ++end;
    }
    return end;
}

std::vector<std::string> pretokenize_impl(std::u32string_view codepoints) {
    std::vector<std::string> pieces;
    size_t position = 0;
    while (position < codepoints.size()) {
        if (is_contraction_at(codepoints, position, {U's'}) ||
            is_contraction_at(codepoints, position, {U't'}) ||
            is_contraction_at(codepoints, position, {U'm'}) ||
            is_contraction_at(codepoints, position, {U'd'})) {
            pieces.push_back(utf32_slice_to_utf8(
                codepoints, position, position + 2));
            position += 2;
            continue;
        }
        if (is_contraction_at(codepoints, position, {U'r', U'e'}) ||
            is_contraction_at(codepoints, position, {U'v', U'e'}) ||
            is_contraction_at(codepoints, position, {U'l', U'l'})) {
            pieces.push_back(utf32_slice_to_utf8(
                codepoints, position, position + 3));
            position += 3;
            continue;
        }

        const char32_t current = codepoints[position];
        if (is_unicode_letter(current)) {
            const size_t end = consume_letters(codepoints, position);
            pieces.push_back(
                utf32_slice_to_utf8(codepoints, position, end));
            position = end;
            continue;
        }
        if (position + 1 < codepoints.size() &&
            current != U'\r' &&
            current != U'\n' &&
            !is_unicode_number(current) &&
            is_unicode_letter(codepoints[position + 1])) {
            const size_t end = consume_letters(codepoints, position + 1);
            pieces.push_back(
                utf32_slice_to_utf8(codepoints, position, end));
            position = end;
            continue;
        }

        if (is_unicode_number(current)) {
            pieces.push_back(utf32_slice_to_utf8(
                codepoints, position, position + 1));
            ++position;
            continue;
        }

        size_t symbol_begin = std::string::npos;
        if (current == U' ' &&
            position + 1 < codepoints.size() &&
            !is_whitespace(codepoints[position + 1]) &&
            !is_unicode_letter(codepoints[position + 1]) &&
            !is_unicode_number(codepoints[position + 1])) {
            symbol_begin = position + 1;
        } else if (!is_whitespace(current) &&
                   !is_unicode_letter(current) &&
                   !is_unicode_number(current)) {
            symbol_begin = position;
        }
        if (symbol_begin != std::string::npos) {
            size_t end = consume_symbols(codepoints, symbol_begin);
            end = consume_newlines(codepoints, end);
            pieces.push_back(
                utf32_slice_to_utf8(codepoints, position, end));
            position = end;
            continue;
        }

        if (is_whitespace(current)) {
            size_t whitespace_end = position;
            while (whitespace_end < codepoints.size() &&
                   is_whitespace(codepoints[whitespace_end])) {
                ++whitespace_end;
            }

            size_t newline_begin = position;
            while (newline_begin < whitespace_end &&
                   codepoints[newline_begin] != U'\r' &&
                   codepoints[newline_begin] != U'\n') {
                ++newline_begin;
            }
            if (newline_begin < whitespace_end) {
                const size_t end =
                    consume_newlines(codepoints, newline_begin);
                pieces.push_back(
                    utf32_slice_to_utf8(codepoints, position, end));
                position = end;
                continue;
            }

            size_t end = whitespace_end;
            if (end < codepoints.size() && end - position > 1) {
                --end;
            }
            pieces.push_back(
                utf32_slice_to_utf8(codepoints, position, end));
            position = end;
            continue;
        }

        pieces.push_back(utf32_slice_to_utf8(
            codepoints, position, position + 1));
        ++position;
    }
    return pieces;
}

std::vector<std::string> pretokenize(std::u32string_view text) {
    return pretokenize_impl(text);
}

} // namespace qwen3_tts::text_tokenizer_internal
