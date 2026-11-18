#include "test_assert.h"
#include "text_tokenizer_internal.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace qwen3_tts::text_tokenizer_internal;

void test_unicode_properties() {
    CHECK(is_unicode_letter(U'A'));
    CHECK(is_unicode_letter(U'\u00e9'));
    CHECK(is_unicode_letter(U'\u4e2d'));
    CHECK(is_unicode_letter(static_cast<char32_t>(0x10400)));
    CHECK(!is_unicode_letter(U'1'));
    CHECK(!is_unicode_letter(static_cast<char32_t>(0x0301)));
    CHECK(!is_unicode_letter(static_cast<char32_t>(0x2160)));

    CHECK(is_unicode_number(U'1'));
    CHECK(is_unicode_number(static_cast<char32_t>(0x0660)));
    CHECK(is_unicode_number(static_cast<char32_t>(0xff11)));
    CHECK(is_unicode_number(static_cast<char32_t>(0x2160)));
    CHECK(is_unicode_number(static_cast<char32_t>(0x00b2)));
    CHECK(!is_unicode_number(U'A'));
    CHECK(!is_unicode_number(U'\u4e2d'));
    CHECK(!is_unicode_number(static_cast<char32_t>(0x0301)));

    CHECK(is_whitespace(U'\t'));
    CHECK(is_whitespace(U'\n'));
    CHECK(is_whitespace(U' '));
    CHECK(is_whitespace(static_cast<char32_t>(0x00a0)));
    CHECK(is_whitespace(static_cast<char32_t>(0x3000)));
    CHECK(!is_whitespace(static_cast<char32_t>(0x200b)));
    CHECK(!is_whitespace(U'A'));
    CHECK(!is_unicode_letter(static_cast<char32_t>(0x1f600)));
    CHECK(!is_unicode_number(static_cast<char32_t>(0x1f600)));
}

void test_utf8() {
    std::u32string codepoints;
    std::string error;
    const std::string mixed =
        std::string(u8"A") + u8"\u00e9" + u8"\u4e2d" + u8"\U0001f600";
    CHECK(decode_utf8(mixed, codepoints, error));
    CHECK_EQ_I64(codepoints.size(), 4);
    CHECK(codepoints[0] == U'A');
    CHECK(codepoints[1] == static_cast<char32_t>(0x00e9));
    CHECK(codepoints[2] == U'\u4e2d');
    CHECK(codepoints[3] == static_cast<char32_t>(0x1f600));
    CHECK_EQ_STR(
        utf32_slice_to_utf8(codepoints, 0, codepoints.size()),
        mixed);

    const std::vector<std::string> invalid_inputs = {
        std::string(1, static_cast<char>(0xff)),
        std::string("\xc0\xaf", 2),
        std::string("\xed\xa0\x80", 3),
        std::string("\xe2\x82", 2),
        std::string("\xe2\x41\xa1", 3),
    };
    for (const std::string& invalid : invalid_inputs) {
        CHECK(!decode_utf8(invalid, codepoints, error));
        CHECK(error.find("byte offset") != std::string::npos);
    }

    std::u32string all_scalars;
    all_scalars.reserve(0x110000 - 0x800);
    for (char32_t value = 0; value <= 0x10ffff; ++value) {
        if (value < 0xd800 || value > 0xdfff) {
            all_scalars.push_back(value);
        }
    }
    const std::string encoded = utf32_slice_to_utf8(
        all_scalars,
        0,
        all_scalars.size());
    CHECK(decode_utf8(encoded, codepoints, error));
    CHECK(codepoints == all_scalars);
}

void test_nfc() {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {u8"\u00e9", u8"e\u0301"},
        {u8"\u00c5", u8"A\u030a"},
        {u8"\u00c5", u8"\u212b"},
        {u8"\uac00", u8"\u1100\u1161"},
        {u8"\u304c", u8"\u304b\u3099"},
        {u8"\u1e0a", u8"D\u0307"},
    };
    for (const auto& [composed, decomposed] : cases) {
        std::string normalized;
        std::u32string codepoints;
        CHECK_STATUS(normalize_nfc(decomposed, normalized, codepoints));
        CHECK_EQ_STR(normalized, composed);

        std::string normalized_again;
        CHECK_STATUS(normalize_nfc(normalized, normalized_again, codepoints));
        CHECK_EQ_STR(normalized_again, normalized);
    }

    const std::string invalid(1, static_cast<char>(0xff));
    std::string normalized;
    std::u32string codepoints;
    CHECK(!normalize_nfc(invalid, normalized, codepoints));
}

void test_pretokenize() {
    struct TestCase {
        const char* text;
        std::vector<std::string> expected;
    };
    const std::vector<TestCase> cases = {
        {"Hello world", {"Hello", " world"}},
        {"abc  def", {"abc", " ", " def"}},
        {"123", {"1", "2", "3"}},
        {" !!!", {" !!!"}},
        {"abc\n\ndef", {"abc", "\n\n", "def"}},
        {"hello's", {"hello", "'s"}},
        {"it's we're I'VE", {"it", "'s", " we", "'re", " I", "'VE"}},
        {"a\r\n\r\nb", {"a", "\r\n\r\n", "b"}},
        {"a   ", {"a", "   "}},
        {"a  b", {"a", " ", " b"}},
        {"1A", {"1", "A"}},
        {"!!!\n", {"!!!\n"}},
        {"\t\n", {"\t\n"}},
        {u8"e\u0301", {"e", u8"\u0301"}},
        {u8"\uff11", {u8"\uff11"}},
        {u8"\u4e2dA", {u8"\u4e2dA"}},
    };

    for (const TestCase& item : cases) {
        std::u32string codepoints;
        std::string error;
        CHECK(decode_utf8(item.text, codepoints, error));
        CHECK(pretokenize(codepoints) == item.expected);
    }
}

} // namespace

int main() {
    test_unicode_properties();
    test_utf8();
    test_nfc();
    test_pretokenize();
    return 0;
}
