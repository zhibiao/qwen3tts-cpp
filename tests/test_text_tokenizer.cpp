#include "config.h"
#include "gguf_model.h"
#include "test_assert.h"
#include "text_tokenizer.h"

#include <memory>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

void expect_text_ids(
    const qwen3_tts::TextTokenizer& tokenizer,
    const std::string& text,
    const std::vector<int32_t>& expected) {
    std::vector<int32_t> ids;
    CHECK_STATUS(tokenizer.encode(text, ids));
    if (ids != expected) {
        std::cerr << "token mismatch for: " << text << "\nexpected:";
        for (int32_t id : expected) {
            std::cerr << ' ' << id;
        }
        std::cerr << "\nactual:";
        for (int32_t id : ids) {
            std::cerr << ' ' << id;
        }
        std::cerr << '\n';
    }
    CHECK(ids == expected);
}

} // namespace

int main() {
    std::unique_ptr<qwen3_tts::GGUFModel> model;
    CHECK_STATUS(qwen3_tts::GGUFModel::open(QWEN3_TTS_TEST_GGUF, model));
    qwen3_tts::TalkerGGUFConfig config;
    CHECK_STATUS(qwen3_tts::load_talker_config(*model, config));

    qwen3_tts::TextTokenizer tokenizer;
    CHECK_STATUS(tokenizer.load(config.text_tokenizer, config.prompt));

    std::vector<int32_t> ids;
    CHECK_STATUS(tokenizer.encode("a", ids));
    CHECK_EQ_I64(ids.size(), 1);
    CHECK_EQ_I64(ids[0], 64);

    CHECK_STATUS(tokenizer.encode("!", ids));
    CHECK_EQ_I64(ids.size(), 1);
    CHECK_EQ_I64(ids[0], 0);

    CHECK_STATUS(tokenizer.encode(" a", ids));
    CHECK_EQ_I64(ids.size(), 1);
    CHECK_EQ_I64(ids[0], 264);

    CHECK_STATUS(tokenizer.encode("  a", ids));
    CHECK((ids == std::vector<int32_t>{220, 264}));

    CHECK_STATUS(tokenizer.encode("<tts_pad>", ids));
    CHECK_EQ_I64(ids.size(), 1);
    CHECK_EQ_I64(ids[0], 151671);

    CHECK_STATUS(tokenizer.encode("<tts_text_bos_single>", ids));
    CHECK_EQ_I64(ids.size(), 1);
    CHECK_EQ_I64(ids[0], 151674);

    expect_text_ids(tokenizer, "Hello", {9707});
    expect_text_ids(tokenizer, "Hello, world!", {9707, 11, 1879, 0});
    expect_text_ids(tokenizer, "你好，世界。", {108386, 3837, 99489, 1773});
    expect_text_ids(
        tokenizer,
        "<|im_start|>assistant\n你好，世界。<|im_end|>\n<|im_start|>assistant\n",
        {151644, 77091, 198, 108386, 3837, 99489, 1773, 151645, 198, 151644, 77091, 198});

    std::vector<int32_t> assistant_ids;
    CHECK_STATUS(tokenizer.encode_assistant("你好，世界。", assistant_ids));
    CHECK((assistant_ids == std::vector<int32_t>{
        151644, 77091, 198, 108386, 3837, 99489, 1773,
        151645, 198, 151644, 77091, 198}));

    std::vector<int32_t> reference_ids;
    CHECK_STATUS(tokenizer.encode_reference("你好，世界。", reference_ids));
    CHECK((reference_ids == std::vector<int32_t>{
        151644, 77091, 198, 108386, 3837, 99489, 1773, 151645, 198}));

    expect_text_ids(
        tokenizer,
        "it's we're I'VE",
        {275, 594, 582, 2299, 358, 6, 4491});
    expect_text_ids(tokenizer, "1A", {16, 32});
    expect_text_ids(tokenizer, "123", {16, 17, 18});
    expect_text_ids(tokenizer, " !!!", {32870});
    expect_text_ids(tokenizer, "abc  def", {13683, 220, 707});
    expect_text_ids(tokenizer, "a   ", {64, 262});
    expect_text_ids(tokenizer, "a  b", {64, 220, 293});
    expect_text_ids(tokenizer, "abc\n\ndef", {13683, 271, 750});
    expect_text_ids(tokenizer, "!!!\n", {79295});
    expect_text_ids(tokenizer, "a\r\n\r\nb", {64, 871, 65});

    const std::vector<
        std::pair<std::string, std::string>> nfc_equivalents = {
        {u8"\u00e9", u8"e\u0301"},
        {u8"\u00c5", u8"A\u030a"},
        {u8"\u00c5", u8"\u212b"},
        {u8"\uac00", u8"\u1100\u1161"},
        {u8"\u304c", u8"\u304b\u3099"},
        {u8"\u1e0a", u8"D\u0307"},
    };
    qwen3_tts::TextTokenizerConfig nfc_config =
        config.text_tokenizer;
    nfc_config.nfc_normalization = true;
    qwen3_tts::TextTokenizer nfc_tokenizer;
    CHECK_STATUS(nfc_tokenizer.load(nfc_config, config.prompt));
    for (const auto& [composed, decomposed] : nfc_equivalents) {
        std::vector<int32_t> composed_ids;
        std::vector<int32_t> decomposed_ids;
        CHECK_STATUS(nfc_tokenizer.encode(composed, composed_ids));
        CHECK_STATUS(nfc_tokenizer.encode(decomposed, decomposed_ids));
        CHECK(composed_ids == decomposed_ids);
    }

    qwen3_tts::TextTokenizerConfig raw_config =
        config.text_tokenizer;
    raw_config.nfc_normalization = false;
    qwen3_tts::TextTokenizer raw_tokenizer;
    CHECK_STATUS(raw_tokenizer.load(raw_config, config.prompt));

    std::vector<int32_t> raw_composed;
    std::vector<int32_t> raw_decomposed;
    CHECK_STATUS(raw_tokenizer.encode(u8"\u00e9", raw_composed));
    CHECK_STATUS(raw_tokenizer.encode(u8"e\u0301", raw_decomposed));
    CHECK(raw_composed != raw_decomposed);

    const std::string invalid_utf8(
        1,
        static_cast<char>(0xff));
    CHECK(!tokenizer.encode(invalid_utf8, ids));
    CHECK(!raw_tokenizer.encode(invalid_utf8, ids));

    return 0;
}
