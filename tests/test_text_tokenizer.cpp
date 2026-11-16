#include "config.h"
#include "gguf_model.h"
#include "test_assert.h"
#include "text_tokenizer.h"

#include <memory>
#include <string>
#include <vector>

namespace {

void expect_text_ids(
    const qwen3_tts::TextTokenizer& tokenizer,
    const std::string& text,
    const std::vector<int32_t>& expected) {
    std::vector<int32_t> ids;
    CHECK_STATUS(tokenizer.encode(text, ids));
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

    return 0;
}
