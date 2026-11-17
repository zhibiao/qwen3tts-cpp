#include "config.h"
#include "gguf_model.h"
#include "test_assert.h"
#include "text_tokenizer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace {

std::vector<int32_t> ids_from_json(const nlohmann::json& values) {
    std::vector<int32_t> result;
    result.reserve(values.size());
    for (const auto& value : values) {
        result.push_back(value.get<int32_t>());
    }
    return result;
}

} // namespace

int main() {
    std::ifstream file(QWEN3_TTS_TEXT_GOLDEN, std::ios::binary);
    CHECK(file.good());
    nlohmann::json fixture;
    file >> fixture;
    CHECK_EQ_STR(
        fixture.at("format").get<std::string>(),
        "qwen3-tts-text-tokenizer-golden-v1");

    std::unique_ptr<qwen3_tts::GGUFModel> model;
    CHECK_STATUS(qwen3_tts::GGUFModel::open(QWEN3_TTS_TEST_GGUF, model));
    qwen3_tts::TalkerGGUFConfig config;
    CHECK_STATUS(qwen3_tts::load_talker_config(*model, config));

    qwen3_tts::TextTokenizer tokenizer;
    CHECK_STATUS(tokenizer.load(config.text_tokenizer, config.prompt));

    int passed = 0;
    for (const auto& item : fixture.at("cases")) {
        const std::string name = item.at("name").get<std::string>();
        const std::string kind = item.at("kind").get<std::string>();
        const std::string text = item.at("text").get<std::string>();
        const std::vector<int32_t> expected =
            ids_from_json(item.at("ids"));
        std::vector<int32_t> actual;
        if (kind == "raw") {
            CHECK_STATUS(tokenizer.encode(text, actual));
        } else if (kind == "assistant") {
            CHECK_STATUS(tokenizer.encode_assistant(text, actual));
        } else if (kind == "reference") {
            CHECK_STATUS(tokenizer.encode_reference(text, actual));
        } else if (kind == "instruction") {
            CHECK_STATUS(tokenizer.encode_instruction(text, actual));
        } else {
            CHECK(false);
        }

        if (actual != expected) {
            std::cerr << "case=" << name << " kind=" << kind << '\n';
            std::cerr << " actual size=" << actual.size()
                      << " expected size=" << expected.size() << '\n';
            const size_t common = std::min(actual.size(), expected.size());
            for (size_t i = 0; i < common; ++i) {
                if (actual[i] != expected[i]) {
                    std::cerr << " first mismatch at " << i
                              << ": actual=" << actual[i]
                              << " expected=" << expected[i] << '\n';
                    break;
                }
            }
        }
        CHECK(actual == expected);
        CHECK(!actual.empty());
        for (int32_t id : actual) {
            CHECK(id >= 0);
            CHECK(id < config.text_tokenizer.vocab_size);
        }
        ++passed;
    }

    CHECK_EQ_I64(passed, static_cast<int64_t>(fixture.at("cases").size()));
    std::cout << "TextTokenizer scenarios passed: " << passed << '\n';
    return 0;
}
