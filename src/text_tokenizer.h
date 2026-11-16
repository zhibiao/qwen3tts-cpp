#pragma once

#include "config.h"
#include "status.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qwen3_tts {

class TextTokenizer {
public:
    Status load(
        const TextTokenizerConfig& config,
        const PromptConfig& prompt);

    Status encode(
        const std::string& text,
        std::vector<int32_t>& token_ids) const;

    Status encode_assistant(
        const std::string& text,
        std::vector<int32_t>& token_ids) const;

    Status encode_reference(
        const std::string& text,
        std::vector<int32_t>& token_ids) const;

    Status encode_instruction(
        const std::string& text,
        std::vector<int32_t>& token_ids) const;

    int32_t special_token_id(const std::string& name) const;
    bool has_token(const std::string& token) const;

private:
    struct AddedToken {
        std::string text;
        int32_t id = 0;
    };

    Status encode_regular(
        const std::string& text,
        std::vector<int32_t>& token_ids) const;

    Status encode_piece(
        const std::string& piece,
        std::vector<int32_t>& token_ids) const;

    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, uint32_t> merge_ranks_;
    std::vector<AddedToken> added_tokens_;
    PromptConfig prompt_;
};

} // namespace qwen3_tts
