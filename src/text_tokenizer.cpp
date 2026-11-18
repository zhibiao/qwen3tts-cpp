#include "text_tokenizer.h"

#include "text_tokenizer_internal.h"

#include <algorithm>
#include <array>
#include <limits>

namespace qwen3_tts {

namespace {

void append_utf8(std::string& output, char32_t value) {
    if (value <= 0x7f) {
        output.push_back(static_cast<char>(value));
    } else if (value <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (value >> 6)));
        output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    } else if (value <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (value >> 12)));
        output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (value >> 18)));
        output.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    }
}

const std::array<char32_t, 256>& byte_encoder() {
    static const std::array<char32_t, 256> table = [] {
        std::array<char32_t, 256> result{};
        std::array<bool, 256> assigned{};
        uint32_t next = 0;
        auto assign_range = [&](int begin, int end) {
            for (int byte = begin; byte <= end; ++byte) {
                result[static_cast<size_t>(byte)] = static_cast<char32_t>(byte);
                assigned[static_cast<size_t>(byte)] = true;
            }
        };
        assign_range(0x21, 0x7e);
        assign_range(0xa1, 0xac);
        assign_range(0xae, 0xff);
        for (size_t byte = 0; byte < result.size(); ++byte) {
            if (!assigned[byte]) {
                result[byte] = static_cast<char32_t>(256 + next++);
            }
        }
        return result;
    }();
    return table;
}

} // namespace

Status TextTokenizer::load(
    const TextTokenizerConfig& config,
    const PromptConfig& prompt) {
    token_to_id_.clear();
    merge_ranks_.clear();
    added_tokens_.clear();
    prompt_ = prompt;
    nfc_normalization_ = config.nfc_normalization;

    const std::vector<std::string>& tokens = config.tokens;
    if (tokens.empty()) {
        return Status::error("text tokenizer vocabulary is empty");
    }
    for (size_t id = 0; id < tokens.size(); ++id) {
        if (tokens[id].empty()) {
            return Status::error("empty token at ID " + std::to_string(id));
        }
        if (!token_to_id_.emplace(tokens[id], static_cast<int32_t>(id)).second) {
            return Status::error("duplicate tokenizer token: " + tokens[id]);
        }
    }

    const std::vector<std::string>& merges = config.merges;
    if (merges.empty()) {
        return Status::error("text tokenizer merges are empty");
    }
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string& merge = merges[rank];
        if (merge.find(' ') == std::string::npos) {
            return Status::error("invalid BPE merge: " + merge);
        }
        if (!merge_ranks_.emplace(merge, static_cast<uint32_t>(rank)).second) {
            return Status::error("duplicate BPE merge: " + merge);
        }
    }

    if (config.added_token_names.size() != config.added_token_ids.size()) {
        return Status::error("added token name/id arrays have different sizes");
    }
    for (size_t i = 0; i < config.added_token_names.size(); ++i) {
        const std::string& name = config.added_token_names[i];
        const int32_t id = config.added_token_ids[i];
        if (id < 0 || static_cast<size_t>(id) >= tokens.size()) {
            return Status::error("added token ID out of range: " + name);
        }
        if (token_to_id_.find(name) == token_to_id_.end()) {
            return Status::error("added token is missing from token table: " + name);
        }
        added_tokens_.push_back({name, id});
    }
    std::sort(added_tokens_.begin(), added_tokens_.end(), [](const AddedToken& lhs, const AddedToken& rhs) {
        if (lhs.text.size() != rhs.text.size()) {
            return lhs.text.size() > rhs.text.size();
        }
        return lhs.id < rhs.id;
    });

    if (prompt_.assistant_prefix.empty() ||
        prompt_.assistant_suffix.empty() ||
        prompt_.reference_prefix.empty() ||
        prompt_.reference_suffix.empty() ||
        prompt_.instruction_prefix.empty() ||
        prompt_.instruction_suffix.empty()) {
        return Status::error("text tokenizer prompt configuration is incomplete");
    }
    return Status::ok();
}

int32_t TextTokenizer::special_token_id(const std::string& name) const {
    const auto it = token_to_id_.find(name);
    return it == token_to_id_.end() ? -1 : it->second;
}

bool TextTokenizer::has_token(const std::string& token) const {
    return token_to_id_.find(token) != token_to_id_.end();
}

Status TextTokenizer::encode(
    const std::string& text,
    std::vector<int32_t>& token_ids) const {
    token_ids.clear();
    size_t position = 0;
    while (position < text.size()) {
        size_t next_position = std::string::npos;
        int32_t next_id = -1;
        size_t next_length = 0;
        for (const AddedToken& added : added_tokens_) {
            if (added.text.empty() || position + added.text.size() > text.size()) {
                continue;
            }
            if (text.compare(position, added.text.size(), added.text) == 0) {
                next_position = position;
                next_id = added.id;
                next_length = added.text.size();
                break;
            }
        }
        if (next_position != std::string::npos) {
            token_ids.push_back(next_id);
            position += next_length;
            continue;
        }

        size_t next_added = text.size();
        for (const AddedToken& added : added_tokens_) {
            const size_t found = text.find(added.text, position);
            if (found != std::string::npos && found < next_added) {
                next_added = found;
            }
        }
        Status status = encode_regular(text.substr(position, next_added - position), token_ids);
        if (!status) {
            return status;
        }
        position = next_added;
    }
    return Status::ok();
}

Status TextTokenizer::encode_regular(
    const std::string& text,
    std::vector<int32_t>& token_ids) const {
    if (text.empty()) {
        return Status::ok();
    }

    std::string error;
    std::string normalized;
    std::u32string codepoints;
    if (nfc_normalization_) {
        Status status = text_tokenizer_internal::normalize_nfc(
            text,
            normalized,
            codepoints);
        if (!status) {
            return status;
        }
    } else if (!text_tokenizer_internal::decode_utf8(
                   text,
                   codepoints,
                   error)) {
        return Status::error(error);
    }

    for (const std::string& piece :
         text_tokenizer_internal::pretokenize(codepoints)) {
        Status status = encode_piece(piece, token_ids);
        if (!status) {
            return status;
        }
    }
    return Status::ok();
}

Status TextTokenizer::encode_piece(
    const std::string& piece,
    std::vector<int32_t>& token_ids) const {
    const auto& encoder = byte_encoder();
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (const unsigned char byte : piece) {
        std::string symbol;
        append_utf8(symbol, encoder[byte]);
        symbols.push_back(std::move(symbol));
    }

    while (symbols.size() > 1) {
        uint32_t best_rank = std::numeric_limits<uint32_t>::max();
        size_t best_index = std::numeric_limits<size_t>::max();
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const std::string pair = symbols[i] + " " + symbols[i + 1];
            const auto it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_index = i;
            }
        }
        if (best_index == std::numeric_limits<size_t>::max()) {
            break;
        }
        symbols[best_index] += symbols[best_index + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }

    for (const std::string& symbol : symbols) {
        const auto it = token_to_id_.find(symbol);
        if (it == token_to_id_.end()) {
            return Status::error("BPE symbol is missing from vocabulary");
        }
        token_ids.push_back(it->second);
    }
    return Status::ok();
}

Status TextTokenizer::encode_assistant(
    const std::string& text,
    std::vector<int32_t>& token_ids) const {
    return encode(prompt_.assistant_prefix + text + prompt_.assistant_suffix, token_ids);
}

Status TextTokenizer::encode_reference(
    const std::string& text,
    std::vector<int32_t>& token_ids) const {
    return encode(prompt_.reference_prefix + text + prompt_.reference_suffix, token_ids);
}

Status TextTokenizer::encode_instruction(
    const std::string& text,
    std::vector<int32_t>& token_ids) const {
    return encode(prompt_.instruction_prefix + text + prompt_.instruction_suffix, token_ids);
}

} // namespace qwen3_tts
