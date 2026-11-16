#pragma once

#include "config.h"
#include "status.h"
#include "types.h"

#include "ggml-backend.h"

#include <memory>

namespace qwen3_tts {

class Qwen3TTSContext;

class SpeechTokenizer {
public:
    SpeechTokenizer();
    ~SpeechTokenizer();

    Status load(
        const SpeechTokenizerConfig& config,
        Qwen3TTSContext& context);

    Status decode(
        Qwen3TTSContext& context,
        const SpeechCodeMatrix& codes,
        AudioBuffer& audio);

private:
    class Decoder;
    class Encoder;
    std::unique_ptr<Decoder> decoder_;
};

} // namespace qwen3_tts
