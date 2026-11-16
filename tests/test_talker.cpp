#include "config.h"
#include "qwen3_tts.h"
#include "talker.h"
#include "test_assert.h"

#include <cmath>
#include <vector>

int main() {
    qwen3_tts::Qwen3TTSContext context;
    qwen3_tts::Qwen3TTSContextParams params;
    params.n_threads = 1;
    CHECK_STATUS(context.load(
        QWEN3_TTS_TEST_GGUF,
        QWEN3_TTS_TEST_SPEECH_GGUF,
        params));
    qwen3_tts::TalkerGGUFConfig config;
    CHECK_STATUS(qwen3_tts::load_talker_config(
        context.talker_gguf(),
        config));

    qwen3_tts::Talker talker;
    CHECK_STATUS(talker.load_tensors(config.decoder, context));

    constexpr int64_t frames = 2;
    const int64_t hidden_size = config.decoder.embedding_length;
    const int64_t codec_vocab = config.decoder.codec_vocab_size;
    std::vector<float> input(
        static_cast<size_t>(frames * hidden_size));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 7) * 0.001f;
    }
    std::vector<float> hidden;
    std::vector<float> logits;
    CHECK_STATUS(talker.prefill(context, input, frames, hidden, logits));
    CHECK_EQ_I64(hidden.size(), frames * hidden_size);
    CHECK_EQ_I64(logits.size(), frames * codec_vocab);
    for (float value : hidden) {
        CHECK(std::isfinite(value));
    }
    for (float value : logits) {
        CHECK(std::isfinite(value));
    }
    std::vector<float> step_input(
        static_cast<size_t>(hidden_size),
        0.002f);
    CHECK_STATUS(talker.decode_step(context, step_input, hidden, logits));
    CHECK_EQ_I64(hidden.size(), hidden_size);
    CHECK_EQ_I64(logits.size(), codec_vocab);
    for (float value : hidden) {
        CHECK(std::isfinite(value));
    }
    for (float value : logits) {
        CHECK(std::isfinite(value));
    }

    std::vector<float> invalid_step(
        static_cast<size_t>(hidden_size - 1),
        0.0f);
    CHECK(!talker.decode_step(context, invalid_step, hidden, logits));

    std::vector<float> invalid;
    CHECK(!talker.prefill(context, invalid, frames, hidden, logits));
    return 0;
}
