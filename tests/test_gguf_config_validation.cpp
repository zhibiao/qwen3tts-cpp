#include "config.h"
#include "gguf_model.h"
#include "test_assert.h"

#include <iostream>
#include <memory>
#include <string>

int main() {
    std::unique_ptr<qwen3_tts::GGUFModel> talker;
    std::unique_ptr<qwen3_tts::GGUFModel> speech;
    CHECK_STATUS(qwen3_tts::GGUFModel::open(QWEN3_TTS_TEST_GGUF, talker));
    CHECK_STATUS(qwen3_tts::GGUFModel::open(
        QWEN3_TTS_TEST_SPEECH_GGUF,
        speech));

    std::unique_ptr<qwen3_tts::GGUFModel> missing;
    CHECK(!qwen3_tts::GGUFModel::open(
        "missing-qwen3-tts-model.gguf",
        missing));
    CHECK(!talker->get_string("missing.metadata").size());
    CHECK(talker->tensor("missing.tensor") == nullptr);

    qwen3_tts::TalkerGGUFConfig talker_config;
    qwen3_tts::SpeechTokenizerGGUFConfig speech_config;
    CHECK(!qwen3_tts::load_talker_config(*speech, talker_config));
    CHECK(!qwen3_tts::load_speech_config(*talker, speech_config));
    CHECK_STATUS(qwen3_tts::load_talker_config(*talker, talker_config));
    CHECK_STATUS(qwen3_tts::load_speech_config(*speech, speech_config));

    qwen3_tts::Qwen3TTSConfig combined;
    CHECK(!qwen3_tts::load_qwen3tts_config(
        *speech,
        *talker,
        combined));
    CHECK_STATUS(qwen3_tts::load_qwen3tts_config(
        *talker,
        *speech,
        combined));
    CHECK_EQ_I64(
        combined.talker.decoder.num_code_groups,
        combined.speech.tokenizer.decoder.num_quantizers);

#if defined(QWEN3_TTS_TEST_GGUF_Q8)
    std::unique_ptr<qwen3_tts::GGUFModel> quantized;
    CHECK_STATUS(qwen3_tts::GGUFModel::open(
        QWEN3_TTS_TEST_GGUF_Q8,
        quantized));
    qwen3_tts::TalkerGGUFConfig quantized_config;
    CHECK_STATUS(qwen3_tts::load_talker_config(
        *quantized,
        quantized_config));
    const ggml_tensor* codec_head =
        quantized->tensor("talker.codec_head.weight");
    CHECK(codec_head != nullptr);
    CHECK(codec_head->type == GGML_TYPE_Q8_0);
    CHECK_EQ_I64(
        quantized_config.decoder.embedding_length,
        talker_config.decoder.embedding_length);
#endif

    std::cout << "GGUF/config validation passed\n";
    return 0;
}
