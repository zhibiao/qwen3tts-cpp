#include "config.h"
#include "gguf_model.h"
#include "test_assert.h"

#include <memory>
#include <string>
#include <iostream>

int main() {
    std::unique_ptr<qwen3_tts::GGUFModel> talker;
    CHECK_STATUS(qwen3_tts::GGUFModel::open(QWEN3_TTS_TEST_GGUF, talker));
    std::unique_ptr<qwen3_tts::GGUFModel> speech;
    CHECK_STATUS(
        qwen3_tts::GGUFModel::open(QWEN3_TTS_TEST_SPEECH_GGUF, speech));

    CHECK_EQ_STR(talker->get_string("general.architecture"), "qwen3-tts");
    CHECK_EQ_STR(talker->get_string("qwen3_tts.file_role"), "talker");
    CHECK_EQ_STR(talker->get_string("qwen3_tts.model_type"), "custom_voice");
    CHECK_EQ_STR(
        talker->get_string("qwen3_tts.tokenizer_type"),
        "qwen3_tts_tokenizer_12hz");
    CHECK_EQ_I64(talker->get_u32("qwen3_tts.schema_version"), 3);
    CHECK_EQ_I64(
        talker->get_u32("qwen3_tts.talker.num_code_groups"), 16);
    CHECK_EQ_I64(
        talker->get_i32("qwen3_tts.talker.codec_eos_token_id"),
        2150);
    CHECK_EQ_I64(talker->tensor_count(), 402);
    CHECK_EQ_I64(
        talker->get_string_array("tokenizer.ggml.tokens").size(),
        151676);
    CHECK_EQ_I64(
        talker->get_string_array("tokenizer.ggml.merges").size(),
        151387);
    CHECK_EQ_I64(
        talker->get_string_array("qwen3_tts.speaker.names").size(),
        9);
    CHECK_EQ_I64(
        talker->get_string_array("qwen3_tts.language.names").size(),
        12);
    CHECK(talker->has_tensor("talker.text_embedding.weight"));
    CHECK(talker->has_tensor("talker.codec_head.weight"));

    CHECK_EQ_STR(
        speech->get_string("general.architecture"),
        "qwen3-tts-tokenizer");
    CHECK_EQ_STR(
        speech->get_string("qwen3_tts.file_role"),
        "speech_tokenizer");
    CHECK_EQ_I64(speech->get_u32("qwen3_tts.schema_version"), 3);
    CHECK_EQ_I64(
        speech->get_u32("qwen3_tts.speech.output_sample_rate"),
        24000);
    CHECK_EQ_I64(
        speech->get_u32("qwen3_tts.speech.decoder.codebook_size"),
        2048);
    CHECK_EQ_I64(speech->tensor_count(), 496);
    CHECK(speech->has_tensor("sdec.pre_conv.conv.weight"));
    CHECK(speech->has_tensor("sdec.q.q0.rvq.l.0.cb.sum"));

    const ggml_tensor* embedding =
        talker->tensor("talker.text_embedding.weight");
    CHECK(embedding != nullptr);
    CHECK_EQ_I64(embedding->ne[0], 2048);
    CHECK_EQ_I64(embedding->ne[1], 151936);
    CHECK(embedding->type == GGML_TYPE_BF16);
    CHECK(embedding->data != nullptr);
    CHECK(talker->tensor("missing.tensor") == nullptr);

    qwen3_tts::Qwen3TTSConfig config;
    const qwen3_tts::Status config_status =
        qwen3_tts::load_qwen3tts_config(*talker, *speech, config);
    if (!config_status) {
        std::cerr << config_status.message() << '\n';
    }
    CHECK_STATUS(config_status);
    CHECK_EQ_STR(config.talker.header.general.architecture, "qwen3-tts");
    CHECK(
        config.talker.identity.model_type ==
        qwen3_tts::TtsModelType::CustomVoice);
    CHECK_EQ_I64(config.talker.decoder.embedding_length, 1024);
    CHECK_EQ_I64(config.talker.decoder.num_code_groups, 16);
    CHECK_EQ_I64(config.talker.code_predictor.hidden_size, 1024);
    CHECK_EQ_I64(config.speech.tokenizer.decode_upsample_rate, 1920);
    CHECK_EQ_I64(config.talker.generation.max_new_tokens, 8192);
    CHECK_EQ_I64(config.talker.decoder.num_ctx, 2048);
    CHECK(!config.talker.text_tokenizer.nfc_normalization);
    return 0;
}
