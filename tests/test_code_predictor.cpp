#include "config.h"
#include "code_predictor.h"
#include "qwen3_tts.h"
#include "test_assert.h"

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

    std::vector<float> hidden(
        static_cast<size_t>(config.code_predictor.input_embedding_size),
        0.001f);
    qwen3_tts::Qwen3TTSContext unloaded_context;
    qwen3_tts::CodePredictor unloaded_predictor;
    std::vector<int32_t> codes;
    CHECK(!unloaded_predictor.predict_frame(
        unloaded_context,
        hidden,
        2149,
        codes));
    CHECK(!unloaded_predictor.load(
        config.code_predictor,
        unloaded_context));

    qwen3_tts::CodePredictor predictor;
    CHECK_STATUS(predictor.load(config.code_predictor, context));
    CHECK_STATUS(predictor.init_graphs(context));

    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = static_cast<float>(i % 7) * 0.001f;
    }
    constexpr int32_t c0 = 2149;
    CHECK_STATUS(predictor.predict_frame(context, hidden, c0, codes));
    CHECK_EQ_I64(codes.size(), config.code_predictor.num_code_groups);
    CHECK_EQ_I64(codes[0], c0);
    for (size_t i = 1; i < codes.size(); ++i) {
        CHECK(codes[i] >= 0);
        CHECK(codes[i] < config.code_predictor.vocab_size);
    }

    std::vector<int32_t> repeated_codes;
    CHECK_STATUS(predictor.predict_frame(context, hidden, c0, repeated_codes));
    CHECK(codes == repeated_codes);

    std::vector<float> invalid_hidden(
        static_cast<size_t>(
            config.code_predictor.input_embedding_size - 1),
        0.0f);
    CHECK(!predictor.predict_frame(context, invalid_hidden, c0, codes));
    CHECK(!predictor.predict_frame(context, hidden, -1, codes));
    CHECK(!predictor.predict_frame(
        context,
        hidden,
        config.code_predictor.talker_codec_vocab_size,
        codes));

    return 0;
}
