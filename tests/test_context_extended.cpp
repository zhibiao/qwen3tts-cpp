#include "qwen3_tts.h"
#include "test_assert.h"

#include <iostream>

int main() {
    qwen3_tts::Qwen3TTSContextParams invalid;
    invalid.device = "CPU";
    invalid.graph_nodes = 0;
    qwen3_tts::Qwen3TTSContext invalid_context;
    CHECK(!invalid_context.init_runtime(invalid));

    qwen3_tts::Qwen3TTSContextParams params;
    params.device = "CPU";
    params.n_threads = 1;

    qwen3_tts::Qwen3TTSContext swapped;
    CHECK_STATUS(swapped.load(
        QWEN3_TTS_TEST_SPEECH_GGUF,
        QWEN3_TTS_TEST_GGUF,
        params));
    qwen3_tts::Qwen3TTSModel swapped_model;
    CHECK(!swapped_model.load(swapped));

    qwen3_tts::Qwen3TTSContext context;
    CHECK_STATUS(context.load(
        QWEN3_TTS_TEST_GGUF,
        QWEN3_TTS_TEST_SPEECH_GGUF,
        params));
    qwen3_tts::Qwen3TTSModel model;
    CHECK_STATUS(model.load(context));
    const qwen3_tts::CodeGenerationOptions defaults =
        model.generation_defaults();
    CHECK(defaults.max_new_tokens.has_value());
    CHECK(defaults.codec_eos_token_id.has_value());
    CHECK(defaults.talker.has_value());
    CHECK(defaults.code_predictor.has_value());

#if defined(QWEN3_TTS_TEST_GGUF_Q8)
    qwen3_tts::Qwen3TTSContext quantized_context;
    CHECK_STATUS(quantized_context.load(
        QWEN3_TTS_TEST_GGUF_Q8,
        QWEN3_TTS_TEST_SPEECH_GGUF,
        params));
    qwen3_tts::Qwen3TTSModel quantized_model;
    CHECK_STATUS(quantized_model.load(quantized_context));
    CHECK_EQ_I64(
        *quantized_model.generation_defaults().max_new_tokens,
        *defaults.max_new_tokens);
#endif

    std::cout << "Context/model lifecycle passed\n";
    return 0;
}
