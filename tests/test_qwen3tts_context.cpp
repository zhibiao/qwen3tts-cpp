#include "qwen3_tts.h"
#include "test_assert.h"

#include <string>

int main() {
    qwen3_tts::Qwen3TTSContextParams params;
    params.device = "CPU";
    params.n_threads = 1;

    qwen3_tts::Qwen3TTSContext runtime_only;
    CHECK_STATUS(runtime_only.init_runtime(params));
    CHECK(runtime_only.backend() != nullptr);
    CHECK(!runtime_only.init_runtime(params));
    ggml_backend_sched_t scheduler = nullptr;
    CHECK_STATUS(runtime_only.create_scheduler(scheduler));
    CHECK(scheduler != nullptr);
    ggml_backend_sched_free(scheduler);

    qwen3_tts::Qwen3TTSContext invalid_device;
    qwen3_tts::Qwen3TTSContextParams invalid_params;
    invalid_params.device = "Qwen3TTSDeviceThatDoesNotExist";
    CHECK(!invalid_device.init_runtime(invalid_params));
    scheduler = nullptr;
    CHECK(!invalid_device.create_scheduler(scheduler));
    CHECK(scheduler == nullptr);

    qwen3_tts::Qwen3TTSContext context;
    CHECK_STATUS(context.load(
        QWEN3_TTS_TEST_GGUF,
        QWEN3_TTS_TEST_SPEECH_GGUF,
        params));
    CHECK(context.is_ready());
    CHECK(!context.load(
        QWEN3_TTS_TEST_GGUF,
        QWEN3_TTS_TEST_SPEECH_GGUF,
        params));

    qwen3_tts::Qwen3TTSModel model;
    CHECK(!model.load(invalid_device));
    CHECK_STATUS(model.load(context));
    CHECK(!model.load(context));
    CHECK_EQ_I64(*model.generation_defaults().max_new_tokens, 8192);
    return 0;
}
