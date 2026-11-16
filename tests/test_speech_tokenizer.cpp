#include "config.h"
#include "qwen3_tts.h"
#include "speech_tokenizer.h"
#include "test_assert.h"

#include <cmath>
#include <iostream>

int main() {
    qwen3_tts::Qwen3TTSContext context;
    qwen3_tts::Qwen3TTSContextParams params;
    params.n_threads = 1;
    CHECK_STATUS(context.load(
        QWEN3_TTS_TEST_GGUF,
        QWEN3_TTS_TEST_SPEECH_GGUF,
        params));

    qwen3_tts::SpeechTokenizerGGUFConfig config;
    CHECK_STATUS(qwen3_tts::load_speech_config(
        context.speech_gguf(),
        config));

    qwen3_tts::SpeechTokenizer tokenizer;
    const qwen3_tts::Status load_status = tokenizer.load(
        config.tokenizer,
        context);
    if (!load_status) {
        std::cerr << load_status.message() << '\n';
        return 1;
    }

    qwen3_tts::SpeechCodeMatrix codes;
    codes.frames = 2;
    codes.groups = 16;
    codes.data.resize(static_cast<size_t>(codes.frames) * codes.groups);
    for (int32_t frame = 0; frame < codes.frames; ++frame) {
        for (int32_t group = 0; group < codes.groups; ++group) {
            codes.data[static_cast<size_t>(frame) * codes.groups + group] =
                (frame + group) % 8;
        }
    }

    qwen3_tts::AudioBuffer audio;
    const qwen3_tts::Status decode_status =
        tokenizer.decode(context, codes, audio);
    if (!decode_status) {
        std::cerr << decode_status.message() << '\n';
        return 1;
    }
    CHECK_EQ_I64(audio.sample_rate, 24000);
    CHECK_EQ_I64(audio.channels, 1);
    CHECK_EQ_I64(audio.samples.size(), 2 * 1920);

    double energy = 0.0;
    for (float sample : audio.samples) {
        CHECK(std::isfinite(sample));
        CHECK(sample >= -1.0f && sample <= 1.0f);
        energy += static_cast<double>(sample) * sample;
    }
    CHECK(energy > 1e-12);

    codes.frames = 3;
    codes.data.resize(static_cast<size_t>(codes.frames) * codes.groups);
    for (size_t i = 0; i < codes.data.size(); ++i) {
        codes.data[i] = static_cast<int32_t>(i % 16);
    }
    CHECK_STATUS(tokenizer.decode(context, codes, audio));
    CHECK_EQ_I64(audio.samples.size(), 3 * 1920);

    qwen3_tts::SpeechCodeMatrix invalid = codes;
    invalid.frames = 0;
    CHECK(!tokenizer.decode(context, invalid, audio));
    CHECK(audio.samples.empty());

    invalid = codes;
    invalid.groups = 15;
    CHECK(!tokenizer.decode(context, invalid, audio));
    CHECK(audio.samples.empty());

    invalid = codes;
    invalid.data.pop_back();
    CHECK(!tokenizer.decode(context, invalid, audio));
    CHECK(audio.samples.empty());

    invalid = codes;
    invalid.data[0] = 2048;
    CHECK(!tokenizer.decode(context, invalid, audio));
    CHECK(audio.samples.empty());

    return 0;
}
