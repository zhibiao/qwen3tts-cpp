#include "gguf_model.h"
#include "qwen3_tts.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr size_t kGroups = 16;
constexpr int32_t kMaxNewTokens = 50;
constexpr size_t kPrefillRows = 15;
constexpr size_t kHidden = 1024;

template <typename T>
std::vector<T> read_binary(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    CHECK(file.good());
    CHECK_EQ_I64(file.tellg(), static_cast<std::streamsize>(count * sizeof(T)));
    file.seekg(0);
    std::vector<T> data(count);
    file.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(T)));
    CHECK(file.good());
    return data;
}

template <typename T>
std::vector<T> read_all_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    CHECK(file.good());
    const std::streamsize size = file.tellg();
    CHECK(size >= 0);
    CHECK_EQ_I64(size % static_cast<std::streamsize>(sizeof(T)), 0);
    file.seekg(0);
    std::vector<T> data(
        static_cast<size_t>(size) / sizeof(T));
    file.read(
        reinterpret_cast<char*>(data.data()),
        size);
    CHECK(file.good());
    return data;
}

double cosine_similarity(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        dot += static_cast<double>(actual[i]) * expected[i];
        actual_norm += static_cast<double>(actual[i]) * actual[i];
        expected_norm += static_cast<double>(expected[i]) * expected[i];
    }
    return dot / std::sqrt(actual_norm * expected_norm);
}

} // namespace

int main() {
    std::unique_ptr<qwen3_tts::GGUFModel> talker_model;
    CHECK_STATUS(
        qwen3_tts::GGUFModel::open(QWEN3_TTS_TEST_GGUF_F32, talker_model));
    std::unique_ptr<qwen3_tts::GGUFModel> speech_model;
    CHECK_STATUS(qwen3_tts::GGUFModel::open(
        QWEN3_TTS_TEST_SPEECH_GGUF,
        speech_model));

    const std::string fixture_dir = QWEN3_TTS_CUSTOMVOICE_GOLDEN_DIR;
    const auto expected_codes = read_all_binary<int32_t>(
        fixture_dir + "/custom_voice_codes.bin");
    const auto expected_prefill = read_binary<float>(
        fixture_dir + "/custom_voice_prefill.bin",
        kPrefillRows * kHidden);
    const auto expected_pad = read_binary<float>(
        fixture_dir + "/custom_voice_tts_pad.bin",
        kHidden);
    const auto expected_audio = read_all_binary<float>(
        fixture_dir + "/custom_voice_audio.bin");
    CHECK_EQ_I64(expected_codes.size() % kGroups, 0);
    const int64_t expected_frames =
        static_cast<int64_t>(expected_codes.size() / kGroups);

    qwen3_tts::Qwen3TTSContext context;
    qwen3_tts::Qwen3TTSContextParams params;
    params.n_threads = 1;
    CHECK_STATUS(context.load(
        QWEN3_TTS_TEST_GGUF_F32,
        QWEN3_TTS_TEST_SPEECH_GGUF,
        params));
    qwen3_tts::Qwen3TTSModel model;
    CHECK_STATUS(model.load(context));
    const qwen3_tts::CodeGenerationOptions defaults =
        model.generation_defaults();
    CHECK_EQ_I64(*defaults.max_new_tokens, 8192);
    CHECK(defaults.talker.has_value());
    CHECK(defaults.code_predictor.has_value());
    qwen3_tts::Qwen3TTSConfig config;
    CHECK_STATUS(qwen3_tts::load_qwen3tts_config(
        *talker_model,
        *speech_model,
        config));

    const auto expected_text_ids = read_binary<int32_t>(
        fixture_dir + "/custom_voice_text_ids.bin",
        12);
    qwen3_tts::TextTokenizer text_tokenizer;
    CHECK_STATUS(text_tokenizer.load(
        config.talker.text_tokenizer,
        config.talker.prompt));
    std::vector<int32_t> actual_text_ids;
    CHECK_STATUS(text_tokenizer.encode_assistant(
        "你好，世界。",
        actual_text_ids));
    CHECK(actual_text_ids == expected_text_ids);

    qwen3_tts::CustomVoiceOptions voice;
    voice.speaker = "serena";
    voice.language = "Chinese";
    qwen3_tts::CodeGenerationOptions options;
    options.max_new_tokens = kMaxNewTokens;
    qwen3_tts::SamplingOptions talker_sampling;
    talker_sampling.do_sample = false;
    talker_sampling.repetition_penalty = 1.0f;
    options.talker = talker_sampling;
    qwen3_tts::SamplingOptions predictor_sampling;
    predictor_sampling.do_sample = false;
    predictor_sampling.repetition_penalty = 1.0f;
    options.code_predictor = predictor_sampling;

    std::vector<float> actual_prefill;
    std::vector<float> actual_pad;
    CHECK_STATUS(model.build_custom_voice_prompt(
        "你好，世界。",
        voice,
        actual_prefill,
        actual_pad));
    CHECK_EQ_I64(actual_prefill.size(), expected_prefill.size());
    float max_prefill_error = 0.0f;
    for (size_t i = 0; i < actual_prefill.size(); ++i) {
        max_prefill_error = std::max(
            max_prefill_error,
            std::fabs(actual_prefill[i] - expected_prefill[i]));
    }
    std::cout << "CustomVoice prefill: cosine="
              << cosine_similarity(actual_prefill, expected_prefill)
              << " max_abs_error=" << max_prefill_error << '\n';
    CHECK(cosine_similarity(actual_prefill, expected_prefill) >= 0.999999);
    CHECK(max_prefill_error <= 2e-4f);
    std::cout << "CustomVoice pad: cosine="
              << cosine_similarity(actual_pad, expected_pad) << '\n';
    CHECK(cosine_similarity(actual_pad, expected_pad) >= 0.999999);

    qwen3_tts::SpeechCodeMatrix codes;
    CHECK_STATUS(model.generate_custom_voice_codes(
        "你好，世界。",
        voice,
        options,
        codes));
    CHECK_EQ_I64(codes.frames, expected_frames);
    CHECK_EQ_I64(codes.groups, kGroups);
    for (int32_t frame = 0; frame < expected_frames; ++frame) {
        for (size_t group = 0; group < kGroups; ++group) {
            const size_t index =
                static_cast<size_t>(frame * kGroups) + group;
            if (codes.data[index] != expected_codes[index]) {
                std::cout << "First code mismatch at frame " << frame
                          << " group " << group
                          << ": actual=" << codes.data[index]
                          << " expected=" << expected_codes[index] << '\n';
            }
        }
    }
    CHECK(codes.data == expected_codes);

    qwen3_tts::AudioBuffer audio;
    CHECK_STATUS(model.generate_custom_voice(
        "你好，世界。",
        voice,
        options,
        audio));
    CHECK_EQ_I64(audio.sample_rate, 24000);
    CHECK_EQ_I64(audio.channels, 1);
    CHECK_EQ_I64(audio.samples.size(), expected_audio.size());
    float max_audio_error = 0.0f;
    for (size_t i = 0; i < audio.samples.size(); ++i) {
        max_audio_error = std::max(
            max_audio_error,
            std::fabs(audio.samples[i] - expected_audio[i]));
    }
    std::cout << "CustomVoice audio: cosine="
              << cosine_similarity(audio.samples, expected_audio)
              << " max_abs_error=" << max_audio_error << '\n';
    CHECK(cosine_similarity(audio.samples, expected_audio) >= 0.999f);
    CHECK(max_audio_error <= 1e-2f);

    voice.speaker = "not-a-speaker";
    CHECK(!model.generate_custom_voice_codes(
        "你好，世界。",
        voice,
        options,
        codes));
    voice.speaker = "serena";
    voice.language = "not-a-language";
    CHECK(!model.generate_custom_voice_codes(
        "你好，世界。",
        voice,
        options,
        codes));

    voice.language = "Chinese";
    voice.instruct = "Speak slowly.";
    CHECK(!model.generate_custom_voice_codes(
        "ä½ å¥½ï¼Œä¸–ç•Œã€‚",
        voice,
        options,
        codes));

    std::cout << "CustomVoice prompt, codes and PCM match PyTorch golden\n";
    return 0;
}
