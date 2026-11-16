#include "gguf_model.h"
#include "qwen3_tts.h"
#include "test_assert.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr size_t kHidden = 1024;
constexpr size_t kGroups = 16;
constexpr int64_t kPrefillFrames = 2;
constexpr int64_t kTrailingFrames = 2;
constexpr int32_t kMaxNewFrames = 3;

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

} // namespace

int main() {
    const std::string fixture_dir = QWEN3_TTS_STREAM_GOLDEN_DIR;
    const auto prefill = read_binary<float>(
        fixture_dir + "/qwen3tts_prefill.bin",
        kPrefillFrames * kHidden);
    const auto trailing = read_binary<float>(
        fixture_dir + "/qwen3tts_trailing.bin",
        kTrailingFrames * kHidden);
    const auto pad = read_binary<float>(
        fixture_dir + "/qwen3tts_pad.bin",
        kHidden);
    const auto expected_codes = read_binary<int32_t>(
        fixture_dir + "/qwen3tts_codes.bin",
        kMaxNewFrames * kGroups);

    qwen3_tts::Qwen3TTSContext context;
    qwen3_tts::Qwen3TTSContextParams params;
    params.n_threads = 1;
    CHECK_STATUS(context.load(
        QWEN3_TTS_TEST_GGUF_F32,
        QWEN3_TTS_TEST_SPEECH_GGUF,
        params));
    qwen3_tts::Qwen3TTSModel model;
    CHECK_STATUS(model.load(context));

    qwen3_tts::SpeechCodeMatrix codes;
    qwen3_tts::CodeGenerationOptions options;
    options.max_new_tokens = kMaxNewFrames;
    qwen3_tts::SamplingOptions talker_sampling;
    talker_sampling.do_sample = false;
    talker_sampling.repetition_penalty = 1.0f;
    options.talker = talker_sampling;
    qwen3_tts::SamplingOptions predictor_sampling;
    predictor_sampling.do_sample = false;
    predictor_sampling.repetition_penalty = 1.0f;
    options.code_predictor = predictor_sampling;
    CHECK_STATUS(model.generate_codes(
        context,
        prefill,
        kPrefillFrames,
        trailing,
        kTrailingFrames,
        pad,
        options,
        codes));
    CHECK_EQ_I64(codes.frames, kMaxNewFrames);
    CHECK_EQ_I64(codes.groups, kGroups);
    CHECK(codes.data == expected_codes);

    qwen3_tts::SpeechCodeMatrix repeated;
    CHECK_STATUS(model.generate_codes(
        context,
        prefill,
        kPrefillFrames,
        trailing,
        kTrailingFrames,
        pad,
        options,
        repeated));
    CHECK(codes.data == repeated.data);

    options.codec_eos_token_id = expected_codes[0];
    qwen3_tts::SpeechCodeMatrix stopped;
    CHECK_STATUS(model.generate_codes(
        context,
        prefill,
        kPrefillFrames,
        trailing,
        kTrailingFrames,
        pad,
        options,
        stopped));
    CHECK_EQ_I64(stopped.frames, 0);
    CHECK(stopped.data.empty());

    std::cout << "Qwen3TTS multi-frame codes match PyTorch golden\n";
    return 0;
}
