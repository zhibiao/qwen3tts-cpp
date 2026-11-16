#pragma once

#include "code_predictor.h"
#include "config.h"
#include "sampler.h"
#include "speech_tokenizer.h"
#include "status.h"
#include "talker.h"
#include "text_tokenizer.h"
#include "types.h"

#include "ggml-backend.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen3_tts {

class GGUFModel;

struct CodeGenerationOptions {
    std::optional<int32_t> max_new_tokens;
    std::optional<int32_t> codec_eos_token_id;
    std::optional<SamplingOptions> talker;
    std::optional<SamplingOptions> code_predictor;
    std::function<void(const std::string&)> log;
    uint64_t seed = 0;
};

struct CustomVoiceOptions {
    std::string speaker = "serena";
    std::string language = "Chinese";
    std::string instruct;
};

struct Qwen3TTSContextParams {
    std::string device;
    int32_t n_threads = 0;
    int32_t graph_nodes = 8192;
};

class Qwen3TTSContext {
public:
    Qwen3TTSContext();
    ~Qwen3TTSContext();

    Qwen3TTSContext(const Qwen3TTSContext&) = delete;
    Qwen3TTSContext& operator=(const Qwen3TTSContext&) = delete;

    Status init_runtime(const Qwen3TTSContextParams& params);

    Status load(
        const std::string& talker_path,
        const std::string& speech_path,
        const Qwen3TTSContextParams& params = {});

    bool is_ready() const noexcept;
    ggml_backend_t backend() const;
    const GGUFModel& talker_gguf() const;
    const GGUFModel& speech_gguf() const;
    Status create_scheduler(ggml_backend_sched_t& scheduler) const;

private:
    void release() noexcept;

    std::unique_ptr<GGUFModel> talker_gguf_;
    std::unique_ptr<GGUFModel> speech_gguf_;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    int32_t graph_nodes_ = 0;
    bool runtime_ready_ = false;
    bool gguf_ready_ = false;
};

class Qwen3TTSModel {
public:
    Qwen3TTSModel();
    ~Qwen3TTSModel();

    Qwen3TTSModel(const Qwen3TTSModel&) = delete;
    Qwen3TTSModel& operator=(const Qwen3TTSModel&) = delete;

    Status load(Qwen3TTSContext& context);

    CodeGenerationOptions generation_defaults() const;

    Status generate_codes(
        Qwen3TTSContext& context,
        const std::vector<float>& prefill_embeddings,
        int64_t prefill_frames,
        const std::vector<float>& trailing_text_hidden,
        int64_t trailing_frames,
        const std::vector<float>& tts_pad_embedding,
        const CodeGenerationOptions& options,
        SpeechCodeMatrix& codes);

    Status generate_custom_voice_codes(
        Qwen3TTSContext& context,
        const std::string& text,
        const CustomVoiceOptions& voice_options,
        const CodeGenerationOptions& options,
        SpeechCodeMatrix& codes);

    Status generate_custom_voice(
        Qwen3TTSContext& context,
        const std::string& text,
        const CustomVoiceOptions& voice_options,
        const CodeGenerationOptions& options,
        AudioBuffer& audio);

    Status decode_codes(
        Qwen3TTSContext& context,
        const SpeechCodeMatrix& codes,
        AudioBuffer& audio);

    Status build_custom_voice_prompt(
        Qwen3TTSContext& context,
        const std::string& text,
        const CustomVoiceOptions& voice_options,
        std::vector<float>& prefill_embeddings,
        std::vector<float>& tts_pad_embedding);

private:
    int32_t named_id(
        const std::unordered_map<std::string, int32_t>& values,
        const std::string& name) const;

    std::unique_ptr<Talker> talker_;
    std::unique_ptr<CodePredictor> code_predictor_;
    std::unique_ptr<SpeechTokenizer> speech_tokenizer_;
    TextTokenizer text_tokenizer_;
    Sampler talker_sampler_;
    Qwen3TTSConfig config_;
    std::unordered_map<std::string, int32_t> speaker_ids_;
    std::unordered_map<std::string, int32_t> language_ids_;
    std::unordered_map<std::string, std::string> speaker_dialects_;
    bool loaded_ = false;
};

} // namespace qwen3_tts
