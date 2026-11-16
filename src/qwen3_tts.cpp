#include "qwen3_tts.h"

#include "gguf_model.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace qwen3_tts {

namespace {

std::string lowercase(const std::string& value) {
    std::string result = value;
    std::transform(
        result.begin(),
        result.end(),
        result.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
    return result;
}

void append_rows(
    std::vector<float>& destination,
    const std::vector<float>& source,
    int64_t rows,
    int64_t hidden) {
    if (source.size() != static_cast<size_t>(rows * hidden)) {
        throw std::runtime_error("embedding row count mismatch");
    }
    destination.insert(destination.end(), source.begin(), source.end());
}

int32_t default_thread_count() {
    const uint32_t logical = std::thread::hardware_concurrency();
    return static_cast<int32_t>(std::max<uint32_t>(logical / 2, 1));
}

void set_backend_threads(ggml_backend_t backend, int32_t n_threads) {
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_reg_t registry =
        device == nullptr ? nullptr : ggml_backend_dev_backend_reg(device);
    if (registry == nullptr) {
        return;
    }
    auto set_threads =
        reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(
                registry,
                "ggml_backend_set_n_threads"));
    if (set_threads != nullptr) {
        set_threads(backend, n_threads);
    }
}

ggml_backend_sched_t make_scheduler(
    ggml_backend_t backend,
    ggml_backend_t cpu_backend,
    int32_t graph_nodes) {
    ggml_backend_t backends[2] = {backend, cpu_backend};
    ggml_backend_buffer_type_t buffer_types[2] = {nullptr, nullptr};
    const int count = backend == cpu_backend ? 1 : 2;
    buffer_types[0] = ggml_backend_get_default_buffer_type(backend);
    if (count == 2) {
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        ggml_backend_buffer_type_t host =
            device == nullptr
            ? nullptr
            : ggml_backend_dev_host_buffer_type(device);
        buffer_types[1] = host != nullptr
            ? host
            : ggml_backend_get_default_buffer_type(cpu_backend);
    }
    return ggml_backend_sched_new(
        backends,
        buffer_types,
        count,
        static_cast<size_t>(graph_nodes),
        false,
        true);
}

} // namespace

Qwen3TTSContext::Qwen3TTSContext() = default;

Qwen3TTSContext::~Qwen3TTSContext() {
    release();
}

Status Qwen3TTSContext::init_runtime(
    const Qwen3TTSContextParams& params) {
    if (runtime_ready_) {
        return Status::error("Qwen3TTSContext runtime is already initialized");
    }
    if (params.graph_nodes <= 0) {
        return Status::error("Qwen3TTSContext graph_nodes must be positive");
    }

    ggml_backend_load_all();
    backend_ = params.device.empty()
        ? ggml_backend_init_best()
        : ggml_backend_init_by_name(params.device.c_str(), nullptr);
    if (backend_ == nullptr) {
        return Status::error(
            params.device.empty()
            ? "failed to initialize the best GGML backend"
            : "failed to initialize GGML device: " + params.device);
    }

    const int32_t n_threads = params.n_threads > 0
        ? params.n_threads
        : default_thread_count();
    set_backend_threads(backend_, n_threads);
    if (std::strcmp(ggml_backend_name(backend_), "CPU") == 0) {
        cpu_backend_ = backend_;
    } else {
        cpu_backend_ = ggml_backend_init_by_name("CPU", nullptr);
        if (cpu_backend_ == nullptr) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
            return Status::error("failed to initialize GGML CPU fallback");
        }
        set_backend_threads(cpu_backend_, n_threads);
    }

    graph_nodes_ = params.graph_nodes;
    runtime_ready_ = true;
    return Status::ok();
}

bool Qwen3TTSContext::is_ready() const noexcept {
    return runtime_ready_ && gguf_ready_ &&
        talker_gguf_ != nullptr && speech_gguf_ != nullptr;
}

ggml_backend_t Qwen3TTSContext::backend() const {
    return backend_;
}

const GGUFModel& Qwen3TTSContext::talker_gguf() const {
    return *talker_gguf_;
}

const GGUFModel& Qwen3TTSContext::speech_gguf() const {
    return *speech_gguf_;
}

Status Qwen3TTSContext::create_scheduler(
    ggml_backend_sched_t& scheduler) const {
    scheduler = nullptr;
    if (!runtime_ready_ || backend_ == nullptr || cpu_backend_ == nullptr) {
        return Status::error("Qwen3TTSContext runtime is not initialized");
    }
    scheduler = make_scheduler(
        backend_,
        cpu_backend_,
        graph_nodes_);
    if (scheduler == nullptr) {
        return Status::error("failed to create Qwen3TTS graph scheduler");
    }
    return Status::ok();
}

Status Qwen3TTSContext::load(
    const std::string& talker_path,
    const std::string& speech_path,
    const Qwen3TTSContextParams& params) {
    if (gguf_ready_) {
        return Status::error("Qwen3TTSContext is already loaded");
    }
    if (!runtime_ready_) {
        Status status = init_runtime(params);
        if (!status) {
            return status;
        }
    }

    std::unique_ptr<GGUFModel> talker_gguf;
    Status status = GGUFModel::open(talker_path, talker_gguf);
    if (!status) {
        return status;
    }
    std::unique_ptr<GGUFModel> speech_gguf;
    status = GGUFModel::open(speech_path, speech_gguf);
    if (!status) {
        return status;
    }

    talker_gguf_ = std::move(talker_gguf);
    speech_gguf_ = std::move(speech_gguf);
    gguf_ready_ = true;
    return Status::ok();
}

void Qwen3TTSContext::release() noexcept {
    if (cpu_backend_ != nullptr && cpu_backend_ != backend_) {
        ggml_backend_free(cpu_backend_);
    }
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
    }
    backend_ = nullptr;
    cpu_backend_ = nullptr;
    graph_nodes_ = 0;
    speech_gguf_.reset();
    talker_gguf_.reset();
    runtime_ready_ = false;
    gguf_ready_ = false;
}

Qwen3TTSModel::Qwen3TTSModel() = default;

Qwen3TTSModel::~Qwen3TTSModel() = default;

Status Qwen3TTSModel::load(Qwen3TTSContext& context) {
    if (loaded_) {
        return Status::error("Qwen3TTSModel is already loaded");
    }
    if (!context.is_ready()) {
        return Status::error("Qwen3TTSContext is not ready");
    }

    Qwen3TTSConfig config;
    Status status = load_qwen3tts_config(
        context.talker_gguf(),
        context.speech_gguf(),
        config);
    if (!status) {
        return status;
    }
    if (config.talker.identity.model_type != TtsModelType::CustomVoice) {
        return Status::error(
            "Qwen3TTSModel currently supports custom_voice models only");
    }
    if (config.talker.identity.tokenizer_type !=
        TokenizerType::Qwen3TTS12Hz) {
        return Status::error(
            "Qwen3TTSModel currently supports the 12 Hz tokenizer only");
    }

    auto talker = std::make_unique<Talker>();
    status = talker->load_tensors(
        config.talker.decoder,
        context);
    if (!status) {
        return status;
    }
    auto code_predictor = std::make_unique<CodePredictor>();
    status = code_predictor->load(
        config.talker.code_predictor,
        context);
    if (!status) {
        return status;
    }
    status = code_predictor->init_graphs(context);
    if (!status) {
        return status;
    }
    status = text_tokenizer_.load(
        config.talker.text_tokenizer,
        config.talker.prompt);
    if (!status) {
        return status;
    }
    auto speech_tokenizer = std::make_unique<SpeechTokenizer>();
    status = speech_tokenizer->load(
        config.speech.tokenizer,
        context);
    if (!status) {
        return status;
    }

    speaker_ids_.clear();
    speaker_dialects_.clear();
    for (const SpeakerConfig& speaker : config.talker.voices.speakers) {
        speaker_ids_[lowercase(speaker.name)] = speaker.id;
        speaker_dialects_[lowercase(speaker.name)] =
            lowercase(speaker.dialect);
    }
    language_ids_.clear();
    for (const LanguageConfig& language : config.talker.voices.languages) {
        language_ids_[lowercase(language.name)] = language.id;
    }
    config.talker.text_tokenizer.tokens.clear();
    config.talker.text_tokenizer.merges.clear();
    config.talker.text_tokenizer.added_token_names.clear();
    config.talker.text_tokenizer.added_token_ids.clear();

    talker_ = std::move(talker);
    code_predictor_ = std::move(code_predictor);
    speech_tokenizer_ = std::move(speech_tokenizer);
    config_ = std::move(config);
    context_ = &context;
    loaded_ = true;
    return Status::ok();
}

int32_t Qwen3TTSModel::named_id(
    const std::unordered_map<std::string, int32_t>& values,
    const std::string& name) const {
    const auto it = values.find(lowercase(name));
    return it == values.end() ? -1 : it->second;
}

CodeGenerationOptions Qwen3TTSModel::generation_defaults() const {
    CodeGenerationOptions options;
    if (!loaded_) {
        return options;
    }
    options.max_new_tokens = config_.talker.generation.max_new_tokens;
    options.codec_eos_token_id = config_.talker.decoder.codec_eos_token_id;

    SamplingOptions talker;
    talker.do_sample = config_.talker.generation.do_sample;
    talker.temperature = config_.talker.generation.temperature;
    talker.top_k = config_.talker.generation.top_k;
    talker.top_p = config_.talker.generation.top_p;
    talker.repetition_penalty = config_.talker.generation.repetition_penalty;
    options.talker = talker;

    SamplingOptions predictor;
    predictor.do_sample = config_.talker.generation.subtalker_do_sample;
    predictor.temperature = config_.talker.generation.subtalker_temperature;
    predictor.top_k = config_.talker.generation.subtalker_top_k;
    predictor.top_p = config_.talker.generation.subtalker_top_p;
    predictor.repetition_penalty =
        config_.talker.generation.subtalker_repetition_penalty;
    options.code_predictor = predictor;
    return options;
}

Status Qwen3TTSModel::build_custom_voice_prompt(
    const std::string& text,
    const CustomVoiceOptions& voice_options,
    std::vector<float>& prefill_embeddings,
    std::vector<float>& tts_pad_embedding) {
    if (context_ == nullptr) {
        return Status::error("Qwen3TTSModel is not loaded");
    }
    Qwen3TTSContext& context = *context_;
    prefill_embeddings.clear();
    tts_pad_embedding.clear();
    if (!loaded_) {
        return Status::error("Qwen3TTS is not loaded");
    }
    if (text.empty()) {
        return Status::error("CustomVoice text is empty");
    }
    if (!voice_options.instruct.empty()) {
        return Status::error(
            "CustomVoice instruct is configured but not implemented");
    }

    const std::string speaker_name = lowercase(voice_options.speaker);
    const int32_t speaker_id = named_id(
        speaker_ids_,
        speaker_name);
    if (speaker_id < 0) {
        return Status::error(
            "unsupported CustomVoice speaker: " + voice_options.speaker);
    }

    int32_t language_id = -1;
    const std::string language_name = lowercase(voice_options.language);
    if (language_name != "auto") {
        language_id = named_id(language_ids_, language_name);
        if (language_id < 0) {
            return Status::error(
                "unsupported CustomVoice language: " + voice_options.language);
        }
    }
    const auto dialect = speaker_dialects_.find(speaker_name);
    if ((language_name == "chinese" || language_name == "auto") &&
        dialect != speaker_dialects_.end() &&
        !dialect->second.empty()) {
        language_id = named_id(language_ids_, dialect->second);
        if (language_id < 0) {
            return Status::error(
                "speaker dialect is missing from language IDs: " +
                dialect->second);
        }
    }

    std::vector<int32_t> input_ids;
    Status status = text_tokenizer_.encode_assistant(text, input_ids);
    if (!status) {
        return status;
    }
    if (input_ids.size() < 8) {
        return Status::error("CustomVoice assistant prompt is too short");
    }
    const std::vector<int32_t> role_ids(
        input_ids.begin(),
        input_ids.begin() + 3);
    const std::vector<int32_t> text_ids(
        input_ids.begin() + 3,
        input_ids.end() - 5);

    const std::vector<int32_t> special_ids = {
        config_.talker.text_tokenizer.tts_bos_token_id,
        config_.talker.text_tokenizer.tts_eos_token_id,
        config_.talker.text_tokenizer.tts_pad_token_id,
    };
    std::vector<int32_t> all_text_ids;
    all_text_ids.reserve(
        role_ids.size() + text_ids.size() + special_ids.size());
    all_text_ids.insert(all_text_ids.end(), role_ids.begin(), role_ids.end());
    all_text_ids.insert(all_text_ids.end(), text_ids.begin(), text_ids.end());
    all_text_ids.insert(
        all_text_ids.end(),
        special_ids.begin(),
        special_ids.end());

    std::vector<float> all_text_embedding;
    status = talker_->embed_text_tokens(
        context,
        all_text_ids,
        all_text_embedding);
    if (!status) {
        return status;
    }
    const int64_t hidden = config_.talker.decoder.embedding_length;
    const size_t role_size =
        static_cast<size_t>(role_ids.size() * hidden);
    const size_t text_size =
        static_cast<size_t>(text_ids.size() * hidden);
    const std::vector<float> role_embedding(
        all_text_embedding.begin(),
        all_text_embedding.begin() + role_size);
    const std::vector<float> text_embedding(
        all_text_embedding.begin() + role_size,
        all_text_embedding.begin() + role_size + text_size);
    const std::vector<float> special_text_embedding(
        all_text_embedding.begin() + role_size + text_size,
        all_text_embedding.end());
    const std::vector<float> tts_bos_embedding(
        special_text_embedding.begin(),
        special_text_embedding.begin() + hidden);
    const std::vector<float> tts_eos_embedding(
        special_text_embedding.begin() + hidden,
        special_text_embedding.begin() + 2 * hidden);
    const std::vector<float> tts_pad_row(
        special_text_embedding.begin() + 2 * hidden,
        special_text_embedding.end());

    std::vector<int32_t> control_ids = language_id < 0
        ? std::vector<int32_t>{
            config_.talker.decoder.codec_nothink_id,
            config_.talker.decoder.codec_think_bos_id,
            config_.talker.decoder.codec_think_eos_id,
        }
        : std::vector<int32_t>{
            config_.talker.decoder.codec_think_id,
            config_.talker.decoder.codec_think_bos_id,
            language_id,
            config_.talker.decoder.codec_think_eos_id,
        };
    control_ids.push_back(speaker_id);
    control_ids.push_back(config_.talker.decoder.codec_pad_token_id);

    const int64_t control_rows = static_cast<int64_t>(control_ids.size());
    std::vector<int32_t> all_codec_ids = control_ids;
    all_codec_ids.push_back(config_.talker.decoder.codec_pad_token_id);
    all_codec_ids.push_back(config_.talker.decoder.codec_bos_token_id);

    std::vector<float> all_codec_embedding;
    status = talker_->embed_codec_tokens(
        context,
        all_codec_ids,
        all_codec_embedding);
    if (!status) {
        return status;
    }
    const size_t codec_row_size = static_cast<size_t>(hidden);
    const size_t control_size =
        static_cast<size_t>(control_rows) * codec_row_size;
    const std::vector<float> control_embedding(
        all_codec_embedding.begin(),
        all_codec_embedding.begin() + control_size);
    const std::vector<float> codec_pad_embedding(
        all_codec_embedding.begin() + control_size,
        all_codec_embedding.begin() + control_size + codec_row_size);
    const std::vector<float> codec_bos_embedding(
        all_codec_embedding.begin() + control_size + codec_row_size,
        all_codec_embedding.end());

    prefill_embeddings.reserve(
        static_cast<size_t>(
            3 + control_rows + text_ids.size() + 2) * hidden);
    append_rows(prefill_embeddings, role_embedding, 3, hidden);

    for (int64_t row = 0; row < control_rows; ++row) {
        const float* text_side = row + 1 == control_rows
            ? tts_bos_embedding.data()
            : tts_pad_row.data();
        const float* codec_side =
            control_embedding.data() + row * hidden;
        for (int64_t i = 0; i < hidden; ++i) {
            prefill_embeddings.push_back(
                text_side[i] + codec_side[i]);
        }
    }

    const int64_t text_rows = static_cast<int64_t>(text_ids.size());
    for (int64_t row = 0; row < text_rows; ++row) {
        for (int64_t i = 0; i < hidden; ++i) {
            prefill_embeddings.push_back(
                text_embedding[static_cast<size_t>(row * hidden + i)] +
                codec_pad_embedding[static_cast<size_t>(i)]);
        }
    }
    for (int64_t i = 0; i < hidden; ++i) {
        prefill_embeddings.push_back(
            tts_eos_embedding[static_cast<size_t>(i)] +
            codec_pad_embedding[static_cast<size_t>(i)]);
    }
    for (int64_t i = 0; i < hidden; ++i) {
        prefill_embeddings.push_back(
            tts_pad_row[static_cast<size_t>(i)] +
            codec_bos_embedding[static_cast<size_t>(i)]);
    }

    tts_pad_embedding = tts_pad_row;
    return Status::ok();
}

Status Qwen3TTSModel::generate_custom_voice_codes(
    const std::string& text,
    const CustomVoiceOptions& voice_options,
    const CodeGenerationOptions& options,
    SpeechCodeMatrix& codes) {
    if (context_ == nullptr) {
        return Status::error("Qwen3TTSModel is not loaded");
    }
    Qwen3TTSContext& context = *context_;
    codes = {};
    std::vector<float> prefill;
    std::vector<float> tts_pad_embedding;
    const auto prompt_start = std::chrono::steady_clock::now();
    Status status = build_custom_voice_prompt(
        text,
        voice_options,
        prefill,
        tts_pad_embedding);
    if (!status) {
        return status;
    }
    if (options.log != nullptr) {
        std::ostringstream line;
        line << std::fixed << std::setprecision(2)
             << "prompt: "
             << std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    prompt_start).count()
             << " ms";
        options.log(line.str());
    }

    return generate_codes(
        prefill,
        static_cast<int64_t>(
            prefill.size() / config_.talker.decoder.embedding_length),
        {},
        0,
        tts_pad_embedding,
        options,
        codes);
}

Status Qwen3TTSModel::generate_custom_voice(
    const std::string& text,
    const CustomVoiceOptions& voice_options,
    const CodeGenerationOptions& options,
    AudioBuffer& audio) {
    if (context_ == nullptr) {
        return Status::error("Qwen3TTSModel is not loaded");
    }
    Qwen3TTSContext& context = *context_;
    audio = {};
    SpeechCodeMatrix codes;
    Status status = generate_custom_voice_codes(
        text,
        voice_options,
        options,
        codes);
    if (!status) {
        return status;
    }
    return speech_tokenizer_->decode(context, codes, audio);
}

Status Qwen3TTSModel::decode_codes(
    const SpeechCodeMatrix& codes,
    AudioBuffer& audio) {
    if (context_ == nullptr) {
        return Status::error("Qwen3TTSModel is not loaded");
    }
    Qwen3TTSContext& context = *context_;
    if (!loaded_) {
        audio = {};
        return Status::error("Qwen3TTS is not loaded");
    }
    return speech_tokenizer_->decode(context, codes, audio);
}

Status Qwen3TTSModel::generate_codes(
    const std::vector<float>& prefill_embeddings,
    int64_t prefill_frames,
    const std::vector<float>& trailing_text_hidden,
    int64_t trailing_frames,
    const std::vector<float>& tts_pad_embedding,
    const CodeGenerationOptions& options,
    SpeechCodeMatrix& codes) {
    if (context_ == nullptr) {
        return Status::error("Qwen3TTSModel is not loaded");
    }
    Qwen3TTSContext& context = *context_;
    codes = {};
    if (!loaded_) {
        return Status::error("Qwen3TTS is not loaded");
    }
    const int64_t hidden = config_.talker.decoder.embedding_length;
    const int64_t codec_vocab = config_.talker.decoder.codec_vocab_size;
    const int32_t num_code_groups = config_.talker.decoder.num_code_groups;
    codes.groups = num_code_groups;
    if (prefill_frames <= 0 ||
        prefill_embeddings.size() !=
            static_cast<size_t>(prefill_frames * hidden)) {
        return Status::error("invalid Qwen3TTS prefill embeddings");
    }
    if (trailing_frames < 0 ||
        trailing_text_hidden.size() !=
            static_cast<size_t>(trailing_frames * hidden)) {
        return Status::error("invalid Qwen3TTS trailing text hidden");
    }
    if (tts_pad_embedding.size() != static_cast<size_t>(hidden)) {
        return Status::error(
            "Qwen3TTS tts_pad embedding size does not match Talker hidden");
    }
    const int32_t max_new_tokens = options.max_new_tokens.value_or(
        config_.talker.generation.max_new_tokens);
    if (max_new_tokens <= 0) {
        return Status::error("Qwen3TTS max_new_tokens must be positive");
    }
    if (options.seed != 0) {
        talker_sampler_.seed(options.seed);
        code_predictor_->set_seed(options.seed ^ 0x9e3779b97f4a7c15ULL);
    }
    const int32_t eos_token_id = options.codec_eos_token_id.value_or(
        config_.talker.decoder.codec_eos_token_id);

    SamplingOptions talker_sampling;
    talker_sampling.do_sample = config_.talker.generation.do_sample;
    talker_sampling.temperature = config_.talker.generation.temperature;
    talker_sampling.top_k = config_.talker.generation.top_k;
    talker_sampling.top_p = config_.talker.generation.top_p;
    talker_sampling.repetition_penalty =
        config_.talker.generation.repetition_penalty;
    if (options.talker) {
        talker_sampling = *options.talker;
    }

    SamplingOptions predictor_sampling;
    predictor_sampling.do_sample = config_.talker.generation.subtalker_do_sample;
    predictor_sampling.temperature = config_.talker.generation.subtalker_temperature;
    predictor_sampling.top_k = config_.talker.generation.subtalker_top_k;
    predictor_sampling.top_p = config_.talker.generation.subtalker_top_p;
    predictor_sampling.repetition_penalty =
        config_.talker.generation.subtalker_repetition_penalty;
    if (options.code_predictor) {
        predictor_sampling = *options.code_predictor;
    }

    std::vector<float> last_hidden;
    std::vector<float> logits;
    const auto prefill_start = std::chrono::steady_clock::now();
    Status status = talker_->prefill(
        context,
        prefill_embeddings,
        prefill_frames,
        last_hidden,
        logits);
    if (!status) {
        return status;
    }
    if (options.log != nullptr) {
        std::ostringstream line;
        line << std::fixed << std::setprecision(2)
             << "talker prefill: "
             << std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    prefill_start).count()
             << " ms frames=" << prefill_frames;
        options.log(line.str());
    }
    if (last_hidden.size() !=
            static_cast<size_t>(prefill_frames * hidden) ||
        logits.size() !=
            static_cast<size_t>(prefill_frames * codec_vocab)) {
        return Status::error("unexpected Talker prefill output shape");
    }
    last_hidden.erase(
        last_hidden.begin(),
        last_hidden.end() - hidden);
    logits.erase(
        logits.begin(),
        logits.end() - codec_vocab);

    codes.frames = 0;
    codes.data.clear();
    codes.data.reserve(
        static_cast<size_t>(max_new_tokens * num_code_groups));
    std::vector<int32_t> c0_history;
    std::vector<uint8_t> talker_allowed_mask(
        static_cast<size_t>(codec_vocab),
        0);
    for (int32_t i = 0; i < config_.talker.code_predictor.vocab_size; ++i) {
        talker_allowed_mask[static_cast<size_t>(i)] = 1;
    }
    if (eos_token_id >= 0 && eos_token_id < codec_vocab) {
        talker_allowed_mask[static_cast<size_t>(eos_token_id)] = 1;
    }
    for (int32_t frame_index = 0;
         frame_index < max_new_tokens;
         ++frame_index) {
        const auto frame_start = std::chrono::steady_clock::now();
        if (last_hidden.size() != static_cast<size_t>(hidden) ||
            logits.size() != static_cast<size_t>(codec_vocab)) {
            return Status::error("unexpected Talker output shape");
        }

        int32_t c0 = -1;
        const auto sampling_start = std::chrono::steady_clock::now();
        status = talker_sampler_.select(
            logits,
            talker_sampling,
            c0_history,
            talker_allowed_mask,
            c0);
        if (!status) {
            return status;
        }
        const double sampling_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                sampling_start).count();
        if (c0 == eos_token_id) {
            if (options.log != nullptr) {
                std::ostringstream line;
                line << std::fixed << std::setprecision(2)
                     << "frame " << frame_index + 1 << '/'
                     << max_new_tokens
                     << ": eos sample=" << sampling_ms
                     << " ms total="
                     << std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() -
                            frame_start).count()
                     << " ms";
                options.log(line.str());
            }
            break;
        }
        c0_history.push_back(c0);

        std::vector<int32_t> frame;
        const auto predictor_start = std::chrono::steady_clock::now();
        std::function<void(const std::string&)> predictor_log;
        if (options.log != nullptr) {
            predictor_log =
                [log = options.log, frame = frame_index + 1](
                    const std::string& line) {
                    log("frame " + std::to_string(frame) +
                        " " + line);
                };
        }
        status = code_predictor_->predict_frame(
            context,
            last_hidden,
            c0,
            frame,
            predictor_sampling,
            nullptr,
            predictor_log);
        if (!status) {
            return status;
        }
        const double predictor_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                predictor_start).count();

        codes.data.insert(codes.data.end(), frame.begin(), frame.end());
        codes.frames += 1;
        if (frame_index + 1 >= max_new_tokens) {
            if (options.log != nullptr) {
                std::ostringstream line;
                line << std::fixed << std::setprecision(2)
                     << "frame " << frame_index + 1 << '/'
                     << max_new_tokens
                     << ": sample=" << sampling_ms
                     << " ms predictor=" << predictor_ms
                     << " ms total="
                     << std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() -
                            frame_start).count()
                     << " ms";
                options.log(line.str());
            }
            break;
        }

        std::vector<float> codec_embedding;
        const auto embedding_start = std::chrono::steady_clock::now();
        status = code_predictor_->embed_codes(
            context,
            frame,
            codec_embedding);
        if (!status) {
            return status;
        }
        const double embedding_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                embedding_start).count();

        std::vector<float> next_embedding(static_cast<size_t>(hidden));
        const auto condition_start = std::chrono::steady_clock::now();
        const float* condition = frame_index < trailing_frames
            ? trailing_text_hidden.data() + frame_index * hidden
            : tts_pad_embedding.data();
        for (int64_t i = 0; i < hidden; ++i) {
            next_embedding[static_cast<size_t>(i)] =
                codec_embedding[static_cast<size_t>(i)] + condition[i];
        }
        const double condition_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                condition_start).count();

        const auto decode_start = std::chrono::steady_clock::now();
        status = talker_->decode_step(
            context,
            next_embedding,
            last_hidden,
            logits);
        if (!status) {
            return status;
        }
        const double decode_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                decode_start).count();
        if (options.log != nullptr) {
            std::ostringstream line;
            line << std::fixed << std::setprecision(2)
                 << "frame " << frame_index + 1 << '/'
                 << max_new_tokens
                 << ": sample=" << sampling_ms
                 << " ms predictor=" << predictor_ms
                 << " ms embed=" << embedding_ms
                 << " ms condition=" << condition_ms
                 << " ms talker-decode=" << decode_ms
                 << " ms total="
                 << std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() -
                        frame_start).count()
                 << " ms";
            options.log(line.str());
        }
    }
    return Status::ok();
}

} // namespace qwen3_tts
