#include "config.h"

#include "gguf_model.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace qwen3_tts {

namespace {

constexpr uint32_t kSchemaVersion = 3;

Status required_key(
    const GGUFModel& model,
    const std::string& key,
    gguf_type type) {
    if (!model.has_key(key)) {
        return Status::error("GGUF is missing required key: " + key);
    }
    const gguf_type actual = model.key_type(key);
    if (actual != type) {
        return Status::error(
            "GGUF key has unexpected type: " + key +
            " (expected " + std::to_string(static_cast<int>(type)) +
            ", got " + std::to_string(static_cast<int>(actual)) + ")");
    }
    return Status::ok();
}

Status read_bool(const GGUFModel& model, const std::string& key, bool& value) {
    Status status = required_key(model, key, GGUF_TYPE_BOOL);
    if (!status) {
        return status;
    }
    value = model.get_bool(key);
    return Status::ok();
}

Status read_i32(const GGUFModel& model, const std::string& key, int32_t& value) {
    Status status = required_key(model, key, GGUF_TYPE_INT32);
    if (!status) {
        return status;
    }
    value = model.get_i32(key);
    return Status::ok();
}

Status read_u32(const GGUFModel& model, const std::string& key, int32_t& value) {
    Status status = required_key(model, key, GGUF_TYPE_UINT32);
    if (!status) {
        return status;
    }
    const uint32_t raw = model.get_u32(key);
    if (raw > static_cast<uint32_t>(INT32_MAX)) {
        return Status::error("GGUF value is too large: " + key);
    }
    value = static_cast<int32_t>(raw);
    return Status::ok();
}

Status read_u32_raw(const GGUFModel& model, const std::string& key, uint32_t& value) {
    Status status = required_key(model, key, GGUF_TYPE_UINT32);
    if (!status) {
        return status;
    }
    value = model.get_u32(key);
    return Status::ok();
}

Status read_f32(const GGUFModel& model, const std::string& key, float& value) {
    Status status = required_key(model, key, GGUF_TYPE_FLOAT32);
    if (!status) {
        return status;
    }
    value = model.get_f32(key);
    return Status::ok();
}

Status read_string(
    const GGUFModel& model,
    const std::string& key,
    std::string& value) {
    Status status = required_key(model, key, GGUF_TYPE_STRING);
    if (!status) {
        return status;
    }
    value = model.get_string(key);
    return Status::ok();
}

gguf_type array_element_type(const GGUFModel& model, const std::string& key) {
    const gguf_context* gguf = model.raw_gguf();
    const int64_t id = gguf_find_key(gguf, key.c_str());
    return id < 0 ? GGUF_TYPE_COUNT : gguf_get_arr_type(gguf, id);
}

Status required_array(
    const GGUFModel& model,
    const std::string& key,
    gguf_type element_type) {
    Status status = required_key(model, key, GGUF_TYPE_ARRAY);
    if (!status) {
        return status;
    }
    if (array_element_type(model, key) != element_type) {
        return Status::error(
            "GGUF array has unexpected element type: " + key);
    }
    return Status::ok();
}

Status read_string_array(
    const GGUFModel& model,
    const std::string& key,
    std::vector<std::string>& values) {
    Status status = required_array(model, key, GGUF_TYPE_STRING);
    if (!status) {
        return status;
    }
    values = model.get_string_array(key);
    return Status::ok();
}

Status read_i32_array(
    const GGUFModel& model,
    const std::string& key,
    std::vector<int32_t>& values) {
    Status status = required_array(model, key, GGUF_TYPE_INT32);
    if (!status) {
        return status;
    }
    values = model.get_i32_array(key);
    return Status::ok();
}

Status require_positive(int32_t value, const std::string& name) {
    return value > 0
        ? Status::ok()
        : Status::error("configuration value must be positive: " + name);
}

Status require_non_negative(int32_t value, const std::string& name) {
    return value >= 0
        ? Status::ok()
        : Status::error("configuration value must be non-negative: " + name);
}

Status parse_model_type(const std::string& value, TtsModelType& type) {
    if (value == "custom_voice") {
        type = TtsModelType::CustomVoice;
    } else if (value == "base") {
        type = TtsModelType::Base;
    } else if (value == "voice_design") {
        type = TtsModelType::VoiceDesign;
    } else {
        return Status::error("unsupported tts_model_type: " + value);
    }
    return Status::ok();
}

Status parse_tokenizer_type(const std::string& value, TokenizerType& type) {
    if (value == "qwen3_tts_tokenizer_12hz") {
        type = TokenizerType::Qwen3TTS12Hz;
        return Status::ok();
    }
    return Status::error("unsupported tokenizer_type: " + value);
}

Status parse_role(const std::string& value, GGUFModelRole& role) {
    if (value == "talker") {
        role = GGUFModelRole::Talker;
        return Status::ok();
    }
    if (value == "speech_tokenizer") {
        role = GGUFModelRole::SpeechTokenizer;
        return Status::ok();
    }
    return Status::error("unsupported GGUF file_role: " + value);
}

Status load_prompt_config(
    const GGUFModel& model,
    PromptConfig& prompt) {
    const std::pair<const char*, std::string*> values[] = {
        {"qwen3_tts.prompt.assistant_prefix", &prompt.assistant_prefix},
        {"qwen3_tts.prompt.assistant_suffix", &prompt.assistant_suffix},
        {"qwen3_tts.prompt.reference_prefix", &prompt.reference_prefix},
        {"qwen3_tts.prompt.reference_suffix", &prompt.reference_suffix},
        {"qwen3_tts.prompt.instruction_prefix", &prompt.instruction_prefix},
        {"qwen3_tts.prompt.instruction_suffix", &prompt.instruction_suffix},
    };
    for (const auto& [key, destination] : values) {
        Status status = read_string(model, key, *destination);
        if (!status) {
            return status;
        }
    }
    return Status::ok();
}

Status load_text_tokenizer_config(
    const GGUFModel& model,
    TextTokenizerConfig& config) {
    Status status = read_string(
        model,
        "tokenizer.ggml.model",
        config.model);
    if (!status) {
        return status;
    }
    status = read_u32(
        model,
        "qwen3_tts.text.vocab_size",
        config.vocab_size);
    if (!status) {
        return status;
    }
    const std::pair<const char*, int32_t*> ids[] = {
        {"tokenizer.ggml.padding_token_id", &config.padding_token_id},
        {"tokenizer.ggml.eos_token_id", &config.eos_token_id},
        {"qwen3_tts.text.im_start_token_id", &config.im_start_token_id},
        {"qwen3_tts.text.im_end_token_id", &config.im_end_token_id},
        {"qwen3_tts.text.tts_pad_token_id", &config.tts_pad_token_id},
        {"qwen3_tts.text.tts_bos_token_id", &config.tts_bos_token_id},
        {"qwen3_tts.text.tts_eos_token_id", &config.tts_eos_token_id},
    };
    for (const auto& [key, destination] : ids) {
        status = read_i32(model, key, *destination);
        if (!status) {
            return status;
        }
    }
    status = read_string_array(
        model,
        "tokenizer.ggml.tokens",
        config.tokens);
    if (!status) {
        return status;
    }
    status = read_string_array(
        model,
        "tokenizer.ggml.merges",
        config.merges);
    if (!status) {
        return status;
    }
    status = read_string_array(
        model,
        "tokenizer.ggml.added_token_names",
        config.added_token_names);
    if (!status) {
        return status;
    }
    status = read_i32_array(
        model,
        "tokenizer.ggml.added_token_ids",
        config.added_token_ids);
    if (!status) {
        return status;
    }
    if (model.has_key("qwen3_tts.text.nfc_normalization")) {
        status = read_bool(
            model,
            "qwen3_tts.text.nfc_normalization",
            config.nfc_normalization);
        if (!status) {
            return status;
        }
    }
    if (config.model != "qwen2") {
        return Status::error(
            "unsupported text tokenizer model: " + config.model);
    }
    if (config.vocab_size <= 0 ||
        static_cast<size_t>(config.vocab_size) < config.tokens.size()) {
        return Status::error("invalid text tokenizer vocabulary size");
    }
    if (config.tokens.empty() || config.merges.empty()) {
        return Status::error("text tokenizer vocabulary is empty");
    }
    if (config.added_token_names.size() != config.added_token_ids.size()) {
        return Status::error(
            "text tokenizer added token arrays have different sizes");
    }
    for (int32_t id : {
             config.padding_token_id,
             config.eos_token_id,
             config.im_start_token_id,
             config.im_end_token_id,
             config.tts_pad_token_id,
             config.tts_bos_token_id,
             config.tts_eos_token_id}) {
        if (id < 0 || id >= static_cast<int32_t>(config.tokens.size())) {
            return Status::error(
                "text tokenizer special token ID is outside the vocabulary");
        }
    }
    return Status::ok();
}

Status load_talker_parameters(
    const GGUFModel& model,
    TalkerConfig& config) {
    const std::pair<const char*, int32_t*> dimensions[] = {
        {"qwen3_tts.talker.codec_vocab_size", &config.codec_vocab_size},
        {"qwen3_tts.talker.text_vocab_size", &config.text_vocab_size},
        {"qwen3_tts.talker.text_embedding_length", &config.text_embedding_length},
        {"qwen3_tts.talker.embedding_length", &config.embedding_length},
        {"qwen3_tts.talker.feed_forward_length", &config.feed_forward_length},
        {"qwen3_tts.talker.block_count", &config.block_count},
        {"qwen3_tts.talker.attention_head_count", &config.attention_head_count},
        {"qwen3_tts.talker.attention_head_count_kv", &config.attention_head_count_kv},
        {"qwen3_tts.talker.attention_key_length", &config.attention_key_length},
        {"qwen3_tts.talker.context_length", &config.context_length},
        {"qwen3_tts.talker.sliding_window", &config.sliding_window},
        {"qwen3_tts.talker.position_id_per_second", &config.position_id_per_second},
        {"qwen3_tts.talker.num_code_groups", &config.num_code_groups},
    };
    for (const auto& [key, destination] : dimensions) {
        Status status = read_u32(model, key, *destination);
        if (!status) {
            return status;
        }
    }
    if (model.has_key("qwen3_tts.talker.num_ctx")) {
        Status status = read_u32(
            model,
            "qwen3_tts.talker.num_ctx",
            config.num_ctx);
        if (!status) {
            return status;
        }
    }
    const std::pair<const char*, int32_t*> token_ids[] = {
        {"qwen3_tts.talker.codec_pad_token_id", &config.codec_pad_token_id},
        {"qwen3_tts.talker.codec_bos_token_id", &config.codec_bos_token_id},
        {"qwen3_tts.talker.codec_eos_token_id", &config.codec_eos_token_id},
        {"qwen3_tts.talker.codec_think_id", &config.codec_think_id},
        {"qwen3_tts.talker.codec_nothink_id", &config.codec_nothink_id},
        {"qwen3_tts.talker.codec_think_bos_id", &config.codec_think_bos_id},
        {"qwen3_tts.talker.codec_think_eos_id", &config.codec_think_eos_id},
    };
    for (const auto& [key, destination] : token_ids) {
        Status token_status = read_i32(model, key, *destination);
        if (!token_status) {
            return token_status;
        }
    }
    Status status = read_string(
        model,
        "qwen3_tts.talker.hidden_activation",
        config.hidden_activation);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "qwen3_tts.talker.rms_norm_epsilon",
        config.rms_norm_epsilon);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "qwen3_tts.talker.attention_bias",
        config.attention_bias);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "qwen3_tts.talker.use_sliding_window",
        config.use_sliding_window);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "qwen3_tts.talker.rope_theta",
        config.rope_theta);
    if (!status) {
        return status;
    }
    status = read_string(
        model,
        "qwen3_tts.talker.rope_type",
        config.rope_type);
    if (!status) {
        return status;
    }
    status = read_i32_array(
        model,
        "qwen3_tts.talker.rope_sections",
        config.rope_sections);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "qwen3_tts.talker.rope_interleaved",
        config.rope_interleaved);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "qwen3_tts.talker.rope_plain_neox",
        config.rope_plain_neox);
    if (!status) {
        return status;
    }

    for (const auto& [value, name] : std::vector<std::pair<int32_t, const char*>>{
             {config.codec_vocab_size, "talker.codec_vocab_size"},
             {config.text_vocab_size, "talker.text_vocab_size"},
             {config.text_embedding_length, "talker.text_embedding_length"},
             {config.embedding_length, "talker.embedding_length"},
             {config.feed_forward_length, "talker.feed_forward_length"},
             {config.block_count, "talker.block_count"},
             {config.attention_head_count, "talker.attention_head_count"},
             {config.attention_head_count_kv, "talker.attention_head_count_kv"},
             {config.attention_key_length, "talker.attention_key_length"},
             {config.context_length, "talker.context_length"},
             {config.num_ctx, "talker.num_ctx"},
             {config.position_id_per_second, "talker.position_id_per_second"},
             {config.num_code_groups, "talker.num_code_groups"}}) {
        status = require_positive(value, name);
        if (!status) {
            return status;
        }
    }
    if (config.attention_head_count % config.attention_head_count_kv != 0) {
        return Status::error(
            "Talker attention heads must be divisible by KV heads");
    }
    if (config.num_ctx > config.context_length) {
        return Status::error(
            "Talker num_ctx exceeds the model context_length");
    }
    if (config.rope_sections.empty() || config.rope_sections.size() > 3) {
        return Status::error("Talker mRoPE sections are invalid");
    }
    for (int32_t group_id : {
             config.codec_pad_token_id,
             config.codec_bos_token_id,
             config.codec_eos_token_id,
             config.codec_think_id,
             config.codec_nothink_id,
             config.codec_think_bos_id,
             config.codec_think_eos_id}) {
        if (group_id < 0 || group_id >= config.codec_vocab_size) {
            return Status::error(
                "Talker codec control token ID is outside the vocabulary");
        }
    }
    return Status::ok();
}

Status load_code_predictor_config(
    const GGUFModel& model,
    CodePredictorConfig& config,
    const TalkerConfig& talker) {
    const std::pair<const char*, int32_t*> values[] = {
        {"qwen3_tts.code_predictor.vocab_size", &config.vocab_size},
        {"qwen3_tts.code_predictor.talker_codec_vocab_size", &config.talker_codec_vocab_size},
        {"qwen3_tts.code_predictor.hidden_size", &config.hidden_size},
        {"qwen3_tts.code_predictor.input_embedding_size", &config.input_embedding_size},
        {"qwen3_tts.code_predictor.intermediate_size", &config.intermediate_size},
        {"qwen3_tts.code_predictor.block_count", &config.block_count},
        {"qwen3_tts.code_predictor.attention_head_count", &config.attention_head_count},
        {"qwen3_tts.code_predictor.attention_head_count_kv", &config.attention_head_count_kv},
        {"qwen3_tts.code_predictor.attention_key_length", &config.attention_key_length},
        {"qwen3_tts.code_predictor.context_length", &config.context_length},
        {"qwen3_tts.code_predictor.sliding_window", &config.sliding_window},
        {"qwen3_tts.code_predictor.num_code_groups", &config.num_code_groups},
    };
    for (const auto& [key, destination] : values) {
        Status status = read_u32(model, key, *destination);
        if (!status) {
            return status;
        }
    }
    Status status = read_string(
        model,
        "qwen3_tts.code_predictor.hidden_activation",
        config.hidden_activation);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "qwen3_tts.code_predictor.rms_norm_epsilon",
        config.rms_norm_epsilon);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "qwen3_tts.code_predictor.attention_bias",
        config.attention_bias);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "qwen3_tts.code_predictor.use_sliding_window",
        config.use_sliding_window);
    if (!status) {
        return status;
    }
    status = read_string_array(
        model,
        "qwen3_tts.code_predictor.layer_types",
        config.layer_types);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "qwen3_tts.code_predictor.rope_theta",
        config.rope_theta);
    if (!status) {
        return status;
    }
    status = read_string(
        model,
        "qwen3_tts.code_predictor.rope_type",
        config.rope_type);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "qwen3_tts.code_predictor.has_input_projection",
        config.has_input_projection);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "qwen3_tts.code_predictor.input_projection_bias",
        config.input_projection_bias);
    if (!status) {
        return status;
    }

    for (const auto& [value, name] : std::vector<std::pair<int32_t, const char*>>{
             {config.vocab_size, "code_predictor.vocab_size"},
             {config.talker_codec_vocab_size, "code_predictor.talker_codec_vocab_size"},
             {config.hidden_size, "code_predictor.hidden_size"},
             {config.input_embedding_size, "code_predictor.input_embedding_size"},
             {config.intermediate_size, "code_predictor.intermediate_size"},
             {config.block_count, "code_predictor.block_count"},
             {config.attention_head_count, "code_predictor.attention_head_count"},
             {config.attention_head_count_kv, "code_predictor.attention_head_count_kv"},
             {config.attention_key_length, "code_predictor.attention_key_length"},
             {config.context_length, "code_predictor.context_length"},
             {config.num_code_groups, "code_predictor.num_code_groups"}}) {
        status = require_positive(value, name);
        if (!status) {
            return status;
        }
    }
    if (config.attention_head_count % config.attention_head_count_kv != 0) {
        return Status::error(
            "CodePredictor attention heads must be divisible by KV heads");
    }
    if (config.num_code_groups != talker.num_code_groups) {
        return Status::error(
            "Talker and CodePredictor num_code_groups differ");
    }
    if (config.input_embedding_size != talker.embedding_length) {
        return Status::error(
            "CodePredictor input embedding size does not match Talker hidden size");
    }
    if (config.talker_codec_vocab_size != talker.codec_vocab_size) {
        return Status::error(
            "CodePredictor talker codec vocabulary does not match Talker");
    }
    if (config.has_input_projection !=
        (config.input_embedding_size != config.hidden_size)) {
        return Status::error(
            "CodePredictor input projection flag does not match dimensions");
    }
    if (config.layer_types.size() !=
        static_cast<size_t>(config.block_count)) {
        return Status::error(
            "CodePredictor layer_types size does not match block_count");
    }
    return Status::ok();
}

Status load_speech_parameters(
    const GGUFModel& model,
    SpeechTokenizerConfig& config) {
    Status status = read_string(
        model,
        "qwen3_tts.speech.model_type",
        config.model_type);
    if (!status) {
        return status;
    }
    const std::pair<const char*, int32_t*> rates[] = {
        {"qwen3_tts.speech.input_sample_rate", &config.input_sample_rate},
        {"qwen3_tts.speech.output_sample_rate", &config.output_sample_rate},
        {"qwen3_tts.speech.encode_downsample_rate", &config.encode_downsample_rate},
        {"qwen3_tts.speech.decode_upsample_rate", &config.decode_upsample_rate},
        {"qwen3_tts.speech.encoder_valid_num_quantizers", &config.encoder_valid_num_quantizers},
    };
    for (const auto& [key, destination] : rates) {
        status = read_u32(model, key, *destination);
        if (!status) {
            return status;
        }
    }

    SpeechDecoderConfig& decoder = config.decoder;
    const std::pair<const char*, int32_t*> decoder_values[] = {
        {"qwen3_tts.speech.decoder.latent_dim", &decoder.latent_dim},
        {"qwen3_tts.speech.decoder.codebook_dim", &decoder.codebook_dim},
        {"qwen3_tts.speech.decoder.codebook_size", &decoder.codebook_size},
        {"qwen3_tts.speech.decoder.semantic_codebook_size", &decoder.semantic_codebook_size},
        {"qwen3_tts.speech.decoder.decoder_dim", &decoder.decoder_dim},
        {"qwen3_tts.speech.decoder.vector_quantization_hidden_dimension", &decoder.vector_quantization_hidden_dimension},
        {"qwen3_tts.speech.decoder.hidden_size", &decoder.hidden_size},
        {"qwen3_tts.speech.decoder.intermediate_size", &decoder.intermediate_size},
        {"qwen3_tts.speech.decoder.block_count", &decoder.block_count},
        {"qwen3_tts.speech.decoder.attention_head_count", &decoder.attention_head_count},
        {"qwen3_tts.speech.decoder.attention_head_count_kv", &decoder.attention_head_count_kv},
        {"qwen3_tts.speech.decoder.attention_key_length", &decoder.attention_key_length},
        {"qwen3_tts.speech.decoder.context_length", &decoder.context_length},
        {"qwen3_tts.speech.decoder.sliding_window", &decoder.sliding_window},
        {"qwen3_tts.speech.decoder.num_quantizers", &decoder.num_quantizers},
        {"qwen3_tts.speech.decoder.num_semantic_quantizers", &decoder.num_semantic_quantizers},
    };
    for (const auto& [key, destination] : decoder_values) {
        status = read_u32(model, key, *destination);
        if (!status) {
            return status;
        }
    }
    status = read_string(
        model,
        "qwen3_tts.speech.decoder.hidden_activation",
        decoder.hidden_activation);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "qwen3_tts.speech.decoder.rms_norm_epsilon",
        decoder.rms_norm_epsilon);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "qwen3_tts.speech.decoder.layer_scale_initial_scale",
        decoder.layer_scale_initial_scale);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "qwen3_tts.speech.decoder.attention_bias",
        decoder.attention_bias);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "qwen3_tts.speech.decoder.use_sliding_window",
        decoder.use_sliding_window);
    if (!status) {
        return status;
    }
    status = read_string_array(
        model,
        "qwen3_tts.speech.decoder.layer_types",
        decoder.layer_types);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "qwen3_tts.speech.decoder.rope_theta",
        decoder.rope_theta);
    if (!status) {
        return status;
    }
    status = read_string(
        model,
        "qwen3_tts.speech.decoder.rope_type",
        decoder.rope_type);
    if (!status) {
        return status;
    }
    status = read_i32_array(
        model,
        "qwen3_tts.speech.decoder.upsample_rates",
        decoder.upsample_rates);
    if (!status) {
        return status;
    }
    status = read_i32_array(
        model,
        "qwen3_tts.speech.decoder.upsampling_ratios",
        decoder.upsampling_ratios);
    if (!status) {
        return status;
    }

    if (config.model_type != "qwen3_tts_tokenizer_12hz") {
        return Status::error(
            "unsupported speech tokenizer model_type: " + config.model_type);
    }
    for (const auto& [value, name] : std::vector<std::pair<int32_t, const char*>>{
             {config.input_sample_rate, "speech.input_sample_rate"},
             {config.output_sample_rate, "speech.output_sample_rate"},
             {config.encode_downsample_rate, "speech.encode_downsample_rate"},
             {config.decode_upsample_rate, "speech.decode_upsample_rate"},
             {config.encoder_valid_num_quantizers, "speech.encoder_valid_num_quantizers"},
             {decoder.latent_dim, "speech.decoder.latent_dim"},
             {decoder.codebook_dim, "speech.decoder.codebook_dim"},
             {decoder.codebook_size, "speech.decoder.codebook_size"},
             {decoder.semantic_codebook_size, "speech.decoder.semantic_codebook_size"},
             {decoder.decoder_dim, "speech.decoder.decoder_dim"},
             {decoder.hidden_size, "speech.decoder.hidden_size"},
             {decoder.intermediate_size, "speech.decoder.intermediate_size"},
             {decoder.block_count, "speech.decoder.block_count"},
             {decoder.attention_head_count, "speech.decoder.attention_head_count"},
             {decoder.attention_head_count_kv, "speech.decoder.attention_head_count_kv"},
             {decoder.attention_key_length, "speech.decoder.attention_key_length"},
             {decoder.context_length, "speech.decoder.context_length"},
             {decoder.num_quantizers, "speech.decoder.num_quantizers"},
             {decoder.num_semantic_quantizers, "speech.decoder.num_semantic_quantizers"}}) {
        status = require_positive(value, name);
        if (!status) {
            return status;
        }
    }
    if (decoder.codebook_dim % 2 != 0) {
        return Status::error(
            "SpeechDecoder vector quantization dimensions are inconsistent");
    }
    if (decoder.attention_head_count % decoder.attention_head_count_kv != 0) {
        return Status::error(
            "SpeechDecoder attention heads must be divisible by KV heads");
    }
    if (decoder.layer_types.size() !=
        static_cast<size_t>(decoder.block_count)) {
        return Status::error(
            "SpeechDecoder layer_types size does not match block_count");
    }
    if (decoder.num_semantic_quantizers >= decoder.num_quantizers) {
        return Status::error(
            "SpeechDecoder semantic quantizers must be fewer than all quantizers");
    }
    int64_t upsample_rate = 1;
    for (int32_t rate : decoder.upsample_rates) {
        if (rate <= 0) {
            return Status::error("SpeechDecoder upsample rate is invalid");
        }
        upsample_rate *= rate;
    }
    for (int32_t ratio : decoder.upsampling_ratios) {
        if (ratio <= 0) {
            return Status::error("SpeechDecoder upsampling ratio is invalid");
        }
        upsample_rate *= ratio;
    }
    if (upsample_rate != config.decode_upsample_rate) {
        return Status::error(
            "SpeechDecoder upsample rates do not match decode_upsample_rate");
    }
    return Status::ok();
}

Status load_generation_config(
    const GGUFModel& model,
    GenerationConfig& config) {
    Status status = read_u32(
        model,
        "qwen3_tts.generation.max_new_tokens",
        config.max_new_tokens);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "qwen3_tts.generation.do_sample",
        config.do_sample);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "qwen3_tts.generation.temperature",
        config.temperature);
    if (!status) {
        return status;
    }
    status = read_u32(
        model,
        "qwen3_tts.generation.top_k",
        config.top_k);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "qwen3_tts.generation.top_p",
        config.top_p);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "qwen3_tts.generation.repetition_penalty",
        config.repetition_penalty);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "qwen3_tts.generation.subtalker_do_sample",
        config.subtalker_do_sample);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "qwen3_tts.generation.subtalker_temperature",
        config.subtalker_temperature);
    if (!status) {
        return status;
    }
    status = read_u32(
        model,
        "qwen3_tts.generation.subtalker_top_k",
        config.subtalker_top_k);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "qwen3_tts.generation.subtalker_top_p",
        config.subtalker_top_p);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "qwen3_tts.generation.subtalker_repetition_penalty",
        config.subtalker_repetition_penalty);
    if (!status) {
        return status;
    }
    if (config.max_new_tokens <= 0 || config.top_k < 0 ||
        config.subtalker_top_k < 0 ||
        !std::isfinite(config.temperature) ||
        !std::isfinite(config.subtalker_temperature) ||
        !std::isfinite(config.top_p) ||
        !std::isfinite(config.subtalker_top_p) ||
        !std::isfinite(config.subtalker_repetition_penalty) ||
        !std::isfinite(config.repetition_penalty) ||
        config.temperature <= 0.0f ||
        config.subtalker_temperature <= 0.0f ||
        config.repetition_penalty < 1.0f ||
        config.subtalker_repetition_penalty < 1.0f) {
        return Status::error("GGUF generation configuration is invalid");
    }
    return Status::ok();
}

Status load_header(
    const GGUFModel& model,
    GGUFModelRole expected_role,
    const std::string& expected_architecture,
    GGUFHeaderConfig& config) {
    Status status = read_string(
        model,
        "general.architecture",
        config.general.architecture);
    if (!status) {
        return status;
    }
    status = read_string(
        model,
        "general.name",
        config.general.name);
    if (!status) {
        return status;
    }
    status = read_u32_raw(
        model,
        "general.file_type",
        config.general.file_type);
    if (!status) {
        return status;
    }
    status = read_u32_raw(
        model,
        "general.quantization_version",
        config.general.quantization_version);
    if (!status) {
        return status;
    }
    status = read_u32_raw(
        model,
        "qwen3_tts.schema_version",
        config.schema_version);
    if (!status) {
        return status;
    }
    if (config.schema_version != kSchemaVersion) {
        return Status::error(
            "unsupported Qwen3-TTS GGUF schema version " +
            std::to_string(config.schema_version) +
            "; expected " + std::to_string(kSchemaVersion));
    }
    std::string role;
    status = read_string(model, "qwen3_tts.file_role", role);
    if (!status) {
        return status;
    }
    status = parse_role(role, config.role);
    if (!status) {
        return status;
    }
    if (config.role != expected_role) {
        return Status::error("GGUF file_role does not match requested model");
    }
    if (config.general.architecture != expected_architecture) {
        return Status::error(
            "unsupported GGUF architecture: " +
            config.general.architecture);
    }
    return Status::ok();
}

Status load_voice_config(
    const GGUFModel& model,
    VoiceConfig& config) {
    std::vector<std::string> speaker_names;
    std::vector<std::string> speaker_dialects;
    std::vector<int32_t> speaker_ids;
    Status status = read_string_array(
        model,
        "qwen3_tts.speaker.names",
        speaker_names);
    if (!status) {
        return status;
    }
    status = read_i32_array(
        model,
        "qwen3_tts.speaker.ids",
        speaker_ids);
    if (!status) {
        return status;
    }
    status = read_string_array(
        model,
        "qwen3_tts.speaker.dialects",
        speaker_dialects);
    if (!status) {
        return status;
    }
    if (speaker_names.size() != speaker_ids.size() ||
        speaker_names.size() != speaker_dialects.size()) {
        return Status::error(
            "speaker name/id/dialect arrays have different sizes");
    }

    config.speakers.clear();
    config.speakers.reserve(speaker_names.size());
    for (size_t i = 0; i < speaker_names.size(); ++i) {
        if (speaker_names[i].empty()) {
            return Status::error("speaker name is empty");
        }
        config.speakers.push_back({
            speaker_names[i],
            speaker_ids[i],
            speaker_dialects[i],
        });
    }

    std::vector<std::string> language_names;
    std::vector<int32_t> language_ids;
    status = read_string_array(
        model,
        "qwen3_tts.language.names",
        language_names);
    if (!status) {
        return status;
    }
    status = read_i32_array(
        model,
        "qwen3_tts.language.ids",
        language_ids);
    if (!status) {
        return status;
    }
    if (language_names.size() != language_ids.size()) {
        return Status::error(
            "language name/id arrays have different sizes");
    }
    config.languages.clear();
    config.languages.reserve(language_names.size());
    for (size_t i = 0; i < language_names.size(); ++i) {
        if (language_names[i].empty()) {
            return Status::error("language name is empty");
        }
        config.languages.push_back({language_names[i], language_ids[i]});
    }
    return Status::ok();
}

} // namespace

Status load_talker_config(
    const GGUFModel& loader,
    TalkerGGUFConfig& config) {
    config = {};
    Status status = load_header(
        loader,
        GGUFModelRole::Talker,
        "qwen3-tts",
        config.header);
    if (!status) {
        return status;
    }

    std::string model_type;
    status = read_string(loader, "qwen3_tts.model_type", model_type);
    if (!status) {
        return status;
    }
    status = parse_model_type(model_type, config.identity.model_type);
    if (!status) {
        return status;
    }
    status = read_string(
        loader,
        "qwen3_tts.model_size",
        config.identity.model_size);
    if (!status) {
        return status;
    }
    std::string tokenizer_type;
    status = read_string(
        loader,
        "qwen3_tts.tokenizer_type",
        tokenizer_type);
    if (!status) {
        return status;
    }
    status = parse_tokenizer_type(
        tokenizer_type,
        config.identity.tokenizer_type);
    if (!status) {
        return status;
    }

    status = load_text_tokenizer_config(loader, config.text_tokenizer);
    if (!status) {
        return status;
    }
    status = load_prompt_config(loader, config.prompt);
    if (!status) {
        return status;
    }
    status = load_talker_parameters(loader, config.decoder);
    if (!status) {
        return status;
    }
    status = load_code_predictor_config(
        loader,
        config.code_predictor,
        config.decoder);
    if (!status) {
        return status;
    }
    status = load_generation_config(loader, config.generation);
    if (!status) {
        return status;
    }
    status = load_voice_config(loader, config.voices);
    if (!status) {
        return status;
    }
    if (config.text_tokenizer.vocab_size !=
        config.decoder.text_vocab_size) {
        return Status::error(
            "text tokenizer vocab size does not match Talker text vocab size");
    }
    return Status::ok();
}

Status load_speech_config(
    const GGUFModel& loader,
    SpeechTokenizerGGUFConfig& config) {
    config = {};
    Status status = load_header(
        loader,
        GGUFModelRole::SpeechTokenizer,
        "qwen3-tts-tokenizer",
        config.header);
    if (!status) {
        return status;
    }
    std::string tokenizer_type;
    status = read_string(
        loader,
        "qwen3_tts.tokenizer_type",
        tokenizer_type);
    if (!status) {
        return status;
    }
    status = parse_tokenizer_type(tokenizer_type, config.tokenizer_type);
    if (!status) {
        return status;
    }
    return load_speech_parameters(loader, config.tokenizer);
}

Status load_qwen3tts_config(
    const GGUFModel& talker_loader,
    const GGUFModel& speech_loader,
    Qwen3TTSConfig& config) {
    config = {};
    Status status = load_talker_config(talker_loader, config.talker);
    if (!status) {
        return status;
    }
    status = load_speech_config(speech_loader, config.speech);
    if (!status) {
        return status;
    }
    if (config.talker.header.schema_version !=
        config.speech.header.schema_version) {
        return Status::error("Talker and Speech GGUF schema versions differ");
    }
    if (config.talker.identity.tokenizer_type !=
        config.speech.tokenizer_type) {
        return Status::error(
            "Talker and Speech tokenizer types differ");
    }
    if (config.talker.code_predictor.vocab_size !=
        config.speech.tokenizer.decoder.codebook_size) {
        return Status::error(
            "CodePredictor vocabulary does not match SpeechDecoder codebooks");
    }
    if (config.talker.decoder.num_code_groups !=
        config.speech.tokenizer.decoder.num_quantizers) {
        return Status::error(
            "Talker and SpeechDecoder code group counts differ");
    }
    return Status::ok();
}

} // namespace qwen3_tts
