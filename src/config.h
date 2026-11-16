#pragma once

#include "status.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qwen3_tts {

class GGUFModel;

enum class TtsModelType {
    CustomVoice,
    Base,
    VoiceDesign,
    Unknown,
};

enum class TokenizerType {
    Qwen3TTS12Hz,
    Unknown,
};

enum class GGUFModelRole {
    Talker,
    SpeechTokenizer,
    Unknown,
};

struct ModelGeneralConfig {
    std::string architecture;
    std::string name;
    uint32_t file_type = 0;
    uint32_t quantization_version = 0;
};

struct GGUFHeaderConfig {
    ModelGeneralConfig general;
    uint32_t schema_version = 0;
    GGUFModelRole role = GGUFModelRole::Unknown;
};

struct ModelIdentityConfig {
    TtsModelType model_type = TtsModelType::Unknown;
    std::string model_size;
    TokenizerType tokenizer_type = TokenizerType::Unknown;
};

struct PromptConfig {
    std::string assistant_prefix;
    std::string assistant_suffix;
    std::string reference_prefix;
    std::string reference_suffix;
    std::string instruction_prefix;
    std::string instruction_suffix;
};

struct TextTokenizerConfig {
    std::string model;
    int32_t vocab_size = 0;
    int32_t padding_token_id = -1;
    int32_t eos_token_id = -1;
    int32_t im_start_token_id = -1;
    int32_t im_end_token_id = -1;
    int32_t tts_pad_token_id = -1;
    int32_t tts_bos_token_id = -1;
    int32_t tts_eos_token_id = -1;
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<std::string> added_token_names;
    std::vector<int32_t> added_token_ids;
};

struct TalkerConfig {
    int32_t codec_vocab_size = 0;
    int32_t text_vocab_size = 0;
    int32_t text_embedding_length = 0;
    int32_t embedding_length = 0;
    int32_t feed_forward_length = 0;
    int32_t block_count = 0;
    int32_t attention_head_count = 0;
    int32_t attention_head_count_kv = 0;
    int32_t attention_key_length = 0;
    std::string hidden_activation;
    int32_t context_length = 0;
    int32_t num_ctx = 2048;
    float rms_norm_epsilon = 0.0f;
    bool attention_bias = false;
    bool use_sliding_window = false;
    int32_t sliding_window = 0;
    float rope_theta = 0.0f;
    std::string rope_type;
    std::vector<int32_t> rope_sections;
    bool rope_interleaved = false;
    bool rope_plain_neox = false;
    int32_t position_id_per_second = 0;
    int32_t num_code_groups = 0;
    int32_t codec_pad_token_id = -1;
    int32_t codec_bos_token_id = -1;
    int32_t codec_eos_token_id = -1;
    int32_t codec_think_id = -1;
    int32_t codec_nothink_id = -1;
    int32_t codec_think_bos_id = -1;
    int32_t codec_think_eos_id = -1;
};

struct CodePredictorConfig {
    int32_t vocab_size = 0;
    int32_t talker_codec_vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t input_embedding_size = 0;
    int32_t intermediate_size = 0;
    int32_t block_count = 0;
    int32_t attention_head_count = 0;
    int32_t attention_head_count_kv = 0;
    int32_t attention_key_length = 0;
    std::string hidden_activation;
    int32_t context_length = 0;
    float rms_norm_epsilon = 0.0f;
    bool attention_bias = false;
    bool use_sliding_window = false;
    int32_t sliding_window = 0;
    std::vector<std::string> layer_types;
    float rope_theta = 0.0f;
    std::string rope_type;
    int32_t num_code_groups = 0;
    bool has_input_projection = false;
    bool input_projection_bias = false;
};

struct SpeechDecoderConfig {
    int32_t latent_dim = 0;
    int32_t codebook_dim = 0;
    int32_t codebook_size = 0;
    int32_t semantic_codebook_size = 0;
    int32_t decoder_dim = 0;
    int32_t vector_quantization_hidden_dimension = 0;
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t block_count = 0;
    int32_t attention_head_count = 0;
    int32_t attention_head_count_kv = 0;
    int32_t attention_key_length = 0;
    std::string hidden_activation;
    int32_t context_length = 0;
    float rms_norm_epsilon = 0.0f;
    float layer_scale_initial_scale = 0.0f;
    bool attention_bias = false;
    bool use_sliding_window = false;
    int32_t sliding_window = 0;
    std::vector<std::string> layer_types;
    float rope_theta = 0.0f;
    std::string rope_type;
    int32_t num_quantizers = 0;
    int32_t num_semantic_quantizers = 0;
    std::vector<int32_t> upsample_rates;
    std::vector<int32_t> upsampling_ratios;
};

struct SpeechTokenizerConfig {
    std::string model_type;
    int32_t input_sample_rate = 0;
    int32_t output_sample_rate = 0;
    int32_t encode_downsample_rate = 0;
    int32_t decode_upsample_rate = 0;
    int32_t encoder_valid_num_quantizers = 0;
    SpeechDecoderConfig decoder;
};

struct GenerationConfig {
    int32_t max_new_tokens = 0;
    bool do_sample = true;
    float temperature = 0.0f;
    int32_t top_k = 0;
    float top_p = 0.0f;
    float repetition_penalty = 0.0f;
    bool subtalker_do_sample = true;
    float subtalker_temperature = 0.0f;
    int32_t subtalker_top_k = 0;
    float subtalker_top_p = 0.0f;
    float subtalker_repetition_penalty = 0.0f;
};

struct SpeakerConfig {
    std::string name;
    int32_t id = -1;
    std::string dialect;
};

struct LanguageConfig {
    std::string name;
    int32_t id = -1;
};

struct VoiceConfig {
    std::vector<SpeakerConfig> speakers;
    std::vector<LanguageConfig> languages;
};

struct TalkerGGUFConfig {
    GGUFHeaderConfig header;
    ModelIdentityConfig identity;
    TextTokenizerConfig text_tokenizer;
    PromptConfig prompt;
    TalkerConfig decoder;
    CodePredictorConfig code_predictor;
    GenerationConfig generation;
    VoiceConfig voices;
};

struct SpeechTokenizerGGUFConfig {
    GGUFHeaderConfig header;
    TokenizerType tokenizer_type = TokenizerType::Unknown;
    SpeechTokenizerConfig tokenizer;
};

struct Qwen3TTSConfig {
    TalkerGGUFConfig talker;
    SpeechTokenizerGGUFConfig speech;
};

Status load_talker_config(
    const GGUFModel& loader,
    TalkerGGUFConfig& config);

Status load_speech_config(
    const GGUFModel& loader,
    SpeechTokenizerGGUFConfig& config);

Status load_qwen3tts_config(
    const GGUFModel& talker_loader,
    const GGUFModel& speech_loader,
    Qwen3TTSConfig& config);

} // namespace qwen3_tts
