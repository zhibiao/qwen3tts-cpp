#include "code_predictor.h"

#include "gguf_model.h"
#include "qwen3_tts.h"


#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen3_tts {

namespace {

ggml_tensor* cast_f32(ggml_context* ctx, ggml_tensor* tensor) {
    return tensor->type == GGML_TYPE_F32 ? tensor : ggml_cast(ctx, tensor, GGML_TYPE_F32);
}

ggml_tensor* repeat_columns(
    ggml_context* ctx,
    ggml_tensor* vector,
    ggml_tensor* like) {
    return ggml_repeat(
        ctx,
        ggml_reshape_2d(ctx, cast_f32(ctx, vector), vector->ne[0], 1),
        like);
}

ggml_tensor* linear(
    ggml_context* ctx,
    ggml_tensor* weight,
    ggml_tensor* input) {
    return ggml_mul_mat(ctx, weight, cast_f32(ctx, input));
}

ggml_tensor* rms_norm(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* weight,
    float eps) {
    ggml_tensor* normalized = ggml_rms_norm(ctx, input, eps);
    return ggml_mul(ctx, normalized, weight);
}

bool is_norm_weight(const std::string& name) {
    return name.find("norm.weight") != std::string::npos;
}

void copy_source_tensor(
    ggml_tensor* destination,
    const ggml_tensor* source) {
    if (destination->type == source->type) {
        ggml_backend_tensor_set(
            destination,
            source->data,
            0,
            ggml_nbytes(source));
        return;
    }
    if (destination->type != GGML_TYPE_F32) {
        throw std::runtime_error(
            "unsupported CodePredictor weight conversion");
    }

    std::vector<float> converted(
        static_cast<size_t>(ggml_nelements(source)));
    if (source->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(
            static_cast<const ggml_fp16_t*>(source->data),
            converted.data(),
            static_cast<int64_t>(converted.size()));
    } else if (source->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row(
            static_cast<const ggml_bf16_t*>(source->data),
            converted.data(),
            static_cast<int64_t>(converted.size()));
    } else {
        throw std::runtime_error(
            "unsupported CodePredictor norm weight dtype");
    }
    ggml_backend_tensor_set(
        destination,
        converted.data(),
        0,
        converted.size() * sizeof(float));
}

ggml_tensor* attention(
    ggml_context* ctx,
    ggml_tensor* input,
    const std::string& prefix,
    ggml_tensor* positions,
    ggml_tensor* cache_positions,
    ggml_tensor* attention_mask,
    ggml_tensor* cache_k,
    ggml_tensor* cache_v,
    int64_t layer,
    int64_t sequence_length,
    std::vector<ggml_tensor*>& cache_writes,
    const std::unordered_map<std::string, ggml_tensor*>& weights,
    const CodePredictorConfig& config) {
    const int64_t head_dim = config.attention_key_length;
    const int64_t heads = config.attention_head_count;
    const int64_t kv_heads = config.attention_head_count_kv;
    const auto get = [&](const std::string& suffix) {
        const auto it = weights.find(prefix + suffix);
        if (it == weights.end()) {
            throw std::runtime_error("missing GGUF tensor: " + prefix + suffix);
        }
        return it->second;
    };

    ggml_tensor* query = linear(ctx, get("q_proj.weight"), input);
    ggml_tensor* key = linear(ctx, get("k_proj.weight"), input);
    ggml_tensor* value = linear(ctx, get("v_proj.weight"), input);

    query = ggml_reshape_3d(ctx, query, head_dim, heads, sequence_length);
    key = ggml_reshape_3d(ctx, key, head_dim, kv_heads, sequence_length);
    value = ggml_reshape_3d(ctx, value, head_dim, kv_heads, sequence_length);

    query = rms_norm(
        ctx, query, get("q_norm.weight"), config.rms_norm_epsilon);
    key = rms_norm(
        ctx, key, get("k_norm.weight"), config.rms_norm_epsilon);

    query = ggml_rope_ext(
        ctx,
        query,
        positions,
        nullptr,
        head_dim,
        GGML_ROPE_TYPE_NEOX,
        config.context_length,
        config.rope_theta,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f);
    key = ggml_rope_ext(
        ctx,
        key,
        positions,
        nullptr,
        head_dim,
        GGML_ROPE_TYPE_NEOX,
        config.context_length,
        config.rope_theta,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f);

    const size_t cache_layer_offset =
        static_cast<size_t>(layer) * cache_k->nb[3];
    ggml_tensor* key_cache_layer = ggml_view_4d(
        ctx,
        cache_k,
        head_dim,
        config.num_code_groups,
        kv_heads,
        1,
        cache_k->nb[1],
        cache_k->nb[2],
        cache_k->nb[3],
        cache_layer_offset);
    ggml_tensor* value_cache_layer = ggml_view_4d(
        ctx,
        cache_v,
        head_dim,
        config.num_code_groups,
        kv_heads,
        1,
        cache_v->nb[1],
        cache_v->nb[2],
        cache_v->nb[3],
        cache_layer_offset);

    ggml_tensor* key_rows = ggml_permute(ctx, key, 0, 2, 1, 3);
    ggml_tensor* value_rows = ggml_permute(ctx, value, 0, 2, 1, 3);
    key_cache_layer = ggml_set_rows(
        ctx,
        key_cache_layer,
        key_rows,
        cache_positions);
    value_cache_layer = ggml_set_rows(
        ctx,
        value_cache_layer,
        value_rows,
        cache_positions);
    cache_writes.push_back(key_cache_layer);
    cache_writes.push_back(value_cache_layer);

    ggml_tensor* key_all = ggml_view_4d(
        ctx,
        key_cache_layer,
        head_dim,
        config.num_code_groups,
        kv_heads,
        1,
        key_cache_layer->nb[1],
        key_cache_layer->nb[2],
        key_cache_layer->nb[3],
        0);
    ggml_tensor* value_all = ggml_view_4d(
        ctx,
        value_cache_layer,
        head_dim,
        config.num_code_groups,
        kv_heads,
        1,
        value_cache_layer->nb[1],
        value_cache_layer->nb[2],
        value_cache_layer->nb[3],
        0);

    ggml_tensor* query_flash = ggml_permute(ctx, query, 0, 2, 1, 3);
    ggml_tensor* output = ggml_flash_attn_ext(
        ctx,
        query_flash,
        key_all,
        value_all,
        attention_mask,
        1.0f / std::sqrt(static_cast<float>(head_dim)),
        0.0f,
        0.0f);
    output = ggml_cont_2d(
        ctx,
        output,
        heads * head_dim,
        sequence_length);
    return linear(ctx, get("o_proj.weight"), output);
}

ggml_tensor* predictor_layer(
    ggml_context* ctx,
    ggml_tensor* input,
    const std::string& prefix,
    ggml_tensor* positions,
    ggml_tensor* cache_positions,
    ggml_tensor* attention_mask,
    ggml_tensor* cache_k,
    ggml_tensor* cache_v,
    int64_t layer,
    int64_t sequence_length,
    std::vector<ggml_tensor*>& cache_writes,
    const std::unordered_map<std::string, ggml_tensor*>& weights,
    const CodePredictorConfig& config) {
    const auto get = [&](const std::string& suffix) {
        const auto it = weights.find(prefix + suffix);
        if (it == weights.end()) {
            throw std::runtime_error("missing GGUF tensor: " + prefix + suffix);
        }
        return it->second;
    };

    ggml_tensor* residual = cast_f32(ctx, input);
    ggml_tensor* hidden = rms_norm(
        ctx,
        residual,
        get("input_layernorm.weight"),
        config.rms_norm_epsilon);
    hidden = attention(
        ctx,
        hidden,
        prefix + "self_attn.",
        positions,
        cache_positions,
        attention_mask,
        cache_k,
        cache_v,
        layer,
        sequence_length,
        cache_writes,
        weights,
        config);
    hidden = ggml_add(ctx, residual, hidden);

    residual = hidden;
    hidden = rms_norm(
        ctx,
        hidden,
        get("post_attention_layernorm.weight"),
        config.rms_norm_epsilon);
    ggml_tensor* gate = linear(
        ctx, get("mlp.gate_proj.weight"), hidden);
    ggml_tensor* up = linear(
        ctx, get("mlp.up_proj.weight"), hidden);
    hidden = ggml_swiglu_split(ctx, gate, up);
    hidden = linear(ctx, get("mlp.down_proj.weight"), hidden);
    return ggml_add(ctx, residual, hidden);
}

void read_f32(
    ggml_tensor* tensor,
    std::vector<float>& output) {
    output.resize(static_cast<size_t>(tensor->ne[0]));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(
            tensor,
            output.data(),
            0,
            output.size() * sizeof(float));
        return;
    }
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(output.size());
        ggml_backend_tensor_get(
            tensor,
            half.data(),
            0,
            half.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(half.data(), output.data(), half.size());
        return;
    }
    throw std::runtime_error("unsupported CodePredictor logits dtype");
}

} // namespace

CodePredictor::CodePredictor() = default;

CodePredictor::~CodePredictor() {
    release_graphs();
    if (kv_buffer_ != nullptr) {
        ggml_backend_buffer_free(kv_buffer_);
    }
    if (kv_context_ != nullptr) {
        ggml_free(kv_context_);
    }
    release_weights();
}

ggml_tensor* CodePredictor::weight(
    const std::string& name,
    int64_t ne0,
    int64_t ne1) const {
    const auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("missing GGUF tensor: " + name);
    }
    if (it->second->ne[0] != ne0 ||
        (ne1 >= 0 && it->second->ne[1] != ne1)) {
        throw std::runtime_error("unexpected shape for GGUF tensor: " + name);
    }
    return it->second;
}

Status CodePredictor::load(
    const CodePredictorConfig& config,
    Qwen3TTSContext& runtime) {
    if (runtime.backend() == nullptr) {
        return Status::error("CodePredictor context backend is null");
    }
    release_graphs();
    if (kv_buffer_ != nullptr) {
        ggml_backend_buffer_free(kv_buffer_);
        kv_buffer_ = nullptr;
    }
    if (kv_context_ != nullptr) {
        ggml_free(kv_context_);
        kv_context_ = nullptr;
    }
    release_weights();
    n_past_ = 0;
    config_ = config;
    const GGUFModel& model = runtime.talker_gguf();

    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    weight_context_ = ggml_init(params);
    if (weight_context_ == nullptr) {
        return Status::error("failed to create CodePredictor weight context");
    }

    try {
        for (int64_t i = 0; i < model.tensor_count(); ++i) {
            const ggml_tensor* source = model.tensor(i);
            if (source == nullptr) {
                continue;
            }
            const std::string name = source->name;
            if (name.rfind(
                    "talker.code_predictor.lm_head.", 0) == 0) {
                continue;
            }
            if (name.rfind(
                    "talker.code_predictor.codec_embedding.", 0) == 0) {
                continue;
            }
            const bool predictor_tensor =
                name.rfind("talker.code_predictor.", 0) == 0;
            const bool c0_embedding =
                name == "talker.codec_embedding.weight";
            if (!predictor_tensor && !c0_embedding) {
                continue;
            }
            ggml_tensor* tensor = is_norm_weight(name)
                ? ggml_new_tensor_1d(
                    weight_context_, GGML_TYPE_F32, source->ne[0])
                : ggml_dup_tensor(weight_context_, source);
            ggml_set_name(tensor, source->name);
            if (!weights_.emplace(name, tensor).second) {
                throw std::runtime_error("duplicate GGUF tensor: " + name);
            }
        }

        std::vector<const ggml_tensor*> lm_heads(
            static_cast<size_t>(config_.num_code_groups - 1),
            nullptr);
        std::vector<const ggml_tensor*> codec_embeddings(
            static_cast<size_t>(config_.num_code_groups - 1),
            nullptr);
        ggml_type codec_embedding_type = GGML_TYPE_COUNT;
        ggml_type lm_head_type = GGML_TYPE_COUNT;
        for (int64_t group = 0;
             group < config_.num_code_groups - 1;
             ++group) {
            const std::string embedding_name =
                "talker.code_predictor.codec_embedding." +
                std::to_string(group) + ".weight";
            const ggml_tensor* embedding = model.tensor(embedding_name);
            if (embedding == nullptr) {
                throw std::runtime_error(
                    "missing GGUF tensor: " + embedding_name);
            }
            if (embedding->ne[0] != config_.input_embedding_size ||
                embedding->ne[1] != config_.vocab_size) {
                throw std::runtime_error(
                    "unexpected codec embedding shape: " +
                    embedding_name);
            }
            if (codec_embedding_type == GGML_TYPE_COUNT) {
                codec_embedding_type = embedding->type;
            } else if (embedding->type != codec_embedding_type) {
                throw std::runtime_error(
                    "CodePredictor codec embeddings must share one dtype");
            }
            codec_embeddings[static_cast<size_t>(group)] = embedding;

            const std::string name =
                "talker.code_predictor.lm_head." +
                std::to_string(group) + ".weight";
            const ggml_tensor* source = model.tensor(name);
            if (source == nullptr) {
                throw std::runtime_error(
                    "missing GGUF tensor: " + name);
            }
            if (source->ne[0] != config_.hidden_size ||
                source->ne[1] != config_.vocab_size) {
                throw std::runtime_error(
                    "unexpected LM head shape: " + name);
            }
            if (lm_head_type == GGML_TYPE_COUNT) {
                lm_head_type = source->type;
            } else if (source->type != lm_head_type) {
                throw std::runtime_error(
                    "CodePredictor LM heads must share one dtype");
            }
            lm_heads[static_cast<size_t>(group)] = source;
        }
        packed_codec_embeddings_ = ggml_new_tensor_2d(
            weight_context_,
            codec_embedding_type,
            config_.input_embedding_size,
            config_.vocab_size * (config_.num_code_groups - 1));
        ggml_set_name(
            packed_codec_embeddings_,
            "talker.code_predictor.codec_embeddings.weight");
        packed_lm_heads_ = ggml_new_tensor_3d(
            weight_context_,
            lm_head_type,
            config_.hidden_size,
            config_.vocab_size,
            config_.num_code_groups - 1);
        ggml_set_name(
            packed_lm_heads_,
            "talker.code_predictor.lm_heads.weight");

        weight_buffer_ = ggml_backend_alloc_ctx_tensors(
            weight_context_,
            runtime.backend());
        if (weight_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate CodePredictor weight buffer");
        }
        ggml_backend_buffer_set_usage(
            weight_buffer_,
            GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (const auto& [name, tensor] : weights_) {
            const ggml_tensor* source = model.tensor(name);
            copy_source_tensor(tensor, source);
        }
        const size_t codec_embedding_bytes = static_cast<size_t>(
            ggml_nbytes(codec_embeddings.front()));
        if (static_cast<size_t>(ggml_nbytes(
                packed_codec_embeddings_)) !=
            codec_embedding_bytes *
                static_cast<size_t>(config_.num_code_groups - 1)) {
            throw std::runtime_error(
                "CodePredictor packed codec embedding size is invalid");
        }
        for (int64_t group = 0;
             group < config_.num_code_groups - 1;
             ++group) {
            const ggml_tensor* source =
                codec_embeddings[static_cast<size_t>(group)];
            if (static_cast<size_t>(ggml_nbytes(source)) !=
                codec_embedding_bytes) {
                throw std::runtime_error(
                    "CodePredictor codec embedding sizes differ");
            }
            ggml_backend_tensor_set(
                packed_codec_embeddings_,
                source->data,
                static_cast<size_t>(group) *
                    codec_embedding_bytes,
                codec_embedding_bytes);
        }
        const size_t lm_head_bytes = static_cast<size_t>(
            ggml_nbytes(lm_heads.front()));
        if (packed_lm_heads_->nb[2] != lm_head_bytes) {
            throw std::runtime_error(
                "CodePredictor packed LM head stride is invalid");
        }
        for (int64_t group = 0;
             group < config_.num_code_groups - 1;
             ++group) {
            const ggml_tensor* source =
                lm_heads[static_cast<size_t>(group)];
            if (static_cast<size_t>(ggml_nbytes(source)) !=
                lm_head_bytes) {
                throw std::runtime_error(
                    "CodePredictor LM head sizes differ");
            }
            ggml_backend_tensor_set(
                packed_lm_heads_,
                source->data,
                static_cast<size_t>(group) * lm_head_bytes,
                lm_head_bytes);
        }

        weight(
            "talker.codec_embedding.weight",
            config_.input_embedding_size,
            config_.talker_codec_vocab_size);
        weight("talker.code_predictor.norm.weight", config_.hidden_size);
        if (config_.has_input_projection) {
            weight(
                "talker.code_predictor.small_to_mtp_projection.weight",
                config_.input_embedding_size,
                config_.hidden_size);
            if (config_.input_projection_bias) {
                weight(
                    "talker.code_predictor.small_to_mtp_projection.bias",
                    config_.hidden_size);
            }
        }
        for (int64_t layer = 0; layer < config_.block_count; ++layer) {
            const std::string prefix =
                "talker.code_predictor.layers." +
                std::to_string(layer) + ".";
            weight(prefix + "input_layernorm.weight", config_.hidden_size);
            weight(prefix + "post_attention_layernorm.weight", config_.hidden_size);
            weight(
                prefix + "self_attn.q_proj.weight",
                config_.hidden_size,
                config_.attention_head_count * config_.attention_key_length);
            weight(
                prefix + "self_attn.k_proj.weight",
                config_.hidden_size,
                config_.attention_head_count_kv * config_.attention_key_length);
            weight(
                prefix + "self_attn.v_proj.weight",
                config_.hidden_size,
                config_.attention_head_count_kv * config_.attention_key_length);
            weight(
                prefix + "self_attn.o_proj.weight",
                config_.attention_head_count * config_.attention_key_length,
                config_.hidden_size);
            weight(prefix + "self_attn.q_norm.weight", config_.attention_key_length);
            weight(prefix + "self_attn.k_norm.weight", config_.attention_key_length);
            weight(
                prefix + "mlp.gate_proj.weight",
                config_.hidden_size,
                config_.intermediate_size);
            weight(
                prefix + "mlp.up_proj.weight",
                config_.hidden_size,
                config_.intermediate_size);
            weight(
                prefix + "mlp.down_proj.weight",
                config_.intermediate_size,
                config_.hidden_size);
        }
        ggml_init_params kv_params{};
        kv_params.mem_size = ggml_tensor_overhead() * 2 + 1024;
        kv_params.no_alloc = true;
        kv_context_ = ggml_init(kv_params);
        if (kv_context_ == nullptr) {
            throw std::runtime_error("failed to create CodePredictor KV context");
        }
        cache_k_ = ggml_new_tensor_4d(
            kv_context_,
            GGML_TYPE_F16,
            config_.attention_key_length,
            config_.num_code_groups,
            config_.attention_head_count_kv,
            config_.block_count);
        cache_v_ = ggml_new_tensor_4d(
            kv_context_,
            GGML_TYPE_F16,
            config_.attention_key_length,
            config_.num_code_groups,
            config_.attention_head_count_kv,
            config_.block_count);
        kv_buffer_ = ggml_backend_alloc_ctx_tensors(
            kv_context_,
            runtime.backend());
        if (kv_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate CodePredictor KV buffer");
        }

    } catch (const std::exception& error) {
        return Status::error(error.what());
    }
    return Status::ok();
}

ggml_cgraph* CodePredictor::build_prefill_graph(ggml_context* ctx) {
    // prefill 一次喂两个 token：Talker 最后一层 hidden + 本帧 c0。
    constexpr int64_t sequence_length = 2;

    ggml_tensor* code_input = ggml_new_tensor_1d(
        ctx,
        GGML_TYPE_I32,
        1);
    ggml_set_name(code_input, "code_predictor_code");
    ggml_set_input(code_input);

    ggml_tensor* talker_input = ggml_new_tensor_2d(
        ctx,
        GGML_TYPE_F32,
        config_.input_embedding_size,
        1);
    ggml_set_name(talker_input, "code_predictor_talker_hidden");
    ggml_set_input(talker_input);

    ggml_tensor* c0_embedding = ggml_get_rows(
        ctx,
        weight(
            "talker.codec_embedding.weight",
            config_.input_embedding_size,
            config_.talker_codec_vocab_size),
        code_input);
    ggml_tensor* hidden = ggml_concat(
        ctx,
        cast_f32(ctx, talker_input),
        cast_f32(ctx, c0_embedding),
        1);

    if (config_.has_input_projection) {
        hidden = linear(
            ctx,
            weight(
                "talker.code_predictor.small_to_mtp_projection.weight",
                config_.input_embedding_size,
                config_.hidden_size),
            hidden);
        if (config_.input_projection_bias) {
            hidden = ggml_add(
                ctx,
                hidden,
                repeat_columns(
                    ctx,
                    weight(
                        "talker.code_predictor.small_to_mtp_projection.bias",
                        config_.hidden_size),
                    hidden));
        }
    }

    ggml_tensor* positions = ggml_new_tensor_1d(
        ctx,
        GGML_TYPE_I32,
        sequence_length);
    ggml_set_name(positions, "code_predictor_positions");
    ggml_set_input(positions);

    ggml_tensor* cache_positions = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, sequence_length);
    ggml_set_name(cache_positions, "code_predictor_cache_positions");
    ggml_set_input(cache_positions);

    // mask 的第一维固定成 num_code_groups（KV cache 容量），不随 n_past 变。
    ggml_tensor* attention_mask = ggml_new_tensor_4d(
        ctx,
        GGML_TYPE_F16,
        config_.num_code_groups,
        sequence_length,
        1,
        1);
    ggml_set_name(attention_mask, "code_predictor_attention_mask");
    ggml_set_input(attention_mask);

    ggml_tensor* head = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, 1, 1);
    ggml_set_name(head, "code_predictor_head");
    ggml_set_input(head);

    std::vector<ggml_tensor*> cache_writes;
    ggml_tensor* graph_output = build_predictor_body(
        ctx,
        hidden,
        positions,
        cache_positions,
        attention_mask,
        head,
        sequence_length,
        cache_writes);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
    for (ggml_tensor* write : cache_writes) {
        ggml_build_forward_expand(graph, write);
    }
    ggml_build_forward_expand(graph, graph_output);
    return graph;
}

ggml_cgraph* CodePredictor::build_decode_graph(ggml_context* ctx) {
    // decode 每步一个 token，从打包的 codec embedding 表里取一行。
    constexpr int64_t sequence_length = 1;

    ggml_tensor* embedding_row = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, 1, 1);
    ggml_set_name(
        embedding_row,
        "code_predictor_embedding_row");
    ggml_set_input(embedding_row);
    ggml_tensor* hidden = cast_f32(ctx, ggml_get_rows(
        ctx,
        packed_codec_embeddings_,
        embedding_row));

    if (config_.has_input_projection) {
        hidden = linear(
            ctx,
            weight(
                "talker.code_predictor.small_to_mtp_projection.weight",
                config_.input_embedding_size,
                config_.hidden_size),
            hidden);
        if (config_.input_projection_bias) {
            hidden = ggml_add(
                ctx,
                hidden,
                repeat_columns(
                    ctx,
                    weight(
                        "talker.code_predictor.small_to_mtp_projection.bias",
                        config_.hidden_size),
                    hidden));
        }
    }

    ggml_tensor* positions = ggml_new_tensor_1d(
        ctx,
        GGML_TYPE_I32,
        sequence_length);
    ggml_set_name(positions, "code_predictor_positions");
    ggml_set_input(positions);

    ggml_tensor* cache_positions = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, sequence_length);
    ggml_set_name(cache_positions, "code_predictor_cache_positions");
    ggml_set_input(cache_positions);

    // mask 的第一维固定成 num_code_groups（KV cache 容量），不随 n_past 变。
    ggml_tensor* attention_mask = ggml_new_tensor_4d(
        ctx,
        GGML_TYPE_F16,
        config_.num_code_groups,
        sequence_length,
        1,
        1);
    ggml_set_name(attention_mask, "code_predictor_attention_mask");
    ggml_set_input(attention_mask);

    ggml_tensor* head = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, 1, 1);
    ggml_set_name(head, "code_predictor_head");
    ggml_set_input(head);

    std::vector<ggml_tensor*> cache_writes;
    ggml_tensor* graph_output = build_predictor_body(
        ctx,
        hidden,
        positions,
        cache_positions,
        attention_mask,
        head,
        sequence_length,
        cache_writes);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
    for (ggml_tensor* write : cache_writes) {
        ggml_build_forward_expand(graph, write);
    }
    ggml_build_forward_expand(graph, graph_output);
    return graph;
}

ggml_tensor* CodePredictor::build_predictor_body(
    ggml_context* ctx,
    ggml_tensor* hidden,
    ggml_tensor* positions,
    ggml_tensor* cache_positions,
    ggml_tensor* attention_mask,
    ggml_tensor* head,
    int64_t sequence_length,
    std::vector<ggml_tensor*>& cache_writes) {
    for (int64_t layer = 0; layer < config_.block_count; ++layer) {
        hidden = predictor_layer(
            ctx,
            hidden,
            "talker.code_predictor.layers." +
                std::to_string(layer) + ".",
            positions,
            cache_positions,
            attention_mask,
            cache_k_,
            cache_v_,
            layer,
            sequence_length,
            cache_writes,
            weights_,
            config_);
    }

    hidden = rms_norm(
        ctx,
        hidden,
        weight("talker.code_predictor.norm.weight", config_.hidden_size),
        config_.rms_norm_epsilon);
    ggml_tensor* last_hidden = ggml_view_2d(
        ctx,
        hidden,
        config_.hidden_size,
        1,
        hidden->nb[1],
        static_cast<size_t>(sequence_length - 1) * hidden->nb[1]);
    ggml_tensor* output_logits = ggml_reshape_2d(
        ctx,
        ggml_mul_mat_id(
            ctx,
            packed_lm_heads_,
            last_hidden,
            head),
        config_.vocab_size,
        1);
    ggml_set_name(output_logits, "code_predictor_logits");
    ggml_set_output(output_logits);
    return output_logits;
}

Status CodePredictor::init_graphs(Qwen3TTSContext& runtime) {
    if (runtime.backend() == nullptr ||
        cache_k_ == nullptr ||
        cache_v_ == nullptr) {
        return Status::error("CodePredictor is not loaded");
    }

    // 三张图要么都建好、要么都不建，避免半初始化状态。
    Status status = init_embed_graph(runtime);
    if (!status) {
        release_graphs();
        return status;
    }
    status = init_prefill_graph(runtime);
    if (!status) {
        release_graphs();
        return status;
    }
    status = init_decode_graph(runtime);
    if (!status) {
        release_graphs();
        return status;
    }
    return Status::ok();
}

Status CodePredictor::run_prefill_graph(
    Qwen3TTSContext& runtime,
    const float* talker_hidden,
    int32_t code,
    int32_t head,
    std::vector<float>& logits) {
    logits.clear();
    if (runtime.backend() == nullptr ||
        cache_k_ == nullptr ||
        cache_v_ == nullptr) {
        return Status::error("CodePredictor is not loaded");
    }
    if (talker_hidden == nullptr) {
        return Status::error("CodePredictor talker hidden is null");
    }
    if (head < 0 || head >= config_.num_code_groups - 1) {
        return Status::error("invalid CodePredictor head index");
    }
    constexpr int64_t sequence_length = 2;
    if (n_past_ + sequence_length > config_.num_code_groups) {
        return Status::error("CodePredictor KV cache capacity exceeded");
    }

    PrefillGraph& cache = prefill_graph_;
    if (cache.context == nullptr) {
        return Status::error(
            "CodePredictor prefill graph is not initialized");
    }
    std::vector<int32_t> position_values(
        static_cast<size_t>(sequence_length));
    for (int64_t i = 0; i < sequence_length; ++i) {
        position_values[static_cast<size_t>(i)] =
            static_cast<int32_t>(n_past_ + i);
    }
    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.code,
        &code,
        0,
        sizeof(code));
    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.positions,
        position_values.data(),
        0,
        position_values.size() * sizeof(int32_t));
    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.cache_positions,
        position_values.data(),
        0,
        position_values.size() * sizeof(int32_t));
    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.head,
        &head,
        0,
        sizeof(head));

    const ggml_fp16_t mask_allowed = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t mask_blocked = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> mask(
        static_cast<size_t>(
            config_.num_code_groups * sequence_length),
        mask_blocked);
    for (int64_t query = 0; query < sequence_length; ++query) {
        const int64_t valid_keys = n_past_ + query + 1;
        for (int64_t key = 0; key < valid_keys; ++key) {
            mask[static_cast<size_t>(
                query * config_.num_code_groups + key)] =
                mask_allowed;
        }
    }
    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.attention_mask,
        mask.data(),
        0,
        mask.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.talker_hidden,
        talker_hidden,
        0,
        static_cast<size_t>(config_.input_embedding_size) *
            sizeof(float));

    if (ggml_backend_sched_graph_compute(
            cache.scheduler,
            cache.graph) != GGML_STATUS_SUCCESS) {
        return Status::error(
            "cached CodePredictor prefill graph computation failed");
    }

    read_f32(cache.output_logits, logits);
    if (logits.size() != static_cast<size_t>(config_.vocab_size)) {
        return Status::error(
            "CodePredictor produced invalid logits size");
    }
    n_past_ += sequence_length;
    return Status::ok();
}

Status CodePredictor::init_prefill_graph(Qwen3TTSContext& runtime) {
    PrefillGraph& cache = prefill_graph_;
    if (cache.context != nullptr) {
        return Status::ok();
    }

    cache.metadata.resize(
        ggml_tensor_overhead() * 8192 + ggml_graph_overhead());
    ggml_init_params params{};
    params.mem_size = cache.metadata.size();
    params.mem_buffer = cache.metadata.data();
    params.no_alloc = true;
    cache.context = ggml_init(params);
    if (cache.context == nullptr) {
        cache.metadata.clear();
        return Status::error(
            "failed to create cached CodePredictor prefill graph context");
    }

    Status status = Status::ok();
    try {
        cache.graph = build_prefill_graph(cache.context);
        cache.code = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_code");
        cache.talker_hidden = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_talker_hidden");
        cache.positions = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_positions");
        cache.cache_positions = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_cache_positions");
        cache.attention_mask = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_attention_mask");
        cache.head = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_head");
        cache.output_logits = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_logits");
        if (cache.code == nullptr ||
            cache.talker_hidden == nullptr ||
            cache.positions == nullptr ||
            cache.cache_positions == nullptr ||
            cache.attention_mask == nullptr ||
            cache.head == nullptr ||
            cache.output_logits == nullptr) {
            throw std::runtime_error(
                "missing cached CodePredictor prefill graph tensors");
        }

        status = runtime.create_scheduler(cache.scheduler);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        // 与 decode 图同理：默认规则会把输入派到 CUDA_Host，导致每次 compute
        // 都为每个输入做一次设备同步 + H2D。绑到 CUDA 后 copy_input 不再触发。
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.code, runtime.backend());
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.talker_hidden, runtime.backend());
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.positions, runtime.backend());
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.cache_positions, runtime.backend());
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.attention_mask, runtime.backend());
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.head, runtime.backend());
        if (!ggml_backend_sched_alloc_graph(
                cache.scheduler,
                cache.graph)) {
            throw std::runtime_error(
                "failed to allocate cached CodePredictor prefill graph");
        }
    } catch (const std::exception& error) {
        status = Status::error(error.what());
    }

    if (!status) {
        if (cache.scheduler != nullptr) {
            ggml_backend_sched_free(cache.scheduler);
            cache.scheduler = nullptr;
        }
        ggml_free(cache.context);
        cache = {};
    }
    return status;
}

Status CodePredictor::init_decode_graph(Qwen3TTSContext& runtime) {
    DecodeGraph& cache = decode_graph_;
    if (cache.context != nullptr) {
        return Status::ok();
    }

    cache.metadata.resize(
        ggml_tensor_overhead() * 8192 + ggml_graph_overhead());
    ggml_init_params params{};
    params.mem_size = cache.metadata.size();
    params.mem_buffer = cache.metadata.data();
    params.no_alloc = true;
    cache.context = ggml_init(params);
    if (cache.context == nullptr) {
        cache.metadata.clear();
        return Status::error(
            "failed to create cached CodePredictor graph context");
    }

    Status status = Status::ok();
    try {
        cache.graph = build_decode_graph(cache.context);
        cache.embedding_row = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_embedding_row");
        cache.positions = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_positions");
        cache.cache_positions = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_cache_positions");
        cache.attention_mask = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_attention_mask");
        cache.head = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_head");
        cache.output_logits = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_logits");
        if (cache.embedding_row == nullptr ||
            cache.positions == nullptr ||
            cache.cache_positions == nullptr ||
            cache.attention_mask == nullptr ||
            cache.head == nullptr ||
            cache.output_logits == nullptr) {
            throw std::runtime_error(
                "missing cached CodePredictor graph tensors");
        }

        status = runtime.create_scheduler(cache.scheduler);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        // 默认规则会把所有 GGML_TENSOR_FLAG_INPUT 张量派到最后一个 backend
        // （CUDA_Host，pinned 主机内存），于是每次 compute 都要为每个输入做
        // 一次设备同步 + H2D。显式绑到 CUDA 之后它们成为 split 内部张量，
        // copy_input 不再触发，输入可以直接异步写进显存。
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.embedding_row, runtime.backend());
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.positions, runtime.backend());
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.cache_positions, runtime.backend());
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.attention_mask, runtime.backend());
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.head, runtime.backend());
        if (!ggml_backend_sched_alloc_graph(
                cache.scheduler,
                cache.graph)) {
            throw std::runtime_error(
                "failed to allocate cached CodePredictor graph");
        }
    } catch (const std::exception& error) {
        status = Status::error(error.what());
    }

    if (!status) {
        if (cache.scheduler != nullptr) {
            ggml_backend_sched_free(cache.scheduler);
            cache.scheduler = nullptr;
        }
        ggml_free(cache.context);
        cache = {};
    }
    return status;
}

Status CodePredictor::run_decode_graph(
    Qwen3TTSContext& runtime,
    int32_t code,
    int32_t head,
    std::vector<float>& logits) {
    logits.clear();
    if (runtime.backend() == nullptr ||
        cache_k_ == nullptr ||
        cache_v_ == nullptr) {
        return Status::error("CodePredictor is not loaded");
    }
    if (head < 0 || head >= config_.num_code_groups - 1) {
        return Status::error("invalid CodePredictor head index");
    }
    if (n_past_ + 1 > config_.num_code_groups) {
        return Status::error("CodePredictor KV cache capacity exceeded");
    }

    DecodeGraph& cache = decode_graph_;
    if (cache.context == nullptr) {
        return Status::error(
            "CodePredictor decode graph is not initialized");
    }
    const int32_t position = static_cast<int32_t>(n_past_);
    const int32_t embedding_row =
        (head - 1) * config_.vocab_size + code;
    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.embedding_row,
        &embedding_row,
        0,
        sizeof(embedding_row));
    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.positions,
        &position,
        0,
        sizeof(position));
    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.cache_positions,
        &position,
        0,
        sizeof(position));
    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.head,
        &head,
        0,
        sizeof(head));

    const ggml_fp16_t mask_allowed = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t mask_blocked = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> mask(
        static_cast<size_t>(config_.num_code_groups),
        mask_blocked);
    for (int64_t key = 0; key <= n_past_; ++key) {
        mask[static_cast<size_t>(key)] = mask_allowed;
    }
    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.attention_mask,
        mask.data(),
        0,
        mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(
            cache.scheduler,
            cache.graph) != GGML_STATUS_SUCCESS) {
        return Status::error(
            "cached CodePredictor graph computation failed");
    }

    read_f32(cache.output_logits, logits);
    if (logits.size() != static_cast<size_t>(config_.vocab_size)) {
        return Status::error(
            "CodePredictor produced invalid logits size");
    }
    n_past_ += 1;
    return Status::ok();
}

Status CodePredictor::predict_frame(
    Qwen3TTSContext& runtime,
    const std::vector<float>& talker_hidden,
    int32_t c0,
    std::vector<int32_t>& codes,
    const SamplingOptions& sampling,
    std::vector<float>* step_logits,
    std::function<void(const std::string&)> log) {
    codes.clear();
    if (step_logits != nullptr) {
        step_logits->clear();
        step_logits->resize(
            static_cast<size_t>(config_.num_code_groups - 1) *
            config_.vocab_size);
    }
    if (talker_hidden.size() !=
        static_cast<size_t>(config_.input_embedding_size)) {
        return Status::error(
            "CodePredictor Talker hidden size does not match configuration");
    }
    if (c0 < 0 || c0 >= config_.talker_codec_vocab_size) {
        return Status::error(
            "CodePredictor c0 is outside the Talker codec vocabulary");
    }
    // 下面要直接 memset 两张 KV cache，所以"是否已加载"由这里把关；
    // run_prefill_graph / run_decode_graph 只校验自己这次调用的参数。
    if (runtime.backend() == nullptr ||
        cache_k_ == nullptr ||
        cache_v_ == nullptr) {
        return Status::error("CodePredictor is not loaded");
    }

    n_past_ = 0;
    ggml_backend_tensor_memset(
        cache_k_,
        0,
        0,
        ggml_nbytes(cache_k_));
    ggml_backend_tensor_memset(
        cache_v_,
        0,
        0,
        ggml_nbytes(cache_v_));
    codes.assign(config_.num_code_groups, -1);
    codes[0] = c0;
    std::vector<int32_t> history = {c0};

    std::vector<float> logits;
    const auto boot_graph_start = std::chrono::steady_clock::now();
    Status status = run_prefill_graph(
        runtime,
        talker_hidden.data(),
        c0,
        0,
        logits);
    if (!status) {
        return status;
    }
    const double boot_graph_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() -
            boot_graph_start).count();
    const auto boot_sample_start = std::chrono::steady_clock::now();
    status = sampler_.select(logits, sampling, history, {}, codes[1]);
    if (!status) {
        return status;
    }
    history.push_back(codes[1]);
    if (log != nullptr) {
        std::ostringstream line;
        line << std::fixed << std::setprecision(2)
             << "predictor bootstrap: graph=" << boot_graph_ms
             << " ms sample="
             << std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    boot_sample_start).count()
             << " ms code[1]=" << codes[1];
        log(line.str());
    }
    if (step_logits != nullptr) {
        std::copy(
            logits.begin(),
            logits.end(),
            step_logits->begin());
    }

    for (int32_t group = 1; group < config_.num_code_groups - 1; ++group) {
        const auto graph_start = std::chrono::steady_clock::now();
        status = run_decode_graph(
            runtime,
            codes[static_cast<size_t>(group)],
            group,
            logits);
        if (!status) {
            return status;
        }
        const double graph_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                graph_start).count();
        const auto sample_start = std::chrono::steady_clock::now();
        status = sampler_.select(
            logits,
            sampling,
            history,
            {},
            codes[static_cast<size_t>(group + 1)]);
        if (!status) {
            return status;
        }
        history.push_back(codes[static_cast<size_t>(group + 1)]);
        if (log != nullptr) {
            std::ostringstream line;
            line << std::fixed << std::setprecision(2)
                 << "predictor group " << group << ": graph="
                 << graph_ms << " ms sample="
                 << std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() -
                        sample_start).count()
                 << " ms code[" << group + 1 << "]="
                 << codes[static_cast<size_t>(group + 1)];
            log(line.str());
        }
        if (step_logits != nullptr) {
            std::copy(
                logits.begin(),
                logits.end(),
                step_logits->begin() +
                static_cast<size_t>(group) * config_.vocab_size);
        }
    }
    return Status::ok();
}

void CodePredictor::set_seed(uint64_t seed) {
    sampler_.seed(seed);
}

Status CodePredictor::embed_codes(
    Qwen3TTSContext& runtime,
    const std::vector<int32_t>& codes,
    std::vector<float>& embedding) {
    return run_embed_graph(runtime, codes, embedding);
}

ggml_cgraph* CodePredictor::build_embed_graph(ggml_context* ctx) {
    // 一帧的 num_code_groups 个码字：逐组查表取一行，再全部相加。
    // 第 0 组走 Talker 的 codec embedding 表，其余走打包后的残差码本。
    ggml_tensor* sum = nullptr;
    for (int64_t group = 0; group < config_.num_code_groups; ++group) {
        ggml_tensor* code_input = ggml_new_tensor_1d(
            ctx,
            GGML_TYPE_I32,
            1);
        const std::string input_name =
            "code_predictor_embedding_code." + std::to_string(group);
        ggml_set_name(code_input, input_name.c_str());
        ggml_set_input(code_input);

        ggml_tensor* table = group == 0
            ? weight(
                "talker.codec_embedding.weight",
                config_.input_embedding_size,
                config_.talker_codec_vocab_size)
            : packed_codec_embeddings_;
        ggml_tensor* row = cast_f32(
            ctx,
            ggml_get_rows(ctx, table, code_input));
        sum = sum == nullptr ? row : ggml_add(ctx, sum, row);
    }
    ggml_set_name(sum, "code_predictor_embedding_sum");
    ggml_set_output(sum);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, sum);
    return graph;
}

Status CodePredictor::run_embed_graph(
    Qwen3TTSContext& runtime,
    const std::vector<int32_t>& codes,
    std::vector<float>& embedding) {
    embedding.clear();
    if (runtime.backend() == nullptr) {
        return Status::error("CodePredictor is not loaded");
    }
    if (codes.size() != static_cast<size_t>(config_.num_code_groups)) {
        return Status::error(
            "CodePredictor embedding code count does not match configuration");
    }
    if (codes[0] < 0 || codes[0] >= config_.talker_codec_vocab_size) {
        return Status::error(
            "CodePredictor c0 is outside the Talker codec vocabulary");
    }
    for (size_t group = 1; group < codes.size(); ++group) {
        if (codes[group] < 0 || codes[group] >= config_.vocab_size) {
            return Status::error(
                "CodePredictor residual code is outside the codebook");
        }
    }

    EmbedGraph& cache = embed_graph_;
    if (cache.context == nullptr) {
        return Status::error(
            "CodePredictor embedding graph is not initialized");
    }

    for (int64_t group = 0; group < config_.num_code_groups; ++group) {
        const int32_t embedding_row = group == 0
            ? codes[0]
            : (group - 1) * config_.vocab_size +
                codes[static_cast<size_t>(group)];
        ggml_backend_tensor_set_async(
            runtime.backend(),
            cache.code_inputs[static_cast<size_t>(group)],
            &embedding_row,
            0,
            sizeof(int32_t));
    }

    if (ggml_backend_sched_graph_compute(
            cache.scheduler,
            cache.graph) != GGML_STATUS_SUCCESS) {
        return Status::error(
            "cached CodePredictor embedding graph computation failed");
    }

    read_f32(cache.output_sum, embedding);
    if (embedding.size() !=
        static_cast<size_t>(config_.input_embedding_size)) {
        return Status::error(
            "CodePredictor produced invalid embedding size");
    }
    return Status::ok();
}

Status CodePredictor::init_embed_graph(Qwen3TTSContext& runtime) {
    EmbedGraph& cache = embed_graph_;
    if (cache.context != nullptr) {
        return Status::ok();
    }

    cache.metadata.resize(
        ggml_tensor_overhead() * 128 + ggml_graph_overhead());
    ggml_init_params params{};
    params.mem_size = cache.metadata.size();
    params.mem_buffer = cache.metadata.data();
    params.no_alloc = true;
    cache.context = ggml_init(params);
    if (cache.context == nullptr) {
        cache.metadata.clear();
        return Status::error(
            "failed to create cached CodePredictor embedding graph context");
    }

    Status status = Status::ok();
    try {
        cache.graph = build_embed_graph(cache.context);
        cache.code_inputs.assign(
            static_cast<size_t>(config_.num_code_groups),
            nullptr);
        for (int64_t group = 0; group < config_.num_code_groups; ++group) {
            const std::string input_name =
                "code_predictor_embedding_code." + std::to_string(group);
            ggml_tensor* input =
                ggml_graph_get_tensor(cache.graph, input_name.c_str());
            if (input == nullptr) {
                throw std::runtime_error(
                    "missing cached CodePredictor embedding input: " +
                    input_name);
            }
            cache.code_inputs[static_cast<size_t>(group)] = input;
        }
        cache.output_sum = ggml_graph_get_tensor(
            cache.graph,
            "code_predictor_embedding_sum");
        if (cache.output_sum == nullptr) {
            throw std::runtime_error(
                "missing cached CodePredictor embedding graph output");
        }

        status = runtime.create_scheduler(cache.scheduler);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        // 与 decode / prefill 同理：绑到 CUDA 避免每个输入一次同步 + H2D。
        for (ggml_tensor* input : cache.code_inputs) {
            ggml_backend_sched_set_tensor_backend(
                cache.scheduler, input, runtime.backend());
        }
        if (!ggml_backend_sched_alloc_graph(
                cache.scheduler,
                cache.graph)) {
            throw std::runtime_error(
                "failed to allocate cached CodePredictor embedding graph");
        }
    } catch (const std::exception& error) {
        status = Status::error(error.what());
    }

    if (!status) {
        if (cache.scheduler != nullptr) {
            ggml_backend_sched_free(cache.scheduler);
            cache.scheduler = nullptr;
        }
        ggml_free(cache.context);
        cache = {};
    }
    return status;
}

void CodePredictor::release_graphs() noexcept {
    if (embed_graph_.scheduler != nullptr) {
        ggml_backend_sched_free(embed_graph_.scheduler);
    }
    if (embed_graph_.context != nullptr) {
        ggml_free(embed_graph_.context);
    }
    embed_graph_ = {};
    if (prefill_graph_.scheduler != nullptr) {
        ggml_backend_sched_free(prefill_graph_.scheduler);
    }
    if (prefill_graph_.context != nullptr) {
        ggml_free(prefill_graph_.context);
    }
    prefill_graph_ = {};
    if (decode_graph_.scheduler != nullptr) {
        ggml_backend_sched_free(decode_graph_.scheduler);
    }
    if (decode_graph_.context != nullptr) {
        ggml_free(decode_graph_.context);
    }
    decode_graph_ = {};
}

void CodePredictor::release_weights() noexcept {
    if (weight_buffer_ != nullptr) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_context_ != nullptr) {
        ggml_free(weight_context_);
        weight_context_ = nullptr;
    }
    packed_codec_embeddings_ = nullptr;
    packed_lm_heads_ = nullptr;
    weights_.clear();
}

} // namespace qwen3_tts
