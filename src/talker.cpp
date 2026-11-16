#include "talker.h"

#include "gguf_model.h"
#include "qwen3_tts.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
            "unsupported Talker weight conversion");
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
            "unsupported Talker norm weight dtype");
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
    int64_t kv_len,
    std::vector<ggml_tensor*>& cache_writes,
    const std::unordered_map<std::string, ggml_tensor*>& weights,
    const TalkerConfig& config) {
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
        ctx,
        query,
        get("q_norm.weight"),
        config.rms_norm_epsilon);
    key = rms_norm(
        ctx,
        key,
        get("k_norm.weight"),
        config.rms_norm_epsilon);

    if (config.rope_plain_neox) {
        ggml_tensor* rope_positions = ggml_view_1d(
            ctx,
            positions,
            sequence_length,
            0);
        query = ggml_rope_ext(
            ctx,
            query,
            rope_positions,
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
            rope_positions,
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
    } else {
        int sections[GGML_MROPE_SECTIONS] = {};
        for (size_t i = 0; i < config.rope_sections.size(); ++i) {
            sections[i] = config.rope_sections[i];
        }
        query = ggml_rope_multi(
            ctx,
            query,
            positions,
            nullptr,
            head_dim,
            sections,
            config.rope_interleaved
                ? GGML_ROPE_TYPE_IMROPE
                : GGML_ROPE_TYPE_NEOX,
            config.context_length,
            config.rope_theta,
            1.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f);
        key = ggml_rope_multi(
            ctx,
            key,
            positions,
            nullptr,
            head_dim,
            sections,
            config.rope_interleaved
                ? GGML_ROPE_TYPE_IMROPE
                : GGML_ROPE_TYPE_NEOX,
            config.context_length,
            config.rope_theta,
            1.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f);
    }

    const size_t cache_layer_offset =
        static_cast<size_t>(layer) * cache_k->nb[3];
    ggml_tensor* key_cache_layer = ggml_view_4d(
        ctx,
        cache_k,
        head_dim,
        config.num_ctx,
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
        config.num_ctx,
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

    ggml_tensor* key_all = key_rows;
    ggml_tensor* value_all = value_rows;
    if (kv_len > sequence_length) {
        key_all = ggml_view_4d(
            ctx,
            key_cache_layer,
            head_dim,
            kv_len,
            kv_heads,
            1,
            key_cache_layer->nb[1],
            key_cache_layer->nb[2],
            key_cache_layer->nb[3],
            0);
        value_all = ggml_view_4d(
            ctx,
            value_cache_layer,
            head_dim,
            kv_len,
            kv_heads,
            1,
            value_cache_layer->nb[1],
            value_cache_layer->nb[2],
            value_cache_layer->nb[3],
            0);
    }

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

ggml_tensor* talker_layer(
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
    int64_t kv_len,
    std::vector<ggml_tensor*>& cache_writes,
    const std::unordered_map<std::string, ggml_tensor*>& weights,
    const TalkerConfig& config) {
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
        kv_len,
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

ggml_tensor* require(
    const std::unordered_map<std::string, ggml_tensor*>& weights,
    const std::string& name,
    int64_t ne0,
    int64_t ne1 = -1) {
    const auto it = weights.find(name);
    if (it == weights.end()) {
        throw std::runtime_error("missing GGUF tensor: " + name);
    }
    if (it->second->ne[0] != ne0 ||
        (ne1 >= 0 && it->second->ne[1] != ne1)) {
        throw std::runtime_error("unexpected shape for GGUF tensor: " + name);
    }
    return it->second;
}

void copy_frame_major(
    ggml_backend_t backend,
    ggml_tensor* tensor,
    int64_t channels,
    int64_t frames,
    std::vector<float>& output) {
    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    output.resize(static_cast<size_t>(channels * frames));

    if (tensor->type == GGML_TYPE_F32) {
        const float* data = reinterpret_cast<const float*>(raw.data());
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t channel = 0; channel < channels; ++channel) {
                output[static_cast<size_t>(frame * channels + channel)] =
                    data[frame * channels + channel];
            }
        }
        return;
    }
    if (tensor->type == GGML_TYPE_F16) {
        const ggml_fp16_t* data = reinterpret_cast<const ggml_fp16_t*>(raw.data());
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t channel = 0; channel < channels; ++channel) {
                output[static_cast<size_t>(frame * channels + channel)] =
                    ggml_fp16_to_fp32(data[frame * channels + channel]);
            }
        }
        return;
    }
    throw std::runtime_error("unsupported Talker output dtype");
}

ggml_context* create_graph_context(
    std::vector<uint8_t>& metadata,
    size_t tensor_budget) {
    // 复用同一块缓冲区：只在容量不够时扩容（那一次会清零）。
    // ggml 建张量时对每个 descriptor 整体覆盖写，不依赖缓冲区初始为 0。
    const size_t needed =
        ggml_tensor_overhead() * tensor_budget + ggml_graph_overhead();
    if (metadata.size() < needed) {
        metadata.resize(needed);
    }
    ggml_init_params params{};
    params.mem_size = needed;
    params.mem_buffer = metadata.data();
    params.no_alloc = true;
    return ggml_init(params);
}

std::vector<int32_t> make_rope_positions(
    int64_t n_past,
    int64_t frames) {
    std::vector<int32_t> positions(static_cast<size_t>(frames * 4), 0);
    for (int64_t i = 0; i < frames; ++i) {
        const int32_t position = static_cast<int32_t>(n_past + i);
        positions[static_cast<size_t>(i)] = position;
        positions[static_cast<size_t>(frames + i)] = position;
        positions[static_cast<size_t>(2 * frames + i)] = position;
    }
    return positions;
}

std::vector<int32_t> make_cache_positions(
    int64_t n_past,
    int64_t frames) {
    std::vector<int32_t> positions(static_cast<size_t>(frames));
    for (int64_t i = 0; i < frames; ++i) {
        positions[static_cast<size_t>(i)] =
            static_cast<int32_t>(n_past + i);
    }
    return positions;
}

std::vector<ggml_fp16_t> make_causal_mask(int64_t frames) {
    const ggml_fp16_t allowed = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t blocked = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> mask(
        static_cast<size_t>(frames * frames),
        blocked);
    for (int64_t query = 0; query < frames; ++query) {
        for (int64_t key = 0; key <= query; ++key) {
            mask[static_cast<size_t>(query * frames + key)] = allowed;
        }
    }
    return mask;
}

} // namespace

Talker::Talker() = default;

Talker::~Talker() {
    release_scheduler();
    if (kv_buffer != nullptr) {
        ggml_backend_buffer_free(kv_buffer);
    }
    if (kv_context != nullptr) {
        ggml_free(kv_context);
    }
    release_weights();
}

Status Talker::load_tensors(
    const TalkerConfig& config,
    Qwen3TTSContext& runtime) {
    if (runtime.backend() == nullptr) {
        return Status::error("Talker context backend is null");
    }
    release_scheduler();
    if (kv_buffer != nullptr) {
        ggml_backend_buffer_free(kv_buffer);
        kv_buffer = nullptr;
    }
    if (kv_context != nullptr) {
        ggml_free(kv_context);
        kv_context = nullptr;
    }
    release_weights();
    config_ = config;
    n_past = 0;
    const GGUFModel& model = runtime.talker_gguf();

    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    context = ggml_init(params);
    if (context == nullptr) {
        return Status::error("failed to create Talker weight context");
    }

    try {
        for (int64_t i = 0; i < model.tensor_count(); ++i) {
            const ggml_tensor* source = model.tensor(i);
            if (source == nullptr) {
                continue;
            }
            const std::string name = source->name;
            if (name.rfind("talker.", 0) != 0 ||
                name.rfind("talker.code_predictor.", 0) == 0) {
                continue;
            }
            ggml_tensor* tensor = is_norm_weight(name)
                ? ggml_new_tensor_1d(
                    context, GGML_TYPE_F32, source->ne[0])
                : ggml_dup_tensor(context, source);
            ggml_set_name(tensor, source->name);
            if (!weights.emplace(name, tensor).second) {
                throw std::runtime_error("duplicate GGUF tensor: " + name);
            }
        }

        weight_buffer = ggml_backend_alloc_ctx_tensors(
            context,
            runtime.backend());
        if (weight_buffer == nullptr) {
            return Status::error("failed to allocate Talker weight buffer");
        }
        ggml_backend_buffer_set_usage(
            weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (const auto& [name, tensor] : weights) {
            const ggml_tensor* source = model.tensor(name);
            copy_source_tensor(tensor, source);
        }

        ggml_init_params kv_params{};
        kv_params.mem_size =
            ggml_tensor_overhead() * 2 + 1024;
        kv_params.no_alloc = true;
        kv_context = ggml_init(kv_params);
        if (kv_context == nullptr) {
            return Status::error("failed to create Talker KV context");
        }
        cache_k = ggml_new_tensor_4d(
            kv_context,
            GGML_TYPE_F16,
            config_.attention_key_length,
            config_.num_ctx,
            config_.attention_head_count_kv,
            config_.block_count);
        cache_v = ggml_new_tensor_4d(
            kv_context,
            GGML_TYPE_F16,
            config_.attention_key_length,
            config_.num_ctx,
            config_.attention_head_count_kv,
            config_.block_count);
        kv_buffer = ggml_backend_alloc_ctx_tensors(
            kv_context,
            runtime.backend());
        if (kv_buffer == nullptr) {
            return Status::error("failed to allocate Talker KV buffer");
        }

        require(
            weights,
            "talker.text_embedding.weight",
            config_.text_embedding_length,
            config_.text_vocab_size);
        require(
            weights,
            "talker.text_projection.linear_fc1.weight",
            config_.text_embedding_length,
            config_.text_embedding_length);
        require(
            weights,
            "talker.text_projection.linear_fc2.weight",
            config_.text_embedding_length,
            config_.embedding_length);
        require(
            weights,
            "talker.codec_embedding.weight",
            config_.embedding_length,
            config_.codec_vocab_size);
        require(weights, "talker.norm.weight", config_.embedding_length);
        require(
            weights,
            "talker.codec_head.weight",
            config_.embedding_length,
            config_.codec_vocab_size);
        for (int64_t layer = 0; layer < config_.block_count; ++layer) {
            const std::string prefix =
                "talker.layers." + std::to_string(layer) + ".";
            require(
                weights,
                prefix + "input_layernorm.weight",
                config_.embedding_length);
            require(
                weights,
                prefix + "post_attention_layernorm.weight",
                config_.embedding_length);
            require(
                weights,
                prefix + "self_attn.q_proj.weight",
                config_.embedding_length,
                config_.attention_head_count * config_.attention_key_length);
            require(
                weights,
                prefix + "self_attn.k_proj.weight",
                config_.embedding_length,
                config_.attention_head_count_kv * config_.attention_key_length);
            require(
                weights,
                prefix + "self_attn.v_proj.weight",
                config_.embedding_length,
                config_.attention_head_count_kv * config_.attention_key_length);
            require(
                weights,
                prefix + "self_attn.o_proj.weight",
                config_.attention_head_count * config_.attention_key_length,
                config_.embedding_length);
            require(
                weights,
                prefix + "self_attn.q_norm.weight",
                config_.attention_key_length);
            require(
                weights,
                prefix + "self_attn.k_norm.weight",
                config_.attention_key_length);
            require(
                weights,
                prefix + "mlp.gate_proj.weight",
                config_.embedding_length,
                config_.feed_forward_length);
            require(
                weights,
                prefix + "mlp.up_proj.weight",
                config_.embedding_length,
                config_.feed_forward_length);
            require(
                weights,
                prefix + "mlp.down_proj.weight",
                config_.feed_forward_length,
                config_.embedding_length);
        }
    } catch (const std::exception& error) {
        return Status::error(error.what());
    }
    // decode 的 KV 视图会覆盖到尚未写入的槽，这些槽必须先是有限值：
    // mask 只把权重压到 0，而 0 * NaN = NaN。清零一次之后，每个槽只会再被
    // 真实的 K/V 覆盖，不需要每帧重复。
    ggml_backend_tensor_memset(cache_k, 0, 0, ggml_nbytes(cache_k));
    ggml_backend_tensor_memset(cache_v, 0, 0, ggml_nbytes(cache_v));
    return init_scheduler(runtime);
}

Status Talker::prefill(
    Qwen3TTSContext& runtime,
    const std::vector<float>& input_embeddings,
    int64_t frames,
    std::vector<float>& last_hidden,
    std::vector<float>& logits) {
    return run_prefill_graph(
        runtime,
        input_embeddings,
        frames,
        last_hidden,
        logits);
}

Status Talker::decode_step(
    Qwen3TTSContext& runtime,
    const std::vector<float>& input_embedding,
    std::vector<float>& last_hidden,
    std::vector<float>& logits) {
    if (input_embedding.size() !=
        static_cast<size_t>(config_.embedding_length)) {
        return Status::error(
            "Talker decode_step expects [" +
            std::to_string(config_.embedding_length) + "]");
    }
    return run_decode_graph(
        runtime,
        input_embedding,
        last_hidden,
        logits);
}

Status Talker::embed_text_tokens(
    Qwen3TTSContext& runtime,
    const std::vector<int32_t>& token_ids,
    std::vector<float>& embeddings) {
    return run_text_embed_graph(runtime, token_ids, embeddings);
}

Status Talker::embed_codec_tokens(
    Qwen3TTSContext& runtime,
    const std::vector<int32_t>& token_ids,
    std::vector<float>& embeddings) {
    return run_codec_embed_graph(runtime, token_ids, embeddings);
}

ggml_tensor* Talker::build_talker_body(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* positions,
    ggml_tensor* cache_positions,
    ggml_tensor* attention_mask,
    int64_t frames,
    int64_t kv_len,
    std::vector<ggml_tensor*>& cache_writes) {
    ggml_tensor* hidden = input;
    for (int64_t layer = 0; layer < config_.block_count; ++layer) {
        hidden = talker_layer(
            ctx,
            hidden,
            "talker.layers." + std::to_string(layer) + ".",
            positions,
            cache_positions,
            attention_mask,
            cache_k,
            cache_v,
            layer,
            frames,
            kv_len,
            cache_writes,
            weights,
            config_);
    }

    hidden = rms_norm(
        ctx,
        hidden,
        require(
            weights,
            "talker.norm.weight",
            config_.embedding_length),
        config_.rms_norm_epsilon);
    ggml_set_output(hidden);
    ggml_set_name(hidden, "talker_hidden");
    ggml_tensor* output_logits = linear(
        ctx,
        require(
            weights,
            "talker.codec_head.weight",
            config_.embedding_length,
            config_.codec_vocab_size),
        hidden);
    ggml_set_name(output_logits, "talker_logits");
    ggml_set_output(output_logits);
    return output_logits;
}

ggml_cgraph* Talker::build_text_embed_graph(
    ggml_context* ctx,
    int64_t frames) {
    ggml_tensor* input = ggml_new_tensor_1d(
        ctx,
        GGML_TYPE_I32,
        frames);
    ggml_set_name(input, "talker_text_ids");
    ggml_set_input(input);

    // 查表 [2048, frames] -> text_projection(2048 -> 2048 -> 1024)
    ggml_tensor* hidden = cast_f32(ctx, ggml_get_rows(
        ctx,
        require(
            weights,
            "talker.text_embedding.weight",
            config_.text_embedding_length,
            config_.text_vocab_size),
        input));
    hidden = linear(
        ctx,
        require(
            weights,
            "talker.text_projection.linear_fc1.weight",
            config_.text_embedding_length,
            config_.text_embedding_length),
        hidden);
    hidden = ggml_add(
        ctx,
        hidden,
        repeat_columns(
            ctx,
            require(
                weights,
                "talker.text_projection.linear_fc1.bias",
                config_.text_embedding_length),
            hidden));
    hidden = ggml_silu(ctx, hidden);
    hidden = linear(
        ctx,
        require(
            weights,
            "talker.text_projection.linear_fc2.weight",
            config_.text_embedding_length,
            config_.embedding_length),
        hidden);
    hidden = ggml_add(
        ctx,
        hidden,
        repeat_columns(
            ctx,
            require(
                weights,
                "talker.text_projection.linear_fc2.bias",
                config_.embedding_length),
            hidden));
    ggml_set_name(hidden, "talker_text_embedding");
    ggml_set_output(hidden);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, hidden);
    return graph;
}

ggml_cgraph* Talker::build_codec_embed_graph(
    ggml_context* ctx,
    int64_t frames) {
    ggml_tensor* input = ggml_new_tensor_1d(
        ctx,
        GGML_TYPE_I32,
        frames);
    ggml_set_name(input, "talker_codec_ids");
    ggml_set_input(input);

    // 只有查表，没有投影。
    ggml_tensor* hidden = cast_f32(ctx, ggml_get_rows(
        ctx,
        require(
            weights,
            "talker.codec_embedding.weight",
            config_.embedding_length,
            config_.codec_vocab_size),
        input));
    ggml_set_name(hidden, "talker_codec_embedding");
    ggml_set_output(hidden);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, hidden);
    return graph;
}

ggml_cgraph* Talker::build_prefill_graph(
    ggml_context* ctx,
    int64_t frames) {
    ggml_tensor* input = ggml_new_tensor_2d(
        ctx,
        GGML_TYPE_F32,
        config_.embedding_length,
        frames);
    ggml_set_name(input, "talker_input");
    ggml_set_input(input);

    ggml_tensor* positions_tensor = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, frames * 4);
    ggml_set_name(positions_tensor, "talker_positions");
    ggml_set_input(positions_tensor);

    ggml_tensor* cache_positions = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, frames);
    ggml_set_name(cache_positions, "talker_cache_positions");
    ggml_set_input(cache_positions);

    // Prefill 一次看完整段 prompt，必须显式给出因果掩码。
    ggml_tensor* attention_mask = ggml_new_tensor_4d(
        ctx,
        GGML_TYPE_F16,
        frames,
        frames,
        1,
        1);
    ggml_set_name(attention_mask, "talker_attention_mask");
    ggml_set_input(attention_mask);

    std::vector<ggml_tensor*> cache_writes;
    ggml_tensor* output_logits = build_talker_body(
        ctx,
        input,
        positions_tensor,
        cache_positions,
        attention_mask,
        frames,
        frames,          // prefill：视图就是本次写入的 frames 行
        cache_writes);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
    for (ggml_tensor* write : cache_writes) {
        ggml_build_forward_expand(graph, write);
    }
    ggml_build_forward_expand(graph, output_logits);
    return graph;
}

ggml_cgraph* Talker::build_decode_graph(
    ggml_context* ctx,
    int64_t kv_len) {
    constexpr int64_t frames = 1;

    ggml_tensor* input = ggml_new_tensor_2d(
        ctx,
        GGML_TYPE_F32,
        config_.embedding_length,
        frames);
    ggml_set_name(input, "talker_input");
    ggml_set_input(input);

    ggml_tensor* positions_tensor = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, frames * 4);
    ggml_set_name(positions_tensor, "talker_positions");
    ggml_set_input(positions_tensor);

    ggml_tensor* cache_positions = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, frames);
    ggml_set_name(cache_positions, "talker_cache_positions");
    ggml_set_input(cache_positions);

    // KV 视图宽度固定成 kv_len，实际用到第几个由 mask 表达：
    // 前 n_past+1 个为 0，其余 -inf。
    ggml_tensor* attention_mask = ggml_new_tensor_4d(
        ctx,
        GGML_TYPE_F16,
        kv_len,
        frames,
        1,
        1);
    ggml_set_name(attention_mask, "talker_attention_mask");
    ggml_set_input(attention_mask);

    std::vector<ggml_tensor*> cache_writes;
    ggml_tensor* output_logits = build_talker_body(
        ctx,
        input,
        positions_tensor,
        cache_positions,
        attention_mask,
        frames,
        kv_len,
        cache_writes);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
    for (ggml_tensor* write : cache_writes) {
        ggml_build_forward_expand(graph, write);
    }
    ggml_build_forward_expand(graph, output_logits);
    return graph;
}

Status Talker::run_text_embed_graph(
    Qwen3TTSContext& runtime,
    const std::vector<int32_t>& token_ids,
    std::vector<float>& embeddings) {
    embeddings.clear();
    if (weight_buffer == nullptr ||
        runtime.backend() == nullptr ||
        embedding_scheduler_ == nullptr) {
        return Status::error("Talker is not loaded");
    }
    if (token_ids.empty()) {
        return Status::error("Talker text embedding input is empty");
    }
    for (int32_t token_id : token_ids) {
        if (token_id < 0 || token_id >= config_.text_vocab_size) {
            return Status::error(
                "Talker text token ID is outside the vocabulary");
        }
    }

    const int64_t frames = static_cast<int64_t>(token_ids.size());
    ggml_context* ctx = create_graph_context(graph_metadata_, 128);
    if (ctx == nullptr) {
        return Status::error(
            "failed to create Talker text embedding context");
    }

    Status status = Status::ok();
    try {
        ggml_cgraph* graph = build_text_embed_graph(ctx, frames);
        ggml_tensor* input =
            ggml_graph_get_tensor(graph, "talker_text_ids");
        ggml_tensor* output =
            ggml_graph_get_tensor(graph, "talker_text_embedding");
        if (input == nullptr || output == nullptr) {
            throw std::runtime_error(
                "missing Talker text embedding graph tensors");
        }

        ggml_backend_sched_reset(embedding_scheduler_);
        if (!ggml_backend_sched_alloc_graph(
                embedding_scheduler_,
                graph)) {
            throw std::runtime_error(
                "failed to allocate Talker text embedding graph");
        }
        ggml_backend_tensor_set(
            input,
            token_ids.data(),
            0,
            token_ids.size() * sizeof(int32_t));
        if (ggml_backend_sched_graph_compute(
                embedding_scheduler_,
                graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(
                "Talker text embedding graph computation failed");
        }

        copy_frame_major(
            runtime.backend(),
            output,
            config_.embedding_length,
            frames,
            embeddings);
    } catch (const std::exception& error) {
        status = Status::error(error.what());
    }

    ggml_free(ctx);
    return status;
}

Status Talker::run_codec_embed_graph(
    Qwen3TTSContext& runtime,
    const std::vector<int32_t>& token_ids,
    std::vector<float>& embeddings) {
    embeddings.clear();
    if (weight_buffer == nullptr ||
        runtime.backend() == nullptr ||
        embedding_scheduler_ == nullptr) {
        return Status::error("Talker is not loaded");
    }
    if (token_ids.empty()) {
        return Status::error("Talker codec embedding input is empty");
    }
    for (int32_t token_id : token_ids) {
        if (token_id < 0 || token_id >= config_.codec_vocab_size) {
            return Status::error(
                "Talker codec token ID is outside the vocabulary");
        }
    }

    const int64_t frames = static_cast<int64_t>(token_ids.size());
    ggml_context* ctx = create_graph_context(graph_metadata_, 32);
    if (ctx == nullptr) {
        return Status::error(
            "failed to create Talker codec embedding context");
    }

    Status status = Status::ok();
    try {
        ggml_cgraph* graph = build_codec_embed_graph(ctx, frames);
        ggml_tensor* input =
            ggml_graph_get_tensor(graph, "talker_codec_ids");
        ggml_tensor* output =
            ggml_graph_get_tensor(graph, "talker_codec_embedding");
        if (input == nullptr || output == nullptr) {
            throw std::runtime_error(
                "missing Talker codec embedding graph tensors");
        }

        ggml_backend_sched_reset(embedding_scheduler_);
        if (!ggml_backend_sched_alloc_graph(
                embedding_scheduler_,
                graph)) {
            throw std::runtime_error(
                "failed to allocate Talker codec embedding graph");
        }
        ggml_backend_tensor_set(
            input,
            token_ids.data(),
            0,
            token_ids.size() * sizeof(int32_t));
        if (ggml_backend_sched_graph_compute(
                embedding_scheduler_,
                graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(
                "Talker codec embedding graph computation failed");
        }

        copy_frame_major(
            runtime.backend(),
            output,
            config_.embedding_length,
            frames,
            embeddings);
    } catch (const std::exception& error) {
        status = Status::error(error.what());
    }

    ggml_free(ctx);
    return status;
}

Status Talker::run_prefill_graph(
    Qwen3TTSContext& runtime,
    const std::vector<float>& input_embeddings,
    int64_t frames,
    std::vector<float>& last_hidden,
    std::vector<float>& logits) {
    last_hidden.clear();
    logits.clear();
    if (weight_buffer == nullptr ||
        runtime.backend() == nullptr ||
        prefill_scheduler_ == nullptr) {
        return Status::error("Talker is not loaded");
    }
    if (frames <= 0 ||
        input_embeddings.size() !=
            static_cast<size_t>(frames * config_.embedding_length)) {
        return Status::error("invalid Talker input embedding size");
    }
    if (frames > config_.num_ctx) {
        return Status::error(
            "Talker KV cache capacity exceeded; increase "
            "qwen3_tts.talker.num_ctx");
    }

    ggml_context* ctx = create_graph_context(graph_metadata_, 8192);
    if (ctx == nullptr) {
        return Status::error("failed to create Talker graph context");
    }

    Status status = Status::ok();
    try {
        ggml_cgraph* graph = build_prefill_graph(ctx, frames);
        ggml_tensor* input =
            ggml_graph_get_tensor(graph, "talker_input");
        ggml_tensor* positions_tensor =
            ggml_graph_get_tensor(graph, "talker_positions");
        ggml_tensor* cache_positions_tensor =
            ggml_graph_get_tensor(graph, "talker_cache_positions");
        ggml_tensor* attention_mask =
            ggml_graph_get_tensor(graph, "talker_attention_mask");
        if (input == nullptr ||
            positions_tensor == nullptr ||
            cache_positions_tensor == nullptr ||
            attention_mask == nullptr) {
            throw std::runtime_error("missing Talker graph inputs");
        }

        const std::vector<int32_t> positions =
            make_rope_positions(0, frames);
        const std::vector<int32_t> cache_positions =
            make_cache_positions(0, frames);
        const std::vector<ggml_fp16_t> mask = make_causal_mask(frames);

        ggml_backend_sched_reset(prefill_scheduler_);
        if (!ggml_backend_sched_alloc_graph(prefill_scheduler_, graph)) {
            throw std::runtime_error(
                "failed to allocate Talker prefill graph");
        }
        ggml_backend_tensor_set(
            input,
            input_embeddings.data(),
            0,
            input_embeddings.size() * sizeof(float));
        ggml_backend_tensor_set(
            positions_tensor,
            positions.data(),
            0,
            positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            cache_positions_tensor,
            cache_positions.data(),
            0,
            cache_positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            attention_mask,
            mask.data(),
            0,
            mask.size() * sizeof(ggml_fp16_t));

        if (ggml_backend_sched_graph_compute(
                prefill_scheduler_,
                graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(
                "Talker prefill graph computation failed");
        }
        ggml_tensor* hidden_out =
            ggml_graph_get_tensor(graph, "talker_hidden");
        ggml_tensor* logits_out =
            ggml_graph_get_tensor(graph, "talker_logits");
        if (hidden_out == nullptr || logits_out == nullptr) {
            throw std::runtime_error(
                "missing Talker prefill graph outputs");
        }
        copy_frame_major(
            runtime.backend(),
            hidden_out,
            config_.embedding_length,
            frames,
            last_hidden);
        copy_frame_major(
            runtime.backend(),
            logits_out,
            config_.codec_vocab_size,
            frames,
            logits);
        n_past = frames;
    } catch (const std::exception& error) {
        status = Status::error(error.what());
    }

    ggml_free(ctx);
    return status;
}

Status Talker::run_decode_graph(
    Qwen3TTSContext& runtime,
    const std::vector<float>& input_embedding,
    std::vector<float>& last_hidden,
    std::vector<float>& logits) {
    last_hidden.clear();
    logits.clear();
    if (weight_buffer == nullptr ||
        runtime.backend() == nullptr) {
        return Status::error("Talker is not loaded");
    }
    if (input_embedding.size() !=
        static_cast<size_t>(config_.embedding_length)) {
        return Status::error("invalid Talker input embedding size");
    }
    if (n_past + 1 > config_.num_ctx) {
        return Status::error(
            "Talker KV cache capacity exceeded; increase "
            "qwen3_tts.talker.num_ctx");
    }

    // KV 视图宽度按 256 取整（CUDA flash attention 内核本来就按 256 分块），
    // 使得同一桶内每一步的形状完全一致。
    constexpr int64_t kKvBucket = 256;
    const int64_t kv_len = std::min(
        static_cast<int64_t>(config_.num_ctx),
        ((n_past + 1 + kKvBucket - 1) / kKvBucket) * kKvBucket);

    if (decode_graph_.context == nullptr || decode_graph_.kv_len != kv_len) {
        // 首次或跨桶：换一张形状匹配的常驻图。
        release_decode_graph();
        Status status = init_decode_graph(runtime, kv_len);
        if (!status) {
            return status;
        }
    }

    DecodeGraph& cache = decode_graph_;
    const std::vector<int32_t> positions = make_rope_positions(n_past, 1);
    const std::vector<int32_t> cache_positions =
        make_cache_positions(n_past, 1);

    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.input,
        input_embedding.data(),
        0,
        input_embedding.size() * sizeof(float));
    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.positions,
        positions.data(),
        0,
        positions.size() * sizeof(int32_t));
    ggml_backend_tensor_set_async(
        runtime.backend(),
        cache.cache_positions,
        cache_positions.data(),
        0,
        cache_positions.size() * sizeof(int32_t));

    // mask：前 n_past+1 个槽可见（0.0），其余屏蔽（-inf）。
    std::vector<ggml_fp16_t> mask(
        static_cast<size_t>(kv_len),
        ggml_fp32_to_fp16(-INFINITY));
    for (int64_t key = 0; key <= n_past; ++key) {
        mask[static_cast<size_t>(key)] = ggml_fp32_to_fp16(0.0f);
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
            "cached Talker decode graph computation failed");
    }

    copy_frame_major(
        runtime.backend(),
        cache.output_hidden,
        config_.embedding_length,
        1,
        last_hidden);
    copy_frame_major(
        runtime.backend(),
        cache.output_logits,
        config_.codec_vocab_size,
        1,
        logits);
    n_past += 1;
    return Status::ok();
}

Status Talker::init_decode_graph(
    Qwen3TTSContext& runtime,
    int64_t kv_len) {
    DecodeGraph& cache = decode_graph_;
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
            "failed to create cached Talker graph context");
    }

    Status status = Status::ok();
    try {
        cache.graph = build_decode_graph(cache.context, kv_len);
        cache.input = ggml_graph_get_tensor(cache.graph, "talker_input");
        cache.positions =
            ggml_graph_get_tensor(cache.graph, "talker_positions");
        cache.cache_positions =
            ggml_graph_get_tensor(cache.graph, "talker_cache_positions");
        cache.attention_mask =
            ggml_graph_get_tensor(cache.graph, "talker_attention_mask");
        cache.output_hidden =
            ggml_graph_get_tensor(cache.graph, "talker_hidden");
        cache.output_logits =
            ggml_graph_get_tensor(cache.graph, "talker_logits");
        if (cache.input == nullptr ||
            cache.positions == nullptr ||
            cache.cache_positions == nullptr ||
            cache.attention_mask == nullptr ||
            cache.output_hidden == nullptr ||
            cache.output_logits == nullptr) {
            throw std::runtime_error(
                "missing cached Talker decode graph tensors");
        }

        status = runtime.create_scheduler(cache.scheduler);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        // 默认规则会把输入派到 CUDA_Host，导致每次 compute 都为每个输入
        // 做一次设备同步 + H2D。绑到 CUDA 后 copy_input 不再触发。
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.input, runtime.backend());
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.positions, runtime.backend());
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.cache_positions, runtime.backend());
        ggml_backend_sched_set_tensor_backend(
            cache.scheduler, cache.attention_mask, runtime.backend());
        if (!ggml_backend_sched_alloc_graph(
                cache.scheduler,
                cache.graph)) {
            throw std::runtime_error(
                "failed to allocate cached Talker decode graph");
        }
        cache.kv_len = kv_len;
    } catch (const std::exception& error) {
        status = Status::error(error.what());
    }

    if (!status) {
        release_decode_graph();
    }
    return status;
}

void Talker::release_decode_graph() noexcept {
    if (decode_graph_.scheduler != nullptr) {
        ggml_backend_sched_free(decode_graph_.scheduler);
    }
    if (decode_graph_.context != nullptr) {
        ggml_free(decode_graph_.context);
    }
    decode_graph_ = {};
}

Status Talker::init_scheduler(Qwen3TTSContext& runtime) {
    release_scheduler();
    Status status = runtime.create_scheduler(prefill_scheduler_);
    if (!status) {
        return status;
    }
    status = runtime.create_scheduler(embedding_scheduler_);
    if (!status) {
        release_scheduler();
        return status;
    }
    return Status::ok();
}

void Talker::release_scheduler() noexcept {
    release_decode_graph();
    if (embedding_scheduler_ != nullptr) {
        ggml_backend_sched_free(embedding_scheduler_);
        embedding_scheduler_ = nullptr;
    }
    if (prefill_scheduler_ != nullptr) {
        ggml_backend_sched_free(prefill_scheduler_);
        prefill_scheduler_ = nullptr;
    }
    graph_metadata_.clear();
    graph_metadata_.shrink_to_fit();
}

void Talker::release_weights() noexcept {
    if (weight_buffer != nullptr) {
        ggml_backend_buffer_free(weight_buffer);
        weight_buffer = nullptr;
    }
    if (context != nullptr) {
        ggml_free(context);
        context = nullptr;
    }
    weights.clear();
}

} // namespace qwen3_tts
