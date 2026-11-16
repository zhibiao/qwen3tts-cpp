#include "speech_tokenizer.h"

#include "gguf_model.h"
#include "qwen3_tts.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qwen3_tts {

namespace {

ggml_tensor* cast_f32(ggml_context* ctx, ggml_tensor* tensor) {
    return tensor->type == GGML_TYPE_F32 ? tensor : ggml_cast(ctx, tensor, GGML_TYPE_F32);
}

ggml_tensor* repeat_columns(ggml_context* ctx, ggml_tensor* vector, ggml_tensor* like) {
    ggml_tensor* column = ggml_reshape_2d(ctx, vector, vector->ne[0], 1);
    return ggml_repeat(ctx, column, like);
}

ggml_tensor* repeat_rows(ggml_context* ctx, ggml_tensor* vector, ggml_tensor* like) {
    ggml_tensor* row = ggml_reshape_2d(ctx, vector, 1, vector->ne[0]);
    return ggml_repeat(ctx, row, like);
}

ggml_tensor* linear(
    ggml_context* ctx,
    ggml_tensor* weight,
    ggml_tensor* bias,
    ggml_tensor* input) {
    ggml_tensor* output = ggml_mul_mat(ctx, weight, cast_f32(ctx, input));
    if (bias != nullptr) {
        output = ggml_add(ctx, output, repeat_columns(ctx, cast_f32(ctx, bias), output));
    }
    return output;
}

ggml_tensor* transpose_2d(ggml_context* ctx, ggml_tensor* input) {
    return ggml_cont(ctx, ggml_transpose(ctx, input));
}

ggml_tensor* rms_norm(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* weight,
    float eps) {
    ggml_tensor* normalized = ggml_rms_norm(ctx, input, eps);
    return ggml_mul(ctx, normalized, repeat_columns(ctx, cast_f32(ctx, weight), normalized));
}

ggml_tensor* layer_norm(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* weight,
    ggml_tensor* bias,
    float eps) {
    ggml_tensor* normalized = ggml_norm(ctx, input, eps);
    normalized = ggml_mul(
        ctx,
        normalized,
        repeat_columns(ctx, cast_f32(ctx, weight), normalized));
    if (bias != nullptr) {
        normalized = ggml_add(
            ctx,
            normalized,
            repeat_columns(ctx, cast_f32(ctx, bias), normalized));
    }
    return normalized;
}

ggml_tensor* causal_conv(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* kernel,
    ggml_tensor* bias,
    int dilation = 1,
    bool depthwise = false) {
    const int64_t input_length = input->ne[0];
    const int64_t input_channels = input->ne[1];
    const int64_t output_channels = kernel->ne[2];
    const int kernel_size = static_cast<int>((kernel->ne[0] - 1) * dilation + 1);
    const int padding = kernel_size - 1;

    if (!depthwise && kernel_size == 1) {
        ggml_tensor* weight_2d = ggml_reshape_2d(
            ctx,
            kernel,
            input_channels,
            output_channels);
        ggml_tensor* transposed = transpose_2d(ctx, input);
        transposed = linear(ctx, weight_2d, bias, transposed);
        return transpose_2d(ctx, transposed);
    }

    ggml_tensor* input_3d = ggml_reshape_3d(ctx, cast_f32(ctx, input), input_length, input_channels, 1);
    ggml_tensor* raw = depthwise
        ? ggml_conv_1d_dw(ctx, kernel, input_3d, 1, padding, dilation)
        : ggml_conv_1d(ctx, kernel, input_3d, 1, padding, dilation);
    ggml_tensor* output = ggml_cont_2d(
        ctx,
        ggml_view_2d(ctx, raw, input_length, output_channels, raw->nb[1], 0),
        input_length,
        output_channels);
    if (bias != nullptr) {
        output = ggml_add(
            ctx,
            output,
            repeat_rows(ctx, cast_f32(ctx, bias), output));
    }
    return output;
}

ggml_tensor* causal_transposed_conv(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* kernel,
    ggml_tensor* bias,
    int stride,
    bool matrix_path) {
    const int64_t input_length = input->ne[0];
    const int64_t input_channels = input->ne[1];
    const int64_t output_length = input_length * stride;
    const int64_t output_channels = kernel->ne[1];
    if (std::getenv("QWEN3TTS_PROFILE_SPEECH") != nullptr) {
        std::fprintf(
            stderr,
            "SpeechDecoder transposed conv: input=[%lld,%lld] "
            "kernel=[%lld,%lld,%lld] type=%s stride=%d\n",
            static_cast<long long>(input_length),
            static_cast<long long>(input_channels),
            static_cast<long long>(kernel->ne[0]),
            static_cast<long long>(kernel->ne[1]),
            static_cast<long long>(kernel->ne[2]),
            ggml_type_name(kernel->type),
            stride);
    }
    if (matrix_path) {
        ggml_tensor* input_transposed =
            transpose_2d(ctx, cast_f32(ctx, input));
        ggml_tensor* phases = nullptr;
        for (int phase = 0; phase < stride; ++phase) {
            const auto tap = [&](int k, ggml_tensor* source) {
                ggml_tensor* kernel_tap = ggml_view_2d(
                    ctx,
                    kernel,
                    input_channels,
                    output_channels,
                    kernel->nb[1],
                    static_cast<size_t>(k) * kernel->nb[2]);
                return ggml_mul_mat(ctx, kernel_tap, source);
            };

            ggml_tensor* phase_output = nullptr;
            int offset = 0;
            for (int k = phase;
                 k < kernel->ne[2];
                 k += stride, ++offset) {
                if (offset >= input_length) {
                    break;
                }
                ggml_tensor* source = offset == 0
                    ? input_transposed
                    : ggml_view_2d(
                        ctx,
                        input_transposed,
                        input_channels,
                        input_length - offset,
                        input_transposed->nb[1],
                        0);
                ggml_tensor* contribution = tap(k, source);
                if (offset > 0) {
                    contribution = ggml_pad_ext(
                        ctx,
                        contribution,
                        0,
                        0,
                        offset,
                        0,
                        0,
                        0,
                        0,
                        0);
                }
                phase_output = phase_output == nullptr
                    ? contribution
                    : ggml_add(ctx, phase_output, contribution);
            }
            phase_output = transpose_2d(ctx, phase_output);
            phases = phases == nullptr
                ? phase_output
                : ggml_concat(ctx, phases, phase_output, 2);
        }
        ggml_tensor* output = ggml_cont_2d(
            ctx,
            ggml_permute(ctx, phases, 1, 2, 0, 3),
            output_length,
            output_channels);
        if (bias != nullptr) {
            output = ggml_add(
                ctx,
                output,
                repeat_rows(ctx, cast_f32(ctx, bias), output));
        }
        return output;
    }

    ggml_tensor* raw = ggml_conv_transpose_1d(
        ctx,
        kernel,
        cast_f32(ctx, input),
        stride,
        0,
        1);
    const int right_crop = static_cast<int>(kernel->ne[0]) - stride;
    const int64_t cropped_length = raw->ne[0] - right_crop;
    ggml_tensor* output = ggml_cont_2d(
        ctx,
        ggml_view_2d(ctx, raw, cropped_length, raw->ne[1], raw->nb[1], 0),
        cropped_length,
        raw->ne[1]);
    if (bias != nullptr) {
        output = ggml_add(
            ctx,
            output,
            repeat_rows(ctx, cast_f32(ctx, bias), output));
    }
    return output;
}

ggml_tensor* snake_beta(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* alpha,
    ggml_tensor* beta) {
    ggml_tensor* x = cast_f32(ctx, input);
    ggml_tensor* alpha_exp = ggml_exp(ctx, cast_f32(ctx, alpha));
    ggml_tensor* beta_exp = ggml_scale_bias(ctx, ggml_exp(ctx, cast_f32(ctx, beta)), 1.0f, 1e-9f);
    ggml_tensor* alpha_full = repeat_rows(ctx, alpha_exp, x);
    ggml_tensor* beta_full = repeat_rows(ctx, beta_exp, x);
    ggml_tensor* sine = ggml_sin(ctx, ggml_mul(ctx, x, alpha_full));
    ggml_tensor* periodic = ggml_div(ctx, ggml_sqr(ctx, sine), beta_full);
    return ggml_add(ctx, x, periodic);
}

ggml_tensor* convnext_block(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* dw_weight,
    ggml_tensor* dw_bias,
    ggml_tensor* norm_weight,
    ggml_tensor* norm_bias,
    ggml_tensor* pw1_weight,
    ggml_tensor* pw1_bias,
    ggml_tensor* pw2_weight,
    ggml_tensor* pw2_bias,
    ggml_tensor* gamma) {
    ggml_tensor* residual = cast_f32(ctx, input);
    ggml_tensor* hidden = causal_conv(ctx, residual, dw_weight, dw_bias, 1, true);
    hidden = transpose_2d(ctx, hidden);
    hidden = layer_norm(ctx, hidden, norm_weight, norm_bias, 1e-6f);
    hidden = linear(ctx, pw1_weight, pw1_bias, hidden);
    hidden = ggml_gelu(ctx, hidden);
    hidden = linear(ctx, pw2_weight, pw2_bias, hidden);
    hidden = ggml_mul(ctx, hidden, repeat_columns(ctx, cast_f32(ctx, gamma), hidden));
    hidden = transpose_2d(ctx, hidden);
    return ggml_add(ctx, residual, hidden);
}

ggml_tensor* decoder_residual_unit(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* act1_alpha,
    ggml_tensor* act1_beta,
    ggml_tensor* conv1_weight,
    ggml_tensor* conv1_bias,
    ggml_tensor* act2_alpha,
    ggml_tensor* act2_beta,
    ggml_tensor* conv2_weight,
    ggml_tensor* conv2_bias,
    int dilation) {
    ggml_tensor* residual = cast_f32(ctx, input);
    ggml_tensor* hidden = snake_beta(ctx, residual, act1_alpha, act1_beta);
    hidden = causal_conv(ctx, hidden, conv1_weight, conv1_bias, dilation, false);
    hidden = snake_beta(ctx, hidden, act2_alpha, act2_beta);
    hidden = causal_conv(ctx, hidden, conv2_weight, conv2_bias, 1, false);
    return ggml_add(ctx, residual, hidden);
}

bool is_transposed_conv_weight(const std::string& name) {
    return (name.rfind("sdec.upsample.", 0) == 0 &&
            name.find(".0.conv.weight") != std::string::npos) ||
        (name.rfind("sdec.decoder.", 0) == 0 &&
         name.find(".block.1.conv.weight") != std::string::npos);
}

ggml_tensor* attention(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* q_weight,
    ggml_tensor* k_weight,
    ggml_tensor* v_weight,
    ggml_tensor* o_weight,
    ggml_tensor* positions,
    ggml_tensor* mask,
    const SpeechDecoderConfig& config) {
    const int64_t sequence_length = input->ne[1];
    const int64_t head_dim = config.attention_key_length;
    const int64_t heads = config.attention_head_count;
    const int64_t kv_heads = config.attention_head_count_kv;
    ggml_tensor* query = linear(ctx, q_weight, nullptr, input);
    ggml_tensor* key = linear(ctx, k_weight, nullptr, input);
    ggml_tensor* value = linear(ctx, v_weight, nullptr, input);

    query = ggml_reshape_3d(ctx, query, head_dim, heads, sequence_length);
    key = ggml_reshape_3d(ctx, key, head_dim, kv_heads, sequence_length);
    value = ggml_reshape_3d(ctx, value, head_dim, kv_heads, sequence_length);

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

    query = ggml_permute(ctx, query, 0, 2, 1, 3);
    key = ggml_permute(ctx, key, 0, 2, 1, 3);
    value = ggml_permute(ctx, value, 0, 2, 1, 3);

    ggml_tensor* output = ggml_flash_attn_ext(
        ctx,
        query,
        key,
        value,
        mask,
        1.0f / std::sqrt(static_cast<float>(head_dim)),
        0.0f,
        0.0f);
    output = ggml_cont_2d(
        ctx,
        output,
        heads * head_dim,
        sequence_length);
    return linear(ctx, o_weight, nullptr, output);
}

ggml_tensor* transformer_layer(
    ggml_context* ctx,
    ggml_tensor* input,
    const std::string& prefix,
    ggml_tensor* positions,
    ggml_tensor* mask,
    const std::unordered_map<std::string, ggml_tensor*>& weights,
    const SpeechDecoderConfig& config) {
    const auto get = [&](const std::string& suffix) -> ggml_tensor* {
        const auto it = weights.find(prefix + suffix);
        if (it == weights.end()) {
            throw std::runtime_error("missing GGUF tensor: " + prefix + suffix);
        }
        return it->second;
    };

    ggml_tensor* residual = cast_f32(ctx, input);
    ggml_tensor* normalized = rms_norm(
        ctx,
        residual,
        get("input_layernorm.weight"),
        config.rms_norm_epsilon);
    ggml_tensor* attended = attention(
        ctx,
        normalized,
        get("sa.q_proj.weight"),
        get("sa.k_proj.weight"),
        get("sa.v_proj.weight"),
        get("sa.o_proj.weight"),
        positions,
        mask,
        config);
    attended = ggml_mul(
        ctx,
        attended,
        repeat_columns(ctx, cast_f32(ctx, get("sa_scale.scale")), attended));
    ggml_tensor* hidden = ggml_add(ctx, residual, attended);

    residual = hidden;
    normalized = rms_norm(
        ctx,
        hidden,
        get("post_norm.weight"),
        config.rms_norm_epsilon);
    ggml_tensor* gate = linear(ctx, get("mlp.gate_proj.weight"), nullptr, normalized);
    gate = ggml_silu(ctx, gate);
    ggml_tensor* up = linear(ctx, get("mlp.up_proj.weight"), nullptr, normalized);
    hidden = ggml_mul(ctx, gate, up);
    hidden = linear(ctx, get("mlp.down_proj.weight"), nullptr, hidden);
    hidden = ggml_mul(
        ctx,
        hidden,
        repeat_columns(ctx, cast_f32(ctx, get("mlp_layer_scale.scale")), hidden));
    return ggml_add(ctx, residual, hidden);
}

} // namespace

class SpeechTokenizer::Decoder {
public:
    Decoder();
    ~Decoder();

    Status load_tensors(
        const SpeechTokenizerConfig& config,
        Qwen3TTSContext& context);
    ggml_cgraph* build_graph(
        ggml_context* ctx,
        int64_t frames);
    Status execute_graph(
        Qwen3TTSContext& context,
        ggml_cgraph* graph,
        const SpeechCodeMatrix& codes,
        AudioBuffer& audio);
    Status decode(
        Qwen3TTSContext& context,
        const SpeechCodeMatrix& codes,
        AudioBuffer& audio);

private:
    ggml_tensor* weight(const std::string& name) const;
    ggml_tensor* transposed_weight(const std::string& name) const;
    ggml_tensor* weight(
        const std::string& name,
        int64_t ne0,
        int64_t ne1 = -1,
        int64_t ne2 = -1,
        int64_t ne3 = -1) const;
    Status init_scheduler(Qwen3TTSContext& context);
    void release_scheduler() noexcept;
    void release_weights() noexcept;

    const GGUFModel* model_ = nullptr;
    ggml_context* weight_context_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> weights_;
    SpeechTokenizerConfig config_;
    ggml_backend_sched_t decode_scheduler_ = nullptr;
    bool use_matrix_transposed_conv_ = false;
};

SpeechTokenizer::Decoder::Decoder() = default;

SpeechTokenizer::Decoder::~Decoder() {
    release_scheduler();
    release_weights();
}

ggml_tensor* SpeechTokenizer::Decoder::weight(const std::string& name) const {
    const auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("missing GGUF tensor: " + name);
    }
    return it->second;
}

ggml_tensor* SpeechTokenizer::Decoder::transposed_weight(
    const std::string& name) const {
    const auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("missing GGUF tensor: " + name);
    }
    return it->second;
}

ggml_tensor* SpeechTokenizer::Decoder::weight(
    const std::string& name,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3) const {
    ggml_tensor* tensor = weight(name);
    if (tensor->ne[0] != ne0 ||
        (ne1 >= 0 && tensor->ne[1] != ne1) ||
        (ne2 >= 0 && tensor->ne[2] != ne2) ||
        (ne3 >= 0 && tensor->ne[3] != ne3)) {
        throw std::runtime_error("unexpected shape for GGUF tensor: " + name);
    }
    return tensor;
}

void SpeechTokenizer::Decoder::release_weights() noexcept {
    if (weight_buffer_ != nullptr) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_context_ != nullptr) {
        ggml_free(weight_context_);
        weight_context_ = nullptr;
    }
    weights_.clear();
    model_ = nullptr;
    use_matrix_transposed_conv_ = false;
}

Status SpeechTokenizer::Decoder::init_scheduler(
    Qwen3TTSContext& runtime) {
    release_scheduler();
    return runtime.create_scheduler(decode_scheduler_);
}

void SpeechTokenizer::Decoder::release_scheduler() noexcept {
    if (decode_scheduler_ != nullptr) {
        ggml_backend_sched_free(decode_scheduler_);
        decode_scheduler_ = nullptr;
    }
}

Status SpeechTokenizer::Decoder::load_tensors(
    const SpeechTokenizerConfig& config,
    Qwen3TTSContext& runtime) {
    if (runtime.backend() == nullptr) {
        return Status::error("SpeechDecoder context backend is null");
    }
    release_scheduler();
    release_weights();
    const GGUFModel& model = runtime.speech_gguf();
    model_ = &model;
    config_ = config;

    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    weight_context_ = ggml_init(params);
    if (weight_context_ == nullptr) {
        return Status::error("failed to create SpeechDecoder weight context");
    }

    try {
        const bool gpu_backend =
            ggml_backend_dev_type(
                ggml_backend_get_device(runtime.backend())) !=
            GGML_BACKEND_DEVICE_TYPE_CPU;
        bool matrix_transposed_conv_ready = gpu_backend;
        for (int64_t i = 0; i < model.tensor_count(); ++i) {
            const ggml_tensor* source = model.tensor(i);
            if (source == nullptr ||
                std::string(source->name).rfind("sdec.", 0) != 0) {
                continue;
            }
            if (source->type != GGML_TYPE_F32) {
                throw std::runtime_error(
                    "SpeechDecoder tensor must remain F32: " +
                    std::string(source->name));
            }
            const bool transpose_conv_weight =
                gpu_backend &&
                is_transposed_conv_weight(source->name);
            if (gpu_backend &&
                is_transposed_conv_weight(source->name) &&
                ggml_blck_size(source->type) != 1) {
                matrix_transposed_conv_ready = false;
            }
            ggml_tensor* tensor = transpose_conv_weight
                ? ggml_new_tensor_3d(
                    weight_context_,
                    source->type,
                    source->ne[2],
                    source->ne[1],
                    source->ne[0])
                : ggml_dup_tensor(weight_context_, source);
            if (tensor == nullptr) {
                return Status::error(
                    "failed to create tensor descriptor: " +
                    std::string(source->name));
            }
            ggml_set_name(tensor, source->name);
            weights_.emplace(source->name, tensor);
        }
        use_matrix_transposed_conv_ = matrix_transposed_conv_ready;
    } catch (const std::exception& error) {
        return Status::error(error.what());
    }

    weight_buffer_ = ggml_backend_alloc_ctx_tensors(
        weight_context_,
        runtime.backend());
    if (weight_buffer_ == nullptr) {
        return Status::error("failed to allocate SpeechDecoder weight buffer");
    }
    ggml_backend_buffer_set_usage(
        weight_buffer_,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (auto& [name, tensor] : weights_) {
        const ggml_tensor* source = model.tensor(name);
        if (is_transposed_conv_weight(name) &&
            tensor->ne[2] == source->ne[0]) {
            const int64_t kernel_size = source->ne[0];
            const int64_t output_channels = source->ne[1];
            const int64_t input_channels = source->ne[2];
            const size_t type_size = ggml_type_size(source->type);
            const uint8_t* source_values =
                static_cast<const uint8_t*>(source->data);
            std::vector<uint8_t> transformed(
                static_cast<size_t>(ggml_nelements(source)) * type_size);
            for (int64_t input_channel = 0;
                 input_channel < input_channels;
                 ++input_channel) {
                for (int64_t output_channel = 0;
                     output_channel < output_channels;
                     ++output_channel) {
                    for (int64_t k = 0; k < kernel_size; ++k) {
                        const size_t source_index = static_cast<size_t>(
                            k +
                            kernel_size *
                                (output_channel +
                                 output_channels * input_channel));
                        const size_t destination_index = static_cast<size_t>(
                            input_channel +
                            input_channels *
                                (output_channel +
                                 output_channels * k));
                        std::memcpy(
                            transformed.data() +
                                destination_index * type_size,
                            source_values + source_index * type_size,
                            type_size);
                    }
                }
            }
            ggml_backend_tensor_set(
                tensor,
                transformed.data(),
                0,
                transformed.size());
        } else {
            ggml_backend_tensor_set(
                tensor,
                source->data,
                0,
                ggml_nbytes(source));
        }
    }

    try {
        const SpeechDecoderConfig& decoder = config_.decoder;
        const int64_t vector_dim = decoder.codebook_dim / 2;
        weight(
            "sdec.pre_conv.conv.weight",
            3,
            decoder.codebook_dim,
            decoder.latent_dim);
        weight("sdec.q.q0.rvq.l.0.cb.sum", vector_dim, decoder.codebook_size);
        weight(
            "sdec.q.q1.rvq.l." +
                std::to_string(decoder.num_quantizers -
                               decoder.num_semantic_quantizers - 1) +
                ".cb.sum",
            vector_dim,
            decoder.codebook_size);
        weight(
            "sdec.pt.l." +
                std::to_string(decoder.block_count - 1) +
                ".mlp.down_proj.weight",
            decoder.intermediate_size,
            decoder.hidden_size);
        const int64_t output_dim =
            decoder.decoder_dim >> decoder.upsample_rates.size();
        weight(
            "sdec.decoder." +
                std::to_string(decoder.upsample_rates.size() + 2) +
                ".conv.weight",
            7,
            output_dim,
            1);
    } catch (const std::exception& error) {
        return Status::error(error.what());
    }

    return init_scheduler(runtime);
}

ggml_cgraph* SpeechTokenizer::Decoder::build_graph(
    ggml_context* ctx,
    int64_t frames) {
    if (model_ == nullptr || weight_context_ == nullptr) {
        throw std::runtime_error("SpeechDecoder is not loaded");
    }
    if (frames <= 0) {
        throw std::runtime_error("SpeechDecoder frame count must be positive");
    }

    const int64_t sequence_length = frames;
    if (ctx == nullptr) {
        throw std::runtime_error("SpeechDecoder graph context is null");
    }

    const SpeechDecoderConfig& config = config_.decoder;
    const int64_t vector_dim = config.codebook_dim / 2;
    std::vector<ggml_tensor*> code_tensors(config.num_quantizers);
    {
        for (int32_t layer = 0; layer < config.num_quantizers; ++layer) {
            code_tensors[layer] =
                ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sequence_length);
            const std::string name = "speech_codes." + std::to_string(layer);
            ggml_set_name(code_tensors[layer], name.c_str());
            ggml_set_input(code_tensors[layer]);
        }

        ggml_tensor* positions_tensor =
            ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sequence_length);
        ggml_set_name(positions_tensor, "positions");
        ggml_set_input(positions_tensor);
        ggml_tensor* mask_tensor = ggml_new_tensor_4d(
            ctx,
            GGML_TYPE_F16,
            sequence_length,
            sequence_length,
            1,
            1);
        ggml_set_name(mask_tensor, "attention_mask");
        ggml_set_input(mask_tensor);

        const auto decode_codebook = [&](const std::string& prefix, int layer) {
            ggml_tensor* sum = ggml_get_rows(
                ctx,
                weight(
                    prefix + ".cb.sum",
                    vector_dim,
                    config.codebook_size),
                code_tensors[static_cast<size_t>(layer)]);
            ggml_tensor* usage = ggml_get_rows(
                ctx,
                ggml_reshape_2d(
                    ctx,
                    weight(prefix + ".cb.usage", config.codebook_size),
                    1,
                    config.codebook_size),
                code_tensors[static_cast<size_t>(layer)]);
            sum = cast_f32(ctx, sum);
            usage = ggml_clamp(ctx, cast_f32(ctx, usage), 1e-5f, FLT_MAX);
            usage = ggml_reshape_2d(ctx, usage, 1, sequence_length);
            usage = ggml_repeat(ctx, usage, sum);
            return transpose_2d(ctx, ggml_div(ctx, sum, usage));
        };

        ggml_tensor* semantic = nullptr;
        for (int32_t layer = 0;
             layer < config.num_semantic_quantizers;
             ++layer) {
            ggml_tensor* decoded = decode_codebook(
                "sdec.q.q0.rvq.l." + std::to_string(layer),
                layer);
            semantic = semantic == nullptr
                ? decoded
                : ggml_add(ctx, semantic, decoded);
        }
        semantic = causal_conv(
            ctx,
            semantic,
            weight(
                "sdec.q.q0.output_proj.weight",
                1,
                vector_dim,
                config.codebook_dim),
            nullptr);

        ggml_tensor* acoustic = nullptr;
        const int32_t acoustic_count =
            config.num_quantizers - config.num_semantic_quantizers;
        for (int32_t layer = 0; layer < acoustic_count; ++layer) {
            ggml_tensor* decoded = decode_codebook(
                "sdec.q.q1.rvq.l." + std::to_string(layer),
                config.num_semantic_quantizers + layer);
            acoustic = acoustic == nullptr ? decoded : ggml_add(ctx, acoustic, decoded);
        }
        acoustic = causal_conv(
            ctx,
            acoustic,
            weight(
                "sdec.q.q1.output_proj.weight",
                1,
                vector_dim,
                config.codebook_dim),
            nullptr);

        ggml_tensor* hidden = ggml_add(ctx, semantic, acoustic);
        hidden = causal_conv(
            ctx,
            hidden,
            weight(
                "sdec.pre_conv.conv.weight",
                3,
                config.codebook_dim,
                config.latent_dim),
            weight("sdec.pre_conv.conv.bias", config.latent_dim));
        hidden = transpose_2d(ctx, hidden);
        hidden = linear(
            ctx,
            weight(
                "sdec.pt.input_proj.weight",
                config.latent_dim,
                config.hidden_size),
            weight("sdec.pt.input_proj.bias", config.hidden_size),
            hidden);

        for (int32_t layer = 0; layer < config.block_count; ++layer) {
            hidden = transformer_layer(
                ctx,
                hidden,
                "sdec.pt.l." + std::to_string(layer) + ".",
                positions_tensor,
                mask_tensor,
                weights_,
                config);
        }
        hidden = rms_norm(
            ctx,
            hidden,
            weight("sdec.pt.norm.weight", config.hidden_size),
            config.rms_norm_epsilon);
        hidden = linear(
            ctx,
            weight(
                "sdec.pt.output_proj.weight",
                config.hidden_size,
                config.latent_dim),
            weight("sdec.pt.output_proj.bias", config.latent_dim),
            hidden);
        hidden = transpose_2d(ctx, hidden);

        for (size_t block = 0;
             block < config.upsampling_ratios.size();
             ++block) {
            const std::string prefix = "sdec.upsample." + std::to_string(block);
            const int32_t ratio = config.upsampling_ratios[block];
            hidden = causal_transposed_conv(
                ctx,
                hidden,
                transposed_weight(
                    prefix + ".0.conv.weight"),
                weight(prefix + ".0.conv.bias", config.latent_dim),
                ratio,
                use_matrix_transposed_conv_);
            hidden = convnext_block(
                ctx,
                hidden,
                weight(prefix + ".1.dwconv.conv.weight"),
                weight(prefix + ".1.dwconv.conv.bias", config.latent_dim),
                weight(prefix + ".1.norm.weight", config.latent_dim),
                weight(prefix + ".1.norm.bias", config.latent_dim),
                weight(
                    prefix + ".1.pwconv1.weight",
                    config.latent_dim,
                    config.latent_dim * 4),
                weight(
                    prefix + ".1.pwconv1.bias",
                    config.latent_dim * 4),
                weight(
                    prefix + ".1.pwconv2.weight",
                    config.latent_dim * 4,
                    config.latent_dim),
                weight(prefix + ".1.pwconv2.bias", config.latent_dim),
                weight(prefix + ".1.gamma", config.latent_dim));
        }

        hidden = causal_conv(
            ctx,
            hidden,
            weight(
                "sdec.decoder.0.conv.weight",
                7,
                config.latent_dim,
                config.decoder_dim),
            weight("sdec.decoder.0.conv.bias", config.decoder_dim));

        int32_t channels = config.decoder_dim;
        for (size_t block = 0;
             block < config.upsample_rates.size();
             ++block) {
            const std::string prefix = "sdec.decoder." + std::to_string(block + 1);
            const int32_t stride = config.upsample_rates[block];
            const int32_t output_channels = channels / 2;
            hidden = snake_beta(
                ctx,
                hidden,
                weight(prefix + ".block.0.alpha", channels),
                weight(prefix + ".block.0.beta", channels));
            hidden = causal_transposed_conv(
                ctx,
                hidden,
                transposed_weight(
                    prefix + ".block.1.conv.weight"),
                weight(prefix + ".block.1.conv.bias", output_channels),
                stride,
                use_matrix_transposed_conv_);

            const int32_t dilations[] = {1, 3, 9};
            for (int32_t unit = 0; unit < 3; ++unit) {
                const std::string unit_prefix =
                    prefix + ".block." + std::to_string(unit + 2);
                hidden = decoder_residual_unit(
                    ctx,
                    hidden,
                    weight(unit_prefix + ".act1.alpha", output_channels),
                    weight(unit_prefix + ".act1.beta", output_channels),
                    weight(
                        unit_prefix + ".conv1.conv.weight",
                        7,
                        output_channels,
                        output_channels),
                    weight(unit_prefix + ".conv1.conv.bias", output_channels),
                    weight(unit_prefix + ".act2.alpha", output_channels),
                    weight(unit_prefix + ".act2.beta", output_channels),
                    weight(
                        unit_prefix + ".conv2.conv.weight",
                        1,
                        output_channels,
                        output_channels),
                    weight(unit_prefix + ".conv2.conv.bias", output_channels),
                    dilations[unit]);
            }
            channels = output_channels;
        }

        const std::string final_snake =
            "sdec.decoder." +
            std::to_string(config.upsample_rates.size() + 1);
        hidden = snake_beta(
            ctx,
            hidden,
            weight(final_snake + ".alpha", channels),
            weight(final_snake + ".beta", channels));
        const std::string final_conv =
            "sdec.decoder." +
            std::to_string(config.upsample_rates.size() + 2);
        hidden = causal_conv(
            ctx,
            hidden,
            weight(final_conv + ".conv.weight", 7, channels, 1),
            weight(final_conv + ".conv.bias", 1));
        ggml_tensor* output = ggml_clamp(ctx, hidden, -1.0f, 1.0f);
        ggml_set_name(output, "speech_decoder_output");
        ggml_set_output(output);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(graph, output);
        return graph;
    }
}

Status SpeechTokenizer::Decoder::execute_graph(
    Qwen3TTSContext& runtime,
    ggml_cgraph* graph,
    const SpeechCodeMatrix& codes,
    AudioBuffer& audio) {
    audio = {};
    if (runtime.backend() == nullptr ||
        graph == nullptr ||
        decode_scheduler_ == nullptr) {
        return Status::error("SpeechDecoder execution state is not initialized");
    }

    const SpeechDecoderConfig& config = config_.decoder;
    const int64_t sequence_length = codes.frames;
    const bool profile = std::getenv("QWEN3TTS_PROFILE_SPEECH") != nullptr;
    auto profile_start = std::chrono::steady_clock::now();
    const auto report_profile = [&](const char* label) {
        if (!profile) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(
            stderr,
            "SpeechDecoder %s: %.2f ms\n",
            label,
            std::chrono::duration<double, std::milli>(
                now - profile_start).count());
        profile_start = now;
    };
    if (sequence_length > config.context_length) {
        return Status::error("SpeechDecoder context length exceeded");
    }
    std::vector<std::vector<int32_t>> layer_codes(config.num_quantizers);
    for (int32_t layer = 0; layer < config.num_quantizers; ++layer) {
        layer_codes[layer].resize(static_cast<size_t>(sequence_length));
        for (int64_t frame = 0; frame < sequence_length; ++frame) {
            layer_codes[layer][frame] =
                codes.data[
                    static_cast<size_t>(frame) * config.num_quantizers +
                    layer];
        }
    }

    std::vector<int32_t> positions(static_cast<size_t>(sequence_length));
    for (int64_t i = 0; i < sequence_length; ++i) {
        positions[i] = static_cast<int32_t>(i);
    }

    const size_t mask_elements =
        static_cast<size_t>(sequence_length) *
        static_cast<size_t>(sequence_length);
    std::vector<ggml_fp16_t> mask(
        mask_elements,
        ggml_fp32_to_fp16(-INFINITY));
    for (int64_t query = 0; query < sequence_length; ++query) {
        const int64_t first_key =
            config.use_sliding_window
            ? std::max<int64_t>(0, query - config.sliding_window + 1)
            : 0;
        for (int64_t key = first_key; key <= query; ++key) {
            const size_t index =
                static_cast<size_t>(query) * sequence_length + key;
            mask[index] = ggml_fp32_to_fp16(0.0f);
        }
    }

    ggml_backend_sched_reset(decode_scheduler_);
    if (!ggml_backend_sched_alloc_graph(decode_scheduler_, graph)) {
        return Status::error("failed to allocate SpeechDecoder graph");
    }
    report_profile("graph allocation");
    if (profile) {
        std::fprintf(
            stderr,
            "SpeechDecoder scheduler splits: %d\n",
            ggml_backend_sched_get_n_splits(decode_scheduler_));
    }
    for (size_t layer = 0; layer < layer_codes.size(); ++layer) {
        const std::string name = "speech_codes." + std::to_string(layer);
        ggml_tensor* input = ggml_graph_get_tensor(graph, name.c_str());
        if (input == nullptr) {
            return Status::error("missing graph input: " + name);
        }
        ggml_backend_tensor_set(
            input,
            layer_codes[layer].data(),
            0,
            layer_codes[layer].size() * sizeof(int32_t));
    }
    ggml_tensor* positions_input = ggml_graph_get_tensor(graph, "positions");
    ggml_tensor* mask_input = ggml_graph_get_tensor(graph, "attention_mask");
    if (positions_input == nullptr || mask_input == nullptr) {
        return Status::error("missing SpeechDecoder positions/mask input");
    }
    ggml_backend_tensor_set(
        positions_input,
        positions.data(),
        0,
        positions.size() * sizeof(int32_t));
    ggml_backend_tensor_set(
        mask_input,
        mask.data(),
        0,
        mask.size() * sizeof(ggml_fp16_t));
    report_profile("input upload");

    if (ggml_backend_sched_graph_compute(
            decode_scheduler_,
            graph) != GGML_STATUS_SUCCESS) {
        return Status::error("SpeechDecoder graph computation failed");
    }
    report_profile("compute");

    ggml_tensor* output =
        ggml_graph_get_tensor(graph, "speech_decoder_output");
    if (output == nullptr) {
        return Status::error("missing SpeechDecoder output tensor");
    }

    const int64_t expected_samples =
        sequence_length *
        static_cast<int64_t>(config_.decode_upsample_rate);
    if (output->ne[0] != expected_samples || output->ne[1] != 1) {
        return Status::error(
            "unexpected SpeechDecoder output shape [" +
            std::to_string(output->ne[0]) + ", " +
            std::to_string(output->ne[1]) + "]");
    }

    if (output->type == GGML_TYPE_F32) {
        audio.samples.resize(static_cast<size_t>(expected_samples));
        ggml_backend_tensor_get(
            output,
            audio.samples.data(),
            0,
            audio.samples.size() * sizeof(float));
    } else if (output->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(static_cast<size_t>(expected_samples));
        ggml_backend_tensor_get(
            output,
            half.data(),
            0,
            half.size() * sizeof(ggml_fp16_t));
        audio.samples.resize(half.size());
        ggml_fp16_to_fp32_row(half.data(), audio.samples.data(), half.size());
    } else {
        return Status::error("unsupported SpeechDecoder output dtype");
    }
    report_profile("output readback");
    audio.sample_rate = config_.output_sample_rate;
    audio.channels = 1;
    return Status::ok();
}

Status SpeechTokenizer::Decoder::decode(
    Qwen3TTSContext& runtime,
    const SpeechCodeMatrix& codes,
    AudioBuffer& audio) {
    audio = {};
    const SpeechDecoderConfig& config = config_.decoder;
    if (codes.frames <= 0 || codes.groups != config.num_quantizers) {
        return Status::error(
            "SpeechDecoder code group count does not match configuration");
    }
    if (codes.data.size() !=
        static_cast<size_t>(codes.frames) * config.num_quantizers) {
        return Status::error(
            "SpeechCodeMatrix data size does not match frames/groups");
    }
    for (int32_t code : codes.data) {
        if (code < 0 || code >= config.codebook_size) {
            return Status::error(
                "speech code is outside the decoder codebook");
        }
    }

    std::vector<uint8_t> graph_metadata(
        ggml_tensor_overhead() * 8192 + ggml_graph_overhead());

    ggml_init_params params{};
    params.mem_size = graph_metadata.size();
    params.mem_buffer = graph_metadata.data();
    params.no_alloc = true;
    ggml_context* graph_ctx = ggml_init(params);
    if (graph_ctx == nullptr) {
        return Status::error("failed to create SpeechDecoder graph context");
    }

    Status status = Status::ok();
    try {
        const auto build_start = std::chrono::steady_clock::now();
        ggml_cgraph* graph = build_graph(
            graph_ctx,
            codes.frames);
        if (std::getenv("QWEN3TTS_PROFILE_SPEECH") != nullptr) {
            std::fprintf(
                stderr,
                "SpeechDecoder graph build: %.2f ms\n",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - build_start).count());
        }
        status = execute_graph(
            runtime,
            graph,
            codes,
            audio);
    } catch (const std::exception& error) {
        status = Status::error(error.what());
    }

    ggml_free(graph_ctx);
    return status;
}

SpeechTokenizer::SpeechTokenizer()
    : decoder_(std::make_unique<Decoder>()) {
}

SpeechTokenizer::~SpeechTokenizer() = default;

Status SpeechTokenizer::load(
    const SpeechTokenizerConfig& config,
    Qwen3TTSContext& runtime) {
    if (decoder_ == nullptr) {
        decoder_ = std::make_unique<Decoder>();
    }
    return decoder_->load_tensors(config, runtime);
}

Status SpeechTokenizer::decode(
    Qwen3TTSContext& runtime,
    const SpeechCodeMatrix& codes,
    AudioBuffer& audio) {
    if (decoder_ == nullptr) {
        return Status::error("SpeechTokenizer is not initialized");
    }
    return decoder_->decode(runtime, codes, audio);
}

} // namespace qwen3_tts
