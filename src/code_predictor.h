#pragma once

#include "config.h"
#include "status.h"
#include "sampler.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen3_tts {

class Qwen3TTSContext;

class CodePredictor {
public:
    CodePredictor();
    ~CodePredictor();

    Status load(
        const CodePredictorConfig& config,
        Qwen3TTSContext& context);

    // 唯一入口：跑完一帧的全部 num_code_groups 个码字。
    // 默认 greedy（不采样），需要随机采样时显式传入 sampling。
    Status predict_frame(
        Qwen3TTSContext& context,
        const std::vector<float>& talker_hidden,
        int32_t c0,
        std::vector<int32_t>& codes,
        const SamplingOptions& sampling = SamplingOptions::greedy(),
        std::vector<float>* step_logits = nullptr,
        std::function<void(const std::string&)> log = {});

    void set_seed(uint64_t seed);

    // 建好 prefill / decode 两张常驻图（各含独立 scheduler），由外部在 load
    // 之后调用一次。重复调用无副作用；失败时两张图都保持未建状态。
    Status init_graphs(Qwen3TTSContext& context);

    Status embed_codes(
        Qwen3TTSContext& context,
        const std::vector<int32_t>& codes,
        std::vector<float>& embedding);

private:
    // 常驻图：建一次（init_graphs）、alloc 一次，之后每帧/每步只往输入张量
    // 写值再 compute。生命周期内不 reset，CUDA Graph 才能持续命中。
    struct PrefillGraph {
        ggml_context* context = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_backend_sched_t scheduler = nullptr;
        std::vector<uint8_t> metadata;
        ggml_tensor* code = nullptr;
        ggml_tensor* talker_hidden = nullptr;
        ggml_tensor* positions = nullptr;
        ggml_tensor* cache_positions = nullptr;
        ggml_tensor* attention_mask = nullptr;
        ggml_tensor* head = nullptr;
        ggml_tensor* output_logits = nullptr;
    };

    struct DecodeGraph {
        ggml_context* context = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_backend_sched_t scheduler = nullptr;
        std::vector<uint8_t> metadata;
        ggml_tensor* embedding_row = nullptr;
        ggml_tensor* positions = nullptr;
        ggml_tensor* cache_positions = nullptr;
        ggml_tensor* attention_mask = nullptr;
        ggml_tensor* head = nullptr;
        ggml_tensor* output_logits = nullptr;
    };

    struct EmbedGraph {
        ggml_context* context = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_backend_sched_t scheduler = nullptr;
        std::vector<uint8_t> metadata;
        std::vector<ggml_tensor*> code_inputs;
        ggml_tensor* output_sum = nullptr;
    };

    ggml_tensor* weight(
        const std::string& name,
        int64_t ne0,
        int64_t ne1 = -1) const;
    // CodePredictor 的三张图，各自一个构建函数。
    // 输入/输出靠名字与执行侧约定：
    //   embed  : code_predictor_embedding_code.<group>（每组一个）
    //            -> code_predictor_embedding_sum
    //   prefill: code_predictor_talker_hidden + code_predictor_code
    //            + positions / cache_positions / attention_mask / head
    //            -> code_predictor_logits
    //   decode : code_predictor_embedding_row
    //            + positions / cache_positions / attention_mask / head
    //            -> code_predictor_logits
    // 三张图都只输出 logits；选 token 一律在 CPU 上由 Sampler 完成。
    ggml_cgraph* build_embed_graph(ggml_context* ctx);
    ggml_cgraph* build_prefill_graph(ggml_context* ctx);
    ggml_cgraph* build_decode_graph(ggml_context* ctx);

    // prefill / decode 共用的主干：5 层 + final norm + lm_head。
    ggml_tensor* build_predictor_body(
        ggml_context* ctx,
        ggml_tensor* hidden,
        ggml_tensor* positions,
        ggml_tensor* cache_positions,
        ggml_tensor* attention_mask,
        ggml_tensor* head,
        int64_t sequence_length,
        std::vector<ggml_tensor*>& cache_writes);

    // 执行：三张图都复用 init_graphs 建好的常驻图。
    Status run_embed_graph(
        Qwen3TTSContext& context,
        const std::vector<int32_t>& codes,
        std::vector<float>& embedding);
    Status run_prefill_graph(
        Qwen3TTSContext& context,
        const float* talker_hidden,
        int32_t code,
        int32_t head,
        std::vector<float>& logits);
    Status run_decode_graph(
        Qwen3TTSContext& context,
        int32_t code,
        int32_t head,
        std::vector<float>& logits);
    Status init_embed_graph(Qwen3TTSContext& context);
    Status init_prefill_graph(Qwen3TTSContext& context);
    Status init_decode_graph(Qwen3TTSContext& context);
    void release_graphs() noexcept;
    void release_weights() noexcept;

    ggml_context* weight_context_ = nullptr;
    ggml_context* kv_context_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    ggml_backend_buffer_t kv_buffer_ = nullptr;
    ggml_tensor* cache_k_ = nullptr;
    ggml_tensor* cache_v_ = nullptr;
    ggml_tensor* packed_codec_embeddings_ = nullptr;
    ggml_tensor* packed_lm_heads_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> weights_;
    CodePredictorConfig config_;
    Sampler sampler_;
    int64_t n_past_ = 0;
    EmbedGraph embed_graph_;
    PrefillGraph prefill_graph_;
    DecodeGraph decode_graph_;
};

} // namespace qwen3_tts
