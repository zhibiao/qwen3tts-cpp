#pragma once

#include "config.h"
#include "status.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen3_tts {

class Qwen3TTSContext;

class Talker {
public:
    Talker();
    ~Talker();

    Status load_tensors(
        const TalkerConfig& config,
        Qwen3TTSContext& context);

    Status prefill(
        Qwen3TTSContext& context,
        const std::vector<float>& input_embeddings,
        int64_t frames,
        std::vector<float>& last_hidden,
        std::vector<float>& logits);

    Status decode_step(
        Qwen3TTSContext& context,
        const std::vector<float>& input_embedding,
        std::vector<float>& last_hidden,
        std::vector<float>& logits);

    Status embed_text_tokens(
        Qwen3TTSContext& context,
        const std::vector<int32_t>& token_ids,
        std::vector<float>& embeddings);

    Status embed_codec_tokens(
        Qwen3TTSContext& context,
        const std::vector<int32_t>& token_ids,
        std::vector<float>& embeddings);

private:
    // 常驻的 decode 图：按 kv_len（256 的倍数）建一次，桶不变就一直复用。
    // 生命周期内不 reset，CUDA Graph 才能持续命中。
    struct DecodeGraph {
        ggml_context* context = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_backend_sched_t scheduler = nullptr;
        std::vector<uint8_t> metadata;
        int64_t kv_len = 0;
        ggml_tensor* input = nullptr;
        ggml_tensor* positions = nullptr;
        ggml_tensor* cache_positions = nullptr;
        ggml_tensor* attention_mask = nullptr;
        ggml_tensor* output_hidden = nullptr;
        ggml_tensor* output_logits = nullptr;
    };

    // Talker 的四张图，各自一个构建函数。
    // 输入/输出张量靠名字与执行侧约定：
    //   text  : talker_text_ids        -> talker_text_embedding
    //   codec : talker_codec_ids       -> talker_codec_embedding
    //   prefill: talker_input/positions/cache_positions/attention_mask
    //            -> talker_hidden / talker_logits
    //   decode : talker_input/positions/cache_positions/attention_mask
    //            -> talker_hidden / talker_logits
    ggml_cgraph* build_text_embed_graph(
        ggml_context* ctx,
        int64_t frames);
    ggml_cgraph* build_codec_embed_graph(
        ggml_context* ctx,
        int64_t frames);
    ggml_cgraph* build_prefill_graph(
        ggml_context* ctx,
        int64_t frames);
    ggml_cgraph* build_decode_graph(
        ggml_context* ctx,
        int64_t kv_len);

    // prefill / decode 共用的主干：28 层 + final norm + codec_head。
    // kv_len = KV 视图宽度（prefill 为 frames，decode 为按桶取整后的常量）。
    ggml_tensor* build_talker_body(
        ggml_context* ctx,
        ggml_tensor* input,
        ggml_tensor* positions,
        ggml_tensor* cache_positions,
        ggml_tensor* attention_mask,
        int64_t frames,
        int64_t kv_len,
        std::vector<ggml_tensor*>& cache_writes);

    // 执行：四张图各一个执行函数，与上面的 build_* 一一对应。
    //   text  / codec 走独立的 embedding_scheduler_，
    //   prefill 走 prefill_scheduler_，
    //   decode 复用 decode_graph_（自带 scheduler，永不 reset）。
    Status run_text_embed_graph(
        Qwen3TTSContext& context,
        const std::vector<int32_t>& token_ids,
        std::vector<float>& embeddings);
    Status run_codec_embed_graph(
        Qwen3TTSContext& context,
        const std::vector<int32_t>& token_ids,
        std::vector<float>& embeddings);
    Status run_prefill_graph(
        Qwen3TTSContext& context,
        const std::vector<float>& input_embeddings,
        int64_t frames,
        std::vector<float>& last_hidden,
        std::vector<float>& logits);
    Status run_decode_graph(
        Qwen3TTSContext& context,
        const std::vector<float>& input_embedding,
        std::vector<float>& last_hidden,
        std::vector<float>& logits);
    Status init_decode_graph(
        Qwen3TTSContext& context,
        int64_t kv_len);
    void release_decode_graph() noexcept;
    Status init_scheduler(Qwen3TTSContext& context);
    void release_scheduler() noexcept;
    void release_weights() noexcept;

    ggml_context* context = nullptr;
    ggml_context* kv_context = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_backend_buffer_t kv_buffer = nullptr;
    ggml_tensor* cache_k = nullptr;
    ggml_tensor* cache_v = nullptr;
    std::unordered_map<std::string, ggml_tensor*> weights;
    TalkerConfig config_;
    int64_t n_past = 0;
    std::vector<uint8_t> graph_metadata_;
    DecodeGraph decode_graph_;
    ggml_backend_sched_t prefill_scheduler_ = nullptr;
    ggml_backend_sched_t embedding_scheduler_ = nullptr;
};

} // namespace qwen3_tts
