#pragma once

#include "status.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qwen3_tts {

class GGUFModel {
public:
    GGUFModel() = default;
    ~GGUFModel();

    static Status open(
        const std::string& path,
        std::unique_ptr<GGUFModel>& model);

    bool has_key(const std::string& key) const;
    gguf_type key_type(const std::string& key) const;

    bool get_bool(const std::string& key, bool fallback = false) const;
    int32_t get_i32(const std::string& key, int32_t fallback = 0) const;
    uint32_t get_u32(const std::string& key, uint32_t fallback = 0) const;
    int64_t get_i64(const std::string& key, int64_t fallback = 0) const;
    float get_f32(const std::string& key, float fallback = 0.0f) const;
    std::string get_string(const std::string& key, const std::string& fallback = {}) const;

    std::vector<std::string> get_string_array(const std::string& key) const;
    std::vector<int32_t> get_i32_array(const std::string& key) const;

    int64_t tensor_count() const;
    int64_t kv_count() const;
    std::vector<std::string> keys() const;
    const ggml_tensor* tensor(int64_t index) const;
    const ggml_tensor* tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const;

    const std::string& path() const noexcept { return path_; }
    const gguf_context* raw_gguf() const noexcept { return gguf_; }

private:
    int64_t find_key(const std::string& key) const;
    Status map_file();
    void unmap_file() noexcept;

    std::string path_;
    gguf_context* gguf_ = nullptr;
    ggml_context* tensor_context_ = nullptr;
    void* mapped_data_ = nullptr;
    size_t mapped_size_ = 0;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#else
    int file_descriptor_ = -1;
#endif
};

} // namespace qwen3_tts
