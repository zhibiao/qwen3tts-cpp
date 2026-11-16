#include "gguf_model.h"

#include <filesystem>
#include <limits>
#include <utility>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace qwen3_tts {

namespace {

bool has_type(gguf_type actual, gguf_type expected) {
    return actual == expected;
}

} // namespace

GGUFModel::~GGUFModel() {
    if (tensor_context_ != nullptr) {
        ggml_free(tensor_context_);
        tensor_context_ = nullptr;
    }
    if (gguf_ != nullptr) {
        gguf_free(gguf_);
        gguf_ = nullptr;
    }
    unmap_file();
}

Status GGUFModel::open(
    const std::string& path,
    std::unique_ptr<GGUFModel>& model) {
    auto result = std::unique_ptr<GGUFModel>(new GGUFModel());
    result->path_ = path;

    Status status = result->map_file();
    if (!status) {
        return status;
    }

    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;

    result->gguf_ = gguf_init_from_file(path.c_str(), params);
    if (result->gguf_ == nullptr) {
        return Status::error("failed to load GGUF file: " + path);
    }

    const int64_t tensor_count = gguf_get_n_tensors(result->gguf_);
    ggml_init_params tensor_params{};
    tensor_params.mem_size =
        ggml_tensor_overhead() * static_cast<size_t>(tensor_count) + 1024;
    tensor_params.mem_buffer = nullptr;
    tensor_params.no_alloc = true;
    result->tensor_context_ = ggml_init(tensor_params);
    if (result->tensor_context_ == nullptr) {
        return Status::error("failed to create GGUF tensor context");
    }

    const uint64_t data_offset = gguf_get_data_offset(result->gguf_);
    for (int64_t i = 0; i < tensor_count; ++i) {
        const int64_t* ne = gguf_get_tensor_ne(result->gguf_, i);
        const int n_dims =
            ne[3] > 1 ? 4 :
            ne[2] > 1 ? 3 :
            ne[1] > 1 ? 2 : 1;
        ggml_tensor* tensor = ggml_new_tensor(
            result->tensor_context_,
            gguf_get_tensor_type(result->gguf_, i),
            n_dims,
            ne);
        if (tensor == nullptr) {
            return Status::error("failed to create GGUF tensor descriptor");
        }
        ggml_set_name(tensor, gguf_get_tensor_name(result->gguf_, i));

        const uint64_t offset =
            data_offset + gguf_get_tensor_offset(result->gguf_, i);
        const size_t size = gguf_get_tensor_size(result->gguf_, i);
        if (offset > result->mapped_size_ ||
            size > result->mapped_size_ - offset) {
            return Status::error("GGUF tensor data is outside mapped file");
        }
        tensor->data = static_cast<uint8_t*>(result->mapped_data_) + offset;
    }

    model = std::move(result);
    return Status::ok();
}

Status GGUFModel::map_file() {
#if defined(_WIN32)
    const std::wstring wide_path = std::filesystem::u8path(path_).wstring();
    HANDLE file = CreateFileW(
        wide_path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return Status::error("failed to open GGUF file for mapping: " + path_);
    }

    LARGE_INTEGER file_size{};
    if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart <= 0) {
        CloseHandle(file);
        return Status::error("failed to get GGUF file size: " + path_);
    }
    if (static_cast<uint64_t>(file_size.QuadPart) >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        CloseHandle(file);
        return Status::error("GGUF file is too large for this platform: " + path_);
    }

    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        CloseHandle(file);
        return Status::error("failed to create GGUF file mapping: " + path_);
    }

    void* data = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (data == nullptr) {
        CloseHandle(mapping);
        CloseHandle(file);
        return Status::error("failed to map GGUF file: " + path_);
    }

    file_handle_ = file;
    mapping_handle_ = mapping;
    mapped_data_ = data;
    mapped_size_ = static_cast<size_t>(file_size.QuadPart);
    return Status::ok();
#else
    const int descriptor = ::open(path_.c_str(), O_RDONLY);
    if (descriptor < 0) {
        return Status::error("failed to open GGUF file for mapping: " + path_);
    }
    struct stat file_stat {};
    if (fstat(descriptor, &file_stat) != 0 || file_stat.st_size <= 0) {
        ::close(descriptor);
        return Status::error("failed to get GGUF file size: " + path_);
    }
    if (static_cast<uint64_t>(file_stat.st_size) >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        ::close(descriptor);
        return Status::error("GGUF file is too large for this platform: " + path_);
    }
    void* data = mmap(
        nullptr,
        static_cast<size_t>(file_stat.st_size),
        PROT_READ,
        MAP_PRIVATE,
        descriptor,
        0);
    if (data == MAP_FAILED) {
        ::close(descriptor);
        return Status::error("failed to map GGUF file: " + path_);
    }
    file_descriptor_ = descriptor;
    mapped_data_ = data;
    mapped_size_ = static_cast<size_t>(file_stat.st_size);
    return Status::ok();
#endif
}

void GGUFModel::unmap_file() noexcept {
#if defined(_WIN32)
    if (mapped_data_ != nullptr) {
        UnmapViewOfFile(mapped_data_);
    }
    if (mapping_handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(mapping_handle_));
    }
    if (file_handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(file_handle_));
    }
    mapped_data_ = nullptr;
    mapping_handle_ = nullptr;
    file_handle_ = nullptr;
#else
    if (mapped_data_ != nullptr && mapped_size_ != 0) {
        munmap(mapped_data_, mapped_size_);
    }
    if (file_descriptor_ >= 0) {
        ::close(file_descriptor_);
    }
    mapped_data_ = nullptr;
    file_descriptor_ = -1;
#endif
    mapped_size_ = 0;
}

int64_t GGUFModel::find_key(const std::string& key) const {
    return gguf_ == nullptr ? -1 : gguf_find_key(gguf_, key.c_str());
}

bool GGUFModel::has_key(const std::string& key) const {
    return find_key(key) >= 0;
}

gguf_type GGUFModel::key_type(const std::string& key) const {
    const int64_t id = find_key(key);
    return id < 0 ? GGUF_TYPE_COUNT : gguf_get_kv_type(gguf_, id);
}

bool GGUFModel::get_bool(const std::string& key, bool fallback) const {
    const int64_t id = find_key(key);
    return id >= 0 && has_type(gguf_get_kv_type(gguf_, id), GGUF_TYPE_BOOL)
        ? gguf_get_val_bool(gguf_, id)
        : fallback;
}

int32_t GGUFModel::get_i32(const std::string& key, int32_t fallback) const {
    const int64_t id = find_key(key);
    return id >= 0 && has_type(gguf_get_kv_type(gguf_, id), GGUF_TYPE_INT32)
        ? gguf_get_val_i32(gguf_, id)
        : fallback;
}

uint32_t GGUFModel::get_u32(const std::string& key, uint32_t fallback) const {
    const int64_t id = find_key(key);
    return id >= 0 && has_type(gguf_get_kv_type(gguf_, id), GGUF_TYPE_UINT32)
        ? gguf_get_val_u32(gguf_, id)
        : fallback;
}

int64_t GGUFModel::get_i64(const std::string& key, int64_t fallback) const {
    const int64_t id = find_key(key);
    return id >= 0 && has_type(gguf_get_kv_type(gguf_, id), GGUF_TYPE_INT64)
        ? gguf_get_val_i64(gguf_, id)
        : fallback;
}

float GGUFModel::get_f32(const std::string& key, float fallback) const {
    const int64_t id = find_key(key);
    return id >= 0 && has_type(gguf_get_kv_type(gguf_, id), GGUF_TYPE_FLOAT32)
        ? gguf_get_val_f32(gguf_, id)
        : fallback;
}

std::string GGUFModel::get_string(const std::string& key, const std::string& fallback) const {
    const int64_t id = find_key(key);
    if (id < 0 || !has_type(gguf_get_kv_type(gguf_, id), GGUF_TYPE_STRING)) {
        return fallback;
    }
    const char* value = gguf_get_val_str(gguf_, id);
    return value == nullptr ? fallback : std::string(value);
}

std::vector<std::string> GGUFModel::get_string_array(const std::string& key) const {
    std::vector<std::string> result;
    const int64_t id = find_key(key);
    if (id < 0 || !has_type(gguf_get_kv_type(gguf_, id), GGUF_TYPE_ARRAY) ||
        !has_type(gguf_get_arr_type(gguf_, id), GGUF_TYPE_STRING)) {
        return result;
    }
    const size_t count = gguf_get_arr_n(gguf_, id);
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const char* value = gguf_get_arr_str(gguf_, id, i);
        result.emplace_back(value == nullptr ? "" : value);
    }
    return result;
}

std::vector<int32_t> GGUFModel::get_i32_array(const std::string& key) const {
    std::vector<int32_t> result;
    const int64_t id = find_key(key);
    if (id < 0 || !has_type(gguf_get_kv_type(gguf_, id), GGUF_TYPE_ARRAY) ||
        !has_type(gguf_get_arr_type(gguf_, id), GGUF_TYPE_INT32)) {
        return result;
    }
    const size_t count = gguf_get_arr_n(gguf_, id);
    const int32_t* values = static_cast<const int32_t*>(gguf_get_arr_data(gguf_, id));
    if (values == nullptr) {
        return result;
    }
    result.assign(values, values + count);
    return result;
}

int64_t GGUFModel::tensor_count() const {
    return gguf_ == nullptr ? 0 : gguf_get_n_tensors(gguf_);
}

int64_t GGUFModel::kv_count() const {
    return gguf_ == nullptr ? 0 : gguf_get_n_kv(gguf_);
}

std::vector<std::string> GGUFModel::keys() const {
    std::vector<std::string> result;
    const int64_t count = kv_count();
    result.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        result.emplace_back(gguf_get_key(gguf_, i));
    }
    return result;
}

const ggml_tensor* GGUFModel::tensor(int64_t index) const {
    if (tensor_context_ == nullptr || index < 0 || index >= tensor_count()) {
        return nullptr;
    }
    return ggml_get_tensor(tensor_context_, gguf_get_tensor_name(gguf_, index));
}

const ggml_tensor* GGUFModel::tensor(const std::string& name) const {
    return tensor_context_ == nullptr
        ? nullptr
        : ggml_get_tensor(tensor_context_, name.c_str());
}

bool GGUFModel::has_tensor(const std::string& name) const {
    return gguf_ != nullptr && gguf_find_tensor(gguf_, name.c_str()) >= 0;
}

} // namespace qwen3_tts
