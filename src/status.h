#pragma once

#include <string>
#include <utility>

namespace qwen3_tts {

class Status {
public:
    static Status ok() {
        return Status(true, {});
    }

    static Status error(std::string message) {
        return Status(false, std::move(message));
    }

    explicit operator bool() const noexcept {
        return ok_;
    }

    bool is_ok() const noexcept {
        return ok_;
    }

    const std::string& message() const noexcept {
        return message_;
    }

private:
    Status(bool ok, std::string message)
        : ok_(ok), message_(std::move(message)) {
    }

    bool ok_;
    std::string message_;
};

} // namespace qwen3_tts
