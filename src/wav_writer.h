#pragma once

#include "status.h"
#include "types.h"

#include <string>

namespace qwen3_tts {

Status write_wav(
    const std::string& path,
    const AudioBuffer& audio);

} // namespace qwen3_tts
