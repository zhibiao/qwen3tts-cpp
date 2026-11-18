# Cross-platform build

The project uses C++17, CMake, and vendored GGML. CPU builds support Windows
x64 and Linux x86_64. CUDA is an optional backend and requires an NVIDIA
toolkit and driver.

## Supported configurations

| Platform | Compiler | Preset |
|---|---|---|
| Windows 10/11 x64 | Visual Studio 2026 | `windows-cpu`, `windows-cuda` |
| Linux x86_64 | GCC 11+ or Clang 14+ | `linux-cpu`, `linux-cuda` |

Requirements:

- CMake 3.21 or newer
- C++17 compiler
- CUDA Toolkit for CUDA presets
- Python only for model conversion and Python helper tests

## Windows

CPU Release:

```powershell
cmake --preset windows-cpu
cmake --build --preset windows-cpu --parallel
```

CUDA Release:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
cmake --preset windows-cuda
cmake --build --preset windows-cuda --parallel
```

Artifacts:

```text
build/windows-cpu/bin/Release/qwen3tts-cli.exe
build/windows-cuda/bin/Release/qwen3tts-cli.exe
```

## Linux

Install basic dependencies:

```bash
sudo apt install build-essential cmake
```

CPU Release:

```bash
cmake --preset linux-cpu
cmake --build --preset linux-cpu --parallel
```

CUDA Release:

```bash
cmake --preset linux-cuda \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build --preset linux-cuda --parallel
```

Artifacts:

```text
build/linux-cpu/bin/qwen3tts-cli
build/linux-cuda/bin/qwen3tts-cli
```

On Linux, shared GGML backends and the executable use an `$ORIGIN` RPATH so
they can be stored together in `bin/`.

## Tests

Default presets set `BUILD_TESTING=OFF`. Test commands, coverage, and RTF
benchmarks are documented in [TESTING.md](TESTING.md).

## CUDA architectures

`windows-cuda` and `linux-cuda` use `native`. They compile only the
architecture detected on the build machine.

For redistributable builds use:

```powershell
cmake --preset windows-cuda-portable
cmake --build --preset windows-cuda-portable --parallel
```

```bash
cmake --preset linux-cuda-portable
cmake --build --preset linux-cuda-portable --parallel
```

The portable preset uses:

```text
75-real;75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real;121a-real
```

## Paths and fixtures

Models are read from `models/` and golden fixtures from `testdata/` by
default. They can be overridden without changing source files:

```text
QWEN3TTS_MODEL_DIR
QWEN3TTS_TEST_DATA_DIR
```

Linux example:

```bash
cmake --preset linux-cuda \
  -DQWEN3TTS_MODEL_DIR=/data/qwen3-tts/models \
  -DQWEN3TTS_TEST_DATA_DIR=/data/qwen3-tts/testdata
```

Windows example:

```powershell
cmake --preset windows-cuda `
  -DQWEN3TTS_MODEL_DIR="D:/qwen3-tts/models" `
  -DQWEN3TTS_TEST_DATA_DIR="D:/qwen3-tts/testdata"
```

Source, CMake, and test paths use `/`, which is accepted on both platforms.
Runtime APIs accept UTF-8 paths.

## Without presets

Windows:

```powershell
cmake -S . -B build/windows-cpu -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DQWEN3TTS_ENABLE_CUDA=OFF
cmake --build build/windows-cpu --config Release --parallel
```

Linux:

```bash
cmake -S . -B build/linux-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DQWEN3TTS_ENABLE_CUDA=OFF
cmake --build build/linux-cpu --parallel
```
