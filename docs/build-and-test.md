# 构建、运行与测试

## 依赖

- CMake 3.20+
- Visual Studio Community 2026，C++ x64 tools + Windows SDK
- Python 3.12 仅用于 GGUF 转换
- 可选：CUDA Toolkit 13.5

## 构建目录与产物

项目使用 Visual Studio 18 2026 多配置生成器。构建目录统一为 `build`，所有可执行文件
和共享库都输出到同一配置目录：

```text
build/bin/Debug/
build/bin/Release/
  qwen3tts-cli.exe
  ggml.dll  ggml-base.dll  ggml-cuda.dll  ggml-cpu-<变体>.dll
```

默认构建为共享库（`BUILD_SHARED_LIBS=ON`），backend 以 DLL 形式在运行时加载
（`GGML_BACKEND_DL=ON`）。`ggml_backend_load_all()` 扫描 EXE 所在目录，按
`ggml-cuda.dll` / `ggml-cpu-*.dll` 自动挑选可用后端，所以**这些 DLL 必须和 EXE
在同一个目录**。启动时日志会打印实际加载了哪些：

```text
load_backend: loaded CUDA backend from ...\ggml-cuda.dll
load_backend: loaded CPU backend from ...\ggml-cpu-haswell.dll
```

想回到单体静态构建：`-DBUILD_SHARED_LIBS=OFF`（`GGML_BACKEND_DL` 会随之关闭，
EXE 又变回单个文件）。

模型 GGUF 位于项目根目录的 `models`。CPU 与 CUDA 构建共用 `build`，重新配置
backend 并编译后，对应配置目录里的产物会被覆盖。

## CPU 构建

```powershell
cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -T host=x64 `
  -DBUILD_TESTING=ON `
  -DQWEN3TTS_ENABLE_CUDA=OFF

cmake --build build --config Debug --parallel
cmake --build build --config Release --parallel
```

Debug 使用 `/Od`，性能不代表最终速度。Release 使用 `/O2`。

## CUDA 构建

当前验证环境：

```text
CUDA Toolkit 13.5
GPU: NVIDIA GeForce RTX 2060
Compute capability: 7.5
Architecture: 75
```

在同一个 `build` 目录启用 CUDA：

```powershell
cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -T host=x64 `
  -DBUILD_TESTING=OFF `
  -DQWEN3TTS_ENABLE_CUDA=ON `
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.5/bin/nvcc.exe"

cmake --build build --config Debug --target qwen3tts-cli --parallel
cmake --build build --config Release --target qwen3tts-cli --parallel
```

`QWEN3TTS_CUDA_ARCHITECTURES` 默认是

```text
75-real;86-real;89-real;120-real;120-virtual
```

即 Turing / Ampere / Ada / Blackwell 各生成 SASS，最高架构额外生成 PTX 供更新的
显卡 JIT。只想为本机快速编一次可以显式传单架构：

```powershell
cmake -S . -B build -DQWEN3TTS_CUDA_ARCHITECTURES=75
```

规则：

- `-real` 只生成 SASS，`-virtual` 只生成 PTX，裸数字两种都生成
- `+PTX` 不是合法后缀，CMake 会直接报错
- CUDA 13 已移除 Maxwell / Pascal / Volta，写 `61` 或 `70` 会编译失败

配置阶段会打印实际生效的架构列表：

```text
-- Qwen3TTS CUDA architectures: 75-real;86-real;89-real;120-real;120-virtual
```

编译完成后可以核对二进制里到底有哪些：

```powershell
& "$env:CUDA_PATH\bin\cuobjdump.exe" -lelf .\build\bin\Release\ggml-cuda.dll |
  Select-String "sm_" | % { [regex]::Match($_.Line,'sm_\w+').Value } | Group-Object
```

CUDA CLI 依赖 `cublas` 和 `cublasLt`（`cudart` 已静态链接）。运行前加入 runtime 路径：

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\x64;" + $env:PATH
```

### VS Code / MSBuild 找不到 CUDA Toolkit

若出现：

```text
The CUDA Toolkit directory '' does not exist.
```

说明 `nvcc.exe` 可以被 CMake 找到，但 MSBuild 的 CUDA build customization 没拿到
`CudaToolkitDir`。先完全重启 VS Code，使其继承系统：

```text
CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5
CUDA_PATH_V13_5=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5
```

必要时从已设置 CUDA 环境的终端启动 VS Code。

## CLI

列出当前构建实际注册的设备：

```powershell
.\build\bin\Release\qwen3tts-cli.exe --list-devices
```

该命令不需要 `--model`、`--speech-tokenizer` 或 `--text`。设备名称可以直接用于
`--device`，例如 `CPU`、`CUDA0`。

```powershell
.\build\bin\Release\qwen3tts-cli.exe `
  --model .\testdata\qwen3-tts-talker-0.6b-customvoice-bf16.gguf `
  --speech-tokenizer .\testdata\qwen3-tts-speech-tokenizer-12hz-f32.gguf `
  --text "你好，世界。" `
  --speaker serena `
  --language Chinese `
  --max-new-tokens 128 `
  --seed 1234 `
  --greedy `
  --device CPU `
  --threads 8 `
  --output .\testdata\output.wav
```

CUDA 使用：

```powershell
.\build\bin\Release\qwen3tts-cli.exe `
  --model .\testdata\qwen3-tts-talker-0.6b-customvoice-bf16.gguf `
  --speech-tokenizer .\testdata\qwen3-tts-speech-tokenizer-12hz-f32.gguf `
  --text "你好，世界。" `
  --speaker serena `
  --language Chinese `
  --max-new-tokens 16 `
  --seed 1234 `
  --greedy `
  --device CUDA0 `
  --threads 8 `
  --output .\testdata\cuda-release.wav
```

`--threads 0` 或省略时使用：

```text
max(逻辑线程数 / 2, 1)
```

CLI 默认显示：

- 模型加载耗时
- Prompt 和 Talker prefill 耗时
- Talker 每一帧的 sampling、CodePredictor 总计、embedding、condition、decode 耗时
- CodePredictor bootstrap 和 14 个 residual group 的 graph、sampling、code
- 实际生成帧数
- SpeechDecoder 耗时
- WAV 写入耗时

Debug 构建会直接显示 ggml allocator 日志。Release 定义了 `NDEBUG`，
`ggml_gallocr_needs_realloc` 等 DEBUG 日志会被编译掉。

## 测试

测试目标统一由 `tests/CMakeLists.txt` 定义，测试可执行文件输出到
`build/bin/<Config>`。

```powershell
ctest --test-dir build -C Debug --output-on-failure
ctest --test-dir build -C Release --output-on-failure
```

本次验证结果：

```text
CPU Debug:   14/14 enabled tests passed，34.17s
CPU Release: 14/14 enabled tests passed，21.43s
qwen3tts_customvoice_golden disabled
```

测试覆盖：

- GGUF v3
- TextTokenizer
- SpeechTokenizer Decoder
- SpeechDecoder transposed-convolution math
- Talker
- CodePredictor
- sampler
- WAV writer
- 0.6B/1.7B golden
- Qwen3TTS multi-frame golden

CUDA 不运行 CPU golden 作为通过标准。CPU/CUDA 浮点累加顺序不同，可能产生不同
greedy code。CUDA Debug 已通过 0.6B 16 帧冒烟测试；Release 使用相同命令。

## 常见问题

### 看起来卡住

CodePredictor 每帧需要 15 个串行步骤，CPU 生成期间可能长时间没有新行。
当前 CLI 每 64 帧输出一次进度。可先用 `--max-new-tokens 16` 做冒烟测试。

### 大量 realloc 日志

Debug 构建会显示 scheduler 调试日志。

```text
graph has different number of nodes
```

表示切换到了不同图布局，不是模型错误。

### 找不到 CUDA runtime DLL

CUDA 构建运行期只需要 `cublas64_13.dll` 和 `cublasLt64_13.dll`（`cudart` 已静态
链接）。确认以下目录在 `PATH` 中，或把两个 DLL 拷到 EXE 旁边：

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\x64
```

### 启动时报 no kernel image / unsupported PTX

```text
CUDA error: the provided PTX was compiled with an unsupported toolchain
CUDA error: no kernel image is available for execution on the device
```

两种原因：

1. **二进制里没有目标显卡的架构**。用 `cuobjdump -lelf` 确认 `sm_XX` 是否包含
   目标卡的 compute capability，没有就重新配置 `QWEN3TTS_CUDA_ARCHITECTURES` 并
   重编（改架构会触发 `ggml-cuda` 全量重编）。
2. **Toolkit 比驱动新**。`nvidia-smi` 顶部的 `CUDA UMD Version` 必须 ≥ 编译用的
   toolkit 版本，否则任何需要 PTX JIT 的路径都会失败。要么换匹配的 toolkit，
   要么升级驱动。

### 只跑在 CPU 上（有显卡但没生效）

先看启动日志有没有这两行：

```text
load_backend: loaded CUDA backend from ...\ggml-cuda.dll
load_backend: loaded CPU backend from ...\ggml-cpu-haswell.dll
```

缺少 CUDA 那一行说明 `ggml-cuda.dll` 没被加载（不在 EXE 目录，或 `cublas` 缺失）。
可以在 CPU 上验证同一份包：

```powershell
.\build\bin\Release\qwen3tts-cli.exe --list-devices
```

### 性能过低

先确认使用 `build\bin\Release` 中的 EXE，而不是 `build\bin\Debug`。
