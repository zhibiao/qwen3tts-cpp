# Qwen3-TTS.cpp

基于 C++/ggml 的 Qwen3-TTS 12 Hz CustomVoice 推理实现。

当前状态：

- 支持 0.6B / 1.7B CustomVoice
- 支持 speaker、language、speaker dialect
- TextTokenizer -> Talker -> CodePredictor -> SpeechDecoder 全流程
- 输出 24 kHz PCM16 WAV
- Talker、CodePredictor 和 SpeechDecoder 均使用 `ggml_flash_attn_ext`
- Talker、CodePredictor 的 MLP 使用融合 SwiGLU，RMSNorm 权重使用 F32 runtime 副本
- CodePredictor 使用固定 16 位 KV view、打包 LM Head/code embedding 和缓存 decode graph
- CodePredictor 使用固定 decode graph 与持久 scheduler plan，SpeechDecoder 使用 1 个 scheduler
- Talker KV cache 默认 `num_ctx=2048`，模型 RoPE context 仍保留 GGUF 配置
- GGUF schema v3，Talker 与 SpeechTokenizer 分成两个文件
- CPU/CUDA Debug/Release 启用测试均为 `14/14 passed`
- CUDA SpeechDecoder 保持模型权重 dtype，并使用 phase-decomposed transposed convolution

当前不支持：

- 非空 `instruct`
- Base / Voice Clone
- VoiceDesign
- SpeechTokenizer Encoder 和 codec 命令行工具
- C++ GGUF 量化工具
- 流式生成 API
- 完整 CUDA 性能调优

## 文档

- [架构与推理流程](docs/architecture.md)
- [GGUF v3 与模型转换](docs/gguf.md)
- [构建、运行与测试](docs/build-and-test.md)
- [性能分析与优化计划](docs/performance.md)
- [后续路线图](docs/roadmap.md)

## 目录与产物

模型文件位于 `testdata`：

```text
testdata/qwen3-tts-talker-0.6b-customvoice-bf16.gguf
testdata/qwen3-tts-talker-1.7b-customvoice-bf16.gguf
testdata/qwen3-tts-speech-tokenizer-12hz-f32.gguf
```

项目所有产物统一输出到 `build/bin/<Config>`：

```text
build/bin/Debug/
build/bin/Release/
  qwen3tts-cli.exe
  ggml.dll  ggml-base.dll  ggml-cuda.dll  ggml-cpu-<变体>.dll
```

默认构建共享库，backend 以 DLL 形式在运行时加载，所以**这些 DLL 必须和 EXE 放在
同一目录**。CPU 和 CUDA 共用同一个 `build` 目录，切换 `QWEN3TTS_ENABLE_CUDA` 后
重新编译时，对应配置目录里的产物会被当前 backend 的版本覆盖。

## 快速运行

先构建 Release，然后运行：

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

Debug 构建请将 EXE 路径替换为 `.\build\bin\Debug\qwen3tts-cli.exe`。

列出当前构建实际注册的 backend 设备：

```powershell
.\build\bin\Release\qwen3tts-cli.exe --list-devices
```

该命令不需要模型参数。CUDA 构建的示例输出：

```text
Devices: 2
0: CUDA0 | GPU | NVIDIA GeForce RTX 2060 | 5113.0 / 6143.7 MiB | 0000:01:00.0
1: CPU | CPU | AMD Ryzen 7 4800H with Radeon Graphics | 6468.5 / 16253.3 MiB
```

列出模型内置的说话人（只读 GGUF 元数据，不加载权重）：

```powershell
.\build\bin\Release\qwen3tts-cli.exe `
  --model .\testdata\qwen3-tts-talker-0.6b-customvoice-q8_0.gguf `
  --list-speakers
```

```text
Speakers: 9
0: aiden | id=2861
1: dylan | id=2878 | dialect=beijing_dialect
...
5: serena | id=3066
```

名字可直接用于 `--speaker`；带 `dialect` 的说话人会据此自动选择语言。

CLI 默认显示模型加载、Prompt/prefill、Talker 每帧和 CodePredictor
每个 code group 的耗时、SpeechDecoder 和 WAV 写入耗时。Debug 构建会直接
输出 ggml allocator 日志。

## 构建

当前 VS Code 使用：

```text
Generator: Visual Studio 18 2026
Platform:  x64
Toolset:   host=x64
Build dir: build
```

CPU 构建配置：

```powershell
cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -T host=x64 `
  -DBUILD_TESTING=ON `
  -DQWEN3TTS_ENABLE_CUDA=OFF
```

CPU Debug/Release：

```powershell
cmake --build build --config Debug --parallel
cmake --build build --config Release --parallel
```

CUDA 构建仍使用同一个 `build` 目录。本项目验证环境为 CUDA 13.5、RTX 2060、
compute capability 7.5：

```powershell
cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -T host=x64 `
  -DBUILD_TESTING=OFF `
  -DQWEN3TTS_ENABLE_CUDA=ON `
  -DQWEN3TTS_CUDA_ARCHITECTURES=75 `
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.5/bin/nvcc.exe"
```

```powershell
cmake --build build --config Debug --target qwen3tts-cli --parallel
cmake --build build --config Release --target qwen3tts-cli --parallel
```

CUDA EXE 运行时需要 CUDA runtime DLL：

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\x64;" + $env:PATH
```

VS Code 中直接点击 Build 时，请在 CMake Tools 中把
`QWEN3TTS_ENABLE_CUDA` 设为 ON。若 VS 报告 `CudaToolkitDir` 或
`The CUDA Toolkit directory '' does not exist`，先完整重启 VS Code，让其继承系统
`CUDA_PATH`；仍无效时，在启动 VS Code 的终端中设置 CUDA 环境后再打开。

## 测试

测试目标统一定义在 `tests/CMakeLists.txt`，仅在 `BUILD_TESTING=ON` 时加入构建。

CPU 完整测试：

```powershell
ctest --test-dir build -C Debug --output-on-failure
ctest --test-dir build -C Release --output-on-failure
```

期望结果：

```text
14/14 enabled tests passed
qwen3tts_customvoice_golden disabled
```

本次验证结果：

```text
CPU Debug     14/14 passed，34.17s
CPU Release   14/14 passed，21.43s
```

CUDA 使用 0.6B、16 帧 CLI 冒烟测试：

```powershell
.\build\bin\Release\qwen3tts-cli.exe `
  --model .\testdata\qwen3-tts-talker-0.6b-customvoice-bf16.gguf `
  --speech-tokenizer .\testdata\qwen3-tts-speech-tokenizer-12hz-f32.gguf `
  --text "今天，中国已经成为亚太贸易自由化和本地区发达供应链的重要基石。" `
  --speaker serena `
  --language Chinese `
  --max-new-tokens 2048 `
  --seed 1234 `
  --greedy `
  --device CUDA0 `
  --threads 8 `
  --output .\testdata\cuda-release.wav
```

验收标准是识别 `CUDA0`、生成 16 帧并写出 24 kHz WAV。CUDA Debug 已在本机通过；
Debug 和 Release 分别使用 `.\build\bin\Debug\qwen3tts-cli.exe` 与
`.\build\bin\Release\qwen3tts-cli.exe`。

CPU 与 CUDA 的 greedy code 可能因浮点累加顺序不同而分叉，不宜直接用 CPU golden
结果判定 CUDA 推理失败。

## HF 转 GGUF

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt

python .\convert_hf_to_gguf.py `
  --input .\vendor\qwen3-tts\models\Qwen3-TTS-12Hz-0.6B-CustomVoice `
  --output-dir .\testdata
```

转换器保留源 tensor dtype，不提供 `--dtype` 参数。

## Talker Q8_0 量化

```powershell
.\.venv\Scripts\python.exe .\quantize_gguf.py `
  --input .\testdata\qwen3-tts-talker-0.6b-customvoice-bf16.gguf `
  --output .\testdata\qwen3-tts-talker-0.6b-customvoice-q8_0.gguf `
  --type q8_0
```

当前只量化 Talker 的大型二维权重，norm、bias 和敏感参数保持源 dtype。
SpeechTokenizer/SpeechDecoder 不支持量化，必须继续使用原始 F32 GGUF。

0.6B 和 1.7B Talker 均已验证：

```text
0.6B BF16: 1,816,930,272 bytes
0.6B Q8_0:   967,829,600 bytes

1.7B BF16: 3,838,705,792 bytes
1.7B Q8_0: 2,041,954,560 bytes
```

量化 Talker 可直接作为现有 C++ CLI 的 `--model`，无需修改运行时代码。
Q8_0 仍是有损量化。

Talker KV cache 默认按 2048 帧分配，可通过 GGUF KV
`qwen3_tts.talker.num_ctx` 覆盖。Prompt 与生成长度之和不能超过该值。
