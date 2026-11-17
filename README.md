# Qwen3-TTS.cpp

基于 C++17 和 ggml 的 Qwen3-TTS 12 Hz 推理实现。Python/PyTorch 参考源码位于
[`docs/qwen3-tts`](docs/qwen3-tts)，C++ 运行时只读取 GGUF，不依赖 PyTorch。

## 当前支持

- Qwen3-TTS 12 Hz CustomVoice 0.6B / 1.7B
- speaker、language 和 speaker dialect
- TextTokenizer -> Talker -> CodePredictor -> SpeechDecoder 完整链路
- 24 kHz PCM16 WAV 输出
- CPU 和 CUDA backend
- Talker `q8_0` / `bf16` 权重转换

当前不支持：

- 非空 `instruct`
- Base / Voice Clone 和 VoiceDesign
- SpeechTokenizer Encoder
- 流式生成 API
- C++ GGUF 量化工具和完整 CUDA 性能调优

## 架构

```text
text
  -> TextTokenizer
  -> CustomVoice prompt embeddings
  -> Talker prefill / decode (c0)
  -> CodePredictor (c1..c15)
  -> 16-code, 12 Hz audio frame
  -> SpeechTokenizer Decoder
  -> 24 kHz PCM16 WAV
```

主要模块：

| 路径 | 作用 |
|---|---|
| `src/talker.*` | Talker Transformer、RoPE、Flash Attention、KV cache |
| `src/code_predictor.*` | 每帧预测 c1..c15，固定 KV view 和 decode graph |
| `src/speech_tokenizer.*` | RVQ、Transformer、上采样和 DAC SpeechDecoder |
| `src/text_tokenizer.*` | byte-level BPE 和 Qwen 特殊 prompt |
| `src/qwen3_tts.*` | 运行时上下文、模型生命周期、CustomVoice prompt 和生成循环 |
| `src/gguf_model.*` | GGUF metadata/tensor 读取 |
| `app/qwen3tts_cli.cpp` | 命令行加载、生成、解码 WAV |
| `tests/` | 单元测试、PyTorch golden 和 fixture 导出脚本 |

`Qwen3TTSContext` 必须比 `Qwen3TTSModel` 生命周期更长。前者持有 GGML backend、
CPU fallback 和两个 GGUF，后者持有 tokenizer、Talker、CodePredictor 和 Decoder。

## 模型文件

运行时需要两个 GGUF：

```text
testdata/qwen3-tts-talker-0.6b-customvoice-bf16.gguf
testdata/qwen3-tts-talker-1.7b-customvoice-bf16.gguf
testdata/qwen3-tts-speech-tokenizer-12hz-f32.gguf
```

Talker GGUF 包含 TextTokenizer、Talker、CodePredictor、speaker/language 和生成
配置；SpeechTokenizer GGUF 包含 Decoder、RVQ、ConvNeXt 和 DAC。

所有可执行文件和 GGML backend DLL 输出到同一配置目录：

```text
build/bin/Debug/
build/bin/Release/
```

共享库构建时，`ggml*.dll` 必须和 `qwen3tts-cli.exe` 放在同一目录。

## 快速运行

先构建 Release，然后执行：

```powershell
.\build\bin\Release\qwen3tts-cli.exe `
  --model .\testdata\qwen3-tts-talker-0.6b-customvoice-bf16.gguf `
  --speech-tokenizer .\testdata\qwen3-tts-speech-tokenizer-12hz-f32.gguf `
  --text "今年是中国第三次担任APEC东道主。中方在全国16座城市累计举办300多场会议和活动。" `
  --speaker serena `
  --language Chinese `
  --max-new-tokens 512 `
  --seed 1234 `
  --greedy `
  --device CUDA0 `
  --threads 8 `
  --output .\testdata\output.wav
```

列出 backend 设备或模型支持的 speaker：

```powershell
.\build\bin\Release\qwen3tts-cli.exe --list-devices
.\build\bin\Release\qwen3tts-cli.exe `
  --model .\testdata\qwen3-tts-talker-0.6b-customvoice-q8_0.gguf `
  --list-speakers
```

常用参数：`--device CPU|CUDA0`、`--greedy`、`--talker-greedy`、
`--subtalker-greedy`、`--repetition-penalty`、`--codes-output`。

## 构建

依赖 CMake 3.20+、Visual Studio C++ x64；CUDA 构建还需要 CUDA Toolkit。

CPU Release：

```powershell
cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -T host=x64 `
  -DBUILD_TESTING=ON `
  -DQWEN3TTS_ENABLE_CUDA=OFF
cmake --build build --config Release --parallel
```

CUDA Release：

```powershell
cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -T host=x64 `
  -DBUILD_TESTING=OFF `
  -DQWEN3TTS_ENABLE_CUDA=ON `
  -DQWEN3TTS_CUDA_ARCHITECTURES=75
cmake --build build --config Release --target qwen3tts-cli --parallel
```

运行 CUDA 版本前，确保 CUDA runtime DLL 位于 `PATH` 或 EXE 旁边。CUDA 与 CPU
的 greedy code 可能因浮点累加顺序不同而产生分歧，不应直接用 CPU golden 判定
CUDA 推理失败。

## HF 转 GGUF

转换脚本需要 Python、PyTorch、safetensors 和 `gguf-py`：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt

python .\convert_hf_to_gguf.py `
  --input .\docs\qwen3-tts\models\Qwen3-TTS-12Hz-0.6B-CustomVoice `
  --output-dir .\testdata
```

转换器保留源 tensor dtype。默认把 SpeechTokenizer Encoder tensor 一并写入 GGUF，
但当前 C++ 运行时只加载 Decoder。

Talker 权重转换支持 `q8_0` 和 `bf16`：

```powershell
.\.venv\Scripts\python.exe .\quantize_gguf.py `
  --input .\testdata\qwen3-tts-talker-0.6b-customvoice-bf16.gguf `
  --output .\testdata\qwen3-tts-talker-0.6b-customvoice-q8_0.gguf `
  --type q8_0
```

转换只作用于 Talker 的大型二维权重，norm、bias 和敏感参数保持源 dtype。
`--type` 可选 `q8_0`、`bf16`。SpeechTokenizer/SpeechDecoder 必须继续使用原始
F32 GGUF。`q8_0` 和 `bf16` 都可能改变数值，不保证生成 code 与原 BF16/F32
文件完全相同。

## 测试

```powershell
ctest --test-dir build -C Debug --output-on-failure
ctest --test-dir build -C Release --output-on-failure
```

测试覆盖 GGUF/config、TextTokenizer、Talker、CodePredictor、SpeechDecoder、
transposed convolution、sampler、WAV writer、模型生命周期、GGUF 转换器和
Talker 量化脚本。Golden 包括 Talker/CodePredictor 数值、CustomVoice prompt、
固定 codes 的 SpeechDecoder 音频、0.6B/1.7B 文本到 codes/audio 端到端结果，
以及可跳过无 CUDA 环境的 CUDA Talker/CodePredictor golden。

CLI 端到端场景覆盖短中文、中文标点、英文、中英混合、长混合文本和 1.7B 短句；
同一批文本另有 PyTorch codes/audio golden，当前 0.6B 五类场景和 1.7B 短混合
句均通过。每个 CLI 场景生成 WAV 后还会校验 RIFF header、采样率和帧对齐。

## 性能与限制

Release 下主要瓶颈是 CodePredictor。每帧包含一次 bootstrap 和 14 个串行 decode
步骤，约占 CPU 生成时间的 69%；Talker decode 约占 21%。已使用融合 SwiGLU、
无损 F32 RMSNorm、固定 KV view、打包 code embedding/LM head、图内 argmax 和
持久 scheduler plan。

0.6B、16 帧、8 threads 的参考数据约为：CPU 码流生成 2.0 s、SpeechDecoder
1.36 s；RTX 2060 CUDA 生成约 0.70-0.79 s。性能会随模型、硬件和输入变化。

Talker KV cache 默认 `num_ctx=2048`，可通过 GGUF
`qwen3_tts.talker.num_ctx` 覆盖。prompt 与生成帧数之和不能超过该值。

Debug 构建较慢，并会输出更多 ggml allocator 日志；性能测试应使用
`build/bin/Release`。
