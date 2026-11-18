# Qwen3-TTS.cpp

基于 C++17 和 ggml 的 Qwen3-TTS 12 Hz 推理实现。Python/PyTorch 参考源码位于
[`docs/qwen3-tts`](docs/qwen3-tts)，C++ 运行时只读取 GGUF，不依赖 PyTorch。

## 当前支持

- Qwen3-TTS 12 Hz CustomVoice 0.6B / 1.7B
- speaker、language 和 speaker dialect
- TextTokenizer -> Talker -> CodePredictor -> SpeechDecoder 完整链路
- TextTokenizer 使用 uni-algo 处理 UTF-8、Unicode 属性和可选 NFC，保留原生 Qwen byte-level BPE
- 24 kHz PCM16 WAV 输出
- CPU 和 CUDA backend
- Windows x64 与 Linux x86_64；默认 CPU 构建，CUDA 显式开启
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
| `src/text_tokenizer*.{h,cpp}` | uni-algo Unicode/NFC、原生 Qwen byte-level BPE 和特殊 prompt |
| `src/qwen3_tts.*` | 运行时上下文、模型生命周期、CustomVoice prompt 和生成循环 |
| `src/gguf_model.*` | GGUF metadata/tensor 读取 |
| `app/qwen3tts_cli.cpp` | 命令行加载、生成、解码 WAV |
| `tests/` | 单元测试、PyTorch golden 和 fixture 导出脚本 |

`Qwen3TTSContext` 必须比 `Qwen3TTSModel` 生命周期更长。前者持有 GGML backend、
CPU fallback 和两个 GGUF，后者持有 tokenizer、Talker、CodePredictor 和 Decoder。

## 模型文件

运行时需要两个 GGUF：

```text
models/qwen3-tts-talker-0.6b-customvoice-bf16.gguf
models/qwen3-tts-talker-1.7b-customvoice-bf16.gguf
models/qwen3-tts-talker-0.6b-customvoice-q8_0.gguf
models/qwen3-tts-talker-1.7b-customvoice-q8_0.gguf
models/qwen3-tts-speech-tokenizer-12hz-f32.gguf
```

Talker GGUF 包含 TextTokenizer、Talker、CodePredictor、speaker/language 和生成
配置；SpeechTokenizer GGUF 包含 Decoder、RVQ、ConvNeXt 和 DAC。

所有可执行文件和 GGML backend 共享库输出到同一配置目录：

```text
build/windows-cpu/bin/Release/
build/windows-cuda/bin/Release/
build/linux-cpu/bin/
build/linux-cuda/bin/
```

Windows 下 `ggml*.dll` 与 `qwen3tts-cli.exe` 放在同一目录；Linux 构建使用
`$ORIGIN` RPATH，同目录下的 `libggml*.so` 会自动被找到。

## 快速运行

先构建 Release，然后执行：

```powershell
.\build\windows-cuda\bin\Release\qwen3tts-cli.exe `
  --model .\models\qwen3-tts-talker-0.6b-customvoice-bf16.gguf `
  --speech-tokenizer .\models\qwen3-tts-speech-tokenizer-12hz-f32.gguf `
  --text "今年是中国第三次担任APEC东道主。中方在全国16座城市累计举办300多场会议和活动。" `
  --speaker serena `
  --language Chinese `
  --max-new-tokens 512 `
  --seed 1234 `
  --greedy `
  --device CUDA0 `
  --threads 8 `
  --output .\output.wav
```

Linux：

```bash
./build/linux-cuda/bin/qwen3tts-cli \
  --model models/qwen3-tts-talker-0.6b-customvoice-bf16.gguf \
  --speech-tokenizer models/qwen3-tts-speech-tokenizer-12hz-f32.gguf \
  --text "今年是中国第三次担任APEC东道主。" \
  --speaker serena \
  --language Chinese \
  --max-new-tokens 512 \
  --seed 1234 \
  --greedy \
  --device CUDA0 \
  --threads 8 \
  --output output.wav
```

列出 backend 设备或模型支持的 speaker：

```powershell
.\build\windows-cuda\bin\Release\qwen3tts-cli.exe --list-devices
.\build\windows-cuda\bin\Release\qwen3tts-cli.exe `
  --model .\models\qwen3-tts-talker-0.6b-customvoice-q8_0.gguf `
  --list-speakers
```

Linux：

```bash
./build/linux-cuda/bin/qwen3tts-cli --list-devices
./build/linux-cuda/bin/qwen3tts-cli \
  --model models/qwen3-tts-talker-0.6b-customvoice-q8_0.gguf \
  --list-speakers
```

常用参数：`--device CPU|CUDA0`、`--greedy`、`--talker-greedy`、
`--subtalker-greedy`、`--repetition-penalty`、`--codes-output`。

## 构建

需要 CMake 3.21+、C++17 编译器。CPU 是默认 backend，CUDA 需显式开启。
完整跨平台说明见 [docs/BUILDING.md](docs/BUILDING.md)。

Windows CPU：

```powershell
cmake --preset windows-cpu
cmake --build --preset windows-cpu --parallel
```

Windows CUDA：

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
cmake --preset windows-cuda
cmake --build --preset windows-cuda --parallel
```

Linux CPU：

```bash
cmake --preset linux-cpu
cmake --build --preset linux-cpu --parallel
```

Linux CUDA：

```bash
cmake --preset linux-cuda \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build --preset linux-cuda --parallel
```

`windows-cuda` / `linux-cuda` 使用本机 `native` 架构；需要兼容多代 GPU 时使用
`windows-cuda-portable` 或 `linux-cuda-portable`。

运行 CUDA 版本前，确保 CUDA runtime 可被动态链接器找到。CUDA 与 CPU 的
greedy code 可能因浮点累加顺序不同而产生分歧，不应直接用 CPU golden 判定
CUDA 推理失败。

模型和测试 fixture 默认从 `models/`、`testdata/` 读取，可通过
`QWEN3TTS_MODEL_DIR` 和 `QWEN3TTS_TEST_DATA_DIR` 覆盖。

## HF 转 GGUF

转换脚本需要 Python、PyTorch、safetensors 和 `gguf-py`：

Windows：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt

python .\convert_hf_to_gguf.py `
  --input .\docs\qwen3-tts\models\Qwen3-TTS-12Hz-0.6B-CustomVoice `
  --output-dir .\models
```

Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python convert_hf_to_gguf.py \
  --input docs/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --output-dir models
```

转换器保留源 tensor dtype。默认把 SpeechTokenizer Encoder tensor 一并写入 GGUF，
但当前 C++ 运行时只加载 Decoder。

Talker 权重转换支持 `q8_0` 和 `bf16`：

```powershell
.\.venv\Scripts\python.exe .\quantize_gguf.py `
  --input .\models\qwen3-tts-talker-0.6b-customvoice-bf16.gguf `
  --output .\models\qwen3-tts-talker-0.6b-customvoice-q8_0.gguf `
  --type q8_0
```

Linux：

```bash
.venv/bin/python quantize_gguf.py \
  --input models/qwen3-tts-talker-0.6b-customvoice-bf16.gguf \
  --output models/qwen3-tts-talker-0.6b-customvoice-q8_0.gguf \
  --type q8_0
```

转换只作用于 Talker 的大型二维权重，norm、bias 和敏感参数保持源 dtype。
`--type` 可选 `q8_0`、`bf16`。SpeechTokenizer/SpeechDecoder 必须继续使用原始
F32 GGUF。`q8_0` 和 `bf16` 都可能改变数值，不保证生成 code 与原 BF16/F32
文件完全相同。

## 测试

测试使用独立的 `*-tests` preset，默认构建不会编译测试：

```text
cmake --preset windows-cpu-tests
cmake --build --preset windows-cpu-tests --parallel
ctest --preset windows-cpu

cmake --preset windows-cuda-tests
cmake --build --preset windows-cuda-tests --parallel
ctest --preset windows-cuda

cmake --preset linux-cpu-tests
cmake --build --preset linux-cpu-tests --parallel
ctest --preset linux-cpu

cmake --preset linux-cuda-tests
cmake --build --preset linux-cuda-tests --parallel
ctest --preset linux-cuda
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

0.6B、16 帧、8 threads 的 RTX 2060 CUDA 参考数据为：码流生成 0.57 s、
SpeechDecoder 0.08 s。性能会随模型、硬件和输入变化。

Talker KV cache 默认 `num_ctx=2048`，可通过 GGUF
`qwen3_tts.talker.num_ctx` 覆盖。prompt 与生成帧数之和不能超过该值。

Debug 构建较慢，并会输出更多 ggml allocator 日志；性能测试应使用对应 preset
的 Release 产物，例如 `build/windows-cuda/bin/Release` 或
`build/linux-cuda/bin`。

## RTF 基准

测试设备为 RTX 2060 6 GB、Ryzen 7 4800H、Windows `windows-cuda` Release。
每项耗时取 3 次运行的中位数。
固定参数：

```text
text = 今天，中国已经成为亚太贸易自由化和本地区发达供应链的重要基石。
greedy
max-new-tokens = 16
seed = 1234
threads = 8
device = CUDA0
```

本次 16 帧对应 1.28 s 音频。

RTF =
`(码流生成 + SpeechDecoder) / 音频时长`，不含模型加载和 WAV 写入。

| 模型 | 格式 | 码流生成 | Decoder | RTF |
|---|---:|---:|---:|---:|
| 0.6B | BF16 | 0.57s | 0.08s | 0.508 |
| 0.6B | Q8_0 | 0.37s | 0.11s | 0.375 |
| 1.7B | BF16 | 0.78s | 0.08s | 0.672 |
| 1.7B | Q8_0 | 0.45s | 0.11s | 0.438 |

Q8_0 相比 BF16 的 RTF 降低约 26%–35%，但属于有损量化。相同输入下生成的
greedy code 可能分叉，因此不能仅凭 RTF 选择 Q8_0，还需要音频质量与 ASR
可懂度验收。
