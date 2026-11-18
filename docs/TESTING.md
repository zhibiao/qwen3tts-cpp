# Testing

默认 `windows-*` 和 `linux-*` preset 设置 `BUILD_TESTING=OFF`，只构建应用。
需要测试时使用对应的 `*-tests` preset。

## 测试命令

Windows CPU：

```powershell
cmake --preset windows-cpu-tests
cmake --build --preset windows-cpu-tests --parallel
ctest --preset windows-cpu
```

Windows CUDA：

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
cmake --preset windows-cuda-tests
cmake --build --preset windows-cuda-tests --parallel
ctest --preset windows-cuda
```

Linux CPU：

```bash
cmake --preset linux-cpu-tests
cmake --build --preset linux-cpu-tests --parallel
ctest --preset linux-cpu
```

Linux CUDA：

```bash
cmake --preset linux-cuda-tests \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build --preset linux-cuda-tests --parallel
ctest --preset linux-cuda
```

## 覆盖范围

- GGUF metadata/tensor 读取和配置校验
- TextTokenizer：UTF-8、Unicode category、NFC、Qwen byte-level BPE、added tokens
- Talker：prefill、decode、KV cache、Flash Attention
- CodePredictor：固定图、打包 embedding/LM head、greedy argmax
- SpeechDecoder：RVQ、Transformer、上采样和 WAV 输出
- sampler、transposed convolution、模型生命周期、转换器和量化脚本

Golden 测试包括：

- Talker/CodePredictor 与 PyTorch 的 hidden/logits 数值对照
- CustomVoice prompt 与固定 codes 的 SpeechDecoder 音频
- 0.6B/1.7B 文本到 codes/audio 端到端结果
- 无 CUDA 环境下可跳过的 CUDA golden

CLI 场景覆盖短中文、中文标点、英文、中英混合、长混合文本和 1.7B 短句。
这些场景使用同一批 PyTorch codes/audio golden，并校验 WAV RIFF header、
采样率和帧对齐。

## RTF 基准

测试设备：

```text
GPU: NVIDIA GeForce RTX 2060 6 GB
CPU: AMD Ryzen 7 4800H
build: Windows windows-cuda Release
```

固定参数：

```text
text = 今天，中国已经成为亚太贸易自由化和本地区发达供应链的重要基石。
greedy
max-new-tokens = 16
seed = 1234
threads = 8
device = CUDA0
```

16 帧对应 1.28 s 音频。每项耗时取 3 次运行的中位数。

```text
RTF = (码流生成 + SpeechDecoder) / 音频时长
```

模型加载和 WAV 写入不计入 RTF。

| 模型 | 格式 | 码流生成 | Decoder | RTF |
|---|---:|---:|---:|---:|
| 0.6B | BF16 | 0.57s | 0.08s | 0.508 |
| 0.6B | Q8_0 | 0.37s | 0.11s | 0.375 |
| 1.7B | BF16 | 0.78s | 0.08s | 0.672 |
| 1.7B | Q8_0 | 0.45s | 0.11s | 0.438 |

Q8_0 相比 BF16 的 RTF 降低约 26%–35%，但 Q8_0 是有损量化。相同输入下生成的
greedy code 可能分叉，因此还需要通过音频质量和 ASR 可懂度验收。
