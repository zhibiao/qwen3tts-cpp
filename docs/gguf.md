# GGUF v3 与模型转换

## 双文件布局

| 文件 | 内容 |
|---|---|
| Talker GGUF | TextTokenizer、Talker、CodePredictor、speaker/language、generation 配置 |
| SpeechTokenizer GGUF | SpeechTokenizer Decoder、RVQ、ConvNeXt、DAC |

当前运行文件：

```text
testdata/qwen3-tts-talker-0.6b-customvoice-bf16.gguf
testdata/qwen3-tts-talker-1.7b-customvoice-bf16.gguf
testdata/qwen3-tts-speech-tokenizer-12hz-f32.gguf
```

## Python 环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
```

转换使用 PyTorch、safetensors 和 `gguf-py`。C++ runtime 不读取 safetensors，
也不依赖 PyTorch。

## HF 转 GGUF

```powershell
python .\convert_hf_to_gguf.py `
  --input .\vendor\qwen3-tts\models\Qwen3-TTS-12Hz-0.6B-CustomVoice `
  --output-dir .\testdata
```

转换器保留源 tensor dtype，不再支持 `--dtype`。

## 配置来源

模型维度、层数、tokenizer、generation、speaker 和 language 都从 GGUF KV
读取，不在 C++ 中按 0.6B/1.7B 写死。

speaker 元数据：

```text
qwen3_tts.speaker.names
qwen3_tts.speaker.ids
qwen3_tts.speaker.dialects
```

## Encoder 状态

SpeechTokenizer Encoder tensor 可以存在于 GGUF，但当前 C++ runtime 只实现
Decoder。

## 量化

当前提供 Python `quantize_gguf.py`，第一阶段只支持 Talker Q8_0：

```text
convert_hf_to_gguf.py : HF -> 源 dtype GGUF
quantize_gguf.py      : Talker GGUF -> Q8_0 GGUF
```

量化规则：

- 只接受 `qwen3_tts.file_role=talker`
- 二维、完整 block 对齐的 `.weight` 转为 Q8_0
- norm、bias、scale 和其余敏感 tensor 保持源 dtype
- 完整复制 GGUF KV metadata
- SpeechTokenizer/SpeechDecoder 永久保持原 F32 文件，不支持量化
- C++ 加载器会拒绝任何非 F32 的 `sdec.*` tensor

```powershell
.\.venv\Scripts\python.exe .\quantize_gguf.py `
  --input .\testdata\qwen3-tts-talker-0.6b-customvoice-bf16.gguf `
  --output .\testdata\qwen3-tts-talker-0.6b-customvoice-q8_0.gguf `
  --type q8_0
```

C++ runtime 直接使用 `gguf_get_tensor_type()` 创建权重描述符，因此 Q8_0 Talker
无需修改加载代码。运行时仍搭配原始 SpeechTokenizer GGUF：

```powershell
.\build\bin\Release\qwen3tts-cli.exe `
  --model .\testdata\qwen3-tts-talker-0.6b-customvoice-q8_0.gguf `
  --speech-tokenizer .\testdata\qwen3-tts-speech-tokenizer-12hz-f32.gguf `
  --text "你好，世界。" `
  --speaker serena `
  --language Chinese `
  --device CUDA0 `
  --output .\testdata\qwen3tts-q8.wav
```

Q8_0 仍是有损量化，不保证生成的 speech code 与 BF16/F32 文件完全相同。
SpeechDecoder 的最终音频精度由原始 F32 SpeechTokenizer GGUF 保证。

已验证文件规模：

```text
0.6B Talker BF16: 1,816,930,272 bytes
0.6B Talker Q8_0:   967,829,600 bytes

1.7B Talker BF16: 3,838,705,792 bytes
1.7B Talker Q8_0: 2,041,954,560 bytes
```

0.6B 输出包含 266 个 Q8_0 tensor 和 136 个保留源 dtype tensor；1.7B 输出包含
267 个 Q8_0 tensor 和 137 个保留源 dtype tensor。
