# 架构与推理流程

## 当前范围

当前实现支持 Qwen3-TTS 12Hz CustomVoice 0.6B 和 1.7B。Base、Voice Clone、
VoiceDesign 和 instruct 尚未进入运行时实现。

## 运行时对象

`Qwen3TTSContext` 持有：

- Talker GGUF
- SpeechTokenizer GGUF
- GGML backend 和 CPU fallback
- scheduler 工厂

`Qwen3TTSModel` 持有：

- 模型配置
- TextTokenizer
- Talker
- CodePredictor
- SpeechTokenizer
- speaker/language 映射
- sampler 和生成状态

Context 必须先于 Model 声明，并保证生命周期更长。

```cpp
Qwen3TTSContext context;
Status status = context.load(talker_path, speech_path, params);

Qwen3TTSModel model;
status = model.load(context);
```

Talker 和 CodePredictor 分别持有 step 与 embedding scheduler；SpeechDecoder
持有 1 个 scheduler。三者共用 Context 中的 backend。

## 完整请求流程

```text
文本
  -> TextTokenizer
  -> CustomVoice Prompt embeddings
  -> Talker prefill
  -> 每帧 Talker decode
  -> CodePredictor 预测 c1..c15
  -> 16-code 音频帧
  -> SpeechTokenizer Decoder
  -> 24 kHz PCM16 WAV
```

一个 12Hz 音频帧为：

```text
[c0, c1, ..., c15]
```

`c0` 由 Talker 生成，`c1..c15` 由 CodePredictor 生成。

## CustomVoice Speaker

CustomVoice 推理不读取参考音频：

```text
speaker name -> speaker ID -> codec_embedding.weight[speaker ID]
```

该向量是模型权重中的 speaker embedding 行，不是 codec frame。

0.6B 和 1.7B 当前 speaker 映射相同：

```text
aiden      2861
dylan      2878
eric       2875
ono_anna   2873
ryan       3061
serena     3066
sohee      2864
uncle_fu   3010
vivian     3065
```

Talker hidden 和 codec/speaker embedding 维度：

```text
0.6B: 1024
1.7B: 2048
```

## CustomVoice Prompt

Talker 每一行输入是文本侧和 codec 侧 embedding 的逐元素和：

```text
Talker input row = text embedding + codec embedding
```

有明确 language 时，Prompt 顺序为：

```text
role embeddings
tts_pad + codec_think
tts_pad + codec_think_bos
tts_pad + language_id
tts_pad + codec_think_eos
tts_pad + speaker_embedding
tts_bos + codec_pad
文本 token embeddings + codec_pad
tts_eos + codec_pad
tts_pad + codec_bos
```

`language=Auto` 时使用 `codec_nothink`，控制区少一行。

Prompt 构造当前使用三张图：

1. 一次批量文本 embedding + projection
2. 一次批量 codec/control/speaker lookup
3. 一次 Talker prefill

文本 embedding 原始维度是 2048：

- 0.6B projection 后为 1024
- 1.7B projection 后为 2048

codec、language、speaker 和特殊控制 token 不需要 text projection。

## Talker

Talker 是 decoder-only Transformer，每帧生成 `c0`。

KV cache：

```text
F16
[head_dim, num_ctx, kv_heads, layers]
```

`num_ctx` 默认 2048，可通过 `qwen3_tts.talker.num_ctx` 覆盖。
模型 RoPE 的 `context_length` 保持 GGUF 原始值 32768。

Attention：

- 使用 `ggml_flash_attn_ext`
- Prefill 使用当前 F32 K/V 和 causal mask
- Decode 使用 `ggml_set_rows` 写入当前 K/V
- Decode 无 mask，只读取有效 cache 前缀
- KV cache 生命周期覆盖整次语音生成

Prefill 主要形状：

```text
input:  [hidden, prompt_frames]
logits: [3072, prompt_frames]
```

只保留最后一帧 hidden 和 logits 进入自回归生成。

## CodePredictor

CodePredictor 是单帧内部的小型 decoder Transformer。

KV cache：

```text
F16
[head_dim, num_code_groups, kv_heads, layers]
num_code_groups = 16
```

每帧开始时清空 cache。第一步输入 Talker hidden 和 `c0` embedding，写入两个
cache 位置并预测 `c1`；之后执行 14 个单 token decode 步骤。

Attention 同样使用 `ggml_flash_attn_ext`：

- bootstrap 使用 causal mask
- decode 使用固定 16 位 KV view，由 mask 屏蔽未使用位置

15 个 residual code embedding 打包为一个二维查表权重，15 个 lm_head
打包为一个三维权重。decode 通过动态组合行号和 `MUL_MAT_ID` 选择当前权重。

固定形状 decode graph 在首次使用时完成分配并缓存；后续只更新 code、
embedding row、position、mask 和 head。greedy 路径在图内执行 `ARGMAX`。

## 帧 embedding 反馈

完整帧 embedding 是 16 个 code embedding 的和：

```text
talker.codec_embedding[c0]
+ code_predictor.codec_embedding.0[c1]
+ ...
+ code_predictor.codec_embedding.14[c15]
```

非流式 CustomVoice 随后加入 `tts_pad`：

```text
next_embedding = frame_embedding + tts_pad
```

该向量作为下一次 Talker decode 的输入。

## SpeechTokenizer

C++ 当前仅实现 Decoder：

```text
codes: [frames, 16]
audio: [frames * 1920, 1]
```

输出为 24000 Hz。GGUF 可以包含 Encoder tensor，但运行时尚未加载。

## 数值兼容性

Talker 和 CodePredictor 都使用 online-softmax flash attention。短 golden
序列和最终语音内容正确，但长 greedy 生成不再保证与旧手工 attention 路径
bitwise 相同。

## 当前调度策略

- Talker：step 和 embedding 各 1 个 scheduler
- CodePredictor：prefill、缓存 decode 和 embedding 各使用独立 scheduler
- SpeechDecoder：1 个 scheduler

CodePredictor embedding graph 不再覆盖 step/bootstrap graph 的 allocator plan。
固定 decode graph 避免在每个 residual group 重建 allocator plan；Debug 构建
仍会直接输出 ggml allocator 日志。

## Backend 状态

- CPU backend 已完成完整验证。
- CUDA 13.5 / RTX 2060 已完成 0.6B 和 1.7B 实验验证。
- CUDA 与 CPU 的贪心 code 不保证 bitwise 相同。
