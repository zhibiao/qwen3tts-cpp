# 后续路线图

## 已完成

- GGUF v3 双文件
- TextTokenizer
- Talker prefill/decode
- CodePredictor
- SpeechTokenizer Decoder
- 0.6B/1.7B CustomVoice
- speaker/language/dialect
- Talker 和 CodePredictor flash attention
- CLI 阶段日志和低频进度
- 13 个启用测试

## 第一优先：性能

1. 评估打包 QKV 与 CUDA Graphs。
2. 降低非 greedy sampling 的 logits 同步。
3. CodePredictor embedding 合入 Talker decode。
4. Prompt embedding 与 Talker prefill 合并。
5. 优化 SpeechDecoder。

## 已完成：CodePredictor 固定图与融合

- MLP SwiGLU 和 RMSNorm 融合模式
- 固定 16 位 KV view 与 mask
- residual embedding/LM Head 打包及 `MUL_MAT_ID`
- greedy 图内 argmax
- 固定 decode graph 和 scheduler plan 复用

## 已完成：Talker Q8_0 量化

```text
quantize_gguf.py
```

已支持 Talker Q8_0，0.6B/1.7B 均可直接加载。SpeechTokenizer/SpeechDecoder
不在量化范围内，必须保持原始 F32，避免影响最终音频精度。

## 已完成：限制 Talker KV cache

Talker cache 默认长度已改成 2048：

```text
qwen3_tts.talker.num_ctx = 2048
```

可通过 GGUF KV 覆盖。Prompt 加生成长度超过 cache 时会明确报错。模型 RoPE
`context_length` 不受影响。

## 第四优先：SpeechTokenizer Encoder

实现 Encoder 加载和 codec 工具：

```text
WAV -> RVQ codes
RVQ codes -> WAV
```

这是 Base voice cloning 的前置条件。

## 第五优先：instruct

当前 `CustomVoiceOptions.instruct` 非空时直接返回未实现。后续需要：

- `TextTokenizer::encode_instruction`
- 文本 projection
- Prompt 前端插入
- 根据模型配置决定 0.6B/1.7B 是否支持

## 第六优先：Base 和 VoiceDesign

Base voice cloning 需要：

- SpeechTokenizer Encoder
- speaker encoder
- reference codes 或 x-vector-only 模式
- ICL Prompt

VoiceDesign 需要独立的高层 Prompt 和模型校验。

## 当前非目标

- 不在 CustomVoice 运行时处理参考音频
- 不提供 SpeechTokenizer Encoder
- 不提供流式生成 API
- 不静默忽略 instruct
- 不增加无明确用途的包装类
