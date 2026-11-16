# 性能分析与优化计划

## 当前基线

当前统一使用 `build`，Release 配置使用：

```text
/O2 /Ob2 /DNDEBUG /MD
```

## 最新 CUDA 优化结果

RTX 2060、16 帧、8 threads、Release、固定输入：

```text
                        优化前      当前
0.6B BF16 总生成时间     1.13s      0.70-0.79s
0.6B CodePredictor      51.80ms     28.05-33.01ms
0.6B Talker decode      12.86ms     8.79-10.68ms
0.6B Q8_0 总生成时间                 0.69s

1.7B BF16 总生成时间     1.30s      0.92s
1.7B CodePredictor      52.64ms     29.85ms
1.7B Talker decode      18.81ms     17.07ms
```

已完成的关键优化：

- Talker/CodePredictor 使用 `ggml_swiglu_split` 融合 MatMul、SiLU 和逐元素乘法
- RMSNorm/QNorm/KNorm 使用无损 F32 runtime 权重，删除临时 cast/reshape/repeat 节点
- CodePredictor KV view 固定为 16 位，未使用位置通过 mask 屏蔽
- residual code embedding 和 LM Head 打包，动态使用组合行号和 `MUL_MAT_ID`
- CodePredictor greedy 路径在图内执行 argmax
- 固定形状 decode graph 和 scheduler plan 持久复用

下一步可直接在稳定图基础上评估 CUDA Graphs，以及打包 QKV 后的一次矩阵乘。

0.6B、16 帧、CPU、8 threads、Release 的阶段耗时：

```text
模型加载        2.23s
码流生成        1.99s   // 约 8.0 frames/s
SpeechDecoder   1.36s
WAV 写入        0.00s
```

内部计时结果：

```text
Prompt                 3.2 ms
Talker prefill       184.3 ms
Talker c0 sampling     0.1 ms
CodePredictor       1376.8 ms
Frame embedding        2.0 ms
Talker decode        423.1 ms
Condition add          0.0 ms
```

Code generation 中 CodePredictor 约占 69%，Talker decode 约占 21%，
Talker prefill 约占 9%。12Hz 实时需要 12 frames/s，当前生成约
8 frames/s。

## 主要耗时

### 1. CodePredictor 是最大瓶颈

16 帧中约 1.38 秒用于 CodePredictor。每帧包含一次 bootstrap 和 14 次
串行 decode，是当前最主要的生成耗时。

### 2. Talker decode 是第二瓶颈

每个 12Hz 帧执行：

```text
Talker decode          1
CodePredictor prefill  1
CodePredictor decode  14
CodePredictor embed    1
```

Talker decode 约 0.42 秒，占 12Hz frame 生成约 21%。CodePredictor 的
采样依赖也使整个 15 步过程无法直接变成一个静态图。

CLI 现在逐帧打印 bootstrap 和 14 个 residual group 的 graph/sampling 耗时，
可用于定位 CodePredictor 的 graph 分配和采样成本。

### 3. Scheduler 图布局切换

CodePredictor 的 step 和 embedding 已使用独立 scheduler。embedding graph 只在
首次执行时建立 allocator plan，不再每帧覆盖 bootstrap/decode 的 step plan。
CodePredictor KV cache 已统一为 F16，与 Talker 一致。Flash Attention 不再需要
随 K/V 历史长度增长的 F16 临时 workspace，`node_33 invalid` 已消失。

Talker 的 text/codec embedding 也使用独立 scheduler，与 prefill/decode plan
隔离。Talker embedding 本来只在请求初始化时执行，因此该拆分不会减少常规
reserve 次数，只避免 embedding plan 覆盖后续 prefill/decode。

当前 2 帧 Debug 实测从 `36 次 reserve / 26 次 node invalid` 降为
`10 次 reserve / 0 次 node invalid`。剩余 reserve 来自不同 graph 类型的首次
建立和切换，而不是 KV 历史长度增长。

### 4. Prompt 和 frame embedding 已不是主要瓶颈

Release 计时中 Prompt 约 3.2 ms，frame embedding 总计约 2 ms。
继续优先优化这两项收益有限。

### 5. Backend/CPU 同步

每个 CodePredictor step 都复制完整 logits 到 CPU。128 帧约产生：

```text
128 × 15 = 1920 次 logits 同步
```

### 6. SpeechDecoder

原始 CUDA SpeechDecoder 使用 F32 权重，并通过逐元素
`ggml_conv_transpose_1d` kernel 执行上采样。32 帧实测中 SpeechDecoder
约 `4.09s`，慢于 CPU 的 `2.43s`。

按需诊断显示：

```powershell
$env:QWEN3TTS_PROFILE_SPEECH = "1"
```

原始 CUDA 路径只有 1 个 scheduler split，没有 CPU fallback；约 99.7% 时间位于
compute，graph build、分配和输出回读合计不到 2 ms。因此根因是卷积 kernel，
不是跨设备拷贝或图分配。

已实现的 CUDA SpeechDecoder 优化：

- 权重在 GPU 上严格保持 GGUF/PyTorch 原始 dtype，不做 F16/BF16 降精度
- `kernel_size=1` 卷积直接转换为矩阵乘，绕过 im2col
- transposed convolution 按 stride phase 分解为矩阵乘，权重只重排布局，位级数值不变
- SpeechDecoder attention 使用 `ggml_flash_attn_ext`
- 新增 CPU 数学回归测试，逐元素对比原始 transposed convolution

32 帧优化结果：

```text
CUDA SpeechDecoder  4.09s -> 2.90-3.07s // 约快 27%
CPU SpeechDecoder   2.43s -> 2.33s
```

CUDA Decoder 目前约比 CPU 慢 28%，但相比原始 68% 已明显缩小。

SpeechDecoder Flash Attention 在 32 帧下速度基本持平，但统一使用了 ggml 的
优化 attention 算子。F16/BF16 权重转换不符合严格数值规则，已回退。

## 优化顺序

### P0：保持 Release 基准

```powershell
cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -T host=x64 `
  -DBUILD_TESTING=ON `
  -DQWEN3TTS_ENABLE_CUDA=OFF
cmake --build build --config Release --parallel
```

运行可执行文件为 `build\bin\Release\qwen3tts-cli.exe`。后续优化都应与当前 Release
数据比较。

### P1：优化 CodePredictor 主体执行

已完成固定 decode 图、动态 LM Head、打包 code embedding 和 graph 复用。
继续优先评估 QKV 打包与 CUDA Graphs。

可选方向：

- 复用 graph metadata 和固定输入 buffer
- 分离 bootstrap/decode scheduler，并固定 decode KV view
- 评估 CUDA/专用 CPU kernel
- 将 greedy argmax 放到 backend

### P2：减少采样同步

- greedy：已在 backend 图内做 argmax，只复制一个整数
- sampling：实现设备侧 top-k/top-p，或复用 logits buffer

### P3：CodePredictor embedding 与 Talker decode 融合

该优化可去掉每帧一次 embedding 图和主机往返，但当前 CPU 计时下只有
约 2 ms/16 frames，收益低于 CodePredictor 主体计算。

### P4：Prompt + Talker prefill 合并成一张图

把文本 projection、codec/speaker lookup 和 Prompt 组装移入 Talker prefill
图，去掉 embedding 的 CPU 往返。请求从 3 张图降为 1 张图。

### P5：SpeechDecoder

- CPU 路径测试 4/6/8/12/16 threads
- 评估分块或流式 decode
- 评估 SpeechDecoder 与码流生成并行，隐藏 decode 延迟
- 继续 profiler 定位剩余 F32 非卷积操作

### P6：量化 Talker GGUF

Talker Q8_0 已通过 `quantize_gguf.py` 验证。SpeechTokenizer/SpeechDecoder
不量化，固定使用原始 F32，以保证最终音频精度。

Talker KV cache 默认从 32768 缩短到 2048，F16 cache 从约 3.5 GiB 降到
224 MiB。93 帧测试峰值显存从约 5820 MiB 降到 2758 MiB。

### P7：动态 CPU variants

当前静态 CPU backend 面向本机 AVX2。若要分发给不同 CPU：

```cmake
BUILD_SHARED_LIBS=ON
GGML_BACKEND_DL=ON
GGML_CPU_ALL_VARIANTS=ON
```

可生成 SSE/AVX2/AVX512 等多个 CPU backend DLL，并在启动时选择最佳兼容版本。

### P8：继续优化 CUDA

后续 CUDA 工作：

- 0.6B/1.7B 权重显存
- RTX 2060 6 GB 显存
- 每步 logits 同步成本
- SpeechDecoder 流式执行与 CUDA graph 复用

## CUDA SpeechDecoder 优化结果

0.6B、32 帧、RTX 2060、CUDA0、Release、8 threads：

```text
优化前 SpeechDecoder   4.09s
优化后 SpeechDecoder   2.90-3.07s
```

CPU Release 对照：

```text
SpeechDecoder   2.33s
```

CUDA 生成阶段仍明显快于 CPU；SpeechDecoder 已接近 CPU 速度。下一阶段的主要
瓶颈仍是 CodePredictor 的逐帧串行生成。

## 基准测试规范

固定输入：

```powershell
--text "你好，世界。" `
--speaker serena `
--language Chinese `
--max-new-tokens 16 `
--seed 1234 `
--greedy `
--device CPU
```

记录：

- 加载耗时
- frame 生成速度
- SpeechDecoder 耗时
- 峰值 CPU 内存
- Debug 构建下的 realloc 次数

不要用 Debug 基线对比 Release 结果。
