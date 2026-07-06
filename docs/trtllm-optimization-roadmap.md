# TensorRT-LLM 推理加速路线

这份文档记录当前项目接下来应该怎么做推理加速实验。核心目标不是只把
`tok/s` 做高，而是在可复现的文本 workload 上同时观察：

```text
性能有没有变好
质量有没有变差
资源有没有超限
改动是否可解释
```

## 1. 三条主线

推理加速可以拆成三条主线。你提出的“和硬件斗争、和模型能力斗争”
是前两条，也是最核心的两条；工程上还要补第三条。

第一条是和硬件斗争：

```text
让 GPU 更忙
减少显存占用
减少内存带宽压力
减少调度和 kernel 开销
提高 batch / concurrency 的有效利用
```

对应手段：

```text
concurrency sweep
max_batch_size / max_seq_len / max_num_tokens
KV cache fraction
paged KV cache
remove input padding
attention / MLP plugin
量化
speculative decoding
多 GPU 并行
```

第二条是和模型能力斗争：

```text
速度更快以后，回答是否还对
中文是否变差
代码是否还能运行
摘要是否漏信息
是否更容易重复
是否提前停止
是否格式崩坏
```

对应手段：

```text
固定业务测试集
baseline 输出对比
人工 review
自动 benchmark
公开评测集
输出质量 checklist
```

第三条是和工程可控性斗争：

```text
结果能不能复现
engine 能不能稳定重建
配置有没有快照
模型版本有没有固定
质量证据有没有保存
线上问题能不能定位
```

对应手段：

```text
results/runs/<run_id>/
results/quality/<run_id>/
config snapshot
serving env snapshot
raw JSON
quality diff
run index
```

## 2. 质量是不是玄学

不是纯玄学，但确实有实验成分。

性能指标通常比较硬：

```text
TTFT
TPOT
request throughput
output tok/s
p95 / p99 latency
GPU utilization
memory used
```

质量指标更依赖场景：

```text
准确率
代码 pass/fail
格式是否正确
摘要覆盖率
事实错误
重复率
中文自然度
指令遵循
```

所以质量不能只靠一个数字。更现实的做法是：

```text
公开 benchmark 给底线
业务 case 给真实判断
人工 review 给语义把关
```

激进加速本质上是 trade-off 搜索，不是一次调参就结束。每次只改一个变量，
把性能和质量一起记录下来。

## 3. 当前必须先做的事

### 3.1 生成固定文本 workload

先不要继续用 `smoke.jsonl` 做结论。它只有 4 条，只适合检查流程。

生成 100 条混合测试：

```bash
python3 scripts/make_benchmark_dataset.py --scenario mixed --count 100
```

再分别生成三类专项测试：

```bash
python3 scripts/make_benchmark_dataset.py --scenario short_chat --count 100
python3 scripts/make_benchmark_dataset.py --scenario long_output --count 100
python3 scripts/make_benchmark_dataset.py --scenario long_context --count 100
```

这些文件会放在：

```text
configs/benchmarks/generated/
```

### 3.2 跑 baseline

先不要改 engine 参数，跑当前 BF16 TensorRT-LLM engine：

```bash
TRTLLM_BENCH_DATASET=/workspace/configs/benchmarks/generated/mixed_100.jsonl \
  TRTLLM_PROFILE_REQUESTS=100 \
  TRTLLM_PROFILE_WARMUP=10 \
  TRTLLM_PROFILE_CONCURRENCIES="1 2 4 8" \
  bash scripts/profile_trtllm_loop.sh
```

每轮结果入口：

```text
results/runs/<run_id>/README.md
```

总索引：

```text
results/runs/index.csv
```

### 3.3 看报告里的核心指标

优先看：

```text
avg_input_length
avg_output_length
avg_concurrency
avg_request_latency_ms
avg_ttft_ms
avg_tpot_ms
request_throughput_req_s
system_output_throughput_tok_s
request_latency_p95_ms
request_latency_p99_ms
memory_used_mib
```

判断方式：

```text
TTFT 高
  -> prefill / 长输入 / 调度成本可能是瓶颈

TPOT 高
  -> decode / 权重读取 / KV cache / 内存带宽可能是瓶颈

吞吐随并发上升
  -> GPU 之前没吃满，并发或 batching 有收益

吞吐不涨但延迟继续涨
  -> 已经过饱和点

显存接近上限
  -> batch、seq_len、KV cache 或并发太激进
```

## 4. 当前建议的实验顺序

### 阶段 A：运行时参数

不重建 engine，先扫并发：

```bash
TRTLLM_PROFILE_CONCURRENCIES="1 2 4 8" bash scripts/profile_trtllm_loop.sh
```

再扫 KV cache：

```bash
TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION=0.60 bash scripts/profile_trtllm_loop.sh
TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION=0.75 bash scripts/profile_trtllm_loop.sh
TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION=0.85 bash scripts/profile_trtllm_loop.sh
```

目标是找到：

```text
吞吐增长明显但 p95/p99 延迟还能接受的点
```

### 阶段 B：build-time 参数

这些通常需要重新 build engine：

```text
TRTLLM_MAX_BATCH_SIZE
TRTLLM_MAX_SEQ_LEN
TRTLLM_MAX_NUM_TOKENS
dtype / quantization
```

每次只改一个维度，例如：

```bash
TRTLLM_MAX_BATCH_SIZE=4 \
TRTLLM_MAX_SEQ_LEN=4096 \
TRTLLM_MAX_NUM_TOKENS=4096 \
  bash scripts/trtllm.sh bench-build
```

然后用同一个 dataset 重跑 profiling。

### 阶段 C：质量对比

每个激进优化都要留质量证据：

```text
baseline 输出
optimized 输出
人工 review
自动评分
结论
```

建议目录：

```text
results/quality/<run_id>/
```

后续可以扩展脚本生成：

```text
baseline_outputs.jsonl
optimized_outputs.jsonl
diff.md
review.md
scores.json
```

当前项目提供了轻量质量采样脚本：

```bash
python3 scripts/capture_quality_outputs.py \
  --run-id qwen25_15b_bf16_baseline \
  --label baseline \
  --dataset configs/benchmarks/cases/mixed.jsonl \
  --temperature 0
```

候选 engine 采样后，可以生成 diff：

```bash
python3 scripts/compare_quality_outputs.py \
  --baseline results/quality/qwen25_15b_bf16_baseline/baseline_outputs.jsonl \
  --candidate results/quality/qwen25_15b_int8_candidate/candidate_outputs.jsonl \
  --markdown results/quality/qwen25_15b_int8_candidate/diff.md
```

完整流程见：

```text
docs/trtllm-quality-workflow.md
```

### 阶段 D：公开评测

先选小规模任务，不要一开始跑很大榜单。

候选工具：

```text
lm-evaluation-harness
OpenCompass
HELM
```

建议优先：

```text
lm-evaluation-harness: 通用英文/推理/知识任务
OpenCompass: 中文和综合大模型评测
```

公开评测的目的不是替代业务测试，而是提供一个可复现的质量底线。

## 5. 什么时候需要写 CUDA 或拆模型

当前阶段不应该直接写 CUDA。

优先顺序应该是：

```text
1. 固定 workload
2. 跑性能和质量 baseline
3. 调 TensorRT-LLM 现有参数
4. 使用官方支持的量化和 plugin
5. 看 profiling 证据
6. 只有现有能力解决不了明确瓶颈时，才写 CUDA / plugin
```

需要底层手术的典型条件：

```text
现有 TensorRT-LLM 不支持某个模型结构
某个自定义算子无法表达
profiling 明确显示某个 kernel 是瓶颈
某个 shape 下官方 kernel 非常低效
要实现新 attention / KV cache / sampling 算法
```

否则不要手动拆模型。Qwen2 这类结构 TensorRT-LLM 已经有适配，当前更应该学习
怎么把现有能力用对。

## 6. 判断一次优化是否值得保留

一次优化只有同时满足下面条件，才算值得保留：

```text
同一 dataset 下吞吐提升或延迟下降
p95 / p99 延迟没有不可接受恶化
显存没有接近危险上限
输出质量没有明显退化
配置和结果能在 results/runs/<run_id>/ 复现
```

如果速度提升但质量明显下降，它不是无条件成功，只是一个候选点。

## 7. 当前下一步

现在先做三件事：

```text
1. 生成 mixed_100、short_chat_100、long_output_100、long_context_100
2. 用当前 BF16 engine 分别跑一轮 baseline
3. 从 results/runs/<run_id>/README.md 对比不同 workload 的瓶颈
```

之后再开始激进优化：

```text
先扫 concurrency 和 KV cache
再改 build 参数
最后尝试量化
```

每次只改一个变量。否则性能变了、质量变了，你无法判断原因。
