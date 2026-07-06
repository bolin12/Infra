# TensorRT-LLM 质量对比流程

性能优化之后必须留一份质量证据。尤其是量化、低精度 KV cache、
speculative decoding、自定义 plugin 或 CUDA kernel 这类激进改动。

## 1. 启动服务

先用要评估的 engine 启动 OpenAI-compatible 服务：

```bash
bash scripts/trtllm.sh serve-engine-detached
bash scripts/trtllm.sh health
```

## 2. 采 baseline 输出

在当前稳定 engine 上采一组固定输出：

```bash
python3 scripts/capture_quality_outputs.py \
  --run-id qwen25_15b_bf16_baseline \
  --label baseline \
  --dataset configs/benchmarks/cases/mixed.jsonl \
  --temperature 0
```

输出目录：

```text
results/quality/qwen25_15b_bf16_baseline/
```

核心文件：

```text
baseline_outputs.jsonl
baseline_metadata.json
baseline_review.md
```

## 3. 采优化后输出

改 engine 或优化参数后，重新启动服务，再采候选输出：

```bash
python3 scripts/capture_quality_outputs.py \
  --run-id qwen25_15b_int8_candidate \
  --label candidate \
  --dataset configs/benchmarks/cases/mixed.jsonl \
  --temperature 0
```

## 4. 做输出差异 review

```bash
python3 scripts/compare_quality_outputs.py \
  --baseline results/quality/qwen25_15b_bf16_baseline/baseline_outputs.jsonl \
  --candidate results/quality/qwen25_15b_int8_candidate/candidate_outputs.jsonl \
  --markdown results/quality/qwen25_15b_int8_candidate/diff.md
```

然后人工打开：

```text
results/quality/qwen25_15b_int8_candidate/diff.md
```

逐条标注：

```text
pass
warn
fail
```

## 5. Review 标准

重点看：

```text
是否回答同一个问题
是否有明显事实错误
中文是否退化
格式是否崩坏
代码是否不可用
是否重复
是否提前停止
是否输出明显变短
```

一次激进优化只有同时满足下面条件，才应该保留：

```text
性能指标变好
p95 / p99 延迟可接受
显存没有危险增长
质量 review 没有明显 fail
配置和结果能复现
```

## 6. 当前建议

先用小型固定 case 做质量守门：

```text
configs/benchmarks/cases/short_chat.jsonl
configs/benchmarks/cases/long_output.jsonl
configs/benchmarks/cases/long_context.jsonl
configs/benchmarks/cases/mixed.jsonl
```

等性能优化路线稳定后，再接公开评测工具：

```text
lm-evaluation-harness
OpenCompass
```
