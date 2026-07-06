# TensorRT-LLM Benchmark 配置

这里记录 TensorRT-LLM serving 压测的固定变量。

完整执行流程见 [../../docs/trtllm-runbook.md](../../docs/trtllm-runbook.md)。

## 测试用例

- `smoke.jsonl`：最小连通性测试，只有 4 条，不适合严肃性能结论。
- `cases/short_chat.jsonl`：短输入短输出。
- `cases/long_output.jsonl`：短/中输入长输出，主要观察 decode。
- `cases/long_context.jsonl`：长输入短输出，主要观察 prefill。
- `cases/mixed.jsonl`：混合场景。

生成更大数据集：

```bash
python3 scripts/make_benchmark_dataset.py --scenario short_chat --count 100
python3 scripts/make_benchmark_dataset.py --scenario long_output --count 100
python3 scripts/make_benchmark_dataset.py --scenario long_context --count 100
python3 scripts/make_benchmark_dataset.py --scenario mixed --count 100
```

## 第一版参数

- context length：1k / 4k / 8k / 16k
- max new tokens：128 / 512
- concurrency：1 / 2 / 4 / 8
- `TRTLLM_MAX_BATCH_SIZE`：8 起步
- `TRTLLM_MAX_NUM_TOKENS`：4096 起步
- temperature：0 或固定值
- top_p：固定值
- prompt：从 `prompts/` 读取

## 指标

- TTFT；
- prefill tokens/s；
- decode tokens/s；
- total latency；
- TensorRT engine size；
- GPU 显存和 KV cache blocks；
- GPU utilization；
- power；
- 输出质量备注。
