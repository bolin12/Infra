# TensorRT-LLM Benchmark 配置

这里记录 TensorRT-LLM serving 压测的固定变量。

完整执行流程见 [../../docs/trtllm-runbook.md](../../docs/trtllm-runbook.md)。

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
