# Benchmark 配置

这里记录 benchmark 的固定变量。

## 第一版参数

- context length：1k / 4k / 8k / 16k
- max new tokens：128 / 512
- batch size：1 / 2 / 4 / 8
- temperature：0 或固定值
- top_p：固定值
- prompt：从 `prompts/` 读取

## 指标

- TTFT；
- prefill tokens/s；
- decode tokens/s；
- total latency；
- GPU 显存；
- GPU utilization；
- power；
- 输出质量备注。
