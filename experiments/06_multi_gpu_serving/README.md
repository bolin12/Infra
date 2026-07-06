# 06 Multi GPU Serving

目标：练 tensor parallel、NCCL 和多 GPU serving。

## 实验对象

- vLLM tensor parallel；
- SGLang 多 GPU serving；
- 后期 TensorRT-LLM；
- NCCL smoke test；
- topology 与性能关系。

## 指标

- TP=1/2/4/8 scaling；
- TTFT；
- throughput；
- p95/p99 latency；
- GPU 利用率；
- 通信开销。
