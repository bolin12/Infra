# 01 RTX 4060 Baseline

目标：建立本地 RTX 4060 的推理 baseline。

## 要做什么

- 采集环境：驱动、CUDA、PyTorch、GPU 信息。
- 跑通 2B/4B/8B 模型的最小推理。
- 比较 llama.cpp、ExLlamaV2、vLLM/SGLang 的可运行边界。
- 记录 TTFT、prefill、decode、显存。

## 输出

- 本机硬件记录；
- baseline 表格；
- 第一版推荐后端；
- 失败案例记录。
