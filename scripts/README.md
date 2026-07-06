# scripts

这里放可重复执行的脚本。

第一批脚本目标：

1. `collect_env.sh`：采集本机或云机器环境。
2. `run_local_smoke.sh`：确认 Python、CUDA、PyTorch 和 GPU 可用。
3. 后续增加 llama.cpp、ExLlamaV2、vLLM、SGLang benchmark 脚本。

脚本原则：

- 输出结果保存到 `results/` 或对应实验目录；
- 命令参数要显式，不依赖隐藏状态；
- 每次运行要记录时间、git commit、硬件环境。
