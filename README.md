# TensorRT-LLM Inference Infra

这个仓库只保留 NVIDIA TensorRT-LLM serving 主线：单机单卡默认 `tp_size=1`，对外暴露 OpenAI-compatible API，后续再扩展多 GPU / Triton / 集群部署。

## 文档入口

- [TensorRT-LLM Runbook](./docs/trtllm-runbook.md)
- [Qwen 模型选择](./docs/qwen-trtllm-models.md)
- [Serving 运行笔记](./docs/trtllm-serving-notes.md)

## 快速启动

```bash
bash scripts/trtllm.sh build
bash scripts/trtllm.sh download
bash scripts/trtllm.sh bench-build
bash scripts/trtllm.sh serve-engine-detached
bash scripts/trtllm.sh health
bash scripts/trtllm_smoke.sh
```

默认配置在 [configs/serving/trtllm-single-gpu.env](./configs/serving/trtllm-single-gpu.env)。

## 当前生产路径

- 镜像：`nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3`
- 服务：`trtllm-serve serve`
- API：`/health`、`/v1/models`、`/v1/chat/completions`
- 输入：HuggingFace checkpoint 或 TensorRT-LLM engine
- 单卡参数：`TRTLLM_TP_SIZE=1`、`TRTLLM_PP_SIZE=1`、`TRTLLM_GPUS_PER_NODE=1`

## 常用命令

```bash
# 查看实际启动参数
bash scripts/trtllm.sh config

# 进入 TensorRT-LLM 容器
bash scripts/trtllm.sh shell

# 下载当前配置的 HuggingFace checkpoint 到 HF cache
bash scripts/trtllm.sh download

# 使用官方 trtllm-bench 构建 benchmark engine 工作区
bash scripts/trtllm.sh bench-build

# 使用持久化 engine 启动服务
bash scripts/trtllm.sh serve-engine-detached

# 基于持久化 engine 跑官方 benchmark
bash scripts/trtllm.sh bench-throughput
bash scripts/trtllm.sh bench-latency

# 前台启动，适合看完整构建/加载日志
bash scripts/trtllm.sh serve

# 停止服务
bash scripts/trtllm.sh stop

# 查看服务日志
bash scripts/trtllm.sh logs
```

## 项目结构

```text
.
├── README.md
├── Dockerfile
├── configs/         ← 实验配置（models / hardware / benchmarks）
├── prompts/         ← 固定测试 prompt（中文问答、代码、摘要、数学）
├── scripts/         ← TensorRT-LLM serving、健康检查、环境采集
├── engines/         ← TensorRT-LLM engine，本地生成，不提交
├── results/         ← 跨实验汇总（env / raw / tables / reports / plots）
└── docs/            ← TensorRT-LLM serving 记录
```

## 基本原则

- 模型权重不要提交到仓库
- TensorRT engine 不提交到仓库
- 原始结果、硬件信息、运行命令要保存
- 每次实验必须固定模型、engine、TensorRT-LLM 镜像、context length、batch size
