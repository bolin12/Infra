# TensorRT-LLM Inference Infra

这个仓库只保留 NVIDIA TensorRT-LLM serving 主线：单机单卡默认 `tp_size=1`，对外暴露 OpenAI-compatible API，后续再扩展多 GPU / Triton / 集群部署。

## 文档入口

- [TensorRT-LLM Runbook](./docs/trtllm-runbook.md)
- [Qwen 模型选择](./docs/qwen-trtllm-models.md)
- [Serving 运行笔记](./docs/trtllm-serving-notes.md)
- [推理加速路线](./docs/trtllm-optimization-roadmap.md)
- [质量对比流程](./docs/trtllm-quality-workflow.md)
- [Profiling 闭环练习](./experiments/trtllm/README.md)

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

## 建议工作流

第一次完整实验建议按这个顺序走：

```bash
# 1. 构建和确认 engine
bash scripts/trtllm.sh build
bash scripts/trtllm.sh download
bash scripts/trtllm.sh bench-build
bash scripts/trtllm.sh config

# 2. 生成固定文本 workload
python3 scripts/make_benchmark_dataset.py --scenario mixed --count 100

# 3. 跑性能 profiling
TRTLLM_BENCH_DATASET=/workspace/configs/benchmarks/generated/mixed_100.jsonl \
  TRTLLM_PROFILE_REQUESTS=100 \
  TRTLLM_PROFILE_WARMUP=10 \
  TRTLLM_PROFILE_CONCURRENCIES="1 2 4 8" \
  bash scripts/profile_trtllm_loop.sh

# 4. 启动服务并采质量 baseline
bash scripts/trtllm.sh serve-engine-detached
bash scripts/trtllm.sh health
python3 scripts/capture_quality_outputs.py \
  --run-id qwen25_15b_bf16_baseline \
  --label baseline \
  --dataset configs/benchmarks/cases/mixed.jsonl \
  --temperature 0
```

每轮性能结果看：

```text
results/runs/<run_id>/README.md
```

质量输出看：

```text
results/quality/<run_id>/
```

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

# 快速检查项目关键文件、本地 engine、Docker 镜像和 GPU
bash scripts/check_project.sh

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

# 跑一轮 latency + concurrency sweep + GPU telemetry，并汇总 JSON
bash scripts/profile_trtllm_loop.sh

# 前台启动，适合看完整构建/加载日志
bash scripts/trtllm.sh serve

# 停止服务
bash scripts/trtllm.sh stop

# 查看服务日志
bash scripts/trtllm.sh logs
```

## 脚本职责

```text
scripts/trtllm.sh
  构建镜像、下载模型、构建 engine、启动服务、跑官方 bench。

scripts/profile_trtllm_loop.sh
  跑 latency + throughput concurrency sweep + GPU telemetry，
  并生成 results/runs/<run_id>/README.md。

scripts/summarize_trtllm_json.py
  把 TensorRT-LLM benchmark JSON 汇总成 CSV/Markdown。

scripts/make_benchmark_dataset.py
  生成 short_chat / long_output / long_context / mixed JSONL workload。

scripts/capture_quality_outputs.py
  调 OpenAI-compatible 服务，保存固定 case 的实际输出。

scripts/compare_quality_outputs.py
  对比 baseline/candidate 输出，生成 Markdown diff 供人工 review。

scripts/collect_env.sh
  采集环境、镜像、GPU、配置版本信息。

scripts/check_project.sh
  快速检查关键命令、配置、engine、镜像和 GPU 状态。
```

## 项目结构

```text
.
├── README.md
├── Dockerfile
├── configs/         ← 实验配置（models / hardware / benchmarks）
├── prompts/         ← 固定测试 prompt（中文问答、代码、摘要、数学）
├── scripts/         ← TensorRT-LLM serving、健康检查、环境采集
├── experiments/     ← profiling 闭环和实验说明
├── engines/         ← TensorRT-LLM engine，本地生成，不提交
├── results/         ← 跨实验汇总（env / raw / tables / reports / plots）
└── docs/            ← TensorRT-LLM serving 记录
```

## 结果目录

```text
results/runs/<run_id>/
  每轮性能实验的主目录。优先阅读 README.md。

results/quality/<run_id>/
  每轮质量采样和人工 review 目录。

results/raw/
results/tables/
results/reports/
  兼容旧流程的扁平输出目录。长期查看优先使用 results/runs/。

configs/benchmarks/cases/
  小型固定质量/功能 case，适合人工 review。

configs/benchmarks/generated/
  自动生成的大型性能 workload，不提交 JSONL 产物。
```

## 基本原则

- 模型权重不要提交到仓库
- TensorRT engine 不提交到仓库
- 原始结果、硬件信息、运行命令要保存
- 每次实验必须固定模型、engine、TensorRT-LLM 镜像、context length、batch size
