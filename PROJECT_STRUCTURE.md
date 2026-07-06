# 项目结构说明

## 顶层结构

```text
.
├── README.md
├── TODO.md
├── PROJECT_STRUCTURE.md
├── rtx4060_llm_inference_acceleration_roadmap.md
├── high_end_gpu_and_cluster_inference_practice.md
├── configs/
├── prompts/
├── scripts/
├── src/
├── experiments/
├── results/
├── cloud/
├── cluster/
└── docs/
```

## `configs/`

放实验配置，不放代码。

- `configs/models/`：模型清单、量化格式、下载位置、推荐后端。
- `configs/hardware/`：不同 GPU/机器的硬件记录。
- `configs/benchmarks/`：benchmark 参数，例如 context length、batch size、max tokens。

## `prompts/`

固定测试 prompt。所有 benchmark 都要从这里取 prompt，避免每次随手输入导致结果不可比。

建议分组：

- 中文问答；
- 代码补全；
- 长文本摘要；
- 数学/推理；
- RAG/多轮对话。

## `scripts/`

放可重复执行的脚本。

第一批脚本优先级：

1. 采集硬件和软件环境；
2. 本地 smoke test；
3. llama.cpp benchmark；
4. ExLlamaV2 benchmark；
5. vLLM/SGLang server 启动；
6. API 压测。

## `src/`

后续自己写的 Python/CUDA/benchmark 代码放这里。

初期可以为空，不急着抽象框架。先用脚本跑通实验，再把重复逻辑沉淀成代码。

## `experiments/`

每个实验主题一个目录，包含：

- `README.md`：实验目的、变量、指标；
- `commands.md`：实际运行命令；
- `notes.md`：观察和结论；
- `results/`：该实验的局部结果，最终重要结果再汇总到顶层 `results/`。

## `results/`

放跨实验汇总结果。

建议格式：

- CSV：方便画图和对比；
- JSON：保留结构化元数据；
- Markdown：写分析结论。

## `cloud/`

放云 GPU 平台相关操作记录。

- `cloud/runpod/`：RunPod 模板、启动记录、注意事项。
- `cloud/vastai/`：Vast.ai 机器筛选和风险记录。
- `cloud/lambda/`：Lambda Cloud 机器记录。
- `cloud/modal/`：Modal serverless GPU 实验。

## `cluster/`

放多 GPU 和集群实验。

- `cluster/local-sim/`：Docker Compose、K3s、Ray local cluster 等本地模拟。
- `cluster/multi-gpu/`：单机多卡、NCCL、tensor parallel。
- `cluster/kubernetes/`：K8s/GPU Operator/serving 部署。
- `cluster/slurm/`：Slurm 作业和 HPC 风格实验。

## `docs/`

放长期阅读笔记、论文笔记、框架源码笔记。
