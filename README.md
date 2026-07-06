# LLM Inference Infra Lab

从 RTX 4060 单卡推理练习，到高端 GPU、多卡和小型集群推理实验的学习路线、脚本、配置和实验记录。

## 当前主线

1. **RTX 4060 本地推理基本功**：llama.cpp、ExLlamaV2、vLLM、SGLang、量化、KV cache、speculative decoding。
2. **可迁移 benchmark 套件**：同一套 prompt、模型、参数和结果格式，可以在 4060、4090、A100、H100 上重复运行。
3. **高端 GPU 短租实验**：按小时租 4090/A100/H100，只跑准备好的实验，不在云上临时开发。
4. **多 GPU / 集群推理练习**：tensor parallel、NCCL、topology、Ray/K3s/Docker Compose、serving 压测。

## 工具链

| 类别 | 工具 | 用途 |
|------|------|------|
| 包管理 | uv + Python 3.12 | 环境管理，[配置日志](./docs/devlogs/2026-07-06-python-env-setup.md) |
| 推理引擎 | vLLM、SGLang、llama.cpp | 本地 & 云端 serving |
| 加速库 | Triton、FlashInfer、xformers | kernel 编译 & 高效 attention |
| Profile | Nsight Systems、Nsight Compute | 系统级 timeline & 核函数级分析 |

## 文档入口

- [RTX 4060 推理加速学习路线](./rtx4060_llm_inference_acceleration_roadmap.md)
- [高端 GPU & 集群推理路线](./high_end_gpu_and_cluster_inference_practice.md)
- [开发日志](./docs/devlogs/)

## 实验计划

### Phase 1：建立 4060 baseline

- [x] 确认本机 CUDA、驱动、Python、PyTorch 环境
- [ ] 跑通 `scripts/collect_env.sh`，保存硬件信息
- [ ] 选择 2B、4B、8B 三个模型作为第一批测试对象
- [ ] 准备 GGUF Q4/Q5/Q8 权重
- [ ] 跑通 llama.cpp 或 vLLM 的最小推理
- [ ] 记录 TTFT、prefill tokens/s、decode tokens/s、显存占用

### Phase 2：量化格式对比

- [ ] 固定同一模型、同一 prompt、同一 max tokens
- [ ] 比较 Q4/Q5/Q8、AWQ、GPTQ、EXL2
- [ ] 记录速度、显存、输出质量

### Phase 3：KV cache 和长上下文

- [ ] 测 context length：1k、4k、8k、16k
- [ ] 记录显存曲线和 decode 速度衰减
- [ ] 尝试 sliding window / attention sink

### Phase 4：Speculative decoding

- [ ] 选择 draft model（0.6B/1.7B）+ target model（4B/8B Q4）
- [ ] 测 acceptance rate 和速度对比

### Phase 5：迁移到云 GPU

- [ ] 本地打包 benchmark 脚本和 prompt
- [ ] 短租 4090/L40S → A100 → H100 逐级迁移

### Phase 6：多卡和集群

- [ ] Docker Compose 模拟多 worker serving
- [ ] Ray local cluster / K3s
- [ ] vLLM/SGLang tensor parallel（TP=1/2/4/8）

## 项目结构

```text
.
├── README.md
├── rtx4060_llm_inference_acceleration_roadmap.md
├── high_end_gpu_and_cluster_inference_practice.md
├── configs/         ← 实验配置（models / hardware / benchmarks）
├── prompts/         ← 固定测试 prompt（中文问答、代码、摘要、数学）
├── scripts/         ← 可重复执行脚本（采集环境、benchmark、压测）
├── src/             ← Python/CUDA 代码（用脚本跑通后再沉淀到此）
├── experiments/     ← 每个实验一个目录（README + commands.md + notes.md）
├── results/         ← 跨实验汇总（env / raw / tables / reports / plots）
├── cloud/           ← 云 GPU 平台记录（runpod / vastai / lambda / modal）
├── cluster/         ← 多 GPU & 集群实验（local-sim / multi-gpu / k8s / slurm）
└── docs/            ← 长期笔记（devlogs、论文笔记、源码阅读）
```

## 基本原则

- 模型权重不要提交到仓库
- 原始结果、硬件信息、运行命令要保存
- 每次实验必须固定 prompt、模型、后端版本、context length、batch size
- 云 GPU 只用于跑准备好的实验，避免边租边写代码
