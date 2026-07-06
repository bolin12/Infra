# LLM Inference Infra Lab

这个仓库用于整理从 RTX 4060 单卡推理练习，到高端 GPU、多卡和小型集群推理实验的学习路线、脚本、配置和实验记录。

当前主线：

1. **RTX 4060 本地推理基本功**：llama.cpp、ExLlamaV2、vLLM、SGLang、量化、KV cache、speculative decoding。
2. **可迁移 benchmark 套件**：同一套 prompt、模型、参数和结果格式，可以在 4060、4090、A100、H100 上重复运行。
3. **高端 GPU 短租实验**：按小时租 4090/A100/H100，只跑准备好的实验，不在云上临时开发。
4. **多 GPU / 集群推理练习**：tensor parallel、NCCL、topology、Ray/K3s/Docker Compose、serving 压测。

## 文档入口

- [RTX 4060 个人 PC 上的 LLM 推理加速学习与研究路线](./rtx4060_llm_inference_acceleration_roadmap.md)
- [从 RTX 4060 走向高端 GPU 和集群推理练习的机会路线](./high_end_gpu_and_cluster_inference_practice.md)
- [项目结构说明](./PROJECT_STRUCTURE.md)
- [下一步任务清单](./TODO.md)

## 推荐推进顺序

1. 先完成 `experiments/01_rtx4060_baseline`：建立 4060 baseline。
2. 再完成 `experiments/02_quantization_comparison`：比较 GGUF/AWQ/GPTQ/EXL2。
3. 然后做 `experiments/03_kv_cache_long_context`：测长上下文和 KV cache。
4. 接着做 `experiments/04_speculative_decoding`：验证 draft/target 是否真正加速。
5. 最后把同一套 benchmark 迁移到 `experiments/05_cloud_single_gpu` 和 `experiments/06_multi_gpu_serving`。

## 基本原则

- 模型权重不要提交到仓库。
- 原始结果、硬件信息、运行命令要保存。
- 每次实验必须固定 prompt、模型、后端版本、context length、batch size。
- 云 GPU 只用于跑准备好的实验，避免边租边写代码。
