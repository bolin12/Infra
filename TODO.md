# 下一步任务清单

## Phase 1：建立 4060 baseline

- [ ] 确认本机 CUDA、驱动、Python、PyTorch 环境。
- [ ] 跑通 `scripts/collect_env.sh`，保存硬件信息。
- [ ] 选择 2B、4B、8B 三个模型作为第一批测试对象。
- [ ] 准备 GGUF Q4/Q5/Q8 或 EXL2/AWQ/GPTQ 权重。
- [ ] 跑通 llama.cpp 或 ExLlamaV2 的最小推理。
- [ ] 记录 TTFT、prefill tokens/s、decode tokens/s、显存占用。

## Phase 2：量化格式对比

- [ ] 固定同一模型、同一 prompt、同一 max tokens。
- [ ] 比较 Q4/Q5/Q8、AWQ、GPTQ、EXL2。
- [ ] 记录速度、显存、输出质量。
- [ ] 写出第一版量化推荐结论。

## Phase 3：KV cache 和长上下文

- [ ] 测 context length：1k、4k、8k、16k。
- [ ] 记录显存曲线。
- [ ] 记录 decode 速度随上下文变长的变化。
- [ ] 尝试 sliding window / attention sink 简化实验。

## Phase 4：Speculative decoding

- [ ] 选择 draft model：0.6B 或 1.7B。
- [ ] 选择 target model：4B 或 8B Q4。
- [ ] 测 acceptance rate。
- [ ] 比较开启/关闭 speculative decoding 的速度和显存。

## Phase 5：迁移到云 GPU

- [ ] 本地打包 benchmark 脚本和 prompt。
- [ ] 短租 4090/L40S 做第一轮迁移。
- [ ] 短租 A100 做第二轮迁移。
- [ ] 如果有预算，再短租 H100 做 prefill/FP8/高 batch 对比。

## Phase 6：多卡和集群

- [ ] 本地用 Docker Compose 模拟多 worker serving。
- [ ] 跑通 Ray local cluster 或 K3s。
- [ ] 有多卡资源后，测试 vLLM/SGLang tensor parallel。
- [ ] 记录 `nvidia-smi topo -m` 和 NCCL 信息。
- [ ] 做 TP=1/2/4/8 scaling 分析。
