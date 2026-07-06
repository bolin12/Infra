# 07 CUDA Kernels

目标：写小型 CUDA kernel，补齐推理底层能力。

## 起步顺序

1. RMSNorm；
2. RoPE；
3. GELU / SiLU；
4. INT4 weight-only matmul demo；
5. 简化 attention。

## 输出

- naive 版本；
- profiling 结果；
- 优化版本；
- 和 PyTorch / 现成后端的对比。
