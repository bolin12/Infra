# 03 KV Cache Long Context

目标：研究长上下文下 KV cache 对显存和 decode 速度的影响。

## 实验变量

- context length：1k / 4k / 8k / 16k；
- batch size：1 / 2 / 4；
- 普通 full KV cache；
- sliding window；
- attention sink；
- 后续再考虑 H2O / KIVI 简化复现。

## 输出

- 显存曲线；
- decode 速度曲线；
- 最大可用上下文；
- 质量下降备注。
