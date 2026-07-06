# 04 Speculative Decoding

目标：验证消费级 GPU 上 speculative decoding 是否真正加速。

## 建议组合

- draft：0.6B / 1.7B；
- target：4B / 8B Q4。

## 指标

- acceptance rate；
- 平均接受 token 数；
- tokens/s；
- latency；
- 显存占用；
- 不同任务类型下是否加速。
