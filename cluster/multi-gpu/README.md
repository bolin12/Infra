# Multi GPU

目标：有多 GPU 机器后，练 tensor parallel、NCCL 和 topology 分析。

## 必须记录

```bash
nvidia-smi
nvidia-smi topo -m
```

## 重点问题

- TP=2 是否接近 2 倍？
- TP=4 为什么可能不线性？
- 通信瓶颈来自 PCIe、NVLink 还是 batch 太小？
- 显存是否只是简单相加？
