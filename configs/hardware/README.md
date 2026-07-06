# 硬件配置记录

每次换机器都要记录硬件环境。尤其是云 GPU，不记录环境就很难解释性能差异。

## 必须记录

- GPU 型号和数量；
- 显存大小；
- 驱动版本；
- CUDA 版本；
- PyTorch 版本；
- CPU；
- 内存；
- 磁盘类型；
- `nvidia-smi topo -m`；
- 是否 NVLink；
- 云平台和实例类型；
- 租用价格和日期。

## 建议文件名

```text
rtx4060_local_YYYYMMDD.md
runpod_rtx4090_YYYYMMDD.md
lambda_a100_YYYYMMDD.md
aws_p5_h100x8_YYYYMMDD.md
```
