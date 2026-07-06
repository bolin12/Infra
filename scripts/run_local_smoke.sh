#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import sys

print("python:", sys.version)

try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch import failed: {exc!r}")

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    x = torch.randn((1024, 1024), device=device)
    y = x @ x
    torch.cuda.synchronize()
    print("gpu:", torch.cuda.get_device_name(0))
    print("smoke_sum:", float(y.sum().detach().cpu()))
else:
    print("No CUDA GPU visible.")
PY
