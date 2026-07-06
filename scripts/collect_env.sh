#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-results/env}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${OUT_DIR}/env_${STAMP}.txt"

mkdir -p "${OUT_DIR}"

{
  echo "date: $(date -Is)"
  echo
  echo "## uname"
  uname -a
  echo
  echo "## lsb_release"
  if command -v lsb_release >/dev/null 2>&1; then
    lsb_release -a
  else
    cat /etc/os-release || true
  fi
  echo
  echo "## nvidia-smi"
  nvidia-smi || true
  echo
  echo "## nvidia-smi topo -m"
  nvidia-smi topo -m || true
  echo
  echo "## nvcc"
  nvcc --version || true
  echo
  echo "## python"
  python --version || true
  echo
  echo "## pytorch"
  python - <<'PY' || true
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_version:", torch.version.cuda)
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"device_{i}:", torch.cuda.get_device_name(i))
except Exception as exc:
    print("torch_check_failed:", repr(exc))
PY
} | tee "${OUT_FILE}"

echo "saved: ${OUT_FILE}"
