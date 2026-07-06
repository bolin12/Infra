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
  echo "## docker"
  docker --version || true
  echo
  echo "## docker image"
  docker image inspect infra-trtllm:25.06 --format '{{.Id}} {{.Created}}' || true
  echo
  echo "## serving config"
  if [ -f configs/serving/trtllm-single-gpu.env ]; then
    sed -n '1,160p' configs/serving/trtllm-single-gpu.env
  fi
  echo
  echo "## trtllm container version"
  docker run --rm --gpus all --entrypoint /bin/bash infra-trtllm:25.06 -lc '
python3 - <<'"'"'PY'"'"'
try:
    import tensorrt_llm
    print("tensorrt_llm:", getattr(tensorrt_llm, "__version__", "unknown"))
except Exception as exc:
    print("trtllm_check_failed:", repr(exc))
PY
' || true
} | tee "${OUT_FILE}"

echo "saved: ${OUT_FILE}"
