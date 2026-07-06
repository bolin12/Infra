#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

check_cmd() {
    local name="$1"
    if command -v "$name" >/dev/null 2>&1; then
        printf '[ok] command: %s\n' "$name"
    else
        printf '[missing] command: %s\n' "$name"
    fi
}

check_path() {
    local path="$1"
    if [[ -e "$path" ]]; then
        printf '[ok] path: %s\n' "$path"
    else
        printf '[missing] path: %s\n' "$path"
    fi
}

echo "== commands =="
check_cmd docker
check_cmd nvidia-smi
check_cmd python3
check_cmd jq

echo
echo "== project files =="
check_path configs/serving/trtllm-single-gpu.env
check_path configs/benchmarks/smoke.jsonl
check_path configs/benchmarks/cases/mixed.jsonl
check_path scripts/trtllm.sh
check_path scripts/profile_trtllm_loop.sh
check_path scripts/capture_quality_outputs.py
check_path docs/trtllm-optimization-roadmap.md
check_path docs/trtllm-quality-workflow.md

echo
echo "== generated local state =="
check_path models/hf-cache
check_path engines/bench/Qwen/Qwen2.5-1.5B-Instruct/tp_1_pp_1/rank0.engine
check_path results/runs
check_path results/quality

echo
echo "== config =="
bash scripts/trtllm.sh config

echo
echo "== docker image =="
if docker image inspect infra-trtllm:25.06 >/dev/null 2>&1; then
    docker image inspect infra-trtllm:25.06 --format '[ok] image: {{index .RepoTags 0}} {{.Id}}'
else
    echo "[missing] image: infra-trtllm:25.06"
fi

echo
echo "== gpu =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
fi
