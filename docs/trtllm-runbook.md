# TensorRT-LLM Runbook

This runbook is the operational path for this repository. It assumes a local
single NVIDIA GPU first, then leaves room for larger GPUs later.

## 1. Prerequisites

Host requirements:

- NVIDIA driver works on the host.
- Docker can run GPU containers.
- Network can reach the configured Hugging Face endpoint.

Quick checks:

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

The local config defaults to `https://hf-mirror.com` because direct
`huggingface.co` access may fail from this machine.

## 2. Build The Local Image

```bash
bash scripts/trtllm.sh build
```

The image is intentionally thin. It uses NVIDIA's NGC image:

```text
nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3
```

The container already contains the required TensorRT-LLM tools:

```text
trtllm-serve
trtllm-build
trtllm-bench
trtllm-eval
trtllm-prune
trtllm-refit
tritonserver
```

## 3. Review Configuration

Default config:

```text
configs/serving/trtllm-single-gpu.env
```

Show the resolved command line:

```bash
bash scripts/trtllm.sh config
```

Important defaults:

```text
TRTLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
TRTLLM_TP_SIZE=1
TRTLLM_PP_SIZE=1
TRTLLM_GPUS_PER_NODE=1
TRTLLM_MAX_BATCH_SIZE=8
TRTLLM_MAX_NUM_TOKENS=4096
TRTLLM_MAX_SEQ_LEN=4096
TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION=0.75
```

## 4. Download The Checkpoint

```bash
bash scripts/trtllm.sh download
```

The checkpoint is cached under:

```text
models/hf-cache/
```

This directory is ignored by Git.

## 5. Build A Persistent Engine

```bash
bash scripts/trtllm.sh bench-build
```

The default engine path is:

```text
engines/bench/Qwen/Qwen2.5-1.5B-Instruct/tp_1_pp_1/
```

Expected important files:

```text
rank0.engine
config.json
tokenizer.json
tokenizer_config.json
```

Do not commit `engines/`. TensorRT engines are tied to GPU architecture,
TensorRT-LLM version, TensorRT version, and build-time limits.

## 6. Serve From The Engine

Start in the background:

```bash
bash scripts/trtllm.sh serve-engine-detached
```

Check readiness:

```bash
bash scripts/trtllm.sh health
```

Run an OpenAI-compatible smoke request:

```bash
bash scripts/trtllm_smoke.sh
```

Stop the service:

```bash
bash scripts/trtllm.sh stop
```

View logs:

```bash
bash scripts/trtllm.sh logs
```

## 7. Direct Serve Path

For quick testing, `trtllm-serve` can build/load from the Hugging Face
checkpoint at service startup:

```bash
bash scripts/trtllm.sh serve-detached
```

Use this for quick validation only. Prefer the persistent engine path for
repeatable runs and benchmarking.

## 8. Benchmark

The default benchmark dataset is:

```text
configs/benchmarks/smoke.jsonl
```

Throughput:

```bash
bash scripts/trtllm.sh bench-throughput
```

Latency:

```bash
bash scripts/trtllm.sh bench-latency
```

Reports are written under:

```text
results/raw/
```

Useful metrics to compare:

- TTFT
- TPOT
- output tokens/sec
- request throughput
- average request latency
- engine size
- KV cache allocation

## 9. Change Model

Edit:

```text
configs/serving/trtllm-single-gpu.env
```

Example small Qwen candidates:

```text
TRTLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
TRTLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
TRTLLM_MODEL=Qwen/Qwen3-0.6B
TRTLLM_MODEL=Qwen/Qwen3-1.7B
```

Then rebuild a matching engine:

```bash
bash scripts/trtllm.sh stop || true
bash scripts/trtllm.sh download
bash scripts/trtllm.sh bench-build
bash scripts/trtllm.sh serve-engine-detached
bash scripts/trtllm.sh health
```

If using a local Hugging Face checkpoint directory, set:

```text
TRTLLM_MODEL=/models/my-hf-checkpoint
TRTLLM_MODEL_PATH=/models/my-hf-checkpoint
TRTLLM_TOKENIZER=/models/my-hf-checkpoint
```

## 10. Troubleshooting

### Service Looks Stuck While Starting

Check logs:

```bash
bash scripts/trtllm.sh logs
```

If logs show `Downloading HF model`, verify endpoint connectivity:

```bash
bash scripts/trtllm.sh shell
python3 - <<'PY'
import urllib.request
for url in ["https://hf-mirror.com", "https://huggingface.co"]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            print(url, r.status)
    except Exception as exc:
        print(url, type(exc).__name__, exc)
PY
```

### MPI Process Appears

TensorRT-LLM may launch an OpenMPI singleton process even for local rank 1.
The expected single-GPU evidence is:

```text
MPI size: 1, MPI local size: 1, rank: 0
Set nccl_plugin to None.
```

Do not add SSH setup for local single-GPU serving.

### CUDA Out Of Memory

Reduce one or more:

```text
TRTLLM_MAX_BATCH_SIZE
TRTLLM_MAX_NUM_TOKENS
TRTLLM_MAX_SEQ_LEN
TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION
```

Then rebuild the engine.

### Engine Not Found

Run:

```bash
bash scripts/trtllm.sh config
find engines -maxdepth 6 -type f -name rank0.engine -print
```

Make sure `TRTLLM_ENGINE_DIR` either points to an existing engine directory or
is empty so the default `engines/bench/<model>/tp_1_pp_1` path is used.

## 11. Evidence Capture

Before comparing performance, capture environment information:

```bash
bash scripts/collect_env.sh
```

Output goes to:

```text
results/env/
```
