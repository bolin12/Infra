# Qwen Models for TensorRT-LLM

## Practical Local Default

Use `Qwen/Qwen2.5-1.5B-Instruct` first on the RTX 4060. It has already been
validated locally with `trtllm-serve`, TensorRT-LLM 0.20.0, `tp_size=1`, and
`max_seq_len=4096`.

Validated local engine path:

```text
engines/bench/Qwen/Qwen2.5-1.5B-Instruct/tp_1_pp_1/
```

Validated flow:

```bash
bash scripts/trtllm.sh download
bash scripts/trtllm.sh bench-build
bash scripts/trtllm.sh serve-engine-detached
bash scripts/trtllm.sh health
bash scripts/trtllm_smoke.sh
```

## Candidate Model Types

1. HuggingFace checkpoints

   These are the most portable inputs for this repo. TensorRT-LLM builds a local
   engine for the current GPU and runtime. Good local candidates:

   - `Qwen/Qwen2.5-0.5B-Instruct`
   - `Qwen/Qwen2.5-1.5B-Instruct`
   - `Qwen/Qwen3-0.6B`
   - `Qwen/Qwen3-1.7B`

2. TensorRT-LLM checkpoints

   Community repositories exist for small Qwen models. They are not prebuilt
   engines; build an engine locally before serving.

3. NVIDIA optimized checkpoints

   NVIDIA publishes Qwen3 optimized checkpoints such as FP4/NVFP4 variants.
   Check the listed hardware compatibility before using them. Some are intended
   for Blackwell and are not a good default for RTX 40-series Ada GPUs.

## Why Not Rely on Prebuilt Engines

TensorRT engines are tied to build-time/runtime assumptions such as GPU
architecture, maximum batch size, sequence lengths, beam width, TensorRT, and
TensorRT-LLM versions. Prefer rebuilding engines locally under `./engines`.

## Sources

- NVIDIA TensorRT-LLM Qwen guide:
  https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/README.md
- Hugging Face TensorRT-LLM backend notes:
  https://huggingface.co/docs/text-generation-inference/backends/trtllm
- NVIDIA Qwen3-8B-NVFP4 model card:
  https://huggingface.co/nvidia/Qwen3-8B-NVFP4
- Community Qwen2.5 TensorRT-LLM checkpoint example:
  https://huggingface.co/Shoolife/Qwen2.5-1.5B-Instruct-TensorRT-LLM-Checkpoint-BF16
