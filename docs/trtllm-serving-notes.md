# TensorRT-LLM Serving Notes

## Current Path

- Base image: `nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3`
- Server: `trtllm-serve serve`
- API: OpenAI-compatible `/v1/chat/completions`
- Local default: single GPU, `tp_size=1`, `pp_size=1`, `gpus_per_node=1`

For the complete operating procedure, use
[trtllm-runbook.md](./trtllm-runbook.md).

## Local Network

The local container cannot reach `https://huggingface.co`, but can reach
`https://hf-mirror.com`. The serving config therefore sets:

```text
TRTLLM_HF_ENDPOINT=https://hf-mirror.com
```

Without this, the server appears to hang while downloading the HuggingFace
checkpoint.

## MPI Note

TensorRT-LLM 0.20 may launch an OpenMPI singleton process even for local
single-rank execution. The important runtime evidence is:

```text
MPI size: 1, MPI local size: 1, rank: 0
Set nccl_plugin to None.
```

That is not a multi-node SSH/MPI deployment path. Do not add SSH setup to the
image for local single-GPU serving.
