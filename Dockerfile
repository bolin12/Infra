FROM nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/models/hf-cache \
    HUGGINGFACE_HUB_CACHE=/models/hf-cache \
    TRANSFORMERS_CACHE=/models/hf-cache \
    TRTLLM_LOG_LEVEL=info \
    TORCH_CUDA_ARCH_LIST=8.9

WORKDIR /workspace

CMD ["trtllm-serve", "--help"]
