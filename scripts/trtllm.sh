#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="${TRTLLM_CONFIG:-$PROJECT_ROOT/configs/serving/trtllm-single-gpu.env}"

if [[ -f "$CONFIG_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
    set +a
fi

IMAGE="${TRTLLM_IMAGE:-infra-trtllm:25.06}"
CONTAINER="${TRTLLM_CONTAINER:-infra-trtllm}"
PORT="${TRTLLM_PORT:-8000}"
HOST="${TRTLLM_HOST:-0.0.0.0}"
GPU_DEVICE="${TRTLLM_GPU_DEVICE:-0}"

MODEL="${TRTLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_PATH="${TRTLLM_MODEL_PATH:-}"
TOKENIZER="${TRTLLM_TOKENIZER:-}"
BACKEND="${TRTLLM_BACKEND:-}"
EXTRA_OPTIONS="${TRTLLM_EXTRA_OPTIONS:-}"
BENCH_WORKSPACE="${TRTLLM_BENCH_WORKSPACE:-/engines/bench}"
BENCH_DATASET="${TRTLLM_BENCH_DATASET:-/workspace/configs/benchmarks/smoke.jsonl}"
BENCH_CONCURRENCY="${TRTLLM_BENCH_CONCURRENCY:-1}"
BENCH_REQUESTS="${TRTLLM_BENCH_REQUESTS:-4}"
BENCH_WARMUP="${TRTLLM_BENCH_WARMUP:-1}"
ENGINE_DIR="${TRTLLM_ENGINE_DIR:-$BENCH_WORKSPACE/$MODEL/tp_${TRTLLM_TP_SIZE:-1}_pp_${TRTLLM_PP_SIZE:-1}}"

mkdir -p "$PROJECT_ROOT/models" "$PROJECT_ROOT/engines" "$PROJECT_ROOT/results/raw"

docker_common_args=(
    --gpus "device=$GPU_DEVICE"
    --rm
    --ipc=host
    --shm-size=2g
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "$PROJECT_ROOT:/workspace"
    -v "$PROJECT_ROOT/models:/models"
    -v "$PROJECT_ROOT/engines:/engines"
    -e HF_ENDPOINT="${TRTLLM_HF_ENDPOINT:-https://hf-mirror.com}"
    -e HF_HOME=/models/hf-cache
    -e HUGGINGFACE_HUB_CACHE=/models/hf-cache
    -e TRANSFORMERS_CACHE=/models/hf-cache
    -e CUDA_VISIBLE_DEVICES=0
    -e TORCH_CUDA_ARCH_LIST=8.9
    -w /workspace
)

docker_serve_args=(
    "${docker_common_args[@]}"
    --name "$CONTAINER"
    -p "$PORT:$PORT"
)

serve_cmd=(
    trtllm-serve serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --tp_size "${TRTLLM_TP_SIZE:-1}"
    --pp_size "${TRTLLM_PP_SIZE:-1}"
    --gpus_per_node "${TRTLLM_GPUS_PER_NODE:-1}"
    --max_batch_size "${TRTLLM_MAX_BATCH_SIZE:-8}"
    --max_num_tokens "${TRTLLM_MAX_NUM_TOKENS:-4096}"
    --max_seq_len "${TRTLLM_MAX_SEQ_LEN:-4096}"
    --kv_cache_free_gpu_memory_fraction "${TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION:-0.75}"
    --num_postprocess_workers "${TRTLLM_NUM_POSTPROCESS_WORKERS:-0}"
    --log_level "${TRTLLM_LOG_LEVEL:-info}"
)

if [[ -n "$TOKENIZER" ]]; then
    serve_cmd+=(--tokenizer "$TOKENIZER")
fi

if [[ -n "$BACKEND" ]]; then
    serve_cmd+=(--backend "$BACKEND")
fi

if [[ -n "$EXTRA_OPTIONS" ]]; then
    serve_cmd+=(--extra_llm_api_options "$EXTRA_OPTIONS")
fi

engine_serve_cmd=(
    trtllm-serve serve "$ENGINE_DIR"
    --tokenizer "${TOKENIZER:-$MODEL}"
    --host "$HOST"
    --port "$PORT"
    --tp_size "${TRTLLM_TP_SIZE:-1}"
    --pp_size "${TRTLLM_PP_SIZE:-1}"
    --gpus_per_node "${TRTLLM_GPUS_PER_NODE:-1}"
    --kv_cache_free_gpu_memory_fraction "${TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION:-0.75}"
    --num_postprocess_workers "${TRTLLM_NUM_POSTPROCESS_WORKERS:-0}"
    --log_level "${TRTLLM_LOG_LEVEL:-info}"
)

bench_base_cmd=(
    trtllm-bench
    -m "$MODEL"
    -w "$BENCH_WORKSPACE"
)

if [[ -n "$MODEL_PATH" ]]; then
    bench_base_cmd+=(--model_path "$MODEL_PATH")
fi

build_image() {
    docker build -t "$IMAGE" "$PROJECT_ROOT"
}

ensure_image() {
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        build_image
    fi
}

pretty_json() {
    if command -v jq >/dev/null 2>&1; then
        jq .
    else
        python3 -m json.tool
    fi
}

case "${1:-serve}" in
    build)
        build_image
        ;;
    serve)
        ensure_image
        docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        exec docker run "${docker_serve_args[@]}" "$IMAGE" "${serve_cmd[@]}"
        ;;
    serve-detached)
        ensure_image
        docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        docker run -d "${docker_serve_args[@]}" "$IMAGE" "${serve_cmd[@]}"
        ;;
    serve-engine)
        ensure_image
        docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        exec docker run "${docker_serve_args[@]}" "$IMAGE" "${engine_serve_cmd[@]}"
        ;;
    serve-engine-detached)
        ensure_image
        docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        docker run -d "${docker_serve_args[@]}" "$IMAGE" "${engine_serve_cmd[@]}"
        ;;
    shell)
        ensure_image
        exec docker run -it "${docker_common_args[@]}" --entrypoint /bin/bash "$IMAGE"
        ;;
    download)
        ensure_image
        exec docker run "${docker_common_args[@]}" "$IMAGE" \
            huggingface-cli download "$MODEL"
        ;;
    bench-build)
        ensure_image
        docker run "${docker_common_args[@]}" "$IMAGE" \
            "${bench_base_cmd[@]}" \
            build \
            --tp_size "${TRTLLM_TP_SIZE:-1}" \
            --pp_size "${TRTLLM_PP_SIZE:-1}" \
            --max_batch_size "${TRTLLM_MAX_BATCH_SIZE:-8}" \
            --max_num_tokens "${TRTLLM_MAX_NUM_TOKENS:-4096}" \
            --max_seq_len "${TRTLLM_MAX_SEQ_LEN:-4096}"
        printf 'engine_dir=%s\n' "$ENGINE_DIR"
        ;;
    bench-throughput)
        ensure_image
        stamp="$(date +%Y%m%d_%H%M%S)"
        exec docker run "${docker_common_args[@]}" "$IMAGE" \
            "${bench_base_cmd[@]}" \
            throughput \
            --engine_dir "$ENGINE_DIR" \
            --dataset "$BENCH_DATASET" \
            --num_requests "$BENCH_REQUESTS" \
            --warmup "$BENCH_WARMUP" \
            --concurrency "$BENCH_CONCURRENCY" \
            --kv_cache_free_gpu_mem_fraction "${TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION:-0.75}" \
            --report_json "/workspace/results/raw/throughput_${stamp}.json" \
            --output_json "/workspace/results/raw/throughput_outputs_${stamp}.json"
        ;;
    bench-latency)
        ensure_image
        stamp="$(date +%Y%m%d_%H%M%S)"
        exec docker run "${docker_common_args[@]}" "$IMAGE" \
            "${bench_base_cmd[@]}" \
            latency \
            --engine_dir "$ENGINE_DIR" \
            --dataset "$BENCH_DATASET" \
            --num_requests "$BENCH_REQUESTS" \
            --warmup "$BENCH_WARMUP" \
            --concurrency "$BENCH_CONCURRENCY" \
            --kv_cache_free_gpu_mem_fraction "${TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION:-0.75}" \
            --report_json "/workspace/results/raw/latency_${stamp}.json"
        ;;
    health)
        curl -fsS "http://127.0.0.1:$PORT/health"
        echo
        curl -fsS "http://127.0.0.1:$PORT/v1/models" | pretty_json
        ;;
    stop)
        docker rm -f "$CONTAINER"
        ;;
    logs)
        docker logs -f "$CONTAINER"
        ;;
    config)
        printf 'config=%s\nimage=%s\ncontainer=%s\nmodel=%s\nport=%s\ngpu=%s\n' \
            "$CONFIG_FILE" "$IMAGE" "$CONTAINER" "$MODEL" "$PORT" "$GPU_DEVICE"
        printf 'bench_workspace=%s\n' "$BENCH_WORKSPACE"
        printf 'engine_dir=%s\n' "$ENGINE_DIR"
        printf 'bench_dataset=%s\n' "$BENCH_DATASET"
        printf 'serve command:'
        printf ' %q' "${serve_cmd[@]}"
        printf '\n'
        printf 'engine serve command:'
        printf ' %q' "${engine_serve_cmd[@]}"
        printf '\n'
        ;;
    *)
        echo "Usage: $0 {build|download|serve|serve-detached|serve-engine|serve-engine-detached|bench-build|bench-throughput|bench-latency|shell|health|logs|stop|config}"
        exit 1
        ;;
esac
