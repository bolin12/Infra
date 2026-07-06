#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="${TRTLLM_PROFILE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CONCURRENCIES="${TRTLLM_PROFILE_CONCURRENCIES:-1 2 4 8}"
REQUESTS="${TRTLLM_PROFILE_REQUESTS:-100}"
WARMUP="${TRTLLM_PROFILE_WARMUP:-10}"
RUN_LATENCY="${TRTLLM_PROFILE_RUN_LATENCY:-1}"
RUN_THROUGHPUT="${TRTLLM_PROFILE_RUN_THROUGHPUT:-1}"
CAPTURE_GPU="${TRTLLM_PROFILE_CAPTURE_GPU:-1}"

RAW_DIR="$PROJECT_ROOT/results/raw"
TABLE_DIR="$PROJECT_ROOT/results/tables"
REPORT_DIR="$PROJECT_ROOT/results/reports"
RUNS_DIR="$PROJECT_ROOT/results/runs"
RUN_DIR="$RUNS_DIR/$RUN_ID"
RUN_RAW_DIR="$RUN_DIR/raw"
MARKER="$RAW_DIR/.profile_${RUN_ID}.marker"
GPU_LOG="$RUN_DIR/gpu.csv"
CSV_OUT="$RUN_DIR/summary.csv"
MD_OUT="$RUN_DIR/SUMMARY.md"
RUN_README="$RUN_DIR/README.md"
RUN_INDEX="$RUNS_DIR/index.csv"
GPU_PID=""

mkdir -p "$RAW_DIR" "$TABLE_DIR" "$REPORT_DIR" "$RUN_RAW_DIR"
touch "$MARKER"

cleanup() {
    stop_gpu_capture
    rm -f "$MARKER"
}
trap cleanup EXIT

stop_gpu_capture() {
    if [[ -n "$GPU_PID" ]] && kill -0 "$GPU_PID" >/dev/null 2>&1; then
        kill "$GPU_PID" >/dev/null 2>&1 || true
        wait "$GPU_PID" >/dev/null 2>&1 || true
    fi
    GPU_PID=""
}

start_gpu_capture() {
    if [[ "$CAPTURE_GPU" != "1" ]]; then
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found; skip GPU capture" >&2
        return
    fi
    nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv \
        -lms 500 >"$GPU_LOG" &
    GPU_PID="$!"
    echo "gpu_log=$GPU_LOG"
}

write_run_metadata() {
    {
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'started_at=%s\n' "$(date -Is)"
        printf 'requests=%s\n' "$REQUESTS"
        printf 'warmup=%s\n' "$WARMUP"
        printf 'concurrencies=%s\n' "$CONCURRENCIES"
        printf 'run_latency=%s\n' "$RUN_LATENCY"
        printf 'run_throughput=%s\n' "$RUN_THROUGHPUT"
        printf 'capture_gpu=%s\n' "$CAPTURE_GPU"
        printf 'trtllm_bench_dataset=%s\n' "${TRTLLM_BENCH_DATASET:-}"
        printf 'trtllm_profile_run_id=%s\n' "${TRTLLM_PROFILE_RUN_ID:-}"
    } >"$RUN_DIR/profile.env"

    bash "$PROJECT_ROOT/scripts/trtllm.sh" config >"$RUN_DIR/config.txt"

    if [[ -f "$PROJECT_ROOT/configs/serving/trtllm-single-gpu.env" ]]; then
        cp "$PROJECT_ROOT/configs/serving/trtllm-single-gpu.env" "$RUN_DIR/serving.env"
    fi
}

run_latency() {
    echo "== latency: requests=$REQUESTS warmup=$WARMUP concurrency=1 =="
    TRTLLM_BENCH_REQUESTS="$REQUESTS" \
        TRTLLM_BENCH_WARMUP="$WARMUP" \
        TRTLLM_BENCH_CONCURRENCY=1 \
        bash "$PROJECT_ROOT/scripts/trtllm.sh" bench-latency
}

run_throughput() {
    local concurrency
    for concurrency in $CONCURRENCIES; do
        echo "== throughput: requests=$REQUESTS warmup=$WARMUP concurrency=$concurrency =="
        TRTLLM_BENCH_REQUESTS="$REQUESTS" \
            TRTLLM_BENCH_WARMUP="$WARMUP" \
            TRTLLM_BENCH_CONCURRENCY="$concurrency" \
            bash "$PROJECT_ROOT/scripts/trtllm.sh" bench-throughput
    done
}

summarize_new_reports() {
    mapfile -t artifacts < <(
        find "$RAW_DIR" \
            -type f \
            -newer "$MARKER" \
            -name '*.json' \
            | sort
    )

    if [[ "${#artifacts[@]}" -eq 0 ]]; then
        echo "No new benchmark JSON reports found." >&2
        return 1
    fi

    local artifact
    for artifact in "${artifacts[@]}"; do
        cp -f "$artifact" "$RUN_RAW_DIR/"
    done

    mapfile -t reports < <(
        find "$RUN_RAW_DIR" \
            -type f \
            -name '*.json' \
            ! -name '*outputs*' \
            | sort
    )

    python3 "$PROJECT_ROOT/scripts/summarize_trtllm_json.py" \
        --input "${reports[@]}" \
        --csv "$CSV_OUT" \
        --markdown "$MD_OUT" \
        --gpu-csv "$GPU_LOG" \
        --project-root "$PROJECT_ROOT"

    cp "$CSV_OUT" "$TABLE_DIR/trtllm_profile_${RUN_ID}.csv"
    cp "$MD_OUT" "$REPORT_DIR/trtllm_profile_${RUN_ID}.md"

    write_run_readme
    update_run_index

    echo "summary_csv=$CSV_OUT"
    echo "summary_markdown=$RUN_README"
    echo "run_dir=$RUN_DIR"
}

write_run_readme() {
    local dataset
    dataset="$(awk -F= '/^bench_dataset=/{print $2}' "$RUN_DIR/config.txt" | tail -1)"
    {
        printf '# TensorRT-LLM Profile Run %s\n\n' "$RUN_ID"
        printf '## Run Inputs\n\n'
        printf -- '- requests: `%s`\n' "$REQUESTS"
        printf -- '- warmup: `%s`\n' "$WARMUP"
        printf -- '- concurrencies: `%s`\n' "$CONCURRENCIES"
        printf -- '- dataset: `%s`\n' "${dataset:-unknown}"
        printf -- '- config snapshot: `config.txt`\n'
        printf -- '- serving env snapshot: `serving.env`\n'
        printf -- '- GPU telemetry: `gpu.csv`\n'
        printf -- '- raw reports: `raw/`\n\n'
        cat "$MD_OUT"
    } >"$RUN_README"
}

update_run_index() {
    mkdir -p "$RUNS_DIR"
    if [[ ! -f "$RUN_INDEX" ]]; then
        printf 'run_id,created_at,requests,warmup,concurrencies,run_dir,summary\n' >"$RUN_INDEX"
    fi
    printf '%s,%s,%s,%s,"%s",%s,%s\n' \
        "$RUN_ID" \
        "$(date -Is)" \
        "$REQUESTS" \
        "$WARMUP" \
        "$CONCURRENCIES" \
        "results/runs/$RUN_ID" \
        "results/runs/$RUN_ID/README.md" >>"$RUN_INDEX"
}

echo "run_id=$RUN_ID"
echo "requests=$REQUESTS"
echo "warmup=$WARMUP"
echo "concurrencies=$CONCURRENCIES"
echo "run_dir=$RUN_DIR"

write_run_metadata
start_gpu_capture

if [[ "$RUN_LATENCY" == "1" ]]; then
    run_latency
fi

if [[ "$RUN_THROUGHPUT" == "1" ]]; then
    run_throughput
fi

stop_gpu_capture
summarize_new_reports
