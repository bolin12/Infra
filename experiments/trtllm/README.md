# TensorRT-LLM Profiling Loop

This folder records the profiling workflow for the local TensorRT-LLM practice
project. Generated raw data still lives under `results/`.

## Minimum Loop

```text
config
  -> build or reuse engine
  -> run benchmark
  -> capture GPU telemetry
  -> summarize JSON
  -> change one variable
```

## Baseline Run

Use the existing engine first:

```bash
bash scripts/trtllm.sh config
bash scripts/profile_trtllm_loop.sh
```

The profiling wrapper writes:

```text
results/runs/<run_id>/README.md
results/runs/<run_id>/summary.csv
results/runs/<run_id>/gpu.csv
results/runs/<run_id>/config.txt
results/runs/<run_id>/serving.env
results/runs/<run_id>/raw/*.json
results/runs/index.csv
```

`results/raw/`, `results/tables/`, and `results/reports/` are still written as
flat compatibility outputs. Prefer `results/runs/<run_id>/README.md` when
reading one experiment.

The default smoke dataset has only four text prompts, so it is useful for
checking the loop, not for stable performance conclusions.

## Text Input Datasets

LLM profiling needs a fixed text workload. Generate a larger dataset before
comparing parameters:

```bash
python3 scripts/make_benchmark_dataset.py --scenario mixed --count 100
```

Run the same loop against that dataset:

```bash
TRTLLM_BENCH_DATASET=/workspace/configs/benchmarks/generated/mixed_100.jsonl \
  TRTLLM_PROFILE_REQUESTS=100 \
  TRTLLM_PROFILE_CONCURRENCIES="1 2 4 8" \
  bash scripts/profile_trtllm_loop.sh
```

Available scenarios:

```text
short_chat    short prompts, short outputs
long_output   short/medium prompts, long generated answers
long_context  repeated long prompts, shorter outputs
mixed         mixed input/output pressure
```

## First Variables To Practice

Change only one dimension at a time.

### Runtime-only

These do not require rebuilding the engine:

```bash
TRTLLM_PROFILE_CONCURRENCIES="1 2 4 8" bash scripts/profile_trtllm_loop.sh
TRTLLM_KV_CACHE_FREE_GPU_MEMORY_FRACTION=0.70 bash scripts/profile_trtllm_loop.sh
```

### Build-time

These require rebuilding the engine:

```bash
TRTLLM_MAX_BATCH_SIZE=4 TRTLLM_MAX_SEQ_LEN=4096 TRTLLM_MAX_NUM_TOKENS=4096 \
  bash scripts/trtllm.sh bench-build

bash scripts/profile_trtllm_loop.sh
```

## Read The Result

Start from these columns in `results/tables/trtllm_profile_*.csv`:

```text
avg_request_latency_ms
avg_ttft_ms
avg_tpot_ms
request_throughput_req_s
system_output_throughput_tok_s
avg_concurrency
max_batch_size
max_sequence_length
```

Mental mapping:

```text
TTFT high       -> prefill or request setup is expensive
TPOT high       -> decode step is slow
throughput flat -> GPU may already be saturated or batching is ineffective
latency spikes  -> concurrency, KV cache, or memory pressure may be too high
```

## Manual Summary

To summarize existing benchmark JSON files:

```bash
python3 scripts/summarize_trtllm_json.py \
  --input 'results/raw/latency_*.json' 'results/raw/throughput_*.json'
```
