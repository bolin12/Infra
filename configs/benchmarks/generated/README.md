# Generated Benchmark Datasets

Use `scripts/make_benchmark_dataset.py` to create larger JSONL files for
TensorRT-LLM benchmark runs.

Examples:

```bash
python3 scripts/make_benchmark_dataset.py --scenario short_chat --count 100
python3 scripts/make_benchmark_dataset.py --scenario long_output --count 100
python3 scripts/make_benchmark_dataset.py --scenario long_context --count 100
python3 scripts/make_benchmark_dataset.py --scenario mixed --count 100
```

Then point TensorRT-LLM benchmark at the generated file:

```bash
TRTLLM_BENCH_DATASET=/workspace/configs/benchmarks/generated/mixed_100.jsonl \
  TRTLLM_PROFILE_REQUESTS=100 \
  TRTLLM_PROFILE_CONCURRENCIES="1 2 4 8" \
  bash scripts/profile_trtllm_loop.sh
```

Generated JSONL files are ignored by Git.
