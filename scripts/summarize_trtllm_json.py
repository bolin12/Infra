#!/usr/bin/env python3
"""Summarize TensorRT-LLM benchmark JSON reports into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from statistics import mean
from typing import Any


FIELDS = [
    "file",
    "kind",
    "model",
    "backend",
    "dtype",
    "kv_cache_dtype",
    "quantization",
    "tp_size",
    "pp_size",
    "world_size",
    "max_batch_size",
    "max_num_tokens",
    "max_input_length",
    "max_sequence_length",
    "kv_cache_percentage",
    "num_requests",
    "avg_concurrency",
    "avg_input_length",
    "avg_output_length",
    "min_input_length",
    "max_input_length_dataset",
    "p50_input_length",
    "p90_input_length",
    "min_output_length",
    "max_output_length",
    "p50_output_length",
    "p90_output_length",
    "avg_sequence_length",
    "max_sequence_length_dataset",
    "total_latency_ms",
    "avg_request_latency_ms",
    "request_latency_p50_ms",
    "request_latency_p90_ms",
    "request_latency_p95_ms",
    "request_latency_p99_ms",
    "request_latency_min_ms",
    "request_latency_max_ms",
    "request_throughput_req_s",
    "system_output_throughput_tok_s",
    "system_total_throughput_tok_s",
    "output_throughput_per_user_tok_s",
    "output_throughput_per_gpu_tok_s",
    "avg_ttft_ms",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "avg_tpot_ms",
    "tpot_p50_ms",
    "tpot_p90_ms",
    "tpot_p95_ms",
    "tpot_p99_ms",
    "token_output_speed_tok_s",
    "gen_tps_p50",
    "gen_tps_p90",
    "engine_dir",
    "engine_size_mib",
    "dataset_path",
]


def nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def fmt(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return value


def report_kind(data: dict[str, Any], path: Path) -> str:
    if "streaming_metrics" in data:
        return "latency"
    if "performance" in data:
        return "throughput"
    return path.stem


def maybe_engine_size_mib(engine_dir: Any, project_root: Path) -> float | None:
    if not isinstance(engine_dir, str):
        return None
    candidates = []
    if engine_dir.startswith("/engines/"):
        candidates.append(project_root / engine_dir.removeprefix("/"))
    candidates.append(Path(engine_dir))
    for directory in candidates:
        engine = directory / "rank0.engine"
        if engine.exists():
            return engine.stat().st_size / (1024 * 1024)
    return None


def read_report(path: Path, project_root: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or "performance" not in data:
        return None

    engine_dir = nested(data, "engine", "engine_dir")
    latency = nested(data, "performance", "request_latency_percentiles_ms") or {}
    ttft = nested(data, "streaming_metrics", "ttft_percentiles") or {}
    tpot = nested(data, "streaming_metrics", "tpot_percentiles") or {}
    gen_tps = nested(data, "streaming_metrics", "gen_tps_percentiles") or {}
    isl = nested(data, "dataset", "isl_stats") or {}
    osl = nested(data, "dataset", "osl_stats") or {}

    return {
        "file": str(path),
        "kind": report_kind(data, path),
        "model": nested(data, "engine", "model"),
        "backend": nested(data, "engine", "backend"),
        "dtype": nested(data, "engine", "dtype"),
        "kv_cache_dtype": nested(data, "engine", "kv_cache_dtype"),
        "quantization": nested(data, "engine", "quantization"),
        "tp_size": nested(data, "world_info", "tp_size"),
        "pp_size": nested(data, "world_info", "pp_size"),
        "world_size": nested(data, "world_info", "world_size"),
        "max_batch_size": nested(data, "world_info", "max_batch_size"),
        "max_num_tokens": nested(data, "world_info", "max_num_tokens"),
        "max_input_length": nested(data, "engine", "max_input_length"),
        "max_sequence_length": nested(data, "engine", "max_sequence_length"),
        "kv_cache_percentage": nested(data, "world_info", "kv_cache_percentage"),
        "num_requests": nested(data, "request_info", "num_requests"),
        "avg_concurrency": nested(data, "request_info", "avg_num_concurrent_requests"),
        "avg_input_length": nested(data, "request_info", "avg_input_length"),
        "avg_output_length": nested(data, "request_info", "avg_output_length"),
        "min_input_length": isl.get("minimum"),
        "max_input_length_dataset": isl.get("maximum"),
        "p50_input_length": isl.get("p50"),
        "p90_input_length": isl.get("p90"),
        "min_output_length": osl.get("minimum"),
        "max_output_length": osl.get("maximum"),
        "p50_output_length": osl.get("p50"),
        "p90_output_length": osl.get("p90"),
        "avg_sequence_length": nested(data, "dataset", "seq_len_stats", "average"),
        "max_sequence_length_dataset": nested(data, "dataset", "max_sequence_length"),
        "total_latency_ms": nested(data, "performance", "total_latency_ms"),
        "avg_request_latency_ms": nested(data, "performance", "avg_request_latency_ms"),
        "request_latency_p50_ms": latency.get("p50"),
        "request_latency_p90_ms": latency.get("p90"),
        "request_latency_p95_ms": latency.get("p95"),
        "request_latency_p99_ms": latency.get("p99"),
        "request_latency_min_ms": latency.get("minimum"),
        "request_latency_max_ms": latency.get("maximum"),
        "request_throughput_req_s": nested(data, "performance", "request_throughput_req_s"),
        "system_output_throughput_tok_s": nested(
            data, "performance", "system_output_throughput_tok_s"
        ),
        "system_total_throughput_tok_s": nested(
            data, "performance", "system_total_throughput_tok_s"
        ),
        "output_throughput_per_user_tok_s": nested(
            data, "performance", "output_throughput_per_user_tok_s"
        ),
        "output_throughput_per_gpu_tok_s": nested(
            data, "performance", "output_throughput_per_gpu_tok_s"
        ),
        "avg_ttft_ms": nested(data, "streaming_metrics", "avg_ttft_ms"),
        "ttft_p50_ms": ttft.get("p50"),
        "ttft_p90_ms": ttft.get("p90"),
        "ttft_p95_ms": ttft.get("p95"),
        "ttft_p99_ms": ttft.get("p99"),
        "avg_tpot_ms": nested(data, "streaming_metrics", "avg_tpot_ms"),
        "tpot_p50_ms": tpot.get("p50"),
        "tpot_p90_ms": tpot.get("p90"),
        "tpot_p95_ms": tpot.get("p95"),
        "tpot_p99_ms": tpot.get("p99"),
        "token_output_speed_tok_s": nested(
            data, "streaming_metrics", "token_output_speed_tok_s"
        ),
        "gen_tps_p50": gen_tps.get("p50"),
        "gen_tps_p90": gen_tps.get("p90"),
        "engine_dir": engine_dir,
        "engine_size_mib": maybe_engine_size_mib(engine_dir, project_root),
        "dataset_path": nested(data, "dataset", "dataset_path"),
    }


def expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        matches = glob.glob(item)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(item))
    return sorted(set(paths))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in FIELDS})


def numeric_cell(row: dict[str, Any], key: str) -> str:
    return str(fmt(row.get(key)))


def gpu_summary(path: Path | None) -> dict[str, dict[str, float]] | None:
    if path is None or not path.exists():
        return None

    def first_number(text: str) -> float | None:
        first = text.strip().split()[0] if text.strip() else ""
        if first in {"[N/A]", "N/A", ""}:
            return None
        try:
            return float(first)
        except ValueError:
            return None

    columns = {
        "utilization.gpu [%]": "gpu_util_pct",
        "utilization.memory [%]": "mem_util_pct",
        "memory.used [MiB]": "memory_used_mib",
        "memory.total [MiB]": "memory_total_mib",
        "power.draw [W]": "power_w",
        "temperature.gpu": "temperature_c",
    }
    samples: dict[str, list[float]] = {value: [] for value in columns.values()}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        field_map = {field.strip(): field for field in reader.fieldnames or []}
        for row in reader:
            for source, dest in columns.items():
                if source not in field_map:
                    continue
                value = first_number(row[field_map[source]])
                if value is not None:
                    samples[dest].append(value)

    result: dict[str, dict[str, float]] = {}
    for key, values in samples.items():
        if values:
            result[key] = {
                "min": min(values),
                "avg": mean(values),
                "max": max(values),
                "samples": float(len(values)),
            }
    return result


def write_gpu_section(handle: Any, gpu: dict[str, dict[str, float]] | None) -> None:
    handle.write("\n## GPU Telemetry\n\n")
    if not gpu:
        handle.write("No GPU telemetry CSV was provided.\n")
        return
    handle.write("| metric | min | avg | max | samples |\n")
    handle.write("| --- | --- | --- | --- | --- |\n")
    for metric, values in gpu.items():
        handle.write(
            "| "
            + " | ".join(
                [
                    metric,
                    fmt(values["min"]),
                    fmt(values["avg"]),
                    fmt(values["max"]),
                    str(int(values["samples"])),
                ]
            )
            + " |\n"
        )


def write_markdown(rows: list[dict[str, Any]], path: Path, gpu_csv: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overview = [
        "kind",
        "avg_concurrency",
        "num_requests",
        "avg_input_length",
        "avg_output_length",
        "avg_request_latency_ms",
        "avg_ttft_ms",
        "avg_tpot_ms",
        "request_throughput_req_s",
        "system_output_throughput_tok_s",
    ]
    latency_cols = [
        "kind",
        "avg_concurrency",
        "request_latency_p50_ms",
        "request_latency_p90_ms",
        "request_latency_p95_ms",
        "request_latency_p99_ms",
        "request_latency_min_ms",
        "request_latency_max_ms",
    ]
    streaming_cols = [
        "kind",
        "avg_concurrency",
        "avg_ttft_ms",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "avg_tpot_ms",
        "tpot_p50_ms",
        "tpot_p90_ms",
        "tpot_p95_ms",
        "tpot_p99_ms",
        "token_output_speed_tok_s",
    ]
    dataset_cols = [
        "kind",
        "num_requests",
        "min_input_length",
        "avg_input_length",
        "max_input_length_dataset",
        "min_output_length",
        "avg_output_length",
        "max_output_length",
        "avg_sequence_length",
    ]
    engine_cols = [
        "model",
        "backend",
        "dtype",
        "quantization",
        "tp_size",
        "pp_size",
        "max_batch_size",
        "max_num_tokens",
        "max_sequence_length",
        "engine_size_mib",
    ]

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# TensorRT-LLM Profile Summary\n\n")

        handle.write("## Key Metrics\n\n")
        write_table(handle, overview, rows)

        handle.write("\n## Latency Percentiles\n\n")
        write_table(handle, latency_cols, rows)

        handle.write("\n## Streaming Percentiles\n\n")
        write_table(handle, streaming_cols, rows)

        handle.write("\n## Dataset Shape\n\n")
        write_table(handle, dataset_cols, rows)

        handle.write("\n## Engine And Runtime\n\n")
        write_table(handle, engine_cols, rows[:1])

        write_gpu_section(handle, gpu_summary(gpu_csv))

        handle.write("\n## Metric Guide\n\n")
        handle.write("- `avg_concurrency`: benchmark-observed active request concurrency.\n")
        handle.write("- `avg_request_latency_ms`: average end-to-end request time.\n")
        handle.write("- `avg_ttft_ms`: time to first token; mainly reflects prefill and scheduling.\n")
        handle.write("- `avg_tpot_ms`: time per output token; mainly reflects decode speed.\n")
        handle.write("- `request_throughput_req_s`: completed requests per second.\n")
        handle.write("- `system_output_throughput_tok_s`: generated output tokens per second.\n")
        handle.write("- `avg_input_length` / `avg_output_length`: workload shape; compare runs only when these are controlled.\n")
        handle.write("- `engine_size_mib`: local `rank0.engine` size when it can be resolved.\n")

        handle.write("\n## Source Files\n\n")
        for row in rows:
            handle.write(f"- `{row['file']}`\n")


def write_table(handle: Any, columns: list[str], rows: list[dict[str, Any]]) -> None:
    handle.write("| " + " | ".join(columns) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        handle.write("| " + " | ".join(numeric_cell(row, column) for column in columns) + " |\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        default=["results/raw/*.json"],
        help="JSON report files or glob patterns.",
    )
    parser.add_argument("--csv", default="results/tables/trtllm_profile_summary.csv")
    parser.add_argument("--markdown", default="results/reports/trtllm_profile_summary.md")
    parser.add_argument("--gpu-csv", type=Path, help="Optional nvidia-smi CSV from this run.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    rows = []
    for path in expand_inputs(args.input):
        if not path.exists():
            continue
        row = read_report(path, args.project_root)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("No TensorRT-LLM benchmark reports found.")

    write_csv(rows, Path(args.csv))
    write_markdown(rows, Path(args.markdown), args.gpu_csv)
    print(f"wrote {args.csv}")
    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
