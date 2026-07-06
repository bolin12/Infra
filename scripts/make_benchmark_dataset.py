#!/usr/bin/env python3
"""Create JSONL datasets for TensorRT-LLM text-generation benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT_DIR = Path("prompts")
DEFAULT_OUTPUT_DIR = Path("configs/benchmarks/generated")

LONG_CONTEXT_PASSAGES = [
    """TensorRT-LLM profiling should separate prefill and decode. Prefill reads
the full prompt, builds attention state, and fills KV cache. Decode generates
one token at a time and repeatedly reads model weights plus cached keys and
values. A short chat workload can hide prefill cost, while a long document
summary workload can make TTFT dominate the user experience.""",
    """A production inference report should record model id, checkpoint
revision, engine path, TensorRT-LLM version, GPU model, driver version, maximum
sequence length, maximum batch size, KV cache fraction, benchmark dataset,
concurrency, warmup count, request count, sampling parameters, and raw JSON
outputs. Without these fields, two benchmark numbers are rarely comparable.""",
    """When concurrency increases, LLM serving systems can batch more decode
steps together. This often raises total output token throughput, but it may also
increase request latency and KV cache pressure. The useful operating point is
usually before throughput flattens and before p95 or p99 latency becomes
unacceptable.""",
    """Quantization reduces weight precision and can reduce memory bandwidth
pressure. It may improve tokens per second and reduce engine size, but it can
also change output quality. A quantized model must be compared against the same
text workload and, ideally, a small quality checklist in addition to raw speed
metrics.""",
]

CODE_PROMPTS = [
    """补全下面的 Python 函数，要求处理异常输入，并给出简短注释：

def batch_tokenize(texts, tokenizer, max_length):
    \"\"\"Tokenize a list of strings for LLM inference.\"\"\"""",
    """写一个 Bash 函数 `latest_profile_report`，返回 results/runs 里最新一轮
profiling 的 README.md 路径，并处理目录不存在的情况。""",
    """给出一个 Python 片段，读取 TensorRT-LLM benchmark JSON，打印 TTFT、TPOT、
request throughput 和 output token throughput。""",
]

SHORT_CHAT_PROMPTS = [
    "用一句中文说明 TensorRT-LLM engine 和 HuggingFace checkpoint 的区别。",
    "解释 TTFT 和 TPOT 的区别，要求简短。",
    "请列出本地单卡推理服务需要固定记录的三个关键参数。",
    "用三点说明为什么 benchmark 必须固定文本输入。",
]


def read_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8").strip()


def repeated_text(text: str, repeat: int) -> str:
    return "\n\n".join(text for _ in range(max(1, repeat)))


def build_records(scenario: str, count: int) -> list[dict[str, object]]:
    basic = read_prompt("basic_cn_qa.txt")
    code = read_prompt("code_completion.txt")
    long_context = read_prompt("long_context_summary.txt")

    templates: list[tuple[str, int]]
    if scenario == "short_chat":
        templates = [
            (prompt, 64 + (index % 2) * 32)
            for index, prompt in enumerate([basic, *SHORT_CHAT_PROMPTS])
        ]
    elif scenario == "long_output":
        templates = [
            ("写一份 TensorRT-LLM 单卡推理 profiling 学习笔记，包含步骤、指标和判断方法。", 512),
            (code, 384),
            *[(prompt, 384) for prompt in CODE_PROMPTS],
            ("围绕 LLM 推理服务的吞吐、延迟、显存和输入长度，写一段系统化说明。", 512),
            (basic, 384),
        ]
    elif scenario == "long_context":
        base_context = "\n\n".join(LONG_CONTEXT_PASSAGES)
        templates = [
            (repeated_text(base_context, 4), 128),
            (repeated_text(base_context, 6), 128),
            (repeated_text(long_context, 8), 128),
            (repeated_text(code, 6), 160),
            (repeated_text(basic, 12), 160),
        ]
    elif scenario == "mixed":
        base_context = "\n\n".join(LONG_CONTEXT_PASSAGES)
        templates = [
            (basic, 64),
            (code, 256),
            (CODE_PROMPTS[0], 384),
            (repeated_text(base_context, 3), 128),
            (repeated_text(long_context, 4), 160),
            ("写一份 TensorRT-LLM profiling 实验报告，包含结论和下一步。", 512),
        ]
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    records = []
    for index in range(count):
        prompt, output_tokens = templates[index % len(templates)]
        records.append(
            {
                "task_id": index + 1,
                "prompt": prompt,
                "output_tokens": output_tokens,
            }
        )
    return records


def write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["short_chat", "long_output", "long_context", "mixed"],
        default="mixed",
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or DEFAULT_OUTPUT_DIR / f"{args.scenario}_{args.count}.jsonl"
    records = build_records(args.scenario, args.count)
    write_jsonl(records, output)
    print(f"wrote {output}")
    print(f"records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
