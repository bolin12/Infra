#!/usr/bin/env python3
"""Capture model outputs from an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("configs/benchmarks/cases/mixed.jsonl")


def read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def post_chat(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            elapsed_ms = (time.perf_counter() - started) * 1000
            data = json.loads(body)
            return {"ok": True, "elapsed_ms": elapsed_ms, "response": data}
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "elapsed_ms": elapsed_ms, "error": body, "status": exc.code}
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {"ok": False, "elapsed_ms": elapsed_ms, "error": repr(exc)}


def extract_text(response: dict[str, Any]) -> str:
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def write_review_template(path: Path, output_jsonl: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Quality Review\n\n")
        handle.write(f"- outputs: `{output_jsonl.name}`\n")
        handle.write("- verdict: `pass / warn / fail`\n\n")
        handle.write("## Checklist\n\n")
        handle.write("- [ ] answers the question\n")
        handle.write("- [ ] no obvious factual regression\n")
        handle.write("- [ ] no unwanted repetition\n")
        handle.write("- [ ] no premature stop\n")
        handle.write("- [ ] expected language and format are preserved\n")
        handle.write("- [ ] code outputs look executable when applicable\n\n")
        handle.write("## Cases\n\n")
        for record in records:
            handle.write(f"### Case {record['task_id']}\n\n")
            handle.write("- verdict: `pass / warn / fail`\n")
            handle.write("- notes:\n\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    endpoint = f"http://{args.host}:{args.port}/v1/chat/completions"
    output_dir = args.output_dir or Path("results/quality") / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / f"{args.label}_outputs.jsonl"
    review_md = output_dir / f"{args.label}_review.md"
    metadata = output_dir / f"{args.label}_metadata.json"

    records = read_jsonl(args.dataset, args.limit)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            result = post_chat(
                endpoint=endpoint,
                model=args.model,
                prompt=record["prompt"],
                max_tokens=int(record.get("output_tokens", 128)),
                temperature=args.temperature,
                timeout=args.timeout,
            )
            response = result.get("response") if result["ok"] else {}
            output = extract_text(response if isinstance(response, dict) else {})
            item = {
                "task_id": record.get("task_id"),
                "label": args.label,
                "prompt": record["prompt"],
                "requested_output_tokens": record.get("output_tokens"),
                "ok": result["ok"],
                "elapsed_ms": result["elapsed_ms"],
                "output": output,
                "raw_response": response,
                "error": result.get("error"),
                "status": result.get("status"),
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"case={record.get('task_id')} ok={result['ok']} elapsed_ms={result['elapsed_ms']:.1f}")

    metadata.write_text(
        json.dumps(
            {
                "label": args.label,
                "run_id": args.run_id,
                "dataset": str(args.dataset),
                "endpoint": endpoint,
                "model": args.model,
                "temperature": args.temperature,
                "num_cases": len(records),
                "output_jsonl": str(output_jsonl),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_review_template(review_md, output_jsonl, records)
    print(f"wrote {output_jsonl}")
    print(f"wrote {metadata}")
    print(f"wrote {review_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
