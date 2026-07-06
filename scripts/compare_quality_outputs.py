#!/usr/bin/env python3
"""Create a Markdown diff between two quality output JSONL files."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any


def read_outputs(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["task_id"])] = row
    return rows


def shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=3000)
    args = parser.parse_args()

    baseline = read_outputs(args.baseline)
    candidate = read_outputs(args.candidate)
    task_ids = sorted(set(baseline) | set(candidate), key=lambda value: int(value))

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    with args.markdown.open("w", encoding="utf-8") as handle:
        handle.write("# Quality Output Diff\n\n")
        handle.write(f"- baseline: `{args.baseline}`\n")
        handle.write(f"- candidate: `{args.candidate}`\n")
        handle.write("- verdict: `pass / warn / fail`\n\n")
        handle.write("## Review Checklist\n\n")
        handle.write("- [ ] candidate answers the same question\n")
        handle.write("- [ ] no obvious factual regression\n")
        handle.write("- [ ] no format regression\n")
        handle.write("- [ ] no unwanted repetition or premature stop\n")
        handle.write("- [ ] speed gain justifies any minor quality change\n\n")
        for task_id in task_ids:
            base = baseline.get(task_id, {})
            cand = candidate.get(task_id, {})
            prompt = base.get("prompt") or cand.get("prompt") or ""
            base_output = shorten(base.get("output", ""), args.max_chars)
            cand_output = shorten(cand.get("output", ""), args.max_chars)
            diff = difflib.unified_diff(
                base_output.splitlines(),
                cand_output.splitlines(),
                fromfile="baseline",
                tofile="candidate",
                lineterm="",
            )
            handle.write(f"## Case {task_id}\n\n")
            handle.write("- verdict: `pass / warn / fail`\n")
            handle.write("- notes:\n\n")
            handle.write("<details><summary>Prompt</summary>\n\n")
            handle.write(prompt + "\n\n")
            handle.write("</details>\n\n")
            handle.write("```diff\n")
            handle.write("\n".join(diff))
            handle.write("\n```\n\n")

    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
