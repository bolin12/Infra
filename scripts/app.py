#!/usr/bin/env python3
"""LLM Infra Lab — 本地推理实验面板

Usage:
    uv run scripts/app.py                  # 启动 Web UI (默认端口 7860)
    uv run scripts/app.py --port 8080      # 指定端口
    uv run scripts/app.py --share          # 生成公网链接（临时）
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from llama_cpp import Llama

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── 模型清单 ──────────────────────────────────────────────

MODELS = {}
for name, path in [
    ("Qwen3.5-0.8B (Q8_0)", MODEL_DIR / "qwen35-0_8B" / "Qwen_Qwen3.5-0.8B-Q8_0.gguf"),
    ("Qwen3.5-2B (Q4_K_M)", MODEL_DIR / "qwen35-2B" / "Qwen_Qwen3.5-2B-Q4_K_M.gguf"),
    ("Qwen3.5-4B (Q4_K_M)", MODEL_DIR / "qwen35-4B" / "Qwen_Qwen3.5-4B-Q4_K_M.gguf"),
    ("Qwen3.5-9B (Q4_K_M)", MODEL_DIR / "qwen35-9B" / "Qwen_Qwen3.5-9B-Q4_K_M.gguf"),
    ("Qwen3-8B (Q4_K_M)",   MODEL_DIR / "qwen3-8B" / "Qwen_Qwen3-8B-Q4_K_M.gguf"),
]:
    if path.exists():
        size_mb = path.stat().st_size / 1024**2
        MODELS[name] = {"path": str(path), "size_mb": size_mb}

# 只保留存在的模型
MODEL_CHOICES = list(MODELS.keys())
DEFAULT_MODEL = MODEL_CHOICES[0]

# ── 全局 LLM 实例 ──────────────────────────────────────────

_active_llm: dict = {"instance": None, "model_name": None}


def get_llm(model_name: str, n_ctx: int = 4096):
    if _active_llm["model_name"] != model_name or _active_llm["instance"] is None:
        if _active_llm["instance"]:
            del _active_llm["instance"]
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        info = MODELS[model_name]
        _active_llm["instance"] = Llama(
            model_path=info["path"],
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            chat_format="chatml",
            verbose=False,
        )
        _active_llm["model_name"] = model_name
    return _active_llm["instance"]


# ── Tab 1: Chat ───────────────────────────────────────────

def chat_fn(message, history, model_name, temperature, max_tokens):
    if not message.strip():
        return "", history

    llm = get_llm(model_name)
    messages = []
    for h in history or []:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    partial = ""
    for chunk in response:
        token = chunk["choices"][0]["delta"].get("content", "")
        partial += token
        yield "", history + [(message, partial)]


def build_chat_tab():
    with gr.Column():
        gr.Markdown("## 💬 Chat")
        with gr.Row():
            model_dd = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label="Model")
            temp_sl = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
            max_tok = gr.Slider(64, 2048, value=512, step=64, label="Max Tokens")
        chatbot = gr.Chatbot(height=500)
        msg_box = gr.Textbox(placeholder="Type your message here...", label="Message")

    msg_box.submit(chat_fn, [msg_box, chatbot, model_dd, temp_sl, max_tok], [msg_box, chatbot])


# ── Tab 2: Benchmark ───────────────────────────────────────

def benchmark_fn(model_name, prompt_text, ctx_len, max_tok, reps):
    """Run a quick benchmark and return formatted results."""
    llm = get_llm(model_name, n_ctx=int(ctx_len))
    messages = [{"role": "user", "content": prompt_text}]

    results = []
    for i in range(int(reps)):
        t0 = time.time()
        resp = llm.create_chat_completion(
            messages=messages,
            max_tokens=int(max_tok),
            temperature=0.0,  # deterministic for benchmarking
            stream=False,
        )
        elapsed = time.time() - t0
        tokens = resp["usage"]["completion_tokens"]
        tps = tokens / elapsed
        vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        results.append({
            "run": i + 1,
            "tokens": tokens,
            "time_s": round(elapsed, 2),
            "tokens_per_sec": round(tps, 1),
            "vram_gb": round(vram, 3),
        })

    # Save to CSV
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / "raw" / f"bench_{stamp}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "tokens", "time_s", "tokens_per_sec", "vram_gb"])
        w.writeheader()
        w.writerows(results)

    # Format output
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    header = f"| Run | Tokens | Time | Tokens/s | VRAM |\n|-----|--------|------|----------|------|\n"
    rows = "\n".join(
        f"| {r['run']} | {r['tokens']} | {r['time_s']}s | **{r['tokens_per_sec']}** | {r['vram_gb']} GB |"
        for r in results
    )
    summary = f"\n\n**Average: {avg_tps:.1f} tokens/s**  |  Saved: `{csv_path.name}`"
    return header + rows + summary


def build_benchmark_tab():
    with gr.Column():
        gr.Markdown("## 📊 Benchmark")
        with gr.Row():
            model_dd = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label="Model")
            ctx_sl = gr.Slider(256, 8192, value=2048, step=256, label="Context Length")
        with gr.Row():
            max_tok = gr.Slider(32, 1024, value=256, step=32, label="Max Tokens")
            reps_sl = gr.Slider(1, 5, value=3, step=1, label="Repetitions")
        prompt_box = gr.Textbox(
            value="请用中文简要介绍深度学习中的注意力机制。",
            lines=3, label="Prompt"
        )
        btn = gr.Button("▶ Run Benchmark", variant="primary")
        result_md = gr.Markdown("")

    btn.click(benchmark_fn, [model_dd, prompt_box, ctx_sl, max_tok, reps_sl], [result_md])


# ── Tab 3: Results ─────────────────────────────────────────

def list_results():
    raw_dir = RESULTS_DIR / "raw"
    if not raw_dir.exists():
        return "No benchmark results yet."
    csv_files = sorted(raw_dir.glob("*.csv"), reverse=True)
    if not csv_files:
        return "No benchmark results yet."
    lines = ["| File | Rows | Created |", "|------|------|---------|"]
    for f in csv_files[:20]:
        with open(f) as fh:
            n = sum(1 for _ in fh) - 1  # minus header
        ts = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"| `{f.name}` | {n} | {ts} |")
    return "\n".join(lines)


def build_results_tab():
    with gr.Column():
        gr.Markdown("## 📈 Results")
        btn = gr.Button("🔄 Refresh")
        table_md = gr.Markdown("")
        btn.click(list_results, [], [table_md])
        # Load on tab switch
        table_md.value = list_results()


# ── Tab 4: Models ──────────────────────────────────────────

def model_inventory():
    lines = ["| Model | Size | VRAM Est. | Path |", "|-------|------|-----------|------|"]
    for name, info in MODELS.items():
        size = f"{info['size_mb']:.0f} MB" if info["size_mb"] < 1024 else f"{info['size_mb']/1024:.1f} GB"
        lines.append(f"| {name} | {size} | — | `{info['path']}` |")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    total = sum(m["size_mb"] for m in MODELS.values())
    total_str = f"{total:.0f} MB" if total < 1024 else f"{total/1024:.1f} GB"
    return f"**GPU:** {gpu}  |  **Total models:** {total_str}\n\n" + "\n".join(lines)


def build_models_tab():
    with gr.Column():
        gr.Markdown("## 📦 Models")
        md = gr.Markdown(model_inventory())
        btn = gr.Button("🔄 Refresh")
        btn.click(model_inventory, [], [md])


# ── System Info ────────────────────────────────────────────

def system_info():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    vram = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "N/A"
    return f"**GPU:** {gpu} ({vram})  |  **PyTorch:** {torch.__version__}  |  **CUDA:** {torch.version.cuda}"


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Infra Lab Web UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Generate public link")
    args = parser.parse_args()

    with gr.Blocks(title="LLM Infra Lab") as app:
        gr.Markdown("# 🧠 LLM Inference Infra Lab")
        gr.Markdown(system_info())

        with gr.Tabs():
            with gr.TabItem("💬 Chat"):
                build_chat_tab()
            with gr.TabItem("📊 Benchmark"):
                build_benchmark_tab()
            with gr.TabItem("📈 Results"):
                build_results_tab()
            with gr.TabItem("📦 Models"):
                build_models_tab()

    print(f"\n🚀 Starting at http://localhost:{args.port}")
    app.queue().launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
