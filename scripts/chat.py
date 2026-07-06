#!/usr/bin/env python3
"""命令行 LLM 对话 — 用本地 GGUF 模型聊天

Usage:
    uv run scripts/chat.py                              # 默认 4B
    uv run scripts/chat.py --model 0.8B                 # 快速对话
    uv run scripts/chat.py --model 9B                   # 大模型
    uv run scripts/chat.py --temperature 0.3            # 更确定
"""

import argparse
import os
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

MODELS = {
    "0.8B": str(MODEL_DIR / "qwen35-0_8B" / "Qwen_Qwen3.5-0.8B-Q8_0.gguf"),
    "2B":   str(MODEL_DIR / "qwen35-2B" / "Qwen_Qwen3.5-2B-Q4_K_M.gguf"),
    "4B":   str(MODEL_DIR / "qwen35-4B" / "Qwen_Qwen3.5-4B-Q4_K_M.gguf"),
    "9B":   str(MODEL_DIR / "qwen35-9B" / "Qwen_Qwen3.5-9B-Q4_K_M.gguf"),
    "8B":   str(MODEL_DIR / "qwen3-8B" / "Qwen_Qwen3-8B-Q4_K_M.gguf"),
}


def main():
    parser = argparse.ArgumentParser(description="Chat with local GGUF models")
    parser.add_argument("--model", default="4B", choices=list(MODELS.keys()),
                        help="Model to use (default: 4B)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--context", type=int, default=4096,
                        help="Context window size")
    args = parser.parse_args()

    model_path = MODELS[args.model]
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("  Run: uv run scripts/download_models.py")
        sys.exit(1)

    print(f"Loading {args.model} model...")
    t0 = time.time()

    from llama_cpp import Llama
    llm = Llama(
        model_path=model_path,
        n_ctx=args.context,
        n_gpu_layers=-1,
        chat_format="chatml",
        verbose=False,
    )
    print(f"✓ Loaded in {time.time() - t0:.1f}s  (VRAM: ~{MODELS[args.model].split('/')[-2]} ready)")
    print("Type /quit to exit, /clear to reset context.\n")

    messages = []
    while True:
        try:
            user = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user:
            continue
        if user.lower() in ("/quit", "/exit", "/q"):
            print("Bye!")
            break
        if user.lower() == "/clear":
            messages = []
            print("Context cleared.\n")
            continue

        messages.append({"role": "user", "content": user})

        t0 = time.time()
        print("AI  > ", end="", flush=True)

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True,
        )

        full = ""
        for chunk in response:
            token = chunk["choices"][0]["delta"].get("content", "")
            print(token, end="", flush=True)
            full += token
        print(f"\n  ({len(full)} chars, {time.time() - t0:.1f}s)\n")

        messages.append({"role": "assistant", "content": full})


if __name__ == "__main__":
    main()
