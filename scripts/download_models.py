#!/usr/bin/env python3
"""Download Qwen GGUF models for local inference.

Usage:
    uv run scripts/download_models.py                        # all local models
    uv run scripts/download_models.py --model 0_8B           # single model
    uv run scripts/download_models.py --source modelscope    # use ModelScope (no proxy)
    uv run scripts/download_models.py --dry-run              # list only

Defaults to hf-mirror.com (no proxy needed in China).
    --source official  to use huggingface.co directly (requires proxy).
"""

import argparse
import os
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG = PROJECT_ROOT / "configs" / "models" / "qwen.toml"


def load_models():
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)["models"]


def download_hf(key, cfg, quant):
    """Download via HuggingFace (uses HF_ENDPOINT if set, else hf-mirror.com)."""
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from huggingface_hub import hf_hub_download, list_repo_files

    repo = cfg["repo"]
    files = list_repo_files(repo)
    ggufs = [f for f in files if f.endswith(".gguf") and quant in f]
    if not ggufs:
        print(f"  ✗ {quant} not found in {repo}")
        return

    filename = ggufs[0]
    dest = MODEL_DIR / key
    dest.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=dest,
        local_dir_use_symlinks=False,
        # tqdm progress bar is built into hf_hub_download
    )
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"  ✓ {size_mb:.0f} MiB → {path}")


def download_ms(key, cfg, quant):
    """Download via ModelScope (no proxy needed, faster in China)."""
    from modelscope.hub.file_download import model_file_download

    repo = cfg["repo"].replace("bartowski/", "")  # strip HF org
    # ModelScope uses different orgs; try a few known ones
    candidates = [
        f"llm-mirror/{repo}",           # community GGUF mirror
        f"Qwen/{repo.replace('Qwen_', '')}",  # Qwen official
        cfg["repo"],                     # as-is (may work)
    ]

    for ms_repo in candidates:
        try:
            dest = MODEL_DIR / key
            dest.mkdir(parents=True, exist_ok=True)
            path = model_file_download(
                model_id=ms_repo,
                file_path=None,  # will list files
                local_dir=str(dest),
                progress=True,
            )
            if path:
                size_mb = Path(path).stat().st_size / 1024**2
                print(f"  ✓ {size_mb:.0f} MiB → {path}  (via ModelScope)")
                return
        except Exception:
            continue

    print(f"  ✗ ModelScope mirror not found for {repo}, falling back to HF")
    download_hf(key, cfg, quant)


def download(key, cfg, source, quant_override=None, dry_run=False):
    quant = quant_override or cfg["quant"]
    print(f"\n{'[DRY RUN] ' if dry_run else ''}{key}  "
          f"({cfg['size']} × {quant}  → ~{cfg['vram']} VRAM, ~{cfg['disk']} disk)")
    print(f"  repo:   {cfg['repo']}")
    print(f"  source: {source}")

    if dry_run:
        return

    if source == "modelscope":
        download_ms(key, cfg, quant)
    else:
        download_hf(key, cfg, quant)


def main():
    parser = argparse.ArgumentParser(description="Download Qwen GGUF models")
    parser.add_argument("--model", help="Model key: 0_8B, 2B, 4B, 9B, 8B, 27B")
    parser.add_argument("--quant", help="Override quant (Q4_K_M, Q8_0)")
    parser.add_argument("--source", default="huggingface",
                        choices=["huggingface", "modelscope"],
                        help="huggingface (needs proxy) or modelscope (direct)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all", action="store_true", help="Include cloud models (27B+)")
    args = parser.parse_args()

    all_models = load_models()
    local_max = 5  # priority > 5 = cloud-only

    models_to_dl = {}
    if args.model:
        needle = args.model.replace(".", "_")
        for k, v in all_models.items():
            if needle in k or v.get("size") == args.model:
                models_to_dl[k] = v
                break
        if not models_to_dl:
            print(f"Unknown model: {args.model}")
            print(f"Available: {sorted(all_models.keys())}")
            print(f"Sizes: {sorted(set(v['size'] for v in all_models.values()))}")
            return
    else:
        models_to_dl = {
            k: v for k, v in all_models.items()
            if v["priority"] <= local_max or args.all
        }
        models_to_dl = dict(sorted(models_to_dl.items(), key=lambda x: x[1]["priority"]))

    total = sum(
        float(v["disk"].replace(" GB", "")) for v in models_to_dl.values()
    )
    print(f"📦 {len(models_to_dl)} models, ~{total:.1f} GB total, source={args.source}")

    for key, cfg in models_to_dl.items():
        download(key, cfg, args.source, args.quant, args.dry_run)

    if not args.dry_run:
        print(f"\n✅ Done. Models in {MODEL_DIR}/")


if __name__ == "__main__":
    main()
