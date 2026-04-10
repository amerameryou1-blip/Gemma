#!/usr/bin/env python3
"""
launcher.py — Downloads all pipeline files from GitHub, then runs them.

This script:
  1. Downloads every .py file from the GitHub repo raw URLs
  2. Saves them to the current working directory
  3. Runs the full pipeline in one Python process

Usage on Kaggle:
  python launcher.py --prompt "Your prompt here"

Files are fetched from:
  https://raw.githubusercontent.com/amerameryou1-blip/Gemma/refs/heads/main/
"""
import argparse
import os
import sys
import time
import traceback
import urllib.request

# ── Configuration ──────────────────────────────────────────────────
REPO = "amerameryou1-blip/Gemma"
BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/refs/heads/{BRANCH}"

# Order matters: setup first, then tpu_init, tokenizer, model_loader, inference
FILES = [
    "setup.py",
    "tpu_init.py",
    "tokenizer_setup.py",
    "model_loader.py",
    "inference.py",
]

DEFAULT_MODEL_DIR = "/kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1"


def download_file(filename: str, dest_dir: str = ".") -> str:
    """Download a single .py file from GitHub raw URL and save to disk."""
    url = f"{RAW_BASE}/{filename}"
    dest = os.path.join(dest_dir, filename)
    print(f"[launcher] Downloading {filename} from {url} ...")
    try:
        req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        size_kb = len(data) / 1024
        print(f"[launcher]   ✓ {filename} saved ({size_kb:.1f} KB)")
        return dest
    except Exception as e:
        print(f"[launcher]   ✗ Failed to download {filename}: {e}")
        raise


def download_all(dest_dir: str = ".") -> list:
    """Download all pipeline files from GitHub."""
    print("=" * 60)
    print("  Downloading pipeline files from GitHub")
    print(f"  Repo: {REPO}  Branch: {BRANCH}")
    print("=" * 60)
    downloaded = []
    for fname in FILES:
        path = download_file(fname, dest_dir)
        downloaded.append(path)
    print(f"[launcher] All {len(downloaded)} files downloaded successfully.")
    print()
    return downloaded


def run_setup():
    """Step 1: Install dependencies."""
    print(">>> STEP 1/5: Installing dependencies")
    print()
    sys.path.insert(0, ".")
    if "setup" in sys.modules:
        del sys.modules["setup"]
    import setup
    setup.main()
    print()


def run_tpu_init():
    """Step 2: Detect and initialize TPU."""
    print(">>> STEP 2/5: Initializing TPU")
    print()
    if "tpu_init" in sys.modules:
        del sys.modules["tpu_init"]
    from tpu_init import init_tpu, get_mesh
    device_count = init_tpu()
    mesh = get_mesh(device_count)
    print()
    return device_count, mesh


def run_tokenizer(model_dir: str):
    """Step 3: Load tokenizer."""
    print(">>> STEP 3/5: Loading tokenizer")
    print()
    if "tokenizer_setup" in sys.modules:
        del sys.modules["tokenizer_setup"]
    from tokenizer_setup import load_tokenizer
    tokenizer = load_tokenizer(model_dir)
    print()
    return tokenizer


def run_model_loader(model_dir: str):
    """Step 4: Load Flax model in BF16."""
    print(">>> STEP 4/5: Loading model (BF16)")
    print()
    if "model_loader" in sys.modules:
        del sys.modules["model_loader"]
    from model_loader import load_model
    model, params = load_model(model_dir)
    print()
    return model, params


def run_inference(model, params, tokenizer, prompt, max_length, temperature, top_k):
    """Step 5: Warmup + generate text."""
    print(">>> STEP 5/5: Running inference")
    print()
    if "inference" in sys.modules:
        del sys.modules["inference"]
    from inference import warmup, generate

    # Warmup triggers XLA compilation so the real prompt doesn't timeout
    warmup(model, params, tokenizer)
    print()

    result = generate(
        model=model,
        params=params,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gemma 4 31B — Full BF16 inference on Kaggle TPU v5e-8"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Path to pre-downloaded Flax model weights",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantum computing in simple terms.",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50)",
    )
    parser.add_argument(
        "--skip_setup",
        action="store_true",
        help="Skip dependency installation (use if already installed)",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip downloading files from GitHub (use if files already exist locally)",
    )
    args = parser.parse_args()

    overall_start = time.time()

    print()
    print("=" * 60)
    print("  Gemma 4 31B — Kaggle TPU v5e-8 Launcher")
    print("  Full BF16 | Zero Quantization | End-to-End")
    print(f"  Model: {args.model_dir}")
    print("=" * 60)
    print()

    try:
        # ── Step 0: Download files from GitHub ─────────────────────
        if not args.skip_download:
            download_all()
        else:
            print("[launcher] Skipping download (--skip_download)")
            print()

        # ── Step 1: Setup ─────────────────────────────────────────
        if not args.skip_setup:
            run_setup()
        else:
            print(">>> STEP 1/5: Skipping setup (--skip_setup)")
            print()

        # ── Step 2: TPU Init ──────────────────────────────────────
        run_tpu_init()

        # ── Step 3: Tokenizer ─────────────────────────────────────
        tokenizer = run_tokenizer(args.model_dir)

        # ── Step 4: Model ─────────────────────────────────────────
        model, params = run_model_loader(args.model_dir)

        # ── Step 5: Inference ─────────────────────────────────────
        result = run_inference(
            model=model,
            params=params,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        # ── Summary ───────────────────────────────────────────────
        total_time = time.time() - overall_start
        print()
        print("=" * 60)
        print(f"Total runtime: {total_time:.1f}s")
        print("Status: SUCCESS")
        print("=" * 60)

        return result

    except Exception as e:
        # Print the FULL traceback so the user can see exactly what failed
        print()
        print("=" * 60)
        print("FATAL ERROR — Full traceback:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
