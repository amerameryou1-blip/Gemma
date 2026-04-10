#!/usr/bin/env python3
"""
launcher.py — Main orchestrator for Gemma 4 31B on Kaggle TPU v5e-8.

This script runs the entire pipeline in order:
  1. setup.py        → Install dependencies
  2. tpu_init.py     → Initialize and verify TPU
  3. tokenizer_setup → Load tokenizer
  4. model_loader.py → Load Flax model in BF16
  5. inference.py    → Generate text

Usage:
  python launcher.py --prompt "Your prompt here"

All steps run in a single Python process to avoid re-loading the model.

Default model path (Kaggle pre-installed):
  /kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1
"""
import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Gemma 4 31B — Full BF16 inference on Kaggle TPU v5e-8"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1",
        help="Path to pre-downloaded Flax model weights (Kaggle default: /kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1)",
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
    args = parser.parse_args()

    overall_start = time.time()

    print()
    print("=" * 60)
    print("  Gemma 4 31B — Kaggle TPU v5e-8 Launcher")
    print("  Full BF16 | Zero Quantization | End-to-End")
    print(f"  Model: {args.model_dir}")
    print("=" * 60)
    print()

    # ── Step 1: Setup ─────────────────────────────────────────────
    if not args.skip_setup:
        print(">>> STEP 1/5: Installing dependencies")
        print()
        try:
            import setup
            setup.main()
        except SystemExit as e:
            if e.code != 0:
                print(f"[launcher] FATAL: Setup failed with exit code {e.code}")
                sys.exit(1)
        print()
    else:
        print(">>> STEP 1/5: Skipping setup (--skip_setup)")
        print()

    # ── Step 2: TPU Init ──────────────────────────────────────────
    print(">>> STEP 2/5: Initializing TPU")
    print()
    from tpu_init import init_tpu, get_mesh
    device_count = init_tpu()
    mesh = get_mesh(device_count)
    print()

    # ── Step 3: Tokenizer ─────────────────────────────────────────
    print(">>> STEP 3/5: Loading tokenizer")
    print()
    from tokenizer_setup import load_tokenizer
    tokenizer = load_tokenizer(args.model_dir)
    print()

    # ── Step 4: Model ─────────────────────────────────────────────
    print(">>> STEP 4/5: Loading model (BF16)")
    print()
    from model_loader import load_model
    model, params = load_model(args.model_dir)
    print()

    # ── Step 5: Inference ─────────────────────────────────────────
    print(">>> STEP 5/5: Running inference")
    print()
    from inference import warmup, generate

    # Warmup triggers XLA compilation so the real prompt doesn't timeout
    warmup(model, params, tokenizer)
    print()

    result = generate(
        model=model,
        params=params,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # ── Summary ───────────────────────────────────────────────────
    total_time = time.time() - overall_start
    print()
    print("=" * 60)
    print(f"Total runtime: {total_time:.1f}s")
    print("Status: SUCCESS")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
