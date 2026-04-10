#!/usr/bin/env python3
"""
setup.py — Install all dependencies for Gemma 4 31B Flax on Kaggle TPU v5e-8
Run this FIRST. It installs everything needed before any other script runs.
"""
import subprocess
import sys


def run(cmd: str) -> int:
    """Run a shell command and return exit code."""
    print(f"[setup] $ {cmd}")
    return subprocess.call(cmd, shell=True)


def main():
    print("=" * 60)
    print("Gemma 4 31B — Installing dependencies for TPU v5e-8")
    print("=" * 60)

    # Kaggle TPU images ship with JAX but we pin versions to avoid
    # silent breakage. The libtpu-nightly wheel is REQUIRED for TPU
    # execution — without it JAX falls back to CPU silently.
    run(
        "pip install --quiet "
        '"jax[tpu]>=0.4.30" '
        "-f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
    )

    # FlaxGemma4ForCausalLM lives in transformers. We need >= 4.49.0
    # for Gemma 4 architecture support (Gemma4Config was added in that release).
    run("pip install --quiet \"transformers>=4.49.0\"")

    # SentencePiece is required by the Gemma tokenizer.
    run("pip install --quiet sentencepiece")

    # Accelerate is needed by some transformers internals.
    run("pip install --quiet accelerate")

    # ── Verify installs ───────────────────────────────────────────
    print()
    print("[setup] Verifying installations...")

    try:
        import jax
        print(f"  jax          {jax.__version__}")
    except ImportError:
        print("  jax          FAILED TO IMPORT")
        sys.exit(1)

    try:
        import flax
        print(f"  flax         {flax.__version__}")
    except ImportError:
        print("  flax         FAILED TO IMPORT")
        sys.exit(1)

    try:
        import transformers
        print(f"  transformers {transformers.__version__}")
    except ImportError:
        print("  transformers FAILED TO IMPORT")
        sys.exit(1)

    try:
        import sentencepiece
        print(f"  sentencepiece OK")
    except ImportError:
        print("  sentencepiece FAILED TO IMPORT")
        sys.exit(1)

    print()
    print("[setup] All dependencies installed successfully.")


if __name__ == "__main__":
    main()
