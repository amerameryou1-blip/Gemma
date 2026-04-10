#!/usr/bin/env python3
"""
setup.py — Install dependencies for Gemma 4 31B on Kaggle TPU v5e-8.

CRITICAL: Kaggle already ships with JAX + libtpu pre-installed and
configured for the TPU runtime. Reinstalling JAX BREAKS the TPU
connection (wrong libtpu version, wrong wheel). This script only
installs transformers, sentencepiece, and accelerate.
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

    # ── DO NOT reinstall JAX on Kaggle ─────────────────────────────
    # Kaggle's TPU Docker image already has JAX + libtpu compiled
    # for the exact TPU chip. Reinstalling jax[tpu] pulls a wheel
    # with a mismatched libtpu, causing "No TPU devices detected".
    # We only verify JAX is importable, then skip.
    print()
    print("[setup] Checking JAX (Kaggle pre-installed — not reinstalling)...")
    try:
        import jax
        print(f"  jax          {jax.__version__}  (pre-installed, OK)")
    except ImportError:
        print("  jax          NOT FOUND — installing as fallback...")
        run(
            "pip install --quiet "
            '"jax[tpu]>=0.4.30" '
            "-f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        )

    try:
        import flax
        print(f"  flax         {flax.__version__}")
    except ImportError:
        print("  flax         NOT FOUND — installing...")
        run("pip install --quiet flax")

    # ── Install non-JAX dependencies ───────────────────────────────
    # transformers >= 4.49.0 for Gemma 4 architecture support
    run("pip install --quiet \"transformers>=4.49.0\"")

    # SentencePiece is required by the Gemma tokenizer
    run("pip install --quiet sentencepiece")

    # Accelerate is needed by some transformers internals
    run("pip install --quiet accelerate")

    # ── Verify installs ───────────────────────────────────────────
    print()
    print("[setup] Verifying installations...")

    try:
        import jax
        print(f"  jax          {jax.__version__}")
    except ImportError:
        raise RuntimeError(
            "jax failed to import. "
            "Kaggle should have it pre-installed. "
            "Check that the TPU accelerator is enabled."
        )

    try:
        import flax
        print(f"  flax         {flax.__version__}")
    except ImportError:
        raise RuntimeError("flax failed to import. Run: pip install flax")

    try:
        import transformers
        print(f"  transformers {transformers.__version__}")
    except ImportError:
        raise RuntimeError(
            "transformers failed to import. "
            "Run: pip install transformers>=4.49.0"
        )

    try:
        import sentencepiece
        print(f"  sentencepiece OK")
    except ImportError:
        raise RuntimeError(
            "sentencepiece failed to import. Run: pip install sentencepiece"
        )

    print()
    print("[setup] All dependencies installed successfully.")


if __name__ == "__main__":
    main()
