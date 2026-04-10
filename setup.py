#!/usr/bin/env python3
"""
setup.py — Install dependencies for Gemma 4 31B on Kaggle TPU v5e-8.

CRITICAL: Kaggle already ships with JAX + libtpu pre-installed and
configured for the TPU runtime. Reinstalling JAX BREAKS the TPU
connection (wrong libtpu version, wrong wheel). This script only
installs transformers and sentencepiece.
"""
import subprocess
import sys


def pip(cmd: str) -> None:
    """Run pip install and print output."""
    print(f"[setup] $ pip {cmd}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + cmd.split(),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pip install failed: {cmd}")


def main():
    print("=" * 60)
    print("Gemma 4 31B — Installing dependencies for TPU v5e-8")
    print("=" * 60)

    # ── DO NOT reinstall JAX on Kaggle ─────────────────────────────
    # Kaggle's TPU Docker image already has JAX + libtpu compiled
    # for the exact TPU chip. Reinstalling jax[tpu] pulls a wheel
    # with a mismatched libtpu, causing "No TPU devices detected".
    print()
    print("[setup] Checking pre-installed packages...")

    try:
        import jax
        print(f"  jax          {jax.__version__}  (pre-installed, OK)")
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
        pip("install flax")
        import flax
        print(f"  flax         {flax.__version__}")

    # ── Install non-JAX dependencies ───────────────────────────────
    # transformers >= 4.49.0 for Gemma 4 architecture support
    pip("install --quiet \"transformers>=4.49.0\"")

    # SentencePiece is required by the Gemma tokenizer
    pip("install --quiet sentencepiece")

    # ── Verify installs ───────────────────────────────────────────
    print()
    print("[setup] Verifying installations...")

    import jax
    print(f"  jax          {jax.__version__}")

    import flax
    print(f"  flax         {flax.__version__}")

    import transformers
    print(f"  transformers {transformers.__version__}")

    import sentencepiece
    print(f"  sentencepiece OK")

    print()
    print("[setup] All dependencies installed successfully.")


if __name__ == "__main__":
    main()
