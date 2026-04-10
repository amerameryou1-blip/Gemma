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


def pip_install(packages: list[str], extra_args: list[str] | None = None) -> None:
    """Run pip install with proper argument handling.

    FIX: The old pip() function had two bugs:
      1. It double-added "install" (cmd had "install" + function prepended it)
      2. cmd.split() kept literal quote characters in package names
    This version takes a clean list of package names and builds the
    subprocess args correctly.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    print(f"[setup] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"pip install failed for: {packages}")


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
        pip_install(["flax"])
        import flax
        print(f"  flax         {flax.__version__}")

    # ── Install transformers ───────────────────────────────────────
    # Gemma 4 architecture support was added in transformers 4.49+.
    # We try the latest first, then fall back to specific versions.
    print()
    print("[setup] Installing transformers...")

    transformers_installed = False
    for version_spec in ["transformers>=4.49.0", "transformers>=4.48.0", "transformers"]:
        try:
            pip_install([version_spec])
            import transformers
            print(f"[setup] transformers {transformers.__version__} installed OK")
            transformers_installed = True
            break
        except RuntimeError:
            print(f"[setup] Failed with {version_spec}, trying next...")

    if not transformers_installed:
        raise RuntimeError(
            "Could not install transformers. "
            "Check network connectivity on Kaggle."
        )

    # ── Install SentencePiece ──────────────────────────────────────
    print("[setup] Installing sentencepiece...")
    pip_install(["sentencepiece"])

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
