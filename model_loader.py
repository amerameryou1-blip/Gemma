#!/usr/bin/env python3
"""
model_loader.py — Load Gemma 4 31B Flax weights in full BF16 precision.

This is the critical file. It loads the pre-downloaded Flax checkpoint
into memory on the TPU.

Default model_dir is the Kaggle pre-installed path:
  /kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1

Key design decisions:
  - dtype=jnp.bfloat16  → Full BF16, zero quantization
  - _do_init=False      → Skip random init, load from checkpoint only
  - from_pretrained()   → Loads Flax weights directly (no conversion needed)

Known failure points and fixes:
  1. Shape mismatch: If the Flax weights were converted from PyTorch and
     the conversion script had a bug, shapes may not match the Flax model
     definition. Fix: Re-convert weights using the official script.
  2. OOM during loading: Flax loads all params into host RAM first, then
     transfers to TPU. If host RAM < 70 GB, this will fail. Kaggle's
     high-RAM environment provides enough RAM for this.
  3. XLA compilation error: The first forward pass triggers XLA compilation
     which can take 2-5 minutes. This is normal — do not interrupt.
"""
import os
import time

import jax
import jax.numpy as jnp


def load_model(model_dir: str):
    """
    Load Gemma 4 31B Flax model in full BF16.

    Returns:
        model: FlaxGemma4ForCausalLM instance (with _do_init=False)
        params: Frozen dict of model parameters in BF16
    """
    print("=" * 60)
    print("Model Loading — Gemma 4 31B Flax BF16")
    print("=" * 60)

    # Imports after TPU init to ensure JAX uses TPU
    from transformers import FlaxGemma4ForCausalLM

    if not os.path.isdir(model_dir):
        raise RuntimeError(
            f"Model directory not found: {model_dir}\n"
            f"Expected: /kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1"
        )

    print(f"[model_loader] Loading from: {model_dir}")
    print("[model_loader] Precision: BF16 (jnp.bfloat16)")
    print("[model_loader] This may take 2-5 minutes...")

    t0 = time.time()

    # _do_init=False prevents random initialization — we load from
    # the checkpoint only. This is critical for large models because
    # random init would waste memory and time.
    #
    # dtype=jnp.bfloat16 ensures all computations happen in BF16.
    # This is NOT quantization — BF16 is a standard floating point
    # format with the same dynamic range as FP32 but half the memory.
    try:
        model, params = FlaxGemma4ForCausalLM.from_pretrained(
            model_dir,
            dtype=jnp.bfloat16,
            _do_init=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model: {e}\n\n"
            f"Common causes:\n"
            f"  1. Weights are PyTorch format, not Flax format\n"
            f"     → Use the Flax variant from the Kaggle dataset\n"
            f"  2. Weights are corrupted or incomplete\n"
            f"     → Re-download from Kaggle dataset\n"
            f"  3. FlaxGemma4ForCausalLM not available in your\n"
            f"     transformers version\n"
            f"     → pip install --upgrade transformers"
        ) from e

    elapsed = time.time() - t0
    print(f"[model_loader] Model loaded in {elapsed:.1f}s")

    # Verify parameter count
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"[model_loader] Parameter count: {param_count:,} ({param_count/1e9:.1f}B)")

    if param_count < 30e9:
        print(f"[model_loader] WARNING: Expected ~31B params, got {param_count/1e9:.1f}B")
        print("[model_loader] You may have loaded the wrong model variant.")

    # Verify dtype
    leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"[model_loader] Parameter dtype: {leaf.dtype}")
    if leaf.dtype != jnp.bfloat16:
        print(f"[model_loader] WARNING: Expected bfloat16, got {leaf.dtype}")
        print("[model_loader] Model may not be running in full BF16.")

    # Verify model config
    config = model.config
    print(f"[model_loader] Model config:")
    print(f"[model_loader]   hidden_size    : {config.hidden_size}")
    print(f"[model_loader]   num_layers     : {config.num_hidden_layers}")
    print(f"[model_loader]   num_attention_heads: {config.num_attention_heads}")
    print(f"[model_loader]   vocab_size     : {config.vocab_size}")

    print()
    print("[model_loader] Model loaded successfully in full BF16.")

    return model, params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1",
    )
    args = parser.parse_args()

    model, params = load_model(args.model_dir)
