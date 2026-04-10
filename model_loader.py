#!/usr/bin/env python3
"""
model_loader.py — Load Gemma 4 31B Flax weights in full BF16 precision
                  and shard across 8 TPU v5e chips.

Default model_dir is the Kaggle pre-installed path:
  /kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1

Key design decisions:
  - dtype=jnp.bfloat16  → Full BF16, zero quantization
  - _do_init=False      → Skip random init, load from checkpoint only
  - from_pretrained()   → Loads Flax weights directly (no conversion needed)

Tensor parallelism sharding strategy:
  - embed_tokens (vocab_size, hidden_size): shard along axis 1 (hidden dim)
    because vocab is too large to shard and each device needs full vocab.
  - All other 2D weight matrices: shard along axis 0 (input dim)
  - 1D tensors (layernorm, etc.): replicate across all devices
"""
import os
import time

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def shard_params(params, mesh: Mesh):
    """
    Shard model parameters across TPU devices using tensor parallelism.

    Strategy:
      - 2D matrices where shape[0] > 100000 (embedding): PartitionSpec(None, "tp")
        Shards along hidden dimension so every device has full vocab.
      - All other 2D matrices: PartitionSpec("tp", None)
        Shards along the first (input) dimension.
      - 1D tensors (layernorm, biases): PartitionSpec()
        Replicated — too small to shard.
    """
    def shard_leaf(x):
        if x.ndim == 2:
            if x.shape[0] > 100000:
                # Embedding matrix: shard along hidden dim (axis 1)
                sharding = NamedSharding(mesh, PartitionSpec(None, "tp"))
            else:
                # All other 2D weight matrices: shard along axis 0
                sharding = NamedSharding(mesh, PartitionSpec("tp", None))
        elif x.ndim == 1:
            # Layernorm weights, biases — replicate
            sharding = NamedSharding(mesh, PartitionSpec())
        else:
            # Higher-dimensional tensors: shard along first axis
            sharding = NamedSharding(mesh, PartitionSpec("tp"))
        return jax.device_put(x, sharding)

    return jax.tree_util.tree_map(shard_leaf, params)


def load_model(model_dir: str):
    """
    Load Gemma 4 31B Flax model in full BF16 and shard across 8 TPU chips.

    Returns:
        model: Flax model instance (with _do_init=False)
        params: Sharded model parameters in BF16
        mesh: JAX mesh used for sharding
    """
    print("=" * 60)
    print("Model Loading — Gemma 4 31B Flax BF16 + Tensor Parallelism")
    print("=" * 60)

    # Imports after TPU init to ensure JAX uses TPU
    try:
        from transformers import FlaxAutoModelForCausalLM
    except ImportError:
        try:
            from transformers import FlaxGemma4ForCausalLM as FlaxAutoModelForCausalLM
        except ImportError:
            from transformers import FlaxGemmaForCausalLM as FlaxAutoModelForCausalLM

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
        model, params = FlaxAutoModelForCausalLM.from_pretrained(
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
            f"  3. Flax model class not available in your\n"
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

    # ── Create mesh and shard params across 8 TPU chips ──────────
    print()
    print("[model_loader] Creating mesh and sharding params...")
    t1 = time.time()

    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("tp",))
    print(f"[model_loader] Mesh: {len(devices)} devices, axis=('tp',)")

    params = shard_params(params, mesh)

    shard_time = time.time() - t1
    print(f"[model_loader] Sharding completed in {shard_time:.1f}s")

    # Verify sharding
    sharded_leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"[model_loader] Sharded param shape: {sharded_leaf.shape}")
    print(f"[model_loader] Sharded param devices: {sharded_leaf.sharding}")

    # Verify model config
    config = model.config
    print(f"[model_loader] Model config:")
    print(f"[model_loader]   hidden_size    : {config.hidden_size}")
    print(f"[model_loader]   num_layers     : {config.num_hidden_layers}")
    print(f"[model_loader]   num_attention_heads: {config.num_attention_heads}")
    print(f"[model_loader]   vocab_size     : {config.vocab_size}")

    print()
    print("[model_loader] Model loaded and sharded successfully in full BF16.")

    return model, params, mesh


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1",
    )
    args = parser.parse_args()

    model, params, mesh = load_model(args.model_dir)
