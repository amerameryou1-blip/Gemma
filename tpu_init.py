#!/usr/bin/env python3
"""
tpu_init.py — Detect, initialize, and verify the Kaggle TPU v5e-8.

This MUST run before model loading. If TPU detection fails, everything
else will silently fall back to CPU and OOM.

Known failure points this script guards against:
  - JAX_PLATFORMS not set → JAX uses CPU
  - libtpu not found → JAX devices() returns []
  - TPU not visible → jax.device_count() == 0
"""
import os


def init_tpu() -> int:
    """
    Initialize TPU and return the number of TPU devices found.
    Raises RuntimeError if TPU is not available.
    """
    print("=" * 60)
    print("TPU Initialization")
    print("=" * 60)

    # Force JAX to use TPU. On Kaggle, JAX may default to CPU if this
    # env var is not set. Must be set BEFORE any jax import.
    os.environ["JAX_PLATFORMS"] = "tpu"
    os.environ["JAX_TRACEBACK_FILTERING"] = "off"

    # Now import JAX (after env vars are set)
    import jax
    import jax.numpy as jnp

    # Detect devices
    devices = jax.devices()
    device_count = len(devices)

    print(f"  JAX version : {jax.__version__}")
    print(f"  TPU devices : {device_count}")

    if device_count == 0:
        raise RuntimeError(
            "No TPU devices detected!\n"
            "  1. Make sure the TPU accelerator is enabled in Kaggle settings\n"
            "  2. libtpu should be pre-installed on Kaggle TPU images\n"
            "  3. Do NOT pip install jax[tpu] — it breaks the pre-installed libtpu"
        )

    for i, dev in enumerate(devices):
        print(f"  Device [{i}] : {dev.device_kind} (id={dev.id})")

    # Verify we have 8 chips (v5e-8)
    if device_count != 8:
        print(f"  WARNING: Expected 8 TPU chips, found {device_count}")
        print("  This may not be a TPU v5e-8 instance. Continuing anyway...")

    # Verify TPU backend is actually TPU (not CPU fallback)
    backend = jax.default_backend()
    print(f"  Backend     : {backend}")
    if backend != "tpu":
        raise RuntimeError(
            f"Backend is '{backend}', expected 'tpu'. "
            f"JAX_PLATFORMS='tpu' was set but JAX still chose {backend}."
        )

    # Memory check: TPU v5e has 16 GB HBM per chip, 128 GB total.
    # Gemma 4 31B in BF16 = ~58.3 GB weights.
    total_hbm_gb = device_count * 16
    weights_bf16_gb = 31e9 * 2 / 1e9
    print(f"  Total HBM   : {total_hbm_gb} GB ({device_count} chips x 16 GB)")
    print(f"  Weights BF16: ~{weights_bf16_gb:.1f} GB")
    print(f"  Headroom    : ~{total_hbm_gb - weights_bf16_gb:.1f} GB for KV cache + activations")

    print()
    print("[tpu_init] TPU initialized successfully.")
    return device_count


def get_mesh(device_count: int):
    """
    Create a JAX mesh for tensor parallelism across 8 TPU chips.
    Gemma 4 31B needs tensor parallelism — a single v5e chip (16 GB)
    cannot hold the full model.

    We use a 1D mesh: (8,) — all 8 chips in a single tensor-parallel group.
    """
    import jax
    from jax.sharding import Mesh

    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("tp",))
    return mesh


if __name__ == "__main__":
    count = init_tpu()
    mesh = get_mesh(count)
    print(f"[tpu_init] Mesh created: {mesh}")
