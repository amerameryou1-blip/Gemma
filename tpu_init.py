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
import sys


def init_tpu() -> int:
    """
    Initialize TPU and return the number of TPU devices found.
    Exits with code 1 if TPU is not available.
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
        print()
        print("[tpu_init] FATAL: No TPU devices detected!")
        print("[tpu_init] Possible causes:")
        print("[tpu_init]   1. libtpu-nightly not installed")
        print("[tpu_init]   2. JAX_PLATFORMS not set to 'tpu'")
        print("[tpu_init]   3. Kaggle TPU accelerator not enabled")
        sys.exit(1)

    for i, dev in enumerate(devices):
        print(f"  Device [{i}] : {dev.device_kind} (id={dev.id})")

    # Verify we have 8 chips (v5e-8)
    if device_count != 8:
        print()
        print(f"[tpu_init] WARNING: Expected 8 TPU chips, found {device_count}")
        print("[tpu_init] This may not be a TPU v5e-8 instance.")
        print("[tpu_init] Continuing anyway...")

    # Verify TPU backend is actually TPU (not CPU fallback)
    backend = jax.default_backend()
    print(f"  Backend     : {backend}")
    if backend != "tpu":
        print(f"[tpu_init] FATAL: Backend is '{backend}', expected 'tpu'")
        sys.exit(1)

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
