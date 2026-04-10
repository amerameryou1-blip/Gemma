#!/usr/bin/env python3
"""
inference.py — Run text generation with Gemma 4 31B on TPU v5e-8.

This is the final step. It takes the loaded model and params, tokenizes
a prompt, and generates text.

Known failure points and fixes:
  1. XLA compilation timeout: The first generation call triggers XLA
     compilation which can take 2-5 minutes. If Kaggle kills the cell
     before compilation finishes, re-run. Fix: Add a warmup call with
     a very short prompt first.
  2. Shape mismatch in generate: If the prompt is longer than max_length,
     generation will fail. Fix: Ensure max_length > len(input_ids).
  3. OOM during generation: KV cache grows linearly with sequence length.
     If max_length is too large, TPU memory will be exhausted. Fix:
     Reduce max_length or use shorter prompts.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np


def generate(
    model,
    params,
    tokenizer,
    prompt: str,
    max_length: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    seed: int = 42,
) -> str:
    """
    Generate text from a prompt using the Flax model on TPU.

    Returns the generated text (prompt + completion).
    """
    print("=" * 60)
    print("Inference — Gemma 4 31B")
    print("=" * 60)
    print(f"  Prompt     : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"  Max length : {max_length}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k      : {top_k}")
    print()

    # return_tensors="np" gives numpy arrays which JAX can consume.
    # We do NOT use return_tensors="jax" because the tokenizer doesn't
    # always return JAX arrays correctly on TPU.
    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", np.ones_like(input_ids))

    prompt_len = input_ids.shape[1]
    print(f"[inference] Prompt tokens: {prompt_len}")

    if prompt_len >= max_length:
        raise RuntimeError(
            f"Prompt length ({prompt_len}) >= max_length ({max_length}). "
            f"Increase max_length or use a shorter prompt."
        )

    # FlaxGemma4ForCausalLM.generate() handles the autoregressive loop
    # internally. It uses JAX's pmap/scan under the hood for TPU
    # efficiency.
    #
    # The first call triggers XLA compilation — this is slow (2-5 min)
    # but subsequent calls are fast.
    print("[inference] Starting generation (first call includes XLA compilation)...")
    t0 = time.time()

    try:
        output_ids = model.generate(
            jnp.array(input_ids),
            params=params,
            prng_key=jax.random.PRNGKey(seed),
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=temperature > 0.0,
        )
    except Exception as e:
        raise RuntimeError(
            f"Generation failed: {e}\n\n"
            f"Common causes:\n"
            f"  1. XLA compilation error — re-run the cell\n"
            f"  2. OOM — reduce max_length\n"
            f"  3. Shape mismatch — check model/params compatibility"
        ) from e

    elapsed = time.time() - t0
    gen_tokens = output_ids.shape[1] - prompt_len
    tokens_per_sec = gen_tokens / elapsed if elapsed > 0 else 0

    print(f"[inference] Generation completed in {elapsed:.1f}s")
    print(f"[inference] Generated {gen_tokens} tokens ({tokens_per_sec:.2f} tok/s)")

    # output_ids is a JAX array. Convert to numpy for the tokenizer.
    output_np = np.array(output_ids)
    generated_text = tokenizer.decode(output_np[0], skip_special_tokens=False)

    print()
    print("-" * 60)
    print("GENERATED OUTPUT:")
    print("-" * 60)
    print(generated_text)
    print("-" * 60)
    print()
    print("[inference] Done.")

    return generated_text


def warmup(model, params, tokenizer):
    """
    Run a tiny warmup generation to trigger XLA compilation.
    This prevents the first real prompt from timing out.
    """
    print("[inference] Running warmup generation (triggers XLA compilation)...")
    t0 = time.time()
    inputs = tokenizer("Hi", return_tensors="np")
    _ = model.generate(
        jnp.array(inputs["input_ids"]),
        params=params,
        prng_key=jax.random.PRNGKey(0),
        max_length=32,
        do_sample=False,
    )
    elapsed = time.time() - t0
    print(f"[inference] Warmup completed in {elapsed:.1f}s")
    print("[inference] XLA compilation done. Subsequent calls will be fast.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1",
    )
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms.")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    # Initialize TPU
    from tpu_init import init_tpu, get_mesh
    device_count = init_tpu()
    mesh = get_mesh(device_count)

    # Load tokenizer
    from tokenizer_setup import load_tokenizer
    tokenizer = load_tokenizer(args.model_dir)

    # Load model
    from model_loader import load_model
    model, params = load_model(args.model_dir)

    # Warmup (triggers XLA compilation)
    warmup(model, params, tokenizer)

    # Generate
    generate(
        model=model,
        params=params,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )
