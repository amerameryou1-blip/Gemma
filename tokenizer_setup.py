#!/usr/bin/env python3
"""
tokenizer_setup.py — Load the Gemma 4 tokenizer.

The tokenizer is architecture-agnostic (same for PyTorch, Flax, JAX).
It must be loaded from the same model directory as the weights.

Default model_dir is the Kaggle pre-installed path:
  /kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1
"""
import argparse
import os
import sys


def load_tokenizer(model_dir: str):
    """
    Load and return the Gemma 4 tokenizer.

    Known issues:
    - The tokenizer uses SentencePiece. If sentencepiece is not installed,
      this will fail with ImportError.
    - Gemma 4 uses a different tokenizer than Gemma 3. Make sure the
      tokenizer files (tokenizer.model, tokenizer_config.json) are in
      the model directory.
    """
    from transformers import AutoTokenizer

    if not os.path.isdir(model_dir):
        print(f"[tokenizer] FATAL: Model directory not found: {model_dir}")
        sys.exit(1)

    # Verify tokenizer files exist
    required_files = ["tokenizer.model", "tokenizer_config.json"]
    for fname in required_files:
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            print(f"[tokenizer] WARNING: Missing tokenizer file: {fname}")
            print(f"[tokenizer] The tokenizer may still load if files are named differently.")

    print(f"[tokenizer] Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Gemma 4 tokenizer: pad_token must be set explicitly — Gemma models
    # don't have one by default. Using eos_token as pad_token is the
    # standard fix for all Gemma variants.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[tokenizer] Set pad_token = eos_token")

    print(f"[tokenizer] Vocabulary size: {tokenizer.vocab_size}")
    print(f"[tokenizer] Model max length: {tokenizer.model_max_length}")
    print("[tokenizer] Tokenizer loaded successfully.")

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1",
    )
    args = parser.parse_args()

    tok = load_tokenizer(args.model_dir)

    # Quick smoke test
    test = "Hello, world!"
    tokens = tok(test, return_tensors="np")
    decoded = tok.decode(tokens["input_ids"][0])
    print(f"[tokenizer] Smoke test: '{test}' -> {tokens['input_ids'].shape} -> '{decoded}'")
