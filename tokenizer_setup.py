#!/usr/bin/env python3
"""
tokenizer_setup.py — Load the Gemma 4 tokenizer.

FIX for the TypeError: not a string error:
  AutoTokenizer.from_pretrained() resolves to GemmaTokenizer (Gemma 1/2)
  which fails because vocab_file in the Flax tokenizer_config.json is
  not a plain string path.

  Solution: Load SentencePiece directly from tokenizer.model file.
  This bypasses the broken AutoTokenizer path entirely.

Default model_dir is the Kaggle pre-installed path:
  /kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1
"""
import json
import os

import numpy as np
import sentencepiece as spm


class SimpleGemmaTokenizer:
    """Minimal tokenizer wrapper around SentencePiece for Gemma 4."""

    def __init__(self, sp_model, special_tokens_dict):
        self.sp = sp_model
        self.vocab_size = sp_model.GetPieceSize()
        self.bos_token = special_tokens_dict.get("bos_token", "<bos>")
        self.eos_token = special_tokens_dict.get("eos_token", "<eos>")
        self.unk_token = special_tokens_dict.get("unk_token", "<unk>")
        self.pad_token = special_tokens_dict.get("pad_token", self.eos_token)
        self.model_max_length = 131072
        self.pad_token_id = self.sp.PieceToId(self.pad_token)
        self.eos_token_id = self.sp.PieceToId(self.eos_token)
        self.bos_token_id = self.sp.PieceToId(self.bos_token)
        self.unk_token_id = self.sp.PieceToId(self.unk_token)

    def __call__(self, text, return_tensors=None, padding=False, **kwargs):
        """Tokenize text and return dict compatible with model.generate()."""
        if isinstance(text, list):
            ids = [self.sp.EncodeAsIds(t) for t in text]
        else:
            ids = [self.sp.EncodeAsIds(text)]

        # Add BOS token if not already present
        for i in range(len(ids)):
            if ids[i][0] != self.bos_token_id:
                ids[i] = [self.bos_token_id] + ids[i]

        if return_tensors == "np" or return_tensors == "numpy":
            return {
                "input_ids": np.array(ids, dtype=np.int64),
                "attention_mask": np.ones((len(ids), len(ids[0])), dtype=np.int64),
            }
        return {"input_ids": ids}

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        """Decode token IDs back to text."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        if skip_special_tokens:
            skip_ids = {
                self.sp.PieceToId(self.bos_token),
                self.sp.PieceToId(self.eos_token),
                self.sp.PieceToId(self.unk_token),
                self.sp.PieceToId(self.pad_token),
            }
            token_ids = [t for t in token_ids if t not in skip_ids]

        return self.sp.DecodeIds(token_ids)


def load_tokenizer(model_dir: str):
    """
    Load and return the Gemma 4 tokenizer via direct SentencePiece.

    This is the ONLY approach that works with the Flax model directory
    because AutoTokenizer resolves to the wrong tokenizer class.
    """
    if not os.path.isdir(model_dir):
        raise RuntimeError(
            f"Model directory not found: {model_dir}\n"
            f"Expected: /kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1"
        )

    # ── Find tokenizer.model ──────────────────────────────────────
    tokenizer_model_path = None
    for candidate in [
        os.path.join(model_dir, "tokenizer.model"),
        os.path.join(model_dir, "tokenizer", "tokenizer.model"),
    ]:
        if os.path.isfile(candidate):
            tokenizer_model_path = candidate
            break

    if tokenizer_model_path is None:
        contents = sorted(os.listdir(model_dir))
        raise RuntimeError(
            f"tokenizer.model not found in {model_dir}\n"
            f"Directory contents: {contents}"
        )

    print(f"[tokenizer] Found tokenizer.model at: {tokenizer_model_path}")

    # ── Load special tokens from tokenizer_config.json ────────────
    config_path = os.path.join(model_dir, "tokenizer_config.json")
    special_tokens = {}
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            tc = json.load(f)
        for key in ["bos_token", "eos_token", "unk_token", "pad_token"]:
            val = tc.get(key, "")
            if isinstance(val, dict):
                val = val.get("content", "")
            if val:
                special_tokens[key] = val
        print(f"[tokenizer] Special tokens from config: {special_tokens}")

    # ── Load SentencePiece directly ───────────────────────────────
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model_path)
    print(f"[tokenizer] SentencePiece loaded. Vocab size: {sp.GetPieceSize()}")

    tokenizer = SimpleGemmaTokenizer(sp, special_tokens)
    print(f"[tokenizer] Vocabulary size: {tokenizer.vocab_size}")
    print("[tokenizer] Tokenizer loaded successfully via direct SentencePiece.")
    return tokenizer


if __name__ == "__main__":
    import argparse
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
