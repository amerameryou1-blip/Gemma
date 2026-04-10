#!/usr/bin/env python3
"""
tokenizer_setup.py — Load the Gemma 4 tokenizer.

FIX for the TypeError: not a string error:
  The Flax model directory's tokenizer_config.json does NOT map to a
  Gemma4Tokenizer class in transformers (it maps to the old GemmaTokenizer
  which expects a string vocab_file path but gets something else).

  Solution: Use AutoProcessor instead of AutoTokenizer. Gemma 4 is a
  multimodal model and its text interface is through the Processor,
  which wraps the SentencePiece tokenizer correctly.

  If AutoProcessor is not available, fall back to loading the
  SentencePiece model directly from the tokenizer.model file.

Default model_dir is the Kaggle pre-installed path:
  /kaggle/input/models/google/gemma-4/flax/gemma-4-31b-it/1
"""
import argparse
import os
import sys


def load_tokenizer(model_dir: str):
    """
    Load and return the Gemma 4 tokenizer/processor.

    Strategy (in order of preference):
      1. AutoProcessor.from_pretrained() — Gemma 4 uses a multimodal processor
      2. GemmaTokenizer with explicit vocab_file path to tokenizer.model
      3. Direct SentencePiece loading as last resort
    """
    if not os.path.isdir(model_dir):
        print(f"[tokenizer] FATAL: Model directory not found: {model_dir}")
        sys.exit(1)

    # ── Attempt 1: AutoProcessor (preferred for Gemma 4) ──────────
    # Gemma 4 is multimodal. AutoProcessor handles the tokenizer correctly
    # and has the apply_chat_template method we need.
    print(f"[tokenizer] Attempt 1: Loading AutoProcessor from: {model_dir}")
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )

        # AutoProcessor wraps the tokenizer — access it via .tokenizer
        if hasattr(processor, 'tokenizer'):
            tokenizer = processor.tokenizer
            print("[tokenizer] Loaded via AutoProcessor (has .tokenizer attribute)")
        else:
            # Some processors ARE the tokenizer
            tokenizer = processor
            print("[tokenizer] AutoProcessor is itself the tokenizer")

        # Gemma: pad_token must be set explicitly
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("[tokenizer] Set pad_token = eos_token")

        print(f"[tokenizer] Vocabulary size: {tokenizer.vocab_size}")
        print("[tokenizer] Tokenizer loaded successfully via AutoProcessor.")
        return tokenizer

    except Exception as e:
        print(f"[tokenizer] AutoProcessor failed: {e}")
        print("[tokenizer] Falling back to Attempt 2...")

    # ── Attempt 2: Direct SentencePiece loading ───────────────────
    # Find the tokenizer.model file — it may be at the root of model_dir
    # or in a subdirectory.
    print("[tokenizer] Attempt 2: Loading SentencePiece model directly...")

    tokenizer_model_path = None
    for candidate in [
        os.path.join(model_dir, "tokenizer.model"),
        os.path.join(model_dir, "tokenizer", "tokenizer.model"),
    ]:
        if os.path.isfile(candidate):
            tokenizer_model_path = candidate
            break

    if tokenizer_model_path is None:
        # List what IS in the directory to help debug
        print(f"[tokenizer] FATAL: tokenizer.model not found in {model_dir}")
        print(f"[tokenizer] Directory contents:")
        for item in sorted(os.listdir(model_dir)):
            print(f"  {item}")
        sys.exit(1)

    print(f"[tokenizer] Found tokenizer.model at: {tokenizer_model_path}")

    # Load tokenizer_config.json to get special tokens
    import json
    config_path = os.path.join(model_dir, "tokenizer_config.json")
    special_tokens = {}
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            tc = json.load(f)
        # Extract special token strings (they may be nested dicts)
        for key in ["bos_token", "eos_token", "unk_token", "pad_token"]:
            val = tc.get(key, "")
            if isinstance(val, dict):
                val = val.get("content", "")
            if val:
                special_tokens[key] = val
        print(f"[tokenizer] Special tokens from config: {special_tokens}")

    # Load SentencePiece directly
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model_path)
    print(f"[tokenizer] SentencePiece loaded. Vocab size: {sp.GetPieceSize()}")

    # Wrap it in a minimal tokenizer interface
    class SimpleGemmaTokenizer:
        """Minimal tokenizer wrapper around SentencePiece for Gemma 4."""

        def __init__(self, sp_model, special_tokens_dict):
            self.sp = sp_model
            self.vocab_size = sp_model.GetPieceSize()
            self.bos_token = special_tokens_dict.get("bos_token", "<bos>")
            self.eos_token = special_tokens_dict.get("eos_token", "<eos>")
            self.unk_token = special_tokens_dict.get("unk_token", "<unk>")
            self.pad_token = special_tokens_dict.get("pad_token", self.eos_token)
            self.model_max_length = 131072  # Gemma 4 supports 128K context

        def __call__(self, text, return_tensors=None, padding=False, **kwargs):
            """Tokenize text and return dict compatible with model.generate()."""
            if isinstance(text, list):
                ids = [self.sp.EncodeAsIds(t) for t in text]
            else:
                ids = [self.sp.EncodeAsIds(text)]

            # Add BOS token if not already present
            bos_id = self.sp.PieceToId(self.bos_token)
            for i in range(len(ids)):
                if ids[i][0] != bos_id:
                    ids[i] = [bos_id] + ids[i]

            import numpy as np
            if return_tensors == "np" or return_tensors == "numpy":
                return {"input_ids": np.array(ids), "attention_mask": np.ones((len(ids), len(ids[0])), dtype=np.int64)}
            return {"input_ids": ids}

        def decode(self, token_ids, skip_special_tokens=False, **kwargs):
            """Decode token IDs back to text."""
            if hasattr(token_ids, 'tolist'):
                token_ids = token_ids.tolist()
            if isinstance(token_ids[0], list):
                token_ids = token_ids[0]

            if skip_special_tokens:
                bos_id = self.sp.PieceToId(self.bos_token)
                eos_id = self.sp.PieceToId(self.eos_token)
                unk_id = self.sp.PieceToId(self.unk_token)
                pad_id = self.sp.PieceToId(self.pad_token)
                skip_ids = {bos_id, eos_id, unk_id, pad_id}
                token_ids = [t for t in token_ids if t not in skip_ids]

            return self.sp.DecodeIds(token_ids)

    tokenizer = SimpleGemmaTokenizer(sp, special_tokens)
    print(f"[tokenizer] Vocabulary size: {tokenizer.vocab_size}")
    print("[tokenizer] Tokenizer loaded successfully via direct SentencePiece.")
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
