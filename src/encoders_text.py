"""
Tokenizer utilities for Project 3 (memory model).
"""
from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer


def load_tokenizer(name: str, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(name, cache_dir=str(cache_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else "[PAD]"
    return tok
