"""Shared fixtures for the llm-from-scratch smoke test suite."""

from pathlib import Path

import pytest
import torch
from src.llm_from_scratch.tokenizer import get_tokenizer

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tokenizer():
    return get_tokenizer()


@pytest.fixture(scope="session")
def mini_corpus_text() -> str:
    return (FIXTURES_DIR / "mini_corpus.txt").read_text(encoding="utf-8")


@pytest.fixture
def tiny_config(tokenizer) -> dict:
    """
    A GPTModel config shaped like GPT_CONFIG_124M but tiny enough to
    forward/train in well under a second on CPU. vocab_size must match the
    real GPT-2 BPE tokenizer since create_dataloader_v1 always encodes with
    tiktoken's "gpt2" encoding.
    """
    return {
        "vocab_size": tokenizer.n_vocab,
        "context_length": 32,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }


@pytest.fixture(autouse=True)
def _fixed_seed():
    torch.manual_seed(42)
