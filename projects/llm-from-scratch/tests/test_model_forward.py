"""
Smoke test for GPTModel: does a forward pass run and produce the expected
output shape? This checks shapes and finiteness, not generation quality.
"""

import pytest
import torch
from src.llm_from_scratch.model import GPTModel


@pytest.mark.smoke
def test_forward_pass_output_shape(tiny_config):
    model = GPTModel(tiny_config)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, tiny_config["vocab_size"], (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, tiny_config["vocab_size"])
    assert torch.isfinite(logits).all()


@pytest.mark.smoke
def test_forward_pass_respects_context_length(tiny_config):
    model = GPTModel(tiny_config)
    model.eval()

    seq_len = tiny_config["context_length"]
    input_ids = torch.randint(0, tiny_config["vocab_size"], (1, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, seq_len, tiny_config["vocab_size"])
