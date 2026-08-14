"""
Smoke test for the training step: do two optimizer steps on a tiny model
run without producing NaN/Inf loss? This checks that the training loop is
mechanically sound (gradients flow, loss is a valid number) -- it makes no
claim that loss decreases or that the model learns anything useful.

Seed is fixed (see conftest._fixed_seed), the model is CPU-only and tiny,
and the whole test runs in well under 30 seconds.
"""

import math

import pytest
import torch
from src.llm_from_scratch.data import create_dataloader_v1
from src.llm_from_scratch.model import GPTModel
from src.llm_from_scratch.training import calc_loss_batch


@pytest.mark.smoke
def test_two_training_steps_produce_finite_loss(tiny_config, mini_corpus_text):
    device = torch.device("cpu")

    dataloader = create_dataloader_v1(
        mini_corpus_text,
        batch_size=2,
        max_length=tiny_config["context_length"] // 2,
        stride=tiny_config["context_length"] // 2,
        shuffle=False,
        drop_last=True,
    )

    model = GPTModel(tiny_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    model.train()
    for step, (input_batch, target_batch) in enumerate(dataloader):
        if step >= 2:
            break

        optimizer.zero_grad()
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    assert len(losses) == 2
    for loss_value in losses:
        assert math.isfinite(loss_value)
