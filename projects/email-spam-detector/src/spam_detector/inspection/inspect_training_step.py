"""
Inspect one real GPT-2 classification training step.

This script verifies that:

- gradients are calculated;
- frozen embeddings receive no gradients;
- the classification head receives gradients;
- the classification head changes after optimizer.step();
- frozen pretrained parameters remain unchanged;
- the loss stays finite.

This is NOT full model training.
"""

import math

import torch

from spam_detector.model.classifier import (
    configure_for_classification,
)
from spam_detector.model.dataloaders import (
    create_dataloaders,
)
from spam_detector.model.metrics import (
    calc_loss_batch,
)
from spam_detector.model.pretrained import (
    create_pretrained_gpt2_small,
)
from spam_detector.model.training import (
    create_optimizer,
)


def gradient_norm(
    parameter,
) -> float:
    """
    Return the L2 norm of a parameter gradient.
    """

    if parameter.grad is None:
        return 0.0

    return parameter.grad.norm().item()


def main() -> None:

    
    # Device
    
    
    # We deliberately stay on CPU for this small pipeline test.

    device = torch.device(
        "cpu"
    )

    print(
        "Device:",
        device,
    )

    
    # Load and configure model
    

    print(
        "\nLoading pretrained GPT-2..."
    )

    model = (
        create_pretrained_gpt2_small()
    )

    model = configure_for_classification(
        model
    )

    model.to(device)

    
    # Optimizer
    

    optimizer = create_optimizer(
        model
    )

    print(
        "\nOptimizer:",
        optimizer.__class__.__name__,
    )

    print(
        "Learning rate:",
        optimizer.param_groups[0]["lr"],
    )

    print(
        "Weight decay:",
        optimizer.param_groups[0][
            "weight_decay"
        ],
    )

    
    # Create one small real batch
    

    (
        train_loader,
        _,
        _,
    ) = create_dataloaders(
        batch_size=2,
        num_workers=0,
    )

    batch = next(
        iter(train_loader)
    )

    print("\n" + "=" * 70)

    print(
        "Training batch"
    )

    print("=" * 70)

    print(
        "Input shape:",
        batch["input_ids"].shape,
    )

    print(
        "Lengths:",
        batch["lengths"],
    )

    print(
        "EOS indices:",
        batch["eos_indices"],
    )

    print(
        "Labels:",
        batch["labels"],
    )

    print(
        "Sources:",
        batch["sources"],
    )

    
    # Store selected weights BEFORE training
    

    classification_head_before = (
        model.out_head.weight
        .detach()
        .clone()
    )

    # We only store a tiny slice of the frozen embedding matrix.
    frozen_embedding_before = (
        model.tok_emb.weight[
            0,
            :10,
        ]
        .detach()
        .clone()
    )

    
    # Forward pass
    

    model.train()

    optimizer.zero_grad()

    loss_before = calc_loss_batch(
        batch=batch,
        model=model,
        device=device,
    )

    print(
        f"\nLoss before update: "
        f"{loss_before.item():.6f}"
    )

    
    # Backpropagation
    

    loss_before.backward()

    
    # Inspect gradients
    

    print("\nGradient norms:")

    print(
        "Output head:",
        gradient_norm(
            model.out_head.weight
        ),
    )

    print(
        "Final LayerNorm:",
        gradient_norm(
            model.final_norm.scale
        ),
    )

    print(
        "Last Transformer block:",
        gradient_norm(
            model.trf_blocks[
                -1
            ].att.W_query.weight
        ),
    )

    print(
        "Token embedding:",
        gradient_norm(
            model.tok_emb.weight
        ),
    )

    # Frozen embedding should have no gradient at all.
    assert (
        model.tok_emb.weight.grad
        is None
    )

    # Trainable components should have gradients.
    assert (
        model.out_head.weight.grad
        is not None
    )

    assert (
        model.final_norm.scale.grad
        is not None
    )

    assert (
        model.trf_blocks[
            -1
        ].att.W_query.weight.grad
        is not None
    )

    print(
        "\nGradient checks passed."
    )

    
    # Perform optimizer update
    

    optimizer.step()

    
    # Compare parameters before and after
    

    classification_head_changed = (
        not torch.equal(
            classification_head_before,
            model.out_head.weight.detach(),
        )
    )

    frozen_embedding_unchanged = (
        torch.equal(
            frozen_embedding_before,
            model.tok_emb.weight[
                0,
                :10,
            ].detach(),
        )
    )

    print(
        "\nClassification head changed:",
        classification_head_changed,
    )

    print(
        "Frozen embedding unchanged:",
        frozen_embedding_unchanged,
    )

    assert classification_head_changed

    assert frozen_embedding_unchanged

    
    # Recalculate loss on the SAME batch
    

    model.eval()

    with torch.no_grad():

        loss_after = calc_loss_batch(
            batch=batch,
            model=model,
            device=device,
        )

    print(
        f"\nLoss after update: "
        f"{loss_after.item():.6f}"
    )

    print(
        "Loss difference:",
        f"{loss_after.item() - loss_before.item():.6f}",
    )

    
    # Sanity checks
    

    assert math.isfinite(
        loss_before.item()
    )

    assert math.isfinite(
        loss_after.item()
    )

    assert (
        loss_before.item() >= 0
    )

    assert (
        loss_after.item() >= 0
    )

    print("\n" + "=" * 70)

    print(
        "One complete training step "
        "passed successfully."
    )


if __name__ == "__main__":
    main()