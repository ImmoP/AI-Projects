"""
Smoke test for the complete GPT-2 classification training loop.

This intentionally trains for only two optimization steps.

A reduced context length is used to make the test fast on CPU.
This is NOT the configuration for final model training.
"""

import math

import torch

from spam_detector.model.classifier import (
    configure_for_classification,
)
from spam_detector.model.dataloaders import (
    create_dataloaders,
)
from spam_detector.model.pretrained import (
    create_pretrained_gpt2_small,
)
from spam_detector.model.training import (
    create_optimizer,
    train_classifier,
)


def main() -> None:

    
    # Device
    

    device = torch.device(
        "cpu"
    )

    print(
        "Device:",
        device,
    )

    
    # Model
    

    print(
        "\nLoading pretrained GPT-2..."
    )

    model = (
        create_pretrained_gpt2_small()
    )

    model = (
        configure_for_classification(
            model
        )
    )

    model.to(device)

    
    # Data
    
    
    # context_length=128 is ONLY used to make this CPU smoke test
    # fast.
    
    # Final training will use our selected context length of 1024.

    (
        train_loader,
        eval_loader,
        _,
    ) = create_dataloaders(
        batch_size=2,
        num_workers=0,
        context_length=128,
    )

    
    # Optimizer
    

    optimizer = create_optimizer(
        model
    )

    
    # Run two real optimization steps
    

    print("\n" + "=" * 70)

    print(
        "Starting two-step "
        "training smoke test"
    )

    print("=" * 70)

    history = train_classifier(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=1,
        eval_freq=1,
        eval_iter=1,
        max_steps=2,
    )


    # Inspect history
    

    print("\nTraining history:")

    print(
        "Steps:",
        history["steps"],
    )

    print(
        "Examples seen:",
        history["examples_seen"],
    )

    print(
        "Train losses:",
        history["train_loss"],
    )

    print(
        "Eval losses:",
        history["eval_loss"],
    )

    print(
        "Train accuracies:",
        history["train_accuracy"],
    )

    print(
        "Eval accuracies:",
        history["eval_accuracy"],
    )

    
    # Sanity checks
    

    assert history[
        "steps"
    ] == [1, 2]

    assert history[
        "examples_seen"
    ] == [2, 4]

    assert len(
        history["train_loss"]
    ) == 2

    assert len(
        history["eval_loss"]
    ) == 2

    for loss in (
        history["train_loss"]
        + history["eval_loss"]
    ):
        assert math.isfinite(
            loss
        )

        assert loss >= 0

    assert len(
        history[
            "train_accuracy"
        ]
    ) == 1

    assert len(
        history[
            "eval_accuracy"
        ]
    ) == 1

    print("\n" + "=" * 70)

    print(
        "Complete training-loop "
        "smoke test passed."
    )


if __name__ == "__main__":
    main()