"""
Inspect classification loss and accuracy before finetuning.

The classification head is still randomly initialized, so good
classification performance is NOT expected.

This script verifies only that:

- classification loss can be calculated;
- accuracy can be calculated;
- real DataLoader batches work;
- train and evaluation loaders can be evaluated;
- all returned metric values are valid.
"""

import math

import torch

from spam_detector.model.classifier import (
    classification_forward,
    configure_for_classification,
)
from spam_detector.model.dataloaders import (
    create_dataloaders,
)
from spam_detector.model.metrics import (
    accuracy_from_logits,
    calc_accuracy_loader,
    calc_loss_batch,
    calc_loss_loader,
)
from spam_detector.model.pretrained import (
    create_pretrained_gpt2_small,
)


def main() -> None:

    
    # Device
    
    
    # Keep this first inspection on CPU.
    
    # Device optimization will be handled separately when we
    # implement the actual training pipeline.

    device = torch.device(
        "cpu"
    )

    print(
        "Device:",
        device,
    )

    
    # Load pretrained GPT-2
    

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

    
    # Create small DataLoaders
    
    
    # batch_size=2 keeps this inspection lightweight.
    
    # This is NOT our final training batch size.

    (
        train_loader,
        eval_loader,
        test_loader,
    ) = create_dataloaders(
        batch_size=2,
        num_workers=0,
    )

    
    # Inspect one real batch
    

    batch = next(
        iter(train_loader)
    )

    print("\n" + "=" * 70)

    print(
        "Single-batch metric test"
    )

    print("=" * 70)

    print(
        "Input shape:",
        batch["input_ids"].shape,
    )

    print(
        "Sequence lengths:",
        batch["lengths"],
    )

    print(
        "EOS indices:",
        batch["eos_indices"],
    )

    print(
        "True labels:",
        batch["labels"],
    )

    print(
        "Sources:",
        batch["sources"],
    )

    
    # Obtain logits
    

    model.eval()

    with torch.no_grad():

        logits = classification_forward(
            model=model,
            input_ids=(
                batch[
                    "input_ids"
                ].to(device)
            ),
            eos_indices=(
                batch[
                    "eos_indices"
                ].to(device)
            ),
        )

    labels = batch[
        "labels"
    ].to(device)

    predictions = torch.argmax(
        logits,
        dim=-1,
    )

    accuracy = (
        accuracy_from_logits(
            logits,
            labels,
        )
    )

    
    # Calculate batch loss
    

    with torch.no_grad():

        loss = calc_loss_batch(
            batch=batch,
            model=model,
            device=device,
        )

    print(
        "\nClassification logits:"
    )

    print(
        logits
    )

    print(
        "\nPredictions:",
        predictions,
    )

    print(
        "True labels:",
        labels,
    )

    print(
        f"\nBatch loss: "
        f"{loss.item():.4f}"
    )

    print(
        f"Batch accuracy: "
        f"{accuracy * 100:.2f}%"
    )


    # Sanity checks
    

    assert (
        logits.shape
        == torch.Size(
            [
                labels.shape[0],
                2,
            ]
        )
    )

    assert (
        loss.ndim == 0
    )

    assert math.isfinite(
        loss.item()
    )

    assert (
        loss.item() >= 0
    )

    assert (
        0.0
        <= accuracy
        <= 1.0
    )

    print(
        "\nSingle-batch checks passed."
    )

    
    # Test DataLoader-level metric functions
    
    
    # Only three batches are used here.
    
    # We are testing the metric pipeline, not evaluating the
    # classifier scientifically yet.

    print("\n" + "=" * 70)

    print(
        "DataLoader metric test"
    )

    print("=" * 70)

    num_test_batches = 3

    train_loss = calc_loss_loader(
        data_loader=train_loader,
        model=model,
        device=device,
        num_batches=num_test_batches,
    )

    eval_loss = calc_loss_loader(
        data_loader=eval_loader,
        model=model,
        device=device,
        num_batches=num_test_batches,
    )

    train_accuracy = (
        calc_accuracy_loader(
            data_loader=train_loader,
            model=model,
            device=device,
            num_batches=num_test_batches,
        )
    )

    eval_accuracy = (
        calc_accuracy_loader(
            data_loader=eval_loader,
            model=model,
            device=device,
            num_batches=num_test_batches,
        )
    )

    print(
        f"Train loss "
        f"({num_test_batches} batches): "
        f"{train_loss:.4f}"
    )

    print(
        f"Eval loss "
        f"({num_test_batches} batches): "
        f"{eval_loss:.4f}"
    )

    print(
        f"Train accuracy "
        f"({num_test_batches} batches): "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Eval accuracy "
        f"({num_test_batches} batches): "
        f"{eval_accuracy * 100:.2f}%"
    )

    
    # Final sanity checks
    

    assert math.isfinite(
        train_loss
    )

    assert math.isfinite(
        eval_loss
    )

    assert train_loss >= 0
    assert eval_loss >= 0

    assert (
        0.0
        <= train_accuracy
        <= 1.0
    )

    assert (
        0.0
        <= eval_accuracy
        <= 1.0
    )

    print("\n" + "=" * 70)

    print(
        "All metric checks "
        "passed successfully."
    )


if __name__ == "__main__":
    main()