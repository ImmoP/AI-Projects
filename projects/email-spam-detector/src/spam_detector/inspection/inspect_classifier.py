"""
Inspect the pretrained GPT-2 spam classifier.

This script verifies that:

- pretrained GPT-2 can be converted into a classifier;
- the output head has two classes;
- most GPT-2 parameters are frozen;
- the final Transformer block is trainable;
- the final LayerNorm is trainable;
- a real email batch produces logits with shape [batch_size, 2];
- EOS-based classification works with dynamic padding.
"""

import torch

from spam_detector.model.classifier import (
    NUM_CLASSES,
    classification_forward,
    configure_for_classification,
)
from spam_detector.model.dataloaders import (
    create_dataloaders,
)
from spam_detector.model.pretrained import (
    create_pretrained_gpt2_small,
)


def count_parameters(
    model,
):
    """
    Count total and trainable model parameters.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    return (
        total_parameters,
        trainable_parameters,
    )


def main() -> None:

    # ----------------------------------------------------------
    # Load pretrained GPT-2
    # ----------------------------------------------------------

    print(
        "Loading pretrained GPT-2..."
    )

    model = (
        create_pretrained_gpt2_small()
    )

    
    # Convert GPT-2 into classifier
    

    model = configure_for_classification(
        model
    )

    model.eval()

    
    # Inspect output head
    

    print("\n" + "=" * 70)

    print(
        "Classification output head:"
    )

    print(
        model.out_head
    )

    assert (
        model.out_head.in_features
        == 768
    )

    assert (
        model.out_head.out_features
        == NUM_CLASSES
    )

    
    # Count parameters
    

    (
        total_parameters,
        trainable_parameters,
    ) = count_parameters(
        model
    )

    frozen_parameters = (
        total_parameters
        - trainable_parameters
    )

    print(
        "\nTotal parameters:",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    print(
        "Frozen parameters:",
        f"{frozen_parameters:,}",
    )

    print(
        "Trainable percentage:",
        f"{100 * trainable_parameters / total_parameters:.2f}%",
    )

    
    # Verify frozen embeddings
    

    assert all(
        not parameter.requires_grad
        for parameter
        in model.tok_emb.parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter
        in model.pos_emb.parameters()
    )

    
    # Verify earlier Transformer blocks are frozen
    

    for block in (
        model.trf_blocks[:-1]
    ):
        assert all(
            not parameter.requires_grad
            for parameter
            in block.parameters()
        )

    
    # Verify last Transformer block is trainable
    

    assert all(
        parameter.requires_grad
        for parameter
        in model.trf_blocks[
            -1
        ].parameters()
    )

    
    # Verify final LayerNorm is trainable
    

    assert all(
        parameter.requires_grad
        for parameter
        in model.final_norm.parameters()
    )

    
    # Verify new output head is trainable
    

    assert all(
        parameter.requires_grad
        for parameter
        in model.out_head.parameters()
    )

    print(
        "\nLayer freezing checks passed."
    )

    
    # Create a SMALL real email batch
    
    #
    # batch_size=2 is intentional here.
    #
    # We only want to verify the end-to-end forward pass on the
    # Mac, not perform actual training yet.

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

    input_ids = batch[
        "input_ids"
    ]

    eos_indices = batch[
        "eos_indices"
    ]

    labels = batch[
        "labels"
    ]

    
    # Print real batch information


    print("\n" + "=" * 70)

    print(
        "Input shape:",
        input_ids.shape,
    )

    print(
        "Sequence lengths:",
        batch["lengths"],
    )

    print(
        "EOS indices:",
        eos_indices,
    )

    print(
        "Labels:",
        labels,
    )

    print(
        "Sources:",
        batch["sources"],
    )

    
    # Classification forward pass
    

    with torch.no_grad():
        logits = (
            classification_forward(
                model=model,
                input_ids=input_ids,
                eos_indices=eos_indices,
            )
        )

    print(
        "\nClassification logits:"
    )

    print(
        logits
    )

    print(
        "\nClassification logits shape:",
        logits.shape,
    )

    
    # Convert logits into predicted classes
    

    predictions = torch.argmax(
        logits,
        dim=-1,
    )

    print(
        "\nPredicted classes:",
        predictions,
    )

    print(
        "True labels:",
        labels,
    )

    
    # Sanity checks
    

    assert (
        logits.shape
        == torch.Size(
            [input_ids.shape[0], 2]
        )
    )

    assert (
        predictions.shape
        == labels.shape
    )

    assert torch.all(
        (
            predictions == 0
        )
        |
        (
            predictions == 1
        )
    )

    print("\n" + "=" * 70)

    print(
        "All classifier checks "
        "passed successfully."
    )


if __name__ == "__main__":
    main()