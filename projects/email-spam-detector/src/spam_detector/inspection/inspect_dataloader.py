"""
Inspect the spam email DataLoaders.

This script verifies that:

- training, evaluation, and test DataLoaders can be created;
- batches contain the expected tensors;
- dynamic padding works inside a real DataLoader;
- EOS positions remain correct;
- labels are valid;
- sequence lengths never exceed GPT-2's context length.
"""

import torch

from spam_detector.model.collate import (
    GPT2_PAD_TOKEN_ID,
)
from spam_detector.model.dataloaders import (
    DEFAULT_BATCH_SIZE,
    create_dataloaders,
)
from spam_detector.model.tokenization import (
    DEFAULT_CONTEXT_LENGTH,
    GPT2_EOS_TOKEN_ID,
)


def inspect_batch(
    batch: dict,
    name: str,
) -> None:
    """
    Inspect and validate one DataLoader batch.
    """

    print("\n" + "=" * 70)

    print(f"{name} batch")

    print("=" * 70)

    
    # Print tensor shapes
    

    print(
        "input_ids shape:",
        batch["input_ids"].shape,
    )

    print(
        "labels shape:",
        batch["labels"].shape,
    )

    print(
        "eos_indices shape:",
        batch["eos_indices"].shape,
    )

    print(
        "attention_mask shape:",
        batch["attention_mask"].shape,
    )

    
    # Print batch information
    

    print(
        "\nSequence lengths:"
    )
    print(
        batch["lengths"]
    )

    print(
        "\nEOS indices:"
    )
    print(
        batch["eos_indices"]
    )

    print(
        "\nLabels:"
    )
    print(
        batch["labels"]
    )

    print(
        "\nTruncated:"
    )
    print(
        batch["was_truncated"]
    )

    print(
        "\nSources:"
    )
    print(
        batch["sources"]
    )

    
    # Basic shape checks
    

    batch_size = (
        batch["input_ids"].shape[0]
    )

    batch_sequence_length = (
        batch["input_ids"].shape[1]
    )

    assert (
        batch_size
        <= DEFAULT_BATCH_SIZE
    )

    assert (
        batch_sequence_length
        <= DEFAULT_CONTEXT_LENGTH
    )

    assert (
        batch["attention_mask"].shape
        == batch["input_ids"].shape
    )

    assert (
        batch["labels"].shape[0]
        == batch_size
    )

    assert (
        batch["eos_indices"].shape[0]
        == batch_size
    )

    
    # Validate every email in the batch
    

    for batch_index in range(
        batch_size
    ):
        sequence_length = (
            batch["lengths"][
                batch_index
            ].item()
        )

        eos_index = (
            batch["eos_indices"][
                batch_index
            ].item()
        )

        # EOS must be the last real token.
        assert (
            eos_index
            == sequence_length - 1
        )

        # The token at eos_index must really be EOS.
        assert (
            batch["input_ids"][
                batch_index,
                eos_index,
            ].item()
            == GPT2_EOS_TOKEN_ID
        )

        # Number of real tokens in the attention mask
        # must equal the stored sequence length.
        assert (
            batch["attention_mask"][
                batch_index
            ].sum().item()
            == sequence_length
        )

        # Label must be either ham or spam.
        label = (
            batch["labels"][
                batch_index
            ].item()
        )

        assert label in {0, 1}

        
        # Validate padding after EOS
        

        padding_tokens = (
            batch["input_ids"][
                batch_index,
                eos_index + 1:,
            ]
        )

        if padding_tokens.numel() > 0:
            assert torch.all(
                padding_tokens
                == GPT2_PAD_TOKEN_ID
            )

        padding_mask = (
            batch["attention_mask"][
                batch_index,
                eos_index + 1:,
            ]
        )

        if padding_mask.numel() > 0:
            assert torch.all(
                padding_mask == 0
            )

    print(
        "\nBatch checks passed."
    )


def main() -> None:
    
    # Create DataLoaders
    

    train_loader, eval_loader, test_loader = (
        create_dataloaders(
            batch_size=DEFAULT_BATCH_SIZE,
            num_workers=0,
            context_length=DEFAULT_CONTEXT_LENGTH,
        )
    )

    
    # Print dataset sizes
    

    print(
        "Training emails:",
        len(train_loader.dataset),
    )

    print(
        "Evaluation emails:",
        len(eval_loader.dataset),
    )

    print(
        "Test emails:",
        len(test_loader.dataset),
    )

    print()

    print(
        "Training batches:",
        len(train_loader),
    )

    print(
        "Evaluation batches:",
        len(eval_loader),
    )

    print(
        "Test batches:",
        len(test_loader),
    )

    
    # Load one actual batch from every DataLoader
    

    train_batch = next(
        iter(train_loader)
    )

    eval_batch = next(
        iter(eval_loader)
    )

    test_batch = next(
        iter(test_loader)
    )

    
    # Inspect the batches
    

    inspect_batch(
        train_batch,
        "Training",
    )

    inspect_batch(
        eval_batch,
        "Evaluation",
    )

    inspect_batch(
        test_batch,
        "Test",
    )

    print("\n" + "=" * 70)

    print(
        "All DataLoader checks "
        "passed successfully."
    )


if __name__ == "__main__":
    main()