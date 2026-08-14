"""
Inspect the dynamic batch padding implementation.

This script verifies that:

- emails with different lengths can be combined into one batch;
- shorter emails are padded correctly;
- EOS indices remain correct after padding;
- the attention mask correctly distinguishes real tokens
  from padding tokens;
- labels and metadata are preserved.
"""

import torch

from spam_detector.model.collate import (
    GPT2_PAD_TOKEN_ID,
    dynamic_email_collate,
)
from spam_detector.model.dataset import (
    SpamEmailDataset,
)
from spam_detector.model.tokenization import (
    DEFAULT_CONTEXT_LENGTH,
    GPT2_EOS_TOKEN_ID,
    get_gpt2_tokenizer,
)
from spam_detector.paths import DATA_DIR


def main() -> None:
    
    # Create the dataset
    

    tokenizer = get_gpt2_tokenizer()

    dataset = SpamEmailDataset(
        parquet_path=(
            DATA_DIR
            / "combined_train_clean.parquet"
        ),
        tokenizer=tokenizer,
        context_length=DEFAULT_CONTEXT_LENGTH,
    )

    
    # Select several emails with different sequence lengths
    

    example_indices = [
        0,
        1,
        42,
        1_000,
        10_000,
    ]

    samples = [
        dataset[index]
        for index in example_indices
    ]

    print("Individual sequence lengths:")

    for index, sample in zip(
        example_indices,
        samples,
    ):
        print(
            f"Index {index:5d}: "
            f"{len(sample['input_ids'])} tokens"
        )

    
    # Create one dynamically padded batch
    

    batch = dynamic_email_collate(
        samples
    )

    print("\n" + "=" * 70)

    print("Batch shapes:")

    print(
        "input_ids:",
        batch["input_ids"].shape,
    )

    print(
        "labels:",
        batch["labels"].shape,
    )

    print(
        "eos_indices:",
        batch["eos_indices"].shape,
    )

    print(
        "attention_mask:",
        batch["attention_mask"].shape,
    )

    
    # Show batch-level information
    

    print("\nSequence lengths:")
    print(batch["lengths"])

    print("\nEOS indices:")
    print(batch["eos_indices"])

    print("\nLabels:")
    print(batch["labels"])

    print("\nSources:")
    print(batch["sources"])

    
    # Verify every sample
    

    for batch_index in range(
        len(samples)
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

        print("\n" + "-" * 70)

        print(
            f"Batch item: {batch_index}"
        )

        print(
            f"Real sequence length: "
            f"{sequence_length}"
        )

        print(
            f"EOS index: "
            f"{eos_index}"
        )

        print(
            "Attention mask sum:",
            batch["attention_mask"][
                batch_index
            ].sum().item(),
        )

        
        # Check 1:
        # EOS index must be the final real token.
        

        assert (
            eos_index
            == sequence_length - 1
        )

        
        # Check 2:
        # The token at eos_index must be EOS.
        

        assert (
            batch["input_ids"][
                batch_index,
                eos_index,
            ].item()
            == GPT2_EOS_TOKEN_ID
        )

        
        # Check 3:
        # Number of ones in the attention mask must equal
        # the number of real tokens.
        

        assert (
            batch["attention_mask"][
                batch_index
            ].sum().item()
            == sequence_length
        )

        
        # Check 4:
        # All positions AFTER EOS must contain padding tokens.
        

        padding_after_eos = (
            batch["input_ids"][
                batch_index,
                eos_index + 1:,
            ]
        )

        if padding_after_eos.numel() > 0:
            assert torch.all(
                padding_after_eos
                == GPT2_PAD_TOKEN_ID
            )

        
        # Check 5:
        # Attention-mask positions AFTER EOS must be zero.
        

        mask_after_eos = (
            batch["attention_mask"][
                batch_index,
                eos_index + 1:,
            ]
        )

        if mask_after_eos.numel() > 0:
            assert torch.all(
                mask_after_eos == 0
            )

    
    # Final batch-level checks
    

    expected_max_length = max(
        len(sample["input_ids"])
        for sample in samples
    )

    assert (
        batch["input_ids"].shape[1]
        == expected_max_length
    )

    assert (
        batch["attention_mask"].shape
        == batch["input_ids"].shape
    )

    assert (
        batch["labels"].shape[0]
        == len(samples)
    )

    print("\n" + "=" * 70)

    print(
        "All dynamic padding checks "
        "passed successfully."
    )


if __name__ == "__main__":
    main()