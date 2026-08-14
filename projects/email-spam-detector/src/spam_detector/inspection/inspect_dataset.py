"""
Inspect the SpamEmailDataset implementation.

This script checks that individual emails can be loaded and
tokenized correctly before a DataLoader is created.
"""

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
    # Load the GPT-2 tokenizer.
    tokenizer = get_gpt2_tokenizer()

    # Create the training dataset.
    train_dataset = SpamEmailDataset(
        parquet_path=(
            DATA_DIR
            / "combined_train_clean.parquet"
        ),
        tokenizer=tokenizer,
        context_length=DEFAULT_CONTEXT_LENGTH,
    )

    print(
        "Number of training emails:",
        len(train_dataset),
    )

    # Select several examples from different positions
    # in the dataset.
    example_indices = [
        0,
        1,
        42,
        1_000,
        10_000,
    ]

    for index in example_indices:
        sample = train_dataset[index]

        print("\n" + "=" * 70)

        print(f"Dataset index: {index}")

        print(
            "Sequence length:",
            len(sample["input_ids"]),
        )

        print(
            "Label:",
            sample["label"].item(),
        )

        print(
            "EOS index:",
            sample["eos_index"],
        )

        print(
            "Original token count:",
            sample["original_token_count"],
        )

        print(
            "Truncated:",
            sample["was_truncated"],
        )

        print(
            "Source:",
            sample["source"],
        )

        print(
            "Source split:",
            sample["source_split"],
        )

        print(
            "Last token:",
            sample["input_ids"][-1].item(),
        )

        print(
            "Last token is EOS:",
            (
                sample["input_ids"][-1].item()
                == GPT2_EOS_TOKEN_ID
            ),
        )

        
        # Sanity checks
        

        # Every sequence must fit into GPT-2's
        # configured context length.
        assert (
            len(sample["input_ids"])
            <= DEFAULT_CONTEXT_LENGTH
        )

        # eos_index must point to the final token.
        assert (
            sample["eos_index"]
            == len(sample["input_ids"]) - 1
        )

        # The final token must be GPT-2's EOS token.
        assert (
            sample["input_ids"][
                sample["eos_index"]
            ].item()
            == GPT2_EOS_TOKEN_ID
        )

        # Labels must be either ham or spam.
        assert (
            sample["label"].item()
            in {0, 1}
        )

    print("\n" + "=" * 70)

    print(
        "All dataset checks passed successfully."
    )


if __name__ == "__main__":
    main()