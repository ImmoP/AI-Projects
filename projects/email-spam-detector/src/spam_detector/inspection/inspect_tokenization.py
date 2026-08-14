"""
Inspect the new email tokenization strategy.

This script does not modify any dataset.
It only tokenizes a small sample and checks whether the
truncation logic behaves as expected.
"""

import pandas as pd

from spam_detector.model.tokenization import (
    DEFAULT_CONTEXT_LENGTH,
    get_gpt2_tokenizer,
    tokenize_email,
)
from spam_detector.paths import DATA_DIR


def main() -> None:
    # Load the clean training dataset.
    train = pd.read_parquet(
        DATA_DIR / "combined_train_clean.parquet"
    )

    # Load the GPT-2 tokenizer.
    tokenizer = get_gpt2_tokenizer()

    # Select a small reproducible random sample.
    sample = train.sample(
        n=20,
        random_state=42,
    )

    # Count how many sampled emails require truncation.
    truncated_count = 0

    # Inspect every email in the sample.
    for _, row in sample.iterrows():

        # Tokenize one email.
        result = tokenize_email(
            sender=row["sender"],
            subject=row["subject"],
            text=row["text"],
            tokenizer=tokenizer,
            context_length=DEFAULT_CONTEXT_LENGTH,
        )

        # Count truncated emails.
        if result.was_truncated:
            truncated_count += 1

        print("\n" + "=" * 70)

        print(f"Label: {row['label']}")
        print(f"Source: {row['source']}")

        print(
            "Original tokens:",
            result.original_token_count,
        )

        print(
            "Final tokens:",
            result.final_token_count,
        )

        print(
            "Truncated:",
            result.was_truncated,
        )

        print(
            "Metadata truncated:",
            result.metadata_truncated,
        )

        print(
            "Last token is EOS:",
            result.input_ids[-1] == 50256,
        )

        # Make sure the sequence never exceeds
        # the configured GPT-2 context length.
        assert (
            len(result.input_ids)
            <= DEFAULT_CONTEXT_LENGTH
        )

        # Make sure every sequence ends with the
        # GPT-2 end-of-text token.
        assert result.input_ids[-1] == 50256

        # If the email was truncated, decode the
        # resulting tokens and inspect both ends.
        if result.was_truncated:
            decoded = tokenizer.decode(
                result.input_ids[:-1]
            )

            print("\nDecoded truncated email:")

            print("\nFirst 500 characters:")
            print(decoded[:500])

            print("\nLast 500 characters:")
            print(decoded[-500:])

    # Print a final summary after all emails
    # in the sample have been inspected.
    print("\n" + "=" * 70)

    print(
        f"Truncated emails in sample: "
        f"{truncated_count}/{len(sample)}"
    )

    print(
        "All tokenized emails respect the "
        "context length."
    )


if __name__ == "__main__":
    main()