"""
Analyze the lengths of the formatted training emails.

This analysis is necessary before choosing a maximum token length
or context length for the model.

Data flow:

combined_train_clean.parquet
        ↓
pd.read_parquet()
        ↓
DataFrame
        ↓
add_model_text()
        ↓
format_email() for every row
        ↓
model_text
        ↓
character_count, word_count, and token_count
        ↓
check for empty model inputs
        ↓
calculate percentiles
        ↓
identify extremely long emails
        ↓
compare possible context lengths
"""

import pandas as pd
import tiktoken

from spam_detector.model.text_formatting import format_email
from spam_detector.paths import DATA_DIR

# Number of training emails used for the token-length analysis.
TOKEN_SAMPLE_SIZE = 20_000

# Candidate maximum context lengths that will be compared.
CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048]

# Percentiles used for the length statistics.
LENGTH_PERCENTILES = [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
]

# Number of extremely long emails shown in the output.
NUMBER_OF_LONGEST_EMAILS = 5

# Number of characters shown from the beginning and end of long emails.
PREVIEW_LENGTH = 1_000


def add_model_text(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create one formatted model input for every email.

    The sender, subject, and body columns are combined by using
    the format_email() function from text_formatting.py.

    A copy of the DataFrame is created so that the original
    DataFrame is not modified directly.
    """
    dataframe = dataframe.copy()

    dataframe["model_text"] = [
        format_email(sender, subject, text)
        for sender, subject, text in zip(
            dataframe["sender"],
            dataframe["subject"],
            dataframe["text"],
        )
    ]

    return dataframe


def count_tokens(text: str, tokenizer) -> int:
    """
    Count how many GPT-2 tokens are contained in one text.

    This function is used only for length analysis. It does not
    create the final padded and truncated training inputs yet.
    """
    token_ids = tokenizer.encode(
        text,
        allowed_special=set(),
    )

    return len(token_ids)


def print_basic_length_statistics(
    dataframe: pd.DataFrame,
) -> None:
    """
    Print character and word-length statistics for the training set.
    """
    dataframe["character_count"] = (
        dataframe["model_text"].str.len()
    )

    dataframe["word_count"] = (
        dataframe["model_text"]
        .str.split()
        .str.len()
    )

    empty_model_texts = (
        dataframe["model_text"]
        .str.strip()
        .eq("")
        .sum()
    )

    print("Empty model texts:")
    print(empty_model_texts)

    print("\nCharacter lengths:")
    print(
        dataframe["character_count"].describe(
            percentiles=LENGTH_PERCENTILES,
        )
    )

    print("\nWord lengths:")
    print(
        dataframe["word_count"].describe(
            percentiles=LENGTH_PERCENTILES,
        )
    )


def print_longest_email_examples(
    dataframe: pd.DataFrame,
) -> None:
    """
    Print information and short previews for the longest emails.

    Only the first and last PREVIEW_LENGTH characters are displayed
    so that extremely long emails are not printed completely.
    """
    longest = dataframe.nlargest(
        NUMBER_OF_LONGEST_EMAILS,
        "character_count",
    )

    print("\nLongest examples:")

    print(
        longest[
            [
                "character_count",
                "word_count",
                "label",
                "source",
            ]
        ].to_string(index=False)
    )

    print("\nLongest email previews:")

    for _, row in longest.iterrows():
        model_text = row["model_text"]

        print("\n" + "=" * 80)
        print(f"Source: {row['source']}")
        print(f"Label: {row['label']}")
        print(f"Characters: {row['character_count']}")
        print(f"Words: {row['word_count']}")

        print(f"\nFirst {PREVIEW_LENGTH} characters:")
        print(model_text[:PREVIEW_LENGTH])

        print(f"\nLast {PREVIEW_LENGTH} characters:")
        print(model_text[-PREVIEW_LENGTH:])


def analyze_token_lengths(
    dataframe: pd.DataFrame,
) -> None:
    """
    Analyze GPT-2 token lengths using a random training sample.

    A sample is used because several emails are extremely large.
    The fixed random state ensures that the same sample is selected
    every time the script is executed.
    """
    tokenizer = tiktoken.get_encoding("gpt2")

    sample_size = min(
        TOKEN_SAMPLE_SIZE,
        len(dataframe),
    )

    sample = dataframe.sample(
        n=sample_size,
        random_state=42,
    ).copy()

    sample["token_count"] = (
        sample["model_text"]
        .map(
            lambda value: count_tokens(
                value,
                tokenizer,
            )
        )
    )

    print(
        f"\nToken lengths — random sample of "
        f"{sample_size:,} training emails:"
    )

    print(
        sample["token_count"].describe(
            percentiles=LENGTH_PERCENTILES,
        )
    )

    print("\nShare above candidate context lengths:")

    for context_length in CONTEXT_LENGTHS:
        share_above = (
            sample["token_count"] > context_length
        ).mean()

        number_above = (
            sample["token_count"] > context_length
        ).sum()

        print(
            f"Above {context_length:4d} tokens: "
            f"{share_above:.2%} "
            f"({number_above:,} emails)"
        )

    print("\nLongest sampled emails by token count:")

    print(
        sample.nlargest(
            10,
            "token_count",
        )[
            [
                "token_count",
                "character_count",
                "word_count",
                "label",
                "source",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    """
    Load the clean training dataset and run all length analyses.
    """
    input_path = (
        DATA_DIR / "combined_train_clean.parquet"
    )

    print(f"Loading training data from:\n{input_path}")

    train = pd.read_parquet(input_path)

    print(f"\nTraining emails loaded: {len(train):,}")

    # Convert sender, subject, and body into one model input.
    train = add_model_text(train)

    # Calculate character and word lengths.
    print_basic_length_statistics(train)

    # Inspect the largest email examples.
    print_longest_email_examples(train)

    # Analyze token lengths using the GPT-2 tokenizer.
    analyze_token_lengths(train)


if __name__ == "__main__":
    main()