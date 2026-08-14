"""
PyTorch dataset for the spam email classification project.

The dataset connects the cleaned Parquet files with the GPT-2
tokenization pipeline.

For every email, the dataset returns:

- input_ids:
    GPT-2 token IDs of the email.

- label:
    0 = ham
    1 = spam

- eos_index:
    Position of the final <|endoftext|> token.
    This position will later be used for classification.

- was_truncated:
    Indicates whether the original email exceeded the
    configured context length.

- original_token_count:
    Number of tokens before truncation.

Padding is intentionally NOT performed here.

Different emails are allowed to have different sequence lengths.
Dynamic padding will be applied later by a custom collate function
when several emails are combined into one batch.
"""

from pathlib import Path

import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset

from spam_detector.model.tokenization import (
    DEFAULT_CONTEXT_LENGTH,
    tokenize_email,
)

# Columns that must exist in every model-ready Parquet file.
REQUIRED_COLUMNS = {
    "sender",
    "subject",
    "text",
    "label",
    "source",
    "source_split",
}


class SpamEmailDataset(Dataset):
    """
    PyTorch Dataset for spam email classification.

    Parameters
    ----------
    parquet_path:
        Path to one of the cleaned Parquet files.

    tokenizer:
        GPT-2 tokenizer used to convert email text into token IDs.

    context_length:
        Maximum number of tokens allowed for one email.
        GPT-2 supports at most 1024 tokens.
    """

    def __init__(
        self,
        parquet_path: Path,
        tokenizer: tiktoken.Encoding,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ):
        # Store the path for debugging and documentation.
        self.parquet_path = Path(parquet_path)

        # Make sure the requested file actually exists.
        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset file does not exist: "
                f"{self.parquet_path}"
            )

        # Load the cleaned email dataset.
        self.data = pd.read_parquet(
            self.parquet_path
        )

        # Check whether all required columns are present.
        missing_columns = (
            REQUIRED_COLUMNS
            - set(self.data.columns)
        )

        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        # Labels must be available for every email.
        if self.data["label"].isna().any():
            raise ValueError(
                "Dataset contains missing labels."
            )

        # Store the tokenizer.
        self.tokenizer = tokenizer

        # Store the maximum allowed sequence length.
        self.context_length = context_length


    def __len__(self) -> int:
        """
        Return the number of emails in the dataset.

        This method is required by PyTorch Dataset.
        """
        return len(self.data)


    def __getitem__(self, index: int) -> dict:
        """
        Load and tokenize one email.

        PyTorch calls this method when accessing an item such as:

            dataset[0]

        The email is tokenized only when it is requested.
        """
        # Select one email from the DataFrame.
        row = self.data.iloc[index]

        # Convert sender, subject, and body into GPT-2 token IDs.
        tokenized = tokenize_email(
            sender=row["sender"],
            subject=row["subject"],
            text=row["text"],
            tokenizer=self.tokenizer,
            context_length=self.context_length,
        )

        # Convert the token IDs into a PyTorch tensor.
        input_ids = torch.tensor(
            tokenized.input_ids,
            dtype=torch.long,
        )

        # Convert the class label into a PyTorch tensor.
        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        # The EOS token is always the final real token.
        #
        # Example:
        #
        # input_ids:
        # [15496, 11, 314, 716, 50256]
        #
        # positions:
        #    0    1    2    3      4
        #
        # eos_index = 4
        eos_index = len(tokenized.input_ids) - 1

        return {
            "input_ids": input_ids,
            "label": label,
            "eos_index": eos_index,

            # The following values are useful for debugging
            # and later evaluation.
            "was_truncated": tokenized.was_truncated,
            "original_token_count":
                tokenized.original_token_count,

            # Keep the source metadata so that performance can
            # later be evaluated separately for public and
            # private emails.
            "source": str(row["source"]),
            "source_split": str(
                row["source_split"]
            ),
        }