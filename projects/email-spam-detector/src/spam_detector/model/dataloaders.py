"""
Create PyTorch DataLoaders for spam email classification.

The DataLoaders connect:

clean Parquet files
    ↓
SpamEmailDataset
    ↓
dynamic_email_collate
    ↓
PyTorch batches

Training data is shuffled.

Evaluation and test data are kept in their original order because
there is no reason to shuffle data during evaluation.
"""

import torch
from torch.utils.data import DataLoader

from spam_detector.model.collate import (
    dynamic_email_collate,
)
from spam_detector.model.dataset import (
    SpamEmailDataset,
)
from spam_detector.model.tokenization import (
    DEFAULT_CONTEXT_LENGTH,
    get_gpt2_tokenizer,
)
from spam_detector.paths import DATA_DIR

DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 0
DEFAULT_RANDOM_SEED = 123


def create_dataloaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    random_seed: int = DEFAULT_RANDOM_SEED,
):
    """
    Create training, evaluation, and test DataLoaders.

    Parameters
    ----------
    batch_size:
        Number of emails in one batch.

    num_workers:
        Number of worker processes used for data loading.

        We initially use 0 because tokenization is performed
        on demand inside SpamEmailDataset and this configuration
        is simple and reliable across macOS and Windows.

    context_length:
        Maximum number of GPT-2 tokens per email.

    random_seed:
        Seed used to make training-data shuffling reproducible.

    Returns
    -------
    tuple
        train_loader,
        eval_loader,
        test_loader
    """

    
    # Validate parameters
    

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative."
        )

    
    # Create one shared GPT-2 tokenizer
    

    tokenizer = get_gpt2_tokenizer()

    
    # Create datasets
    

    train_dataset = SpamEmailDataset(
        parquet_path=(
            DATA_DIR
            / "combined_train_clean.parquet"
        ),
        tokenizer=tokenizer,
        context_length=context_length,
    )

    eval_dataset = SpamEmailDataset(
        parquet_path=(
            DATA_DIR
            / "combined_eval_clean.parquet"
        ),
        tokenizer=tokenizer,
        context_length=context_length,
    )

    test_dataset = SpamEmailDataset(
        parquet_path=(
            DATA_DIR
            / "combined_test_clean.parquet"
        ),
        tokenizer=tokenizer,
        context_length=context_length,
    )

    
    # Create reproducible random generator
    

    generator = torch.Generator()

    generator.manual_seed(
        random_seed
    )

    
    # Training DataLoader
    

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,

        # Shuffle the training examples every epoch.
        shuffle=True,

        # Tokenization currently runs in the main process.
        num_workers=num_workers,

        # Do not discard an incomplete final batch.
        drop_last=False,

        # Dynamically pad emails to the longest sequence
        # in the current batch.
        collate_fn=dynamic_email_collate,

        # Make shuffling reproducible.
        generator=generator,
    )

    
    # Evaluation DataLoader
    

    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,

        # Evaluation data does not need random ordering.
        shuffle=False,

        num_workers=num_workers,
        drop_last=False,
        collate_fn=dynamic_email_collate,
    )

    
    # Test DataLoader
    

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=dynamic_email_collate,
    )

    return (
        train_loader,
        eval_loader,
        test_loader,
    )