"""
Batch collation utilities for spam email classification.

Individual emails in SpamEmailDataset have different sequence lengths.

This module combines multiple dataset samples into one PyTorch batch
by dynamically padding all sequences to the length of the longest
email in the current batch.

Example:

Email A: 200 tokens
Email B: 500 tokens
Email C: 800 tokens

After dynamic padding:

Email A: 200 real tokens + 600 padding tokens
Email B: 500 real tokens + 300 padding tokens
Email C: 800 real tokens

The maximum length therefore depends on the current batch instead
of always padding every email to the global 1024-token context length.
"""

import torch

# GPT-2 uses token ID 50256 for <|endoftext|>.
#
# We also use this token as the padding token, following the approach
# used in the book.
GPT2_PAD_TOKEN_ID = 50256


def dynamic_email_collate(
    batch: list[dict],
    pad_token_id: int = GPT2_PAD_TOKEN_ID,
) -> dict:
    """
    Combine multiple email samples into one padded PyTorch batch.

    Parameters
    ----------
    batch:
        List of samples returned by SpamEmailDataset.__getitem__().

        Each item contains:

            input_ids
            label
            eos_index
            was_truncated
            original_token_count
            source
            source_split

    pad_token_id:
        Token ID used to pad shorter sequences.
        GPT-2 uses 50256 (<|endoftext|>) because GPT-2 does not
        define a dedicated padding token.

    Returns
    -------
    dict
        Dictionary containing:

        input_ids:
            Tensor with shape:

                [batch_size, batch_max_length]

        labels:
            Class labels with shape:

                [batch_size]

        eos_indices:
            Position of the real EOS token for every email.

        attention_mask:
            1 for real tokens, including EOS.
            0 for padding tokens.

        lengths:
            Number of real tokens in each email.

        sources:
            Original dataset source for each email.

        source_splits:
            Original data split for each email.

        was_truncated:
            Indicates whether each email was truncated.

        original_token_counts:
            Token counts before truncation.
    """

    
    # Basic validation
    if len(batch) == 0:
        raise ValueError(
            "Cannot create a batch from an empty list."
        )

    # Determine the number of real tokens in every email.
    #
    # Example:
    #
    # [200, 500, 800]
    lengths = [
        len(item["input_ids"])
        for item in batch
    ]

    # Find the longest email in this particular batch.
    #
    # Example:
    #
    # max([200, 500, 800]) -> 800
    batch_max_length = max(lengths)

    batch_size = len(batch)

    
    # Create padded input tensor

    # Initially create a tensor that contains only padding tokens.
    #
    # Shape:
    #
    # [batch_size, batch_max_length]
    #
    # Example:
    #
    # [
    #   [PAD, PAD, PAD, PAD],
    #   [PAD, PAD, PAD, PAD],
    #   [PAD, PAD, PAD, PAD],
    # ]
    input_ids = torch.full(
        size=(
            batch_size,
            batch_max_length,
        ),
        fill_value=pad_token_id,
        dtype=torch.long,
    )

    
    # Create attention mask


    # Start with a mask containing only zeros.
    #
    # 0 = padding position
    # 1 = real token position
    attention_mask = torch.zeros(
        size=(
            batch_size,
            batch_max_length,
        ),
        dtype=torch.long,
    )

    
    # Copy each real token sequence into the padded tensor

    for batch_index, item in enumerate(batch):
        sequence = item["input_ids"]

        sequence_length = len(sequence)

        # Copy the real token sequence into the beginning
        # of the padded row.
        input_ids[
            batch_index,
            :sequence_length,
        ] = sequence

        # Mark all real token positions as 1.
        #
        # This includes the EOS token.
        attention_mask[
            batch_index,
            :sequence_length,
        ] = 1

    
    # Collect labels
    labels = torch.stack(
        [
            item["label"]
            for item in batch
        ]
    )


    # Store the EOS position for every email
    
    eos_indices = torch.tensor(
        [
            item["eos_index"]
            for item in batch
        ],
        dtype=torch.long,
    )

    
    # Store the real sequence lengths

    lengths_tensor = torch.tensor(
        lengths,
        dtype=torch.long,
    )

    
    # Keep metadata for later evaluation
   
    sources = [
        item["source"]
        for item in batch
    ]

    source_splits = [
        item["source_split"]
        for item in batch
    ]

    was_truncated = torch.tensor(
        [
            item["was_truncated"]
            for item in batch
        ],
        dtype=torch.bool,
    )

    original_token_counts = torch.tensor(
        [
            item["original_token_count"]
            for item in batch
        ],
        dtype=torch.long,
    )

    
    # Return the complete batch
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "eos_indices": eos_indices,
        "attention_mask": attention_mask,
        "lengths": lengths_tensor,
        "sources": sources,
        "source_splits": source_splits,
        "was_truncated": was_truncated,
        "original_token_counts":
            original_token_counts,
    }