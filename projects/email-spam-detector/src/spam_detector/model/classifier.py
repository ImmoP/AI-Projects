"""
Utilities for adapting pretrained GPT-2 to email classification.

The pretrained GPT-2 language model originally predicts one of
50,257 vocabulary tokens at every sequence position.

For spam classification, the language-model output head is replaced
with a two-class output head:

    0 = ham
    1 = spam

Only the following components are trainable:

- the new classification head;
- the final LayerNorm;
- the final Transformer block.

The remaining pretrained GPT-2 parameters stay frozen.
"""

import torch
import torch.nn as nn

from spam_detector.model.gpt import GPTModel

# Classification constants


NUM_CLASSES = 2

HAM_LABEL = 0
SPAM_LABEL = 1



# Configure GPT-2 for classification


def configure_for_classification(
    model: GPTModel,
    num_classes: int = NUM_CLASSES,
    random_seed: int = 123,
) -> GPTModel:
    """
    Convert a pretrained GPT-2 language model into a classifier.

    Steps
    -----
    1. Freeze all pretrained parameters.
    2. Replace the language-model output head.
    3. Unfreeze the final Transformer block.
    4. Unfreeze the final LayerNorm.

    Parameters
    ----------
    model:
        Pretrained GPTModel.

    num_classes:
        Number of classification classes.

        For spam detection:

            0 = ham
            1 = spam

    random_seed:
        Seed used when initializing the new classification head.

    Returns
    -------
    GPTModel
        The modified GPT model.
    """

    if num_classes < 2:
        raise ValueError(
            "num_classes must be at least 2."
        )

    
    # Freeze all pretrained GPT-2 parameters
    

    for parameter in model.parameters():
        parameter.requires_grad = False

    
    # Replace the language-model output head
    
    
    # Before:
    
    #     768 -> 50,257
    
    # After:
    
    #     768 -> 2

    # The new layer is randomly initialized and trainable.

    torch.manual_seed(
        random_seed
    )

    embedding_dimension = (
        model.cfg["emb_dim"]
    )

    model.out_head = nn.Linear(
        in_features=embedding_dimension,
        out_features=num_classes,
    )

    
    # Unfreeze final Transformer block
    

    for parameter in (
        model.trf_blocks[-1].parameters()
    ):
        parameter.requires_grad = True

    
    # Unfreeze final LayerNorm
    

    for parameter in (
        model.final_norm.parameters()
    ):
        parameter.requires_grad = True

    return model



# Extract classification logits


def extract_eos_logits(
    sequence_logits: torch.Tensor,
    eos_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Select the model output at each email's real EOS position.

    Parameters
    ----------
    sequence_logits:
        GPT output with shape:

            [batch_size, sequence_length, num_classes]

    eos_indices:
        EOS position of every email:

            [batch_size]

    Returns
    -------
    torch.Tensor
        Classification logits with shape:

            [batch_size, num_classes]

    Why EOS?
    --------
    Each email has a different real sequence length because we use
    dynamic right-padding.

    Example:

        Email 1:
        TOKEN TOKEN EOS PAD PAD

        Email 2:
        TOKEN TOKEN TOKEN TOKEN EOS

    Therefore, sequence_logits[:, -1, :] would be incorrect for
    Email 1 because the final position belongs to padding.

    Instead, we select the individual EOS position of every email.
    """

    
    # Validate sequence logits
    

    if sequence_logits.ndim != 3:
        raise ValueError(
            "sequence_logits must have shape "
            "[batch_size, sequence_length, num_classes]."
        )

    if eos_indices.ndim != 1:
        raise ValueError(
            "eos_indices must have shape [batch_size]."
        )

    batch_size = (
        sequence_logits.shape[0]
    )

    sequence_length = (
        sequence_logits.shape[1]
    )

    if eos_indices.shape[0] != batch_size:
        raise ValueError(
            "Number of EOS indices must match batch size."
        )

    # Make sure indexing occurs on the same device.
    eos_indices = eos_indices.to(
        sequence_logits.device
    )

    
    # Validate EOS positions
    

    if torch.any(
        eos_indices < 0
    ):
        raise ValueError(
            "EOS indices cannot be negative."
        )

    if torch.any(
        eos_indices >= sequence_length
    ):
        raise ValueError(
            "EOS index exceeds sequence length."
        )

    
    # Select each email's EOS representation
    

    batch_indices = torch.arange(
        batch_size,
        device=sequence_logits.device,
    )

    classification_logits = (
        sequence_logits[
            batch_indices,
            eos_indices,
        ]
    )

    return classification_logits



# Forward pass for classification


def classification_forward(
    model: GPTModel,
    input_ids: torch.Tensor,
    eos_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Perform one GPT-2 classification forward pass.

    Input
    -----
    input_ids:

        [batch_size, sequence_length]

    eos_indices:

        [batch_size]

    Output
    ------
    classification_logits:

        [batch_size, num_classes]
    """

    sequence_logits = model(
        input_ids
    )

    classification_logits = (
        extract_eos_logits(
            sequence_logits=sequence_logits,
            eos_indices=eos_indices,
        )
    )

    return classification_logits