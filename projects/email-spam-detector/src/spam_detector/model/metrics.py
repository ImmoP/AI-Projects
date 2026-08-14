"""
Loss and accuracy utilities for GPT-2 email classification.

The classifier outputs two logits for every token position:

    class 0 = ham
    class 1 = spam

Because email batches use dynamic right-padding, classification is
performed at the real EOS position of each email instead of simply
using the final padded tensor position.
"""

import torch
import torch.nn.functional as F

from spam_detector.model.classifier import (
    classification_forward,
)

# Single-batch classification loss


def calc_loss_batch(
    batch: dict,
    model,
    device: torch.device,
) -> torch.Tensor:
    """
    Calculate cross-entropy loss for one email batch.

    This function intentionally does NOT use torch.no_grad().

    That is important because the same function will later be used
    during training, where gradients are required for backpropagation.

    Parameters
    ----------
    batch:
        Batch created by dynamic_email_collate().

    model:
        GPT-2 classification model.

    device:
        Device on which the model and tensors should be evaluated.

    Returns
    -------
    torch.Tensor
        Scalar cross-entropy loss.
    """

    
    # Move required tensors to the model device
    

    input_ids = batch[
        "input_ids"
    ].to(device)

    eos_indices = batch[
        "eos_indices"
    ].to(device)

    labels = batch[
        "labels"
    ].to(device)

    
    # Obtain one pair of class logits per email
    
    #
    # Shape:
    #
    #     [batch_size, 2]

    logits = classification_forward(
        model=model,
        input_ids=input_ids,
        eos_indices=eos_indices,
    )

    
    # Cross-entropy classification loss
    
    #
    # CrossEntropyLoss expects:
    #
    # logits:
    #     [batch_size, num_classes]
    #
    # labels:
    #     [batch_size]
    #
    # Labels are integer class indices:
    #
    #     0 = ham
    #     1 = spam

    loss = F.cross_entropy(
        logits,
        labels,
    )

    return loss



# Accuracy from logits


def accuracy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Calculate classification accuracy from logits.

    Accuracy is:

        number of correct predictions
        -----------------------------
        total number of predictions
    """

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape "
            "[batch_size, num_classes]."
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels must have shape [batch_size]."
        )

    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "Number of logits and labels must match."
        )

    if labels.numel() == 0:
        return float("nan")

    predictions = torch.argmax(
        logits,
        dim=-1,
    )

    correct = (
        predictions == labels
    ).sum().item()

    return (
        correct
        / labels.numel()
    )



# Loss over a DataLoader


def calc_loss_loader(
    data_loader,
    model,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """
    Calculate mean classification loss over a DataLoader.

    If num_batches is provided, only that many batches are used.

    During evaluation, gradient tracking is disabled to reduce
    memory usage and computation time.
    """

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(
            data_loader
        )
    else:
        if num_batches < 1:
            raise ValueError(
                "num_batches must be at least 1."
            )

        num_batches = min(
            num_batches,
            len(data_loader),
        )

    # Remember whether the model was previously
    # in training mode.
    was_training = model.training

    model.eval()

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():

        for batch_index, batch in enumerate(
            data_loader
        ):

            if batch_index >= num_batches:
                break

            batch_size = batch[
                "labels"
            ].shape[0]

            loss = calc_loss_batch(
                batch=batch,
                model=model,
                device=device,
            )

            # calc_loss_batch returns the MEAN loss of the batch.
            #
            # Multiply by the number of examples so that the final
            # result is correctly weighted even if the last batch
            # contains fewer examples.
            total_loss += (
                loss.item()
                * batch_size
            )

            total_examples += (
                batch_size
            )

    # Restore the previous model mode.
    if was_training:
        model.train()

    if total_examples == 0:
        return float("nan")

    return (
        total_loss
        / total_examples
    )



# Accuracy over a DataLoader


def calc_accuracy_loader(
    data_loader,
    model,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """
    Calculate classification accuracy over a DataLoader.

    Accuracy is calculated over all examples in the selected
    batches, not as an unweighted average of batch accuracies.
    """

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(
            data_loader
        )
    else:
        if num_batches < 1:
            raise ValueError(
                "num_batches must be at least 1."
            )

        num_batches = min(
            num_batches,
            len(data_loader),
        )

    was_training = model.training

    model.eval()

    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():

        for batch_index, batch in enumerate(
            data_loader
        ):

            if batch_index >= num_batches:
                break

            input_ids = batch[
                "input_ids"
            ].to(device)

            eos_indices = batch[
                "eos_indices"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            logits = classification_forward(
                model=model,
                input_ids=input_ids,
                eos_indices=eos_indices,
            )

            predictions = torch.argmax(
                logits,
                dim=-1,
            )

            correct_predictions += (
                predictions
                == labels
            ).sum().item()

            total_examples += (
                labels.numel()
            )

    if was_training:
        model.train()

    if total_examples == 0:
        return float("nan")

    return (
        correct_predictions
        / total_examples
    )