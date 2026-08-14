"""
Main training script for the GPT-2 email spam classifier.

This file contains the central training configuration and connects:

- pretrained GPT-2;
- the classification head;
- training and validation DataLoaders;
- AdamW optimization;
- gradient accumulation;
- checkpointing.

For a short pipeline test, set MAX_STEPS to a small integer.

For complete training, use:

    MAX_STEPS = None
"""

import torch

from spam_detector.model.classifier import (
    configure_for_classification,
)
from spam_detector.model.dataloaders import (
    create_dataloaders,
)
from spam_detector.model.pretrained import (
    create_pretrained_gpt2_small,
)
from spam_detector.model.training import (
    create_optimizer,
    train_classifier,
)
from spam_detector.paths import (
    PROJECT_ROOT,
)

# Training configuration

RANDOM_SEED = 123

# Maximum number of tokens used for one email.
# GPT-2 supports up to 1024 token positions.
CONTEXT_LENGTH = 1024

# Number of emails processed simultaneously.
# A batch size of 1 minimizes GPU memory usage.
BATCH_SIZE = 1

# Number of worker processes used by the DataLoader.
# Zero is the safest option for our current setup.
NUM_WORKERS = 0

# Accumulate gradients over several micro-batches before
# updating the model parameters.
#
# With:
#
# BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
#
# the effective batch size is 8.
# GRADIENT_ACCUMULATION_STEPS = 8

# AdamW learning rate.
LEARNING_RATE = 5e-5

# AdamW weight decay regularization.
WEIGHT_DECAY = 0.1

# Number of complete passes through the training dataset.
NUM_EPOCHS = 1

# Evaluate train and validation loss every N optimizer steps.
#
# For the first test:
 #   EVAL_FREQ = 1000
#
# For full training later:
#     EVAL_FREQ = 1000
EVAL_FREQ = 1000

# Number of batches used during each intermediate evaluation.
#
# For the first test:
#     EVAL_ITER = 1
#
# For full training later:
#     EVAL_ITER = 100
EVAL_ITER = 100

# Maximum number of optimizer steps.
#
# Keep this at 2 for the first end-to-end test.
#
# For complete training:
#
#     MAX_STEPS = None
MAX_STEPS = None

# Directory in which training checkpoints are stored.
CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)


# Device selection

def get_device() -> torch.device:
    """
    Select the best available PyTorch device.

    Priority:

    1. NVIDIA CUDA GPU
    2. Apple Metal Performance Shaders (MPS)
    3. CPU
    """

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


# Random seed

def set_random_seed(
    seed: int,
) -> None:
    """
    Set PyTorch random seeds for reproducibility.
    """

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# Configuration output

def print_configuration(
    device: torch.device,
) -> None:
    """
    Print the active training configuration.
    """

    effective_batch_size = (
        BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
    )

    print(
        "\nTraining configuration"
    )

    print(
        "-" * 50
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                device
            ),
        )

    print(
        f"Context length: "
        f"{CONTEXT_LENGTH}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation steps: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Effective batch size: "
        f"{effective_batch_size}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Epochs: "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Evaluation frequency: "
        f"{EVAL_FREQ}"
    )

    print(
        f"Evaluation batches: "
        f"{EVAL_ITER}"
    )

    print(
        f"Maximum optimizer steps: "
        f"{MAX_STEPS}"
    )

    print(
        f"Checkpoint directory: "
        f"{CHECKPOINT_DIR}"
    )

    print(
        "-" * 50
    )


# Model parameter information

def print_parameter_information(
    model,
) -> None:
    """
    Print total, trainable, and frozen parameter counts.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = (
        total_parameters
        - trainable_parameters
    )

    print(
        "\nModel parameters"
    )

    print(
        "-" * 50
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Frozen parameters: "
        f"{frozen_parameters:,}"
    )

    print(
        f"Trainable percentage: "
        f"{100 * trainable_parameters / total_parameters:.2f}%"
    )

    print(
        "-" * 50
    )


# Main training function

def main() -> None:
    """
    Run GPT-2 classification finetuning.
    """

    # Set deterministic random seeds.

    set_random_seed(
        RANDOM_SEED
    )

    # Select the best available hardware device.

    device = get_device()

    print_configuration(
        device
    )

    # Load the original pretrained GPT-2 124M weights.

    print(
        "\nLoading pretrained GPT-2..."
    )

    model = (
        create_pretrained_gpt2_small()
    )

    # Replace the language-model output head with the
    # two-class ham/spam classification head.

    print(
        "\nConfiguring GPT-2 for "
        "email classification..."
    )

    model = (
        configure_for_classification(
            model=model,
            random_seed=RANDOM_SEED,
        )
    )

    # Move the model onto the selected device.

    model.to(
        device
    )

    print_parameter_information(
        model
    )

    # Create training, validation, and test DataLoaders.

    print(
        "\nCreating DataLoaders..."
    )

    (
        train_loader,
        eval_loader,
        test_loader,
    ) = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context_length=CONTEXT_LENGTH,
        random_seed=RANDOM_SEED,
    )

    print(
        f"Training examples: "
        f"{len(train_loader.dataset):,}"
    )

    print(
        f"Validation examples: "
        f"{len(eval_loader.dataset):,}"
    )

    print(
        f"Test examples: "
        f"{len(test_loader.dataset):,}"
    )

    # Create AdamW using only trainable model parameters.

    optimizer = create_optimizer(
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "\nOptimizer:"
    )

    print(
        f"AdamW | "
        f"lr={LEARNING_RATE} | "
        f"weight_decay={WEIGHT_DECAY}"
    )

    # Start classification finetuning.

    print(
        "\nStarting training..."
    )

    history = train_classifier(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        eval_freq=EVAL_FREQ,
        eval_iter=EVAL_ITER,
        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),
        max_steps=MAX_STEPS,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # Print a compact summary when training finishes.

    print(
        "\nTraining finished"
    )

    print(
        "-" * 50
    )

    print(
        f"Epochs completed: "
        f"{len(history['epochs'])}"
    )

    print(
        f"Evaluation points: "
        f"{len(history['steps'])}"
    )

    if history[
        "steps"
    ]:
        print(
            f"Last optimizer step: "
            f"{history['steps'][-1]}"
        )

        print(
            f"Examples seen: "
            f"{history['examples_seen'][-1]:,}"
        )

        print(
            f"Last train loss: "
            f"{history['train_loss'][-1]:.4f}"
        )

        print(
            f"Last validation loss: "
            f"{history['eval_loss'][-1]:.4f}"
        )

    if history[
        "train_accuracy"
    ]:
        print(
            f"Last train accuracy: "
            f"{history['train_accuracy'][-1] * 100:.2f}%"
        )

    if history[
        "eval_accuracy"
    ]:
        print(
            f"Last validation accuracy: "
            f"{history['eval_accuracy'][-1] * 100:.2f}%"
        )

    print(
        "\nCheckpoint directory:"
    )

    print(
        CHECKPOINT_DIR
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        CHECKPOINT_DIR
        / "best_checkpoint.pt"
    )

    print(
        "\nLast checkpoint:"
    )

    print(
        CHECKPOINT_DIR
        / "last_checkpoint.pt"
    )

    print(
        "-" * 50
    )


if __name__ == "__main__":
    main()