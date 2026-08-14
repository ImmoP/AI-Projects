"""
Training utilities for GPT-2 email classification.

This module contains:

- optimizer creation;
- a single optimization step;
- evaluation of training and validation loss;
- checkpoint saving;
- gradient accumulation;
- the complete classification-finetuning loop.
"""

from pathlib import Path

import torch

from spam_detector.model.metrics import (
    calc_accuracy_loader,
    calc_loss_batch,
    calc_loss_loader,
)

# Training defaults

DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_WEIGHT_DECAY = 0.1


# Optimizer

def create_optimizer(
    model,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
):
    """
    Create an AdamW optimizer.

    Only trainable model parameters are passed to the optimizer.
    Frozen GPT-2 parameters are therefore ignored.
    """

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError(
            "The model has no trainable parameters."
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return optimizer


# Single optimization step

def train_step(
    batch: dict,
    model,
    optimizer,
    device: torch.device,
) -> float:
    """
    Perform one complete optimization step.

    This helper does not use gradient accumulation.
    The full train_classifier() function below supports it.
    """

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    loss = calc_loss_batch(
        batch=batch,
        model=model,
        device=device,
    )

    loss.backward()

    optimizer.step()

    return loss.item()


# Evaluate training and validation loss

def evaluate_model(
    model,
    train_loader,
    eval_loader,
    device: torch.device,
    num_batches: int | None = None,
):
    """
    Evaluate training and validation loss.

    Parameters
    ----------
    model:
        GPT-2 classification model.

    train_loader:
        Training DataLoader.

    eval_loader:
        Validation DataLoader.

    device:
        CPU, CUDA, or MPS device.

    num_batches:
        Optional number of batches to evaluate.

        Limiting the number of batches makes intermediate
        evaluation substantially faster.

    Returns
    -------
    tuple
        (train_loss, eval_loss)
    """

    train_loss = calc_loss_loader(
        data_loader=train_loader,
        model=model,
        device=device,
        num_batches=num_batches,
    )

    eval_loss = calc_loss_loader(
        data_loader=eval_loader,
        model=model,
        device=device,
        num_batches=num_batches,
    )

    return (
        train_loss,
        eval_loss,
    )


# Save training checkpoint

def save_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    epoch: int,
    global_step: int,
    examples_seen: int,
    eval_loss: float,
    history: dict,
) -> None:
    """
    Save the current training state.

    The checkpoint contains:

    - model parameters;
    - optimizer state;
    - current epoch;
    - optimizer step;
    - number of training examples seen;
    - validation loss;
    - complete training history.

    Saving the optimizer state allows training to be resumed later
    without resetting AdamW's internal statistics.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "epoch":
            epoch,

        "global_step":
            global_step,

        "examples_seen":
            examples_seen,

        "eval_loss":
            eval_loss,

        "history":
            history,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


# Complete training loop

def train_classifier(
    model,
    train_loader,
    eval_loader,
    optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    gradient_accumulation_steps: int = 1,
    max_steps: int | None = None,
    checkpoint_dir=None,
):
    """
    Finetune GPT-2 for spam classification.

    Parameters
    ----------
    model:
        GPT-2 classification model.

    train_loader:
        Training DataLoader.

    eval_loader:
        Validation DataLoader.

    optimizer:
        PyTorch optimizer.

    device:
        CPU, CUDA, or MPS device.

    num_epochs:
        Number of complete passes through the training dataset.

    eval_freq:
        Evaluate training and validation loss every N optimizer
        steps.

    eval_iter:
        Number of batches used during intermediate evaluation.

    gradient_accumulation_steps:
        Number of micro-batches whose gradients are accumulated
        before optimizer.step().

        Example:

            batch_size = 1
            gradient_accumulation_steps = 8

            effective batch size = 8

    max_steps:
        Optional maximum number of optimizer steps.

        Use a small integer for smoke testing.

        Use:

            max_steps=None

        for complete training.

    checkpoint_dir:
        Optional directory for checkpoints.

        The following files are created:

            best_checkpoint.pt
            last_checkpoint.pt
            final_checkpoint.pt

    Returns
    -------
    dict
        Training history.
    """

    # Validate configuration

    if num_epochs < 1:
        raise ValueError(
            "num_epochs must be at least 1."
        )

    if eval_freq < 1:
        raise ValueError(
            "eval_freq must be at least 1."
        )

    if eval_iter < 1:
        raise ValueError(
            "eval_iter must be at least 1."
        )

    if gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps "
            "must be at least 1."
        )

    if (
        max_steps is not None
        and max_steps < 1
    ):
        raise ValueError(
            "max_steps must be at least 1."
        )

    if checkpoint_dir is not None:
        checkpoint_dir = Path(
            checkpoint_dir
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # Training history

    history = {
        "steps": [],
        "examples_seen": [],
        "train_loss": [],
        "eval_loss": [],
        "epochs": [],
        "train_accuracy": [],
        "eval_accuracy": [],
    }

    global_step = 0
    examples_seen = 0

    best_eval_loss = float(
        "inf"
    )

    stop_training = False

    optimizer.zero_grad(
        set_to_none=True
    )

    # Epoch loop

    for epoch in range(
        num_epochs
    ):

        model.train()

        accumulation_counter = 0

        # Batch loop

        for batch in train_loader:

            # Calculate classification loss

            loss = calc_loss_batch(
                batch=batch,
                model=model,
                device=device,
            )

            # Scale the loss so that accumulated gradients
            # correspond approximately to one larger batch.

            scaled_loss = (
                loss
                / gradient_accumulation_steps
            )

            # Backpropagation

            scaled_loss.backward()

            accumulation_counter += 1

            # Count processed examples

            batch_size = batch[
                "labels"
            ].shape[0]

            examples_seen += (
                batch_size
            )

            # Continue accumulating gradients until the desired
            # number of micro-batches has been reached.

            if (
                accumulation_counter
                < gradient_accumulation_steps
            ):
                continue

            # Update trainable parameters

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            accumulation_counter = 0

            global_step += 1

            # Periodic evaluation

            if (
                global_step
                % eval_freq
                == 0
            ):

                (
                    train_loss,
                    eval_loss,
                ) = evaluate_model(
                    model=model,
                    train_loader=train_loader,
                    eval_loader=eval_loader,
                    device=device,
                    num_batches=eval_iter,
                )

                history[
                    "steps"
                ].append(
                    global_step
                )

                history[
                    "examples_seen"
                ].append(
                    examples_seen
                )

                history[
                    "train_loss"
                ].append(
                    train_loss
                )

                history[
                    "eval_loss"
                ].append(
                    eval_loss
                )

                print(
                    f"Epoch {epoch + 1} | "
                    f"Step {global_step:06d} | "
                    f"Examples {examples_seen:,} | "
                    f"Train loss {train_loss:.4f} | "
                    f"Eval loss {eval_loss:.4f}"
                )

                # Save the most recently evaluated model

                if checkpoint_dir is not None:

                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        checkpoint_path=(
                            checkpoint_dir
                            / "last_checkpoint.pt"
                        ),
                        epoch=epoch + 1,
                        global_step=global_step,
                        examples_seen=examples_seen,
                        eval_loss=eval_loss,
                        history=history,
                    )

                    # Save a separate best model whenever the
                    # validation loss reaches a new minimum.

                    if (
                        eval_loss
                        < best_eval_loss
                    ):

                        best_eval_loss = (
                            eval_loss
                        )

                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            checkpoint_path=(
                                checkpoint_dir
                                / "best_checkpoint.pt"
                            ),
                            epoch=epoch + 1,
                            global_step=global_step,
                            examples_seen=examples_seen,
                            eval_loss=eval_loss,
                            history=history,
                        )

                        print(
                            "New best checkpoint | "
                            f"Eval loss "
                            f"{best_eval_loss:.4f}"
                        )

            # Optional stopping point for smoke tests

            if (
                max_steps is not None
                and global_step
                >= max_steps
            ):

                stop_training = True
                break

        # Handle an incomplete gradient accumulation group
        #
        # Example:
        #
        # gradient_accumulation_steps = 8
        #
        # but only 3 micro-batches remain at the end of an epoch.
        #
        # Those gradients are still used instead of discarded.

        if (
            accumulation_counter > 0
            and not stop_training
        ):

            correction_factor = (
                gradient_accumulation_steps
                / accumulation_counter
            )

            for parameter in model.parameters():

                if parameter.grad is not None:

                    parameter.grad.mul_(
                        correction_factor
                    )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            accumulation_counter = 0

            global_step += 1

        # Calculate accuracy at the end of the epoch

        train_accuracy = (
            calc_accuracy_loader(
                data_loader=train_loader,
                model=model,
                device=device,
                num_batches=eval_iter,
            )
        )

        eval_accuracy = (
            calc_accuracy_loader(
                data_loader=eval_loader,
                model=model,
                device=device,
                num_batches=eval_iter,
            )
        )

        history[
            "epochs"
        ].append(
            epoch + 1
        )

        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )

        history[
            "eval_accuracy"
        ].append(
            eval_accuracy
        )

        print(
            f"\nEpoch {epoch + 1} accuracy | "
            f"Train "
            f"{train_accuracy * 100:.2f}% | "
            f"Eval "
            f"{eval_accuracy * 100:.2f}%\n"
        )

        # Evaluate the exact final state of this epoch

        (
            final_train_loss,
            final_eval_loss,
        ) = evaluate_model(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            num_batches=eval_iter,
        )

        # Add the final state to the history if it has not already
        # been recorded by the regular evaluation schedule.

        if (
            not history["steps"]
            or history["steps"][-1]
            != global_step
        ):

            history[
                "steps"
            ].append(
                global_step
            )

            history[
                "examples_seen"
            ].append(
                examples_seen
            )

            history[
                "train_loss"
            ].append(
                final_train_loss
            )

            history[
                "eval_loss"
            ].append(
                final_eval_loss
            )

        print(
            f"Final epoch state | "
            f"Step {global_step:06d} | "
            f"Examples {examples_seen:,} | "
            f"Train loss {final_train_loss:.4f} | "
            f"Eval loss {final_eval_loss:.4f}"
        )

        # Save the exact final model state

        if checkpoint_dir is not None:

            # final_checkpoint.pt always represents the exact
            # model state at the end of the most recent epoch.

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=(
                    checkpoint_dir
                    / "final_checkpoint.pt"
                ),
                epoch=epoch + 1,
                global_step=global_step,
                examples_seen=examples_seen,
                eval_loss=final_eval_loss,
                history=history,
            )

            # last_checkpoint.pt is also updated so that "last"
            # really means the most recent saved model state.

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=(
                    checkpoint_dir
                    / "last_checkpoint.pt"
                ),
                epoch=epoch + 1,
                global_step=global_step,
                examples_seen=examples_seen,
                eval_loss=final_eval_loss,
                history=history,
            )

            # The final state may also be the best validation
            # model seen during training.

            if (
                final_eval_loss
                < best_eval_loss
            ):

                best_eval_loss = (
                    final_eval_loss
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    checkpoint_path=(
                        checkpoint_dir
                        / "best_checkpoint.pt"
                    ),
                    epoch=epoch + 1,
                    global_step=global_step,
                    examples_seen=examples_seen,
                    eval_loss=final_eval_loss,
                    history=history,
                )

                print(
                    "New best checkpoint at end of epoch | "
                    f"Eval loss "
                    f"{best_eval_loss:.4f}"
                )

            print(
                f"Final checkpoint saved | "
                f"Step {global_step} | "
                f"Examples {examples_seen:,}"
            )

        if stop_training:
            break

    return history