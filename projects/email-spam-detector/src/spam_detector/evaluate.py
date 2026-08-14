"""
Evaluate the finetuned GPT-2 email spam classifier.

This script performs validation-based model selection and evaluates
the selected model on several test datasets.

Workflow:

1. Load best_checkpoint.pt.
2. Evaluate it on the complete validation set.
3. Load final_checkpoint.pt.
4. Evaluate it on the complete validation set.
5. Select the checkpoint with the lower full validation loss.
6. Evaluate the selected checkpoint on:
   - the complete leakage-cleaned combined test set;
   - the public/Hugging Face test set;
   - the private email test set.
7. Save all results as JSON.

Important:

The test sets are never used for model selection.

The combined clean test set is the primary test set.

The separate HF/public and private test sets are additional
domain-specific evaluations.
"""

import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from spam_detector.model.classifier import (
    classification_forward,
    configure_for_classification,
)
from spam_detector.model.collate import (
    dynamic_email_collate,
)
from spam_detector.model.dataset import (
    SpamEmailDataset,
)
from spam_detector.model.pretrained import (
    create_pretrained_gpt2_small,
)
from spam_detector.model.tokenization import (
    get_gpt2_tokenizer,
)
from spam_detector.paths import (
    DATA_DIR,
    PROJECT_ROOT,
)

# Evaluation configuration

RANDOM_SEED = 123

CONTEXT_LENGTH = 1024

# Batch size 1 is safe for the RTX 3050.
#
# Evaluation uses no gradients, so a larger batch size could be
# tested later if faster inference is desired.
EVAL_BATCH_SIZE = 1

NUM_WORKERS = 0

HAM_LABEL = 0
SPAM_LABEL = 1


# Checkpoint paths

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "best_checkpoint.pt"
)

FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "final_checkpoint.pt"
)


# Dataset paths

VALIDATION_DATA = (
    DATA_DIR
    / "combined_eval_clean.parquet"
)

COMBINED_TEST_DATA = (
    DATA_DIR
    / "combined_test_clean.parquet"
)

HF_TEST_DATA = (
    DATA_DIR
    / "hf_test_prepared.parquet"
)

PRIVATE_TEST_DATA = (
    DATA_DIR
    / "private_test.parquet"
)


# Result paths

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "evaluation_results.json"
)


# Device selection

def get_device() -> torch.device:
    """
    Select the best available PyTorch device.

    Priority:

    1. NVIDIA CUDA GPU
    2. Apple MPS
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
    Set PyTorch random seeds.
    """

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# Dataset availability

def validate_required_files() -> None:
    """
    Verify that all checkpoints and datasets required for the
    evaluation exist before starting the expensive model runs.
    """

    required_files = {
        "Best checkpoint":
            BEST_CHECKPOINT,

        "Final checkpoint":
            FINAL_CHECKPOINT,

        "Validation dataset":
            VALIDATION_DATA,

        "Combined clean test dataset":
            COMBINED_TEST_DATA,

        "HF/public test dataset":
            HF_TEST_DATA,

        "Private test dataset":
            PRIVATE_TEST_DATA,
    }

    missing_files = []

    for description, path in required_files.items():

        if not path.exists():

            missing_files.append(
                (
                    description,
                    path,
                )
            )

    if missing_files:

        message_lines = [
            "The following required files are missing:"
        ]

        for description, path in missing_files:

            message_lines.append(
                f"- {description}: {path}"
            )

        raise FileNotFoundError(
            "\n".join(
                message_lines
            )
        )


# Evaluation DataLoader

def create_evaluation_loader(
    parquet_path: Path,
    tokenizer,
) -> DataLoader:
    """
    Create a deterministic DataLoader for evaluation.

    Evaluation DataLoaders never shuffle the dataset.
    """

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {parquet_path}"
        )

    dataset = SpamEmailDataset(
        parquet_path=parquet_path,
        tokenizer=tokenizer,
        context_length=CONTEXT_LENGTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        collate_fn=dynamic_email_collate,
    )

    return loader


# Classification model

def create_model(
    device: torch.device,
):
    """
    Create the GPT-2 classification architecture.

    The original pretrained GPT-2 parameters are loaded first.

    The language-model output head is then replaced with the
    two-class ham/spam classification head.

    Finetuned checkpoint parameters are loaded afterwards.
    """

    print(
        "\nLoading pretrained GPT-2 architecture..."
    )

    model = (
        create_pretrained_gpt2_small()
    )

    model = (
        configure_for_classification(
            model=model,
            random_seed=RANDOM_SEED,
        )
    )

    model.to(
        device
    )

    model.eval()

    return model


# Checkpoint loading

def load_checkpoint(
    model,
    checkpoint_path: Path,
) -> dict:
    """
    Load model parameters from a training checkpoint.

    The checkpoint is loaded into CPU memory first to avoid
    unnecessarily loading AdamW optimizer state into GPU memory.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print(
        f"\nLoading checkpoint: "
        f"{checkpoint_path.name}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    metadata = {
        "epoch":
            int(
                checkpoint.get(
                    "epoch",
                    -1,
                )
            ),

        "global_step":
            int(
                checkpoint.get(
                    "global_step",
                    -1,
                )
            ),

        "examples_seen":
            int(
                checkpoint.get(
                    "examples_seen",
                    -1,
                )
            ),

        "stored_eval_loss":
            float(
                checkpoint.get(
                    "eval_loss",
                    float("nan"),
                )
            ),
    }

    del checkpoint

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metadata


# Complete DataLoader evaluation

def evaluate_loader(
    model,
    data_loader,
    device: torch.device,
    description: str,
) -> dict:
    """
    Evaluate a model over an entire DataLoader.

    Metrics:

    - mean cross-entropy loss;
    - accuracy;
    - spam precision;
    - spam recall;
    - spam F1;
    - ROC-AUC;
    - PR-AUC;
    - confusion matrix;
    - class counts.
    """

    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_labels = []
    all_predictions = []
    all_spam_probabilities = []

    with torch.no_grad():

        for batch in tqdm(
            data_loader,
            desc=description,
        ):

            input_ids = batch[
                "input_ids"
            ].to(
                device
            )

            eos_indices = batch[
                "eos_indices"
            ].to(
                device
            )

            labels = batch[
                "labels"
            ].to(
                device
            )

            logits = classification_forward(
                model=model,
                input_ids=input_ids,
                eos_indices=eos_indices,
            )

            # Use summed loss for each batch.
            #
            # Dividing by total_examples afterwards gives the
            # exact mean cross-entropy over the entire dataset.

            batch_loss = F.cross_entropy(
                logits,
                labels,
                reduction="sum",
            )

            total_loss += (
                batch_loss.item()
            )

            total_examples += (
                labels.numel()
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            spam_probabilities = (
                probabilities[
                    :,
                    SPAM_LABEL,
                ]
            )

            predictions = torch.argmax(
                logits,
                dim=-1,
            )

            all_labels.extend(
                labels
                .cpu()
                .tolist()
            )

            all_predictions.extend(
                predictions
                .cpu()
                .tolist()
            )

            all_spam_probabilities.extend(
                spam_probabilities
                .cpu()
                .tolist()
            )

    if total_examples == 0:
        raise ValueError(
            "Evaluation DataLoader is empty."
        )

    mean_loss = (
        total_loss
        / total_examples
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        pos_label=SPAM_LABEL,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        pos_label=SPAM_LABEL,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        pos_label=SPAM_LABEL,
        zero_division=0,
    )

    confusion = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[
            HAM_LABEL,
            SPAM_LABEL,
        ],
    )

    (
        true_negatives,
        false_positives,
        false_negatives,
        true_posititives,
    ) = confusion.ravel()

    ham_count = sum(
        label == HAM_LABEL
        for label in all_labels
    )

    spam_count = sum(
        label == SPAM_LABEL
        for label in all_labels
    )

    # ROC-AUC and PR-AUC are defined only when both classes
    # occur in the evaluated dataset.

    unique_labels = set(
        all_labels
    )

    if len(
        unique_labels
    ) == 2:

        roc_auc = roc_auc_score(
            all_labels,
            all_spam_probabilities,
        )

        pr_auc = average_precision_score(
            all_labels,
            all_spam_probabilities,
        )

    else:

        roc_auc = None
        pr_auc = None

    results = {
        "examples":
            int(
                total_examples
            ),

        "ham_examples":
            int(
                ham_count
            ),

        "spam_examples":
            int(
                spam_count
            ),

        "loss":
            float(
                mean_loss
            ),

        "accuracy":
            float(
                accuracy
            ),

        "spam_precision":
            float(
                precision
            ),

        "spam_recall":
            float(
                recall
            ),

        "spam_f1":
            float(
                f1
            ),

        "roc_auc":
            (
                float(
                    roc_auc
                )
                if roc_auc is not None
                else None
            ),

        "pr_auc":
            (
                float(
                    pr_auc
                )
                if pr_auc is not None
                else None
            ),

        "confusion_matrix": {
            "true_negatives":
                int(
                    true_negatives
                ),

            "false_positives":
                int(
                    false_positives
                ),

            "false_negatives":
                int(
                    false_negatives
                ),

            "true_positives":
                int(
                    true_posititives
                ),
        },
    }

    return results


# Console metric output

def print_metrics(
    title: str,
    metrics: dict,
) -> None:
    """
    Print evaluation metrics in a readable format.
    """

    print(
        f"\n{title}"
    )

    print(
        "-" * 60
    )

    print(
        f"Examples: "
        f"{metrics['examples']:,}"
    )

    print(
        f"Ham: "
        f"{metrics['ham_examples']:,}"
    )

    print(
        f"Spam: "
        f"{metrics['spam_examples']:,}"
    )

    print(
        f"Loss: "
        f"{metrics['loss']:.6f}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Spam precision: "
        f"{metrics['spam_precision'] * 100:.2f}%"
    )

    print(
        f"Spam recall: "
        f"{metrics['spam_recall'] * 100:.2f}%"
    )

    print(
        f"Spam F1: "
        f"{metrics['spam_f1'] * 100:.2f}%"
    )

    if metrics[
        "roc_auc"
    ] is not None:

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.6f}"
        )

    if metrics[
        "pr_auc"
    ] is not None:

        print(
            f"PR-AUC: "
            f"{metrics['pr_auc']:.6f}"
        )

    confusion = metrics[
        "confusion_matrix"
    ]

    print(
        "\nConfusion matrix"
    )

    print(
        f"True negatives  (ham -> ham):  "
        f"{confusion['true_negatives']:,}"
    )

    print(
        f"False positives (ham -> spam): "
        f"{confusion['false_positives']:,}"
    )

    print(
        f"False negatives (spam -> ham): "
        f"{confusion['false_negatives']:,}"
    )

    print(
        f"True positives  (spam -> spam): "
        f"{confusion['true_positives']:,}"
    )

    print(
        "-" * 60
    )


# Compact comparison table

def print_test_comparison(
    combined_metrics: dict,
    hf_metrics: dict,
    private_metrics: dict,
) -> None:
    """
    Print a compact comparison of the three test domains.
    """

    print(
        "\nTest-set comparison"
    )

    print(
        "-" * 100
    )

    header = (
        f"{'Dataset':<20}"
        f"{'N':>9}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'PR-AUC':>12}"
    )

    print(
        header
    )

    print(
        "-" * 100
    )

    rows = [
        (
            "Combined clean",
            combined_metrics,
        ),
        (
            "HF / Public",
            hf_metrics,
        ),
        (
            "Private",
            private_metrics,
        ),
    ]

    for name, metrics in rows:

        pr_auc = metrics[
            "pr_auc"
        ]

        pr_auc_text = (
            f"{pr_auc:.4f}"
            if pr_auc is not None
            else "N/A"
        )

        print(
            f"{name:<20}"
            f"{metrics['examples']:>9,}"
            f"{metrics['accuracy'] * 100:>11.2f}%"
            f"{metrics['spam_precision'] * 100:>11.2f}%"
            f"{metrics['spam_recall'] * 100:>11.2f}%"
            f"{metrics['spam_f1'] * 100:>11.2f}%"
            f"{pr_auc_text:>12}"
        )

    print(
        "-" * 100
    )


# Save JSON results

def save_results(
    results: dict,
) -> None:
    """
    Save all evaluation results as JSON.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        "\nEvaluation results saved to:"
    )

    print(
        RESULTS_FILE
    )


# Main evaluation

def main() -> None:
    """
    Run validation-based model selection and complete
    multi-domain test evaluation.
    """

    set_random_seed(
        RANDOM_SEED
    )

    validate_required_files()

    device = get_device()

    print(
        "\nEvaluation configuration"
    )

    print(
        "-" * 60
    )

    print(
        f"Device: "
        f"{device}"
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
        f"Evaluation batch size: "
        f"{EVAL_BATCH_SIZE}"
    )

    print(
        "-" * 60
    )

    # Create tokenizer.

    print(
        "\nCreating evaluation datasets..."
    )

    tokenizer = (
        get_gpt2_tokenizer()
    )

    # Create deterministic DataLoaders.

    validation_loader = (
        create_evaluation_loader(
            parquet_path=VALIDATION_DATA,
            tokenizer=tokenizer,
        )
    )

    combined_test_loader = (
        create_evaluation_loader(
            parquet_path=COMBINED_TEST_DATA,
            tokenizer=tokenizer,
        )
    )

    hf_test_loader = (
        create_evaluation_loader(
            parquet_path=HF_TEST_DATA,
            tokenizer=tokenizer,
        )
    )

    private_test_loader = (
        create_evaluation_loader(
            parquet_path=PRIVATE_TEST_DATA,
            tokenizer=tokenizer,
        )
    )

    print(
        f"Validation examples: "
        f"{len(validation_loader.dataset):,}"
    )

    print(
        f"Combined clean test examples: "
        f"{len(combined_test_loader.dataset):,}"
    )

    print(
        f"HF/public test examples: "
        f"{len(hf_test_loader.dataset):,}"
    )

    print(
        f"Private test examples: "
        f"{len(private_test_loader.dataset):,}"
    )

    # Create the model once.
    #
    # Different checkpoint state dictionaries are loaded into the
    # same architecture during evaluation.

    model = create_model(
        device=device
    )

    # Evaluate best_checkpoint.pt over the complete validation set.

    best_metadata = load_checkpoint(
        model=model,
        checkpoint_path=BEST_CHECKPOINT,
    )

    print(
        "\nBest checkpoint metadata:"
    )

    print(
        best_metadata
    )

    best_validation_metrics = (
        evaluate_loader(
            model=model,
            data_loader=validation_loader,
            device=device,
            description=(
                "Best checkpoint validation"
            ),
        )
    )

    print_metrics(
        title=(
            "BEST CHECKPOINT - "
            "FULL VALIDATION SET"
        ),
        metrics=best_validation_metrics,
    )

    # Evaluate final_checkpoint.pt over the same complete
    # validation set.

    final_metadata = load_checkpoint(
        model=model,
        checkpoint_path=FINAL_CHECKPOINT,
    )

    print(
        "\nFinal checkpoint metadata:"
    )

    print(
        final_metadata
    )

    final_validation_metrics = (
        evaluate_loader(
            model=model,
            data_loader=validation_loader,
            device=device,
            description=(
                "Final checkpoint validation"
            ),
        )
    )

    print_metrics(
        title=(
            "FINAL CHECKPOINT - "
            "FULL VALIDATION SET"
        ),
        metrics=final_validation_metrics,
    )

    # Select the checkpoint using validation loss only.
    #
    # No test-set information is used for this decision.

    if (
        best_validation_metrics[
            "loss"
        ]
        <=
        final_validation_metrics[
            "loss"
        ]
    ):

        selected_checkpoint = (
            BEST_CHECKPOINT
        )

        selected_name = (
            "best_checkpoint.pt"
        )

        selected_metadata = (
            best_metadata
        )

        selected_validation_metrics = (
            best_validation_metrics
        )

    else:

        selected_checkpoint = (
            FINAL_CHECKPOINT
        )

        selected_name = (
            "final_checkpoint.pt"
        )

        selected_metadata = (
            final_metadata
        )

        selected_validation_metrics = (
            final_validation_metrics
        )

    print(
        "\nSelected checkpoint"
    )

    print(
        "-" * 60
    )

    print(
        selected_name
    )

    print(
        f"Full validation loss: "
        f"{selected_validation_metrics['loss']:.6f}"
    )

    print(
        f"Optimizer step: "
        f"{selected_metadata['global_step']:,}"
    )

    print(
        f"Training examples seen: "
        f"{selected_metadata['examples_seen']:,}"
    )

    print(
        "-" * 60
    )

    # Load the selected checkpoint before test evaluation.

    selected_metadata = load_checkpoint(
        model=model,
        checkpoint_path=selected_checkpoint,
    )

    # Primary test:
    # leakage-cleaned combined test set.

    combined_test_metrics = (
        evaluate_loader(
            model=model,
            data_loader=combined_test_loader,
            device=device,
            description=(
                "Combined clean test"
            ),
        )
    )

    print_metrics(
        title=(
            "SELECTED MODEL - "
            "COMBINED CLEAN TEST"
        ),
        metrics=combined_test_metrics,
    )

    # Public/Hugging Face domain evaluation.

    hf_test_metrics = (
        evaluate_loader(
            model=model,
            data_loader=hf_test_loader,
            device=device,
            description=(
                "HF/public test"
            ),
        )
    )

    print_metrics(
        title=(
            "SELECTED MODEL - "
            "HF / PUBLIC TEST"
        ),
        metrics=hf_test_metrics,
    )

    # Private email domain evaluation.

    private_test_metrics = (
        evaluate_loader(
            model=model,
            data_loader=private_test_loader,
            device=device,
            description=(
                "Private test"
            ),
        )
    )

    print_metrics(
        title=(
            "SELECTED MODEL - "
            "PRIVATE TEST"
        ),
        metrics=private_test_metrics,
    )

    # Print all test domains next to each other.

    print_test_comparison(
        combined_metrics=combined_test_metrics,
        hf_metrics=hf_test_metrics,
        private_metrics=private_test_metrics,
    )

    # Store complete evaluation output.

    results = {
        "configuration": {
            "context_length":
                CONTEXT_LENGTH,

            "evaluation_batch_size":
                EVAL_BATCH_SIZE,

            "random_seed":
                RANDOM_SEED,

            "device":
                str(
                    device
                ),

            "labels": {
                "ham":
                    HAM_LABEL,

                "spam":
                    SPAM_LABEL,
            },
        },

        "model_selection": {
            "criterion":
                "lowest_full_validation_loss",

            "test_data_used_for_selection":
                False,
        },

        "best_checkpoint": {
            "path":
                str(
                    BEST_CHECKPOINT
                ),

            "metadata":
                best_metadata,

            "validation":
                best_validation_metrics,
        },

        "final_checkpoint": {
            "path":
                str(
                    FINAL_CHECKPOINT
                ),

            "metadata":
                final_metadata,

            "validation":
                final_validation_metrics,
        },

        "selected_checkpoint": {
            "name":
                selected_name,

            "path":
                str(
                    selected_checkpoint
                ),

            "metadata":
                selected_metadata,

            "full_validation_metrics":
                selected_validation_metrics,
        },

        "test_evaluations": {
            "combined_clean": {
                "path":
                    str(
                        COMBINED_TEST_DATA
                    ),

                "role":
                    "primary_test_set",

                "metrics":
                    combined_test_metrics,
            },

            "hf_public": {
                "path":
                    str(
                        HF_TEST_DATA
                    ),

                "role":
                    "public_domain_evaluation",

                "metrics":
                    hf_test_metrics,
            },

            "private": {
                "path":
                    str(
                        PRIVATE_TEST_DATA
                    ),

                "role":
                    "private_domain_evaluation",

                "metrics":
                    private_test_metrics,
            },
        },

        "notes": {
            "combined_clean_test":
                (
                    "Primary leakage-cleaned combined test set."
                ),

            "domain_specific_tests":
                (
                    "HF/public and private test files are reported "
                    "separately as additional domain-specific "
                    "evaluations."
                ),

            "private_test_uncertainty":
                (
                    "The private test set contains relatively few "
                    "spam examples, so private spam recall and F1 "
                    "have substantially greater sampling uncertainty "
                    "than the combined/public metrics."
                ),
        },
    }

    save_results(
        results
    )

    print(
        "\nEvaluation finished successfully."
    )


if __name__ == "__main__":
    main()