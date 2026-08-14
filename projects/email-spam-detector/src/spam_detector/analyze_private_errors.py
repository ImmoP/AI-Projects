"""
Analyze classification errors on the private validation dataset.

This script deliberately uses private_eval.parquet instead of
private_test.parquet.

The private test set should remain untouched for future independent
evaluation. The private validation set can be used for:

- error analysis;
- manual label review;
- threshold development;
- understanding domain shift;
- identifying possible annotation errors.

Outputs are written to:

    data/private_error_analysis/

The output files may contain private email information and should
therefore NOT be committed to Git.
"""


import pandas as pd
import torch
from tqdm import tqdm

from spam_detector.evaluate import (
    create_evaluation_loader,
    create_model,
    get_device,
    load_checkpoint,
    set_random_seed,
)
from spam_detector.model.classifier import (
    classification_forward,
)
from spam_detector.model.tokenization import (
    get_gpt2_tokenizer,
)
from spam_detector.paths import (
    DATA_DIR,
    PROJECT_ROOT,
)

# Configuration

RANDOM_SEED = 123

HAM_LABEL = 0
SPAM_LABEL = 1

PRIVATE_EVAL_DATA = (
    DATA_DIR
    / "private_eval.parquet"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_checkpoint.pt"
)

OUTPUT_DIR = (
    DATA_DIR
    / "private_error_analysis"
)

ALL_PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "all_predictions.csv"
)

FALSE_POSITIVES_FILE = (
    OUTPUT_DIR
    / "false_positives.csv"
)

FALSE_NEGATIVES_FILE = (
    OUTPUT_DIR
    / "false_negatives.csv"
)

HIGH_CONFIDENCE_ERRORS_FILE = (
    OUTPUT_DIR
    / "high_confidence_errors.csv"
)

MANUAL_REVIEW_SAMPLE_FILE = (
    OUTPUT_DIR
    / "manual_review_sample.csv"
)


# Number of examples selected per error type for manual review.

MANUAL_REVIEW_PER_ERROR_TYPE = 10


# Errors above this confidence are additionally written to the
# high-confidence error file.

HIGH_CONFIDENCE_THRESHOLD = 0.90


# Maximum number of body characters included in the CSV preview.

BODY_PREVIEW_LENGTH = 2000


# Helper functions

def label_name(
    label: int,
) -> str:
    """
    Convert numeric labels into readable names.
    """

    if label == HAM_LABEL:
        return "ham"

    if label == SPAM_LABEL:
        return "spam"

    return f"unknown_{label}"


def clean_text_value(
    value,
) -> str:
    """
    Convert a potentially missing value into a clean string.
    """

    if value is None:
        return ""

    if pd.isna(
        value
    ):
        return ""

    return str(
        value
    ).strip()


def create_body_preview(
    text,
    max_length: int = BODY_PREVIEW_LENGTH,
) -> str:
    """
    Create a readable one-line preview of an email body.

    Newlines and repeated whitespace are collapsed so that the
    resulting CSV is easier to inspect in Excel or another table
    viewer.
    """

    text = clean_text_value(
        text
    )

    text = " ".join(
        text.split()
    )

    if len(
        text
    ) <= max_length:
        return text

    return (
        text[:max_length]
        + " ..."
    )


def determine_error_type(
    true_label: int,
    predicted_label: int,
) -> str:
    """
    Return the confusion-matrix category for one prediction.
    """

    if (
        true_label == HAM_LABEL
        and predicted_label == HAM_LABEL
    ):
        return "true_negative"

    if (
        true_label == HAM_LABEL
        and predicted_label == SPAM_LABEL
    ):
        return "false_positive"

    if (
        true_label == SPAM_LABEL
        and predicted_label == HAM_LABEL
    ):
        return "false_negative"

    if (
        true_label == SPAM_LABEL
        and predicted_label == SPAM_LABEL
    ):
        return "true_positive"

    return "unknown"


def determine_review_priority(
    is_error: bool,
    model_confidence: float,
) -> str:
    """
    Assign a simple priority for manual review.

    A high-confidence disagreement between the model and the
    original label is particularly interesting because it may
    indicate:

    - a difficult example;
    - domain shift;
    - a systematic model error;
    - or a potentially incorrect original label.

    It does NOT automatically mean that the label is wrong.
    """

    if not is_error:
        return ""

    if model_confidence >= 0.95:
        return "high"

    if model_confidence >= 0.80:
        return "medium"

    return "low"


# Prediction

def collect_predictions(
    model,
    data_loader,
    device: torch.device,
) -> pd.DataFrame:
    """
    Run the selected model over the entire private validation set.

    DataLoader order is deterministic because evaluation loaders
    use shuffle=False.

    The returned DataFrame contains one row per original email.
    """

    model.eval()

    rows = []

    row_position = 0

    with torch.no_grad():

        for batch in tqdm(
            data_loader,
            desc="Private validation predictions",
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

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            predictions = torch.argmax(
                logits,
                dim=-1,
            )

            spam_probabilities = (
                probabilities[
                    :,
                    SPAM_LABEL,
                ]
            )

            batch_size = labels.shape[
                0
            ]

            for batch_index in range(
                batch_size
            ):

                true_label = int(
                    labels[
                        batch_index
                    ].item()
                )

                predicted_label = int(
                    predictions[
                        batch_index
                    ].item()
                )

                spam_probability = float(
                    spam_probabilities[
                        batch_index
                    ].item()
                )

                ham_probability = (
                    1.0
                    - spam_probability
                )

                model_confidence = max(
                    ham_probability,
                    spam_probability,
                )

                is_error = (
                    true_label
                    != predicted_label
                )

                error_type = determine_error_type(
                    true_label=true_label,
                    predicted_label=predicted_label,
                )

                review_priority = determine_review_priority(
                    is_error=is_error,
                    model_confidence=model_confidence,
                )

                rows.append(
                    {
                        "row_id":
                            row_position,

                        "true_label":
                            true_label,

                        "true_label_name":
                            label_name(
                                true_label
                            ),

                        "predicted_label":
                            predicted_label,

                        "predicted_label_name":
                            label_name(
                                predicted_label
                            ),

                        "spam_probability":
                            spam_probability,

                        "ham_probability":
                            ham_probability,

                        "model_confidence":
                            model_confidence,

                        "is_error":
                            is_error,

                        "error_type":
                            error_type,

                        "review_priority":
                            review_priority,
                    }
                )

                row_position += 1

    return pd.DataFrame(
        rows
    )


# Merge predictions with original email metadata

def build_analysis_dataframe(
    original_data: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine model predictions with the original email information.
    """

    original_data = (
        original_data
        .reset_index(
            drop=True
        )
        .copy()
    )

    predictions = (
        predictions
        .reset_index(
            drop=True
        )
        .copy()
    )

    if len(
        original_data
    ) != len(
        predictions
    ):
        raise ValueError(
            "Prediction count does not match the number "
            "of rows in private_eval.parquet."
        )

    # Preserve the original row position so that examples can be
    # located again in the source Parquet file.

    original_data[
        "row_id"
    ] = range(
        len(
            original_data
        )
    )

    # Keep useful metadata when available.

    metadata_columns = [
        "row_id",
        "sender",
        "subject",
        "text",
        "date",
        "label",
        "source",
        "source_split",
    ]

    available_columns = [
        column
        for column in metadata_columns
        if column in original_data.columns
    ]

    metadata = original_data[
        available_columns
    ].copy()

    analysis = predictions.merge(
        metadata,
        on="row_id",
        how="left",
        validate="one_to_one",
    )

    # Add a shortened version of the body for convenient manual
    # inspection.

    if "text" in analysis.columns:

        analysis[
            "body_preview"
        ] = analysis[
            "text"
        ].apply(
            create_body_preview
        )

    else:

        analysis[
            "body_preview"
        ] = ""

    # Normalize sender and subject fields.

    if "sender" in analysis.columns:

        analysis[
            "sender"
        ] = analysis[
            "sender"
        ].apply(
            clean_text_value
        )

    else:

        analysis[
            "sender"
        ] = ""

    if "subject" in analysis.columns:

        analysis[
            "subject"
        ] = analysis[
            "subject"
        ].apply(
            clean_text_value
        )

    else:

        analysis[
            "subject"
        ] = ""

    return analysis


# Save analysis files

def save_analysis_files(
    analysis: pd.DataFrame,
) -> None:
    """
    Save complete predictions and separate error files.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save all predictions.

    analysis.to_csv(
        ALL_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # False positives:
    # legitimate ham classified as spam.

    false_positives = analysis[
        analysis[
            "error_type"
        ]
        == "false_positive"
    ].copy()

    false_positives = (
        false_positives
        .sort_values(
            by="spam_probability",
            ascending=False,
        )
    )

    false_positives.to_csv(
        FALSE_POSITIVES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # False negatives:
    # spam classified as legitimate ham.

    false_negatives = analysis[
        analysis[
            "error_type"
        ]
        == "false_negative"
    ].copy()

    false_negatives = (
        false_negatives
        .sort_values(
            by="spam_probability",
            ascending=True,
        )
    )

    false_negatives.to_csv(
        FALSE_NEGATIVES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # High-confidence errors are especially useful for identifying
    # possible labeling problems.

    high_confidence_errors = analysis[
        (
            analysis[
                "is_error"
            ]
        )
        &
        (
            analysis[
                "model_confidence"
            ]
            >= HIGH_CONFIDENCE_THRESHOLD
        )
    ].copy()

    high_confidence_errors = (
        high_confidence_errors
        .sort_values(
            by="model_confidence",
            ascending=False,
        )
    )

    high_confidence_errors.to_csv(
        HIGH_CONFIDENCE_ERRORS_FILE,
        index=False,
        encoding="utf-8-sig",
    )


# Manual review sample

def create_manual_review_sample(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a reproducible random sample of false positives and
    false negatives for manual inspection.

    The sample includes empty columns that can be filled in by hand.
    """

    false_positives = analysis[
        analysis[
            "error_type"
        ]
        == "false_positive"
    ].copy()

    false_negatives = analysis[
        analysis[
            "error_type"
        ]
        == "false_negative"
    ].copy()

    fp_sample_size = min(
        MANUAL_REVIEW_PER_ERROR_TYPE,
        len(
            false_positives
        ),
    )

    fn_sample_size = min(
        MANUAL_REVIEW_PER_ERROR_TYPE,
        len(
            false_negatives
        ),
    )

    if fp_sample_size > 0:

        fp_sample = (
            false_positives
            .sample(
                n=fp_sample_size,
                random_state=RANDOM_SEED,
            )
        )

    else:

        fp_sample = (
            false_positives
            .copy()
        )

    if fn_sample_size > 0:

        fn_sample = (
            false_negatives
            .sample(
                n=fn_sample_size,
                random_state=RANDOM_SEED,
            )
        )

    else:

        fn_sample = (
            false_negatives
            .copy()
        )

    sample = pd.concat(
        [
            fp_sample,
            fn_sample,
        ],
        ignore_index=True,
    )

    # Randomize the combined order so that the reviewer is not
    # presented with all false positives followed by all false
    # negatives.

    if len(
        sample
    ) > 1:

        sample = (
            sample
            .sample(
                frac=1,
                random_state=RANDOM_SEED,
            )
            .reset_index(
                drop=True
            )
        )

    # These fields are deliberately left empty for manual review.

    sample[
        "manual_label"
    ] = ""

    sample[
        "original_label_correct"
    ] = ""

    sample[
        "review_notes"
    ] = ""

    # Put the most useful columns first.

    preferred_columns = [
        "row_id",
        "error_type",
        "review_priority",
        "true_label_name",
        "predicted_label_name",
        "spam_probability",
        "model_confidence",
        "sender",
        "subject",
        "body_preview",
        "date",
        "source",
        "manual_label",
        "original_label_correct",
        "review_notes",
    ]

    existing_preferred_columns = [
        column
        for column in preferred_columns
        if column in sample.columns
    ]

    remaining_columns = [
        column
        for column in sample.columns
        if column not in existing_preferred_columns
        and column != "text"
    ]

    sample = sample[
        existing_preferred_columns
        + remaining_columns
    ]

    sample.to_csv(
        MANUAL_REVIEW_SAMPLE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return sample


# Console summary

def print_summary(
    analysis: pd.DataFrame,
    manual_sample: pd.DataFrame,
) -> None:
    """
    Print a compact error-analysis summary.
    """

    total_examples = len(
        analysis
    )

    true_positives = int(
        (
            analysis[
                "error_type"
            ]
            == "true_positive"
        ).sum()
    )

    true_negatives = int(
        (
            analysis[
                "error_type"
            ]
            == "true_negative"
        ).sum()
    )

    false_positives = int(
        (
            analysis[
                "error_type"
            ]
            == "false_positive"
        ).sum()
    )

    false_negatives = int(
        (
            analysis[
                "error_type"
            ]
            == "false_negative"
        ).sum()
    )

    total_errors = (
        false_positives
        + false_negatives
    )

    high_confidence_errors = int(
        (
            (
                analysis[
                    "is_error"
                ]
            )
            &
            (
                analysis[
                    "model_confidence"
                ]
                >= HIGH_CONFIDENCE_THRESHOLD
            )
        ).sum()
    )

    print(
        "\nPrivate validation error analysis"
    )

    print(
        "-" * 60
    )

    print(
        f"Total examples: "
        f"{total_examples:,}"
    )

    print(
        f"True negatives: "
        f"{true_negatives:,}"
    )

    print(
        f"True positives: "
        f"{true_positives:,}"
    )

    print(
        f"False positives: "
        f"{false_positives:,}"
    )

    print(
        f"False negatives: "
        f"{false_negatives:,}"
    )

    print(
        f"Total errors: "
        f"{total_errors:,}"
    )

    if total_examples > 0:

        error_rate = (
            total_errors
            / total_examples
        )

        print(
            f"Error rate: "
            f"{error_rate * 100:.2f}%"
        )

    print(
        f"High-confidence errors "
        f"(confidence >= "
        f"{HIGH_CONFIDENCE_THRESHOLD:.0%}): "
        f"{high_confidence_errors:,}"
    )

    print(
        f"Manual review sample: "
        f"{len(manual_sample):,}"
    )

    print(
        "-" * 60
    )

    print(
        "\nFiles created:"
    )

    print(
        ALL_PREDICTIONS_FILE
    )

    print(
        FALSE_POSITIVES_FILE
    )

    print(
        FALSE_NEGATIVES_FILE
    )

    print(
        HIGH_CONFIDENCE_ERRORS_FILE
    )

    print(
        MANUAL_REVIEW_SAMPLE_FILE
    )


# Main

def main() -> None:
    """
    Run private validation error analysis.
    """

    set_random_seed(
        RANDOM_SEED
    )

    if not PRIVATE_EVAL_DATA.exists():

        raise FileNotFoundError(
            f"Private validation dataset not found: "
            f"{PRIVATE_EVAL_DATA}"
        )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    device = get_device()

    print(
        "\nPrivate error analysis"
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
        f"Dataset: "
        f"{PRIVATE_EVAL_DATA}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH.name}"
    )

    print(
        "-" * 60
    )

    # Load the original private validation data.

    original_data = pd.read_parquet(
        PRIVATE_EVAL_DATA
    )

    print(
        f"\nPrivate validation examples: "
        f"{len(original_data):,}"
    )

    if "label" not in original_data.columns:

        raise ValueError(
            "private_eval.parquet does not contain "
            "a 'label' column."
        )

    print(
        "Class distribution:"
    )

    print(
        original_data[
            "label"
        ].value_counts(
            sort=False
        )
    )

    # Create tokenizer and deterministic evaluation DataLoader.

    tokenizer = (
        get_gpt2_tokenizer()
    )

    data_loader = (
        create_evaluation_loader(
            parquet_path=PRIVATE_EVAL_DATA,
            tokenizer=tokenizer,
        )
    )

    if len(
        data_loader.dataset
    ) != len(
        original_data
    ):

        raise ValueError(
            "Dataset length mismatch between the Parquet file "
            "and the evaluation DataLoader."
        )

    # Create model and load selected checkpoint.

    model = create_model(
        device=device
    )

    checkpoint_metadata = (
        load_checkpoint(
            model=model,
            checkpoint_path=CHECKPOINT_PATH,
        )
    )

    print(
        "\nCheckpoint metadata:"
    )

    print(
        checkpoint_metadata
    )

    # Generate predictions.

    predictions = collect_predictions(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    # Merge predictions with original sender, subject and body.

    analysis = build_analysis_dataframe(
        original_data=original_data,
        predictions=predictions,
    )

    # Save complete and error-specific files.

    save_analysis_files(
        analysis
    )

    # Generate reproducible manual-review sample.

    manual_sample = (
        create_manual_review_sample(
            analysis
        )
    )

    # Print summary.

    print_summary(
        analysis=analysis,
        manual_sample=manual_sample,
    )

    print(
        "\nPrivate error analysis finished successfully."
    )


if __name__ == "__main__":
    main()