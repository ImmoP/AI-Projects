import json
from email.utils import parseaddr
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "private_error_analysis"
    / "all_predictions_adjudicated.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "private_error_analysis"
)


def extract_sender_information(sender):
    """
    Extract display name, email address, and domain
    from the sender field.
    """

    if pd.isna(sender):
        return "", "", ""

    sender = str(sender)

    display_name, email_address = parseaddr(sender)

    email_address = email_address.strip().lower()

    if "@" in email_address:
        domain = email_address.rsplit("@", 1)[1].lower()
    else:
        domain = ""

    return (
        display_name.strip(),
        email_address,
        domain,
    )


def confidence_bucket(confidence):
    """
    Group model confidence into simple ranges.
    """

    if confidence >= 0.99:
        return ">= 0.99"

    if confidence >= 0.90:
        return "0.90 - 0.99"

    if confidence >= 0.75:
        return "0.75 - 0.90"

    if confidence >= 0.50:
        return "0.50 - 0.75"

    return "< 0.50"


def main():
    print(f"Loading:\n{INPUT_PATH}\n")

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "row_id",
        "predicted_label",
        "spam_probability",
        "ham_probability",
        "sender",
        "subject",
        "text",
        "corrected_label",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["predicted_label"] = pd.to_numeric(
        df["predicted_label"],
        errors="coerce",
    )

    df["corrected_label"] = pd.to_numeric(
        df["corrected_label"],
        errors="coerce",
    )

    df["spam_probability"] = pd.to_numeric(
        df["spam_probability"],
        errors="coerce",
    )

    df["ham_probability"] = pd.to_numeric(
        df["ham_probability"],
        errors="coerce",
    )

    # A genuine model error remains when the prediction
    # disagrees with the manually adjudicated label.

    errors = df[
        df["predicted_label"]
        != df["corrected_label"]
    ].copy()

    errors["corrected_label_name"] = (
        errors["corrected_label"]
        .map(
            {
                0: "ham",
                1: "spam",
            }
        )
    )

    errors["adjudicated_error_type"] = "unknown"

    errors.loc[
        (
            (errors["corrected_label"] == 0)
            & (errors["predicted_label"] == 1)
        ),
        "adjudicated_error_type",
    ] = "false_positive"

    errors.loc[
        (
            (errors["corrected_label"] == 1)
            & (errors["predicted_label"] == 0)
        ),
        "adjudicated_error_type",
    ] = "false_negative"

    sender_information = (
        errors["sender"]
        .apply(extract_sender_information)
    )

    errors["sender_display_name"] = [
        value[0]
        for value in sender_information
    ]

    errors["sender_email"] = [
        value[1]
        for value in sender_information
    ]

    errors["sender_domain"] = [
        value[2]
        for value in sender_information
    ]

    # Recalculate confidence directly from the predicted class.

    errors["prediction_confidence"] = errors.apply(
        lambda row: (
            row["spam_probability"]
            if row["predicted_label"] == 1
            else row["ham_probability"]
        ),
        axis=1,
    )

    errors["confidence_bucket"] = (
        errors["prediction_confidence"]
        .apply(confidence_bucket)
    )

    # Create a short preview if one is not already available.

    if "body_preview" not in errors.columns:
        errors["body_preview"] = (
            errors["text"]
            .fillna("")
            .astype(str)
            .str.replace(
                r"\s+",
                " ",
                regex=True,
            )
            .str.slice(
                0,
                500,
            )
        )

    output_columns = [
        "row_id",
        "adjudicated_error_type",
        "corrected_label",
        "corrected_label_name",
        "predicted_label",
        "predicted_label_name",
        "spam_probability",
        "ham_probability",
        "prediction_confidence",
        "confidence_bucket",
        "sender",
        "sender_display_name",
        "sender_email",
        "sender_domain",
        "subject",
        "body_preview",
        "text",
        "date",
        "source",
        "source_split",
        "true_label",
        "true_label_name",
        "label_changed",
        "manually_reviewed",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in errors.columns
    ]

    errors = errors[
        output_columns
    ].sort_values(
        by=[
            "adjudicated_error_type",
            "prediction_confidence",
        ],
        ascending=[
            True,
            False,
        ],
    )

    false_positives = errors[
        errors["adjudicated_error_type"]
        == "false_positive"
    ].copy()

    false_negatives = errors[
        errors["adjudicated_error_type"]
        == "false_negative"
    ].copy()

    genuine_errors_path = (
        OUTPUT_DIR
        / "genuine_model_errors.csv"
    )

    false_positives_path = (
        OUTPUT_DIR
        / "genuine_false_positives.csv"
    )

    false_negatives_path = (
        OUTPUT_DIR
        / "genuine_false_negatives.csv"
    )

    domain_summary_path = (
        OUTPUT_DIR
        / "genuine_error_domain_summary.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "genuine_error_summary.json"
    )

    errors.to_csv(
        genuine_errors_path,
        index=False,
        encoding="utf-8-sig",
    )

    false_positives.to_csv(
        false_positives_path,
        index=False,
        encoding="utf-8-sig",
    )

    false_negatives.to_csv(
        false_negatives_path,
        index=False,
        encoding="utf-8-sig",
    )

    domain_summary = (
        errors
        .groupby(
            [
                "sender_domain",
                "adjudicated_error_type",
            ],
            dropna=False,
        )
        .agg(
            error_count=(
                "row_id",
                "count",
            ),
            mean_spam_probability=(
                "spam_probability",
                "mean",
            ),
            mean_prediction_confidence=(
                "prediction_confidence",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "error_count",
            ascending=False,
        )
    )

    domain_summary.to_csv(
        domain_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    high_confidence_errors = errors[
        errors["prediction_confidence"]
        >= 0.90
    ]

    very_high_confidence_errors = errors[
        errors["prediction_confidence"]
        >= 0.99
    ]

    summary = {
        "total_examples": int(len(df)),
        "genuine_model_errors": int(len(errors)),
        "false_positives": int(len(false_positives)),
        "false_negatives": int(len(false_negatives)),
        "high_confidence_errors_ge_0_90": int(
            len(high_confidence_errors)
        ),
        "very_high_confidence_errors_ge_0_99": int(
            len(very_high_confidence_errors)
        ),
        "mean_error_confidence": (
            float(
                errors["prediction_confidence"].mean()
            )
            if len(errors) > 0
            else None
        ),
        "mean_false_positive_spam_probability": (
            float(
                false_positives[
                    "spam_probability"
                ].mean()
            )
            if len(false_positives) > 0
            else None
        ),
        "mean_false_negative_spam_probability": (
            float(
                false_negatives[
                    "spam_probability"
                ].mean()
            )
            if len(false_negatives) > 0
            else None
        ),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("GENUINE MODEL ERROR ANALYSIS")
    print()

    print(
        f"Total private validation examples: "
        f"{len(df):,}"
    )

    print(
        f"Genuine model errors: "
        f"{len(errors):,}"
    )

    print(
        f"False positives: "
        f"{len(false_positives):,}"
    )

    print(
        f"False negatives: "
        f"{len(false_negatives):,}"
    )

    print()

    print(
        "High-confidence errors "
        f"(confidence >= 0.90): "
        f"{len(high_confidence_errors):,}"
    )

    print(
        "Very high-confidence errors "
        f"(confidence >= 0.99): "
        f"{len(very_high_confidence_errors):,}"
    )

    if len(errors) > 0:
        print(
            f"Mean confidence on errors: "
            f"{errors['prediction_confidence'].mean():.4f}"
        )

    print()

    print("FALSE POSITIVES")
    print()

    if len(false_positives) == 0:
        print("None")
    else:
        for _, row in false_positives.iterrows():
            print(
                f"Row {row['row_id']} | "
                f"P(spam)={row['spam_probability']:.4f} | "
                f"{row['sender_email']} | "
                f"{row['subject']}"
            )

    print()

    print("FALSE NEGATIVES")
    print()

    if len(false_negatives) == 0:
        print("None")
    else:
        for _, row in false_negatives.iterrows():
            print(
                f"Row {row['row_id']} | "
                f"P(spam)={row['spam_probability']:.4f} | "
                f"{row['sender_email']} | "
                f"{row['subject']}"
            )

    print()

    print("Saved:")
    print(genuine_errors_path)
    print(false_positives_path)
    print(false_negatives_path)
    print(domain_summary_path)
    print(summary_path)


if __name__ == "__main__":
    main()