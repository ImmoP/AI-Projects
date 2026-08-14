"""
Evaluate dual Security V1 + V2 fusion with temporal reputation.

Architecture:

    GPT-2 content prediction
        ↓
    temporal sender/domain reputation
        ↓
    only for cold-start messages:
        ↓
    Security V1 + Security V2

The temporal model remains the primary classifier.

Security models are allowed to override the temporal prediction only
when there is no previous sender or domain history.

Spam override:
    Security V1 P(spam) >= 0.995
    OR
    Security V2 P(spam) >= 0.98

Ham override:
    Security V1 P(spam) <= 0.005
    AND
    Security V2 P(spam) <= 0.10

The asymmetric logic is intentional:

- strong evidence from either complementary security model can support
  a spam override;

- a temporal spam prediction is changed back to ham only when both
  security models agree with high confidence.

This is a development-set diagnostic analysis. These thresholds have
been informed by private_eval and must not be interpreted as final
independently validated production thresholds.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from spam_detector.paths import DATA_DIR

TEMPORAL_PATH = (
    DATA_DIR
    / "private_error_analysis"
    / "temporal_sender_reputation_results.csv"
)

SECURITY_V1_PATH = (
    DATA_DIR
    / "private_error_analysis"
    / "security_classifier_eval_predictions.csv"
)

SECURITY_V2_PATH = (
    DATA_DIR
    / "private_error_analysis"
    / "security_classifier_v2_eval_predictions.csv"
)

OUTPUT_DIR = (
    DATA_DIR
    / "private_error_analysis"
)


TRUTH_COLUMN = "corrected_label"

TEMPORAL_PREDICTION_COLUMN = (
    "temporal_hybrid_min_1_sender_domain"
)

SENDER_COUNT_COLUMN = (
    "temporal_sender_count"
)

DOMAIN_COUNT_COLUMN = (
    "temporal_domain_count"
)

V1_PROBABILITY_COLUMN = (
    "security_spam_probability"
)

V2_PROBABILITY_COLUMN = (
    "security_v2_spam_probability"
)


V1_SPAM_THRESHOLD = 0.995
V2_SPAM_THRESHOLD = 0.98

V1_HAM_THRESHOLD = 0.005
V2_HAM_THRESHOLD = 0.10


def check_required_columns(
    dataframe,
    columns,
    dataframe_name,
):
    """
    Verify that all required columns are present.
    """

    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"{dataframe_name} is missing required columns: "
            + ", ".join(missing)
        )


def calculate_metrics(
    y_true,
    predictions,
):
    """
    Calculate binary classification metrics.
    """

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        "total_errors":
            int(
                fp
                + fn
            ),
    }


def print_metrics(
    name,
    metrics,
):
    """
    Print classification metrics.
    """

    print()
    print(name)

    print(
        f"Accuracy:  "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:    "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1:        "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print(
        f"TN={metrics['tn']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"TP={metrics['tp']}"
    )

    print(
        f"Total errors: "
        f"{metrics['total_errors']}"
    )


def main():
    """
    Evaluate dual Security V1 + V2 fusion.
    """

    temporal = pd.read_csv(
        TEMPORAL_PATH
    )

    security_v1 = pd.read_csv(
        SECURITY_V1_PATH
    )

    security_v2 = pd.read_csv(
        SECURITY_V2_PATH
    )

    check_required_columns(
        dataframe=temporal,
        columns=[
            "row_id",
            TRUTH_COLUMN,
            TEMPORAL_PREDICTION_COLUMN,
            SENDER_COUNT_COLUMN,
            DOMAIN_COUNT_COLUMN,
        ],
        dataframe_name="Temporal results",
    )

    check_required_columns(
        dataframe=security_v1,
        columns=[
            "_private_row",
            V1_PROBABILITY_COLUMN,
        ],
        dataframe_name="Security V1 results",
    )

    check_required_columns(
        dataframe=security_v2,
        columns=[
            "_private_row",
            V2_PROBABILITY_COLUMN,
        ],
        dataframe_name="Security V2 results",
    )

    if temporal[
        TRUTH_COLUMN
    ].isna().any():

        missing_count = int(
            temporal[
                TRUTH_COLUMN
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            f"{TRUTH_COLUMN} contains "
            f"{missing_count} missing values."
        )

    if temporal[
        "row_id"
    ].duplicated().any():

        raise ValueError(
            "Temporal results contain duplicate row_id values."
        )

    if security_v1[
        "_private_row"
    ].duplicated().any():

        raise ValueError(
            "Security V1 results contain duplicate row IDs."
        )

    if security_v2[
        "_private_row"
    ].duplicated().any():

        raise ValueError(
            "Security V2 results contain duplicate row IDs."
        )

    security_v1_small = (
        security_v1[
            [
                "_private_row",
                V1_PROBABILITY_COLUMN,
            ]
        ]
        .rename(
            columns={
                "_private_row":
                    "row_id"
            }
        )
    )

    security_v2_small = (
        security_v2[
            [
                "_private_row",
                V2_PROBABILITY_COLUMN,
            ]
        ]
        .rename(
            columns={
                "_private_row":
                    "row_id"
            }
        )
    )

    dataframe = (
        temporal
        .merge(
            security_v1_small,
            on="row_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            security_v2_small,
            on="row_id",
            how="left",
            validate="one_to_one",
        )
    )

    if len(dataframe) != len(temporal):

        raise ValueError(
            "Row count changed after merging security predictions."
        )

    y_true = (
        dataframe[
            TRUTH_COLUMN
        ]
        .astype(int)
        .to_numpy()
    )

    temporal_prediction = (
        dataframe[
            TEMPORAL_PREDICTION_COLUMN
        ]
        .astype(int)
        .to_numpy()
    )

    sender_count = (
        dataframe[
            SENDER_COUNT_COLUMN
        ]
        .fillna(0)
        .to_numpy()
    )

    domain_count = (
        dataframe[
            DOMAIN_COUNT_COLUMN
        ]
        .fillna(0)
        .to_numpy()
    )

    v1_probability = (
        dataframe[
            V1_PROBABILITY_COLUMN
        ]
        .to_numpy()
    )

    v2_probability = (
        dataframe[
            V2_PROBABILITY_COLUMN
        ]
        .to_numpy()
    )

    v1_available = (
        dataframe[
            V1_PROBABILITY_COLUMN
        ]
        .notna()
        .to_numpy()
    )

    v2_available = (
        dataframe[
            V2_PROBABILITY_COLUMN
        ]
        .notna()
        .to_numpy()
    )

    cold_start = (
        (sender_count == 0)
        &
        (domain_count == 0)
    )

    both_security_available = (
        v1_available
        &
        v2_available
    )

    eligible_for_dual_security = (
        cold_start
        &
        both_security_available
    )

    v1_spam_signal = (
        eligible_for_dual_security
        &
        (
            v1_probability
            >= V1_SPAM_THRESHOLD
        )
    )

    v2_spam_signal = (
        eligible_for_dual_security
        &
        (
            v2_probability
            >= V2_SPAM_THRESHOLD
        )
    )

    spam_candidate = (
        v1_spam_signal
        |
        v2_spam_signal
    )

    v1_ham_signal = (
        eligible_for_dual_security
        &
        (
            v1_probability
            <= V1_HAM_THRESHOLD
        )
    )

    v2_ham_signal = (
        eligible_for_dual_security
        &
        (
            v2_probability
            <= V2_HAM_THRESHOLD
        )
    )

    ham_candidate = (
        v1_ham_signal
        &
        v2_ham_signal
    )

    conflicting_candidate = (
        spam_candidate
        &
        ham_candidate
    )

    if conflicting_candidate.any():

        conflicting_ids = (
            dataframe.loc[
                conflicting_candidate,
                "row_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Conflicting spam and ham security signals for row IDs: "
            f"{conflicting_ids}"
        )

    fused_prediction = (
        temporal_prediction
        .copy()
    )

    spam_override = (
        spam_candidate
        &
        (
            temporal_prediction
            == 0
        )
    )

    ham_override = (
        ham_candidate
        &
        (
            temporal_prediction
            == 1
        )
    )

    fused_prediction[
        spam_override
    ] = 1

    fused_prediction[
        ham_override
    ] = 0

    changed = (
        fused_prediction
        != temporal_prediction
    )

    temporal_correct = (
        temporal_prediction
        == y_true
    )

    fusion_correct = (
        fused_prediction
        == y_true
    )

    fixed_temporal_error = (
        changed
        &
        ~temporal_correct
        &
        fusion_correct
    )

    created_new_error = (
        changed
        &
        temporal_correct
        &
        ~fusion_correct
    )

    correct_override = (
        changed
        &
        fusion_correct
    )

    incorrect_override = (
        changed
        &
        ~fusion_correct
    )

    v1_only_spam_signal = (
        v1_spam_signal
        &
        ~v2_spam_signal
    )

    v2_only_spam_signal = (
        ~v1_spam_signal
        &
        v2_spam_signal
    )

    both_spam_signal = (
        v1_spam_signal
        &
        v2_spam_signal
    )

    print(
        "Dual Security V1 + V2 temporal fusion"
    )

    print()

    print(
        f"Messages: "
        f"{len(dataframe):,}"
    )

    print(
        f"Ham: "
        f"{(y_true == 0).sum():,}"
    )

    print(
        f"Spam: "
        f"{(y_true == 1).sum():,}"
    )

    print()

    print(
        f"Cold-start messages: "
        f"{cold_start.sum():,}"
    )

    print(
        f"Cold-start messages with both security models: "
        f"{eligible_for_dual_security.sum():,}"
    )

    print(
        f"Missing V1 probabilities: "
        f"{(~v1_available).sum():,}"
    )

    print(
        f"Missing V2 probabilities: "
        f"{(~v2_available).sum():,}"
    )

    print()

    print(
        "DUAL SECURITY RULE"
    )

    print()

    print(
        "Spam override if:"
    )

    print(
        f"  V1 >= {V1_SPAM_THRESHOLD}"
    )

    print(
        "  OR"
    )

    print(
        f"  V2 >= {V2_SPAM_THRESHOLD}"
    )

    print()

    print(
        "Ham override if:"
    )

    print(
        f"  V1 <= {V1_HAM_THRESHOLD}"
    )

    print(
        "  AND"
    )

    print(
        f"  V2 <= {V2_HAM_THRESHOLD}"
    )

    baseline_metrics = (
        calculate_metrics(
            y_true=y_true,
            predictions=temporal_prediction,
        )
    )

    fusion_metrics = (
        calculate_metrics(
            y_true=y_true,
            predictions=fused_prediction,
        )
    )

    print_metrics(
        name="TEMPORAL BASELINE",
        metrics=baseline_metrics,
    )

    print_metrics(
        name="DUAL SECURITY FUSION",
        metrics=fusion_metrics,
    )

    print()
    print(
        "SECURITY SIGNAL SUMMARY"
    )

    print()

    print(
        f"V1 spam signals: "
        f"{v1_spam_signal.sum():,}"
    )

    print(
        f"V2 spam signals: "
        f"{v2_spam_signal.sum():,}"
    )

    print(
        f"V1-only spam signals: "
        f"{v1_only_spam_signal.sum():,}"
    )

    print(
        f"V2-only spam signals: "
        f"{v2_only_spam_signal.sum():,}"
    )

    print(
        f"Both models spam signal: "
        f"{both_spam_signal.sum():,}"
    )

    print(
        f"Combined spam candidates: "
        f"{spam_candidate.sum():,}"
    )

    print()

    print(
        f"V1 ham signals: "
        f"{v1_ham_signal.sum():,}"
    )

    print(
        f"V2 ham signals: "
        f"{v2_ham_signal.sum():,}"
    )

    print(
        f"Both models ham signal: "
        f"{ham_candidate.sum():,}"
    )

    print()

    print(
        "OVERRIDE SUMMARY"
    )

    print()

    print(
        f"Spam overrides: "
        f"{spam_override.sum():,}"
    )

    print(
        f"Ham overrides: "
        f"{ham_override.sum():,}"
    )

    print(
        f"Total overrides: "
        f"{changed.sum():,}"
    )

    print(
        f"Correct overrides: "
        f"{correct_override.sum():,}"
    )

    print(
        f"Incorrect overrides: "
        f"{incorrect_override.sum():,}"
    )

    print(
        f"Temporal errors fixed: "
        f"{fixed_temporal_error.sum():,}"
    )

    print(
        f"New errors created: "
        f"{created_new_error.sum():,}"
    )

    dataframe[
        "cold_start"
    ] = cold_start

    dataframe[
        "dual_security_eligible"
    ] = eligible_for_dual_security

    dataframe[
        "v1_spam_signal"
    ] = v1_spam_signal

    dataframe[
        "v2_spam_signal"
    ] = v2_spam_signal

    dataframe[
        "v1_ham_signal"
    ] = v1_ham_signal

    dataframe[
        "v2_ham_signal"
    ] = v2_ham_signal

    dataframe[
        "dual_spam_candidate"
    ] = spam_candidate

    dataframe[
        "dual_ham_candidate"
    ] = ham_candidate

    dataframe[
        "dual_spam_override"
    ] = spam_override

    dataframe[
        "dual_ham_override"
    ] = ham_override

    dataframe[
        "temporal_correct"
    ] = temporal_correct

    dataframe[
        "dual_fusion_prediction"
    ] = fused_prediction

    dataframe[
        "dual_fusion_correct"
    ] = fusion_correct

    dataframe[
        "dual_fusion_changed_prediction"
    ] = changed

    dataframe[
        "dual_fusion_fixed_temporal_error"
    ] = fixed_temporal_error

    dataframe[
        "dual_fusion_created_new_error"
    ] = created_new_error

    changed_rows = (
        dataframe[
            dataframe[
                "dual_fusion_changed_prediction"
            ]
        ]
        .copy()
    )

    print()
    print(
        "CHANGED PREDICTIONS"
    )

    print()

    changed_columns = [
        "row_id",
        "sender",
        "subject",
        TRUTH_COLUMN,
        "predicted_label",
        TEMPORAL_PREDICTION_COLUMN,
        V1_PROBABILITY_COLUMN,
        V2_PROBABILITY_COLUMN,
        SENDER_COUNT_COLUMN,
        DOMAIN_COUNT_COLUMN,
        "v1_spam_signal",
        "v2_spam_signal",
        "v1_ham_signal",
        "v2_ham_signal",
        "dual_spam_override",
        "dual_ham_override",
        "dual_fusion_prediction",
        "temporal_correct",
        "dual_fusion_correct",
        "dual_fusion_fixed_temporal_error",
        "dual_fusion_created_new_error",
    ]

    changed_columns = [
        column
        for column in changed_columns
        if column in changed_rows.columns
    ]

    if changed_rows.empty:

        print(
            "No predictions were changed."
        )

    else:

        print(
            changed_rows[
                changed_columns
            ]
            .sort_values(
                [
                    "dual_spam_override",
                    V1_PROBABILITY_COLUMN,
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .to_string(
                index=False
            )
        )

    remaining_errors = (
        dataframe[
            ~dataframe[
                "dual_fusion_correct"
            ]
        ]
        .copy()
    )

    print()
    print(
        "ERRORS AFTER DUAL SECURITY FUSION"
    )

    print()

    error_columns = [
        "row_id",
        "sender",
        "subject",
        TRUTH_COLUMN,
        "predicted_label",
        TEMPORAL_PREDICTION_COLUMN,
        V1_PROBABILITY_COLUMN,
        V2_PROBABILITY_COLUMN,
        SENDER_COUNT_COLUMN,
        DOMAIN_COUNT_COLUMN,
        "cold_start",
        "dual_fusion_prediction",
    ]

    error_columns = [
        column
        for column in error_columns
        if column in remaining_errors.columns
    ]

    if remaining_errors.empty:

        print(
            "No remaining classification errors."
        )

    else:

        print(
            remaining_errors[
                error_columns
            ]
            .sort_values(
                "row_id"
            )
            .to_string(
                index=False
            )
        )

    metrics_output = pd.DataFrame(
        [
            {
                "model":
                    "temporal_baseline",

                **baseline_metrics,
            },
            {
                "model":
                    "dual_security_fusion",

                **fusion_metrics,
            },
        ]
    )

    metrics_path = (
        OUTPUT_DIR
        / "dual_security_temporal_fusion_metrics.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "dual_security_temporal_fusion_predictions.csv"
    )

    changed_path = (
        OUTPUT_DIR
        / "dual_security_temporal_fusion_changed_predictions.csv"
    )

    errors_path = (
        OUTPUT_DIR
        / "dual_security_temporal_fusion_errors.csv"
    )

    metrics_output.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    dataframe.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    changed_rows.to_csv(
        changed_path,
        index=False,
        encoding="utf-8-sig",
    )

    remaining_errors.to_csv(
        errors_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "Saved metrics:"
    )

    print(
        metrics_path
    )

    print()

    print(
        "Saved predictions:"
    )

    print(
        predictions_path
    )

    print()

    print(
        "Saved changed predictions:"
    )

    print(
        changed_path
    )

    print()

    print(
        "Saved remaining errors:"
    )

    print(
        errors_path
    )

    print()
    print(
        "Important:"
    )

    print(
        "These thresholds are development-set diagnostic thresholds."
    )

    print(
        "Do not report them as independently validated final "
        "production thresholds."
    )


if __name__ == "__main__":
    main()