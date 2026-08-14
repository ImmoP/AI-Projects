"""
Evaluate asymmetric fusion of temporal sender/domain reputation
and Security Classifier V2.

Architecture:

    GPT-2 content prediction
        ↓
    temporal sender/domain reputation
        ↓
    Security Classifier V2 for cold-start messages only

The temporal sender/domain model remains the primary classifier.

Security V2 may override the temporal prediction only when there is
no previous sender or domain history.

Spam and ham thresholds are evaluated independently:

    5 spam thresholds
    ×
    5 ham thresholds
    =
    25 threshold combinations

This is a diagnostic development-set analysis.

The best configuration on private_eval must not be interpreted as an
independently validated final operating threshold.
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

SECURITY_PATH = (
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

SECURITY_PROBABILITY_COLUMN = (
    "security_v2_spam_probability"
)


SPAM_THRESHOLDS = [
    0.90,
    0.95,
    0.98,
    0.99,
    0.995,
]

HAM_THRESHOLDS = [
    0.10,
    0.05,
    0.02,
    0.01,
    0.005,
]


def check_required_columns(
    dataframe,
    columns,
    dataframe_name,
):
    """
    Verify that all required columns exist.
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
    }


def print_metrics(
    name,
    metrics,
):
    """
    Print one set of classification metrics.
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


def evaluate_threshold_pair(
    temporal_prediction,
    y_true,
    eligible_for_security,
    security_probability,
    spam_threshold,
    ham_threshold,
):
    """
    Evaluate one asymmetric Security V2 threshold pair.
    """

    fused = (
        temporal_prediction
        .copy()
    )

    spam_candidate = (
        eligible_for_security
        &
        (
            security_probability
            >= spam_threshold
        )
    )

    ham_candidate = (
        eligible_for_security
        &
        (
            security_probability
            <= ham_threshold
        )
    )

    fused[
        spam_candidate
    ] = 1

    fused[
        ham_candidate
    ] = 0

    changed = (
        fused
        != temporal_prediction
    )

    spam_override = (
        changed
        &
        (
            fused
            == 1
        )
    )

    ham_override = (
        changed
        &
        (
            fused
            == 0
        )
    )

    correct_override = (
        changed
        &
        (
            fused
            == y_true
        )
    )

    incorrect_override = (
        changed
        &
        (
            fused
            != y_true
        )
    )

    fixed_temporal_error = (
        changed
        &
        (
            temporal_prediction
            != y_true
        )
        &
        (
            fused
            == y_true
        )
    )

    created_new_error = (
        changed
        &
        (
            temporal_prediction
            == y_true
        )
        &
        (
            fused
            != y_true
        )
    )

    metrics = calculate_metrics(
        y_true=y_true,
        predictions=fused,
    )

    result = {
        "spam_threshold":
            spam_threshold,

        "ham_threshold":
            ham_threshold,

        "spam_candidates":
            int(
                spam_candidate.sum()
            ),

        "ham_candidates":
            int(
                ham_candidate.sum()
            ),

        "spam_overrides":
            int(
                spam_override.sum()
            ),

        "ham_overrides":
            int(
                ham_override.sum()
            ),

        "overrides":
            int(
                changed.sum()
            ),

        "correct_overrides":
            int(
                correct_override.sum()
            ),

        "incorrect_overrides":
            int(
                incorrect_override.sum()
            ),

        "temporal_errors_fixed":
            int(
                fixed_temporal_error.sum()
            ),

        "new_errors_created":
            int(
                created_new_error.sum()
            ),

        **metrics,
    }

    return (
        result,
        fused,
    )


def print_f1_grid(
    results,
):
    """
    Print F1 scores as a spam-threshold × ham-threshold matrix.
    """

    grid = (
        results
        .pivot(
            index="spam_threshold",
            columns="ham_threshold",
            values="f1",
        )
        .sort_index(
            ascending=True
        )
    )

    grid = grid[
        sorted(
            grid.columns,
            reverse=True,
        )
    ]

    print()
    print(
        "F1 GRID"
    )

    print()

    print(
        "Rows    = spam threshold"
    )

    print(
        "Columns = ham threshold"
    )

    print()

    print(
        grid.to_string(
            float_format=lambda x: f"{x:.4f}",
        )
    )


def print_error_grid(
    results,
):
    """
    Print total classification errors for every threshold pair.
    """

    grid = (
        results
        .pivot(
            index="spam_threshold",
            columns="ham_threshold",
            values="total_errors",
        )
        .sort_index(
            ascending=True
        )
    )

    grid = grid[
        sorted(
            grid.columns,
            reverse=True,
        )
    ]

    print()
    print(
        "TOTAL ERROR GRID"
    )

    print()

    print(
        "Rows    = spam threshold"
    )

    print(
        "Columns = ham threshold"
    )

    print()

    print(
        grid.to_string()
    )


def main():
    """
    Evaluate all 25 asymmetric Temporal + Security V2 combinations.
    """

    temporal = pd.read_csv(
        TEMPORAL_PATH
    )

    security = pd.read_csv(
        SECURITY_PATH
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
        dataframe=security,
        columns=[
            "_private_row",
            SECURITY_PROBABILITY_COLUMN,
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
            "Temporal result file contains duplicate row_id values."
        )

    security_small = (
        security[
            [
                "_private_row",
                SECURITY_PROBABILITY_COLUMN,
            ]
        ]
        .rename(
            columns={
                "_private_row":
                    "row_id"
            }
        )
    )

    if security_small[
        "row_id"
    ].duplicated().any():

        raise ValueError(
            "Security V2 prediction file contains duplicate row_id values."
        )

    dataframe = (
        temporal
        .merge(
            security_small,
            on="row_id",
            how="left",
            validate="one_to_one",
        )
    )

    if len(dataframe) != len(temporal):

        raise ValueError(
            "Row count changed after merging temporal and Security V2 data."
        )

    missing_security = int(
        dataframe[
            SECURITY_PROBABILITY_COLUMN
        ]
        .isna()
        .sum()
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

    security_probability = (
        dataframe[
            SECURITY_PROBABILITY_COLUMN
        ]
        .to_numpy()
    )

    security_available = (
        dataframe[
            SECURITY_PROBABILITY_COLUMN
        ]
        .notna()
        .to_numpy()
    )

    cold_start = (
        (sender_count == 0)
        &
        (domain_count == 0)
    )

    eligible_for_security = (
        cold_start
        &
        security_available
    )

    print(
        "Asymmetric Temporal + Security V2 fusion evaluation"
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

    print(
        f"Cold-start messages: "
        f"{cold_start.sum():,}"
    )

    print(
        f"Cold-start messages with Security V2: "
        f"{eligible_for_security.sum():,}"
    )

    print(
        f"Missing Security V2 probabilities: "
        f"{missing_security:,}"
    )

    print()

    print(
        f"Spam thresholds: "
        f"{SPAM_THRESHOLDS}"
    )

    print(
        f"Ham thresholds: "
        f"{HAM_THRESHOLDS}"
    )

    print(
        f"Threshold combinations: "
        f"{len(SPAM_THRESHOLDS) * len(HAM_THRESHOLDS)}"
    )

    print()

    print(
        "Temporal prediction column:"
    )

    print(
        TEMPORAL_PREDICTION_COLUMN
    )

    print()

    print(
        "Security probability column:"
    )

    print(
        SECURITY_PROBABILITY_COLUMN
    )

    baseline_metrics = calculate_metrics(
        y_true=y_true,
        predictions=temporal_prediction,
    )

    print_metrics(
        name="TEMPORAL BASELINE",
        metrics=baseline_metrics,
    )

    result_rows = []

    prediction_map = {}

    for spam_threshold in SPAM_THRESHOLDS:

        for ham_threshold in HAM_THRESHOLDS:

            (
                result,
                fused,
            ) = evaluate_threshold_pair(
                temporal_prediction=temporal_prediction,
                y_true=y_true,
                eligible_for_security=eligible_for_security,
                security_probability=security_probability,
                spam_threshold=spam_threshold,
                ham_threshold=ham_threshold,
            )

            result_rows.append(
                result
            )

            prediction_map[
                (
                    spam_threshold,
                    ham_threshold,
                )
            ] = fused

    results = pd.DataFrame(
        result_rows
    )

    results[
        "total_errors"
    ] = (
        results[
            "fp"
        ]
        + results[
            "fn"
        ]
    )

    results = (
        results
        .sort_values(
            [
                "spam_threshold",
                "ham_threshold",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "ALL 25 ASYMMETRIC SECURITY V2 FUSION RESULTS"
    )

    print()

    display_columns = [
        "spam_threshold",
        "ham_threshold",
        "spam_candidates",
        "ham_candidates",
        "spam_overrides",
        "ham_overrides",
        "overrides",
        "correct_overrides",
        "incorrect_overrides",
        "temporal_errors_fixed",
        "new_errors_created",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "total_errors",
    ]

    print(
        results[
            display_columns
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print_f1_grid(
        results
    )

    print_error_grid(
        results
    )

    ranked = (
        results
        .sort_values(
            [
                "f1",
                "new_errors_created",
                "total_errors",
                "overrides",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "TOP 10 SECURITY V2 DIAGNOSTIC CONFIGURATIONS"
    )

    print()

    print(
        ranked[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    best = ranked.iloc[0]

    best_spam_threshold = float(
        best[
            "spam_threshold"
        ]
    )

    best_ham_threshold = float(
        best[
            "ham_threshold"
        ]
    )

    print()
    print(
        "BEST SECURITY V2 DIAGNOSTIC CONFIGURATION"
    )

    print()

    print(
        best[
            display_columns
        ]
        .to_string()
    )

    print()

    print(
        "Important: this is the best configuration on the "
        "development validation set, not a final independently "
        "validated operating threshold."
    )

    fused = (
        prediction_map[
            (
                best_spam_threshold,
                best_ham_threshold,
            )
        ]
        .copy()
    )

    spam_candidate = (
        eligible_for_security
        &
        (
            security_probability
            >= best_spam_threshold
        )
    )

    ham_candidate = (
        eligible_for_security
        &
        (
            security_probability
            <= best_ham_threshold
        )
    )

    changed = (
        fused
        != temporal_prediction
    )

    spam_override = (
        changed
        &
        (
            fused
            == 1
        )
    )

    ham_override = (
        changed
        &
        (
            fused
            == 0
        )
    )

    dataframe[
        "cold_start"
    ] = cold_start

    dataframe[
        "security_v2_available"
    ] = security_available

    dataframe[
        "security_v2_fusion_eligible"
    ] = eligible_for_security

    dataframe[
        "selected_spam_threshold"
    ] = best_spam_threshold

    dataframe[
        "selected_ham_threshold"
    ] = best_ham_threshold

    dataframe[
        "security_v2_spam_candidate"
    ] = spam_candidate

    dataframe[
        "security_v2_ham_candidate"
    ] = ham_candidate

    dataframe[
        "security_v2_spam_override"
    ] = spam_override

    dataframe[
        "security_v2_ham_override"
    ] = ham_override

    dataframe[
        "fusion_prediction"
    ] = fused

    dataframe[
        "temporal_correct"
    ] = (
        temporal_prediction
        == y_true
    )

    dataframe[
        "fusion_correct"
    ] = (
        fused
        == y_true
    )

    dataframe[
        "fusion_changed_prediction"
    ] = changed

    dataframe[
        "fusion_fixed_temporal_error"
    ] = (
        changed
        &
        ~dataframe[
            "temporal_correct"
        ]
        &
        dataframe[
            "fusion_correct"
        ]
    )

    dataframe[
        "fusion_created_new_error"
    ] = (
        changed
        &
        dataframe[
            "temporal_correct"
        ]
        &
        ~dataframe[
            "fusion_correct"
        ]
    )

    changed_rows = (
        dataframe[
            dataframe[
                "fusion_changed_prediction"
            ]
        ]
        .copy()
    )

    print()
    print(
        "CHANGED PREDICTIONS FOR BEST SECURITY V2 CONFIGURATION"
    )

    print()

    changed_columns = [
        "row_id",
        "sender",
        "subject",
        TRUTH_COLUMN,
        "predicted_label",
        TEMPORAL_PREDICTION_COLUMN,
        SECURITY_PROBABILITY_COLUMN,
        SENDER_COUNT_COLUMN,
        DOMAIN_COUNT_COLUMN,
        "security_v2_spam_override",
        "security_v2_ham_override",
        "fusion_prediction",
        "temporal_correct",
        "fusion_correct",
        "fusion_fixed_temporal_error",
        "fusion_created_new_error",
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
                SECURITY_PROBABILITY_COLUMN,
                ascending=False,
            )
            .to_string(
                index=False
            )
        )

    remaining_errors = (
        dataframe[
            ~dataframe[
                "fusion_correct"
            ]
        ]
        .copy()
    )

    print()
    print(
        "ERRORS AFTER BEST SECURITY V2 FUSION"
    )

    print()

    error_columns = [
        "row_id",
        "sender",
        "subject",
        TRUTH_COLUMN,
        "predicted_label",
        TEMPORAL_PREDICTION_COLUMN,
        SECURITY_PROBABILITY_COLUMN,
        SENDER_COUNT_COLUMN,
        DOMAIN_COUNT_COLUMN,
        "cold_start",
        "fusion_prediction",
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

    results_path = (
        OUTPUT_DIR
        / "security_v2_temporal_asymmetric_grid_results.csv"
    )

    ranked_path = (
        OUTPUT_DIR
        / "security_v2_temporal_asymmetric_ranked_results.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "security_v2_temporal_asymmetric_best_predictions.csv"
    )

    changed_path = (
        OUTPUT_DIR
        / "security_v2_temporal_asymmetric_best_changed_predictions.csv"
    )

    results.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    ranked.to_csv(
        ranked_path,
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

    print()
    print(
        "Saved full Security V2 25-combination grid:"
    )

    print(
        results_path
    )

    print()

    print(
        "Saved ranked Security V2 configurations:"
    )

    print(
        ranked_path
    )

    print()

    print(
        "Saved best Security V2 predictions:"
    )

    print(
        predictions_path
    )

    print()

    print(
        "Saved changed Security V2 predictions:"
    )

    print(
        changed_path
    )


if __name__ == "__main__":
    main()