"""
Train Security/Identity Classifier V2.

V2 removes header-presence features that showed strong dataset/time
confounding in V1.

Instead, the model focuses on:

- visible sender identity consistency
- full sender-domain identity consistency
- explicit SPF/DKIM/DMARC failures
- explicit domain-alignment failures
- combinations of identity and technical evidence

The classifier is trained only on private_train and evaluated on the
adjudicated private_eval split.

The model is intended as a security signal for later fusion with:

- GPT-2 content classification
- temporal sender/domain reputation
"""

import json
import re
from difflib import SequenceMatcher

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spam_detector.paths import DATA_DIR

TRAIN_PATH = (
    DATA_DIR
    / "security_features"
    / "private_train_security_matched.parquet"
)

EVAL_PATH = (
    DATA_DIR
    / "security_features"
    / "private_eval_security_matched.parquet"
)

TEMPORAL_PATH = (
    DATA_DIR
    / "private_error_analysis"
    / "temporal_sender_reputation_results.csv"
)

RESULTS_DIR = (
    DATA_DIR
    / "private_error_analysis"
)

MODEL_DIR = (
    DATA_DIR.parent
    / "models"
    / "security_classifier_v2"
)


FEATURES = [
    "org_domain_identity_similarity",
    "full_domain_identity_similarity",
    "org_domain_name_in_display",
    "full_domain_label_in_display",
    "delegated_subdomain_identity_match",

    "spf_fail",
    "spf_softfail",
    "dkim_fail",
    "dmarc_fail",

    "return_path_org_mismatch",
    "dkim_org_mismatch",
    "reply_to_org_mismatch",

    "authentication_failure_count",
    "alignment_failure_count",

    "identity_mismatch",
    "identity_mismatch_and_auth_failure",
    "identity_mismatch_and_alignment_failure",
    "identity_mismatch_but_full_alignment",
]


IMPORTANT_ROWS = [
    73,
    434,
    915,
    925,
    1249,
    1295,
    1411,
    1466,
    1813,
]


def value_is_missing(value):
    """
    Return True when a value is missing.
    """

    if value is None:
        return True

    try:
        return bool(
            pd.isna(value)
        )

    except TypeError:
        return False


def normalize_identity(value):
    """
    Normalize a display name or domain component.
    """

    if value_is_missing(value):
        return ""

    value = (
        str(value)
        .casefold()
    )

    return re.sub(
        r"[^a-z0-9]+",
        "",
        value,
    )


def get_org_domain_label(domain):
    """
    Extract the first brand-like label from an organizational domain.

    Examples:

        cdkeys.com
            -> cdkeys

        qatarairways.com.qa
            -> qatarairways
    """

    if value_is_missing(domain):
        return ""

    domain = (
        str(domain)
        .casefold()
        .strip()
    )

    if not domain:
        return ""

    return domain.split(".")[0]


def get_full_domain_labels(domain):
    """
    Extract all meaningful labels from the complete From domain.

    Example:

        wewash.zendesk.com
            -> ["wewash", "zendesk"]
    """

    if value_is_missing(domain):
        return []

    labels = []

    for part in str(domain).split("."):

        normalized = (
            normalize_identity(
                part
            )
        )

        if len(normalized) >= 3:
            labels.append(
                normalized
            )

    return labels


def calculate_org_domain_similarity(
    display_name,
    org_domain,
):
    """
    Compare the display name with the organizational-domain label.
    """

    display = normalize_identity(
        display_name
    )

    domain = normalize_identity(
        get_org_domain_label(
            org_domain
        )
    )

    if not display or not domain:
        return np.nan

    return SequenceMatcher(
        None,
        display,
        domain,
    ).ratio()


def calculate_full_domain_similarity(
    display_name,
    from_domain,
):
    """
    Return the highest similarity between the display name and any
    meaningful label in the complete From domain.
    """

    display = normalize_identity(
        display_name
    )

    labels = get_full_domain_labels(
        from_domain
    )

    if not display or not labels:
        return np.nan

    return max(
        SequenceMatcher(
            None,
            display,
            label,
        ).ratio()
        for label in labels
    )


def org_domain_name_in_display(
    display_name,
    org_domain,
):
    """
    Check whether the organizational-domain label occurs in the
    display name.
    """

    display = normalize_identity(
        display_name
    )

    domain = normalize_identity(
        get_org_domain_label(
            org_domain
        )
    )

    if not display or not domain:
        return np.nan

    if len(domain) < 3:
        return np.nan

    return float(
        domain
        in display
    )


def full_domain_label_in_display(
    display_name,
    from_domain,
):
    """
    Check whether any complete From-domain label occurs in the
    visible display name.

    Example:

        WeWash GmbH
        wewash.zendesk.com

        -> True
    """

    display = normalize_identity(
        display_name
    )

    labels = get_full_domain_labels(
        from_domain
    )

    if not display or not labels:
        return np.nan

    return float(
        any(
            label in display
            for label in labels
        )
    )


def explicit_false(value):
    """
    Return 1 only when an alignment result is explicitly False.

    Missing values are not interpreted as mismatches.
    """

    if value_is_missing(value):
        return 0.0

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return float(
            not value
        )

    normalized = (
        str(value)
        .strip()
        .casefold()
    )

    if normalized in {
        "false",
        "0",
        "no",
    }:
        return 1.0

    return 0.0


def explicit_true(value):
    """
    Return 1 only when a value is explicitly True.
    """

    if value_is_missing(value):
        return 0.0

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return float(
            value
        )

    normalized = (
        str(value)
        .strip()
        .casefold()
    )

    if normalized in {
        "true",
        "1",
        "yes",
    }:
        return 1.0

    return 0.0


def auth_result_is(
    value,
    expected,
):
    """
    Return 1 when an authentication result matches the requested
    result string.
    """

    if value_is_missing(value):
        return 0.0

    normalized = (
        str(value)
        .strip()
        .casefold()
    )

    return float(
        normalized
        == expected
    )


def create_features(
    dataframe,
):
    """
    Create V2 security and identity features.
    """

    dataframe = (
        dataframe
        .copy()
    )

    dataframe[
        "org_domain_identity_similarity"
    ] = dataframe.apply(
        lambda row:
            calculate_org_domain_similarity(
                row.get(
                    "security_display_name"
                ),
                row.get(
                    "security_from_org_domain"
                ),
            ),
        axis=1,
    )

    dataframe[
        "full_domain_identity_similarity"
    ] = dataframe.apply(
        lambda row:
            calculate_full_domain_similarity(
                row.get(
                    "security_display_name"
                ),
                row.get(
                    "security_from_domain"
                ),
            ),
        axis=1,
    )

    dataframe[
        "org_domain_name_in_display"
    ] = dataframe.apply(
        lambda row:
            org_domain_name_in_display(
                row.get(
                    "security_display_name"
                ),
                row.get(
                    "security_from_org_domain"
                ),
            ),
        axis=1,
    )

    dataframe[
        "full_domain_label_in_display"
    ] = dataframe.apply(
        lambda row:
            full_domain_label_in_display(
                row.get(
                    "security_display_name"
                ),
                row.get(
                    "security_from_domain"
                ),
            ),
        axis=1,
    )

    dataframe[
        "delegated_subdomain_identity_match"
    ] = (
        dataframe[
            "full_domain_label_in_display"
        ]
        .eq(1.0)
        &
        dataframe[
            "org_domain_name_in_display"
        ]
        .eq(0.0)
    ).astype(float)

    dataframe[
        "spf_fail"
    ] = dataframe[
        "security_spf_result"
    ].map(
        lambda value:
            auth_result_is(
                value,
                "fail",
            )
    )

    dataframe[
        "spf_softfail"
    ] = dataframe[
        "security_spf_result"
    ].map(
        lambda value:
            auth_result_is(
                value,
                "softfail",
            )
    )

    dataframe[
        "dkim_fail"
    ] = dataframe[
        "security_dkim_result"
    ].map(
        lambda value:
            auth_result_is(
                value,
                "fail",
            )
    )

    dataframe[
        "dmarc_fail"
    ] = dataframe[
        "security_dmarc_result"
    ].map(
        lambda value:
            auth_result_is(
                value,
                "fail",
            )
    )

    dataframe[
        "return_path_org_mismatch"
    ] = dataframe[
        "security_from_return_path_org_match"
    ].map(
        explicit_false
    )

    dataframe[
        "dkim_org_mismatch"
    ] = dataframe[
        "security_from_dkim_org_match"
    ].map(
        explicit_false
    )

    dataframe[
        "reply_to_org_mismatch"
    ] = dataframe[
        "security_from_reply_to_org_match"
    ].map(
        explicit_false
    )

    dataframe[
        "authentication_failure_count"
    ] = (
        dataframe[
            "spf_fail"
        ]
        + dataframe[
            "spf_softfail"
        ]
        + dataframe[
            "dkim_fail"
        ]
        + dataframe[
            "dmarc_fail"
        ]
    )

    dataframe[
        "alignment_failure_count"
    ] = (
        dataframe[
            "return_path_org_mismatch"
        ]
        + dataframe[
            "dkim_org_mismatch"
        ]
        + dataframe[
            "reply_to_org_mismatch"
        ]
    )

    dataframe[
        "identity_mismatch"
    ] = (
        dataframe[
            "full_domain_label_in_display"
        ]
        .eq(0.0)
    ).astype(float)

    dataframe[
        "identity_mismatch_and_auth_failure"
    ] = (
        dataframe[
            "identity_mismatch"
        ]
        * (
            dataframe[
                "authentication_failure_count"
            ]
            > 0
        )
        .astype(float)
    )

    dataframe[
        "identity_mismatch_and_alignment_failure"
    ] = (
        dataframe[
            "identity_mismatch"
        ]
        * (
            dataframe[
                "alignment_failure_count"
            ]
            > 0
        )
        .astype(float)
    )

    spf_pass = (
        dataframe[
            "security_spf_result"
        ]
        .map(
            lambda value:
                auth_result_is(
                    value,
                    "pass",
                )
        )
    )

    dkim_pass = (
        dataframe[
            "security_dkim_result"
        ]
        .map(
            lambda value:
                auth_result_is(
                    value,
                    "pass",
                )
        )
    )

    dmarc_pass = (
        dataframe[
            "security_dmarc_result"
        ]
        .map(
            lambda value:
                auth_result_is(
                    value,
                    "pass",
                )
        )
    )

    return_path_match = (
        dataframe[
            "security_from_return_path_org_match"
        ]
        .map(
            explicit_true
        )
    )

    dkim_match = (
        dataframe[
            "security_from_dkim_org_match"
        ]
        .map(
            explicit_true
        )
    )

    dataframe[
        "full_technical_alignment"
    ] = (
        (
            spf_pass
            == 1
        )
        &
        (
            dkim_pass
            == 1
        )
        &
        (
            dmarc_pass
            == 1
        )
        &
        (
            return_path_match
            == 1
        )
        &
        (
            dkim_match
            == 1
        )
    ).astype(float)

    dataframe[
        "identity_mismatch_but_full_alignment"
    ] = (
        dataframe[
            "identity_mismatch"
        ]
        * dataframe[
            "full_technical_alignment"
        ]
    )

    return dataframe


def build_eval_ground_truth(
    dataframe,
):
    """
    Add manually adjudicated validation labels.
    """

    temporal = (
        pd.read_csv(
            TEMPORAL_PATH
        )
    )

    corrections = (
        temporal[
            [
                "row_id",
                "corrected_label",
            ]
        ]
        .drop_duplicates(
            "row_id"
        )
        .rename(
            columns={
                "row_id":
                    "_private_row"
            }
        )
    )

    dataframe = (
        dataframe
        .merge(
            corrections,
            on="_private_row",
            how="left",
            validate="one_to_one",
        )
    )

    dataframe[
        "analysis_label"
    ] = (
        dataframe[
            "corrected_label"
        ]
        .where(
            dataframe[
                "corrected_label"
            ]
            .notna(),
            dataframe[
                "label"
            ],
        )
        .astype(int)
    )

    return dataframe


def build_pipeline():
    """
    Build a regularized logistic-regression classifier.
    """

    feature_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "features",
                feature_pipeline,
                FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=2000,
        random_state=123,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5,
):
    """
    Calculate validation metrics.
    """

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

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

    metrics = {
        "threshold":
            float(
                threshold
            ),

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

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    y_true,
                    probabilities,
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

    return (
        metrics,
        predictions,
    )


def get_coefficients(
    pipeline,
):
    """
    Extract logistic-regression coefficients.
    """

    preprocessor = (
        pipeline.named_steps[
            "preprocessor"
        ]
    )

    classifier = (
        pipeline.named_steps[
            "classifier"
        ]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    coefficients = (
        classifier
        .coef_[0]
    )

    result = pd.DataFrame(
        {
            "feature":
                feature_names,

            "coefficient":
                coefficients,

            "absolute_coefficient":
                np.abs(
                    coefficients
                ),
        }
    )

    return (
        result
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def print_metrics(
    metrics,
):
    """
    Print validation metrics.
    """

    print()
    print(
        "SECURITY CLASSIFIER V2 VALIDATION"
    )

    print()

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
        f"ROC AUC:   "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"PR AUC:    "
        f"{metrics['pr_auc']:.6f}"
    )

    print()

    print(
        f"TN={metrics['tn']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"TP={metrics['tp']}"
    )


def print_important_rows(
    dataframe,
):
    """
    Print security predictions for important diagnostic cases.
    """

    rows = (
        dataframe[
            dataframe[
                "_private_row"
            ]
            .isin(
                IMPORTANT_ROWS
            )
        ]
        .copy()
        .sort_values(
            "_private_row"
        )
    )

    columns = [
        "_private_row",
        "sender",
        "subject",
        "analysis_label",
        "security_v2_spam_probability",
        "security_v2_prediction",
        "org_domain_identity_similarity",
        "full_domain_identity_similarity",
        "org_domain_name_in_display",
        "full_domain_label_in_display",
        "delegated_subdomain_identity_match",
        "authentication_failure_count",
        "alignment_failure_count",
        "identity_mismatch",
        "identity_mismatch_and_auth_failure",
        "identity_mismatch_and_alignment_failure",
        "identity_mismatch_but_full_alignment",
    ]

    print()
    print(
        "IMPORTANT DIAGNOSTIC ROWS"
    )

    print()

    print(
        rows[
            columns
        ]
        .to_string(
            index=False,
        )
    )


def main():
    """
    Train and evaluate Security Classifier V2.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = (
        pd.read_parquet(
            TRAIN_PATH
        )
    )

    eval_df = (
        pd.read_parquet(
            EVAL_PATH
        )
    )

    train_df = (
        create_features(
            train_df
        )
    )

    eval_df = (
        create_features(
            eval_df
        )
    )

    eval_df = (
        build_eval_ground_truth(
            eval_df
        )
    )

    X_train = (
        train_df[
            FEATURES
        ]
    )

    y_train = (
        train_df[
            "label"
        ]
        .astype(int)
    )

    X_eval = (
        eval_df[
            FEATURES
        ]
    )

    y_eval = (
        eval_df[
            "analysis_label"
        ]
        .astype(int)
    )

    print(
        "Training Security Classifier V2"
    )

    print()

    print(
        f"Training messages: "
        f"{len(train_df):,}"
    )

    print(
        f"Training ham: "
        f"{(y_train == 0).sum():,}"
    )

    print(
        f"Training spam: "
        f"{(y_train == 1).sum():,}"
    )

    print()

    print(
        f"Validation messages: "
        f"{len(eval_df):,}"
    )

    print(
        f"Validation ham: "
        f"{(y_eval == 0).sum():,}"
    )

    print(
        f"Validation spam: "
        f"{(y_eval == 1).sum():,}"
    )

    print()

    print(
        "V2 features:"
    )

    for feature in FEATURES:
        print(
            f"  {feature}"
        )

    pipeline = (
        build_pipeline()
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    probabilities = (
        pipeline
        .predict_proba(
            X_eval
        )[:, 1]
    )

    (
        metrics,
        predictions,
    ) = calculate_metrics(
        y_true=y_eval,
        probabilities=probabilities,
        threshold=0.5,
    )

    print_metrics(
        metrics
    )

    print()
    print(
        "Classification report"
    )

    print()

    print(
        classification_report(
            y_eval,
            predictions,
            target_names=[
                "ham",
                "spam",
            ],
            digits=4,
            zero_division=0,
        )
    )

    eval_df[
        "security_v2_spam_probability"
    ] = probabilities

    eval_df[
        "security_v2_prediction"
    ] = predictions

    eval_df[
        "security_v2_correct"
    ] = (
        eval_df[
            "security_v2_prediction"
        ]
        == eval_df[
            "analysis_label"
        ]
    )

    coefficients = (
        get_coefficients(
            pipeline
        )
    )

    print()
    print(
        "V2 COEFFICIENTS"
    )

    print()

    print(
        coefficients[
            [
                "feature",
                "coefficient",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print_important_rows(
        eval_df
    )

    predictions_path = (
        RESULTS_DIR
        / "security_classifier_v2_eval_predictions.csv"
    )

    coefficients_path = (
        RESULTS_DIR
        / "security_classifier_v2_coefficients.csv"
    )

    metrics_path = (
        RESULTS_DIR
        / "security_classifier_v2_metrics.json"
    )

    model_path = (
        MODEL_DIR
        / "security_classifier_v2.joblib"
    )

    output_columns = [
        "_private_row",
        "sender",
        "subject",
        "date",
        "label",
        "analysis_label",
        "security_v2_spam_probability",
        "security_v2_prediction",
        "security_v2_correct",
    ] + FEATURES

    eval_df[
        output_columns
    ].to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    coefficients.to_csv(
        coefficients_path,
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    joblib.dump(
        pipeline,
        model_path,
    )

    print()
    print(
        "Saved V2 predictions:"
    )

    print(
        predictions_path
    )

    print()

    print(
        "Saved V2 coefficients:"
    )

    print(
        coefficients_path
    )

    print()

    print(
        "Saved V2 metrics:"
    )

    print(
        metrics_path
    )

    print()

    print(
        "Saved V2 model:"
    )

    print(
        model_path
    )


if __name__ == "__main__":
    main()