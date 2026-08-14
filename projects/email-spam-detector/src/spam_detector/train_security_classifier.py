"""
Train a small security/identity spam classifier.

The classifier is trained only on the private training split and
evaluated on the private validation split.

It deliberately excludes:

- source
- source_split
- mailbox
- security_match_method
- sender/domain identities
- existing provider spam-filter decisions

This avoids obvious target leakage and prevents the classifier from
simply memorizing individual senders or folders.

The model uses:

- display-name / domain identity consistency
- SPF / DKIM / DMARC results
- organizational-domain alignment
- availability of relevant authentication headers

The resulting probability is intended to become one component of a
later meta-classifier together with:

- GPT-2 content probability
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
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

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
    / "security_classifier"
)


NUMERIC_FEATURES = [
    "identity_similarity",
    "domain_name_in_display",
    "identity_available",
]


BOOLEAN_FEATURES = [
    "security_from_return_path_org_match",
    "security_from_dkim_org_match",
    "security_from_reply_to_org_match",
    "security_has_authentication_results",
    "security_has_dkim_signature",
    "security_has_return_path",
    "security_has_reply_to",
]


CATEGORICAL_FEATURES = [
    "security_spf_result",
    "security_dkim_result",
    "security_dmarc_result",
]


KNOWN_TEMPORAL_ERROR_IDS = [
    73,
    434,
    915,
    1249,
    1295,
    1466,
    1813,
]


def value_is_missing(value):
    """
    Check whether a value is missing.
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
    Normalize a display name or domain label.
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


def get_domain_label(domain):
    """
    Extract the brand-like label from an organizational domain.

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


def calculate_identity_similarity(
    display_name,
    org_domain,
):
    """
    Calculate string similarity between visible sender identity and
    organizational sender domain.
    """

    display = normalize_identity(
        display_name
    )

    domain = normalize_identity(
        get_domain_label(
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


def calculate_domain_name_in_display(
    display_name,
    org_domain,
):
    """
    Check whether the organizational-domain label occurs directly in
    the visible sender name.
    """

    display = normalize_identity(
        display_name
    )

    domain = normalize_identity(
        get_domain_label(
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


def convert_boolean_feature(value):
    """
    Convert True/False-like security values to 1/0.

    Missing values remain NaN.
    """

    if value_is_missing(value):
        return np.nan

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

    if normalized in {
        "false",
        "0",
        "no",
    }:
        return 0.0

    return np.nan


def normalize_auth_result(value):
    """
    Normalize SPF/DKIM/DMARC result strings.
    """

    if value_is_missing(value):
        return "missing"

    value = (
        str(value)
        .strip()
        .casefold()
    )

    if not value:
        return "missing"

    return value


def add_identity_features(
    dataframe,
):
    """
    Add display-name/domain identity features.
    """

    dataframe = (
        dataframe
        .copy()
    )

    dataframe[
        "identity_similarity"
    ] = dataframe.apply(
        lambda row:
            calculate_identity_similarity(
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
        "domain_name_in_display"
    ] = dataframe.apply(
        lambda row:
            calculate_domain_name_in_display(
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
        "identity_available"
    ] = (
        dataframe[
            "identity_similarity"
        ]
        .notna()
        .astype(float)
    )

    return dataframe


def prepare_features(
    dataframe,
):
    """
    Prepare all model features.
    """

    dataframe = (
        add_identity_features(
            dataframe
        )
    )

    for feature in BOOLEAN_FEATURES:

        dataframe[
            feature
        ] = (
            dataframe[
                feature
            ]
            .map(
                convert_boolean_feature
            )
        )

    for feature in CATEGORICAL_FEATURES:

        dataframe[
            feature
        ] = (
            dataframe[
                feature
            ]
            .map(
                normalize_auth_result
            )
        )

    return dataframe


def build_eval_ground_truth(
    eval_df,
):
    """
    Use manually adjudicated disagreement labels for validation.

    corrected_label is used when available. Otherwise the original
    validation label is retained.
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

    eval_df = (
        eval_df
        .merge(
            corrections,
            on="_private_row",
            how="left",
            validate="one_to_one",
        )
    )

    eval_df[
        "analysis_label"
    ] = (
        eval_df[
            "corrected_label"
        ]
        .where(
            eval_df[
                "corrected_label"
            ]
            .notna(),
            eval_df[
                "label"
            ],
        )
        .astype(int)
    )

    return eval_df


def build_pipeline():
    """
    Build a regularized logistic-regression classifier.
    """

    numeric_pipeline = Pipeline(
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

    boolean_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "boolean",
                boolean_pipeline,
                BOOLEAN_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        random_state=123,
    )

    pipeline = Pipeline(
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

    return pipeline


def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5,
):
    """
    Calculate binary-classification metrics.
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
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "tp":
            int(
                tp
            ),
    }

    return (
        metrics,
        predictions,
    )


def get_feature_coefficients(
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

    dataframe = pd.DataFrame(
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
        dataframe
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
        "SECURITY CLASSIFIER VALIDATION"
    )

    print()

    print(
        f"Threshold: "
        f"{metrics['threshold']:.2f}"
    )

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
        "Confusion matrix"
    )

    print(
        f"TN: {metrics['tn']:,}"
    )

    print(
        f"FP: {metrics['fp']:,}"
    )

    print(
        f"FN: {metrics['fn']:,}"
    )

    print(
        f"TP: {metrics['tp']:,}"
    )


def print_known_temporal_errors(
    predictions_df,
):
    """
    Show security-classifier output for the seven remaining temporal
    reputation errors.
    """

    errors = (
        predictions_df[
            predictions_df[
                "_private_row"
            ]
            .isin(
                KNOWN_TEMPORAL_ERROR_IDS
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
        "security_spam_probability",
        "security_prediction",
        "identity_similarity",
        "domain_name_in_display",
        "security_spf_result",
        "security_dkim_result",
        "security_dmarc_result",
        "security_from_return_path_org_match",
        "security_from_dkim_org_match",
    ]

    columns = [
        column
        for column in columns
        if column in errors.columns
    ]

    print()
    print(
        "SEVEN REMAINING TEMPORAL ERRORS"
    )

    print()

    print(
        errors[
            columns
        ]
        .to_string(
            index=False,
        )
    )


def main():
    """
    Train and evaluate the security classifier.
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
        prepare_features(
            train_df
        )
    )

    eval_df = (
        prepare_features(
            eval_df
        )
    )

    eval_df = (
        build_eval_ground_truth(
            eval_df
        )
    )

    feature_columns = (
        NUMERIC_FEATURES
        + BOOLEAN_FEATURES
        + CATEGORICAL_FEATURES
    )

    X_train = (
        train_df[
            feature_columns
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
            feature_columns
        ]
    )

    y_eval = (
        eval_df[
            "analysis_label"
        ]
        .astype(int)
    )

    print(
        "Training security/identity classifier"
    )

    print()

    print(
        f"Training messages: "
        f"{len(train_df):,}"
    )

    print(
        f"Training ham:      "
        f"{(y_train == 0).sum():,}"
    )

    print(
        f"Training spam:     "
        f"{(y_train == 1).sum():,}"
    )

    print()

    print(
        f"Validation messages: "
        f"{len(eval_df):,}"
    )

    print(
        f"Validation ham:      "
        f"{(y_eval == 0).sum():,}"
    )

    print(
        f"Validation spam:     "
        f"{(y_eval == 1).sum():,}"
    )

    print()

    print(
        "Model features:"
    )

    for feature in feature_columns:

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

    eval_output = (
        eval_df
        .copy()
    )

    eval_output[
        "security_spam_probability"
    ] = probabilities

    eval_output[
        "security_prediction"
    ] = predictions

    eval_output[
        "security_correct"
    ] = (
        eval_output[
            "security_prediction"
        ]
        == eval_output[
            "analysis_label"
        ]
    )

    prediction_path = (
        RESULTS_DIR
        / "security_classifier_eval_predictions.csv"
    )

    output_columns = [
        "_private_row",
        "sender",
        "subject",
        "date",
        "label",
        "analysis_label",
        "security_spam_probability",
        "security_prediction",
        "security_correct",
        "identity_similarity",
        "domain_name_in_display",
    ]

    output_columns += [
        feature
        for feature in BOOLEAN_FEATURES
        if feature
        not in output_columns
    ]

    output_columns += [
        feature
        for feature in CATEGORICAL_FEATURES
        if feature
        not in output_columns
    ]

    output_columns = [
        column
        for column in output_columns
        if column in eval_output.columns
    ]

    eval_output[
        output_columns
    ].to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    coefficients = (
        get_feature_coefficients(
            pipeline
        )
    )

    coefficient_path = (
        RESULTS_DIR
        / "security_classifier_coefficients.csv"
    )

    coefficients.to_csv(
        coefficient_path,
        index=False,
        encoding="utf-8-sig",
    )

    metrics_path = (
        RESULTS_DIR
        / "security_classifier_metrics.json"
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

    model_path = (
        MODEL_DIR
        / "security_classifier.joblib"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    print()
    print(
        "TOP SECURITY MODEL COEFFICIENTS"
    )

    print()

    print(
        coefficients[
            [
                "feature",
                "coefficient",
            ]
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print_known_temporal_errors(
        eval_output
    )

    print()
    print(
        "Saved predictions:"
    )

    print(
        prediction_path
    )

    print()

    print(
        "Saved coefficients:"
    )

    print(
        coefficient_path
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
        "Saved model:"
    )

    print(
        model_path
    )


if __name__ == "__main__":
    main()