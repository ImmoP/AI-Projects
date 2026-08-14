"""
Explain the logistic-regression security prediction for one email.

The script decomposes the logit into individual transformed feature
contributions:

    contribution = transformed feature value × model coefficient

Positive contributions push the prediction toward spam.
Negative contributions push the prediction toward ham.
"""

import joblib
import numpy as np
import pandas as pd

from spam_detector.paths import DATA_DIR
from spam_detector.train_security_classifier import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_features,
)

ROW_ID = 1411


MODEL_PATH = (
    DATA_DIR.parent
    / "models"
    / "security_classifier"
    / "security_classifier.joblib"
)

EVAL_PATH = (
    DATA_DIR
    / "security_features"
    / "private_eval_security_matched.parquet"
)


def main():
    pipeline = joblib.load(
        MODEL_PATH
    )

    dataframe = pd.read_parquet(
        EVAL_PATH
    )

    dataframe = prepare_features(
        dataframe
    )

    row = dataframe[
        dataframe["_private_row"] == ROW_ID
    ].copy()

    if len(row) != 1:
        raise ValueError(
            f"Expected one row for {ROW_ID}, "
            f"found {len(row)}."
        )

    feature_columns = (
        NUMERIC_FEATURES
        + BOOLEAN_FEATURES
        + CATEGORICAL_FEATURES
    )

    X = row[
        feature_columns
    ]

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    classifier = pipeline.named_steps[
        "classifier"
    ]

    transformed = preprocessor.transform(
        X
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    coefficients = (
        classifier.coef_[0]
    )

    values = (
        np.asarray(
            transformed
        )
        .reshape(-1)
    )

    contributions = (
        values
        * coefficients
    )

    explanation = pd.DataFrame(
        {
            "feature":
                feature_names,

            "transformed_value":
                values,

            "coefficient":
                coefficients,

            "contribution":
                contributions,
        }
    )

    explanation[
        "absolute_contribution"
    ] = np.abs(
        explanation[
            "contribution"
        ]
    )

    explanation = (
        explanation
        .sort_values(
            "absolute_contribution",
            ascending=False,
        )
    )

    probability = (
        pipeline
        .predict_proba(
            X
        )[0, 1]
    )

    intercept = float(
        classifier.intercept_[0]
    )

    logit = (
        intercept
        + contributions.sum()
    )

    reconstructed_probability = (
        1
        / (
            1
            + np.exp(
                -logit
            )
        )
    )

    print(
        f"Security prediction explanation for row {ROW_ID}"
    )

    print()

    print(
        "Sender:"
    )

    print(
        row.iloc[0]["sender"]
    )

    print()

    print(
        "Subject:"
    )

    print(
        row.iloc[0]["subject"]
    )

    print()

    print(
        f"Model P(spam): "
        f"{probability:.6f}"
    )

    print(
        f"Reconstructed P(spam): "
        f"{reconstructed_probability:.6f}"
    )

    print(
        f"Intercept: "
        f"{intercept:.6f}"
    )

    print()

    print(
        "Largest feature contributions"
    )

    print()

    print(
        explanation[
            [
                "feature",
                "transformed_value",
                "coefficient",
                "contribution",
            ]
        ]
        .head(25)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()

    print(
        "Positive contribution = pushes toward spam"
    )

    print(
        "Negative contribution = pushes toward ham"
    )


if __name__ == "__main__":
    main()