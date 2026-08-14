"""
Analyze security-header features on the private validation dataset.

This script combines:

- private validation labels
- manually adjudicated label corrections
- extracted email security features

The goal is to understand which security signals distinguish spam
from ham before using them in a meta-classifier.
"""

import pandas as pd

from spam_detector.paths import DATA_DIR

SECURITY_PATH = (
    DATA_DIR
    / "security_features"
    / "private_eval_security_matched.parquet"
)

TEMPORAL_PATH = (
    DATA_DIR
    / "private_error_analysis"
    / "temporal_sender_reputation_results.csv"
)


BOOLEAN_FEATURES = [
    "security_spf_pass",
    "security_dkim_pass",
    "security_dmarc_pass",
    "security_spf_fail",
    "security_dkim_fail",
    "security_dmarc_fail",
    "security_from_return_path_exact_match",
    "security_from_return_path_org_match",
    "security_from_dkim_exact_match",
    "security_from_dkim_org_match",
    "security_from_reply_to_exact_match",
    "security_from_reply_to_org_match",
    "security_shared_relay_domain",
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


def build_ground_truth(
    security_df,
    temporal_df,
):
    """
    Combine the original label with manually adjudicated corrections.

    corrected_label is used when available. Otherwise the original
    private label is retained.
    """

    corrections = (
        temporal_df[
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
        security_df
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


def summarize_boolean_feature(
    dataframe,
    feature,
):
    """
    Calculate feature prevalence separately for ham and spam.
    """

    rows = []

    for label, name in [
        (0, "ham"),
        (1, "spam"),
    ]:

        group = dataframe[
            dataframe[
                "analysis_label"
            ]
            == label
        ]

        values = (
            group[
                feature
            ]
            .astype("boolean")
        )

        available = int(
            values.notna().sum()
        )

        positive = int(
            values.fillna(False).sum()
        )

        percentage = (
            positive
            / available
            * 100
            if available
            else float("nan")
        )

        rows.append(
            {
                "feature":
                    feature,

                "class":
                    name,

                "messages":
                    len(group),

                "available":
                    available,

                "positive":
                    positive,

                "positive_pct":
                    percentage,
            }
        )

    return rows


def print_boolean_summary(
    dataframe,
):
    """
    Print ham/spam prevalence for boolean security features.
    """

    rows = []

    for feature in BOOLEAN_FEATURES:

        if feature not in dataframe.columns:
            continue

        rows.extend(
            summarize_boolean_feature(
                dataframe,
                feature,
            )
        )

    summary = pd.DataFrame(
        rows
    )

    pivot = (
        summary
        .pivot(
            index="feature",
            columns="class",
            values="positive_pct",
        )
        .reset_index()
    )

    if (
        "ham"
        in pivot.columns
        and "spam"
        in pivot.columns
    ):

        pivot[
            "spam_minus_ham_pct"
        ] = (
            pivot[
                "spam"
            ]
            - pivot[
                "ham"
            ]
        )

    print()
    print(
        "BOOLEAN SECURITY FEATURES"
    )

    print()

    print(
        pivot
        .sort_values(
            "spam_minus_ham_pct",
            ascending=False,
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )


def print_categorical_summary(
    dataframe,
):
    """
    Print SPF/DKIM/DMARC distributions by class.
    """

    for feature in CATEGORICAL_FEATURES:

        if feature not in dataframe.columns:
            continue

        print()
        print(
            "=" * 80
        )

        print(
            feature
        )

        print(
            "=" * 80
        )

        table = pd.crosstab(
            dataframe[
                feature
            ]
            .fillna(
                "<missing>"
            ),
            dataframe[
                "analysis_label"
            ]
            .map(
                {
                    0: "ham",
                    1: "spam",
                }
            ),
            margins=True,
        )

        print(
            table.to_string()
        )


def print_spoofing_combinations(
    dataframe,
):
    """
    Analyze combinations that may represent stronger spoofing signals.
    """

    dataframe = (
        dataframe
        .copy()
    )

    dataframe[
        "signal_dmarc_fail"
    ] = (
        dataframe[
            "security_dmarc_fail"
        ]
        .fillna(False)
    )

    dataframe[
        "signal_dkim_alignment_failure"
    ] = (
        dataframe[
            "security_dkim_fail"
        ]
        .fillna(False)
        &
        dataframe[
            "security_from_dkim_org_match"
        ]
        .eq(False)
        .fillna(False)
    )

    dataframe[
        "signal_return_path_mismatch"
    ] = (
        dataframe[
            "security_from_return_path_org_match"
        ]
        .eq(False)
        .fillna(False)
    )

    dataframe[
        "strong_spoofing_signal"
    ] = (
        dataframe[
            "signal_dmarc_fail"
        ]
        |
        dataframe[
            "signal_dkim_alignment_failure"
        ]
    )

    features = [
        "signal_dmarc_fail",
        "signal_dkim_alignment_failure",
        "signal_return_path_mismatch",
        "strong_spoofing_signal",
    ]

    print()
    print(
        "COMBINED SECURITY SIGNALS"
    )

    print()

    rows = []

    for feature in features:

        for label, name in [
            (0, "ham"),
            (1, "spam"),
        ]:

            group = dataframe[
                dataframe[
                    "analysis_label"
                ]
                == label
            ]

            positive = int(
                group[
                    feature
                ]
                .sum()
            )

            rows.append(
                {
                    "signal":
                        feature,

                    "class":
                        name,

                    "positive":
                        positive,

                    "messages":
                        len(group),

                    "percentage":
                        (
                            positive
                            / len(group)
                            * 100
                        ),
                }
            )

    summary = pd.DataFrame(
        rows
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )


def print_remaining_errors(
    dataframe,
):
    """
    Print the seven previously identified temporal cold-start errors.
    """

    error_ids = [
        73,
        434,
        915,
        1249,
        1295,
        1466,
        1813,
    ]

    errors = dataframe[
        dataframe[
            "_private_row"
        ]
        .isin(
            error_ids
        )
    ].copy()

    columns = [
        "_private_row",
        "sender",
        "subject",
        "analysis_label",
        "security_spf_result",
        "security_dkim_result",
        "security_dmarc_result",
        "security_from_return_path_org_match",
        "security_from_dkim_org_match",
        "security_from_reply_to_org_match",
    ]

    columns = [
        column
        for column in columns
        if column in errors.columns
    ]

    print()
    print(
        "REMAINING TEMPORAL ERRORS"
    )

    print()

    print(
        errors[
            columns
        ]
        .sort_values(
            "_private_row"
        )
        .to_string(
            index=False
        )
    )


def main():
    """
    Run security-feature analysis.
    """

    security_df = (
        pd.read_parquet(
            SECURITY_PATH
        )
    )

    temporal_df = (
        pd.read_csv(
            TEMPORAL_PATH
        )
    )

    dataframe = (
        build_ground_truth(
            security_df=security_df,
            temporal_df=temporal_df,
        )
    )

    print(
        "Private validation security analysis"
    )

    print()

    print(
        f"Messages: "
        f"{len(dataframe):,}"
    )

    print(
        f"Ham:      "
        f"{(dataframe['analysis_label'] == 0).sum():,}"
    )

    print(
        f"Spam:     "
        f"{(dataframe['analysis_label'] == 1).sum():,}"
    )

    print(
        f"Security matched: "
        f"{dataframe['security_matched'].sum():,}"
        f" / "
        f"{len(dataframe):,}"
    )

    print_boolean_summary(
        dataframe
    )

    print_categorical_summary(
        dataframe
    )

    print_spoofing_combinations(
        dataframe
    )

    print_remaining_errors(
        dataframe
    )


if __name__ == "__main__":
    main()