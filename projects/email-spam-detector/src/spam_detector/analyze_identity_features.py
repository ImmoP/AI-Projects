"""
Analyze display-name / sender-domain identity consistency.

The goal is to determine whether the visible sender identity is
consistent with the organizational From domain.

This is intended as a feature for a later meta-classifier, not as a
standalone spam rule.
"""

import re
from difflib import SequenceMatcher

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


def normalize_identity(value):
    """
    Normalize display names and domain labels to comparable strings.
    """

    if value is None or pd.isna(value):
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

    Example:

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


def identity_similarity(
    display_name,
    org_domain,
):
    """
    Compute string similarity between display name and domain label.
    """

    display = (
        normalize_identity(
            display_name
        )
    )

    domain = (
        normalize_identity(
            get_domain_label(
                org_domain
            )
        )
    )

    if not display or not domain:
        return None

    return SequenceMatcher(
        None,
        display,
        domain,
    ).ratio()


def domain_name_in_display(
    display_name,
    org_domain,
):
    """
    Check whether the organizational domain label appears directly
    in the normalized display name.
    """

    display = (
        normalize_identity(
            display_name
        )
    )

    domain = (
        normalize_identity(
            get_domain_label(
                org_domain
            )
        )
    )

    if not display or not domain:
        return None

    if len(domain) < 3:
        return None

    return (
        domain
        in display
    )


def build_ground_truth(
    dataframe,
    temporal,
):
    """
    Apply manually adjudicated validation labels when available.
    """

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


def print_class_summary(
    dataframe,
):
    """
    Print similarity statistics separately for ham and spam.
    """

    print()
    print(
        "IDENTITY SIMILARITY BY CLASS"
    )

    for label, name in [
        (0, "HAM"),
        (1, "SPAM"),
    ]:

        values = (
            dataframe.loc[
                dataframe[
                    "analysis_label"
                ]
                == label,
                "identity_similarity",
            ]
            .dropna()
        )

        print()
        print(
            name
        )

        print(
            f"Messages: "
            f"{len(values):,}"
        )

        print(
            f"Mean:     "
            f"{values.mean():.3f}"
        )

        print(
            f"Median:   "
            f"{values.median():.3f}"
        )

        print(
            f"10%:      "
            f"{values.quantile(0.10):.3f}"
        )

        print(
            f"25%:      "
            f"{values.quantile(0.25):.3f}"
        )

        print(
            f"75%:      "
            f"{values.quantile(0.75):.3f}"
        )

        print(
            f"90%:      "
            f"{values.quantile(0.90):.3f}"
        )


def print_threshold_table(
    dataframe,
):
    """
    Show how different similarity thresholds behave.

    This is diagnostic only and is not used as a final decision rule.
    """

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
    ]

    rows = []

    available = dataframe[
        dataframe[
            "identity_similarity"
        ]
        .notna()
    ].copy()

    for threshold in thresholds:

        suspicious = (
            available[
                "identity_similarity"
            ]
            < threshold
        )

        true_spam = (
            available[
                "analysis_label"
            ]
            == 1
        )

        true_ham = (
            available[
                "analysis_label"
            ]
            == 0
        )

        tp = int(
            (
                suspicious
                & true_spam
            )
            .sum()
        )

        fp = int(
            (
                suspicious
                & true_ham
            )
            .sum()
        )

        fn = int(
            (
                ~suspicious
                & true_spam
            )
            .sum()
        )

        tn = int(
            (
                ~suspicious
                & true_ham
            )
            .sum()
        )

        precision = (
            tp
            / (tp + fp)
            if tp + fp
            else 0
        )

        recall = (
            tp
            / (tp + fn)
            if tp + fn
            else 0
        )

        rows.append(
            {
                "threshold":
                    threshold,

                "TP":
                    tp,

                "FP":
                    fp,

                "FN":
                    fn,

                "TN":
                    tn,

                "precision":
                    precision,

                "recall":
                    recall,
            }
        )

    result = pd.DataFrame(
        rows
    )

    print()
    print(
        "SIMILARITY THRESHOLD DIAGNOSTICS"
    )

    print()

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def print_domain_in_display_summary(
    dataframe,
):
    """
    Compare direct domain-name presence between ham and spam.
    """

    print()
    print(
        "DOMAIN NAME IN DISPLAY"
    )

    for label, name in [
        (0, "HAM"),
        (1, "SPAM"),
    ]:

        values = (
            dataframe.loc[
                dataframe[
                    "analysis_label"
                ]
                == label,
                "domain_name_in_display",
            ]
            .dropna()
        )

        positive = int(
            values
            .eq(True)
            .sum()
        )

        print()

        print(
            f"{name}: "
            f"{positive:,} / "
            f"{len(values):,} "
            f"({positive / len(values) * 100:.2f}%)"
            if len(values)
            else f"{name}: no data"
        )


def print_low_similarity_examples(
    dataframe,
    label,
    name,
):
    """
    Print the lowest-similarity examples for manual inspection.
    """

    subset = (
        dataframe[
            dataframe[
                "analysis_label"
            ]
            == label
        ]
        .dropna(
            subset=[
                "identity_similarity"
            ]
        )
        .sort_values(
            "identity_similarity"
        )
        .head(25)
    )

    columns = [
        "_private_row",
        "security_display_name",
        "security_from_org_domain",
        "identity_similarity",
        "domain_name_in_display",
        "subject",
    ]

    print()
    print(
        f"LOWEST SIMILARITY {name} EXAMPLES"
    )

    print()

    print(
        subset[
            columns
        ]
        .to_string(
            index=False
        )
    )


def main():
    """
    Run identity-consistency analysis.
    """

    dataframe = (
        pd.read_parquet(
            SECURITY_PATH
        )
    )

    temporal = (
        pd.read_csv(
            TEMPORAL_PATH
        )
    )

    dataframe = (
        build_ground_truth(
            dataframe=dataframe,
            temporal=temporal,
        )
    )

    dataframe[
        "identity_similarity"
    ] = dataframe.apply(
        lambda row:
            identity_similarity(
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
            domain_name_in_display(
                row.get(
                    "security_display_name"
                ),
                row.get(
                    "security_from_org_domain"
                ),
            ),
        axis=1,
    )

    print(
        "Private validation identity analysis"
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
        f"Similarity available: "
        f"{dataframe['identity_similarity'].notna().sum():,}"
    )

    print_class_summary(
        dataframe
    )

    print_domain_in_display_summary(
        dataframe
    )

    print_threshold_table(
        dataframe
    )

    print_low_similarity_examples(
        dataframe=dataframe,
        label=0,
        name="HAM",
    )

    print_low_similarity_examples(
        dataframe=dataframe,
        label=1,
        name="SPAM",
    )


if __name__ == "__main__":
    main()