"""
Evaluate temporally valid sender and domain reputation on the
adjudicated private validation dataset.

Only emails from private_train that occurred strictly before the
current validation email may contribute reputation.

Important:
- Exact normalized sender address is used for sender reputation.
- Exact sender domain is used for domain reputation.
- Shared relay infrastructure such as Apple Private Relay is excluded.
- Reputation overrides GPT only when historical labels are completely
  pure:
      spam_rate == 0.0 -> HAM
      spam_rate == 1.0 -> SPAM
- Mixed reputation falls back to GPT.
- Mixed ISO date formats are parsed explicitly with format="mixed".
"""

from email.utils import parseaddr
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "private_train.parquet"
)

EVAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "private_error_analysis"
    / "all_predictions_adjudicated.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "private_error_analysis"
    / "temporal_sender_reputation_results.csv"
)


SHARED_RELAY_DOMAINS = {
    "privaterelay.appleid.com",
}


MIN_COUNTS = [
    1,
    2,
    3,
    5,
    10,
]


def extract_email_address(sender):
    """
    Extract and normalize the sender email address.
    """

    if pd.isna(sender):
        return ""

    _, email_address = parseaddr(
        str(sender)
    )

    return (
        email_address
        .strip()
        .lower()
    )


def extract_domain(email_address):
    """
    Extract the sender domain.
    """

    if not email_address:
        return ""

    if "@" not in email_address:
        return ""

    return (
        email_address
        .rsplit("@", 1)[1]
        .strip()
        .lower()
        .rstrip(".")
    )


def is_shared_relay_domain(domain):
    """
    Check whether a domain belongs to shared relay infrastructure.
    """

    if not domain:
        return False

    domain = (
        str(domain)
        .strip()
        .lower()
        .rstrip(".")
    )

    for relay_domain in SHARED_RELAY_DOMAINS:

        if domain == relay_domain:
            return True

        if domain.endswith(
            "." + relay_domain
        ):
            return True

    return False


def parse_dates(values):
    """
    Parse mixed email date formats and normalize them to UTC.

    Invalid or genuinely missing dates are returned as NaT.
    """

    return pd.to_datetime(
        values,
        errors="coerce",
        utc=True,
        format="mixed",
    )


def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Calculate binary classification metrics.
    """

    (
        tn,
        fp,
        fn,
        tp,
    ) = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
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


def get_temporal_reputation(
    train,
    group_column,
    group_value,
    current_date,
):
    """
    Calculate reputation using only training emails that occurred
    strictly before the current validation email.
    """

    if not group_value:
        return (
            0,
            None,
        )

    historical = train[
        (
            train[group_column]
            == group_value
        )
        &
        (
            train["date_parsed"]
            < current_date
        )
    ]

    count = len(
        historical
    )

    if count == 0:
        return (
            0,
            None,
        )

    spam_count = int(
        historical[
            "label"
        ]
        .sum()
    )

    spam_rate = (
        spam_count
        / count
    )

    return (
        count,
        spam_rate,
    )


def apply_temporal_hybrid(
    row,
    sender_count,
    sender_spam_rate,
    domain_count,
    domain_spam_rate,
    min_count,
    use_domain,
):
    """
    Apply temporal reputation to one validation email.

    Sender reputation has priority over domain reputation.

    Only completely pure historical reputation changes GPT.
    """

    original_prediction = int(
        row[
            "predicted_label"
        ]
    )

    domain = row[
        "sender_domain"
    ]

    if is_shared_relay_domain(
        domain
    ):
        return (
            original_prediction,
            "gpt_shared_relay",
        )

    if (
        sender_count
        >= min_count
    ):

        if sender_spam_rate == 0.0:
            return (
                0,
                "sender_ham",
            )

        if sender_spam_rate == 1.0:
            return (
                1,
                "sender_spam",
            )

    if use_domain:

        if (
            domain_count
            >= min_count
        ):

            if domain_spam_rate == 0.0:
                return (
                    0,
                    "domain_ham",
                )

            if domain_spam_rate == 1.0:
                return (
                    1,
                    "domain_spam",
                )

    return (
        original_prediction,
        "gpt",
    )


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
        "-" * len(name)
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
        f"TN={metrics['tn']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"TP={metrics['tp']}"
    )


def get_prediction_column_name(
    min_count,
    use_domain,
):
    """
    Build the prediction-column name.
    """

    if use_domain:
        suffix = "sender_domain"
    else:
        suffix = "sender_only"

    return (
        f"temporal_hybrid_min_"
        f"{min_count}_"
        f"{suffix}"
    )


def get_source_column_name(
    min_count,
    use_domain,
):
    """
    Build the decision-source column name.
    """

    if use_domain:
        suffix = "sender_domain"
    else:
        suffix = "sender_only"

    return (
        f"temporal_source_min_"
        f"{min_count}_"
        f"{suffix}"
    )


def main():
    print(
        "Loading private training data..."
    )

    train = pd.read_parquet(
        TRAIN_PATH
    ).copy()

    train["label"] = (
        pd.to_numeric(
            train[
                "label"
            ],
            errors="raise",
        )
        .astype(int)
    )

    train["date_parsed"] = (
        parse_dates(
            train[
                "date"
            ]
        )
    )

    train["sender_email"] = (
        train[
            "sender"
        ]
        .apply(
            extract_email_address
        )
    )

    train["sender_domain"] = (
        train[
            "sender_email"
        ]
        .apply(
            extract_domain
        )
    )

    missing_train_dates = int(
        train[
            "date_parsed"
        ]
        .isna()
        .sum()
    )

    print(
        f"Training examples: "
        f"{len(train):,}"
    )

    print(
        f"Training examples without usable date: "
        f"{missing_train_dates:,}"
    )

    print()
    print(
        "Loading adjudicated private validation..."
    )

    df = pd.read_csv(
        EVAL_PATH
    ).copy()

    required_columns = {
        "sender",
        "date",
        "predicted_label",
        "corrected_label",
    }

    missing_columns = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Validation dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    df[
        "predicted_label"
    ] = (
        pd.to_numeric(
            df[
                "predicted_label"
            ],
            errors="raise",
        )
        .astype(int)
    )

    df[
        "corrected_label"
    ] = (
        pd.to_numeric(
            df[
                "corrected_label"
            ],
            errors="raise",
        )
        .astype(int)
    )

    df["date_parsed"] = (
        parse_dates(
            df[
                "date"
            ]
        )
    )

    df["sender_email"] = (
        df[
            "sender"
        ]
        .apply(
            extract_email_address
        )
    )

    df["sender_domain"] = (
        df[
            "sender_email"
        ]
        .apply(
            extract_domain
        )
    )

    missing_eval_dates = int(
        df[
            "date_parsed"
        ]
        .isna()
        .sum()
    )

    print(
        f"Validation examples: "
        f"{len(df):,}"
    )

    print(
        f"Validation examples without usable date: "
        f"{missing_eval_dates:,}"
    )

    if (
        missing_eval_dates
        > 0
    ):
        print()

        print(
            "Warning: validation emails without dates "
            "cannot use temporal reputation."
        )

    sender_counts = []
    sender_spam_rates = []

    domain_counts = []
    domain_spam_rates = []

    print()
    print(
        "Calculating temporally valid reputation..."
    )

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        if pd.isna(
            row[
                "date_parsed"
            ]
        ):

            sender_count = 0
            sender_spam_rate = None

            domain_count = 0
            domain_spam_rate = None

        else:

            (
                sender_count,
                sender_spam_rate,
            ) = get_temporal_reputation(
                train=train,
                group_column="sender_email",
                group_value=row[
                    "sender_email"
                ],
                current_date=row[
                    "date_parsed"
                ],
            )

            if is_shared_relay_domain(
                row[
                    "sender_domain"
                ]
            ):

                domain_count = 0
                domain_spam_rate = None

            else:

                (
                    domain_count,
                    domain_spam_rate,
                ) = get_temporal_reputation(
                    train=train,
                    group_column="sender_domain",
                    group_value=row[
                        "sender_domain"
                    ],
                    current_date=row[
                        "date_parsed"
                    ],
                )

        sender_counts.append(
            sender_count
        )

        sender_spam_rates.append(
            sender_spam_rate
        )

        domain_counts.append(
            domain_count
        )

        domain_spam_rates.append(
            domain_spam_rate
        )

        if (
            position
            % 250
            == 0
        ):
            print(
                f"Processed "
                f"{position:,}/"
                f"{len(df):,}"
            )

    df[
        "temporal_sender_count"
    ] = sender_counts

    df[
        "temporal_sender_spam_rate"
    ] = sender_spam_rates

    df[
        "temporal_domain_count"
    ] = domain_counts

    df[
        "temporal_domain_spam_rate"
    ] = domain_spam_rates

    y_true = (
        df[
            "corrected_label"
        ]
        .astype(int)
    )

    gpt_predictions = (
        df[
            "predicted_label"
        ]
        .astype(int)
    )

    gpt_metrics = (
        calculate_metrics(
            y_true=y_true,
            y_pred=gpt_predictions,
        )
    )

    print_metrics(
        name="GPT BASELINE",
        metrics=gpt_metrics,
    )

    results = []

    for min_count in MIN_COUNTS:

        for use_domain in [
            False,
            True,
        ]:

            predictions = []
            sources = []

            for _, row in df.iterrows():

                (
                    prediction,
                    source,
                ) = apply_temporal_hybrid(
                    row=row,

                    sender_count=int(
                        row[
                            "temporal_sender_count"
                        ]
                    ),

                    sender_spam_rate=row[
                        "temporal_sender_spam_rate"
                    ],

                    domain_count=int(
                        row[
                            "temporal_domain_count"
                        ]
                    ),

                    domain_spam_rate=row[
                        "temporal_domain_spam_rate"
                    ],

                    min_count=
                        min_count,

                    use_domain=
                        use_domain,
                )

                predictions.append(
                    prediction
                )

                sources.append(
                    source
                )

            prediction_column = (
                get_prediction_column_name(
                    min_count=
                        min_count,

                    use_domain=
                        use_domain,
                )
            )

            source_column = (
                get_source_column_name(
                    min_count=
                        min_count,

                    use_domain=
                        use_domain,
                )
            )

            df[
                prediction_column
            ] = predictions

            df[
                source_column
            ] = sources

            metrics = (
                calculate_metrics(
                    y_true=
                        y_true,

                    y_pred=
                        predictions,
                )
            )

            prediction_series = pd.Series(
                predictions,
                index=df.index,
            )

            overrides = int(
                (
                    prediction_series
                    != gpt_predictions
                )
                .sum()
            )

            gpt_wrong = (
                gpt_predictions
                != y_true
            )

            gpt_correct = (
                gpt_predictions
                == y_true
            )

            hybrid_correct = (
                prediction_series
                == y_true
            )

            hybrid_wrong = (
                prediction_series
                != y_true
            )

            corrected_gpt_errors = int(
                (
                    gpt_wrong
                    & hybrid_correct
                )
                .sum()
            )

            new_errors_created = int(
                (
                    gpt_correct
                    & hybrid_wrong
                )
                .sum()
            )

            if use_domain:
                configuration_name = (
                    f"TEMPORAL HYBRID | "
                    f"min_count={min_count} | "
                    f"sender + domain"
                )

            else:
                configuration_name = (
                    f"TEMPORAL HYBRID | "
                    f"min_count={min_count} | "
                    f"sender only"
                )

            print_metrics(
                name=
                    configuration_name,

                metrics=
                    metrics,
            )

            print(
                f"Overrides: "
                f"{overrides}"
            )

            print(
                f"GPT errors corrected: "
                f"{corrected_gpt_errors}"
            )

            print(
                f"New errors created: "
                f"{new_errors_created}"
            )

            results.append(
                {
                    "min_count":
                        min_count,

                    "use_domain":
                        use_domain,

                    "accuracy":
                        metrics[
                            "accuracy"
                        ],

                    "precision":
                        metrics[
                            "precision"
                        ],

                    "recall":
                        metrics[
                            "recall"
                        ],

                    "f1":
                        metrics[
                            "f1"
                        ],

                    "tn":
                        metrics[
                            "tn"
                        ],

                    "fp":
                        metrics[
                            "fp"
                        ],

                    "fn":
                        metrics[
                            "fn"
                        ],

                    "tp":
                        metrics[
                            "tp"
                        ],

                    "overrides":
                        overrides,

                    "corrected_gpt_errors":
                        corrected_gpt_errors,

                    "new_errors_created":
                        new_errors_created,

                    "prediction_column":
                        prediction_column,

                    "source_column":
                        source_column,
                }
            )

    results_df = pd.DataFrame(
        results
    )

    ranked_results = (
        results_df
        .sort_values(
            by=[
                "f1",
                "accuracy",
                "precision",
                "recall",
                "min_count",
                "use_domain",
            ],
            ascending=[
                False,
                False,
                False,
                False,
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
        "TEMPORAL HYBRID COMPARISON"
    )
    print()

    comparison_columns = [
        "min_count",
        "use_domain",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "overrides",
        "corrected_gpt_errors",
        "new_errors_created",
    ]

    print(
        ranked_results[
            comparison_columns
        ]
        .to_string(
            index=False
        )
    )

    best = (
        ranked_results
        .iloc[0]
    )

    print()
    print(
        "BEST TEMPORAL CONFIGURATION"
    )
    print()

    print(
        f"min_count: "
        f"{int(best['min_count'])}"
    )

    print(
        f"use_domain: "
        f"{bool(best['use_domain'])}"
    )

    print(
        f"Accuracy: "
        f"{best['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{best['precision'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{best['recall'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{best['f1'] * 100:.2f}%"
    )

    print(
        f"GPT errors corrected: "
        f"{int(best['corrected_gpt_errors'])}"
    )

    print(
        f"New errors created: "
        f"{int(best['new_errors_created'])}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Detailed temporal results saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()