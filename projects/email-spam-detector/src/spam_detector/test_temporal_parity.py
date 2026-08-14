"""
Verify that the live temporal-reputation implementation in
classify_email.py exactly reproduces the corrected temporal evaluation.

The comparison is calculated directly from source logic rather than
using an older stored CSV as the reference.
"""

import pandas as pd

from spam_detector.classify_email import (
    calculate_temporal_reputation,
    prepare_history,
)
from spam_detector.evaluate_temporal_sender_reputation import (
    EVAL_PATH,
    TRAIN_PATH,
    apply_temporal_hybrid,
    extract_domain,
    extract_email_address,
    get_temporal_reputation,
    is_shared_relay_domain,
)

MIN_COUNT = 1
USE_DOMAIN = True


def values_equal(
    first,
    second,
    tolerance=1e-12,
):
    """
    Compare optional numeric values including NaN and None.
    """

    first_missing = (
        first is None
        or
        pd.isna(first)
    )

    second_missing = (
        second is None
        or
        pd.isna(second)
    )

    if (
        first_missing
        and
        second_missing
    ):
        return True

    if (
        first_missing
        or
        second_missing
    ):
        return False

    try:
        return (
            abs(
                float(first)
                - float(second)
            )
            <= tolerance
        )

    except (
        TypeError,
        ValueError,
    ):
        return (
            first
            == second
        )


def prepare_reference_train():
    """
    Prepare training data exactly as the corrected temporal evaluation.
    """

    train = (
        pd.read_parquet(
            TRAIN_PATH
        )
        .copy()
    )

    train[
        "label"
    ] = (
        pd.to_numeric(
            train[
                "label"
            ],
            errors="raise",
        )
        .astype(int)
    )

    train[
        "date_parsed"
    ] = pd.to_datetime(
        train[
            "date"
        ],
        errors="coerce",
        utc=True,
        format="mixed",
    )

    train[
        "sender_email"
    ] = (
        train[
            "sender"
        ]
        .apply(
            extract_email_address
        )
    )

    train[
        "sender_domain"
    ] = (
        train[
            "sender_email"
        ]
        .apply(
            extract_domain
        )
    )

    return train


def prepare_reference_eval():
    """
    Prepare validation data exactly as the corrected temporal
    evaluation.
    """

    df = (
        pd.read_csv(
            EVAL_PATH
        )
        .copy()
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
        "date_parsed"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce",
        utc=True,
        format="mixed",
    )

    df[
        "sender_email"
    ] = (
        df[
            "sender"
        ]
        .apply(
            extract_email_address
        )
    )

    df[
        "sender_domain"
    ] = (
        df[
            "sender_email"
        ]
        .apply(
            extract_domain
        )
    )

    return df


def calculate_reference_result(
    train,
    row,
):
    """
    Calculate one reference result with the evaluation functions.
    """

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
            train=
                train,

            group_column=
                "sender_email",

            group_value=
                row[
                    "sender_email"
                ],

            current_date=
                row[
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
                train=
                    train,

                group_column=
                    "sender_domain",

                group_value=
                    row[
                        "sender_domain"
                    ],

                current_date=
                    row[
                        "date_parsed"
                    ],
            )

    (
        prediction,
        source,
    ) = apply_temporal_hybrid(
        row=
            row,

        sender_count=
            sender_count,

        sender_spam_rate=
            sender_spam_rate,

        domain_count=
            domain_count,

        domain_spam_rate=
            domain_spam_rate,

        min_count=
            MIN_COUNT,

        use_domain=
            USE_DOMAIN,
    )

    return {
        "prediction":
            int(
                prediction
            ),

        "source":
            source,

        "sender_count":
            int(
                sender_count
            ),

        "sender_spam_rate":
            sender_spam_rate,

        "domain_count":
            int(
                domain_count
            ),

        "domain_spam_rate":
            domain_spam_rate,
    }


def main():
    print(
        "Loading live temporal history..."
    )

    live_history = (
        prepare_history()
    )

    print(
        f"Live history rows: "
        f"{len(live_history):,}"
    )

    print()
    print(
        "Loading reference temporal history..."
    )

    reference_train = (
        prepare_reference_train()
    )

    print(
        f"Reference history rows: "
        f"{len(reference_train):,}"
    )

    print()
    print(
        "Loading adjudicated validation data..."
    )

    df = (
        prepare_reference_eval()
    )

    print(
        f"Validation rows: "
        f"{len(df):,}"
    )

    prediction_mismatches = []

    sender_count_mismatches = []

    domain_count_mismatches = []

    sender_rate_mismatches = []

    domain_rate_mismatches = []

    for (
        index,
        row,
    ) in df.iterrows():

        reference = (
            calculate_reference_result(
                train=
                    reference_train,

                row=
                    row,
            )
        )

        live = (
            calculate_temporal_reputation(
                history=
                    live_history,

                sender=
                    row[
                        "sender"
                    ],

                date=
                    row[
                        "date"
                    ],

                gpt_prediction=
                    int(
                        row[
                            "predicted_label"
                        ]
                    ),
            )
        )

        if (
            live[
                "prediction"
            ]
            !=
            reference[
                "prediction"
            ]
        ):

            prediction_mismatches.append(
                {
                    "row":
                        index,

                    "sender":
                        row[
                            "sender"
                        ],

                    "expected":
                        reference[
                            "prediction"
                        ],

                    "actual":
                        live[
                            "prediction"
                        ],

                    "expected_source":
                        reference[
                            "source"
                        ],

                    "actual_source":
                        live[
                            "source"
                        ],
                }
            )

        if (
            live[
                "sender_count"
            ]
            !=
            reference[
                "sender_count"
            ]
        ):

            sender_count_mismatches.append(
                {
                    "row":
                        index,

                    "sender":
                        row[
                            "sender"
                        ],

                    "expected":
                        reference[
                            "sender_count"
                        ],

                    "actual":
                        live[
                            "sender_count"
                        ],
                }
            )

        if (
            live[
                "domain_count"
            ]
            !=
            reference[
                "domain_count"
            ]
        ):

            domain_count_mismatches.append(
                {
                    "row":
                        index,

                    "sender":
                        row[
                            "sender"
                        ],

                    "expected":
                        reference[
                            "domain_count"
                        ],

                    "actual":
                        live[
                            "domain_count"
                        ],
                }
            )

        if not values_equal(
            live[
                "sender_spam_rate"
            ],

            reference[
                "sender_spam_rate"
            ],
        ):

            sender_rate_mismatches.append(
                {
                    "row":
                        index,

                    "sender":
                        row[
                            "sender"
                        ],

                    "expected":
                        reference[
                            "sender_spam_rate"
                        ],

                    "actual":
                        live[
                            "sender_spam_rate"
                        ],
                }
            )

        if not values_equal(
            live[
                "domain_spam_rate"
            ],

            reference[
                "domain_spam_rate"
            ],
        ):

            domain_rate_mismatches.append(
                {
                    "row":
                        index,

                    "sender":
                        row[
                            "sender"
                        ],

                    "expected":
                        reference[
                            "domain_spam_rate"
                        ],

                    "actual":
                        live[
                            "domain_spam_rate"
                        ],
                }
            )

    print()
    print(
        "TEMPORAL SOURCE-TO-LIVE PARITY RESULTS"
    )

    print(
        f"Messages tested:            "
        f"{len(df):,}"
    )

    print(
        f"Prediction mismatches:      "
        f"{len(prediction_mismatches):,}"
    )

    print(
        f"Sender count mismatches:    "
        f"{len(sender_count_mismatches):,}"
    )

    print(
        f"Domain count mismatches:    "
        f"{len(domain_count_mismatches):,}"
    )

    print(
        f"Sender rate mismatches:     "
        f"{len(sender_rate_mismatches):,}"
    )

    print(
        f"Domain rate mismatches:     "
        f"{len(domain_rate_mismatches):,}"
    )

    all_match = (
        len(
            prediction_mismatches
        ) == 0
        and
        len(
            sender_count_mismatches
        ) == 0
        and
        len(
            domain_count_mismatches
        ) == 0
        and
        len(
            sender_rate_mismatches
        ) == 0
        and
        len(
            domain_rate_mismatches
        ) == 0
    )

    print()

    if all_match:

        print(
            "PASS: Live temporal logic exactly matches "
            "the corrected temporal evaluation."
        )

    else:

        print(
            "FAIL: Live temporal logic still differs "
            "from the corrected temporal evaluation."
        )

    mismatch_groups = [
        (
            "Prediction mismatches",
            prediction_mismatches,
        ),

        (
            "Sender-count mismatches",
            sender_count_mismatches,
        ),

        (
            "Domain-count mismatches",
            domain_count_mismatches,
        ),

        (
            "Sender-rate mismatches",
            sender_rate_mismatches,
        ),

        (
            "Domain-rate mismatches",
            domain_rate_mismatches,
        ),
    ]

    for (
        title,
        mismatches,
    ) in mismatch_groups:

        if not mismatches:
            continue

        print()

        print(
            f"First "
            f"{title.lower()}:"
        )

        print(
            pd.DataFrame(
                mismatches[
                    :20
                ]
            )
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()