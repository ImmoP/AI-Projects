"""
Match extracted security features to the private ML dataset.

The raw Thunderbird mailboxes may have changed since the private
train/eval/test datasets were created. Messages are therefore matched
using stable metadata rather than MBOX position or message_index.

Matching strategy:

1. Exact metadata match:
   source + UTC date + sender + subject

2. Date/subject fallback:
   source + UTC date + subject

   This is used only when the key is unique in the private dataset.
   It helps recover messages whose sender field was malformed during
   the original dataset preparation.

3. Sender/subject fallback:
   source + sender + subject

   This is also used only when the key is unique in the private
   dataset.

4. Equivalent duplicates:
   If several raw MBOX messages match a key, they are accepted only
   when all security-relevant features are identical.

Ambiguous messages with genuinely different security features remain
unmatched.

The output always contains exactly one row per private email.
"""

import re
import unicodedata
from collections import defaultdict
from email.utils import parseaddr

import pandas as pd

from spam_detector.paths import DATA_DIR

SECURITY_PATH = (
    DATA_DIR
    / "security_features"
    / "private_security_features.parquet"
)

PRIVATE_SPLITS = {
    "train":
        DATA_DIR
        / "private_train.parquet",

    "eval":
        DATA_DIR
        / "private_eval.parquet",

    "test":
        DATA_DIR
        / "private_test.parquet",
}

OUTPUT_DIR = (
    DATA_DIR
    / "security_features"
)

DIAGNOSTIC_DIR = (
    OUTPUT_DIR
    / "matching_diagnostics"
)


# These fields define whether several raw MBOX candidates are
# equivalent for the downstream security model.
#
# Candidate-specific provenance fields such as message_index are not
# included because they do not affect the security classification.

EQUIVALENT_SECURITY_COLUMNS = [
    "from_address",
    "from_domain",
    "from_org_domain",
    "return_path_address",
    "return_path_domain",
    "return_path_org_domain",
    "reply_to_address",
    "reply_to_domain",
    "reply_to_org_domain",
    "dkim_domains",
    "dkim_org_domains",
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "arc_spf_result",
    "arc_dkim_result",
    "arc_dmarc_result",
    "has_authentication_results",
    "has_arc_authentication_results",
    "has_received_spf",
    "has_dkim_signature",
    "has_return_path",
    "has_reply_to",
    "from_return_path_exact_match",
    "from_return_path_org_match",
    "from_reply_to_exact_match",
    "from_reply_to_org_match",
    "from_dkim_exact_match",
    "from_dkim_org_match",
    "shared_relay_domain",
    "spf_pass",
    "dkim_pass",
    "dmarc_pass",
    "spf_fail",
    "dkim_fail",
    "dmarc_fail",
]


def normalize_text(value) -> str:
    """
    Normalize a general text value.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    return (
        str(value)
        .strip()
        .casefold()
    )


def normalize_email(value) -> str:
    """
    Extract and normalize the actual sender email address.

    Example:

        "Example Store" <support@example.com>

    becomes:

        support@example.com

    Malformed sender strings are preserved in normalized form so that
    later fallback matching can still work.
    """

    value = normalize_text(
        value
    )

    if not value:
        return ""

    _, address = parseaddr(
        value
    )

    address = (
        address
        .strip()
        .casefold()
    )

    if (
        address
        and "@"
        in address
    ):
        return address

    match = re.search(
        r"[A-Z0-9._%+\-]+"
        r"@"
        r"[A-Z0-9.\-]+"
        r"\.[A-Z]{2,}",
        value,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            match
            .group(0)
            .casefold()
        )

    return value


def normalize_subject(value) -> str:
    """
    Normalize an email subject for deterministic matching.
    """

    value = normalize_text(
        value
    )

    if not value:
        return ""

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    return " ".join(
        value.split()
    )


def parse_timestamp(value):
    """
    Parse one timestamp independently.

    Parsing values individually avoids pandas inferring one datetime
    format for a column containing mixed timestamp formats.
    """

    if value is None:
        return pd.NaT

    try:
        if pd.isna(value):
            return pd.NaT
    except TypeError:
        pass

    value = str(
        value
    ).strip()

    if not value:
        return pd.NaT

    try:
        return pd.to_datetime(
            value,
            errors="coerce",
        )

    except Exception:
        return pd.NaT


def normalize_date_utc_value(value):
    """
    Convert one timestamp to a timezone-aware UTC timestamp.

    Naive timestamps are interpreted as UTC because that is how the
    original private dataset represented timestamps without an
    explicit offset.
    """

    timestamp = parse_timestamp(
        value
    )

    if pd.isna(
        timestamp
    ):
        return pd.NaT

    timestamp = pd.Timestamp(
        timestamp
    )

    if timestamp.tzinfo is None:

        return timestamp.tz_localize(
            "UTC"
        )

    return timestamp.tz_convert(
        "UTC"
    )


def prepare_security(
    dataframe,
):
    """
    Prepare the extracted security dataset for matching.
    """

    dataframe = (
        dataframe
        .copy()
        .reset_index(
            drop=True
        )
    )

    dataframe[
        "_security_row"
    ] = range(
        len(
            dataframe
        )
    )

    dataframe[
        "_match_sender"
    ] = (
        dataframe[
            "from_address"
        ]
        .map(
            normalize_email
        )
    )

    dataframe[
        "_match_subject"
    ] = (
        dataframe[
            "subject"
        ]
        .map(
            normalize_subject
        )
    )

    dataframe[
        "_match_date_utc"
    ] = (
        dataframe[
            "date_utc"
        ]
        .map(
            normalize_date_utc_value
        )
    )

    return dataframe


def prepare_private(
    dataframe,
    split_name,
):
    """
    Prepare one private ML split for matching.
    """

    dataframe = (
        dataframe
        .copy()
        .reset_index(
            drop=True
        )
    )

    dataframe[
        "private_split"
    ] = split_name

    dataframe[
        "_private_row"
    ] = range(
        len(
            dataframe
        )
    )

    dataframe[
        "_match_sender"
    ] = (
        dataframe[
            "sender"
        ]
        .map(
            normalize_email
        )
    )

    dataframe[
        "_match_subject"
    ] = (
        dataframe[
            "subject"
        ]
        .map(
            normalize_subject
        )
    )

    dataframe[
        "_match_date_utc"
    ] = (
        dataframe[
            "date"
        ]
        .map(
            normalize_date_utc_value
        )
    )

    return dataframe


def timestamp_is_missing(
    value,
) -> bool:
    """
    Check whether a timestamp is missing.
    """

    return bool(
        pd.isna(
            value
        )
    )


def build_key(
    row,
    columns,
):
    """
    Build a deterministic matching key.

    A key containing a missing date is rejected.
    """

    values = []

    for column in columns:

        value = row[
            column
        ]

        if column == "_match_date_utc":

            if timestamp_is_missing(
                value
            ):
                return None

        values.append(
            value
        )

    return tuple(
        values
    )


def build_index(
    dataframe,
    columns,
):
    """
    Build a mapping from matching key to dataframe row indices.
    """

    index = defaultdict(
        list
    )

    for row_index, row in dataframe.iterrows():

        key = build_key(
            row,
            columns,
        )

        if key is None:
            continue

        index[
            key
        ].append(
            row_index
        )

    return dict(
        index
    )


def build_key_counts(
    dataframe,
    columns,
):
    """
    Count how often each matching key occurs.
    """

    counts = defaultdict(
        int
    )

    for _, row in dataframe.iterrows():

        key = build_key(
            row,
            columns,
        )

        if key is None:
            continue

        counts[
            key
        ] += 1

    return dict(
        counts
    )


def normalize_signature_dataframe(
    dataframe,
):
    """
    Normalize security values before comparing candidate signatures.
    """

    dataframe = (
        dataframe
        .copy()
    )

    for column in dataframe.columns:

        dataframe[
            column
        ] = (
            dataframe[
                column
            ]
            .map(
                lambda value:
                    "<NA>"
                    if pd.isna(value)
                    else str(value)
            )
        )

    return dataframe


def security_candidates_are_equivalent(
    security_df,
    candidate_indices,
):
    """
    Determine whether several raw MBOX messages have identical
    security-relevant features.
    """

    if len(
        candidate_indices
    ) <= 1:
        return True

    columns = [
        column
        for column
        in EQUIVALENT_SECURITY_COLUMNS
        if column
        in security_df.columns
    ]

    if not columns:
        return False

    signatures = (
        security_df
        .loc[
            candidate_indices,
            columns,
        ]
    )

    signatures = (
        normalize_signature_dataframe(
            signatures
        )
    )

    unique_signatures = (
        signatures
        .drop_duplicates()
    )

    return (
        len(
            unique_signatures
        )
        == 1
    )


def evaluate_matching_stage(
    private_row,
    columns,
    security_index,
    private_counts,
    security_df,
    unique_method,
    equivalent_method,
):
    """
    Evaluate one matching stage.

    A unique security candidate is accepted only when the same key is
    unique in the private dataset.

    Multiple raw candidates are accepted only when their security
    signatures are equivalent.
    """

    key = build_key(
        private_row,
        columns,
    )

    if key is None:

        return {
            "status":
                "no_candidate",

            "security_index":
                None,

            "method":
                None,

            "candidate_count":
                0,

            "signature_count":
                0,
        }

    candidates = (
        security_index.get(
            key,
            [],
        )
    )

    private_count = (
        private_counts.get(
            key,
            0,
        )
    )

    if len(
        candidates
    ) == 0:

        return {
            "status":
                "no_candidate",

            "security_index":
                None,

            "method":
                None,

            "candidate_count":
                0,

            "signature_count":
                0,
        }

    # The fallback key must uniquely identify the private row.
    if private_count != 1:

        return {
            "status":
                "ambiguous",

            "security_index":
                None,

            "method":
                None,

            "candidate_count":
                len(
                    candidates
                ),

            "signature_count":
                None,
        }

    if len(
        candidates
    ) == 1:

        return {
            "status":
                "matched",

            "security_index":
                candidates[0],

            "method":
                unique_method,

            "candidate_count":
                1,

            "signature_count":
                1,
        }

    equivalent = (
        security_candidates_are_equivalent(
            security_df=security_df,
            candidate_indices=candidates,
        )
    )

    if equivalent:

        return {
            "status":
                "matched",

            # Selecting the first representative is safe because all
            # downstream security features are identical.
            "security_index":
                candidates[0],

            "method":
                equivalent_method,

            "candidate_count":
                len(
                    candidates
                ),

            "signature_count":
                1,
        }

    columns_available = [
        column
        for column
        in EQUIVALENT_SECURITY_COLUMNS
        if column
        in security_df.columns
    ]

    signatures = (
        security_df
        .loc[
            candidates,
            columns_available,
        ]
    )

    signatures = (
        normalize_signature_dataframe(
            signatures
        )
    )

    signature_count = len(
        signatures
        .drop_duplicates()
    )

    return {
        "status":
            "ambiguous",

        "security_index":
            None,

        "method":
            None,

        "candidate_count":
            len(
                candidates
            ),

        "signature_count":
            signature_count,
    }


def choose_match(
    private_row,
    security_df,
    exact_index,
    date_subject_index,
    metadata_index,
    private_exact_counts,
    private_date_subject_counts,
    private_metadata_counts,
):
    """
    Match one private email using increasingly relaxed but controlled
    rules.
    """

    exact_columns = [
        "source",
        "_match_date_utc",
        "_match_sender",
        "_match_subject",
    ]

    date_subject_columns = [
        "source",
        "_match_date_utc",
        "_match_subject",
    ]

    metadata_columns = [
        "source",
        "_match_sender",
        "_match_subject",
    ]

    ambiguity_information = None

    # Stage 1:
    # Strict source + date + sender + subject.

    result = (
        evaluate_matching_stage(
            private_row=private_row,
            columns=exact_columns,
            security_index=exact_index,
            private_counts=private_exact_counts,
            security_df=security_df,
            unique_method="utc_exact",
            equivalent_method="equivalent_duplicates_utc",
        )
    )

    if result[
        "status"
    ] == "matched":

        return result

    if result[
        "status"
    ] == "ambiguous":

        ambiguity_information = {
            **result,
            "method":
                "ambiguous_utc",
        }

    # Stage 2:
    # Ignore sender when it was malformed during original dataset
    # preparation, but keep exact source/date/subject.

    result = (
        evaluate_matching_stage(
            private_row=private_row,
            columns=date_subject_columns,
            security_index=date_subject_index,
            private_counts=private_date_subject_counts,
            security_df=security_df,
            unique_method="date_subject_unique",
            equivalent_method="equivalent_duplicates_date_subject",
        )
    )

    if result[
        "status"
    ] == "matched":

        return result

    if (
        result[
            "status"
        ] == "ambiguous"
        and ambiguity_information
        is None
    ):

        ambiguity_information = {
            **result,
            "method":
                "ambiguous_date_subject",
        }

    # Stage 3:
    # Ignore date only when source + sender + subject uniquely identify
    # the private message.

    result = (
        evaluate_matching_stage(
            private_row=private_row,
            columns=metadata_columns,
            security_index=metadata_index,
            private_counts=private_metadata_counts,
            security_df=security_df,
            unique_method="metadata_unique",
            equivalent_method="equivalent_duplicates_metadata",
        )
    )

    if result[
        "status"
    ] == "matched":

        return result

    if (
        result[
            "status"
        ] == "ambiguous"
        and ambiguity_information
        is None
    ):

        ambiguity_information = {
            **result,
            "method":
                "ambiguous_metadata",
        }

    if ambiguity_information is not None:

        return ambiguity_information

    return {
        "status":
            "no_match",

        "security_index":
            None,

        "method":
            "no_match",

        "candidate_count":
            0,

        "signature_count":
            0,
    }


def get_security_output_columns(
    security_df,
):
    """
    Select security columns to attach to private rows.

    Internal matching fields are excluded.
    """

    return [
        column
        for column
        in security_df.columns
        if not column.startswith(
            "_"
        )
    ]


def attach_security_features(
    private_df,
    security_df,
    exact_index,
    date_subject_index,
    metadata_index,
    private_exact_counts,
    private_date_subject_counts,
    private_metadata_counts,
):
    """
    Attach security information while preserving exactly one row per
    private email.
    """

    security_columns = (
        get_security_output_columns(
            security_df
        )
    )

    output_rows = []

    for _, private_row in private_df.iterrows():

        result = (
            choose_match(
                private_row=private_row,
                security_df=security_df,
                exact_index=exact_index,
                date_subject_index=date_subject_index,
                metadata_index=metadata_index,
                private_exact_counts=private_exact_counts,
                private_date_subject_counts=private_date_subject_counts,
                private_metadata_counts=private_metadata_counts,
            )
        )

        security_index = (
            result[
                "security_index"
            ]
        )

        output_row = {
            column:
                private_row[
                    column
                ]

            for column
            in private_df.columns

            if not column.startswith(
                "_match_"
            )
        }

        output_row[
            "security_match_method"
        ] = result[
            "method"
        ]

        output_row[
            "security_match_count"
        ] = int(
            result[
                "candidate_count"
            ]
        )

        output_row[
            "security_signature_count"
        ] = (
            result[
                "signature_count"
            ]
        )

        output_row[
            "security_matched"
        ] = (
            security_index
            is not None
        )

        output_row[
            "security_ambiguous"
        ] = (
            result[
                "status"
            ]
            == "ambiguous"
        )

        output_row[
            "security_equivalent_duplicates"
        ] = (
            result[
                "method"
            ]
            is not None
            and result[
                "method"
            ].startswith(
                "equivalent_duplicates"
            )
        )

        if security_index is not None:

            security_row = (
                security_df.loc[
                    security_index
                ]
            )

            for column in security_columns:

                output_row[
                    f"security_{column}"
                ] = (
                    security_row[
                        column
                    ]
                )

        else:

            for column in security_columns:

                output_row[
                    f"security_{column}"
                ] = None

        output_rows.append(
            output_row
        )

    return pd.DataFrame(
        output_rows
    )


def print_split_summary(
    dataframe,
    split_name,
):
    """
    Print matching statistics for one split.
    """

    total = len(
        dataframe
    )

    matched = int(
        dataframe[
            "security_matched"
        ]
        .sum()
    )

    ambiguous = int(
        dataframe[
            "security_ambiguous"
        ]
        .sum()
    )

    no_match = int(
        dataframe[
            "security_match_method"
        ]
        .eq(
            "no_match"
        )
        .sum()
    )

    print()
    print(
        split_name.upper()
    )

    print(
        f"Private messages:       "
        f"{total:,}"
    )

    print(
        f"Matched:                "
        f"{matched:,} "
        f"({matched / total * 100:.2f}%)"
    )

    methods = (
        dataframe[
            "security_match_method"
        ]
        .value_counts()
    )

    for method, count in methods.items():

        print(
            f"  {method:<30}"
            f"{count:,}"
        )

    print(
        f"Ambiguous:              "
        f"{ambiguous:,}"
    )

    print(
        f"No match:               "
        f"{no_match:,}"
    )


def save_diagnostics(
    dataframe,
):
    """
    Save unresolved cases without storing the email body.
    """

    DIAGNOSTIC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    unresolved = dataframe[
        ~dataframe[
            "security_matched"
        ]
    ].copy()

    diagnostic_columns = [
        "private_split",
        "_private_row",
        "source",
        "date",
        "sender",
        "subject",
        "label",
        "security_match_method",
        "security_match_count",
        "security_signature_count",
        "security_ambiguous",
    ]

    diagnostic_columns = [
        column
        for column
        in diagnostic_columns
        if column
        in unresolved.columns
    ]

    unresolved = unresolved[
        diagnostic_columns
    ]

    output_path = (
        DIAGNOSTIC_DIR
        / "unresolved_security_matches.csv"
    )

    unresolved.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return output_path


def save_summary(
    combined,
):
    """
    Save matching statistics per split.
    """

    rows = []

    for (
        split_name,
        group,
    ) in combined.groupby(
        "private_split",
        sort=False,
    ):

        total = len(
            group
        )

        matched = int(
            group[
                "security_matched"
            ]
            .sum()
        )

        ambiguous = int(
            group[
                "security_ambiguous"
            ]
            .sum()
        )

        no_match = int(
            group[
                "security_match_method"
            ]
            .eq(
                "no_match"
            )
            .sum()
        )

        equivalent_duplicates = int(
            group[
                "security_equivalent_duplicates"
            ]
            .sum()
        )

        row = {
            "split":
                split_name,

            "messages":
                int(
                    total
                ),

            "matched":
                matched,

            "match_rate_pct":
                (
                    matched
                    / total
                    * 100
                ),

            "equivalent_duplicates":
                equivalent_duplicates,

            "ambiguous":
                ambiguous,

            "no_match":
                no_match,
        }

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    output_path = (
        OUTPUT_DIR
        / "security_matching_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        summary,
        output_path,
    )


def main():
    """
    Match all private train/eval/test messages to security features.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Security feature matching"
    )

    security_df = (
        pd.read_parquet(
            SECURITY_PATH
        )
    )

    security_df = (
        prepare_security(
            security_df
        )
    )

    print(
        f"Security messages: "
        f"{len(security_df):,}"
    )

    private_frames = []

    for (
        split_name,
        split_path,
    ) in PRIVATE_SPLITS.items():

        private_df = (
            pd.read_parquet(
                split_path
            )
        )

        private_df = (
            prepare_private(
                dataframe=private_df,
                split_name=split_name,
            )
        )

        private_frames.append(
            private_df
        )

    private_all = pd.concat(
        private_frames,
        ignore_index=True,
    )

    private_all[
        "_global_private_row"
    ] = range(
        len(
            private_all
        )
    )

    exact_columns = [
        "source",
        "_match_date_utc",
        "_match_sender",
        "_match_subject",
    ]

    date_subject_columns = [
        "source",
        "_match_date_utc",
        "_match_subject",
    ]

    metadata_columns = [
        "source",
        "_match_sender",
        "_match_subject",
    ]

    exact_index = (
        build_index(
            security_df,
            exact_columns,
        )
    )

    date_subject_index = (
        build_index(
            security_df,
            date_subject_columns,
        )
    )

    metadata_index = (
        build_index(
            security_df,
            metadata_columns,
        )
    )

    private_exact_counts = (
        build_key_counts(
            private_all,
            exact_columns,
        )
    )

    private_date_subject_counts = (
        build_key_counts(
            private_all,
            date_subject_columns,
        )
    )

    private_metadata_counts = (
        build_key_counts(
            private_all,
            metadata_columns,
        )
    )

    matched_frames = []

    for split_name in PRIVATE_SPLITS:

        split_private = (
            private_all[
                private_all[
                    "private_split"
                ]
                .eq(
                    split_name
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        matched = (
            attach_security_features(
                private_df=split_private,
                security_df=security_df,
                exact_index=exact_index,
                date_subject_index=date_subject_index,
                metadata_index=metadata_index,
                private_exact_counts=private_exact_counts,
                private_date_subject_counts=private_date_subject_counts,
                private_metadata_counts=private_metadata_counts,
            )
        )

        if len(
            matched
        ) != len(
            split_private
        ):

            raise RuntimeError(
                f"Output row count mismatch for "
                f"{split_name}: "
                f"{len(matched):,} vs "
                f"{len(split_private):,}"
            )

        print_split_summary(
            dataframe=matched,
            split_name=split_name,
        )

        output_path = (
            OUTPUT_DIR
            / (
                f"private_"
                f"{split_name}_"
                f"security_matched.parquet"
            )
        )

        matched.to_parquet(
            output_path,
            index=False,
        )

        print(
            f"Saved: "
            f"{output_path}"
        )

        matched_frames.append(
            matched
        )

    combined = pd.concat(
        matched_frames,
        ignore_index=True,
    )

    total = len(
        combined
    )

    matched_total = int(
        combined[
            "security_matched"
        ]
        .sum()
    )

    ambiguous_total = int(
        combined[
            "security_ambiguous"
        ]
        .sum()
    )

    no_match_total = int(
        combined[
            "security_match_method"
        ]
        .eq(
            "no_match"
        )
        .sum()
    )

    equivalent_total = int(
        combined[
            "security_equivalent_duplicates"
        ]
        .sum()
    )

    summary, summary_path = (
        save_summary(
            combined
        )
    )

    diagnostic_path = (
        save_diagnostics(
            combined
        )
    )

    print()
    print(
        "MATCHING COMPLETE"
    )

    print(
        f"Private messages total: "
        f"{total:,}"
    )

    print(
        f"Matched total:          "
        f"{matched_total:,}"
    )

    print(
        f"Overall match rate:     "
        f"{matched_total / total * 100:.4f}%"
    )

    print(
        f"Equivalent duplicates: "
        f"{equivalent_total:,}"
    )

    print(
        f"Ambiguous total:        "
        f"{ambiguous_total:,}"
    )

    print(
        f"No match total:         "
        f"{no_match_total:,}"
    )

    print()
    print(
        "Match methods:"
    )

    print(
        combined[
            "security_match_method"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Summary saved:"
    )

    print(
        summary_path
    )

    print()
    print(
        "Unresolved cases saved:"
    )

    print(
        diagnostic_path
    )


if __name__ == "__main__":
    main()
