"""
Extract security-related features from private Thunderbird MBOX files.

The script processes the raw private email sources and creates compact
Parquet files containing sender, authentication, and domain-alignment
features.

No email body content is stored.

Mailbox paths and source identifiers are loaded from the same ignored
local configuration used by private dataset preparation.

Important:
Existing spam-filter decisions such as X-GMX-Antispam or X-Spam-Flag
are deliberately NOT used as model features because they could create
target leakage.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import tldextract

from spam_detector.inspection.inspect_email_headers import (
    extract_auth_result,
    extract_message_information,
    iter_mbox_messages_binary,
)
from spam_detector.paths import DATA_DIR
from spam_detector.private_mailbox_config import (
    DEFAULT_PRIVATE_MAILBOX_CONFIG,
    load_private_mailboxes,
)

OUTPUT_DIR = (
    DATA_DIR
    / "security_features"
)


SHARED_RELAY_DOMAINS = {
    "privaterelay.appleid.com",
}


# Use the bundled Public Suffix List snapshot.
# This avoids downloading anything while the script runs.

TLD_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=None
)


def normalize_domain(
    domain,
) -> str:
    """
    Normalize a domain name.
    """

    if domain is None:
        return ""

    if pd.isna(domain):
        return ""

    return (
        str(domain)
        .strip()
        .lower()
        .rstrip(".")
    )


def get_organizational_domain(
    domain,
) -> str:
    """
    Extract the registrable / organizational domain.

    Examples:

    mail.cdkeys.com
        -> cdkeys.com

    em229451.gamivo.com
        -> gamivo.com

    ebooking.qatarairways.com.qa
        -> qatarairways.com.qa
    """

    domain = normalize_domain(
        domain
    )

    if not domain:
        return ""

    extracted = (
        TLD_EXTRACTOR(
            domain
        )
    )

    if (
        not extracted.domain
        or not extracted.suffix
    ):
        return domain

    return (
        f"{extracted.domain}."
        f"{extracted.suffix}"
    )


def is_shared_relay_domain(
    domain,
) -> bool:
    """
    Determine whether a domain represents shared relay infrastructure.
    """

    domain = normalize_domain(
        domain
    )

    if not domain:
        return False

    for relay_domain in SHARED_RELAY_DOMAINS:

        if domain == relay_domain:
            return True

        if domain.endswith(
            "." + relay_domain
        ):
            return True

    return False


def exact_domain_match(
    first_domain,
    second_domain,
):
    """
    Compare two domains exactly.

    Missing values return None rather than False.
    """

    first_domain = normalize_domain(
        first_domain
    )

    second_domain = normalize_domain(
        second_domain
    )

    if (
        not first_domain
        or not second_domain
    ):
        return None

    return (
        first_domain
        == second_domain
    )


def organizational_domain_match(
    first_domain,
    second_domain,
):
    """
    Compare the registrable organizational domains.

    Example:

    cdkeys.com
    mail.cdkeys.com

    -> True
    """

    first_domain = normalize_domain(
        first_domain
    )

    second_domain = normalize_domain(
        second_domain
    )

    if (
        not first_domain
        or not second_domain
    ):
        return None

    first_org = (
        get_organizational_domain(
            first_domain
        )
    )

    second_org = (
        get_organizational_domain(
            second_domain
        )
    )

    if (
        not first_org
        or not second_org
    ):
        return None

    return (
        first_org
        == second_org
    )


def split_semicolon_values(
    value,
) -> list[str]:
    """
    Split semicolon-separated values from the header inspector.
    """

    if value is None:
        return []

    if pd.isna(value):
        return []

    values = []

    for part in str(value).split(";"):

        part = normalize_domain(
            part
        )

        if part:
            values.append(
                part
            )

    return values


def any_exact_domain_match(
    domain,
    candidate_domains,
):
    """
    Check whether a domain exactly matches any candidate domain.
    """

    domain = normalize_domain(
        domain
    )

    candidate_domains = [
        normalize_domain(
            candidate
        )
        for candidate in candidate_domains
        if normalize_domain(
            candidate
        )
    ]

    if (
        not domain
        or not candidate_domains
    ):
        return None

    return (
        domain
        in candidate_domains
    )


def any_organizational_domain_match(
    domain,
    candidate_domains,
):
    """
    Check whether the organizational domain matches any candidate.
    """

    domain = normalize_domain(
        domain
    )

    if not domain:
        return None

    if not candidate_domains:
        return None

    domain_org = (
        get_organizational_domain(
            domain
        )
    )

    if not domain_org:
        return None

    candidate_org_domains = {
        get_organizational_domain(
            candidate
        )
        for candidate in candidate_domains
        if candidate
    }

    candidate_org_domains.discard(
        ""
    )

    if not candidate_org_domains:
        return None

    return (
        domain_org
        in candidate_org_domains
    )


def parse_date_utc(
    date_value,
):
    """
    Parse an email date and normalize it to UTC.

    Invalid dates are returned as NaT.
    """

    return pd.to_datetime(
        date_value,
        errors="coerce",
        utc=True,
    )


def result_is(
    result,
    expected,
) -> bool:
    """
    Compare an authentication result with a specific value.
    """

    if result is None:
        return False

    return (
        str(result)
        .strip()
        .lower()
        == expected
    )


def extract_security_row(
    message,
    mbox_path,
    message_index,
    source,
):
    """
    Extract security features from one message.
    """

    info = (
        extract_message_information(
            message=message,
            mbox_path=mbox_path,
            message_index=message_index,
        )
    )

    from_domain = normalize_domain(
        info.get(
            "from_domain"
        )
    )

    return_path_domain = normalize_domain(
        info.get(
            "return_path_domain"
        )
    )

    reply_to_domain = normalize_domain(
        info.get(
            "reply_to_domain"
        )
    )

    dkim_domains = (
        split_semicolon_values(
            info.get(
                "dkim_domains"
            )
        )
    )

    from_org_domain = (
        get_organizational_domain(
            from_domain
        )
    )

    return_path_org_domain = (
        get_organizational_domain(
            return_path_domain
        )
    )

    reply_to_org_domain = (
        get_organizational_domain(
            reply_to_domain
        )
    )

    dkim_org_domains = sorted(
        {
            get_organizational_domain(
                domain
            )
            for domain in dkim_domains
            if domain
        }
        - {""}
    )

    # Use Authentication-Results from the receiving mail system
    # separately from ARC results.
    #
    # This is deliberately stricter than treating ARC as equivalent
    # to the receiver's own authentication result.

    authentication_results = (
        info.get(
            "authentication_results",
            "",
        )
        or ""
    )

    arc_authentication_results = (
        info.get(
            "arc_authentication_results",
            "",
        )
        or ""
    )

    spf_result = (
        extract_auth_result(
            authentication_results,
            "spf",
        )
    )

    dkim_result = (
        extract_auth_result(
            authentication_results,
            "dkim",
        )
    )

    dmarc_result = (
        extract_auth_result(
            authentication_results,
            "dmarc",
        )
    )

    arc_spf_result = (
        extract_auth_result(
            arc_authentication_results,
            "spf",
        )
    )

    arc_dkim_result = (
        extract_auth_result(
            arc_authentication_results,
            "dkim",
        )
    )

    arc_dmarc_result = (
        extract_auth_result(
            arc_authentication_results,
            "dmarc",
        )
    )

    date_utc = (
        parse_date_utc(
            info.get(
                "date"
            )
        )
    )

    row = {
        "source":
            source,

        "mailbox":
            info.get(
                "mailbox",
                "",
            ),

        "message_index":
            int(
                message_index
            ),

        "message_id":
            info.get(
                "message_id",
                "",
            ),

        "date_raw":
            info.get(
                "date",
                "",
            ),

        "date_utc":
            date_utc,

        "subject":
            info.get(
                "subject",
                "",
            ),

        "display_name":
            info.get(
                "display_name",
                "",
            ),

        "from_address":
            info.get(
                "from_address",
                "",
            ),

        "from_domain":
            from_domain,

        "from_org_domain":
            from_org_domain,

        "return_path_address":
            info.get(
                "return_path_address",
                "",
            ),

        "return_path_domain":
            return_path_domain,

        "return_path_org_domain":
            return_path_org_domain,

        "reply_to_address":
            info.get(
                "reply_to_address",
                "",
            ),

        "reply_to_domain":
            reply_to_domain,

        "reply_to_org_domain":
            reply_to_org_domain,

        "dkim_domains":
            "; ".join(
                dkim_domains
            ),

        "dkim_org_domains":
            "; ".join(
                dkim_org_domains
            ),

        "spf_result":
            spf_result,

        "dkim_result":
            dkim_result,

        "dmarc_result":
            dmarc_result,

        "arc_spf_result":
            arc_spf_result,

        "arc_dkim_result":
            arc_dkim_result,

        "arc_dmarc_result":
            arc_dmarc_result,

        "has_authentication_results":
            bool(
                info.get(
                    "has_authentication_results",
                    False,
                )
            ),

        "has_arc_authentication_results":
            bool(
                info.get(
                    "has_arc_authentication_results",
                    False,
                )
            ),

        "has_received_spf":
            bool(
                info.get(
                    "has_received_spf",
                    False,
                )
            ),

        "has_dkim_signature":
            bool(
                info.get(
                    "has_dkim_signature",
                    False,
                )
            ),

        "has_return_path":
            bool(
                info.get(
                    "has_return_path",
                    False,
                )
            ),

        "has_reply_to":
            bool(
                info.get(
                    "has_reply_to",
                    False,
                )
            ),

        "from_return_path_exact_match":
            exact_domain_match(
                from_domain,
                return_path_domain,
            ),

        "from_return_path_org_match":
            organizational_domain_match(
                from_domain,
                return_path_domain,
            ),

        "from_reply_to_exact_match":
            exact_domain_match(
                from_domain,
                reply_to_domain,
            ),

        "from_reply_to_org_match":
            organizational_domain_match(
                from_domain,
                reply_to_domain,
            ),

        "from_dkim_exact_match":
            any_exact_domain_match(
                from_domain,
                dkim_domains,
            ),

        "from_dkim_org_match":
            any_organizational_domain_match(
                from_domain,
                dkim_domains,
            ),

        "shared_relay_domain":
            is_shared_relay_domain(
                from_domain
            ),

        "spf_pass":
            result_is(
                spf_result,
                "pass",
            ),

        "dkim_pass":
            result_is(
                dkim_result,
                "pass",
            ),

        "dmarc_pass":
            result_is(
                dmarc_result,
                "pass",
            ),

        "spf_fail":
            result_is(
                spf_result,
                "fail",
            ),

        "dkim_fail":
            result_is(
                dkim_result,
                "fail",
            ),

        "dmarc_fail":
            result_is(
                dmarc_result,
                "fail",
            ),
    }

    return row


def extract_mailbox(
    source,
    mbox_path,
):
    """
    Extract security features from one complete MBOX file.
    """

    print()
    print(
        f"Processing source: {source}"
    )

    print(
        f"MBOX: {mbox_path}"
    )

    rows = []

    for (
        message_index,
        message,
    ) in iter_mbox_messages_binary(
        mbox_path
    ):

        try:

            row = (
                extract_security_row(
                    message=message,
                    mbox_path=mbox_path,
                    message_index=message_index,
                    source=source,
                )
            )

            rows.append(
                row
            )

        except Exception as error:

            print(
                f"Warning: failed to extract "
                f"message {message_index}: "
                f"{error}"
            )

        processed = (
            message_index
            + 1
        )

        if (
            processed % 1000
            == 0
        ):

            print(
                f"Processed "
                f"{processed:,} messages..."
            )

    dataframe = pd.DataFrame(
        rows
    )

    print(
        f"Finished {source}: "
        f"{len(dataframe):,} messages"
    )

    return dataframe


def create_coverage_summary(
    dataframe,
):
    """
    Create per-source feature coverage statistics.
    """

    fields = [
        "has_authentication_results",
        "has_dkim_signature",
        "has_return_path",
        "has_reply_to",
        "spf_pass",
        "dkim_pass",
        "dmarc_pass",
        "spf_fail",
        "dkim_fail",
        "dmarc_fail",
        "from_return_path_org_match",
        "from_dkim_org_match",
        "shared_relay_domain",
    ]

    summary_rows = []

    for source, group in dataframe.groupby(
        "source"
    ):

        row = {
            "source":
                source,

            "messages":
                len(group),
        }

        for field in fields:

            if field not in group.columns:
                continue

            count = int(
                group[
                    field
                ]
                .fillna(False)
                .eq(True)
                .sum()
            )

            row[
                f"{field}_count"
            ] = count

            row[
                f"{field}_pct"
            ] = (
                count
                / len(group)
                * 100
            )

        summary_rows.append(
            row
        )

    return pd.DataFrame(
        summary_rows
    )


def build_summary_json(
    dataframe,
):
    """
    Build general extraction statistics.
    """

    total = len(
        dataframe
    )

    return {
        "total_messages":
            int(total),

        "sources":
            {
                source:
                    int(count)

                for source, count
                in dataframe[
                    "source"
                ]
                .value_counts()
                .to_dict()
                .items()
            },

        "missing_date":
            int(
                dataframe[
                    "date_utc"
                ]
                .isna()
                .sum()
            ),

        "authentication_results_available":
            int(
                dataframe[
                    "has_authentication_results"
                ]
                .sum()
            ),

        "dkim_signature_available":
            int(
                dataframe[
                    "has_dkim_signature"
                ]
                .sum()
            ),

        "return_path_available":
            int(
                dataframe[
                    "has_return_path"
                ]
                .sum()
            ),

        "shared_relay_messages":
            int(
                dataframe[
                    "shared_relay_domain"
                ]
                .sum()
            ),
    }


def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Extract security features from private "
            "Thunderbird MBOX files."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PRIVATE_MAILBOX_CONFIG,
        help=(
            "Path to the ignored private mailbox TOML configuration "
            f"(default: {DEFAULT_PRIVATE_MAILBOX_CONFIG})."
        ),
    )

    return parser.parse_args()


def main():
    """
    Extract security features from all private mail sources.
    """

    arguments = (
        parse_arguments()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mailbox_definitions = load_private_mailboxes(
        arguments.config
    )

    print(
        "Private mailbox security feature extraction"
    )

    print()

    print(
        "Mailboxes:"
    )

    for definition in mailbox_definitions:

        print(
            f"  {definition.source}:"
        )

        print(
            f"    {definition.path}"
        )

    all_frames = []

    for definition in mailbox_definitions:

        dataframe = (
            extract_mailbox(
                source=definition.source,
                mbox_path=definition.path,
            )
        )

        output_path = (
            OUTPUT_DIR
            / f"{definition.source}_security.parquet"
        )

        dataframe.to_parquet(
            output_path,
            index=False,
        )

        print(
            f"Saved:\n{output_path}"
        )

        all_frames.append(
            dataframe
        )

    combined = pd.concat(
        all_frames,
        ignore_index=True,
    )

    combined_path = (
        OUTPUT_DIR
        / "private_security_features.parquet"
    )

    combined.to_parquet(
        combined_path,
        index=False,
    )

    coverage = (
        create_coverage_summary(
            combined
        )
    )

    coverage_path = (
        OUTPUT_DIR
        / "security_feature_coverage.csv"
    )

    coverage.to_csv(
        coverage_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        build_summary_json(
            combined
        )
    )

    summary_path = (
        OUTPUT_DIR
        / "security_feature_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        "SECURITY FEATURE EXTRACTION COMPLETE"
    )

    print()

    print(
        f"Total messages: "
        f"{len(combined):,}"
    )

    print()

    print(
        "Messages per source:"
    )

    print(
        combined[
            "source"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Feature coverage:"
    )

    print()

    print(
        coverage.to_string(
            index=False
        )
    )

    print()

    print(
        "Combined security features:"
    )

    print(
        combined_path
    )

    print()

    print(
        "Coverage report:"
    )

    print(
        coverage_path
    )

    print()

    print(
        "Summary:"
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()
