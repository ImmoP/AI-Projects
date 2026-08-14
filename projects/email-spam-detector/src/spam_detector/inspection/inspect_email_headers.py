"""
Inspect security-relevant email headers in Thunderbird MBOX files.

The script reads MBOX files in binary mode. This is more robust than
Python's mailbox.mbox implementation for exports containing malformed
or non-ASCII MBOX envelope lines.

It inspects headers such as:

- From
- Reply-To
- Return-Path
- Authentication-Results
- ARC-Authentication-Results
- Received-SPF
- DKIM-Signature
- Received
- Message-ID
- X-Spam headers

It also extracts structured information such as:

- From domain
- Reply-To domain
- Return-Path domain
- DKIM signing domain
- SPF result
- DKIM result
- DMARC result
"""

import argparse
import json
import re
from email import policy
from email.header import (
    decode_header,
    make_header,
)
from email.parser import BytesParser
from email.utils import (
    parseaddr,
)
from pathlib import Path

import pandas as pd

from spam_detector.paths import DATA_DIR

OUTPUT_DIR = (
    DATA_DIR
    / "header_inspection"
)

RAW_HEADERS_DIR = (
    OUTPUT_DIR
    / "raw_headers"
)


SECURITY_HEADER_NAMES = {
    "authentication-results",
    "arc-authentication-results",
    "received-spf",
    "dkim-signature",
    "return-path",
    "reply-to",
    "received",
    "x-spam-flag",
    "x-spam-status",
    "x-spam-score",
    "x-spam-level",
    "x-gmx-antispam",
    "x-microsoft-antispam",
    "x-microsoft-antispam-message-info",
    "x-forefront-antispam-report",
}


# Match the date portion at the end of an MBOX separator line.
#
# The sender portion between "From " and the date is deliberately
# allowed to contain arbitrary bytes. Some Thunderbird exports contain
# malformed/non-ASCII bytes in the envelope sender.

MBOX_SEPARATOR_PATTERN = re.compile(
    rb"^From .* "
    rb"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) "
    rb"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    rb"[ 0-9]{1,2} "
    rb"[0-9]{2}:[0-9]{2}:[0-9]{2} "
    rb"[0-9]{4}"
    rb"\r?\n?$"
)


def decode_header_value(
    value,
) -> str:
    """
    Decode a potentially MIME-encoded email header.
    """

    if value is None:
        return ""

    try:
        return str(
            make_header(
                decode_header(
                    str(value)
                )
            )
        )

    except Exception:
        return str(value)


def get_header(
    message,
    name: str,
) -> str:
    """
    Return one decoded header value.
    """

    return decode_header_value(
        message.get(
            name,
            "",
        )
    )


def get_all_headers(
    message,
    name: str,
) -> list[str]:
    """
    Return all occurrences of a header.
    """

    values = message.get_all(
        name,
        [],
    )

    return [
        decode_header_value(
            value
        )
        for value in values
    ]


def join_headers(
    values: list[str],
) -> str:
    """
    Join multiple header values for CSV storage.
    """

    return " | ".join(
        value
        for value in values
        if value
    )


def extract_email_address(
    header_value: str,
) -> str:
    """
    Extract an email address from a sender-related header.
    """

    if not header_value:
        return ""

    _, address = parseaddr(
        header_value
    )

    return (
        address
        .strip()
        .lower()
    )


def extract_domain(
    email_address: str,
) -> str:
    """
    Extract the domain from an email address.
    """

    if not email_address:
        return ""

    if "@" not in email_address:
        return ""

    return (
        email_address
        .rsplit(
            "@",
            1,
        )[1]
        .strip()
        .lower()
        .strip("<>")
        .rstrip(".")
    )


def extract_display_name(
    header_value: str,
) -> str:
    """
    Extract the display name from a sender header.
    """

    if not header_value:
        return ""

    display_name, _ = parseaddr(
        header_value
    )

    return display_name.strip()


def extract_auth_result(
    authentication_results: str,
    mechanism: str,
) -> str:
    """
    Extract SPF, DKIM, or DMARC result from
    Authentication-Results.
    """

    if not authentication_results:
        return ""

    pattern = (
        rf"\b{re.escape(mechanism)}"
        rf"\s*=\s*"
        rf"([a-zA-Z0-9_-]+)"
    )

    match = re.search(
        pattern,
        authentication_results,
        flags=re.IGNORECASE,
    )

    if not match:
        return ""

    return (
        match
        .group(1)
        .lower()
    )


def extract_received_spf_result(
    received_spf: str,
) -> str:
    """
    Extract the result from Received-SPF.
    """

    if not received_spf:
        return ""

    pattern = (
        r"^\s*"
        r"(pass|fail|softfail|neutral|none|"
        r"temperror|permerror)"
    )

    match = re.search(
        pattern,
        received_spf,
        flags=re.IGNORECASE,
    )

    if not match:
        return ""

    return (
        match
        .group(1)
        .lower()
    )


def extract_dkim_domains(
    dkim_signatures: list[str],
) -> list[str]:
    """
    Extract all d= signing domains from DKIM-Signature headers.
    """

    domains = []

    for signature in dkim_signatures:

        matches = re.findall(
            r"(?:^|;)\s*d\s*=\s*([^;\s]+)",
            signature,
            flags=re.IGNORECASE,
        )

        for domain in matches:

            domain = (
                domain
                .strip()
                .lower()
                .rstrip(".")
            )

            if (
                domain
                and domain not in domains
            ):
                domains.append(
                    domain
                )

    return domains


def extract_dkim_selectors(
    dkim_signatures: list[str],
) -> list[str]:
    """
    Extract all s= selectors from DKIM-Signature headers.
    """

    selectors = []

    for signature in dkim_signatures:

        matches = re.findall(
            r"(?:^|;)\s*s\s*=\s*([^;\s]+)",
            signature,
            flags=re.IGNORECASE,
        )

        for selector in matches:

            selector = selector.strip()

            if (
                selector
                and selector not in selectors
            ):
                selectors.append(
                    selector
                )

    return selectors


def extract_message_id_domain(
    message_id: str,
) -> str:
    """
    Extract the domain from Message-ID when possible.
    """

    if not message_id:
        return ""

    match = re.search(
        r"@([^>\s]+)",
        message_id,
    )

    if not match:
        return ""

    return (
        match
        .group(1)
        .lower()
        .rstrip(".")
    )


def domain_match(
    first_domain: str,
    second_domain: str,
):
    """
    Compare two domains exactly.
    """

    if (
        not first_domain
        or not second_domain
    ):
        return None

    return (
        first_domain
        == second_domain
    )


def from_matches_any_dkim_domain(
    from_domain: str,
    dkim_domains: list[str],
):
    """
    Check whether the exact From domain occurs as a DKIM
    signing domain.
    """

    if (
        not from_domain
        or not dkim_domains
    ):
        return None

    return (
        from_domain
        in dkim_domains
    )


def get_interesting_header_names(
    message,
) -> list[str]:
    """
    List security-related headers present in the message.
    """

    found = []

    for name in message.keys():

        name_lower = (
            name
            .lower()
            .strip()
        )

        if (
            name_lower
            in SECURITY_HEADER_NAMES
        ):

            if name not in found:
                found.append(
                    name
                )

            continue

        if any(
            keyword in name_lower
            for keyword in [
                "authentication",
                "antispam",
                "anti-spam",
                "spam-score",
                "spam-status",
                "received-spf",
                "dmarc",
            ]
        ):

            if name not in found:
                found.append(
                    name
                )

    return found


def build_raw_header_text(
    message,
) -> str:
    """
    Create a text representation containing headers only.

    The message body is deliberately excluded.
    """

    lines = []

    for name, value in message.items():

        decoded_value = (
            decode_header_value(
                value
            )
        )

        lines.append(
            f"{name}: {decoded_value}"
        )

    return "\n".join(
        lines
    )


def message_matches_search(
    message,
    contains: str | None,
    sender_contains: str | None,
    subject_contains: str | None,
) -> bool:
    """
    Determine whether an email matches optional search criteria.
    """

    from_header = get_header(
        message,
        "From",
    )

    subject = get_header(
        message,
        "Subject",
    )

    if sender_contains:

        if (
            sender_contains.lower()
            not in from_header.lower()
        ):
            return False

    if subject_contains:

        if (
            subject_contains.lower()
            not in subject.lower()
        ):
            return False

    if contains:

        header_text = (
            build_raw_header_text(
                message
            )
        )

        if (
            contains.lower()
            not in header_text.lower()
        ):
            return False

    return True


def extract_message_information(
    message,
    mbox_path: Path,
    message_index: int,
) -> dict:
    """
    Extract security-relevant information from one email.
    """

    from_header = get_header(
        message,
        "From",
    )

    reply_to_header = get_header(
        message,
        "Reply-To",
    )

    return_path_header = get_header(
        message,
        "Return-Path",
    )

    subject = get_header(
        message,
        "Subject",
    )

    date = get_header(
        message,
        "Date",
    )

    message_id = get_header(
        message,
        "Message-ID",
    )

    authentication_results_values = (
        get_all_headers(
            message,
            "Authentication-Results",
        )
    )

    arc_authentication_results_values = (
        get_all_headers(
            message,
            "ARC-Authentication-Results",
        )
    )

    received_spf_values = (
        get_all_headers(
            message,
            "Received-SPF",
        )
    )

    dkim_signatures = (
        get_all_headers(
            message,
            "DKIM-Signature",
        )
    )

    received_headers = (
        get_all_headers(
            message,
            "Received",
        )
    )

    authentication_results = (
        join_headers(
            authentication_results_values
        )
    )

    arc_authentication_results = (
        join_headers(
            arc_authentication_results_values
        )
    )

    received_spf = (
        join_headers(
            received_spf_values
        )
    )

    combined_authentication = " | ".join(
        value
        for value in [
            authentication_results,
            arc_authentication_results,
        ]
        if value
    )

    from_address = (
        extract_email_address(
            from_header
        )
    )

    reply_to_address = (
        extract_email_address(
            reply_to_header
        )
    )

    return_path_address = (
        extract_email_address(
            return_path_header
        )
    )

    from_domain = (
        extract_domain(
            from_address
        )
    )

    reply_to_domain = (
        extract_domain(
            reply_to_address
        )
    )

    return_path_domain = (
        extract_domain(
            return_path_address
        )
    )

    dkim_domains = (
        extract_dkim_domains(
            dkim_signatures
        )
    )

    dkim_selectors = (
        extract_dkim_selectors(
            dkim_signatures
        )
    )

    spf_result = (
        extract_auth_result(
            combined_authentication,
            "spf",
        )
    )

    dkim_result = (
        extract_auth_result(
            combined_authentication,
            "dkim",
        )
    )

    dmarc_result = (
        extract_auth_result(
            combined_authentication,
            "dmarc",
        )
    )

    received_spf_result = (
        extract_received_spf_result(
            received_spf
        )
    )

    interesting_headers = (
        get_interesting_header_names(
            message
        )
    )

    first_received_headers = (
        received_headers[:3]
    )

    return {
        "mailbox":
            mbox_path.name,

        "mailbox_path":
            str(mbox_path),

        "message_index":
            int(message_index),

        "date":
            date,

        "subject":
            subject,

        "display_name":
            extract_display_name(
                from_header
            ),

        "from_header":
            from_header,

        "from_address":
            from_address,

        "from_domain":
            from_domain,

        "reply_to_header":
            reply_to_header,

        "reply_to_address":
            reply_to_address,

        "reply_to_domain":
            reply_to_domain,

        "return_path_header":
            return_path_header,

        "return_path_address":
            return_path_address,

        "return_path_domain":
            return_path_domain,

        "message_id":
            message_id,

        "message_id_domain":
            extract_message_id_domain(
                message_id
            ),

        "spf_result":
            spf_result,

        "received_spf_result":
            received_spf_result,

        "dkim_result":
            dkim_result,

        "dmarc_result":
            dmarc_result,

        "dkim_domains":
            "; ".join(
                dkim_domains
            ),

        "dkim_selectors":
            "; ".join(
                dkim_selectors
            ),

        "from_return_path_exact_match":
            domain_match(
                from_domain,
                return_path_domain,
            ),

        "from_reply_to_exact_match":
            domain_match(
                from_domain,
                reply_to_domain,
            ),

        "from_dkim_exact_match":
            from_matches_any_dkim_domain(
                from_domain,
                dkim_domains,
            ),

        "has_authentication_results":
            bool(
                authentication_results_values
            ),

        "has_arc_authentication_results":
            bool(
                arc_authentication_results_values
            ),

        "has_received_spf":
            bool(
                received_spf_values
            ),

        "has_dkim_signature":
            bool(
                dkim_signatures
            ),

        "has_return_path":
            bool(
                return_path_header
            ),

        "has_reply_to":
            bool(
                reply_to_header
            ),

        "authentication_results":
            authentication_results,

        "arc_authentication_results":
            arc_authentication_results,

        "received_spf":
            received_spf,

        "dkim_signature":
            join_headers(
                dkim_signatures
            ),

        "first_received_headers":
            join_headers(
                first_received_headers
            ),

        "interesting_header_names":
            "; ".join(
                interesting_headers
            ),
    }


def sanitize_filename(
    value: str,
) -> str:
    """
    Make a string safe for Windows/macOS filenames.
    """

    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    return value.strip(
        "_"
    )


def save_raw_headers(
    message,
    mbox_path: Path,
    message_index: int,
    tag: str,
) -> Path:
    """
    Save the complete header block of a selected message.
    """

    RAW_HEADERS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mailbox_name = (
        sanitize_filename(
            mbox_path.name
        )
    )

    filename = (
        f"{tag}_"
        f"{mailbox_name}_"
        f"message_{message_index:06d}_headers.txt"
    )

    output_path = (
        RAW_HEADERS_DIR
        / filename
    )

    raw_header_text = (
        build_raw_header_text(
            message
        )
    )

    output_path.write_text(
        raw_header_text,
        encoding="utf-8",
        errors="replace",
    )

    return output_path


def is_mbox_separator_line(
    line: bytes,
) -> bool:
    """
    Check whether a raw byte line is an MBOX separator.

    The sender part may contain malformed or non-ASCII bytes.
    """

    if not line.startswith(
        b"From "
    ):
        return False

    return bool(
        MBOX_SEPARATOR_PATTERN.match(
            line
        )
    )


def iter_mbox_messages_binary(
    mbox_path: Path,
):
    """
    Read an MBOX file sequentially in binary mode.

    This avoids the strict ASCII decoding of the MBOX envelope
    line performed by Python's mailbox.mbox implementation.

    The file is streamed message-by-message and therefore does
    not need to be loaded completely into memory.
    """

    parser = BytesParser(
        policy=policy.default
    )

    message_bytes = (
        bytearray()
    )

    message_index = -1

    with open(
        mbox_path,
        "rb",
    ) as file:

        for line in file:

            if is_mbox_separator_line(
                line
            ):

                if (
                    message_index >= 0
                    and message_bytes
                ):

                    try:

                        message = (
                            parser.parsebytes(
                                bytes(
                                    message_bytes
                                )
                            )
                        )

                        yield (
                            message_index,
                            message,
                        )

                    except Exception as error:

                        print(
                            f"Warning: could not parse "
                            f"message {message_index:,}: "
                            f"{error}"
                        )

                message_index += 1

                message_bytes = (
                    bytearray()
                )

                continue

            if message_index >= 0:

                message_bytes.extend(
                    line
                )

        if (
            message_index >= 0
            and message_bytes
        ):

            try:

                message = (
                    parser.parsebytes(
                        bytes(
                            message_bytes
                        )
                    )
                )

                yield (
                    message_index,
                    message,
                )

            except Exception as error:

                print(
                    f"Warning: could not parse "
                    f"message {message_index:,}: "
                    f"{error}"
                )


def inspect_mbox(
    mbox_path: Path,
    limit: int,
    contains: str | None,
    sender_contains: str | None,
    subject_contains: str | None,
    tag: str,
) -> list[dict]:
    """
    Inspect selected emails in one MBOX file.

    The MBOX is read in binary mode to tolerate malformed
    non-ASCII envelope lines.
    """

    if not mbox_path.exists():

        raise FileNotFoundError(
            f"MBOX file not found: "
            f"{mbox_path}"
        )

    print(
        "\nInspecting MBOX:"
    )

    print(
        mbox_path
    )

    rows = []

    scanned = 0

    for (
        message_index,
        message,
    ) in iter_mbox_messages_binary(
        mbox_path
    ):

        scanned += 1

        if not message_matches_search(
            message=message,
            contains=contains,
            sender_contains=sender_contains,
            subject_contains=subject_contains,
        ):
            continue

        information = (
            extract_message_information(
                message=message,
                mbox_path=mbox_path,
                message_index=message_index,
            )
        )

        raw_header_path = (
            save_raw_headers(
                message=message,
                mbox_path=mbox_path,
                message_index=message_index,
                tag=tag,
            )
        )

        information[
            "raw_header_file"
        ] = str(
            raw_header_path
        )

        rows.append(
            information
        )

        print(
            f"Selected message "
            f"{message_index:,}: "
            f"{information['from_address']} | "
            f"{information['subject'][:80]}"
        )

        if len(
            rows
        ) >= limit:
            break

    print(
        f"Messages scanned: "
        f"{scanned:,}"
    )

    print(
        f"Messages selected: "
        f"{len(rows):,}"
    )

    return rows


def create_coverage_summary(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Summarize security-header availability.
    """

    if dataframe.empty:

        return {
            "messages": 0,
        }

    total = len(
        dataframe
    )

    fields = [
        "has_authentication_results",
        "has_arc_authentication_results",
        "has_received_spf",
        "has_dkim_signature",
        "has_return_path",
        "has_reply_to",
    ]

    summary = {
        "messages":
            int(total)
    }

    for field in fields:

        count = int(
            dataframe[
                field
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        summary[
            field
        ] = {
            "count":
                count,

            "percentage":
                float(
                    count
                    / total
                    * 100
                ),
        }

    parsed_fields = [
        "spf_result",
        "received_spf_result",
        "dkim_result",
        "dmarc_result",
        "dkim_domains",
    ]

    for field in parsed_fields:

        count = int(
            dataframe[
                field
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        summary[
            f"parsed_{field}"
        ] = {
            "count":
                count,

            "percentage":
                float(
                    count
                    / total
                    * 100
                ),
        }

    return summary


def print_coverage_summary(
    summary: dict,
) -> None:
    """
    Print header availability statistics.
    """

    total = summary.get(
        "messages",
        0,
    )

    print(
        "\nHeader coverage summary"
    )

    print(
        "-" * 60
    )

    print(
        f"Selected messages: "
        f"{total:,}"
    )

    if total == 0:

        print(
            "No matching messages found."
        )

        print(
            "-" * 60
        )

        return

    labels = {
        "has_authentication_results":
            "Authentication-Results",

        "has_arc_authentication_results":
            "ARC-Authentication-Results",

        "has_received_spf":
            "Received-SPF",

        "has_dkim_signature":
            "DKIM-Signature",

        "has_return_path":
            "Return-Path",

        "has_reply_to":
            "Reply-To",

        "parsed_spf_result":
            "Parsed SPF result",

        "parsed_received_spf_result":
            "Parsed Received-SPF",

        "parsed_dkim_result":
            "Parsed DKIM result",

        "parsed_dmarc_result":
            "Parsed DMARC result",

        "parsed_dkim_domains":
            "Parsed DKIM domain",
    }

    for key, label in labels.items():

        value = summary.get(
            key
        )

        if not value:
            continue

        print(
            f"{label:<30}"
            f"{value['count']:>5} / "
            f"{total:<5} "
            f"({value['percentage']:>6.2f}%)"
        )

    print(
        "-" * 60
    )


def sanitize_tag(
    tag: str,
) -> str:
    """
    Sanitize the output tag.
    """

    tag = sanitize_filename(
        tag
    )

    if not tag:
        return "inspection"

    return tag


def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Inspect authentication and sender headers "
            "in Thunderbird MBOX files."
        )
    )

    parser.add_argument(
        "mbox_paths",
        nargs="+",
        help=(
            "One or more Thunderbird MBOX file paths."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help=(
            "Maximum number of selected messages per MBOX. "
            "Default: 20."
        ),
    )

    parser.add_argument(
        "--contains",
        default=None,
        help=(
            "Select messages whose complete header block "
            "contains this text."
        ),
    )

    parser.add_argument(
        "--sender-contains",
        default=None,
        help=(
            "Select messages whose From header contains "
            "this text."
        ),
    )

    parser.add_argument(
        "--subject-contains",
        default=None,
        help=(
            "Select messages whose Subject contains this text."
        ),
    )

    parser.add_argument(
        "--tag",
        default="inspection",
        help=(
            "Tag used in output filenames."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Run email-header inspection.
    """

    arguments = (
        parse_arguments()
    )

    if arguments.limit < 1:

        raise ValueError(
            "--limit must be at least 1."
        )

    tag = (
        sanitize_tag(
            arguments.tag
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    for path_value in arguments.mbox_paths:

        mbox_path = (
            Path(
                path_value
            )
            .expanduser()
        )

        rows = (
            inspect_mbox(
                mbox_path=mbox_path,
                limit=arguments.limit,
                contains=arguments.contains,
                sender_contains=arguments.sender_contains,
                subject_contains=arguments.subject_contains,
                tag=tag,
            )
        )

        all_rows.extend(
            rows
        )

    dataframe = pd.DataFrame(
        all_rows
    )

    csv_path = (
        OUTPUT_DIR
        / f"header_sample_{tag}.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / f"header_coverage_{tag}.json"
    )

    dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        create_coverage_summary(
            dataframe
        )
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

    print_coverage_summary(
        summary
    )

    print(
        "\nCSV output:"
    )

    print(
        csv_path
    )

    print(
        "\nCoverage summary:"
    )

    print(
        summary_path
    )

    print(
        "\nIndividual raw header files:"
    )

    print(
        RAW_HEADERS_DIR
    )

    print(
        "\nHeader inspection finished successfully."
    )


if __name__ == "__main__":
    main()