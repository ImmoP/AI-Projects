"""Parse the private mailbox exports into a common Parquet schema."""

import argparse
import mailbox
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pandas as pd

from spam_detector.data_processing.extract import extract_mailbox
from spam_detector.paths import DATA_DIR
from spam_detector.private_mailbox_config import (
    DEFAULT_PRIVATE_MAILBOX_CONFIG,
    load_private_mailboxes,
)


def parse_email(file_object):
    """Parse a message from a binary mbox stream."""
    return BytesParser(policy=policy.default).parse(file_object)


def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Parse configured private mailbox exports."
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
    arguments = parse_arguments()
    mailbox_definitions = load_private_mailboxes(arguments.config)

    opened_mailboxes = [
        (
            definition,
            mailbox.mbox(
                definition.path,
                factory=parse_email,
                create=False,
            ),
        )
        for definition in mailbox_definitions
    ]

    records_by_mailbox = []

    try:
        for definition, opened_mailbox in opened_mailboxes:
            records_by_mailbox.append(
                extract_mailbox(
                    mbox=opened_mailbox,
                    label=definition.numeric_label,
                    source=definition.source,
                )
            )
    finally:
        for _, opened_mailbox in opened_mailboxes:
            opened_mailbox.close()

    for index, records in enumerate(records_by_mailbox, start=1):
        print(f"Mailbox {index}:", len(records))

    private_records = [
        record
        for records in records_by_mailbox
        for record in records
    ]
    private_df = pd.DataFrame(private_records)

    for column in ["sender", "subject", "text", "date"]:
        missing = (
            private_df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        print(f"\nFehlend: {column}")
        print(
            private_df.assign(is_missing=missing)
            .groupby("source")["is_missing"]
            .sum()
        )

    output_path = DATA_DIR / "private_emails_parsed.parquet"
    private_df.to_parquet(output_path, index=False)

    print("Saved:", output_path)
    print(private_df.shape)
    print(private_df.columns.tolist())
    print(private_df["label"].value_counts())
    print(private_df["source"].value_counts())


if __name__ == "__main__":
    main()
