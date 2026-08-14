from email.utils import parseaddr
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "private_train.parquet"
)

ERRORS_PATH = (
    PROJECT_ROOT
    / "data"
    / "private_error_analysis"
    / "genuine_model_errors.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "private_error_analysis"
    / "genuine_errors_with_sender_reputation.csv"
)


def extract_email_address(sender):
    """
    Extract the normalized email address from a sender field.
    """

    if pd.isna(sender):
        return ""

    _, email_address = parseaddr(
        str(sender)
    )

    return email_address.strip().lower()


def extract_domain(email_address):
    """
    Extract the domain from an email address.
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
    )


def calculate_reputation_table(
    dataframe,
    group_column,
):
    """
    Calculate historical ham/spam statistics for a sender
    or domain using the private training data only.
    """

    reputation = (
        dataframe
        .groupby(
            group_column,
            dropna=False,
        )
        .agg(
            train_count=(
                "label",
                "size",
            ),
            train_spam_count=(
                "label",
                "sum",
            ),
        )
        .reset_index()
    )

    reputation["train_ham_count"] = (
        reputation["train_count"]
        - reputation["train_spam_count"]
    )

    reputation["train_spam_rate"] = (
        reputation["train_spam_count"]
        / reputation["train_count"]
    )

    return reputation


def main():
    print("Loading private training data...")

    train = pd.read_parquet(
        TRAIN_PATH
    )

    print(
        f"Training examples: {len(train):,}"
    )

    if "sender" not in train.columns:
        raise ValueError(
            "private_train.parquet does not contain a sender column."
        )

    if "label" not in train.columns:
        raise ValueError(
            "private_train.parquet does not contain a label column."
        )

    train["label"] = pd.to_numeric(
        train["label"],
        errors="raise",
    ).astype(int)

    train["sender_email"] = (
        train["sender"]
        .apply(
            extract_email_address
        )
    )

    train["sender_domain"] = (
        train["sender_email"]
        .apply(
            extract_domain
        )
    )

    print(
        f"Ham:  {(train['label'] == 0).sum():,}"
    )

    print(
        f"Spam: {(train['label'] == 1).sum():,}"
    )

    sender_reputation = (
        calculate_reputation_table(
            train,
            "sender_email",
        )
    )

    sender_reputation = (
        sender_reputation
        .rename(
            columns={
                "train_count":
                    "sender_train_count",

                "train_spam_count":
                    "sender_train_spam_count",

                "train_ham_count":
                    "sender_train_ham_count",

                "train_spam_rate":
                    "sender_train_spam_rate",
            }
        )
    )

    domain_reputation = (
        calculate_reputation_table(
            train,
            "sender_domain",
        )
    )

    domain_reputation = (
        domain_reputation
        .rename(
            columns={
                "train_count":
                    "domain_train_count",

                "train_spam_count":
                    "domain_train_spam_count",

                "train_ham_count":
                    "domain_train_ham_count",

                "train_spam_rate":
                    "domain_train_spam_rate",
            }
        )
    )

    print()
    print("Loading genuine model errors...")

    errors = pd.read_csv(
        ERRORS_PATH
    )

    errors["sender_email"] = (
        errors["sender"]
        .apply(
            extract_email_address
        )
    )

    errors["sender_domain"] = (
        errors["sender_email"]
        .apply(
            extract_domain
        )
    )

    errors = errors.merge(
        sender_reputation,
        on="sender_email",
        how="left",
    )

    errors = errors.merge(
        domain_reputation,
        on="sender_domain",
        how="left",
    )

    count_columns = [
        "sender_train_count",
        "sender_train_spam_count",
        "sender_train_ham_count",
        "domain_train_count",
        "domain_train_spam_count",
        "domain_train_ham_count",
    ]

    for column in count_columns:
        errors[column] = (
            errors[column]
            .fillna(0)
            .astype(int)
        )

    errors.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "row_id",
        "adjudicated_error_type",
        "spam_probability",
        "sender_email",
        "sender_domain",
        "sender_train_count",
        "sender_train_ham_count",
        "sender_train_spam_count",
        "sender_train_spam_rate",
        "domain_train_count",
        "domain_train_ham_count",
        "domain_train_spam_count",
        "domain_train_spam_rate",
        "subject",
    ]

    print()
    print("SENDER REPUTATION FOR GENUINE MODEL ERRORS")
    print()

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        250,
    )

    pd.set_option(
        "display.max_colwidth",
        70,
    )

    print(
        errors[
            display_columns
        ].to_string(
            index=False
        )
    )

    known_sender = (
        errors["sender_train_count"]
        > 0
    ).sum()

    known_domain = (
        errors["domain_train_count"]
        > 0
    ).sum()

    print()
    print("SUMMARY")
    print()

    print(
        f"Genuine errors: {len(errors)}"
    )

    print(
        f"Exact sender already present in private training: "
        f"{known_sender}/{len(errors)}"
    )

    print(
        f"Sender domain already present in private training: "
        f"{known_domain}/{len(errors)}"
    )

    print()
    print(
        f"Saved:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()