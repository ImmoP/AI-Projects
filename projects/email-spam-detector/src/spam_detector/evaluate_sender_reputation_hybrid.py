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
    / "sender_reputation_hybrid_results.csv"
)


# Domains that represent shared infrastructure rather than
# a single organization.
#
# Domain-level reputation should not be used for these domains
# because unrelated services may share the same domain.
#
# Exact sender reputation is still allowed.

SHARED_RELAY_DOMAINS = {
    "privaterelay.appleid.com",
}


def extract_email_address(sender):
    """
    Extract and normalize the sender email address.
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
        .rstrip(".")
    )


def is_shared_relay_domain(domain):
    """
    Determine whether a domain belongs to shared relay
    infrastructure.

    Subdomains are included as well.
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


def build_reputation_table(
    dataframe,
    group_column,
):
    """
    Build ham/spam reputation statistics using
    private training data only.
    """

    table = (
        dataframe
        .groupby(
            group_column,
            dropna=False,
        )
        .agg(
            count=(
                "label",
                "size",
            ),
            spam_count=(
                "label",
                "sum",
            ),
        )
        .reset_index()
    )

    table["ham_count"] = (
        table["count"]
        - table["spam_count"]
    )

    table["spam_rate"] = (
        table["spam_count"]
        / table["count"]
    )

    return table


def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Calculate binary classification metrics.
    """

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )
        .ravel()
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_metrics(
    name,
    metrics,
):
    """
    Print evaluation metrics.
    """

    print()
    print(name)
    print("-" * len(name))

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


def apply_hybrid_rule(
    row,
    min_count,
    use_domain=True,
):
    """
    Apply sender reputation as a conservative override.

    Priority:

    1. Start with the GPT-2 prediction.
    2. Do not use reputation for shared relay infrastructure.
    3. Prefer exact sender reputation for normal senders.
    4. Require unanimous historical labels.
    5. Require at least min_count historical examples.
    6. Optionally use domain reputation.
    """

    original_prediction = int(
        row["predicted_label"]
    )

    domain = row["sender_domain"]

    # Shared relay infrastructure should not use either
    # exact-sender or domain reputation.
    #
    # The historical labels may reflect how the receiving
    # mailbox classified messages from individual apps rather
    # than the actual trustworthiness of the relay identity.
    #
    # Example:
    # Example Sender
    # relay-user@example.com

    if is_shared_relay_domain(
        domain
    ):
        return (
            original_prediction,
            "gpt_shared_relay",
        )

    sender_count = int(
        row["sender_rep_count"]
    )

    sender_spam_rate = (
        row["sender_rep_spam_rate"]
    )

    # Exact sender reputation

    if sender_count >= min_count:

        if sender_spam_rate == 0.0:
            return 0, "sender_ham"

        if sender_spam_rate == 1.0:
            return 1, "sender_spam"

    # Domain reputation

    if use_domain:

        domain_count = int(
            row["domain_rep_count"]
        )

        domain_spam_rate = (
            row["domain_rep_spam_rate"]
        )

        if domain_count >= min_count:

            if domain_spam_rate == 0.0:
                return 0, "domain_ham"

            if domain_spam_rate == 1.0:
                return 1, "domain_spam"

    return (
        original_prediction,
        "gpt",
    )


def main():
    print(
        "Loading private training data..."
    )

    train = pd.read_parquet(
        TRAIN_PATH
    )

    train["label"] = (
        pd.to_numeric(
            train["label"],
            errors="raise",
        )
        .astype(int)
    )

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

    train["is_shared_relay_domain"] = (
        train["sender_domain"]
        .apply(
            is_shared_relay_domain
        )
    )

    sender_reputation = (
        build_reputation_table(
            train,
            "sender_email",
        )
        .rename(
            columns={
                "count":
                    "sender_rep_count",

                "spam_count":
                    "sender_rep_spam_count",

                "ham_count":
                    "sender_rep_ham_count",

                "spam_rate":
                    "sender_rep_spam_rate",
            }
        )
    )

    domain_reputation = (
        build_reputation_table(
            train,
            "sender_domain",
        )
        .rename(
            columns={
                "count":
                    "domain_rep_count",

                "spam_count":
                    "domain_rep_spam_count",

                "ham_count":
                    "domain_rep_ham_count",

                "spam_rate":
                    "domain_rep_spam_rate",
            }
        )
    )

    print(
        "Loading adjudicated private validation data..."
    )

    df = pd.read_csv(
        EVAL_PATH
    )

    df["sender_email"] = (
        df["sender"]
        .apply(
            extract_email_address
        )
    )

    df["sender_domain"] = (
        df["sender_email"]
        .apply(
            extract_domain
        )
    )

    df["is_shared_relay_domain"] = (
        df["sender_domain"]
        .apply(
            is_shared_relay_domain
        )
    )

    df = df.merge(
        sender_reputation,
        on="sender_email",
        how="left",
    )

    df = df.merge(
        domain_reputation,
        on="sender_domain",
        how="left",
    )

    count_columns = [
        "sender_rep_count",
        "sender_rep_spam_count",
        "sender_rep_ham_count",
        "domain_rep_count",
        "domain_rep_spam_count",
        "domain_rep_ham_count",
    ]

    for column in count_columns:

        df[column] = (
            df[column]
            .fillna(0)
            .astype(int)
        )

    df["predicted_label"] = (
        pd.to_numeric(
            df["predicted_label"],
            errors="raise",
        )
        .astype(int)
    )

    df["corrected_label"] = (
        pd.to_numeric(
            df["corrected_label"],
            errors="raise",
        )
        .astype(int)
    )

    y_true = (
        df["corrected_label"]
        .to_numpy()
    )

    gpt_predictions = (
        df["predicted_label"]
        .to_numpy()
    )

    baseline_metrics = (
        calculate_metrics(
            y_true,
            gpt_predictions,
        )
    )

    print_metrics(
        "GPT-2 BASELINE",
        baseline_metrics,
    )

    relay_count = int(
        df[
            "is_shared_relay_domain"
        ].sum()
    )

    print()
    print(
        "Shared relay domains excluded from "
        "domain reputation:"
    )

    for domain in sorted(
        SHARED_RELAY_DOMAINS
    ):
        print(
            f"  {domain}"
        )

    print(
        f"\nValidation emails using shared relay "
        f"domains: {relay_count}"
    )

    results = []

    for min_count in [
        1,
        2,
        3,
        5,
        10,
    ]:

        for use_domain in [
            False,
            True,
        ]:

            predictions = []

            sources = []

            for _, row in df.iterrows():

                prediction, source = (
                    apply_hybrid_rule(
                        row=row,
                        min_count=min_count,
                        use_domain=use_domain,
                    )
                )

                predictions.append(
                    prediction
                )

                sources.append(
                    source
                )

            prediction_column = (
                f"hybrid_min_{min_count}_"
                f"{'sender_domain' if use_domain else 'sender_only'}"
            )

            source_column = (
                prediction_column
                + "_source"
            )

            df[
                prediction_column
            ] = predictions

            df[
                source_column
            ] = sources

            metrics = (
                calculate_metrics(
                    y_true,
                    predictions,
                )
            )

            overrides = sum(
                prediction
                != original
                for prediction, original
                in zip(
                    predictions,
                    gpt_predictions,
                )
            )

            corrected_gpt_errors = sum(
                (
                    original != truth
                    and prediction == truth
                )
                for (
                    original,
                    prediction,
                    truth,
                )
                in zip(
                    gpt_predictions,
                    predictions,
                    y_true,
                )
            )

            new_errors = sum(
                (
                    original == truth
                    and prediction != truth
                )
                for (
                    original,
                    prediction,
                    truth,
                )
                in zip(
                    gpt_predictions,
                    predictions,
                    y_true,
                )
            )

            results.append(
                {
                    "min_count":
                        min_count,

                    "use_domain":
                        use_domain,

                    "accuracy":
                        metrics["accuracy"],

                    "precision":
                        metrics["precision"],

                    "recall":
                        metrics["recall"],

                    "f1":
                        metrics["f1"],

                    "tn":
                        metrics["tn"],

                    "fp":
                        metrics["fp"],

                    "fn":
                        metrics["fn"],

                    "tp":
                        metrics["tp"],

                    "overrides":
                        overrides,

                    "corrected_gpt_errors":
                        corrected_gpt_errors,

                    "new_errors_created":
                        new_errors,
                }
            )

            name = (
                f"HYBRID | min_count={min_count} | "
                f"{'sender + domain' if use_domain else 'sender only'}"
            )

            print_metrics(
                name,
                metrics,
            )

            print(
                f"Overrides: {overrides}"
            )

            print(
                "GPT errors corrected: "
                f"{corrected_gpt_errors}"
            )

            print(
                "New errors created: "
                f"{new_errors}"
            )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by=[
                "f1",
                "new_errors_created",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    print()
    print("HYBRID COMPARISON")
    print()

    print(
        results_df.to_string(
            index=False,
        )
    )

    print()
    print("BEST VALIDATION CONFIGURATION")
    print()

    best = (
        results_df
        .iloc[0]
    )

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

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Detailed results saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
