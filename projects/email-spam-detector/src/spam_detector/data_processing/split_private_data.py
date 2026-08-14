"""Deduplicate and stratify the private email dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split

from spam_detector.paths import DATA_DIR


def main():
    private_df = pd.read_parquet(DATA_DIR / "private_emails_parsed.parquet")

    print("Complete duplicate rows:", private_df.duplicated().sum())
    print(
        "Duplicate email content:",
        private_df.duplicated(subset=["sender", "subject", "text"]).sum(),
    )

    label_counts = private_df.groupby(
        ["sender", "subject", "text"],
        dropna=False,
    )["label"].nunique()
    print("Emails with conflicting labels:", (label_counts > 1).sum())

    duplicate_columns = ["sender", "subject", "text"]
    private_unique_df = private_df.drop_duplicates(
        subset=duplicate_columns,
        keep="first",
    ).copy()

    print("Before:", len(private_df))
    print("After:", len(private_unique_df))
    print("Removed:", len(private_df) - len(private_unique_df))
    print("\nLabels after deduplication:")
    print(private_unique_df["label"].value_counts())

    private_train, private_remaining = train_test_split(
        private_unique_df,
        test_size=0.20,
        random_state=42,
        stratify=private_unique_df["label"],
    )
    private_eval, private_test = train_test_split(
        private_remaining,
        test_size=0.50,
        random_state=42,
        stratify=private_remaining["label"],
    )

    private_train = private_train.copy()
    private_eval = private_eval.copy()
    private_test = private_test.copy()

    private_train["source_split"] = "train"
    private_eval["source_split"] = "eval"
    private_test["source_split"] = "test"

    for name, dataframe in [
        ("Train", private_train),
        ("Evaluation", private_eval),
        ("Test", private_test),
    ]:
        print(f"\n{name}: {dataframe.shape}")
        print(dataframe["label"].value_counts())

    private_train.to_parquet(DATA_DIR / "private_train.parquet", index=False)
    private_eval.to_parquet(DATA_DIR / "private_eval.parquet", index=False)
    private_test.to_parquet(DATA_DIR / "private_test.parquet", index=False)


if __name__ == "__main__":
    main()
