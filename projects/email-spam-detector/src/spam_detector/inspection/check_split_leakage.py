"""Report duplicate content and label conflicts across combined splits."""

import pandas as pd

from spam_detector.paths import DATA_DIR


def create_email_key(dataframe):
    dataframe = dataframe.copy()

    for column in ["sender", "subject", "text"]:
        dataframe[column] = (
            dataframe[column]
            .fillna("")
            .astype("string")
            .str.strip()
            .str.lower()
        )

    dataframe["email_key"] = (
        dataframe["sender"]
        + "\n"
        + dataframe["subject"]
        + "\n"
        + dataframe["text"]
    )
    return dataframe


def main():
    train = create_email_key(
        pd.read_parquet(DATA_DIR / "combined_train_clean.parquet")
    )
    evaluation = create_email_key(
        pd.read_parquet(DATA_DIR / "combined_eval_clean.parquet")
    )
    test = create_email_key(pd.read_parquet(DATA_DIR / "combined_test_clean.parquet"))

    train_keys = set(train["email_key"])
    eval_keys = set(evaluation["email_key"])
    test_keys = set(test["email_key"])

    print("Train ↔ Evaluation:", len(train_keys & eval_keys))
    print("Train ↔ Test:", len(train_keys & test_keys))
    print("Evaluation ↔ Test:", len(eval_keys & test_keys))

    for name, dataframe in [
        ("Train", train),
        ("Evaluation", evaluation),
        ("Test", test),
    ]:
        duplicates = dataframe["email_key"].duplicated().sum()
        print(f"{name} duplicates:", duplicates)

    all_data = pd.concat([train, evaluation, test], ignore_index=True)
    label_counts = all_data.groupby("email_key")["label"].nunique()
    print("Conflicting labels:", (label_counts > 1).sum())


if __name__ == "__main__":
    main()
