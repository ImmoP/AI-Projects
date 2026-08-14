"""Remove duplicate email content with test > evaluation > train priority."""

import pandas as pd

from spam_detector.paths import DATA_DIR


def create_email_key(dataframe):
    dataframe = dataframe.copy()
    normalized_parts = []

    for column in ["sender", "subject", "text"]:
        normalized = (
            dataframe[column]
            .fillna("")
            .astype("string")
            .str.strip()
            .str.lower()
        )
        normalized_parts.append(normalized)

    dataframe["email_key"] = (
        normalized_parts[0]
        + "\n"
        + normalized_parts[1]
        + "\n"
        + normalized_parts[2]
    )
    return dataframe


def main():
    train = create_email_key(
        pd.read_parquet(DATA_DIR / "combined_train.parquet")
    )
    evaluation = create_email_key(
        pd.read_parquet(DATA_DIR / "combined_eval.parquet")
    )
    test = create_email_key(pd.read_parquet(DATA_DIR / "combined_test.parquet"))

    print("Before internal deduplication:")
    print("Train:", len(train))
    print("Evaluation:", len(evaluation))
    print("Test:", len(test))

    train = train.drop_duplicates(subset="email_key", keep="first").copy()
    evaluation = evaluation.drop_duplicates(subset="email_key", keep="first").copy()
    test = test.drop_duplicates(subset="email_key", keep="first").copy()

    print("\nAfter internal deduplication:")
    print("Train:", len(train))
    print("Evaluation:", len(evaluation))
    print("Test:", len(test))

    # Protect the test set, then the evaluation set.
    test_keys = set(test["email_key"])
    evaluation = evaluation[~evaluation["email_key"].isin(test_keys)].copy()
    train = train[~train["email_key"].isin(test_keys)].copy()

    evaluation_keys = set(evaluation["email_key"])
    train = train[~train["email_key"].isin(evaluation_keys)].copy()

    train_keys = set(train["email_key"])
    evaluation_keys = set(evaluation["email_key"])
    test_keys = set(test["email_key"])

    print("\nRemaining overlap:")
    print("Train ↔ Evaluation:", len(train_keys & evaluation_keys))
    print("Train ↔ Test:", len(train_keys & test_keys))
    print("Evaluation ↔ Test:", len(evaluation_keys & test_keys))
    print("\nFinal sizes:")
    print("Train:", len(train))
    print("Evaluation:", len(evaluation))
    print("Test:", len(test))

    train = train.drop(columns="email_key")
    evaluation = evaluation.drop(columns="email_key")
    test = test.drop(columns="email_key")

    train.to_parquet(DATA_DIR / "combined_train_clean.parquet", index=False)
    evaluation.to_parquet(DATA_DIR / "combined_eval_clean.parquet", index=False)
    test.to_parquet(DATA_DIR / "combined_test_clean.parquet", index=False)
    print("\nClean datasets saved successfully.")


if __name__ == "__main__":
    main()
