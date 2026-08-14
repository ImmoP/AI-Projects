"""Inspect the original Hugging Face dataset splits."""

import pandas as pd

from spam_detector.paths import DATA_DIR


def main():
    hf_train = pd.read_parquet(DATA_DIR / "train.parquet")
    hf_eval = pd.read_parquet(DATA_DIR / "eval.parquet")
    hf_test = pd.read_parquet(DATA_DIR / "test.parquet")

    print("Train:", hf_train.shape)
    print("Evaluation:", hf_eval.shape)
    print("Test:", hf_test.shape)
    print("\nTrain columns:")
    print(hf_train.columns.tolist())
    print("\nData types:")
    print(hf_train.dtypes)
    print("\nSame columns:")
    print("Train vs Eval:", hf_train.columns.tolist() == hf_eval.columns.tolist())
    print("Train vs Test:", hf_train.columns.tolist() == hf_test.columns.tolist())

    for split_name, dataframe in [
        ("Train", hf_train),
        ("Evaluation", hf_eval),
        ("Test", hf_test),
    ]:
        print(f"\n{split_name} labels:")
        print(dataframe["label"].value_counts(dropna=False).sort_index())
        print("\nProportions:")
        print(
            dataframe["label"]
            .value_counts(normalize=True, dropna=False)
            .sort_index()
        )

    core_columns = ["sender", "subject", "text", "date", "label"]
    splits = [
        ("Train", hf_train),
        ("Evaluation", hf_eval),
        ("Test", hf_test),
    ]

    for split_name, dataframe in splits:
        print(f"\nMissing values — {split_name}")
        for column in core_columns:
            if column == "label" or column == "date":
                missing = dataframe[column].isna().sum()
            else:
                missing = (
                    dataframe[column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq("")
                    .sum()
                )
            print(f"{column}: {missing}")

    for split_name, dataframe in splits:
        print(f"\nOriginal datasets — {split_name}")
        print(dataframe["dataset_name"].value_counts(dropna=False))

    print("\nSender examples:")
    print(
        hf_train["sender"]
        .dropna()
        .sample(20, random_state=42)
        .to_string(index=False)
    )

    examples = hf_train.sample(5, random_state=42)
    for _, row in examples.iterrows():
        print("\n" + "=" * 80)
        print("Sender:")
        print(row["sender"])
        print("\nSubject:")
        print(row["subject"])
        print("\nLabel:")
        print(row["label"])
        print("\nDataset:")
        print(row["dataset_name"])
        print("\nText:")
        print(str(row["text"])[:1000])

    for split_name, dataframe in splits:
        missing_sender = (
            dataframe["sender"].fillna("").astype(str).str.strip().eq("")
        )
        result = (
            dataframe.assign(sender_missing=missing_sender)
            .groupby("dataset_name")["sender_missing"]
            .agg(["sum", "count"])
        )
        result["percentage"] = result["sum"] / result["count"] * 100
        print(f"\nMissing senders by dataset — {split_name}")
        print(result.sort_values("percentage", ascending=False))

    for split_name, dataframe in splits:
        missing_sender = (
            dataframe["sender"].fillna("").astype(str).str.strip().eq("")
        )
        result = (
            dataframe.assign(sender_missing=missing_sender)
            .groupby("label")["sender_missing"]
            .agg(["sum", "count"])
        )
        result["percentage"] = result["sum"] / result["count"] * 100
        print(f"\nMissing senders by label — {split_name}")
        print(result)

    for column in ["sender", "subject", "text", "date"]:
        print(f"\nMissing {column} by dataset — Train")
        if column == "date":
            missing = hf_train[column].isna()
        else:
            missing = (
                hf_train[column].fillna("").astype(str).str.strip().eq("")
            )
        result = (
            hf_train.assign(is_missing=missing)
            .groupby("dataset_name")["is_missing"]
            .agg(["sum", "count"])
        )
        result["percentage"] = result["sum"] / result["count"] * 100
        print(result.sort_values("percentage", ascending=False))

    for split_name, dataframe in splits:
        print(f"\nLabels by dataset — {split_name}")
        result = pd.crosstab(
            dataframe["dataset_name"],
            dataframe["label"],
            normalize="index",
        )
        print(result)


if __name__ == "__main__":
    main()
