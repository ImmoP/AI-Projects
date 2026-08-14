"""Normalize the existing Hugging Face dataset splits."""

import pandas as pd

from spam_detector.paths import DATA_DIR


def prepare_hf_split(dataframe, split_name):
    prepared = dataframe[
        ["sender", "subject", "text", "date", "label", "dataset_name"]
    ].copy()
    prepared = prepared.rename(columns={"dataset_name": "source"})
    prepared["source_split"] = split_name

    for column in ["sender", "subject", "text"]:
        prepared[column] = (
            prepared[column].fillna("").astype("string").str.strip()
        )

    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["label"] = prepared["label"].astype("Int64")

    return prepared


def main():
    hf_train = pd.read_parquet(DATA_DIR / "train.parquet")
    hf_eval = pd.read_parquet(DATA_DIR / "eval.parquet")
    hf_test = pd.read_parquet(DATA_DIR / "test.parquet")

    prepared_train = prepare_hf_split(hf_train, "train")
    prepared_eval = prepare_hf_split(hf_eval, "eval")
    prepared_test = prepare_hf_split(hf_test, "test")

    for split_name, dataframe in [
        ("Train", prepared_train),
        ("Evaluation", prepared_eval),
        ("Test", prepared_test),
    ]:
        print(f"\n{split_name}")
        print("Shape:", dataframe.shape)
        print("Columns:", dataframe.columns.tolist())
        print("\nLabels:")
        print(dataframe["label"].value_counts())
        print("\nSources:")
        print(dataframe["source"].value_counts())

    prepared_train.to_parquet(DATA_DIR / "hf_train_prepared.parquet", index=False)
    prepared_eval.to_parquet(DATA_DIR / "hf_eval_prepared.parquet", index=False)
    prepared_test.to_parquet(DATA_DIR / "hf_test_prepared.parquet", index=False)


if __name__ == "__main__":
    main()
