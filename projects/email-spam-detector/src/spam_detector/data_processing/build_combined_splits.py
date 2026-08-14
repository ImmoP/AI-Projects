"""Combine corresponding public and private dataset splits."""

import pandas as pd

from spam_detector.paths import DATA_DIR


def normalize_dataframe(dataframe):
    dataframe = dataframe.copy()
    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
        format="mixed",
        utc=True,
    )
    dataframe["label"] = dataframe["label"].astype("Int64")

    for column in ["sender", "subject", "text", "source", "source_split"]:
        dataframe[column] = dataframe[column].fillna("").astype("string")

    return dataframe


def main():
    hf_train = pd.read_parquet(DATA_DIR / "hf_train_prepared.parquet")
    hf_eval = pd.read_parquet(DATA_DIR / "hf_eval_prepared.parquet")
    hf_test = pd.read_parquet(DATA_DIR / "hf_test_prepared.parquet")
    private_train = pd.read_parquet(DATA_DIR / "private_train.parquet")
    private_eval = pd.read_parquet(DATA_DIR / "private_eval.parquet")
    private_test = pd.read_parquet(DATA_DIR / "private_test.parquet")

    hf_train = normalize_dataframe(hf_train)
    hf_eval = normalize_dataframe(hf_eval)
    hf_test = normalize_dataframe(hf_test)
    private_train = normalize_dataframe(private_train)
    private_eval = normalize_dataframe(private_eval)
    private_test = normalize_dataframe(private_test)

    combined_train = pd.concat([hf_train, private_train], ignore_index=True)
    combined_eval = pd.concat([hf_eval, private_eval], ignore_index=True)
    combined_test = pd.concat([hf_test, private_test], ignore_index=True)

    for split_name, dataframe in [
        ("Train", combined_train),
        ("Evaluation", combined_eval),
        ("Test", combined_test),
    ]:
        print(f"\n{split_name}")
        print("Shape:", dataframe.shape)
        print("\nData types:")
        print(dataframe.dtypes)
        print("\nLabels:")
        print(dataframe["label"].value_counts())
        print("\nSource split:")
        print(dataframe["source_split"].value_counts())
        print("\nSources:")
        print(dataframe["source"].value_counts())

    combined_train.to_parquet(DATA_DIR / "combined_train.parquet", index=False)
    combined_eval.to_parquet(DATA_DIR / "combined_eval.parquet", index=False)
    combined_test.to_parquet(DATA_DIR / "combined_test.parquet", index=False)
    print("\nCombined datasets saved successfully.")


if __name__ == "__main__":
    main()
