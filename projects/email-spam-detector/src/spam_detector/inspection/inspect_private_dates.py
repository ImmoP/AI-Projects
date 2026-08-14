"""Inspect the normalized date range of the private dataset."""

import pandas as pd

from spam_detector.paths import DATA_DIR


def main():
    private_df = pd.read_parquet(DATA_DIR / "private_emails_parsed.parquet")
    private_df["date"] = pd.to_datetime(
        private_df["date"],
        errors="coerce",
        utc=True,
    )

    print("Earliest date:")
    print(private_df["date"].min())
    print("\nLatest date:")
    print(private_df["date"].max())
    print("\nMissing dates:")
    print(private_df["date"].isna().sum())

    private_df["year"] = private_df["date"].dt.year
    counts_by_year = (
        private_df.groupby(["year", "label"]).size().unstack(fill_value=0)
    )
    print("\nEmails by year and label:")
    print(counts_by_year)


if __name__ == "__main__":
    main()
