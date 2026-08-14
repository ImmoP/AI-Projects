"""Inspect private emails with empty message bodies."""

import pandas as pd

from spam_detector.paths import DATA_DIR


def main():
    private_df = pd.read_parquet(DATA_DIR / "private_emails_parsed.parquet")
    empty_text_mask = (
        private_df["text"].fillna("").astype(str).str.strip().eq("")
    )
    empty_emails = private_df[empty_text_mask]

    print("E-Mails ohne Body:", len(empty_emails))
    print(
        empty_emails[["sender", "subject", "date", "label", "source"]]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
