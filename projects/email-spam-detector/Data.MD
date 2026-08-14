# Data pipeline reference

All inputs and generated datasets remain in the root-level `data/` directory.
The package resolves that location through `spam_detector.paths.DATA_DIR`.
Importing the path module or any processing module does not read data, open a
mailbox, write files, or print output.

## Common schema

Prepared private and public data use these columns:

```text
sender
subject
text
date
label
source
source_split
```

Labels retain their existing meaning: `0` is non-spam and `1` is spam.

## Private email extraction

`spam_detector.data_processing.extract` converts email messages to records. It:

- decodes MIME-encoded sender and subject headers;
- skips attachments and multipart containers;
- prefers plain text and falls back to text extracted from HTML;
- parses valid message dates to ISO format;
- records the supplied label and source plus `source_split="private"`.

`spam_detector.data_processing.prepare_private_data` opens three local private
mailbox exports (one inbox and two spam folders) with a binary email parser,
applies the established labels and source names, and writes
`private_emails_parsed.parquet`. The mailbox exports, their local filenames,
and all derived row-level datasets are private and excluded from Git.

Mailbox definitions come from the ignored
`config/private_mailboxes.local.toml` file; the tracked
`config/private_mailboxes.example.toml` contains placeholders only. Each
ordered entry defines `path`, semantic `label`, and `source` independently.
Labels are never inferred from filenames or source names, and entry order must
be preserved. Private preparation and security-feature extraction must use the
same configuration. Use `--config path/to/private_mailboxes.local.toml` to
override the default local-config path.

## Preparation and splitting

`spam_detector.data_processing.prepare_hf_data` preserves the existing public
train/evaluation/test assignments, maps `dataset_name` to `source`, normalizes
the selected fields, and writes:

```text
hf_train_prepared.parquet
hf_eval_prepared.parquet
hf_test_prepared.parquet
```

`spam_detector.data_processing.split_private_data` deduplicates private messages
by `sender`, `subject`, and `text`, then performs the existing stratified split
with `random_state=42`. It writes:

```text
private_train.parquet
private_eval.parquet
private_test.parquet
```

## Combining and leakage cleaning

`spam_detector.data_processing.build_combined_splits` normalizes dates with
mixed-format parsing in UTC, combines corresponding public and private splits,
and writes `combined_train.parquet`, `combined_eval.parquet`, and
`combined_test.parquet`.

`spam_detector.data_processing.clean_split_leakage` creates a case-normalized
key from `sender`, `subject`, and `text`, removes duplicates within splits, and
removes cross-split overlap with priority `test > evaluation > train`. It writes:

```text
combined_train_clean.parquet
combined_eval_clean.parquet
combined_test_clean.parquet
```

## Inspection

Read-only inspection commands live under `spam_detector.inspection`:

- `inspect_private_data` reports private messages with empty bodies;
- `inspect_private_dates` reports UTC date coverage by year and label;
- `inspect_hf_data` summarizes the original public splits;
- `check_split_leakage` reports duplicate content and conflicting labels in the
  combined splits.

Run package modules with `python -m`; the complete commands are listed in
`README.md`.
