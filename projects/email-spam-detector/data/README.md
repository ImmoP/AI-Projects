# Data Pipeline Overview

The data pipeline prepares private Thunderbird emails and a public Hugging Face phishing-email dataset for training a spam classifier.

Only documentation belongs in this tracked directory. Raw mailbox exports,
private message-level data, prepared datasets, and generated row-level analysis
outputs are sensitive local artifacts and are excluded from Git.

## 1. Path Configuration

`paths.py` defines the project root and the location of the `data/` directory.

Important variables:

* `PROJECT_ROOT`: root directory of the project
* `DATA_DIR`: directory containing the raw and processed datasets

This ensures that all scripts use consistent file paths.

## 2. Private Email Extraction

`extract.py` contains reusable functions for processing Thunderbird Mbox messages.

### Main functions

* `decode_header_field()`: Decodes MIME-encoded email headers such as the sender and subject.
* `extract_text()`: Extracts the email body. Plain text is preferred, while HTML is converted to readable text as a fallback.
* `extract_mail()`: Converts one email into a structured dictionary.
* `extract_mailbox()`: Processes every message in an Mbox file.

Each extracted email contains:

```text
sender
subject
text
date
label
source
source_split
```

## 3. Private Data Preparation

`prepare_private_data.py` processes the private mailbox exports listed in the
ignored `config/private_mailboxes.local.toml` file. The tracked
`config/private_mailboxes.example.toml` contains placeholders only.

Each ordered mailbox entry supplies three deliberately separate values:

* `path`: local mailbox path
* `label`: explicit `ham` or `spam` semantic label
* `source`: stable private dataset source identifier

Labels are not inferred from paths, filenames, or source names. Entry order is
meaningful and must remain unchanged. Security-feature extraction consumes the
same configuration. Both commands use the default ignored local file:

```bash
python -m spam_detector.data_processing.prepare_private_data
python -m spam_detector.inspection.extract_security_features
```

Pass `--config path/to/private_mailboxes.local.toml` to either command to use a
different ignored local configuration.

The extracted emails are combined and saved as:

```text
private_emails_parsed.parquet
```

## 4. Public Dataset Preparation

`prepare_hf_data.py` loads the existing Hugging Face train, evaluation, and test files.

The original columns are converted to the common schema, and:

* `dataset_name` is renamed to `source`
* the original split is stored in `source_split`
* missing text fields are converted to empty strings
* dates and labels are normalized

The results are saved as:

```text
hf_train_prepared.parquet
hf_eval_prepared.parquet
hf_test_prepared.parquet
```

## 5. Private Data Splitting

`split_private_data.py` prepares the private emails for model development.

The script:

1. Removes duplicate emails using `sender`, `subject`, and `text`.
2. Checks whether identical emails have conflicting labels.
3. Creates stratified train, evaluation, and test splits.

The stratified split ensures that all three subsets contain both ham and spam emails.

The output files are:

```text
private_train.parquet
private_eval.parquet
private_test.parquet
```

## 6. Combining Public and Private Data

`build_combined_splits.py` combines corresponding public and private subsets:

```text
HF train + private train → combined train
HF evaluation + private evaluation → combined evaluation
HF test + private test → combined test
```

Before combining them, the script normalizes:

* string columns
* labels
* dates and time zones
* source information

The output files are:

```text
combined_train.parquet
combined_eval.parquet
combined_test.parquet
```

## 7. Leakage Detection

`check_split_leakage.py` checks whether identical emails occur in more than one split.

A normalized email key is created from:

```text
sender + subject + text
```

The script checks:

* duplicate emails inside each split
* overlap between train, evaluation, and test
* identical emails with conflicting labels

## 8. Leakage Removal

`clean_split_leakage.py` removes duplicate emails while protecting the evaluation sets.

The priority is:

```text
test > evaluation > train
```

For example, if the same email appears in both training and test data, it is kept in the test set and removed from the training set.

The final model-ready datasets are:

```text
combined_train_clean.parquet
combined_eval_clean.parquet
combined_test_clean.parquet
```

## Inspection Scripts

The inspection modules validate individual pipeline stages:

* `inspect_private_data.py`: checks private email fields and missing values
* `inspect_private_dates.py`: examines the private date distribution
* `inspect_hf_data.py`: examines public splits, labels, sources, and missing fields
* `check_split_leakage.py`: detects duplicates and overlap between splits

## Final Data Flow

```text
Private Mbox files
        ↓
Email extraction
        ↓
Private Parquet dataset
        ↓
Deduplication and stratified splitting
        ↓
Private train, evaluation, and test sets

Hugging Face Parquet files
        ↓
Schema and datatype normalization
        ↓
Public train, evaluation, and test sets

Public and private splits
        ↓
Combination
        ↓
Leakage detection and removal
        ↓
Clean model-ready datasets
```

The final datasets use the common schema:

```text
sender
subject
text
date
label
source
source_split
```
