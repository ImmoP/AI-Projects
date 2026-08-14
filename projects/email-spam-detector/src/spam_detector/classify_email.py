"""
Classify a single .eml email with the complete spam-detection pipeline.

Pipeline:

    .eml
      ↓
    GPT-2 spam classifier
      ↓
    temporal sender/domain reputation
      ↓
    Security V1
      ↓
    Security V2
      ↓
    cold-start dual-security fusion
      ↓
    final HAM / SPAM prediction

The temporal implementation mirrors the corrected temporal evaluation
logic and uses mixed-format date parsing.
"""

import argparse
import json
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from pathlib import Path

import joblib
import pandas as pd
import torch

from spam_detector.data_processing.extract import (
    extract_mail,
)
from spam_detector.evaluate import (
    BEST_CHECKPOINT,
    CONTEXT_LENGTH,
    create_model,
    get_device,
    load_checkpoint,
)
from spam_detector.inspection.extract_security_features import (
    extract_security_row,
    is_shared_relay_domain,
)
from spam_detector.model.classifier import (
    classification_forward,
)
from spam_detector.model.tokenization import (
    get_gpt2_tokenizer,
    tokenize_email,
)
from spam_detector.paths import (
    DATA_DIR,
    PROJECT_ROOT,
)
from spam_detector.train_security_classifier import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from spam_detector.train_security_classifier import (
    prepare_features as prepare_v1_features,
)
from spam_detector.train_security_classifier_v2 import (
    FEATURES as V2_FEATURES,
)
from spam_detector.train_security_classifier_v2 import (
    create_features as create_v2_features,
)

HAM_LABEL = 0
SPAM_LABEL = 1

TEMPORAL_MIN_COUNT = 1


PRIVATE_TRAIN_PATH = (
    DATA_DIR
    / "private_train.parquet"
)

SECURITY_V1_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "security_classifier"
    / "security_classifier.joblib"
)

SECURITY_V2_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "security_classifier_v2"
    / "security_classifier_v2.joblib"
)


V1_SPAM_THRESHOLD = 0.995
V2_SPAM_THRESHOLD = 0.98

V1_HAM_THRESHOLD = 0.005
V2_HAM_THRESHOLD = 0.10


def label_name(label):
    """
    Convert numeric label to HAM or SPAM.
    """

    if int(label) == SPAM_LABEL:
        return "SPAM"

    return "HAM"


def optional_text(value):
    """
    Convert optional values to printable strings.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (
        TypeError,
        ValueError,
    ):
        pass

    return str(value)


def optional_rate(value):
    """
    Format optional probability/rate values.
    """

    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except (
        TypeError,
        ValueError,
    ):
        pass

    return (
        f"{float(value):.4f}"
    )


def to_json_safe(value):
    """
    Convert pandas/numpy/PyTorch values into JSON-safe types.
    """

    if value is None:
        return None

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        pd.Timestamp,
    ):
        if pd.isna(value):
            return None

        return value.isoformat()

    if isinstance(
        value,
        torch.Tensor,
    ):
        if value.numel() == 1:
            return value.item()

        return (
            value
            .detach()
            .cpu()
            .tolist()
        )

    try:
        if pd.isna(value):
            return None
    except (
        TypeError,
        ValueError,
    ):
        pass

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key):
                to_json_safe(item)

            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            to_json_safe(item)
            for item in value
        ]

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()
        except (
            TypeError,
            ValueError,
        ):
            pass

    return value


def load_eml(eml_path):
    """
    Load one RFC822 .eml file.
    """

    eml_path = (
        Path(
            eml_path
        )
        .expanduser()
        .resolve()
    )

    if not eml_path.exists():
        raise FileNotFoundError(
            f"Email file not found: "
            f"{eml_path}"
        )

    if not eml_path.is_file():
        raise ValueError(
            f"Email path is not a file: "
            f"{eml_path}"
        )

    with eml_path.open(
        "rb"
    ) as file:

        message = BytesParser(
            policy=
                policy.default
        ).parse(
            file
        )

    return (
        eml_path,
        message,
    )


def temporal_extract_email_address(sender):
    """
    Extract and normalize sender email exactly as in the temporal
    evaluation pipeline.
    """

    if pd.isna(sender):
        return ""

    _, email_address = parseaddr(
        str(sender)
    )

    return (
        email_address
        .strip()
        .lower()
    )


def temporal_extract_domain(email_address):
    """
    Extract the sender domain.
    """

    if not email_address:
        return ""

    if "@" not in email_address:
        return ""

    return (
        email_address
        .rsplit("@", 1)[1]
        .strip()
        .lower()
        .rstrip(".")
    )


def parse_temporal_date(value):
    """
    Parse mixed email date formats and normalize them to UTC.
    """

    return pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
        format="mixed",
    )


def prepare_history():
    """
    Load and prepare the frozen private training history.

    This preprocessing mirrors the corrected temporal evaluation.
    """

    if not PRIVATE_TRAIN_PATH.exists():
        raise FileNotFoundError(
            "Private training history not found: "
            f"{PRIVATE_TRAIN_PATH}"
        )

    history = (
        pd.read_parquet(
            PRIVATE_TRAIN_PATH
        )
        .copy()
    )

    required_columns = {
        "sender",
        "date",
        "label",
    }

    missing_columns = (
        required_columns
        - set(
            history.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Private training history is missing columns: "
            f"{sorted(missing_columns)}"
        )

    history[
        "label"
    ] = (
        pd.to_numeric(
            history[
                "label"
            ],
            errors="raise",
        )
        .astype(int)
    )

    history[
        "_date_utc"
    ] = pd.to_datetime(
        history[
            "date"
        ],
        errors="coerce",
        utc=True,
        format="mixed",
    )

    history[
        "_sender_email"
    ] = (
        history[
            "sender"
        ]
        .apply(
            temporal_extract_email_address
        )
    )

    history[
        "_sender_domain"
    ] = (
        history[
            "_sender_email"
        ]
        .apply(
            temporal_extract_domain
        )
    )

    return history


def calculate_temporal_reputation(
    history,
    sender,
    date,
    gpt_prediction,
):
    """
    Apply temporal sender/domain reputation.

    Only historical messages strictly earlier than the target email
    may be used.

    Reputation overrides GPT only when historical labels are pure:

        spam_rate == 0.0 -> HAM
        spam_rate == 1.0 -> SPAM

    Mixed reputation falls back to GPT.
    """

    sender_email = (
        temporal_extract_email_address(
            sender
        )
    )

    sender_domain = (
        temporal_extract_domain(
            sender_email
        )
    )

    target_date = (
        parse_temporal_date(
            date
        )
    )

    result = {
        "sender_email":
            sender_email,

        "sender_domain":
            sender_domain,

        "target_date_utc":
            target_date,

        "date_available":
            not pd.isna(
                target_date
            ),

        "sender_count":
            0,

        "sender_spam_rate":
            None,

        "domain_count":
            0,

        "domain_spam_rate":
            None,

        "prediction":
            int(
                gpt_prediction
            ),

        "source":
            "gpt",

        "shared_relay":
            False,
    }

    if pd.isna(
        target_date
    ):
        result[
            "source"
        ] = (
            "gpt_missing_date"
        )

        return result

    sender_history = history[
        (
            history[
                "_sender_email"
            ]
            == sender_email
        )
        &
        (
            history[
                "_date_utc"
            ]
            < target_date
        )
    ]

    sender_count = len(
        sender_history
    )

    sender_spam_rate = None

    if sender_count > 0:

        sender_spam_count = int(
            sender_history[
                "label"
            ]
            .sum()
        )

        sender_spam_rate = (
            sender_spam_count
            / sender_count
        )

    result[
        "sender_count"
    ] = int(
        sender_count
    )

    result[
        "sender_spam_rate"
    ] = sender_spam_rate

    if is_shared_relay_domain(
        sender_domain
    ):
        result[
            "shared_relay"
        ] = True

        result[
            "domain_count"
        ] = 0

        result[
            "domain_spam_rate"
        ] = None

        result[
            "prediction"
        ] = int(
            gpt_prediction
        )

        result[
            "source"
        ] = (
            "gpt_shared_relay"
        )

        return result

    domain_history = history[
        (
            history[
                "_sender_domain"
            ]
            == sender_domain
        )
        &
        (
            history[
                "_date_utc"
            ]
            < target_date
        )
    ]

    domain_count = len(
        domain_history
    )

    domain_spam_rate = None

    if domain_count > 0:

        domain_spam_count = int(
            domain_history[
                "label"
            ]
            .sum()
        )

        domain_spam_rate = (
            domain_spam_count
            / domain_count
        )

    result[
        "domain_count"
    ] = int(
        domain_count
    )

    result[
        "domain_spam_rate"
    ] = domain_spam_rate

    if (
        sender_count
        >= TEMPORAL_MIN_COUNT
    ):

        if sender_spam_rate == 0.0:
            result[
                "prediction"
            ] = HAM_LABEL

            result[
                "source"
            ] = "sender_ham"

            return result

        if sender_spam_rate == 1.0:
            result[
                "prediction"
            ] = SPAM_LABEL

            result[
                "source"
            ] = "sender_spam"

            return result

    if (
        domain_count
        >= TEMPORAL_MIN_COUNT
    ):

        if domain_spam_rate == 0.0:
            result[
                "prediction"
            ] = HAM_LABEL

            result[
                "source"
            ] = "domain_ham"

            return result

        if domain_spam_rate == 1.0:
            result[
                "prediction"
            ] = SPAM_LABEL

            result[
                "source"
            ] = "domain_spam"

            return result

    result[
        "prediction"
    ] = int(
        gpt_prediction
    )

    result[
        "source"
    ] = "gpt"

    return result


def run_gpt_prediction(
    model,
    tokenizer,
    device,
    email_record,
):
    """
    Run GPT-2 on one email.
    """

    tokenized = (
        tokenize_email(
            sender=
                email_record[
                    "sender"
                ],

            subject=
                email_record[
                    "subject"
                ],

            text=
                email_record[
                    "text"
                ],

            tokenizer=
                tokenizer,

            context_length=
                CONTEXT_LENGTH,
        )
    )

    input_ids = torch.tensor(
        [
            tokenized.input_ids
        ],
        dtype=torch.long,
        device=device,
    )

    eos_indices = torch.tensor(
        [
            len(
                tokenized.input_ids
            )
            - 1
        ],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        logits = (
            classification_forward(
                model=
                    model,

                input_ids=
                    input_ids,

                eos_indices=
                    eos_indices,
            )
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
        )

    ham_probability = float(
        probabilities[
            0,
            HAM_LABEL,
        ]
        .detach()
        .cpu()
        .item()
    )

    spam_probability = float(
        probabilities[
            0,
            SPAM_LABEL,
        ]
        .detach()
        .cpu()
        .item()
    )

    prediction = int(
        torch.argmax(
            probabilities,
            dim=-1,
        )
        .detach()
        .cpu()
        .item()
    )

    return {
        "prediction":
            prediction,

        "ham_probability":
            ham_probability,

        "spam_probability":
            spam_probability,

        "original_token_count":
            int(
                tokenized
                .original_token_count
            ),

        "final_token_count":
            int(
                tokenized
                .final_token_count
            ),

        "was_truncated":
            bool(
                tokenized
                .was_truncated
            ),

        "metadata_truncated":
            bool(
                tokenized
                .metadata_truncated
            ),
    }


def create_live_security_dataframe(
    message,
    eml_path,
    email_record,
):
    """
    Extract live security features and create a one-row DataFrame.
    """

    security_row = (
        extract_security_row(
            message=
                message,

            mbox_path=
                eml_path,

            message_index=
                0,

            source=
                "live_eml",
        )
    )

    row = {
        "sender":
            email_record[
                "sender"
            ],

        "subject":
            email_record[
                "subject"
            ],

        "text":
            email_record[
                "text"
            ],

        "date":
            email_record[
                "date"
            ],

        "label":
            0,

        "source":
            "live_eml",

        "source_split":
            "live",
    }

    for (
        key,
        value,
    ) in security_row.items():

        row[
            f"security_{key}"
        ] = value

    return pd.DataFrame(
        [
            row
        ]
    )


def run_security_v1(
    model,
    security_dataframe,
):
    """
    Calculate Security V1 spam probability.
    """

    features = (
        prepare_v1_features(
            security_dataframe.copy()
        )
    )

    feature_columns = (
        NUMERIC_FEATURES
        + BOOLEAN_FEATURES
        + CATEGORICAL_FEATURES
    )

    missing_columns = [
        column

        for column
        in feature_columns

        if column
        not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            "Security V1 feature preparation "
            "is missing columns: "
            f"{missing_columns}"
        )

    X = features[
        feature_columns
    ]

    probability = float(
        model.predict_proba(
            X
        )[
            0,
            SPAM_LABEL,
        ]
    )

    return {
        "spam_probability":
            probability,

        "prediction":
            int(
                probability
                >= 0.5
            ),
    }


def run_security_v2(
    model,
    security_dataframe,
):
    """
    Calculate Security V2 spam probability.
    """

    features = (
        create_v2_features(
            security_dataframe.copy()
        )
    )

    missing_columns = [
        column

        for column
        in V2_FEATURES

        if column
        not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            "Security V2 feature preparation "
            "is missing columns: "
            f"{missing_columns}"
        )

    X = features[
        V2_FEATURES
    ]

    probability = float(
        model.predict_proba(
            X
        )[
            0,
            SPAM_LABEL,
        ]
    )

    return {
        "spam_probability":
            probability,

        "prediction":
            int(
                probability
                >= 0.5
            ),
    }


def apply_dual_security_fusion(
    temporal_result,
    security_v1_result,
    security_v2_result,
):
    """
    Apply frozen dual-security cold-start fusion.

    Spam:
        V1 >= 0.995 OR V2 >= 0.98

    Ham:
        V1 <= 0.005 AND V2 <= 0.10

    Security may change the temporal prediction only when both
    sender_count and domain_count are zero.
    """

    temporal_prediction = int(
        temporal_result[
            "prediction"
        ]
    )

    sender_count = int(
        temporal_result[
            "sender_count"
        ]
    )

    domain_count = int(
        temporal_result[
            "domain_count"
        ]
    )

    cold_start = (
        sender_count == 0
        and
        domain_count == 0
    )

    v1_probability = float(
        security_v1_result[
            "spam_probability"
        ]
    )

    v2_probability = float(
        security_v2_result[
            "spam_probability"
        ]
    )

    v1_spam_signal = (
        v1_probability
        >= V1_SPAM_THRESHOLD
    )

    v2_spam_signal = (
        v2_probability
        >= V2_SPAM_THRESHOLD
    )

    v1_ham_signal = (
        v1_probability
        <= V1_HAM_THRESHOLD
    )

    v2_ham_signal = (
        v2_probability
        <= V2_HAM_THRESHOLD
    )

    spam_candidate = (
        v1_spam_signal
        or
        v2_spam_signal
    )

    ham_candidate = (
        v1_ham_signal
        and
        v2_ham_signal
    )

    final_prediction = (
        temporal_prediction
    )

    override = "none"

    if not cold_start:

        reason = (
            "Security fusion was not eligible because "
            "historical sender or domain reputation exists."
        )

    elif (
        spam_candidate
        and
        temporal_prediction
        == HAM_LABEL
    ):

        final_prediction = (
            SPAM_LABEL
        )

        override = "spam"

        reason = (
            "Cold-start spam override: Security V1 or V2 "
            "reached the high-confidence spam threshold."
        )

    elif (
        ham_candidate
        and
        temporal_prediction
        == SPAM_LABEL
    ):

        final_prediction = (
            HAM_LABEL
        )

        override = "ham"

        reason = (
            "Cold-start ham override: Security V1 and V2 "
            "both reached their high-confidence ham thresholds."
        )

    else:

        reason = (
            "Cold-start security signals did not justify "
            "changing the temporal/GPT prediction."
        )

    return {
        "cold_start":
            cold_start,

        "v1_spam_signal":
            v1_spam_signal,

        "v2_spam_signal":
            v2_spam_signal,

        "v1_ham_signal":
            v1_ham_signal,

        "v2_ham_signal":
            v2_ham_signal,

        "spam_candidate":
            spam_candidate,

        "ham_candidate":
            ham_candidate,

        "override":
            override,

        "reason":
            reason,

        "prediction":
            int(
                final_prediction
            ),
    }


class SpamEmailClassifier:
    """
    End-to-end email spam classifier.
    """

    def __init__(self):
        self.device = (
            get_device()
        )

        print(
            f"Device: "
            f"{self.device}"
        )

        print(
            "Loading GPT-2 tokenizer..."
        )

        self.tokenizer = (
            get_gpt2_tokenizer()
        )

        print(
            "Loading GPT-2 spam classifier..."
        )

        self.gpt_model = (
            create_model(
                self.device
            )
        )

        load_checkpoint(
            self.gpt_model,
            BEST_CHECKPOINT,
        )

        self.gpt_model.eval()

        print(
            "Loading temporal reputation history..."
        )

        self.history = (
            prepare_history()
        )

        print(
            f"Temporal history: "
            f"{len(self.history):,} emails"
        )

        print(
            "Loading Security V1..."
        )

        if not SECURITY_V1_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Security V1 model not found: "
                f"{SECURITY_V1_MODEL_PATH}"
            )

        self.security_v1_model = (
            joblib.load(
                SECURITY_V1_MODEL_PATH
            )
        )

        print(
            "Loading Security V2..."
        )

        if not SECURITY_V2_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Security V2 model not found: "
                f"{SECURITY_V2_MODEL_PATH}"
            )

        self.security_v2_model = (
            joblib.load(
                SECURITY_V2_MODEL_PATH
            )
        )

        print(
            "Classifier ready."
        )

    def classify_eml(
        self,
        eml_path,
        json_output=None,
        print_result=True,
    ):
        """
        Classify one .eml file.
        """

        (
            eml_path,
            message,
        ) = load_eml(
            eml_path
        )

        email_record = (
            extract_mail(
                message=
                    message,

                label=
                    0,

                source=
                    "live_eml",
            )
        )

        gpt_result = (
            run_gpt_prediction(
                model=
                    self.gpt_model,

                tokenizer=
                    self.tokenizer,

                device=
                    self.device,

                email_record=
                    email_record,
            )
        )

        temporal_result = (
            calculate_temporal_reputation(
                history=
                    self.history,

                sender=
                    email_record[
                        "sender"
                    ],

                date=
                    email_record[
                        "date"
                    ],

                gpt_prediction=
                    gpt_result[
                        "prediction"
                    ],
            )
        )

        security_dataframe = (
            create_live_security_dataframe(
                message=
                    message,

                eml_path=
                    eml_path,

                email_record=
                    email_record,
            )
        )

        security_v1_result = (
            run_security_v1(
                model=
                    self.security_v1_model,

                security_dataframe=
                    security_dataframe,
            )
        )

        security_v2_result = (
            run_security_v2(
                model=
                    self.security_v2_model,

                security_dataframe=
                    security_dataframe,
            )
        )

        fusion_result = (
            apply_dual_security_fusion(
                temporal_result=
                    temporal_result,

                security_v1_result=
                    security_v1_result,

                security_v2_result=
                    security_v2_result,
            )
        )

        security_row = (
            security_dataframe
            .iloc[0]
            .to_dict()
        )

        result = {
            "email": {
                "path":
                    str(
                        eml_path
                    ),

                "sender":
                    email_record[
                        "sender"
                    ],

                "subject":
                    email_record[
                        "subject"
                    ],

                "date":
                    email_record[
                        "date"
                    ],
            },

            "gpt2":
                gpt_result,

            "temporal":
                temporal_result,

            "security_v1":
                security_v1_result,

            "security_v2":
                security_v2_result,

            "security_features": {
                "spf_result":
                    security_row.get(
                        "security_spf_result"
                    ),

                "dkim_result":
                    security_row.get(
                        "security_dkim_result"
                    ),

                "dmarc_result":
                    security_row.get(
                        "security_dmarc_result"
                    ),

                "from_domain":
                    security_row.get(
                        "security_from_domain"
                    ),

                "from_org_domain":
                    security_row.get(
                        "security_from_org_domain"
                    ),

                "return_path_org_match":
                    security_row.get(
                        "security_from_return_path_org_match"
                    ),

                "dkim_org_match":
                    security_row.get(
                        "security_from_dkim_org_match"
                    ),

                "reply_to_org_match":
                    security_row.get(
                        "security_from_reply_to_org_match"
                    ),
            },

            "dual_fusion":
                fusion_result,

            "final_prediction":
                fusion_result[
                    "prediction"
                ],

            "final_label":
                label_name(
                    fusion_result[
                        "prediction"
                    ]
                ),
        }

        result = (
            to_json_safe(
                result
            )
        )

        if print_result:

            self.print_result(
                result
            )

        if json_output is not None:

            self.save_json(
                result=
                    result,

                output_path=
                    json_output,
            )

        return result

    @staticmethod
    def print_result(result):
        """
        Print classification result.
        """

        email = result[
            "email"
        ]

        gpt = result[
            "gpt2"
        ]

        temporal = result[
            "temporal"
        ]

        security_v1 = result[
            "security_v1"
        ]

        security_v2 = result[
            "security_v2"
        ]

        security = result[
            "security_features"
        ]

        fusion = result[
            "dual_fusion"
        ]

        print()
        print(
            "EMAIL"
        )

        print(
            f"Sender:  "
            f"{email['sender']}"
        )

        print(
            f"Subject: "
            f"{email['subject']}"
        )

        print(
            f"Date:    "
            f"{email['date']}"
        )

        print()
        print(
            "GPT-2"
        )

        print(
            "Prediction: "
            f"{label_name(gpt['prediction'])}"
        )

        print(
            "P(spam):    "
            f"{gpt['spam_probability']:.6f}"
        )

        print(
            "P(ham):     "
            f"{gpt['ham_probability']:.6f}"
        )

        print(
            "Tokens:     "
            f"{gpt['final_token_count']} "
            f"(original: "
            f"{gpt['original_token_count']})"
        )

        print(
            "Truncated:  "
            f"{gpt['was_truncated']}"
        )

        print()
        print(
            "TEMPORAL REPUTATION"
        )

        print(
            "Sender: "
            f"{temporal['sender_email']}"
        )

        print(
            "Domain: "
            f"{temporal['sender_domain']}"
        )

        print(
            "Sender history: "
            f"{temporal['sender_count']} "
            "| spam rate: "
            f"{optional_rate(temporal['sender_spam_rate'])}"
        )

        print(
            "Domain history: "
            f"{temporal['domain_count']} "
            "| spam rate: "
            f"{optional_rate(temporal['domain_spam_rate'])}"
        )

        print(
            "Source:     "
            f"{temporal['source']}"
        )

        print(
            "Prediction: "
            f"{label_name(temporal['prediction'])}"
        )

        print()
        print(
            "SECURITY"
        )

        print(
            "V1 P(spam): "
            f"{security_v1['spam_probability']:.6f}"
        )

        print(
            "V2 P(spam): "
            f"{security_v2['spam_probability']:.6f}"
        )

        print(
            "SPF:   "
            f"{optional_text(security['spf_result'])}"
        )

        print(
            "DKIM:  "
            f"{optional_text(security['dkim_result'])}"
        )

        print(
            "DMARC: "
            f"{optional_text(security['dmarc_result'])}"
        )

        print(
            "From domain: "
            f"{optional_text(security['from_domain'])}"
        )

        print(
            "From org domain: "
            f"{optional_text(security['from_org_domain'])}"
        )

        print(
            "Return-Path aligned: "
            f"{security['return_path_org_match']}"
        )

        print(
            "DKIM aligned:        "
            f"{security['dkim_org_match']}"
        )

        print(
            "Reply-To aligned:    "
            f"{security['reply_to_org_match']}"
        )

        print()
        print(
            "DUAL FUSION"
        )

        print(
            "Cold start: "
            f"{fusion['cold_start']}"
        )

        print(
            "V1 spam signal: "
            f"{fusion['v1_spam_signal']}"
        )

        print(
            "V2 spam signal: "
            f"{fusion['v2_spam_signal']}"
        )

        print(
            "V1 ham signal:  "
            f"{fusion['v1_ham_signal']}"
        )

        print(
            "V2 ham signal:  "
            f"{fusion['v2_ham_signal']}"
        )

        print(
            "Override: "
            f"{fusion['override']}"
        )

        print(
            "Reason:   "
            f"{fusion['reason']}"
        )

        print()
        print(
            "FINAL PREDICTION"
        )
        print()

        print(
            f">>> "
            f"{result['final_label']} "
            f"<<<"
        )

        print()

    @staticmethod
    def save_json(
        result,
        output_path,
    ):
        """
        Save classification result as JSON.
        """

        output_path = Path(
            output_path
        )

        if not output_path.is_absolute():

            output_path = (
                PROJECT_ROOT
                / output_path
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"JSON output saved to: "
            f"{output_path}"
        )


def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Classify one .eml email using GPT-2, "
            "temporal reputation, Security V1/V2, "
            "and dual security fusion."
        )
    )

    parser.add_argument(
        "email",
        type=Path,
        help=
            "Path to the .eml file.",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help=
            "Optional JSON output path.",
    )

    return parser.parse_args()


def main():
    args = (
        parse_arguments()
    )

    classifier = (
        SpamEmailClassifier()
    )

    classifier.classify_eml(
        eml_path=
            args.email,

        json_output=
            args.json_output,

        print_result=
            True,
    )


if __name__ == "__main__":
    main()