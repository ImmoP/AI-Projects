"""Load private mailbox definitions from an ignored local TOML file."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import tomllib

from spam_detector.paths import PROJECT_ROOT

SemanticLabel = Literal["ham", "spam"]

DEFAULT_PRIVATE_MAILBOX_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "private_mailboxes.local.toml"
)

_NUMERIC_LABELS: dict[SemanticLabel, int] = {
    "ham": 0,
    "spam": 1,
}


@dataclass(frozen=True, slots=True)
class PrivateMailbox:
    """One ordered private mailbox input definition."""

    path: Path
    label: SemanticLabel
    source: str

    @property
    def numeric_label(self) -> int:
        """Return the explicit numeric model label."""

        return _NUMERIC_LABELS[self.label]


def load_private_mailboxes(
    config_path: str | Path,
    *,
    validate_paths: bool = True,
) -> tuple[PrivateMailbox, ...]:
    """Load and validate mailbox definitions without changing their order."""

    resolved_config_path = Path(config_path).expanduser().resolve()

    with resolved_config_path.open("rb") as file:
        config = tomllib.load(file)

    raw_mailboxes = config.get("mailboxes")

    if not isinstance(raw_mailboxes, list) or not raw_mailboxes:
        raise ValueError(
            "The private mailbox configuration must contain at least "
            "one [[mailboxes]] entry."
        )

    mailboxes: list[PrivateMailbox] = []
    seen_sources: set[str] = set()

    for index, raw_mailbox in enumerate(raw_mailboxes, start=1):
        if not isinstance(raw_mailbox, dict):
            raise ValueError(
                f"Mailbox entry {index} must be a TOML table."
            )

        missing_fields = {
            field
            for field in ("path", "label", "source")
            if field not in raw_mailbox
        }

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Mailbox entry {index} is missing required field(s): "
                f"{missing}."
            )

        configured_path = raw_mailbox["path"]
        label = raw_mailbox["label"]
        source = raw_mailbox["source"]

        if not isinstance(configured_path, str) or not configured_path.strip():
            raise ValueError(
                f"Mailbox entry {index} must have a non-empty path."
            )

        if not isinstance(label, str) or label not in _NUMERIC_LABELS:
            raise ValueError(
                f"Mailbox entry {index} label must be exactly "
                '"ham" or "spam".'
            )

        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"Mailbox entry {index} must have a non-empty source."
            )

        if source in seen_sources:
            raise ValueError(
                f"Mailbox entry {index} duplicates a source identifier."
            )

        mailbox_path = Path(configured_path).expanduser()

        if not mailbox_path.is_absolute():
            mailbox_path = resolved_config_path.parent / mailbox_path

        mailbox_path = mailbox_path.resolve()

        if validate_paths and not mailbox_path.exists():
            raise FileNotFoundError(
                f"Configured mailbox path does not exist for entry {index}."
            )

        seen_sources.add(source)
        mailboxes.append(
            PrivateMailbox(
                path=mailbox_path,
                label=cast(SemanticLabel, label),
                source=source,
            )
        )

    return tuple(mailboxes)
