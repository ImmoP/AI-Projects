"""Deterministic, extension-based categorisation.

Rules deliberately run before any future agent. Files without a matching rule are
returned separately so that the agent only ever sees the unresolved subset.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError

PROJECT_RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "rules.yaml"
PACKAGED_RULES_PATH = Path(__file__).resolve().parent / "config" / "rules.yaml"
DEFAULT_RULES_PATH = (
    PROJECT_RULES_PATH if PROJECT_RULES_PATH.is_file() else PACKAGED_RULES_PATH
)
_RESERVED_DIRECTORY_NAMES = frozenset(
    {
        "con", "prn", "aux", "nul",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects keys PyYAML would otherwise overwrite."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class RuleSet:
    """An immutable extension-to-directory mapping."""

    categories: dict[str, tuple[str, ...]]
    review_directory: str = "_ToReview"
    exclude_patterns: tuple[str, ...] = ()

    def is_excluded(self, filename: str) -> bool:
        """Return whether a basename matches a configured glob exclusion."""
        return any(fnmatchcase(filename, pattern) for pattern in self.exclude_patterns)

    def category_for(self, filename: str) -> str | None:
        """Return the configured category for *filename*, or ``None``."""
        lower_name = filename.casefold()
        for category, extensions in self.categories.items():
            if any(lower_name.endswith(extension) for extension in extensions):
                return category
        return None


def _directory_name(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{field} must be a non-empty directory name")
    if (
        Path(value).name != value
        or "/" in value
        or "\\" in value
        or value.startswith(".")
        or value.endswith((".", " "))
        or any(ord(character) < 32 or character in '<>:"|?*' for character in value)
        # Windows device basenames remain reserved with any extension, e.g.
        # CON.txt and LPT1.docx.
        or unicodedata.normalize("NFC", value).split(".", 1)[0].casefold()
        in _RESERVED_DIRECTORY_NAMES
    ):
        raise ValueError(f"{field} must be a visible, single directory name")
    return value


def load_rules(path: str | Path = DEFAULT_RULES_PATH) -> RuleSet:
    """Load and validate a YAML rule configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        try:
            raw = yaml.load(handle, Loader=_UniqueKeyLoader) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid rules configuration: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("rules.yaml must contain a mapping")

    raw_categories = raw.get("categories")
    if not isinstance(raw_categories, dict) or not raw_categories:
        raise ValueError("rules.yaml must contain a non-empty 'categories' mapping")

    categories: dict[str, tuple[str, ...]] = {}
    category_names: dict[str, str] = {}
    extension_owners: dict[str, str] = {}
    for raw_category, raw_extensions in raw_categories.items():
        category = _directory_name(raw_category, field="category")
        folded_category = unicodedata.normalize("NFC", category).casefold()
        if folded_category in category_names:
            raise ValueError(
                f"category names {category_names[folded_category]!r} and "
                f"{category!r} collide case-insensitively"
            )
        category_names[folded_category] = category
        if not isinstance(raw_extensions, list) or not raw_extensions:
            raise ValueError(f"category {category!r} must contain an extension list")

        extensions: set[str] = set()
        for raw_extension in raw_extensions:
            if not isinstance(raw_extension, str) or not raw_extension.strip():
                raise ValueError(f"invalid extension in category {category!r}")
            extension = raw_extension.strip().casefold()
            if not extension.startswith("."):
                extension = f".{extension}"
            if (
                extension == "."
                or "/" in extension
                or "\\" in extension
                or any(character.isspace() for character in extension)
                or any(character in extension for character in "*?[]")
            ):
                raise ValueError(f"invalid extension {raw_extension!r}")
            if extension in extensions:
                raise ValueError(
                    f"duplicate extension {extension!r} in category {category!r}"
                )
            owner = extension_owners.get(extension)
            if owner is not None:
                raise ValueError(
                    f"extension {extension!r} is assigned to both "
                    f"{owner!r} and {category!r}"
                )
            for existing_extension, existing_owner in extension_owners.items():
                if existing_owner != category and (
                    extension.endswith(existing_extension)
                    or existing_extension.endswith(extension)
                ):
                    raise ValueError(
                        f"extensions {existing_extension!r} in {existing_owner!r} "
                        f"and {extension!r} in {category!r} overlap ambiguously"
                    )
            extensions.add(extension)
            extension_owners[extension] = category

        # Longest first makes compound extensions such as .tar.gz explicit.
        categories[category] = tuple(sorted(extensions, key=len, reverse=True))

    review_directory = _directory_name(
        raw.get("review_directory", "_ToReview"), field="review_directory"
    )
    colliding_category = category_names.get(
        unicodedata.normalize("NFC", review_directory).casefold()
    )
    if colliding_category is not None:
        raise ValueError(
            f"review_directory {review_directory!r} collides with category "
            f"{colliding_category!r} case-insensitively"
        )
    raw_exclude_patterns = raw.get("exclude_patterns", [])
    if not isinstance(raw_exclude_patterns, list):
        raise ValueError("exclude_patterns must be a list of basename globs")
    exclude_patterns: list[str] = []
    for pattern in raw_exclude_patterns:
        if (
            not isinstance(pattern, str)
            or not pattern
            or "/" in pattern
            or "\\" in pattern
        ):
            raise ValueError("exclude_patterns must contain non-empty basename globs")
        exclude_patterns.append(pattern)

    return RuleSet(
        categories=categories,
        review_directory=review_directory,
        exclude_patterns=tuple(exclude_patterns),
    )


def classify_directory(
    directory: str | Path,
    rules: RuleSet,
) -> tuple[list[dict[str, str]], list[str]]:
    """Build rule-based moves for direct child files.

    Files matching ``rules.exclude_patterns``, symlinks and directories are
    omitted entirely: they appear in neither the move list nor the unresolved
    list. Other unmatched files are returned by name for the future agent. No
    file content is inspected. The two-item return signature remains unchanged.
    """
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)

    moves: list[dict[str, str]] = []
    unresolved: list[str] = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if rules.is_excluded(entry.name) or entry.is_symlink() or not entry.is_file():
            continue
        category = rules.category_for(entry.name)
        if category is None:
            unresolved.append(entry.name)
            continue
        moves.append(
            {
                "source": entry.name,
                "destination": f"{category}/{entry.name}",
                "reason": f"Matched extension rule for {category}/",
            }
        )
    return moves, unresolved
