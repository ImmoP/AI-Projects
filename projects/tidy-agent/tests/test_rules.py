from __future__ import annotations

from pathlib import Path

import pytest
from tidy.rules import PACKAGED_RULES_PATH, PROJECT_RULES_PATH, classify_directory, load_rules


def test_rules_classify_known_extensions_and_leave_unknown_for_agent(tmp_path: Path) -> None:
    (tmp_path / "photo.JPG").touch()
    (tmp_path / "backup.tar.gz").touch()
    (tmp_path / "meeting-notes").touch()
    (tmp_path / ".secret.pdf").touch()
    rules = load_rules()

    moves, unresolved = classify_directory(tmp_path, rules)

    assert [(item["source"], item["destination"]) for item in moves] == [
        ("backup.tar.gz", "Archives/backup.tar.gz"),
        ("photo.JPG", "Images/photo.JPG"),
    ]
    assert unresolved == ["meeting-notes"]


def test_rule_scan_ignores_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "image.png"
    target.touch()
    link = tmp_path / "linked.png"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        return

    moves, _ = classify_directory(tmp_path, load_rules())

    assert [item["source"] for item in moves] == ["image.png"]


def test_classification_excludes_word_locks_and_hidden_files(tmp_path: Path) -> None:
    (tmp_path / "~$test.docx").touch()
    (tmp_path / ".DS_Store").touch()
    (tmp_path / "keep.docx").touch()

    moves, unresolved = classify_directory(tmp_path, load_rules())

    assert [item["source"] for item in moves] == ["keep.docx"]
    assert unresolved == []


def test_packaged_rule_fallback_matches_editable_configuration() -> None:
    assert load_rules(PACKAGED_RULES_PATH) == load_rules(PROJECT_RULES_PATH)


@pytest.mark.parametrize(
    "configuration, message",
    [
        (
            "categories:\n  Documents: [.pdf]\n  Archives: [.PDF]\n",
            "assigned to both",
        ),
        (
            "categories:\n  Documents: [.pdf]\n  documents: [.txt]\n",
            "collide case-insensitively",
        ),
        (
            "categories:\n  Documents: [.pdf]\nreview_directory: documents\n",
            "review_directory",
        ),
        (
            "categories:\n  Documents: [.pdf, PDF]\n",
            "duplicate extension",
        ),
        (
            "categories:\n  Documents: ['.']\n",
            "invalid extension",
        ),
        (
            "categories:\n  Archives: [.gz]\n  Backups: [.tar.gz]\n",
            "overlap ambiguously",
        ),
        (
            "categories:\n  CON: [.txt]\n",
            "visible, single directory name",
        ),
        (
            "categories:\n  bad\\\\name: [.txt]\n",
            "visible, single directory name",
        ),
        (
            "categories:\n  Documents: [.pdf]\n  Documents: [.txt]\n",
            "duplicate key",
        ),
        ("- not\n- a\n- mapping\n", "must contain a mapping"),
    ],
)
def test_invalid_or_ambiguous_rule_configurations_fail_fast(
    tmp_path: Path, configuration: str, message: str
) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(configuration, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_rules(path)


@pytest.mark.parametrize(
    "reserved_name",
    [
        "CON",
        "CON.txt",
        "PRN.pdf",
        "AUX",
        "NUL.log",
        "COM1",
        "COM9.txt",
        "LPT1.docx",
        "LPT9",
    ],
)
def test_windows_device_basenames_are_reserved_with_extensions(
    tmp_path: Path, reserved_name: str
) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        f"categories:\n  {reserved_name}: [.txt]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="visible, single directory name"):
        load_rules(path)


@pytest.mark.parametrize("invalid_name", ["Documents.", "Documents "])
def test_directory_names_reject_trailing_periods_and_spaces(
    tmp_path: Path, invalid_name: str
) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        f"categories:\n  '{invalid_name}': [.txt]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="visible, single directory name"):
        load_rules(path)
