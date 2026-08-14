"""Verifies the fail-closed platform gate itself, on every platform.

test_executor.py's mutation-path tests are skipped on Windows because
``PlanExecutor.execute()``/``undo()`` now refuse to mutate at all there (see
that file's module docstring for why). What must still be verified on every
platform -- Windows included, since that's exactly where it matters -- is
that the refusal itself actually happens: loudly, before any mutation, on
every mutating entry point. This file does that by monkeypatching the
reliability flag directly rather than relying on the current platform,
so it exercises the same code path on Linux/macOS CI as it does on Windows.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import tidy.executor as executor_module
from tidy.executor import PlanExecutor, UnverifiedPlatformError, undo


def move(source: str, destination: str) -> dict[str, str]:
    return {"source": source, "destination": destination, "reason": "test"}


@pytest.fixture
def unreliable_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        executor_module, "_METADATA_IDENTITY_VERIFICATION_RELIABLE", False
    )


def test_execute_refuses_to_mutate_on_unverified_platform(
    unreliable_platform: None, tmp_path: Path
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "one.txt").write_text("one", encoding="utf-8")
    journal_dir = tmp_path / "journals"

    with pytest.raises(UnverifiedPlatformError):
        PlanExecutor(root, journal_dir=journal_dir).run(
            [move("one.txt", "Documents/one.txt")], apply=True
        )

    assert (root / "one.txt").is_file()
    assert not (root / "Documents").exists()
    assert not journal_dir.exists()


def test_dry_run_is_unaffected_by_the_unverified_platform_gate(
    unreliable_platform: None, tmp_path: Path
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "one.txt").write_text("one", encoding="utf-8")

    result = PlanExecutor(root, journal_dir=tmp_path / "journals").run(
        [move("one.txt", "Documents/one.txt")], apply=False
    )

    assert result.applied is False
    assert (root / "one.txt").is_file()


def test_undo_refuses_to_mutate_on_unverified_platform(
    unreliable_platform: None, tmp_path: Path
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "Documents").mkdir()
    (root / "Documents" / "one.txt").write_text("one", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    journal_dir.mkdir()
    journal_path = journal_dir / "fake-run.json"
    journal_path.write_text(
        json.dumps(
            {
                "id": "fake-run",
                "state": "committed",
                "root": str(root),
                "operations": [],
                "undone_at": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnverifiedPlatformError):
        undo("fake-run", journal_dir=journal_dir)

    assert (root / "Documents" / "one.txt").is_file()
