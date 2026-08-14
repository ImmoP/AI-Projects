"""Mutation-path tests for tidy.executor -- skipped on Windows.

``PlanExecutor.execute()`` and ``undo()`` now refuse to mutate at all on
platforms where ``_METADATA_IDENTITY_VERIFICATION_RELIABLE`` is False (see
executor.py), which currently means Windows/NTFS: the pre-move
``SourceMetadata`` full-equality re-check -- this executor's core TOCTOU /
tamper-detection guarantee -- was found to spuriously disagree between two
stat() calls on a file nobody touched (an ``st_ctime_ns`` value observed to
shift between a stat() taken immediately after a write-and-close and a
stat() taken microseconds later on the same, untouched file). Since two
stat() calls disagreeing on an untouched file is exactly the failure mode
this check exists to catch, it cannot currently distinguish a real change
from that timing noise on this platform, so every test in this file that
exercises ``apply=True`` or ``undo()`` now hits ``UnverifiedPlatformError``
by design instead of running the behavior it was written to check. This is
an intentional, correct fail-closed refusal, not a skip of convenience --
see the "unverified platform" safety gap tracked in the README Limitations
section and its linked issue. The gate itself (that mutation is refused,
loudly, before anything on disk changes) is verified on every platform,
including Windows, by test_unverified_platform_gate.py.
"""

from __future__ import annotations

import errno
import json
import multiprocessing
import os
import unicodedata
from contextlib import contextmanager
from pathlib import Path

import pytest
import tidy.executor as executor_module
from tidy.executor import (
    DirectoryIdentity,
    JournalError,
    PlanChangedError,
    PlanExecutor,
    RunLockError,
    SourceMetadata,
    undo,
)

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "SourceMetadata's full-equality pre-move check "
        "(device/inode/mode/size/mtime_ns/ctime_ns) does not currently hold "
        "on NTFS for freshly-created files (st_ctime_ns observed to shift "
        "between consecutive stat() calls on an untouched file); "
        "PlanExecutor.execute()/undo() now refuse to mutate at all on this "
        "platform (UnverifiedPlatformError) rather than run with an "
        "unreliable TOCTOU check, so these mutation-path tests don't apply "
        "here. Real Windows behavior for this check is an open design "
        "question, tracked separately -- see README Limitations."
    ),
)


def move(source: str, destination: str) -> dict[str, str]:
    return {"source": source, "destination": destination, "reason": "test"}


def _apply_while_holding_lock(
    root_text: str,
    journal_text: str,
    source: str,
    destination: str,
    ready,
    release,
) -> None:
    import tidy.executor as child_executor

    def hold_before_move(kind: str, source_path: Path, destination_path: Path) -> None:
        del kind, source_path, destination_path
        ready.set()
        if not release.wait(10):
            raise RuntimeError("test barrier timed out")

    child_executor._MUTATION_HOOK = hold_before_move
    child_executor.PlanExecutor(root_text, journal_dir=journal_text).run(
        [move(source, destination)], apply=True
    )


def test_traversal_attempt_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "safe.txt").write_text("safe", encoding="utf-8")

    result = PlanExecutor(root, journal_dir=tmp_path / "journals").run(
        [move("safe.txt", "../escaped.txt")], apply=True
    )

    assert result.entries[0].status == "rejected"
    assert "escapes" in result.entries[0].message
    assert (root / "safe.txt").exists()
    assert not (tmp_path / "escaped.txt").exists()


def test_collision_adds_numeric_suffix(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    (root / "Documents").mkdir(parents=True)
    (root / "report.pdf").write_text("new", encoding="utf-8")
    (root / "Documents" / "report.pdf").write_text("old", encoding="utf-8")

    result = PlanExecutor(root, journal_dir=tmp_path / "journals").run(
        [move("report.pdf", "Documents/report.pdf")], apply=True
    )

    assert result.entries[0].destination == "Documents/report_1.pdf"
    assert (root / "Documents" / "report.pdf").read_text(encoding="utf-8") == "old"
    assert (root / "Documents" / "report_1.pdf").read_text(encoding="utf-8") == "new"


def test_missing_source_is_skipped_without_aborting(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "present.txt").write_text("present", encoding="utf-8")

    result = PlanExecutor(root, journal_dir=tmp_path / "journals").run(
        [
            move("missing.txt", "Documents/missing.txt"),
            move("present.txt", "Documents/present.txt"),
        ],
        apply=True,
    )

    assert [entry.status for entry in result.entries] == ["skipped", "moved"]
    assert (root / "Documents" / "present.txt").exists()


def test_undo_restores_original_state(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    original = root / "photo.jpg"
    original.write_text("photo", encoding="utf-8")
    journal_dir = tmp_path / "journals"

    applied = PlanExecutor(root, journal_dir=journal_dir).run(
        [move("photo.jpg", "Images/photo.jpg")], apply=True
    )
    rolled_back = undo(applied.run_id, journal_dir=journal_dir)

    assert rolled_back.entries[0].status == "restored"
    assert original.read_text(encoding="utf-8") == "photo"
    assert not (root / "Images" / "photo.jpg").exists()
    payload = json.loads(Path(rolled_back.journal_path).read_text(encoding="utf-8"))
    assert payload["state"] == "undone"


def test_partial_undo_is_recorded_and_retryable(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "one.txt").write_text("one", encoding="utf-8")
    (root / "two.txt").write_text("two", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    applied = PlanExecutor(root, journal_dir=journal_dir).run(
        [
            move("one.txt", "Documents/one.txt"),
            move("two.txt", "Documents/two.txt"),
        ],
        apply=True,
    )
    unavailable = tmp_path / "temporarily-unavailable.txt"
    (root / "Documents" / "one.txt").rename(unavailable)

    partial = undo(applied.run_id, journal_dir=journal_dir)

    payload = json.loads(Path(partial.journal_path).read_text(encoding="utf-8"))
    assert partial.journal_state == "partially_undone"
    assert payload["state"] == "partially_undone"
    assert payload["undone_at"] is None
    assert (root / "two.txt").exists()

    unavailable.rename(root / "Documents" / "one.txt")
    completed = undo(applied.run_id, journal_dir=journal_dir)

    payload = json.loads(Path(completed.journal_path).read_text(encoding="utf-8"))
    assert completed.journal_state == "undone"
    assert payload["state"] == "undone"
    assert payload["undone_at"] is not None
    assert (root / "one.txt").read_text(encoding="utf-8") == "one"
    assert (root / "two.txt").read_text(encoding="utf-8") == "two"


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    source = root / "notes.txt"
    source.write_text("notes", encoding="utf-8")
    journal_dir = tmp_path / "journals"

    result = PlanExecutor(root, journal_dir=journal_dir).run(
        [move("notes.txt", "Documents/notes.txt")]
    )

    assert result.entries[0].status == "planned"
    assert source.exists()
    assert not (root / "Documents").exists()
    assert not journal_dir.exists()


def test_journal_exists_before_first_filesystem_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "notes.txt").write_text("notes", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    real_move = executor_module._atomic_rename_no_replace

    def assert_journal_then_move(*args, **kwargs):
        journals = list(journal_dir.glob("*.json"))
        assert len(journals) == 1
        payload = json.loads(journals[0].read_text(encoding="utf-8"))
        assert payload["state"] == "in_progress"
        assert payload["operations"][0]["status"] == "moving"
        return real_move(*args, **kwargs)

    monkeypatch.setattr(executor_module, "_atomic_rename_no_replace", assert_journal_then_move)

    result = PlanExecutor(root, journal_dir=journal_dir).run(
        [move("notes.txt", "Documents/notes.txt")], apply=True
    )

    assert result.journal_state == "committed"


def test_successful_moves_remain_recoverable_after_mid_run_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    for name in ("one.txt", "two.txt"):
        (root / name).write_text(name, encoding="utf-8")
    journal_dir = tmp_path / "journals"
    real_move = executor_module._atomic_rename_no_replace
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated crash")
        return real_move(*args, **kwargs)

    monkeypatch.setattr(executor_module, "_atomic_rename_no_replace", fail_second)

    with pytest.raises(RuntimeError, match="simulated crash"):
        PlanExecutor(root, journal_dir=journal_dir).run(
            [
                move("one.txt", "Documents/one.txt"),
                move("two.txt", "Documents/two.txt"),
            ],
            apply=True,
        )

    payload = json.loads(next(journal_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["state"] == "partially_failed"
    assert payload["moves"] == [
        {"source": "one.txt", "destination": "Documents/one.txt", "reason": "test"}
    ]
    assert (root / "Documents" / "one.txt").exists()


def test_journal_failure_after_move_is_not_reported_as_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "notes.txt").write_text("notes", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    real_write = executor_module._write_json
    writes = 0

    def fail_success_record(path: Path, payload: dict) -> None:
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("disk full")
        real_write(path, payload)

    monkeypatch.setattr(executor_module, "_write_json", fail_success_record)

    with pytest.raises(JournalError, match="not reported as committed"):
        PlanExecutor(root, journal_dir=journal_dir).run(
            [move("notes.txt", "Documents/notes.txt")], apply=True
        )

    payload = json.loads(next(journal_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["state"] != "committed"
    assert payload["operations"][0]["source"] == "notes.txt"


def test_approved_destination_change_requires_a_new_plan(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "report.pdf").write_text("report", encoding="utf-8")
    executor = PlanExecutor(root, journal_dir=tmp_path / "journals")
    preview = executor.run([move("report.pdf", "Documents/report.pdf")])
    (root / "Documents").mkdir()
    (root / "Documents" / "report.pdf").write_text("occupied", encoding="utf-8")

    with pytest.raises(PlanChangedError, match="destination"):
        executor.run(preview.validated_plan, apply=True)

    assert not (root / "Documents" / "report_1.pdf").exists()
    assert not (tmp_path / "journals").exists()


def test_approved_source_replacement_requires_a_new_plan(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    source = root / "report.pdf"
    source.write_text("original", encoding="utf-8")
    executor = PlanExecutor(root, journal_dir=tmp_path / "journals")
    preview = executor.run([move("report.pdf", "Documents/report.pdf")])
    source.unlink()
    source.write_text("replacement with different identity", encoding="utf-8")

    with pytest.raises(PlanChangedError, match="source changed"):
        executor.run(preview.validated_plan, apply=True)

    assert source.read_text(encoding="utf-8").startswith("replacement")
    assert not (tmp_path / "journals").exists()


def test_word_lock_filename_is_a_valid_dry_run_entry(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    lock_file = root / "~$Bachelorarbeit_v4.docx"
    lock_file.touch()

    result = PlanExecutor(root, journal_dir=tmp_path / "journals").run(
        [move(lock_file.name, f"Documents/{lock_file.name}")]
    )

    assert result.entries[0].status == "planned"
    assert result.entries[0].source == lock_file.name


def test_home_shorthand_source_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()

    result = PlanExecutor(root, journal_dir=tmp_path / "journals").run(
        [move("~/something", "Documents/something")]
    )

    assert result.entries[0].status == "rejected"
    assert "home-directory shorthand" in result.entries[0].message


def test_runtime_error_in_one_of_five_entries_is_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    for index in (1, 2, 4, 5):
        (root / f"file-{index}.txt").touch()

    original_validate_move = executor_module._validate_move

    def fail_one_entry(root_path: Path, entry: dict[str, str]):
        if entry.get("source") == "broken.txt":
            raise RuntimeError("synthetic per-entry path failure")
        return original_validate_move(root_path, entry)

    monkeypatch.setattr(executor_module, "_validate_move", fail_one_entry)
    plan = [
        move("file-1.txt", "Documents/file-1.txt"),
        move("file-2.txt", "Documents/file-2.txt"),
        move("broken.txt", "Documents/broken.txt"),
        move("file-4.txt", "Documents/file-4.txt"),
        move("file-5.txt", "Documents/file-5.txt"),
    ]

    result = PlanExecutor(root, journal_dir=tmp_path / "journals").run(plan)

    assert [entry.status for entry in result.entries] == [
        "planned",
        "planned",
        "rejected",
        "planned",
        "planned",
    ]
    assert result.entries[2].message == "synthetic per-entry path failure"


def test_symlink_is_ignored(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    target = root / "target.txt"
    target.write_text("target", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")

    result = PlanExecutor(root, journal_dir=tmp_path / "journals").run(
        [move("link.txt", "Documents/link.txt")], apply=True
    )

    assert result.entries[0].status == "rejected"
    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == "target"


def test_symlinked_destination_directory_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "notes.txt").write_text("notes", encoding="utf-8")
    linked_directory = root / "Documents"
    try:
        linked_directory.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")

    result = PlanExecutor(root, journal_dir=tmp_path / "journals").run(
        [move("notes.txt", "Documents/notes.txt")], apply=True
    )

    assert result.entries[0].status == "rejected"
    assert (root / "notes.txt").exists()
    assert not (outside / "notes.txt").exists()


def test_open_file_is_skipped_when_handle_inspection_is_available(tmp_path: Path) -> None:
    import shutil

    if shutil.which("lsof") is None:
        pytest.skip("lsof is unavailable; executor falls back to advisory locking")
    root = tmp_path / "desktop"
    root.mkdir()
    source = root / "active.txt"
    source.write_text("active", encoding="utf-8")

    with source.open("rb"):
        result = PlanExecutor(root, journal_dir=tmp_path / "journals").run(
            [move("active.txt", "Documents/active.txt")], apply=True
        )

    assert result.entries[0].status == "skipped"
    assert "in use" in result.entries[0].message
    assert source.exists()


def test_undo_uses_suffix_when_original_name_was_recreated(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "report.pdf").write_text("original", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    applied = PlanExecutor(root, journal_dir=journal_dir).run(
        [move("report.pdf", "Documents/report.pdf")], apply=True
    )
    (root / "report.pdf").write_text("replacement", encoding="utf-8")

    rolled_back = undo(applied.run_id, journal_dir=journal_dir)

    assert rolled_back.entries[0].destination == "report_1.pdf"
    assert (root / "report.pdf").read_text(encoding="utf-8") == "replacement"
    assert (root / "report_1.pdf").read_text(encoding="utf-8") == "original"


def test_retrying_undo_never_overwrites_a_saved_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "report.pdf").write_text("original", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    applied = PlanExecutor(root, journal_dir=journal_dir).run(
        [move("report.pdf", "Documents/report.pdf")], apply=True
    )
    real_move = executor_module._atomic_rename_no_replace

    def fail_once(*args, **kwargs):
        raise OSError("temporarily unavailable")

    monkeypatch.setattr(executor_module, "_atomic_rename_no_replace", fail_once)
    partial = undo(applied.run_id, journal_dir=journal_dir)
    assert partial.journal_state == "partially_undone"

    (root / "report.pdf").write_text("replacement", encoding="utf-8")
    monkeypatch.setattr(executor_module, "_atomic_rename_no_replace", real_move)
    completed = undo(applied.run_id, journal_dir=journal_dir)

    assert completed.journal_state == "undone"
    assert (root / "report.pdf").read_text(encoding="utf-8") == "replacement"
    assert (root / "report_1.pdf").read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize(
    "folder_name, message",
    [
        ("", "non-empty"),
        ("bad/name", "only letters"),
        ("bad\\name", "only letters"),
        ("name with spaces", "only letters"),
        ("a" * 41, "at most 40"),
        ("CON", "reserved"),
        ("documents", "collides"),
    ],
)
def test_group_folder_policy_is_enforced_by_executor(
    tmp_path: Path, folder_name: str, message: str
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    filenames = ["one.txt", "two.md", "three.pdf"]
    for filename in filenames:
        (root / filename).touch()

    result = PlanExecutor(root).validate_groups(
        [{"folder_name": folder_name, "files": filenames, "reason": "same topic"}],
        candidate_files=filenames,
        existing_categories=["Documents", "Images"],
    )

    assert result.entries[0].status == "rejected"
    assert message in result.entries[0].message
    assert result.invalid_folder_names == 1
    assert result.moves == ()


def test_small_group_is_discarded_for_fallback(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    filenames = ["draft.docx", "comments.md"]
    for filename in filenames:
        (root / filename).touch()

    result = PlanExecutor(root).validate_groups(
        [
            {
                "folder_name": "Bachelorarbeit",
                "files": filenames,
                "reason": "same thesis",
            }
        ],
        candidate_files=filenames,
        existing_categories=["Documents"],
    )

    assert result.entries[0].status == "discarded"
    assert result.grouped_files == frozenset()
    assert result.moves == ()


def test_group_moves_preserve_filenames_and_reject_duplicate_membership(
    tmp_path: Path,
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    filenames = [f"file-{number}.txt" for number in range(1, 6)]
    for filename in filenames:
        (root / filename).touch()

    result = PlanExecutor(root).validate_groups(
        [
            {
                "folder_name": "Projekt_Änderung",
                "files": filenames[:3],
                "reason": "same project",
            },
            {
                "folder_name": "Second",
                "files": filenames[2:],
                "reason": "overlaps",
            },
        ],
        candidate_files=filenames,
        existing_categories=["Documents"],
    )

    assert [entry.status for entry in result.entries] == ["accepted", "rejected"]
    assert [move["destination"] for move in result.moves] == [
        "Projekt_Änderung/file-1.txt",
        "Projekt_Änderung/file-2.txt",
        "Projekt_Änderung/file-3.txt",
    ]


def test_apply_rejects_destination_parent_replaced_by_symlink_before_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("creating symlinks may require elevated Windows privileges")
    root = tmp_path / "desktop"
    documents = root / "Documents"
    documents.mkdir(parents=True)
    source = root / "report.pdf"
    source.write_text("approved", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    executor = PlanExecutor(root, journal_dir=tmp_path / "journals")
    plan = executor.validate([move("report.pdf", "Documents/report.pdf")])

    def replace_parent(kind: str, source_path: Path, destination_path: Path) -> None:
        del kind, source_path, destination_path
        documents.rename(tmp_path / "approved-documents")
        documents.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(executor_module, "_MUTATION_HOOK", replace_parent)
    with pytest.raises(PlanChangedError, match="symlink|parent"):
        executor.execute(plan)

    assert source.read_text(encoding="utf-8") == "approved"
    assert not (outside / "report.pdf").exists()
    payload = json.loads(next((tmp_path / "journals").glob("*.json")).read_text())
    assert payload["state"] == "partially_failed"
    assert payload["operations"][0]["status"] == "failed"


def test_apply_rejects_destination_parent_replaced_by_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    documents = root / "Documents"
    documents.mkdir(parents=True)
    source = root / "report.pdf"
    source.write_text("approved", encoding="utf-8")
    executor = PlanExecutor(root, journal_dir=tmp_path / "journals")
    plan = executor.validate([move("report.pdf", "Documents/report.pdf")])

    def replace_parent(kind: str, source_path: Path, destination_path: Path) -> None:
        del kind, source_path, destination_path
        documents.rename(tmp_path / "approved-documents")
        documents.mkdir()

    monkeypatch.setattr(executor_module, "_MUTATION_HOOK", replace_parent)
    with pytest.raises(PlanChangedError, match="identity"):
        executor.execute(plan)

    assert source.exists()
    assert not (documents / "report.pdf").exists()


def test_apply_rejects_destination_parent_that_appeared_after_preview(
    tmp_path: Path
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    source = root / "report.pdf"
    source.write_text("approved", encoding="utf-8")
    executor = PlanExecutor(root, journal_dir=tmp_path / "journals")
    plan = executor.validate([move("report.pdf", "Documents/report.pdf")])
    (root / "Documents").mkdir()

    with pytest.raises(PlanChangedError, match="parent appeared"):
        executor.execute(plan)

    assert source.exists()
    assert not (root / "Documents" / "report.pdf").exists()


def test_apply_atomic_no_clobber_rejects_destination_created_at_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    (root / "Documents").mkdir(parents=True)
    source = root / "report.pdf"
    destination = root / "Documents" / "report.pdf"
    source.write_text("approved", encoding="utf-8")
    executor = PlanExecutor(root, journal_dir=tmp_path / "journals")
    plan = executor.validate([move("report.pdf", "Documents/report.pdf")])

    def occupy_destination(kind: str, source_path: Path, destination_path: Path) -> None:
        del kind, source_path
        destination_path.write_text("unapproved", encoding="utf-8")

    monkeypatch.setattr(executor_module, "_MUTATION_HOOK", occupy_destination)
    with pytest.raises(FileExistsError):
        executor.execute(plan)

    assert source.read_text(encoding="utf-8") == "approved"
    assert destination.read_text(encoding="utf-8") == "unapproved"


def test_apply_rejects_source_replacement_at_mutation_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    source = root / "report.pdf"
    source.write_text("approved", encoding="utf-8")
    saved = tmp_path / "approved-report.pdf"
    executor = PlanExecutor(root, journal_dir=tmp_path / "journals")
    plan = executor.validate([move("report.pdf", "Documents/report.pdf")])

    def replace_source(kind: str, source_path: Path, destination_path: Path) -> None:
        del kind, destination_path
        source_path.rename(saved)
        source_path.write_text("replacement", encoding="utf-8")

    monkeypatch.setattr(executor_module, "_MUTATION_HOOK", replace_source)
    with pytest.raises(PlanChangedError, match="source changed"):
        executor.execute(plan)

    assert source.read_text(encoding="utf-8") == "replacement"
    assert saved.read_text(encoding="utf-8") == "approved"
    assert not (root / "Documents" / "report.pdf").exists()


def test_apply_uses_ctime_to_reject_metadata_preserving_content_change(
    tmp_path: Path
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    source = root / "note.txt"
    source.write_text("AAAA", encoding="utf-8")
    executor = PlanExecutor(root, journal_dir=tmp_path / "journals")
    plan = executor.validate([move("note.txt", "Documents/note.txt")])
    approved = plan.moves[0].source_metadata
    source.write_text("BBBB", encoding="utf-8")
    os.utime(source, ns=(approved.mtime_ns, approved.mtime_ns))

    with pytest.raises(PlanChangedError, match="source changed"):
        executor.execute(plan)

    assert source.read_text(encoding="utf-8") == "BBBB"
    assert not (root / "Documents" / "note.txt").exists()


def test_undo_rejects_current_parent_replaced_by_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("creating symlinks may require elevated Windows privileges")
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "report.pdf").write_text("approved", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    applied = PlanExecutor(root, journal_dir=journal_dir).run(
        [move("report.pdf", "Documents/report.pdf")], apply=True
    )
    documents = root / "Documents"
    outside = tmp_path / "outside"
    outside.mkdir()

    def replace_parent(kind: str, source_path: Path, destination_path: Path) -> None:
        del source_path, destination_path
        if kind == "undo":
            documents.rename(tmp_path / "approved-documents")
            documents.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(executor_module, "_MUTATION_HOOK", replace_parent)
    result = undo(applied.run_id, journal_dir=journal_dir)

    assert result.journal_state == "partially_undone"
    assert not (outside / "report.pdf").exists()
    assert not (root / "report.pdf").exists()


def test_undo_rejects_destination_created_at_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "report.pdf").write_text("approved", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    applied = PlanExecutor(root, journal_dir=journal_dir).run(
        [move("report.pdf", "Documents/report.pdf")], apply=True
    )

    def occupy_destination(kind: str, source_path: Path, destination_path: Path) -> None:
        del source_path
        if kind == "undo":
            destination_path.write_text("replacement", encoding="utf-8")

    monkeypatch.setattr(executor_module, "_MUTATION_HOOK", occupy_destination)
    result = undo(applied.run_id, journal_dir=journal_dir)

    assert result.journal_state == "partially_undone"
    assert (root / "report.pdf").read_text(encoding="utf-8") == "replacement"
    assert (root / "Documents" / "report.pdf").read_text(encoding="utf-8") == "approved"


def test_undo_rejects_source_replacement_at_mutation_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "report.pdf").write_text("approved", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    applied = PlanExecutor(root, journal_dir=journal_dir).run(
        [move("report.pdf", "Documents/report.pdf")], apply=True
    )
    saved = tmp_path / "approved-report.pdf"

    def replace_source(kind: str, source_path: Path, destination_path: Path) -> None:
        del destination_path
        if kind == "undo":
            source_path.rename(saved)
            source_path.write_text("replacement", encoding="utf-8")

    monkeypatch.setattr(executor_module, "_MUTATION_HOOK", replace_source)
    result = undo(applied.run_id, journal_dir=journal_dir)

    assert result.journal_state == "partially_undone"
    assert saved.read_text(encoding="utf-8") == "approved"
    assert (root / "Documents" / "report.pdf").read_text(encoding="utf-8") == "replacement"
    assert not (root / "report.pdf").exists()


def test_undo_rejects_original_parent_replaced_by_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    inbox = root / "Inbox"
    inbox.mkdir(parents=True)
    (inbox / "report.pdf").write_text("approved", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    applied = PlanExecutor(root, journal_dir=journal_dir).run(
        [move("Inbox/report.pdf", "Documents/report.pdf")], apply=True
    )

    def replace_parent(kind: str, source_path: Path, destination_path: Path) -> None:
        del source_path, destination_path
        if kind == "undo":
            inbox.rename(tmp_path / "approved-inbox")
            inbox.mkdir()

    monkeypatch.setattr(executor_module, "_MUTATION_HOOK", replace_parent)
    result = undo(applied.run_id, journal_dir=journal_dir)

    assert result.journal_state == "partially_undone"
    assert not (inbox / "report.pdf").exists()
    assert (root / "Documents" / "report.pdf").exists()


def test_cross_filesystem_moves_are_rejected_before_rename() -> None:
    source = SourceMetadata(1, 2, 0o100600, 4, 5, 6)
    destination_parent = DirectoryIdentity(7, 8, 0o40700)

    with pytest.raises(OSError) as captured:
        executor_module._ensure_same_device(source, destination_parent, Path("target"))

    assert captured.value.errno == errno.EXDEV
    assert "intentionally unsupported" in str(captured.value)


def test_executor_rejects_cross_filesystem_move_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    (root / "Documents").mkdir(parents=True)
    source = root / "report.pdf"
    source.write_text("approved", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    real_bound_parent = executor_module._bound_parent
    destination_calls = 0

    @contextmanager
    def different_destination_device(*args, **kwargs):
        nonlocal destination_calls
        if args[1] == "Documents":
            destination_calls += 1
            if destination_calls > 1:
                anchor = kwargs["anchor"]
                kwargs["anchor"] = type(anchor)(
                    anchor.relative_path,
                    DirectoryIdentity(
                        anchor.identity.device - 1,
                        anchor.identity.inode,
                        anchor.identity.mode,
                    ),
                )
        with real_bound_parent(*args, **kwargs) as (descriptor, path, identity):
            if args[1] == "Documents":
                identity = DirectoryIdentity(
                    identity.device + 1, identity.inode, identity.mode
                )
            yield descriptor, path, identity

    monkeypatch.setattr(executor_module, "_bound_parent", different_destination_device)
    with pytest.raises(OSError) as captured:
        PlanExecutor(root, journal_dir=journal_dir).run(
            [move("report.pdf", "Documents/report.pdf")], apply=True
        )

    assert captured.value.errno == errno.EXDEV
    assert source.read_text(encoding="utf-8") == "approved"
    assert not (root / "Documents" / "report.pdf").exists()
    payload = json.loads(next(journal_dir.glob("*.json")).read_text())
    assert payload["operations"][0]["status"] == "failed"


class _SimulatedCrash(BaseException):
    pass


def _crash_after_atomic_move(monkeypatch: pytest.MonkeyPatch) -> None:
    real_move = executor_module._atomic_rename_no_replace

    def move_then_crash(*args, **kwargs):
        real_move(*args, **kwargs)
        raise _SimulatedCrash("process stopped before journal success update")

    monkeypatch.setattr(executor_module, "_atomic_rename_no_replace", move_then_crash)


def test_undo_recovers_move_completed_before_moved_journal_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "report.pdf").write_text("approved", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    _crash_after_atomic_move(monkeypatch)

    with pytest.raises(_SimulatedCrash):
        PlanExecutor(root, journal_dir=journal_dir).run(
            [move("report.pdf", "Documents/report.pdf")], apply=True
        )
    payload = json.loads(next(journal_dir.glob("*.json")).read_text())
    assert payload["operations"][0]["status"] == "moving"

    # Undo uses the same primitive; restore the real implementation captured in
    # the crash wrapper by undoing the monkeypatch entirely.
    monkeypatch.undo()
    result = undo(payload["id"], journal_dir=journal_dir)

    assert result.journal_state == "undone"
    assert (root / "report.pdf").read_text(encoding="utf-8") == "approved"


def test_moving_recovery_recognizes_move_never_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    source = root / "report.pdf"
    source.write_text("approved", encoding="utf-8")
    journal_dir = tmp_path / "journals"

    def crash_before_move(*args, **kwargs):
        del args, kwargs
        raise _SimulatedCrash("process stopped before rename")

    monkeypatch.setattr(
        executor_module, "_atomic_rename_no_replace", crash_before_move
    )
    with pytest.raises(_SimulatedCrash):
        PlanExecutor(root, journal_dir=journal_dir).run(
            [move("report.pdf", "Documents/report.pdf")], apply=True
        )
    payload = json.loads(next(journal_dir.glob("*.json")).read_text())
    monkeypatch.undo()

    result = undo(payload["id"], journal_dir=journal_dir)

    assert result.journal_state == "undone"
    assert source.read_text(encoding="utf-8") == "approved"
    assert not (root / "Documents" / "report.pdf").exists()


def test_moving_recovery_refuses_both_source_and_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "report.pdf").write_text("approved", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    _crash_after_atomic_move(monkeypatch)
    with pytest.raises(_SimulatedCrash):
        PlanExecutor(root, journal_dir=journal_dir).run(
            [move("report.pdf", "Documents/report.pdf")], apply=True
        )
    payload = json.loads(next(journal_dir.glob("*.json")).read_text())
    (root / "report.pdf").write_text("new occupant", encoding="utf-8")
    monkeypatch.undo()

    result = undo(payload["id"], journal_dir=journal_dir)

    assert result.journal_state == "partially_undone"
    assert (root / "report.pdf").read_text(encoding="utf-8") == "new occupant"
    assert (root / "Documents" / "report.pdf").read_text(encoding="utf-8") == "approved"


def test_moving_recovery_refuses_destination_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "report.pdf").write_text("approved", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    _crash_after_atomic_move(monkeypatch)
    with pytest.raises(_SimulatedCrash):
        PlanExecutor(root, journal_dir=journal_dir).run(
            [move("report.pdf", "Documents/report.pdf")], apply=True
        )
    payload = json.loads(next(journal_dir.glob("*.json")).read_text())
    destination = root / "Documents" / "report.pdf"
    destination.rename(tmp_path / "approved-report.pdf")
    destination.write_text("wrong object", encoding="utf-8")
    monkeypatch.undo()

    result = undo(payload["id"], journal_dir=journal_dir)

    assert result.journal_state == "partially_undone"
    assert destination.read_text(encoding="utf-8") == "wrong object"
    assert not (root / "report.pdf").exists()


def test_moving_recovery_refuses_when_neither_path_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "report.pdf").write_text("approved", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    _crash_after_atomic_move(monkeypatch)
    with pytest.raises(_SimulatedCrash):
        PlanExecutor(root, journal_dir=journal_dir).run(
            [move("report.pdf", "Documents/report.pdf")], apply=True
        )
    payload = json.loads(next(journal_dir.glob("*.json")).read_text())
    (root / "Documents" / "report.pdf").unlink()
    monkeypatch.undo()

    result = undo(payload["id"], journal_dir=journal_dir)

    assert result.journal_state == "partially_undone"
    assert not (root / "report.pdf").exists()
    assert not (root / "Documents" / "report.pdf").exists()


@pytest.mark.parametrize(
    "failure_point",
    ["_write_file", "_flush_file", "_fsync_file", "_replace_file", "_fsync_directory"],
)
def test_initial_journal_persistence_failures_prevent_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    source = root / "report.pdf"
    source.write_text("approved", encoding="utf-8")
    journal_dir = tmp_path / "journals"

    def fail(*args, **kwargs):
        del args, kwargs
        raise OSError(f"simulated {failure_point} failure")

    monkeypatch.setattr(executor_module, failure_point, fail)
    with pytest.raises(JournalError, match=failure_point):
        PlanExecutor(root, journal_dir=journal_dir).run(
            [move("report.pdf", "Documents/report.pdf")], apply=True
        )

    assert source.read_text(encoding="utf-8") == "approved"
    assert not (root / "Documents" / "report.pdf").exists()
    for journal_path in journal_dir.glob("*.json"):
        assert json.loads(journal_path.read_text())["state"] != "committed"


@pytest.mark.parametrize(
    "failure_point",
    ["_write_file", "_flush_file", "_fsync_file", "_replace_file", "_fsync_directory"],
)
def test_post_move_journal_failures_leave_recoverable_non_committed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "report.pdf").write_text("approved", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    real_operation = getattr(executor_module, failure_point)
    calls = 0

    def fail_third(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError(f"simulated post-move {failure_point} failure")
        return real_operation(*args, **kwargs)

    monkeypatch.setattr(executor_module, failure_point, fail_third)
    with pytest.raises(JournalError) as captured:
        PlanExecutor(root, journal_dir=journal_dir).run(
            [move("report.pdf", "Documents/report.pdf")], apply=True
        )

    assert captured.value.result is not None
    assert captured.value.result.moved_count == 1
    journal_path = next(journal_dir.glob("*.json"))
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    assert payload["state"] == "partially_failed"
    assert payload["operations"][0]["status"] == "moved"
    assert (root / "Documents" / "report.pdf").exists()

    restored = undo(payload["id"], journal_dir=journal_dir)
    assert restored.journal_state == "undone"
    assert (root / "report.pdf").read_text(encoding="utf-8") == "approved"


def test_journal_never_contains_file_body_content(tmp_path: Path) -> None:
    secret = "PRIVATE DOCUMENT BODY 7e2abf"
    root = tmp_path / "root"
    root.mkdir()
    (root / "mystery").write_text(secret, encoding="utf-8")
    journal_dir = tmp_path / "journals"

    result = PlanExecutor(root, journal_dir=journal_dir).run(
        [
            {
                "source": "mystery",
                "destination": "_ToReview/mystery",
                "reason": "classification",
            }
        ],
        apply=True,
    )

    journal = Path(result.journal_path).read_text(encoding="utf-8")
    assert secret not in journal


def test_two_apply_processes_share_one_exclusive_root_lock(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "one.txt").write_text("one", encoding="utf-8")
    (root / "two.txt").write_text("two", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_apply_while_holding_lock,
        args=(
            str(root),
            str(journal_dir),
            "one.txt",
            "Documents/one.txt",
            ready,
            release,
        ),
    )
    process.start()
    try:
        assert ready.wait(10)
        with pytest.raises(RunLockError, match="already running"):
            PlanExecutor(root, journal_dir=tmp_path / "other-journals").run(
                [move("two.txt", "Documents/two.txt")], apply=True
            )
    finally:
        release.set()
        process.join(10)
    assert process.exitcode == 0
    assert (root / "Documents" / "one.txt").exists()
    assert (root / "two.txt").exists()


def test_apply_and_undo_share_one_exclusive_root_lock(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")
    (root / "new.txt").write_text("new", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    applied = PlanExecutor(root, journal_dir=journal_dir).run(
        [move("old.txt", "Documents/old.txt")], apply=True
    )
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_apply_while_holding_lock,
        args=(
            str(root),
            str(tmp_path / "other-journals"),
            "new.txt",
            "Documents/new.txt",
            ready,
            release,
        ),
    )
    process.start()
    try:
        assert ready.wait(10)
        with pytest.raises(RunLockError, match="already running"):
            undo(applied.run_id, journal_dir=journal_dir)
    finally:
        release.set()
        process.join(10)
    assert process.exitcode == 0
    assert (root / "Documents" / "old.txt").exists()


def test_case_insensitive_plan_collisions_are_suffixed_before_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "one.txt").touch()
    (root / "two.txt").touch()
    monkeypatch.setattr(executor_module, "_filesystem_case_sensitive", lambda _: False)

    plan = PlanExecutor(root).validate(
        [
            move("one.txt", "Documents/Report.txt"),
            move("two.txt", "Documents/report.txt"),
        ]
    )

    assert [item.destination for item in plan.moves] == [
        "Documents/Report.txt",
        "Documents/report_1.txt",
    ]


def test_unicode_normalized_plan_collisions_are_suffixed(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "one.txt").touch()
    (root / "two.txt").touch()
    composed = "Café.txt"
    decomposed = unicodedata.normalize("NFD", composed)

    plan = PlanExecutor(root).validate(
        [
            move("one.txt", f"Documents/{composed}"),
            move("two.txt", f"Documents/{decomposed}"),
        ]
    )

    assert plan.moves[0].destination == f"Documents/{composed}"
    assert plan.moves[1].destination.endswith("_1.txt")
