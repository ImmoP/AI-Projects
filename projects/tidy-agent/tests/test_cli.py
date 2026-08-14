from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import app
import pytest
import tidy.cli as cli_module
import tidy.executor as executor_module
from tidy.classification import ValidatedClassification, validate_classification_response


class StaticClassifier:
    def __init__(self, response, *, peeks=()):
        self.response = response
        self.peeks = tuple(peeks)
        self.calls: list[tuple[list[dict], list[str]]] = []
        self.probes: list[dict] = []

    def classify(self, metadata, categories, *, peek_tool=None):
        self.calls.append((list(metadata), list(categories)))
        if peek_tool is not None:
            for source in self.peeks:
                self.probes.append(json.loads(peek_tool(path=source)))
        return validate_classification_response(
            self.response,
            [item["name"] for item in metadata],
            categories,
        )

    def classify_with_agreement_gate(self, metadata, real_categories, *, review_directory):
        """Simulate E3 from one canned response: both passes 'agreed' to it.

        Plan-construction tests only need a faithful ``ValidatedClassification``
        shape (categories / invalid_sources), not genuine two-pass mechanics --
        those are covered directly against production in test_classification.py.
        Any per-source protocol failure (omitted, invalid, unproposed) collapses
        into ``invalid_sources`` here, matching E3's own single undifferentiated
        "either pass unusable" fallback bucket.
        """
        self.calls.append((list(metadata), list(real_categories)))
        result = validate_classification_response(
            self.response,
            [item["name"] for item in metadata],
            [*real_categories, review_directory],
        )
        failed = (
            set(result.omitted_sources)
            | set(result.invalid_sources)
            | set(result.unproposed_sources)
        )
        sources = [item["name"] for item in metadata]
        return ValidatedClassification(
            result.categories,
            (),
            tuple(source for source in sources if source in failed),
            (),
            result.unknown_sources,
            result.telemetry,
        )


@pytest.mark.smoke
def test_no_agent_plan_uses_rules_and_review_fallback(tmp_path: Path) -> None:
    (tmp_path / "photo.jpg").touch()
    (tmp_path / "mystery").touch()

    bundle = app.build_combined_plan(tmp_path, use_agent=False)

    assert [(move["origin"], move["destination"]) for move in bundle.moves] == [
        ("rule", "Images/photo.jpg"),
        ("rule", "_ToReview/mystery"),
    ]


def test_agent_receives_only_unresolved_metadata(tmp_path: Path) -> None:
    (tmp_path / "known.pdf").touch()
    (tmp_path / "mystery").touch()
    classifier = StaticClassifier(
        '{"decisions":[{"source":"mystery","category":"_ToReview"}]}'
    )

    bundle = app.build_combined_plan(tmp_path, classifier=classifier)

    assert bundle.moves[0]["destination"] == "Documents/known.pdf"
    assert bundle.moves[1]["destination"] == "_ToReview/mystery"
    assert bundle.moves[1]["origin"] == "agent"
    assert [item["name"] for item in classifier.calls[0][0]] == ["mystery"]


def test_content_reading_confines_peeks_to_unresolved_files(tmp_path: Path) -> None:
    """A file an extension rule resolved must not be readable, even if asked."""
    (tmp_path / "known.txt").write_text("resolved by rule", encoding="utf-8")
    (tmp_path / "notiz.md").write_text("Einkaufsliste", encoding="utf-8")
    (tmp_path / "mystery").write_text("Rechnung der Stadtwerke", encoding="utf-8")
    classifier = StaticClassifier(
        '{"decisions":[{"source":"mystery","category":"_ToReview"},'
        '{"source":"notiz.md","category":"_ToReview"}]}',
        peeks=("notiz.md", "known.txt", "mystery"),
    )

    app.build_combined_plan(
        tmp_path,
        classifier=classifier,
        read_contents=True,
        allow_remote_content=True,
    )

    unresolved, resolved, extensionless = classifier.probes
    assert unresolved["readable"] is True
    assert "Einkaufsliste" in unresolved["file_data"]["text"]
    # Resolved by an extension rule: refused by the binding, not by the prompt.
    assert resolved["status"] == "rejected"
    assert resolved["reason"] == "file is not authorized for content reading"
    # Extensionless and unresolved: readable as plain text, which is the whole
    # point of content reading.
    assert extensionless["readable"] is True
    assert extensionless["source"] == "plain_text_head"
    assert "Rechnung" in extensionless["file_data"]["text"]


def test_content_reading_rejects_absolute_and_relative_escapes(tmp_path: Path) -> None:
    """Containment is checked before the allowlist, so escapes stay rejected."""
    secret = tmp_path / "secret.txt"
    secret.write_text("outside the root", encoding="utf-8")
    root = tmp_path / "desk"
    root.mkdir()
    (root / "mystery").write_text("unklar", encoding="utf-8")
    classifier = StaticClassifier(
        '{"decisions":[{"source":"mystery","category":"_ToReview"}]}',
        peeks=(str(secret), "../secret.txt", "sub/../../secret.txt"),
    )

    app.build_combined_plan(
        root,
        classifier=classifier,
        read_contents=True,
        allow_remote_content=True,
    )

    assert [probe["status"] for probe in classifier.probes] == ["rejected"] * 3
    assert all(probe["readable"] is False for probe in classifier.probes)


def test_without_the_flag_no_peek_root_is_bound_at_all(tmp_path: Path) -> None:
    """The default path is unchanged: reading is impossible, not merely unused."""
    (tmp_path / "mystery").write_text("unklar", encoding="utf-8")
    classifier = StaticClassifier(
        '{"decisions":[{"source":"mystery","category":"_ToReview"}]}',
        peeks=("mystery",),
    )

    app.build_combined_plan(
        tmp_path,
        classifier=classifier,
    )

    assert classifier.probes == []


def test_parse_read_contents_flags_default_to_disabled() -> None:
    assert app.parse_args([]).read_contents is False
    assert app.parse_args(["--read-contents"]).read_contents is True
    assert app.parse_args(["--no-read-contents"]).read_contents is False
    assert app.parse_args([]).allow_remote_content is False
    assert app.parse_args(["--allow-remote-content"]).allow_remote_content is True


def _content_classifier_that_peeks() -> StaticClassifier:
    return StaticClassifier(
        '{"decisions":[{"source":"mystery","category":"_ToReview"}]}',
        peeks=("mystery",),
    )


def test_local_endpoint_allows_content_mode(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "mystery").write_text("bounded local evidence", encoding="utf-8")
    classifier = _content_classifier_that_peeks()
    monkeypatch.setenv("API_BASE", "http://127.0.0.1:11434")
    monkeypatch.setattr(
        cli_module,
        "build_classifier",
        lambda *args, **kwargs: classifier,
    )

    bundle = app.build_combined_plan(tmp_path, read_contents=True)

    assert classifier.probes[0]["readable"] is True
    assert bundle.peek_metrics["peek_chars_returned"] > 0


def test_remote_content_without_authorization_fails_before_peek_or_model(
    tmp_path: Path, monkeypatch
) -> None:
    secret = "NEVER READ THIS PRIVATE BODY"
    (tmp_path / "mystery").write_text(secret, encoding="utf-8")
    model_calls: list[str] = []
    monkeypatch.setenv("API_BASE", "https://user:password@api.example.test/v1")
    monkeypatch.setattr(
        cli_module,
        "build_classifier",
        lambda *args, **kwargs: model_calls.append("built"),
    )
    monkeypatch.setattr(
        cli_module,
        "peek_file_for_root",
        lambda *args, **kwargs: pytest.fail("peek tool must not be constructed"),
    )

    with pytest.raises(ValueError, match="--allow-remote-content") as error:
        app.build_combined_plan(tmp_path, read_contents=True)

    assert model_calls == []
    assert secret not in str(error.value)
    assert "password" not in str(error.value)


def test_supplied_agent_with_unknown_locality_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "mystery").write_text("private", encoding="utf-8")
    calls: list[str] = []
    classifier = SimpleNamespace(classify=lambda *args, **kwargs: calls.append("called"))

    with pytest.raises(ValueError, match="locality cannot be verified"):
        app.build_combined_plan(tmp_path, classifier=classifier, read_contents=True)

    assert calls == []


def test_remote_content_with_both_permissions_works(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "mystery").write_text("authorized evidence", encoding="utf-8")
    classifier = _content_classifier_that_peeks()
    monkeypatch.setenv("API_BASE", "https://api.example.test/v1")
    monkeypatch.setattr(
        cli_module,
        "build_classifier",
        lambda *args, **kwargs: classifier,
    )

    app.build_combined_plan(
        tmp_path,
        read_contents=True,
        allow_remote_content=True,
    )

    assert classifier.probes[0]["readable"] is True
    assert "authorized evidence" in classifier.probes[0]["file_data"]["text"]


def test_remote_rejection_never_prints_endpoint_credentials(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    (tmp_path / "mystery").write_text("private", encoding="utf-8")
    monkeypatch.setenv(
        "API_BASE",
        "https://alice:super-secret@private.example.test/v1?key=also-secret",
    )

    result = app.main(["--path", str(tmp_path), "--read-contents"])
    output = capsys.readouterr()

    assert result == 2
    assert "--allow-remote-content" in output.out
    assert "alice" not in output.out + output.err
    assert "super-secret" not in output.out + output.err
    assert "also-secret" not in output.out + output.err


def test_classifier_exception_cannot_echo_file_content(tmp_path: Path) -> None:
    secret = "PRIVATE EXCERPT FROM PROVIDER ERROR"
    (tmp_path / "mystery").write_text(secret, encoding="utf-8")

    def fail(*args, **kwargs) -> None:
        raise RuntimeError(f"provider echoed: {secret}")

    bundle = app.build_combined_plan(
        tmp_path,
        classifier=SimpleNamespace(classify=fail),
        read_contents=True,
        allow_remote_content=True,
    )

    assert secret not in json.dumps(bundle.moves)
    assert bundle.unproposed_sources == ("mystery",)


def test_model_supplied_reason_cannot_echo_content_into_plan(tmp_path: Path) -> None:
    secret = "PRIVATE CONTENT COPIED INTO REASON"
    (tmp_path / "mystery").write_text(secret, encoding="utf-8")
    classifier = StaticClassifier(
        json.dumps(
            {
                "decisions": [
                    {
                        "source": "mystery",
                        "category": "_ToReview",
                        "reason": secret,
                    }
                ]
            }
        )
    )

    bundle = app.build_combined_plan(
        tmp_path,
        classifier=classifier,
        read_contents=True,
        allow_remote_content=True,
    )

    assert secret not in json.dumps(bundle.moves)
    assert bundle.moves[0]["fallback"] == "invalid"


def test_direct_classification_list_is_validated_and_missing_files_fall_back(
    tmp_path: Path,
) -> None:
    (tmp_path / "meeting-notes").touch()
    (tmp_path / "mystery").touch()
    classifier = StaticClassifier(
        '{"decisions":[{"source":"meeting-notes","category":"Documents"}]}'
    )

    bundle = app.build_combined_plan(tmp_path, classifier=classifier)

    assert [(move["source"], move["destination"]) for move in bundle.moves] == [
        ("meeting-notes", "Documents/meeting-notes"),
        ("mystery", "_ToReview/mystery"),
    ]


def test_run_without_any_proposal_is_not_counted_as_per_file_omission(
    tmp_path: Path,
) -> None:
    """No proposal at all is one failure, not one omission per file.

    This distinction belongs to the single-pass ``classify()`` schema
    (``validate_classification_response``), still used unchanged for content
    mode; the default agreement-gate path collapses any per-source protocol
    failure -- on either pass -- into one undifferentiated fallback bucket by
    design (see ``merge_agreement_gate``'s case 6), so this is exercised via
    ``--read-contents`` rather than the bare default.
    """
    (tmp_path / "meeting-notes").touch()
    (tmp_path / "quellcode_alt").touch()
    classifier = StaticClassifier('{"wrong":true}')

    bundle = app.build_combined_plan(
        tmp_path, classifier=classifier, read_contents=True, allow_remote_content=True
    )

    assert bundle.omitted_sources == ()
    assert bundle.unproposed_sources == ("meeting-notes", "quellcode_alt")


def test_omitted_files_are_reported_separately_from_invalid_assignments(
    tmp_path: Path,
) -> None:
    """Silently dropping a file is a different failure from assigning it badly.

    Single-pass-schema distinction; exercised via ``--read-contents`` for the
    same reason as the test above.
    """
    for name in ("meeting-notes", "mystery", "quellcode_alt"):
        (tmp_path / name).touch()
    classifier = StaticClassifier(
        '{"decisions":[{"source":"meeting-notes","category":"Documents"},'
        '{"source":"mystery","category":"Steuerunterlagen_2024"}]}'
    )

    bundle = app.build_combined_plan(
        tmp_path, classifier=classifier, read_contents=True, allow_remote_content=True
    )

    # "quellcode_alt" never appeared in the proposal; "mystery" appeared with a
    # destination outside the whitelist. Both fall back, for different reasons.
    assert bundle.omitted_sources == ("quellcode_alt",)
    assert bundle.invalid_sources == ("mystery",)
    fallbacks = {
        move["source"]: move.get("fallback")
        for move in bundle.moves
        if move.get("fallback")
    }
    assert fallbacks == {"mystery": "invalid", "quellcode_alt": "omitted"}


@pytest.mark.smoke
def test_cli_default_is_dry_run_and_does_not_move(tmp_path: Path, capsys) -> None:
    source = tmp_path / "photo.jpg"
    source.touch()

    exit_code = app.main(["--path", str(tmp_path), "--no-agent"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Dry-run complete" in output
    assert "ORIGIN" in output
    assert source.exists()
    assert not (tmp_path / "Images").exists()


@pytest.mark.smoke
def test_apply_requires_explicit_confirmation(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "photo.jpg"
    source.touch()
    monkeypatch.setattr("builtins.input", lambda _: "no")

    exit_code = app.main(["--path", str(tmp_path), "--no-agent", "--apply"])

    assert exit_code == 0
    assert "Cancelled" in capsys.readouterr().out
    assert source.exists()
    assert not (tmp_path / "Images").exists()


def test_parse_undo_with_and_without_id() -> None:
    assert app.parse_args(["--undo"]).undo == app.LATEST_RUN
    assert app.parse_args(["--undo", "run-123"]).undo == "run-123"


def test_parse_think_flags_default_to_model_behavior() -> None:
    assert app.parse_args([]).think is None
    assert app.parse_args(["--think"]).think is True
    assert app.parse_args(["--no-think"]).think is False


def test_parse_group_flags_default_to_disabled() -> None:
    assert app.parse_args([]).group is False
    assert app.parse_args(["--group"]).group is True
    assert app.parse_args(["--no-group"]).group is False


def test_grouping_sees_all_files_and_overrides_extension_rules(tmp_path: Path) -> None:
    filenames = [
        "Bachelorarbeit_v4.docx",
        "Kritik_Bachelorarbeit_GPT.md",
        "Pruefbericht_Bachelorarbeit_v4.md",
    ]
    for filename in filenames:
        (tmp_path / filename).touch()
    tasks: list[str] = []

    def run(task: str) -> str:
        tasks.append(task)
        return (
            '{"ok": true, "groups": [{"folder_name": "Bachelorarbeit", '
            '"files": ["Bachelorarbeit_v4.docx", '
            '"Kritik_Bachelorarbeit_GPT.md", '
            '"Pruefbericht_Bachelorarbeit_v4.md"], '
            '"reason": "same thesis"}]}'
        )

    bundle = app.build_combined_plan(
        tmp_path,
        group=True,
        group_agent=SimpleNamespace(run=run),
    )

    assert {move["destination"] for move in bundle.moves} == {
        f"Bachelorarbeit/{filename}" for filename in filenames
    }
    assert {move["origin"] for move in bundle.moves} == {"group"}
    assert all(filename in tasks[0] for filename in filenames)


def test_direct_group_list_still_crosses_executor_validation(tmp_path: Path) -> None:
    filenames = ["project-a.txt", "project-b.md", "project-c.pdf"]
    for filename in filenames:
        (tmp_path / filename).touch()
    group_agent = SimpleNamespace(
        run=lambda _: [
            {
                "folder_name": "../unsafe",
                "files": filenames,
                "reason": "same project",
            }
        ]
    )
    classifier = StaticClassifier(
        '{"decisions":[{"source":"project-b.md","category":"_ToReview"}]}'
    )

    bundle = app.build_combined_plan(
        tmp_path,
        group=True,
        group_agent=group_agent,
        classifier=classifier,
    )

    assert bundle.grouping is not None
    assert bundle.grouping.invalid_folder_names == 1
    assert all(not move["destination"].startswith("../") for move in bundle.moves)


def test_discarded_small_group_falls_back_to_existing_logic(tmp_path: Path) -> None:
    (tmp_path / "Bachelorarbeit_v4.docx").touch()
    (tmp_path / "Kritik_Bachelorarbeit.md").touch()

    group_agent = SimpleNamespace(
        run=lambda _: (
            '{"ok": true, "groups": [{"folder_name": "Bachelorarbeit", '
            '"files": ["Bachelorarbeit_v4.docx", "Kritik_Bachelorarbeit.md"], '
            '"reason": "same thesis"}]}'
        )
    )
    classifier = StaticClassifier(
        '{"decisions":[{"source":"Kritik_Bachelorarbeit.md",'
        '"category":"_ToReview"}]}'
    )

    bundle = app.build_combined_plan(
        tmp_path,
        group=True,
        group_agent=group_agent,
        classifier=classifier,
    )

    assert [(move["origin"], move["destination"]) for move in bundle.moves] == [
        ("rule", "Documents/Bachelorarbeit_v4.docx"),
        ("agent", "_ToReview/Kritik_Bachelorarbeit.md"),
    ]
    assert bundle.grouping is not None
    assert bundle.grouping.entries[0].status == "discarded"


@pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "Exercises a real --apply mutation run; PlanExecutor.execute() now "
        "refuses to mutate at all on this platform "
        "(UnverifiedPlatformError -- see tests/test_executor.py's module "
        "docstring and README Limitations), so there is no partial-failure "
        "recovery path to observe here."
    ),
)
def test_partial_apply_failure_prints_structured_recovery_context(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root = tmp_path / "desktop"
    root.mkdir()
    (root / "one.txt").write_text("one", encoding="utf-8")
    (root / "two.txt").write_text("two", encoding="utf-8")
    journal_dir = tmp_path / "journals"
    real_executor = cli_module.PlanExecutor
    real_move = executor_module._atomic_rename_no_replace
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second move failure")
        return real_move(*args, **kwargs)

    monkeypatch.setattr(
        cli_module,
        "PlanExecutor",
        lambda directory: real_executor(directory, journal_dir=journal_dir),
    )
    monkeypatch.setattr(executor_module, "_atomic_rename_no_replace", fail_second)

    exit_code = app.main(
        ["--path", str(root), "--no-agent", "--apply", "--yes"]
    )

    output = capsys.readouterr().out
    journal = json.loads(next(journal_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert exit_code == 2
    assert "Execution partially failed" in output
    assert f"Run ID: {journal['id']}" in output
    assert f"Journal: {journal_dir / (journal['id'] + '.json')}" in output
    assert "Run state: partially_failed" in output
    assert "Completed moves: 1" in output
    assert "Failed move:" in output
    assert f"tidy --undo {journal['id']}" in output
