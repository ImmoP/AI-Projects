"""Offline, deterministic tests for the production E3 agreement-gate integration.

No Ollama dependency: every test drives ``StructuredClassifier`` with a
``FakeModel`` (identical in spirit to ``tests/test_classification.py``) or
exercises the pure validation/merge functions directly. Unit-level coverage
of the schema validator and the merge state table already exists in
``tests/test_calibration_candidates.py`` (which now exercises these same
production functions through their ``evals.calibration_candidates``
re-export); this file adds the integration-level coverage specific to
``StructuredClassifier.classify_with_agreement_gate`` and its production CLI
wiring in ``build_combined_plan`` -- the two-call orchestration, order
perturbation, telemetry, and the plan/security boundary. Nothing here
touches the consumed 41-file Holdout.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import app
from tidy.classification import (
    StructuredClassifier,
    reverse_pass_order,
)

REAL_CATEGORIES = ["Documents", "Code"]
REVIEW_DIRECTORY = "_ToReview"
METADATA = [
    {"name": "one", "size_bytes": 10},
    {"name": "two", "size_bytes": 20},
]


class FakeModel:
    structured_output_mode = "json_schema"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[list[dict], dict]] = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(
            content=response,
            token_usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        )


def _prompt_text(call: tuple[list[dict], dict]) -> str:
    messages, _kwargs = call
    return messages[0]["content"][0]["text"]


# --- Two-call orchestration and order perturbation --------------------------


def test_classify_with_agreement_gate_issues_exactly_two_requests() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"two","decision":"classify","category":"Code"}]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"two","decision":"classify","category":"Code"}]}',
    )
    classifier = StructuredClassifier(model)

    result = classifier.classify_with_agreement_gate(
        METADATA, REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )

    assert len(model.calls) == 2
    assert result.categories == {"one": "Documents", "two": "Code"}
    assert result.telemetry["classification_requests"] == 2
    assert result.telemetry["final_classification_requests"] == 2
    assert result.telemetry["peek_phase_requests"] == 0


def test_second_pass_reverses_source_order_without_adding_or_omitting_any() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"two","decision":"classify","category":"Code"}]}',
        '{"decisions":[{"source":"two","decision":"classify","category":"Code"},'
        '{"source":"one","decision":"classify","category":"Documents"}]}',
    )
    classifier = StructuredClassifier(model)

    classifier.classify_with_agreement_gate(
        METADATA, REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )

    first_prompt = _prompt_text(model.calls[0])
    second_prompt = _prompt_text(model.calls[1])
    first_block = first_prompt[first_prompt.index("<FILENAME_DATA>") :]
    second_block = second_prompt[second_prompt.index("<FILENAME_DATA>") :]
    assert first_block.index("one") < first_block.index("two")
    assert second_block.index("two") < second_block.index("one")
    # Same wording and schema instructions in both passes; only order differs.
    first_head = first_prompt[: first_prompt.index("<FILENAME_DATA>")]
    second_head = second_prompt[: second_prompt.index("<FILENAME_DATA>")]
    assert first_head == second_head


def test_reverse_pass_order_preserves_every_source_exactly_once() -> None:
    metadata = [{"name": "a"}, {"name": "b"}, {"name": "c"}]

    reversed_once = reverse_pass_order(metadata)

    assert {item["name"] for item in reversed_once} == {"a", "b", "c"}
    assert len(reversed_once) == len(metadata)
    assert reversed_once == ({"name": "c"}, {"name": "b"}, {"name": "a"})


def test_agreement_gate_telemetry_counts_every_pass_and_gate_bucket() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"two","decision":"review","category":null}]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"two","decision":"classify","category":"Code"}]}',
    )
    classifier = StructuredClassifier(model)

    result = classifier.classify_with_agreement_gate(
        METADATA, REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )

    telemetry = result.telemetry
    assert telemetry["pass1_classify_count"] == 1
    assert telemetry["pass1_review_count"] == 1
    assert telemetry["pass2_classify_count"] == 2
    assert telemetry["gate_classify_same_category_count"] == 1  # "one"
    assert telemetry["gate_review_then_classify_count"] == 1  # "two"
    assert telemetry["gate_final_automatic_count"] == 1
    assert telemetry["gate_final_review_count"] == 1
    assert result.categories == {"one": "Documents", "two": REVIEW_DIRECTORY}


# --- E3 merge table, exercised end-to-end through the production classifier -


def test_both_passes_classify_same_category_is_accepted() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.categories == {"one": "Documents"}
    assert result.invalid_sources == ()


def test_classify_classify_different_category_reviews() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Code"}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.categories == {"one": REVIEW_DIRECTORY}


def test_classify_then_review_reviews() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
        '{"decisions":[{"source":"one","decision":"review","category":null}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.categories == {"one": REVIEW_DIRECTORY}


def test_review_then_classify_reviews() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"review","category":null}]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.categories == {"one": REVIEW_DIRECTORY}


def test_both_review_reviews() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"review","category":null}]}',
        '{"decisions":[{"source":"one","decision":"review","category":null}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.categories == {"one": REVIEW_DIRECTORY}


def test_invalid_first_pass_then_valid_classify_reviews() -> None:
    model = FakeModel(
        "not-json",
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.categories == {}
    assert result.invalid_sources == ("one",)


def test_valid_classify_then_invalid_second_pass_reviews() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
        "not-json",
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.categories == {}
    assert result.invalid_sources == ("one",)


def test_both_passes_invalid_reviews() -> None:
    model = FakeModel("not-json", "{}")
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.categories == {}
    assert result.invalid_sources == ("one",)


# --- Structured failure modes -----------------------------------------------


def test_malformed_json_in_first_pass_is_handled_safely() -> None:
    model = FakeModel(
        "{not valid json",
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.invalid_sources == ("one",)
    assert result.telemetry["parse_failures"] == 1


def test_schema_invalid_root_is_handled_safely() -> None:
    model = FakeModel(
        '{"moves":[]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.invalid_sources == ("one",)


def test_incomplete_first_pass_omitted_source_reviews() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"two","decision":"classify","category":"Code"}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        METADATA, REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.categories == {"one": "Documents"}
    assert result.invalid_sources == ("two",)


def test_incomplete_second_pass_omitted_source_reviews() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"two","decision":"classify","category":"Code"}]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        METADATA, REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.categories == {"one": "Documents"}
    assert result.invalid_sources == ("two",)


def test_provider_failure_on_either_pass_is_never_treated_as_success() -> None:
    # Both passes always run regardless of pass 1's outcome (E3 never
    # silently falls back to a single successful pass), so a total
    # infrastructure outage means both requests fail independently.
    model = FakeModel(
        RuntimeError("provider unavailable"), RuntimeError("provider unavailable")
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.categories == {}
    assert result.invalid_sources == ("one",)
    assert result.telemetry["provider_errors"] == 2


def test_one_pass_provider_failure_is_never_masked_by_the_other_succeeding() -> None:
    model = FakeModel(
        RuntimeError("provider unavailable"),
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert result.categories == {}
    assert result.invalid_sources == ("one",)
    assert result.telemetry["provider_errors"] == 1


def test_duplicate_source_in_a_pass_is_rejected() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"one","decision":"classify","category":"Code"}]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert "one" not in result.categories
    assert result.invalid_sources == ("one",)
    assert result.telemetry["duplicate_source_responses"] == 1


def test_invented_category_never_becomes_a_decision() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"SecretVault"}]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert "SecretVault" not in result.categories.values()
    assert result.categories == {}
    assert result.invalid_sources == ("one",)


def test_invented_source_is_ignored_and_never_reaches_categories() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"../escape","decision":"classify","category":"Documents"}]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )
    result = StructuredClassifier(model).classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )
    assert "../escape" not in result.categories
    assert set(result.categories) <= {"one"}
    assert "../escape" in result.unknown_sources


# --- Security: no model output can expand filesystem authority -------------


def test_no_destination_outside_authorized_categories_or_review(tmp_path: Path) -> None:
    (tmp_path / "mystery").touch()
    model = FakeModel(
        '{"decisions":[{"source":"mystery","decision":"classify",'
        '"category":"../../etc"}]}',
        '{"decisions":[{"source":"mystery","decision":"classify",'
        '"category":"../../etc"}]}',
    )
    classifier = StructuredClassifier(model)

    bundle = app.build_combined_plan(tmp_path, classifier=classifier)

    assert bundle.moves[0]["destination"] == "_ToReview/mystery"
    for move in bundle.moves:
        category = move["destination"].split("/", 1)[0]
        assert category in {*app.load_rules().categories, "_ToReview"}


def test_no_arbitrary_source_path_can_be_introduced(tmp_path: Path) -> None:
    (tmp_path / "mystery").touch()
    model = FakeModel(
        '{"decisions":[{"source":"mystery","decision":"classify","category":"Documents"},'
        '{"source":"/etc/passwd","decision":"classify","category":"Documents"}]}',
        '{"decisions":[{"source":"mystery","decision":"classify","category":"Documents"}]}',
    )
    classifier = StructuredClassifier(model)

    bundle = app.build_combined_plan(tmp_path, classifier=classifier)

    assert [move["source"] for move in bundle.moves] == ["mystery"]
    assert all("/etc/passwd" not in json.dumps(move) for move in bundle.moves)


def test_classification_never_mutates_the_filesystem(tmp_path: Path) -> None:
    (tmp_path / "mystery").touch()
    model = FakeModel(
        '{"decisions":[{"source":"mystery","decision":"classify","category":"Documents"}]}',
        '{"decisions":[{"source":"mystery","decision":"classify","category":"Documents"}]}',
    )
    classifier = StructuredClassifier(model)

    app.build_combined_plan(tmp_path, classifier=classifier)

    assert (tmp_path / "mystery").exists()
    assert not (tmp_path / "Documents").exists()


# --- Production wiring: E3 is the metadata-only default --------------------


class _SpyClassifier:
    def __init__(self, response: str) -> None:
        self.response = response
        self.gate_calls = 0
        self.single_pass_calls = 0

    def classify_with_agreement_gate(self, metadata, real_categories, *, review_directory):
        self.gate_calls += 1
        from tidy.classification import validate_classification_response

        result = validate_classification_response(
            self.response,
            [item["name"] for item in metadata],
            [*real_categories, review_directory],
        )
        return result

    def classify(self, metadata, categories, **kwargs):
        self.single_pass_calls += 1
        from tidy.classification import validate_classification_response

        return validate_classification_response(
            self.response, [item["name"] for item in metadata], categories
        )


def test_default_unresolved_file_path_uses_the_agreement_gate(tmp_path: Path) -> None:
    (tmp_path / "mystery").touch()
    spy = _SpyClassifier('{"decisions":[{"source":"mystery","category":"Documents"}]}')

    app.build_combined_plan(tmp_path, classifier=spy)

    assert spy.gate_calls == 1
    assert spy.single_pass_calls == 0


def test_single_pass_classification_mode_is_an_explicit_opt_out(tmp_path: Path) -> None:
    (tmp_path / "mystery").touch()
    spy = _SpyClassifier('{"decisions":[{"source":"mystery","category":"Documents"}]}')

    app.build_combined_plan(tmp_path, classifier=spy, classification_mode="single_pass")

    assert spy.gate_calls == 0
    assert spy.single_pass_calls == 1


def test_content_mode_bypasses_the_agreement_gate_entirely(tmp_path: Path) -> None:
    (tmp_path / "mystery").write_text("evidence", encoding="utf-8")
    spy = _SpyClassifier('{"decisions":[{"source":"mystery","category":"Documents"}]}')

    app.build_combined_plan(
        tmp_path, classifier=spy, read_contents=True, allow_remote_content=True
    )

    assert spy.gate_calls == 0
    assert spy.single_pass_calls == 1


def test_metadata_control_bypasses_the_agreement_gate_entirely(tmp_path: Path) -> None:
    (tmp_path / "mystery").touch()
    spy = _SpyClassifier('{"decisions":[{"source":"mystery","category":"Documents"}]}')

    app.build_combined_plan(tmp_path, classifier=spy, metadata_control=True)

    assert spy.gate_calls == 0
    assert spy.single_pass_calls == 1


def test_deterministic_rules_still_bypass_the_agreement_gate_entirely(
    tmp_path: Path,
) -> None:
    (tmp_path / "photo.jpg").touch()
    spy = _SpyClassifier('{"decisions":[]}')

    bundle = app.build_combined_plan(tmp_path, classifier=spy)

    assert spy.gate_calls == 0
    assert bundle.moves == [
        {
            "source": "photo.jpg",
            "destination": "Images/photo.jpg",
            "reason": "Matched extension rule for Images/",
            "origin": "rule",
        }
    ]


def test_agreement_gate_never_receives_a_peek_tool() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )
    classifier = StructuredClassifier(model)

    result = classifier.classify_with_agreement_gate(
        [{"name": "one"}], REAL_CATEGORIES, review_directory=REVIEW_DIRECTORY
    )

    # No PeekSession-derived keys appear at all -- stronger than "zero", this
    # proves no peek/content mechanism was ever constructed for this path.
    assert "peek_calls" not in result.telemetry
    assert "peek_bytes_read" not in result.telemetry
    assert "peek_chars_returned" not in result.telemetry
    assert result.telemetry["peek_phase_requests"] == 0
    assert result.telemetry["peek_requests_model"] == 0
    assert result.telemetry["peek_requests_authorized"] == 0
    assert result.telemetry["content_unavailable"] == 0
    for messages, _kwargs in model.calls:
        assert "UNTRUSTED_FILE_DATA" not in messages[0]["content"][0]["text"]


def test_cli_exposes_no_new_e3_or_agreement_gate_flag() -> None:
    parser_source = Path(
        Path(__file__).parents[1] / "src" / "tidy" / "cli.py"
    ).read_text(encoding="utf-8")
    for needle in ("--e3", "--agreement-gate", "--classification-mode"):
        assert needle not in parser_source

    args = app.parse_args([])
    assert not hasattr(args, "classification_mode")
    assert not hasattr(args, "e3")


def test_cli_help_lists_only_the_existing_documented_flags(capsys) -> None:
    import pytest

    with pytest.raises(SystemExit):
        app.parse_args(["--help"])
    output = capsys.readouterr().out
    for flag in (
        "--group",
        "--no-group",
        "--read-contents",
        "--no-read-contents",
        "--allow-remote-content",
        "--apply",
        "--undo",
        "--yes",
    ):
        assert flag in output
    for absent in ("--e3", "--agreement-gate", "--classification-mode"):
        assert absent not in output
