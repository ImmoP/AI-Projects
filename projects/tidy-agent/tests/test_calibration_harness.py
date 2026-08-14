"""Offline integration tests for the E0-E3 calibration harness.

Every model call goes through ``FakeModel`` (in-process, deterministic,
scripted responses). No Ollama dependency, and nothing here touches the
consumed Holdout — only the reusable ``evals/calibration`` fixture path
default is exercised, and only through ``parse_args`` (no fixture files are
read by these tests).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from evals.run_structured_calibration import (
    CONDITIONS,
    COUNTERBALANCED_SCHEDULE,
    _condition_summary,
    _stability,
    aggregate,
    parse_args,
    run_e0,
    run_e1,
    run_e2,
    run_e3,
    score_run,
)

REVIEW = "_ToReview"
CATEGORIES_WITH_REVIEW = ["Documents", "Code", REVIEW]
REAL_CATEGORIES = ["Documents", "Code"]
METADATA = [
    {"name": "one", "size_bytes": 0},
    {"name": "two", "size_bytes": 0},
    {"name": "three", "size_bytes": 0},
]


class FakeModel:
    structured_output_mode = "json_schema"

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        response = self.responses.pop(0)
        return SimpleNamespace(
            content=response,
            token_usage=SimpleNamespace(input_tokens=5, output_tokens=2),
        )


def test_run_e0_matches_production_metadata_control_shape() -> None:
    model = FakeModel(
        '{"requests":[]}',
        '{"decisions":[{"source":"one","category":"Documents"},'
        '{"source":"two","category":"Code"},'
        '{"source":"three","category":"_ToReview"}]}',
    )

    final, detail, telemetry = run_e0(model, METADATA, CATEGORIES_WITH_REVIEW)

    assert final == {"one": "Documents", "two": "Code", "three": "_ToReview"}
    assert len(model.calls) == 2
    assert telemetry["classification_requests"] == 2
    assert {item["filename"] for item in detail} == {"one", "two", "three"}


def test_run_e1_disagreement_reviews_only_the_disagreeing_file() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","category":"Documents"},'
        '{"source":"two","category":"Code"},'
        '{"source":"three","category":"Documents"}]}',
        # pass 2 sees the reversed order but reports by source name regardless
        '{"decisions":[{"source":"one","category":"Documents"},'
        '{"source":"two","category":"Documents"},'
        '{"source":"three","category":"Documents"}]}',
    )

    final, detail, telemetry = run_e1(model, METADATA, CATEGORIES_WITH_REVIEW, review_directory=REVIEW)

    assert final == {"one": "Documents", "two": REVIEW, "three": "Documents"}
    assert len(model.calls) == 2
    by_source = {item["filename"]: item for item in detail}
    assert by_source["two"]["agreement"] == "disagree"
    assert by_source["one"]["agreement"] == "agree"
    assert telemetry["classification_requests"] == 2


def test_run_e1_second_pass_receives_reversed_order() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","category":"Documents"},'
        '{"source":"two","category":"Documents"},'
        '{"source":"three","category":"Documents"}]}',
        '{"decisions":[{"source":"one","category":"Documents"},'
        '{"source":"two","category":"Documents"},'
        '{"source":"three","category":"Documents"}]}',
    )

    run_e1(model, METADATA, CATEGORIES_WITH_REVIEW, review_directory=REVIEW)

    def _filename_block(prompt: str) -> str:
        return prompt.split("<FILENAME_DATA>")[1].split("</FILENAME_DATA>")[0]

    first_block = _filename_block(model.calls[0]["messages"][0]["content"][0]["text"])
    second_block = _filename_block(model.calls[1]["messages"][0]["content"][0]["text"])
    assert first_block.split() == ["one", "two", "three"]
    assert second_block.split() == ["three", "two", "one"]


def test_run_e2_resolves_explicit_decisions() -> None:
    model = FakeModel(
        '{"decisions":['
        '{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"two","decision":"review","category":null},'
        '{"source":"three","decision":"classify","category":"Code"}]}'
    )

    final, detail, telemetry = run_e2(model, METADATA, REAL_CATEGORIES, review_directory=REVIEW)

    assert final == {"one": "Documents", "two": REVIEW, "three": "Code"}
    assert len(model.calls) == 1
    by_source = {item["filename"]: item for item in detail}
    assert by_source["two"]["pass1_decision"] == "review"
    assert by_source["one"]["pass1_category"] == "Documents"


def test_run_e3_accepts_only_matching_classify_pairs() -> None:
    model = FakeModel(
        '{"decisions":['
        '{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"two","decision":"classify","category":"Documents"},'
        '{"source":"three","decision":"review","category":null}]}',
        '{"decisions":['
        '{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"two","decision":"classify","category":"Code"},'
        '{"source":"three","decision":"classify","category":"Documents"}]}',
    )

    final, detail, telemetry = run_e3(model, METADATA, REAL_CATEGORIES, review_directory=REVIEW)

    assert final == {"one": "Documents", "two": REVIEW, "three": REVIEW}
    by_source = {item["filename"]: item for item in detail}
    assert by_source["one"]["agreement"] == "agree_classify"
    assert by_source["two"]["agreement"] == "disagree_classify"
    assert by_source["three"]["agreement"] == "review_involved"


# --- Aggregation pipeline, driven by hand-built raw run records -----------


def _raw_run(condition: str, repetition: int, sources_final: dict, detail: list[dict]) -> dict:
    return {
        "experiment_id": "test",
        "repetition": repetition,
        "condition": condition,
        "status": "ok",
        "metrics": {
            "status": "ok",
            "sources": list(sources_final),
            "final": sources_final,
            "detail": detail,
        },
    }


EXPECTED = {
    "clearly_documents": ["Documents"],
    "clearly_code": ["Code"],
    "ambiguous_only_review": [REVIEW],
    "either_doc_or_code": ["Documents", "Code"],
}


def test_score_run_flags_unsafe_and_abstention_cases() -> None:
    run = _raw_run(
        "E2",
        1,
        {
            "clearly_documents": "Documents",
            "clearly_code": REVIEW,
            "ambiguous_only_review": "Documents",
            "either_doc_or_code": "Code",
        },
        [],
    )

    cases = score_run(run, EXPECTED, review_directory=REVIEW)
    by_name = {c["filename"]: c for c in cases}

    assert by_name["clearly_documents"]["correct"] is True
    assert by_name["clearly_code"]["false_abstention"] is True
    assert by_name["ambiguous_only_review"]["unsafe_classification_of_review_case"] is True
    assert by_name["either_doc_or_code"]["correct"] is True


def test_condition_summary_computes_unsafe_automation_and_review_quality() -> None:
    run = _raw_run(
        "E2",
        1,
        {
            "clearly_documents": "Documents",
            "clearly_code": REVIEW,
            "ambiguous_only_review": "Documents",
            "either_doc_or_code": "Code",
        },
        [],
    )
    cases = score_run(run, EXPECTED, review_directory=REVIEW)
    summary = _condition_summary([{**run, "cases": cases}])

    # 1 incorrect automatic (ambiguous_only_review classified) out of 4 files
    assert summary["unsafe_automation_rate"] == 1 / 4
    # automation_coverage = non-review decisions / all files = 3/4
    assert summary["automation_coverage"] == 3 / 4
    # only true abstention is none here (ambiguous file was NOT reviewed)
    assert summary["review_quality"]["review_recall"] == 0.0
    # nothing was predicted _ToReview except clearly_code, which is a false abstention
    assert summary["review_quality"]["review_precision"] == 0.0
    assert summary["cost_scenarios"]["balanced"] == 5 * 1 + 1 * 1  # one wrong, one review


def test_stability_detects_unstable_category_and_review_flip() -> None:
    run1 = _raw_run("E1", 1, {"a": "Documents", "b": "Code"}, [])
    run2 = _raw_run("E1", 2, {"a": "Code", "b": "Code"}, [])
    run3 = _raw_run("E1", 3, {"a": "Documents", "b": REVIEW}, [])
    expected = {"a": ["Documents"], "b": ["Code"]}
    scored = [{**r, "cases": score_run(r, expected, review_directory=REVIEW)} for r in (run1, run2, run3)]

    stability = _stability(scored)

    assert "a" in stability["unstable_files"]["unstable_category"]
    assert "b" in stability["unstable_files"]["unstable_review_classify_decision"]


def test_aggregate_end_to_end_over_two_conditions() -> None:
    expected = {"a": ["Documents"], "b": [REVIEW]}
    runs = [
        _raw_run("E0", 1, {"a": "Documents", "b": REVIEW}, []),
        _raw_run(
            "E1",
            1,
            {"a": "Documents", "b": REVIEW},
            [
                {"filename": "a", "pass1_decision": "classify", "pass1_category": "Documents",
                 "pass2_decision": "classify", "pass2_category": "Documents", "agreement": "agree"},
                {"filename": "b", "pass1_decision": None, "pass1_category": None,
                 "pass2_decision": "classify", "pass2_category": "Documents", "agreement": "pass1_invalid"},
            ],
        ),
    ]

    result = aggregate(runs, expected, review_directory=REVIEW)

    assert set(result["summary"]) == {"E0", "E1"}
    assert result["summary"]["E1"]["abstention"]["disagreements_between_passes"] == 1
    assert result["summary"]["E0"]["explicit_review"]["applicable"] is False
    assert result["summary"]["E1"]["explicit_review"]["applicable"] is False


# --- Structural / defaults --------------------------------------------------


def test_counterbalanced_schedule_covers_all_four_conditions_each_rotation() -> None:
    for rotation in COUNTERBALANCED_SCHEDULE:
        assert set(rotation) == set(CONDITIONS)
        assert len(rotation) == 4


def test_parse_args_defaults_to_calibration_fixture_not_holdout() -> None:
    args = parse_args([])

    assert args.repetitions == 5
    assert args.think is False
    assert "calibration" in str(args.fixture)
    assert "holdout" not in str(args.fixture).lower()
    assert "holdout" not in str(args.fixture_manifest).lower()


_HARNESS_SOURCE = Path(__file__).parents[1].joinpath(
    "evals", "run_structured_calibration.py"
).read_text(encoding="utf-8")


def test_harness_never_reads_holdout_fixture_or_reads_content() -> None:
    assert "evals/holdout" not in _HARNESS_SOURCE
    assert "read_contents=True" not in _HARNESS_SOURCE
    assert "allow_remote_content=True" not in _HARNESS_SOURCE
    assert "peek_tool" not in _HARNESS_SOURCE
