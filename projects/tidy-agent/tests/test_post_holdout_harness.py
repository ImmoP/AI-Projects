"""Offline integration tests for the E3/E4/E5 post-Holdout-v2 harness.

Every model call goes through ``FakeModel`` (in-process, deterministic,
scripted responses). No Ollama dependency, and nothing here touches either
consumed Holdout -- only ``run_condition`` and the pure scoring/aggregation
functions are exercised, exactly as ``tests/test_calibration_harness.py``
exercises the E0-E3 harness. Importing or testing this module performs no
live inference: only running ``evals/run_post_holdout_development.py``
directly as a script does that, and no test here does so.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from evals.run_post_holdout_development import (
    CONDITIONS,
    COUNTERBALANCED_SCHEDULE,
    FIXTURES,
    _condition_summary,
    _e4_metrics,
    _e5_metrics,
    _stability,
    aggregate,
    parse_args,
    run_condition,
    score_run,
)

REVIEW = "_ToReview"
REAL_CATEGORIES = ["Documents", "Code", "Images", "Archives", "Installers"]
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


# --- run_condition dispatch --------------------------------------------------


def test_run_condition_e3_makes_exactly_two_calls() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )

    final, _detail, _telemetry = run_condition(
        "E3", model, METADATA[:1], real_categories=REAL_CATEGORIES, review_directory=REVIEW
    )

    assert final == {"one": "Documents"}
    assert len(model.calls) == 2


def test_run_condition_e4_reuses_e3_output_with_no_extra_call() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )

    final, detail, _telemetry = run_condition(
        "E4", model, METADATA[:1], real_categories=REAL_CATEGORIES, review_directory=REVIEW
    )

    assert len(model.calls) == 2  # E4 adds no model call beyond E3's two
    assert final["one"] == "Documents"
    assert detail[0]["veto_reason_code"] == "NO_CONFLICT"


def test_run_condition_e5_uses_role_separated_verifier() -> None:
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
        '{"decisions":[{"source":"one","decision":"accept","category":"Documents"}]}',
    )

    final, detail, _telemetry = run_condition(
        "E5", model, METADATA[:1], real_categories=REAL_CATEGORIES, review_directory=REVIEW
    )

    assert len(model.calls) == 2
    assert final == {"one": "Documents"}
    assert detail[0]["verifier_decision"] == "accept"


def test_run_condition_unknown_raises() -> None:
    try:
        run_condition("E9", FakeModel(), METADATA, real_categories=REAL_CATEGORIES, review_directory=REVIEW)
    except ValueError as exc:
        assert "E9" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# --- Scoring pipeline --------------------------------------------------------


def _raw_run(condition: str, fixture: str, repetition: int, sources_final: dict, detail: list[dict]) -> dict:
    return {
        "experiment_id": "test",
        "fixture": fixture,
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


def test_score_run_flags_correct_and_incorrect_automatic() -> None:
    run = _raw_run(
        "E3",
        "boundary_calibration",
        1,
        {
            "clearly_documents": "Documents",
            "clearly_code": REVIEW,
            "ambiguous_only_review": "Documents",
            "either_doc_or_code": "Code",
        },
        [],
    )

    cases = score_run(run["metrics"], EXPECTED, review_directory=REVIEW)
    by_name = {c["filename"]: c for c in cases}

    assert by_name["clearly_documents"]["correct"] is True
    assert by_name["clearly_code"]["false_abstention"] is True
    assert by_name["ambiguous_only_review"]["incorrect_automatic"] is True
    assert by_name["either_doc_or_code"]["correct"] is True


def test_condition_summary_primary_metrics() -> None:
    run = _raw_run(
        "E3",
        "boundary_calibration",
        1,
        {
            "clearly_documents": "Documents",
            "clearly_code": REVIEW,
            "ambiguous_only_review": "Documents",
            "either_doc_or_code": "Code",
        },
        [],
    )
    cases = score_run(run["metrics"], EXPECTED, review_directory=REVIEW)
    summary = _condition_summary("E3", [{**run, "cases": cases}])

    assert summary["primary"]["raw_counts"]["incorrect_automatic"] == 1
    assert summary["primary"]["unsafe_automation_rate"] == 1 / 4
    assert summary["primary"]["automation_coverage"] == 3 / 4
    assert summary["cost_scenarios"]["balanced"] == 5 * 1 + 1 * 1


def test_e4_metrics_true_and_false_positive_vetoes() -> None:
    detail = [
        {"filename": "wrong_but_vetoed", "e3_category": "Archives", "veto_applicable": True},
        {"filename": "right_but_vetoed", "e3_category": "Documents", "veto_applicable": True},
        {"filename": "right_and_kept", "e3_category": "Documents", "veto_applicable": True},
        {"filename": "wrong_and_kept", "e3_category": "Archives", "veto_applicable": True},
    ]
    final = {
        "wrong_but_vetoed": REVIEW,
        "right_but_vetoed": REVIEW,
        "right_and_kept": "Documents",
        "wrong_and_kept": "Archives",
    }
    expected = {
        "wrong_but_vetoed": ["Documents"],
        "right_but_vetoed": ["Documents"],
        "right_and_kept": ["Documents"],
        "wrong_and_kept": ["Documents"],
    }
    run = _raw_run("E4", "boundary_calibration", 1, final, detail)
    cases = score_run(run["metrics"], expected, review_directory=REVIEW)

    metrics = _e4_metrics(cases)

    assert metrics["e3_automatic_candidates_presented_to_veto"] == 4
    assert metrics["e4_vetoed"] == 2
    assert metrics["e4_accepted"] == 2
    assert metrics["true_positive_vetoes"] == 1  # wrong_but_vetoed
    assert metrics["false_positive_vetoes"] == 1  # right_but_vetoed
    assert metrics["unsafe_e3_errors_surviving_veto"] == 1  # wrong_and_kept
    assert metrics["veto_precision"] == 0.5
    assert metrics["veto_recall_for_e3_automatic_errors"] == 0.5  # 1 of 2 e3 errors caught


def test_e5_metrics_accept_and_rejection_breakdown() -> None:
    detail = [
        {
            "filename": "accepted_right",
            "classifier_decision": "classify",
            "classifier_category": "Documents",
            "verifier_decision": "accept",
            "reason_code": "VERIFIER_ACCEPT",
        },
        {
            "filename": "rejected_was_wrong",
            "classifier_decision": "classify",
            "classifier_category": "Archives",
            "verifier_decision": "review",
            "reason_code": "VERIFIER_REVIEW",
        },
        {
            "filename": "rejected_was_right",
            "classifier_decision": "classify",
            "classifier_category": "Documents",
            "verifier_decision": "review",
            "reason_code": "VERIFIER_REVIEW",
        },
        {
            "filename": "classifier_abstained",
            "classifier_decision": "review",
            "classifier_category": None,
            "verifier_decision": None,
            "reason_code": "CLASSIFIER_REVIEW",
        },
    ]
    final = {
        "accepted_right": "Documents",
        "rejected_was_wrong": REVIEW,
        "rejected_was_right": REVIEW,
        "classifier_abstained": REVIEW,
    }
    expected = {
        "accepted_right": ["Documents"],
        "rejected_was_wrong": ["Documents"],
        "rejected_was_right": ["Documents"],
        "classifier_abstained": ["Documents"],
    }
    run = _raw_run("E5", "boundary_calibration", 1, final, detail)
    cases = score_run(run["metrics"], expected, review_directory=REVIEW)

    metrics = _e5_metrics(cases)

    assert metrics["classifier_classify_count"] == 3
    assert metrics["classifier_review_count"] == 1
    assert metrics["verifier_accept_count"] == 1
    assert metrics["verifier_review_count"] == 2
    assert metrics["accepted_correct"] == 1
    assert metrics["accepted_wrong"] == 0
    assert metrics["rejected_classifier_errors"] == 1  # rejected_was_wrong
    assert metrics["rejected_correct_classifier_proposals"] == 1  # rejected_was_right


def test_stability_flags_unstable_predictions_across_repetitions() -> None:
    run1 = _raw_run("E3", "boundary_calibration", 1, {"a": "Documents", "b": "Code"}, [])
    run2 = _raw_run("E3", "boundary_calibration", 2, {"a": "Code", "b": "Code"}, [])
    expected = {"a": ["Documents"], "b": ["Code"]}
    scored = [
        {**r, "cases": score_run(r["metrics"], expected, review_directory=REVIEW)} for r in (run1, run2)
    ]

    stability = _stability(scored)

    assert stability["unique_file_count"] == 2
    assert "a" in stability["unstable_files"]
    assert "b" not in stability["unstable_files"]


def test_aggregate_reports_unique_file_denominator_not_repetition_count() -> None:
    expected = {"a": ["Documents"], "b": [REVIEW]}
    runs = [
        _raw_run("E3", "boundary_calibration", 1, {"a": "Documents", "b": REVIEW}, []),
        _raw_run("E3", "boundary_calibration", 2, {"a": "Documents", "b": REVIEW}, []),
    ]

    result = aggregate(runs, expected, review_directory=REVIEW)

    # Two repetitions over two files -> four scored observations, but the
    # semantic denominator (unique files) is still reported separately.
    assert result["summary"]["E3"]["primary"]["total_files_scored"] == 4
    assert result["stability"]["E3"]["unique_file_count"] == 2


# --- Structural / defaults --------------------------------------------------


def test_counterbalanced_schedule_covers_exactly_e3_e4_e5() -> None:
    for rotation in COUNTERBALANCED_SCHEDULE:
        assert set(rotation) == {"E3", "E4", "E5"}
        assert len(rotation) == 3
    assert set(CONDITIONS) == {"E3", "E4", "E5"}


def test_parse_args_defaults_to_both_development_fixtures() -> None:
    args = parse_args([])

    assert args.repetitions == 5
    assert args.think is False
    assert set(args.fixtures) == {"calibration", "boundary_calibration"}
    assert set(FIXTURES) == {"calibration", "boundary_calibration"}
    for paths in FIXTURES.values():
        assert "holdout" not in str(paths["fixture"]).lower()
        assert "holdout" not in str(paths["expected"]).lower()
        assert "holdout" not in str(paths["manifest"]).lower()


_HARNESS_SOURCE = Path(__file__).parents[1].joinpath(
    "evals", "run_post_holdout_development.py"
).read_text(encoding="utf-8")


def test_harness_never_reads_either_holdout_fixture_or_content() -> None:
    assert "evals/holdout" not in _HARNESS_SOURCE
    assert "holdout_v2" not in _HARNESS_SOURCE
    assert "read_contents=True" not in _HARNESS_SOURCE
    assert "allow_remote_content=True" not in _HARNESS_SOURCE
    assert "peek_tool" not in _HARNESS_SOURCE


def test_harness_only_runs_live_inference_as_a_direct_script() -> None:
    assert 'if __name__ == "__main__":' in _HARNESS_SOURCE
    # main()/run_experiment() must not be invoked at module import time.
    import evals.run_post_holdout_development as module

    assert not hasattr(module, "_auto_ran")
