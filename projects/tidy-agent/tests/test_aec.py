from __future__ import annotations

from pathlib import Path

from evals.run_structured_aec import (
    CONDITIONS,
    COUNTERBALANCED_SCHEDULE,
    _aggregate,
    _content_evidence_summary,
    _paired_analysis,
    parse_args,
)
from tidy.classification import build_peek_candidates
from tidy.rules import classify_directory, load_rules
from tidy.tools import metadata_for_names


def _metrics(
    prediction: str,
    *,
    correct: bool,
    requested: tuple[str, ...] = (),
    authorized: tuple[str, ...] = (),
    nonempty: bool = False,
) -> dict:
    file_metrics = (
        {
            "mystery": {
                "readable": True,
                "nonempty": nonempty,
                "bytes_read": 10 if nonempty else 0,
                "chars_returned": 8 if nonempty else 0,
            }
        }
        if authorized
        else {}
    )
    return {
        "status": "ok",
        "overall_accuracy": float(correct),
        "overall_correct": int(correct),
        "overall_total": 1,
        "decision_rate": float(prediction != "_ToReview"),
        "decided_accuracy": float(correct and prediction != "_ToReview"),
        "review_rate": float(prediction == "_ToReview"),
        "class_latency_seconds": 1.0,
        "class_input_tokens": 10,
        "class_completion_tokens": 2,
        "classification_requests": 2 if requested else 1,
        "json_object_responses": 2 if requested else 1,
        "peek_requests_model": len(requested),
        "peek_requests_unique": len(requested),
        "peek_requests_authorized": len(authorized),
        "peek_requests_control_selected": len(requested) if not authorized else 0,
        "peek_calls": len(authorized),
        "peek_readable": len(authorized),
        "peek_nonempty": int(nonempty),
        "peek_bytes_read": 10 if nonempty else 0,
        "peek_chars_returned": 8 if nonempty else 0,
        "peek_requested_sources": list(requested),
        "peeked_sources": list(authorized),
        "peek_file_metrics": file_metrics,
        "cases": [
            {
                "filename": "mystery",
                "predicted": prediction,
                "correct": correct,
                "allowed": ["Documents"],
            }
        ],
    }


def _run(condition: str, metrics: dict, repetition: int = 1) -> dict:
    return {
        "condition": condition,
        "repetition": repetition,
        "status": "ok",
        "total_run_latency_seconds": 2.0,
        "metrics": metrics,
    }


def test_aec_defaults_to_five_counterbalanced_warm_repetitions() -> None:
    args = parse_args([])

    assert args.repetitions == 5
    assert args.think is False
    assert set(CONDITIONS) == {"A", "E", "C"}
    assert COUNTERBALANCED_SCHEDULE[:3] == (
        ("A", "E", "C"),
        ("E", "C", "A"),
        ("C", "A", "E"),
    )


def test_development_candidate_filter_excludes_empty_fixture_placeholders() -> None:
    root = Path(__file__).parents[1] / "evals/fixture"
    _, unresolved = classify_directory(root, load_rules())
    candidates = build_peek_candidates(metadata_for_names(root, unresolved))

    assert {item["source"] for item in candidates} == {
        "geraetedump",
        "sitzungsprotokoll_q3",
        "steuerbescheid_2024",
    }
    assert all(item["size_bytes"] > 0 for item in candidates)
    assert all(
        set(item) == {"source", "extension", "size_bytes", "content_reader"}
        for item in candidates
    )


def test_aggregate_defines_informative_rate_as_nonempty_over_authorized() -> None:
    runs = [
        _run("A", _metrics("_ToReview", correct=False)),
        _run(
            "E",
            _metrics("_ToReview", correct=False, requested=("mystery",)),
        ),
        _run(
            "C",
            _metrics(
                "Documents",
                correct=True,
                requested=("mystery",),
                authorized=("mystery",),
                nonempty=True,
            ),
        ),
    ]

    summary = _aggregate(runs)

    assert summary["C"]["metrics"]["informative_peek_rate"]["mean"] == 1.0
    assert summary["E"]["totals"]["peek_requests_authorized"] == 0
    assert summary["E"]["totals"]["peek_requests_control_selected"] == 1


def test_content_improvement_requires_nonempty_delivered_content() -> None:
    runs = [
        _run("A", _metrics("_ToReview", correct=False)),
        _run(
            "E",
            _metrics("_ToReview", correct=False, requested=("mystery",)),
        ),
        _run(
            "C",
            _metrics(
                "Documents",
                correct=True,
                requested=("mystery",),
                authorized=("mystery",),
                nonempty=True,
            ),
        ),
    ]

    transitions, evidence = _paired_analysis(runs)
    content = _content_evidence_summary(evidence)

    assert transitions["A_vs_E"] == {"unchanged": 1}
    assert transitions["E_vs_C"] == {"review → correct": 1}
    assert content["nonempty_content_improved_count"] == 1
    assert content["improved_files"] == ["mystery"]


def test_changed_unpeeked_file_is_not_claimed_as_content_improvement() -> None:
    runs = [
        _run("A", _metrics("_ToReview", correct=False)),
        _run("E", _metrics("_ToReview", correct=False)),
        _run("C", _metrics("Documents", correct=True)),
    ]

    _transitions, evidence = _paired_analysis(runs)
    content = _content_evidence_summary(evidence)

    assert content["nonempty_content_changed_count"] == 0
    assert content["improved_files"] == []
