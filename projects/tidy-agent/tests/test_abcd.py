from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from evals.run_structured_abcd import (
    CONDITIONS,
    PROJECT_ROOT,
    _aggregate_runs,
    _canonical_digest,
    _paired_analysis,
    _render_report,
    _verify_dataset_manifest,
    parse_args,
)
from evals.runtime_only_fixture_files import RUNTIME_ONLY_FILES


def _category(
    predicted: str,
    *,
    correct: bool,
    peeked: bool = False,
    grouped: bool = False,
) -> dict:
    return {
        "status": "ok",
        "overall_accuracy": float(correct),
        "decision_rate": float(predicted != "_ToReview"),
        "decided_accuracy": float(correct and predicted != "_ToReview"),
        "review_rate": float(predicted == "_ToReview"),
        "overall_total": 1,
        "review_count": int(predicted == "_ToReview"),
        "cases": [
            {
                "filename": "mystery",
                "predicted": predicted,
                "correct": correct,
                "unresolved": True,
                "mode": "agent",
            }
        ],
        "all_destinations": {"mystery": predicted},
        "grouped_sources": ["mystery"] if grouped else [],
        "peeked_sources": ["mystery"] if peeked else [],
        "deterministic_rule_sources": [],
        "classification_source_count": 1,
        "class_latency_seconds": 1.0,
        "class_input_tokens": 10,
        "class_completion_tokens": 2,
        "classification_requests": 2 if peeked else 1,
        "json_object_responses": 2 if peeked else 1,
    }


def _run(condition: str, category: dict, repetition: int = 1) -> dict:
    metrics = category
    if CONDITIONS[condition]["group"]:
        metrics = {
            "status": "ok",
            "category": category,
            "input_tokens": 20,
            "completion_tokens": 4,
            "group_latency_seconds": 0.5,
            "group_input_tokens": 10,
            "group_completion_tokens": 2,
            "grouped_total": int(bool(category["grouped_sources"])),
            "groups_proposed": 1,
            "groups_accepted": 1,
            "groups_discarded": 0,
            "groups_rejected": 0,
            "clustering_purity": 1.0,
            "group_cohesion": 1.0,
            "scatter_in_group": 0,
            "grouped_rule_resolved_sources": [],
            "grouped_unresolved_sources": category["grouped_sources"],
        }
    return {
        "condition": condition,
        "repetition": repetition,
        "status": "ok",
        "total_run_latency_seconds": 2.0,
        "metrics": metrics,
    }


@pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "'report:final' (a real, empty dev-fixture case; see evals/expected.yaml) "
        "contains a colon, which is illegal in NTFS filenames and is read as an "
        "Alternate Data Stream separator on Windows. It cannot be materialized "
        "here even under tmp_path, so this manifest check does not run on "
        "Windows. See evals/runtime_only_fixture_files.py."
    ),
)
def test_committed_dev_manifest_matches_every_canonical_file(tmp_path: Path) -> None:
    staged_fixture = tmp_path / "fixture"
    shutil.copytree(PROJECT_ROOT / "evals/fixture", staged_fixture)
    for name, size in RUNTIME_ONLY_FILES.items():
        assert size == 0, f"{name}: only empty runtime-only fixtures are supported here"
        (staged_fixture / name).write_bytes(b"")

    dev = _verify_dataset_manifest(
        PROJECT_ROOT / "evals/dev/fixture_manifest.json",
        staged_fixture,
        PROJECT_ROOT / "evals/expected.yaml",
    )

    assert len(dev["files"]) == 76


def test_committed_holdout_manifest_matches_every_canonical_file() -> None:
    holdout = _verify_dataset_manifest(
        PROJECT_ROOT / "evals/holdout/fixture_manifest.json",
        PROJECT_ROOT / "evals/holdout/fixture",
        PROJECT_ROOT / "evals/holdout/expected.yaml",
    )

    assert len(holdout["files"]) == 41
    assert holdout["designation"].startswith("locked holdout")


def test_manifest_self_hash_rejects_tampering(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "one").write_text("evidence", encoding="utf-8")
    expected = tmp_path / "expected.yaml"
    expected.write_text("files:\n  one: [Documents]\n", encoding="utf-8")
    files = [
        {
            "path": "one",
            "size": 8,
            "sha256": "ee8250fb76e094b34b471f13a73dbbe51d1ae142e9df59d7c0d31ec20f0a0a8e",
        }
    ]
    ground_truth = {
        "path": "expected.yaml",
        "size": expected.stat().st_size,
        "sha256": "wrong",
    }
    manifest = {
        "files": files,
        "ground_truth": ground_truth,
        "dataset_sha256": _canonical_digest(
            {"files": files, "ground_truth": ground_truth}
        ),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        _verify_dataset_manifest(manifest_path, fixture, expected)
    except RuntimeError as error:
        assert "ground truth" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("tampered manifest was accepted")


def test_aggregate_preserves_repeated_values_and_protocol_counts() -> None:
    runs = [
        _run("A", _category("_ToReview", correct=True), repetition=1),
        _run("A", _category("Documents", correct=True), repetition=2),
    ]

    summary = _aggregate_runs(runs)

    accuracy = summary["A"]["metrics"]["strict_category_accuracy"]
    assert accuracy["individual_values"] == [1.0, 1.0]
    assert summary["A"]["reliability_totals"]["classification_requests"] == 2


def test_paired_content_analysis_records_review_to_correct() -> None:
    runs = [
        _run("A", _category("_ToReview", correct=False)),
        _run("B", _category("_ToReview", correct=False)),
        _run("C", _category("Documents", correct=True, peeked=True)),
        _run("D", _category("Documents", correct=True, peeked=True)),
    ]
    expected = {"files": {"mystery": ["Documents"]}, "groups": {}, "scatter": []}

    transitions, per_file = _paired_analysis(runs, expected)

    assert transitions["A_vs_C"] == {"review → correct": 1}
    assert transitions["B_vs_D"] == {"review → correct": 1}
    assert per_file["mystery"]["peeked"]["C"] == [True]


def test_abcd_cli_defaults_to_five_warm_repetitions_without_thinking() -> None:
    args = parse_args([])

    assert args.repetitions == 5
    assert args.think is False


def test_report_renders_complete_four_condition_summary() -> None:
    runs = [
        _run("A", _category("_ToReview", correct=True)),
        _run("B", _category("Documents", correct=True)),
        _run("C", _category("Documents", correct=True, peeked=True)),
        _run("D", _category("Documents", correct=True, peeked=True)),
    ]
    expected = {"files": {"mystery": ["Documents", "_ToReview"]}, "groups": {}}
    summary = _aggregate_runs(runs)
    transitions, _ = _paired_analysis(runs, expected)
    manifest = {
        "source": {"commit": "a" * 40},
        "model": {
            "identifier": "ollama_chat/test",
            "digest": "b" * 64,
            "provider_runtime": "Ollama via LiteLLM",
            "runtime_version": "test",
            "endpoint": "remote endpoint (address redacted)",
            "structured_output_mode": "json_object",
        },
        "execution": {"repetitions": 1},
        "dataset": {
            "fixture_file_count": 1,
            "fixture_dataset_sha256": "c" * 64,
            "ground_truth_sha256": "d" * 64,
        },
    }

    report = _render_report(
        experiment_id="structured-test",
        manifest=manifest,
        summary=summary,
        transitions=transitions,
        errors=[],
    )

    assert "## Core experimental questions" in report
    assert "Default recommendation" in report
    assert "locked holdout" in report
