"""Frozen, counterbalanced E3/E4/E5 post-Holdout-v2 Development comparison.

Metadata-only: no ``--read-contents``, no ``--allow-remote-content``, no
peek tool is ever constructed. This does not touch either consumed Holdout
-- only the two reusable Development fixtures (``evals/calibration`` and
``evals/boundary_calibration``) are read here.

E3 is the frozen production comparator, run unchanged via
``evals.post_holdout_candidates.run_e3`` (itself built only from frozen
production primitives re-exported from ``tidy.classification``). E4 and E5
are the two Development-only candidates from that same module; see its
module docstring for the full per-candidate design and state tables.

This module implements scoring and a future live-experiment entry point; it
performs no model call, no Ollama/LiteLLM request, and no warmup by being
imported or by any test in this repository importing it. Only running it
directly as a script (``python evals/run_post_holdout_development.py``)
performs live inference, and no other file in this repository does that on
its behalf.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from tidy.agent import DEFAULT_MODEL_ID, build_model  # noqa: E402
from tidy.rules import classify_directory, load_rules  # noqa: E402
from tidy.tools import metadata_for_names  # noqa: E402

from evals.post_holdout_candidates import run_e3, run_e4, run_e5  # noqa: E402
from evals.run_evals import _load_expected, _warmup_worker, run_in_subprocess  # noqa: E402
from evals.run_structured_abcd import (  # noqa: E402
    _git,
    _installed_versions,
    _model_manifest,
    _sha256,
    _stage_fixture,
    _verify_dataset_manifest,
)

CONDITIONS = {
    "E3": "production comparator: explicit abstention + two-pass agreement gate (unchanged)",
    "E4": "E3 + deterministic ambiguity/conflict veto (no additional model call)",
    "E5": "role-separated classifier + verifier (classifier pass, then verifier pass)",
}
COUNTERBALANCED_SCHEDULE = (
    ("E3", "E4", "E5"),
    ("E4", "E5", "E3"),
    ("E5", "E3", "E4"),
)
COST_SCENARIOS = {
    "safety_heavy": {"incorrect_automatic": 10, "review": 1, "correct": 0},
    "balanced": {"incorrect_automatic": 5, "review": 1, "correct": 0},
    "coverage_heavy": {"incorrect_automatic": 3, "review": 1, "correct": 0},
}
SECURITY_ZERO_FIELDS = (
    "peek_requests_authorized",
    "content_unavailable",
)

FIXTURES = {
    "calibration": {
        "fixture": PROJECT_ROOT / "evals/calibration/fixture",
        "expected": PROJECT_ROOT / "evals/calibration/expected.yaml",
        "manifest": PROJECT_ROOT / "evals/calibration/fixture_manifest.json",
    },
    "boundary_calibration": {
        "fixture": PROJECT_ROOT / "evals/boundary_calibration/fixture",
        "expected": PROJECT_ROOT / "evals/boundary_calibration/expected.yaml",
        "manifest": PROJECT_ROOT / "evals/boundary_calibration/fixture_manifest.json",
    },
}


def run_condition(
    condition: str,
    model: Any,
    metadata: list[dict[str, Any]],
    *,
    real_categories: list[str],
    review_directory: str,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    if condition == "E3":
        return run_e3(model, metadata, real_categories, review_directory=review_directory)
    if condition == "E4":
        return run_e4(model, metadata, real_categories, review_directory=review_directory)
    if condition == "E5":
        return run_e5(model, metadata, real_categories, review_directory=review_directory)
    raise ValueError(f"unknown condition: {condition}")


# --- Subprocess worker (live path only; not exercised offline) -------------


def _post_holdout_worker(
    result_queue: Any,
    fixture: str,
    condition: str,
    model_id: str | None,
    think: bool | None,
) -> None:
    started = time.perf_counter()
    try:
        rules = load_rules()
        moves, unresolved = classify_directory(Path(fixture), rules)
        if moves:
            raise RuntimeError(
                "post-holdout fixtures must be fully unresolved by deterministic "
                f"rules; {len(moves)} file(s) resolved unexpectedly"
            )
        metadata = metadata_for_names(fixture, unresolved)
        real_categories = list(rules.categories)
        model = build_model(model_id, think=think)
        final, detail, telemetry = run_condition(
            condition,
            model,
            metadata,
            real_categories=real_categories,
            review_directory=rules.review_directory,
        )
        result_queue.put(
            {
                "status": "ok",
                "sources": [str(item["name"]) for item in metadata],
                "final": final,
                "detail": detail,
                "telemetry": telemetry,
                "latency_seconds": time.perf_counter() - started,
            }
        )
    except Exception as exc:  # isolated in a child; converted to report data
        result_queue.put(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "latency_seconds": time.perf_counter() - started,
            }
        )


# --- Provenance --------------------------------------------------------


def _git_source() -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "live post-holdout development evaluation requires a clean Git "
            "worktree; commit or remove all changes first"
        )
    commit = _git("rev-parse", "HEAD")
    script = Path(__file__).resolve()
    return {
        "commit": commit,
        "short_commit": commit[:12],
        "dirty": False,
        "branch": _git("branch", "--show-current"),
        "evaluation_script": "evals/run_post_holdout_development.py",
        "evaluation_script_sha256": _sha256(script),
        "post_holdout_candidates_sha256": _sha256(
            PROJECT_ROOT / "evals/post_holdout_candidates.py"
        ),
    }


# --- Scoring (mirrors evals/run_structured_calibration.py's shape) --------


def _score_case(
    filename: str,
    allowed: list[str],
    predicted: str,
    *,
    review_directory: str,
) -> dict[str, Any]:
    is_review = predicted == review_directory
    ground_truth_review_only = allowed == [review_directory]
    correct = predicted in allowed
    return {
        "filename": filename,
        "allowed": allowed,
        "predicted": predicted,
        "correct": correct,
        "is_review": is_review,
        "ground_truth_review_only": ground_truth_review_only,
        "incorrect_automatic": (not is_review) and (not correct),
        "true_abstention": is_review and ground_truth_review_only,
        "false_abstention": is_review and not ground_truth_review_only,
    }


def score_run(
    run: dict[str, Any], expected: dict[str, list[str]], *, review_directory: str
) -> list[dict[str, Any]]:
    """Join one run's raw predictions with ground truth. Pure and offline-testable."""
    detail_by_source = {item["filename"]: item for item in run.get("detail", [])}
    cases = []
    for source in run.get("sources", []):
        allowed = expected[source]
        predicted = run["final"].get(source, review_directory)
        case = _score_case(source, allowed, predicted, review_directory=review_directory)
        case.update(
            {key: value for key, value in detail_by_source.get(source, {}).items() if key != "filename"}
        )
        cases.append(case)
    return cases


def _cost(cases: list[dict[str, Any]], weights: dict[str, int]) -> int:
    total = 0
    for case in cases:
        if case["incorrect_automatic"]:
            total += weights["incorrect_automatic"]
        elif case["is_review"]:
            total += weights["review"]
    return total


def _primary_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    correct_automatic = sum(1 for c in cases if not c["is_review"] and c["correct"])
    incorrect_automatic = sum(c["incorrect_automatic"] for c in cases)
    review_count = sum(c["is_review"] for c in cases)
    decided = total - review_count
    ground_truth_review_total = sum(c["ground_truth_review_only"] for c in cases)
    true_abstentions = sum(c["true_abstention"] for c in cases)
    return {
        "total_files_scored": total,
        "raw_counts": {
            "correct_automatic": correct_automatic,
            "incorrect_automatic": incorrect_automatic,
            "review": review_count,
        },
        "strict_accuracy": (correct_automatic + true_abstentions) / total if total else None,
        "unsafe_automation_rate": incorrect_automatic / total if total else None,
        "automation_coverage": decided / total if total else None,
        "review_rate": review_count / total if total else None,
        "accuracy_on_decided": correct_automatic / decided if decided else None,
        "review_recall": (
            true_abstentions / ground_truth_review_total if ground_truth_review_total else None
        ),
        "review_precision": true_abstentions / review_count if review_count else None,
    }


def _real_category_subset_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    subset = [c for c in cases if not c["ground_truth_review_only"]]
    n = len(subset)
    correct_automatic = sum(1 for c in subset if not c["is_review"] and c["correct"])
    wrong_automatic = sum(c["incorrect_automatic"] for c in subset)
    false_review = sum(c["is_review"] for c in subset)
    return {"n": n, "correct_automatic": correct_automatic, "wrong_automatic": wrong_automatic, "false_review": false_review}


def _review_subset_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    subset = [c for c in cases if c["ground_truth_review_only"]]
    n = len(subset)
    correctly_reviewed = sum(c["is_review"] for c in subset)
    incorrectly_automated = sum(1 for c in subset if not c["is_review"])
    return {"n": n, "correctly_reviewed": correctly_reviewed, "incorrectly_automated": incorrectly_automated}


def _e3_gate_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Only meaningful for condition E3 (or E4's embedded e3_* fields)."""
    def bucket(c: dict[str, Any]) -> str:
        d1, d2 = c.get("pass1_decision"), c.get("pass2_decision")
        if d1 is None or d2 is None:
            return "invalid"
        if d1 == "classify" and d2 == "classify":
            cat1, cat2 = c.get("pass1_category"), c.get("pass2_category")
            return "classify_classify_same" if cat1 == cat2 and cat1 else "classify_classify_different"
        if d1 == "classify" and d2 == "review":
            return "classify_review"
        if d1 == "review" and d2 == "classify":
            return "review_classify"
        return "review_review"

    buckets: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        buckets.setdefault(bucket(case), []).append(case)
    same = buckets.get("classify_classify_same", [])
    return {
        "counts": {name: len(items) for name, items in buckets.items()},
        "classify_classify_same_branch": {
            "n": len(same),
            "correct": sum(c["correct"] for c in same),
            "incorrect": sum(not c["correct"] for c in same),
        },
    }


def _e4_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Section 29: does the deterministic veto catch correlated E3 errors
    without reviewing too many correct E3 decisions?"""
    presented = [c for c in cases if c.get("veto_applicable") is True]
    accepted = [c for c in presented if not c["is_review"]]
    vetoed = [c for c in presented if c["is_review"]]
    true_positive_vetoes = sum(1 for c in vetoed if c["e3_category"] not in c["allowed"])
    false_positive_vetoes = sum(1 for c in vetoed if c["e3_category"] in c["allowed"])
    e3_errors = sum(1 for c in presented if c["e3_category"] not in c["allowed"])
    unsafe_errors_surviving_veto = sum(
        1 for c in accepted if c["e3_category"] not in c["allowed"]
    )
    return {
        "e3_automatic_candidates_presented_to_veto": len(presented),
        "e4_accepted": len(accepted),
        "e4_vetoed": len(vetoed),
        "true_positive_vetoes": true_positive_vetoes,
        "false_positive_vetoes": false_positive_vetoes,
        "unsafe_e3_errors_surviving_veto": unsafe_errors_surviving_veto,
        "veto_precision": true_positive_vetoes / len(vetoed) if vetoed else None,
        "veto_recall_for_e3_automatic_errors": (
            true_positive_vetoes / e3_errors if e3_errors else None
        ),
    }


def _e5_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Section 30: does role-separated verification catch systematic
    semantic errors that repeated classification agreement misses?"""
    classifier_classify = sum(1 for c in cases if c.get("classifier_decision") == "classify")
    classifier_review = sum(1 for c in cases if c.get("classifier_decision") == "review")
    verifier_accept = sum(1 for c in cases if c.get("verifier_decision") == "accept")
    verifier_review = sum(1 for c in cases if c.get("verifier_decision") == "review")
    invalid_verifier = sum(
        1
        for c in cases
        if c.get("classifier_decision") == "classify" and c.get("verifier_decision") is None
    )
    accepted_cases = [c for c in cases if c.get("reason_code") == "VERIFIER_ACCEPT"]
    accepted_correct = sum(c["correct"] for c in accepted_cases)
    accepted_wrong = sum(not c["correct"] for c in accepted_cases)
    rejected_by_verifier = [
        c for c in cases if c.get("verifier_decision") == "review" and c.get("classifier_decision") == "classify"
    ]
    rejected_classifier_errors = sum(
        1 for c in rejected_by_verifier if c.get("classifier_category") not in c["allowed"]
    )
    rejected_correct_proposals = sum(
        1 for c in rejected_by_verifier if c.get("classifier_category") in c["allowed"]
    )
    return {
        "classifier_classify_count": classifier_classify,
        "classifier_review_count": classifier_review,
        "verifier_accept_count": verifier_accept,
        "verifier_review_count": verifier_review,
        "invalid_verifier_count": invalid_verifier,
        "accepted_correct": accepted_correct,
        "accepted_wrong": accepted_wrong,
        "rejected_classifier_errors": rejected_classifier_errors,
        "rejected_correct_classifier_proposals": rejected_correct_proposals,
    }


def _condition_summary(condition: str, condition_runs: list[dict[str, Any]]) -> dict[str, Any]:
    flat = [case for run in condition_runs for case in run["cases"]]
    summary: dict[str, Any] = {
        "runs": len(condition_runs),
        "primary": _primary_metrics(flat),
        "real_category_subset": _real_category_subset_metrics(flat),
        "review_subset": _review_subset_metrics(flat),
        "cost_scenarios": {name: _cost(flat, weights) for name, weights in COST_SCENARIOS.items()},
    }
    if condition == "E3":
        summary["e3_gate_analysis"] = _e3_gate_metrics(flat)
    if condition == "E4":
        summary["e3_gate_analysis"] = _e3_gate_metrics(flat)
        summary["e4_veto_analysis"] = _e4_metrics(flat)
    if condition == "E5":
        summary["e5_analysis"] = _e5_metrics(flat)
    return summary


def _stability(condition_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Repeated runs at temperature 0 mostly assess protocol/runtime
    stability, not independent semantic samples -- the unique-file count
    stays the semantic denominator; this only flags files whose *decision*
    ever changed across repetitions."""
    by_file: dict[str, list[dict[str, Any]]] = {}
    for run in condition_runs:
        for case in run["cases"]:
            by_file.setdefault(case["filename"], []).append(case)
    buckets = Counter()
    unstable_files: list[str] = []
    for filename, cases in by_file.items():
        predictions = {c["predicted"] for c in cases}
        if len(predictions) > 1:
            buckets["unstable"] += 1
            unstable_files.append(filename)
        elif cases[0]["is_review"]:
            buckets["consistently_reviewed"] += 1
        elif cases[0]["correct"]:
            buckets["consistently_correct"] += 1
        else:
            buckets["consistently_wrong"] += 1
    return {
        "unique_file_count": len(by_file),
        "counts": dict(buckets),
        "unstable_files": unstable_files,
    }


def aggregate(
    runs: list[dict[str, Any]], expected: dict[str, list[str]], *, review_directory: str
) -> dict[str, Any]:
    scored_runs = []
    for run in runs:
        cases = score_run(run["metrics"], expected, review_directory=review_directory)
        scored_runs.append({**run, "cases": cases})
    summary = {}
    stability = {}
    for condition in CONDITIONS:
        condition_runs = [r for r in scored_runs if r["condition"] == condition]
        if not condition_runs:
            continue
        summary[condition] = _condition_summary(condition, condition_runs)
        stability[condition] = _stability(condition_runs)
    return {"summary": summary, "stability": stability}


# --- CLI / orchestration ----------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("MODEL_ID", DEFAULT_MODEL_ID))
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--think", dest="think", action="store_true")
    thinking.add_argument("--no-think", dest="think", action="store_false")
    parser.set_defaults(think=False)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--fixtures",
        nargs="+",
        choices=list(FIXTURES),
        default=list(FIXTURES),
        help="which Development fixtures to run (both by default, reported "
        "separately and combined)",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "evals/results")
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def _combined(
    per_fixture: dict[str, dict[str, Any]], expected_by_fixture: dict[str, dict[str, list[str]]],
    runs_by_fixture: dict[str, list[dict[str, Any]]], *, review_directory: str,
) -> dict[str, Any]:
    """Combined view across fixtures, reported alongside (never instead of)
    each fixture's own separate aggregate."""
    all_runs = [run for runs in runs_by_fixture.values() for run in runs]
    combined_expected: dict[str, list[str]] = {}
    for fixture_name, expected in expected_by_fixture.items():
        for name, allowed in expected.items():
            combined_expected[f"{fixture_name}::{name}"] = allowed
    namespaced_runs = []
    for fixture_name, runs in runs_by_fixture.items():
        for run in runs:
            namespaced = dict(run)
            namespaced["metrics"] = dict(run["metrics"])
            namespaced["metrics"]["sources"] = [
                f"{fixture_name}::{source}" for source in run["metrics"].get("sources", [])
            ]
            namespaced["metrics"]["final"] = {
                f"{fixture_name}::{source}": category
                for source, category in run["metrics"].get("final", {}).items()
            }
            detail = run["metrics"].get("detail", [])
            namespaced["metrics"]["detail"] = [
                {**item, "filename": f"{fixture_name}::{item['filename']}"} for item in detail
            ]
            namespaced_runs.append(namespaced)
    return aggregate(namespaced_runs, combined_expected, review_directory=review_directory)


def run_experiment(args: argparse.Namespace) -> Path:
    source = _git_source()
    schedule = [
        list(COUNTERBALANCED_SCHEDULE[index % len(COUNTERBALANCED_SCHEDULE)])
        for index in range(args.repetitions)
    ]
    model = _model_manifest(args.model, args.think)
    experiment_id = args.experiment_id or (
        f"post-holdout-development-{datetime.now(timezone.utc).strftime('%Y%m%d')}-"
        f"{source['short_commit']}"
    )
    output = args.output_root / experiment_id
    if output.exists():
        raise FileExistsError(f"experiment output already exists: {output}")

    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "python_environment": {
            "operating_system": platform.platform(),
            "architecture": platform.machine(),
            "processor": platform.processor() or "unknown",
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "dependencies": _installed_versions(),
        },
        "model": model,
        "conditions": CONDITIONS,
        "fixtures_run": args.fixtures,
        "execution": {
            "repetitions": args.repetitions,
            "scheduled_runs_per_fixture": args.repetitions * len(CONDITIONS),
            "condition_order": schedule,
            "model_lifecycle": "warm",
            "warmup": "one discarded model request before all measured runs",
            "standard_timeout_seconds": args.timeout,
            "failed_run_policy": "record failure and stop; never substitute model; no reruns",
        },
    }

    with tempfile.TemporaryDirectory(prefix=f"{experiment_id}-") as temporary_name:
        temporary = Path(temporary_name)
        staging = temporary / "artifacts"
        staging.mkdir()

        warmup = run_in_subprocess(_warmup_worker, (args.model, args.think), timeout=args.timeout)
        manifest["execution"]["warmup_result"] = warmup
        if warmup.get("status") != "ok":
            raise RuntimeError("model warmup failed")

        runs_by_fixture: dict[str, list[dict[str, Any]]] = {}
        expected_by_fixture: dict[str, dict[str, list[str]]] = {}
        aborted = False

        for fixture_name in args.fixtures:
            paths = FIXTURES[fixture_name]
            fixture = paths["fixture"].resolve(strict=True)
            expected_path = paths["expected"].resolve(strict=True)
            fixture_manifest = _verify_dataset_manifest(
                paths["manifest"].resolve(strict=True), fixture, expected_path
            )
            expected_by_fixture[fixture_name] = _load_expected(expected_path)
            staged_fixture = temporary / f"fixture-{fixture_name}"
            _stage_fixture(fixture, fixture_manifest, staged_fixture)

            runs: list[dict[str, Any]] = []
            raw_path = staging / f"raw_runs-{fixture_name}.jsonl"
            for repetition, order in enumerate(schedule, 1):
                for position, condition in enumerate(order, 1):
                    started = time.perf_counter()
                    metrics = run_in_subprocess(
                        _post_holdout_worker,
                        (str(staged_fixture), condition, args.model, args.think),
                        timeout=args.timeout,
                    )
                    status = metrics.get("status", "error")
                    if status == "ok":
                        for field in SECURITY_ZERO_FIELDS:
                            value = int(metrics["telemetry"].get(field, 0) or 0)
                            if value:
                                raise RuntimeError(
                                    f"post-holdout development cycle is metadata-only but "
                                    f"{field}={value}"
                                )
                    run = {
                        "experiment_id": experiment_id,
                        "fixture": fixture_name,
                        "repetition": repetition,
                        "sequence_position": position,
                        "condition": condition,
                        "status": status,
                        "total_run_latency_seconds": time.perf_counter() - started,
                        "metrics": metrics,
                    }
                    runs.append(run)
                    with raw_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(run, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    if status != "ok":
                        aborted = True
                        break
                if aborted:
                    break
            runs_by_fixture[fixture_name] = runs
            if aborted:
                break

        final_model = _model_manifest(args.model, args.think)
        unchanged = final_model.get("digest") == model.get("digest")
        manifest["execution"]["model_identity_check"] = "unchanged" if unchanged else "changed during run"
        if not unchanged:
            aborted = True

        expected_run_count = args.repetitions * len(CONDITIONS)
        complete = not aborted and all(
            len(runs_by_fixture.get(name, [])) == expected_run_count for name in args.fixtures
        )
        result: dict[str, Any] = {"experiment_id": experiment_id, "complete": complete}
        if complete:
            review_directory = load_rules().review_directory
            result["per_fixture"] = {
                name: aggregate(runs_by_fixture[name], expected_by_fixture[name], review_directory=review_directory)
                for name in args.fixtures
            }
            if len(args.fixtures) > 1:
                result["combined"] = _combined(
                    result["per_fixture"], expected_by_fixture, runs_by_fixture,
                    review_directory=review_directory,
                )
        (staging / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, output)
    return output


def main(argv: list[str] | None = None) -> int:
    output = run_experiment(parse_args(argv))
    print(f"Post-holdout development artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
