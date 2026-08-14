"""Frozen, counterbalanced E0/E1/E2/E3 abstention-calibration comparison.

Metadata-only: no ``--read-contents``, no ``--allow-remote-content``, no peek
tool is ever constructed. This intentionally does not touch the consumed
41-file Holdout — only ``evals/calibration`` (development-only, reusable) is
read here.

E0 is the existing production two-pass metadata control, used unchanged via
``StructuredClassifier.classify(metadata_control=True)``. E1/E2/E3 are
evaluation-only candidates from ``evals.calibration_candidates``; see that
module's docstring for the exact per-candidate design and state tables.
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
from tidy.classification import (  # noqa: E402
    CLASSIFICATION_JSON_SCHEMA,
    ClassificationBackend,
    StructuredClassifier,
    build_classification_prompt,
    validate_classification_response,
)
from tidy.rules import classify_directory, load_rules  # noqa: E402
from tidy.tools import metadata_for_names  # noqa: E402

from evals.calibration_candidates import (  # noqa: E402
    EXPLICIT_ABSTENTION_JSON_SCHEMA,
    ValidatedAbstentionClassification,
    build_explicit_abstention_prompt,
    merge_agreement_gate,
    merge_disagreement_abstention,
    resolve_explicit_abstention,
    reverse_pass_order,
    validate_explicit_abstention_response,
)
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
    "E0": "two-pass metadata control (production, unchanged)",
    "E1": "deterministic disagreement abstention (two independent passes)",
    "E2": "explicit structured abstention (decision + nullable category)",
    "E3": "explicit abstention + agreement gate (two independent passes)",
}
COUNTERBALANCED_SCHEDULE = (
    ("E0", "E1", "E2", "E3"),
    ("E1", "E2", "E3", "E0"),
    ("E2", "E3", "E0", "E1"),
    ("E3", "E0", "E1", "E2"),
)
COST_SCENARIOS = {
    "safety_heavy": {"incorrect_automatic": 10, "review": 1, "correct": 0},
    "balanced": {"incorrect_automatic": 5, "review": 1, "correct": 0},
    "coverage_heavy": {"incorrect_automatic": 3, "review": 1, "correct": 0},
}
SECURITY_ZERO_FIELDS = (
    "peek_calls",
    "peek_bytes_read",
    "peek_chars_returned",
    "peek_requests_authorized",
    "content_unavailable",
)


# --- Per-condition classification, callable directly for offline testing ---


def run_e0(
    model: Any, metadata: list[dict[str, Any]], categories_with_review: list[str]
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """Unchanged production path. No case-level pass detail is produced."""
    classifier = StructuredClassifier(model)
    result = classifier.classify(metadata, categories_with_review, metadata_control=True)
    sources = [str(item["name"]) for item in metadata]
    final = {
        source: result.categories.get(source, categories_with_review[-1])
        for source in sources
    }
    detail = [{"filename": source} for source in sources]
    return final, detail, result.telemetry


def run_e1(
    model: Any,
    metadata: list[dict[str, Any]],
    categories_with_review: list[str],
    *,
    review_directory: str,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    sources = [str(item["name"]) for item in metadata]
    backend = ClassificationBackend(model)
    raw1 = backend.request(
        build_classification_prompt(metadata, categories_with_review),
        schema_name="tidy_classification",
        schema=CLASSIFICATION_JSON_SCHEMA,
        phase="final",
    )
    pass1 = validate_classification_response(
        raw1, sources, categories_with_review, backend.telemetry
    )
    metadata2 = reverse_pass_order(metadata)
    raw2 = backend.request(
        build_classification_prompt(metadata2, categories_with_review),
        schema_name="tidy_classification",
        schema=CLASSIFICATION_JSON_SCHEMA,
        phase="final",
    )
    pass2 = validate_classification_response(
        raw2, sources, categories_with_review, backend.telemetry
    )
    merged = merge_disagreement_abstention(
        pass1, pass2, sources, review_directory=review_directory
    )
    final = {source: outcome.final for source, outcome in merged.items()}
    detail = [
        {
            "filename": source,
            "pass1_decision": "classify" if outcome.pass1_valid else None,
            "pass1_category": outcome.pass1_category,
            "pass2_decision": "classify" if outcome.pass2_valid else None,
            "pass2_category": outcome.pass2_category,
            "agreement": outcome.agreement,
        }
        for source, outcome in merged.items()
    ]
    return final, detail, backend.telemetry.snapshot()


def run_e2(
    model: Any,
    metadata: list[dict[str, Any]],
    real_categories: list[str],
    *,
    review_directory: str,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    sources = [str(item["name"]) for item in metadata]
    backend = ClassificationBackend(model)
    raw = backend.request(
        build_explicit_abstention_prompt(metadata, real_categories),
        schema_name="tidy_explicit_abstention",
        schema=EXPLICIT_ABSTENTION_JSON_SCHEMA,
        phase="final",
    )
    result = validate_explicit_abstention_response(
        raw, sources, real_categories, backend.telemetry
    )
    final = resolve_explicit_abstention(result, sources, review_directory=review_directory)
    detail = [
        {
            "filename": source,
            "pass1_decision": _decision_of(result, source),
            "pass1_category": _category_of(result, source),
            "invalid_reason": result.invalid_reasons.get(source),
        }
        for source in sources
    ]
    return final, detail, backend.telemetry.snapshot()


def run_e3(
    model: Any,
    metadata: list[dict[str, Any]],
    real_categories: list[str],
    *,
    review_directory: str,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    sources = [str(item["name"]) for item in metadata]
    backend = ClassificationBackend(model)
    raw1 = backend.request(
        build_explicit_abstention_prompt(metadata, real_categories),
        schema_name="tidy_explicit_abstention",
        schema=EXPLICIT_ABSTENTION_JSON_SCHEMA,
        phase="final",
    )
    pass1 = validate_explicit_abstention_response(
        raw1, sources, real_categories, backend.telemetry
    )
    metadata2 = reverse_pass_order(metadata)
    raw2 = backend.request(
        build_explicit_abstention_prompt(metadata2, real_categories),
        schema_name="tidy_explicit_abstention",
        schema=EXPLICIT_ABSTENTION_JSON_SCHEMA,
        phase="final",
    )
    pass2 = validate_explicit_abstention_response(
        raw2, sources, real_categories, backend.telemetry
    )
    gate = merge_agreement_gate(pass1, pass2, sources, review_directory=review_directory)
    final = {source: outcome.final for source, outcome in gate.items()}
    detail = [
        {
            "filename": source,
            "pass1_decision": outcome.pass1_decision,
            "pass1_category": outcome.pass1_category,
            "pass2_decision": outcome.pass2_decision,
            "pass2_category": outcome.pass2_category,
            "agreement": outcome.agreement,
        }
        for source, outcome in gate.items()
    ]
    return final, detail, backend.telemetry.snapshot()


def _decision_of(result: ValidatedAbstentionClassification, source: str) -> str | None:
    decision = result.decisions.get(source)
    return decision.decision if decision is not None else None


def _category_of(result: ValidatedAbstentionClassification, source: str) -> str | None:
    decision = result.decisions.get(source)
    return decision.category if decision is not None else None


def run_condition(
    condition: str,
    model: Any,
    metadata: list[dict[str, Any]],
    *,
    real_categories: list[str],
    categories_with_review: list[str],
    review_directory: str,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    if condition == "E0":
        return run_e0(model, metadata, categories_with_review)
    if condition == "E1":
        return run_e1(model, metadata, categories_with_review, review_directory=review_directory)
    if condition == "E2":
        return run_e2(model, metadata, real_categories, review_directory=review_directory)
    if condition == "E3":
        return run_e3(model, metadata, real_categories, review_directory=review_directory)
    raise ValueError(f"unknown condition: {condition}")


# --- Subprocess worker (live path only; not exercised in Phase 1) ----------


def _calibration_worker(
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
                "calibration fixture must be fully unresolved by deterministic "
                f"rules; {len(moves)} file(s) resolved unexpectedly"
            )
        metadata = metadata_for_names(fixture, unresolved)
        real_categories = list(rules.categories)
        categories_with_review = [*rules.categories, rules.review_directory]
        model = build_model(model_id, think=think)
        final, detail, telemetry = run_condition(
            condition,
            model,
            metadata,
            real_categories=real_categories,
            categories_with_review=categories_with_review,
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


# --- Provenance -------------------------------------------------------------


def _git_source() -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "live calibration evaluation requires a clean Git worktree; commit "
            "or remove all changes first"
        )
    commit = _git("rev-parse", "HEAD")
    script = Path(__file__).resolve()
    return {
        "commit": commit,
        "short_commit": commit[:12],
        "dirty": False,
        "branch": _git("branch", "--show-current"),
        "evaluation_script": "evals/run_structured_calibration.py",
        "evaluation_script_sha256": _sha256(script),
        "calibration_candidates_sha256": _sha256(
            PROJECT_ROOT / "evals/calibration_candidates.py"
        ),
    }


def _experiment_manifest(
    *, source: dict[str, Any], model: dict[str, Any], fixture_manifest: dict[str, Any],
    repetitions: int, schedule: list[list[str]], timeout: float,
) -> dict[str, Any]:
    rules = load_rules()
    rule_paths = [
        PROJECT_ROOT / "config/rules.yaml",
        PROJECT_ROOT / "src/tidy/config/rules.yaml",
    ]
    return {
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
        "tidy_configuration": {
            "rules": [
                {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha256(path)}
                for path in rule_paths
            ],
            "categories": list(rules.categories),
            "review_directory": rules.review_directory,
            "conditions": CONDITIONS,
            "content_mode": False,
            "remote_content_authorized": False,
        },
        "dataset": {
            "designation": "development-calibration",
            "fixture_manifest": "evals/calibration/fixture_manifest.json",
            "fixture_dataset_sha256": fixture_manifest["dataset_sha256"],
            "fixture_file_count": len(fixture_manifest["files"]),
            "ground_truth_sha256": fixture_manifest["ground_truth"]["sha256"],
            "holdout_referenced": False,
        },
        "execution": {
            "repetitions": repetitions,
            "scheduled_runs": repetitions * len(CONDITIONS),
            "condition_order": schedule,
            "model_lifecycle": "warm",
            "warmup": "one discarded model request before all measured runs",
            "standard_timeout_seconds": timeout,
            "failed_run_policy": "record failure and stop; never substitute model; no reruns",
        },
    }


# --- Aggregation --------------------------------------------------------


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
        "unsafe_classification_of_review_case": (not is_review) and ground_truth_review_only,
    }


def score_run(
    run: dict[str, Any], expected: dict[str, list[str]], *, review_directory: str
) -> list[dict[str, Any]]:
    """Join one run's raw predictions with ground truth. Pure and offline-testable."""
    metrics = run["metrics"]
    detail_by_source = {item["filename"]: item for item in metrics.get("detail", [])}
    cases = []
    for source in metrics.get("sources", []):
        allowed = expected[source]
        predicted = metrics["final"].get(source, review_directory)
        case = _score_case(source, allowed, predicted, review_directory=review_directory)
        case.update(
            {
                key: value
                for key, value in detail_by_source.get(source, {}).items()
                if key != "filename"
            }
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


def _condition_summary(
    condition_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    condition = condition_runs[0]["condition"]
    all_cases = [run["cases"] for run in condition_runs]
    flat = [case for cases in all_cases for case in cases]
    total = len(flat)
    correct = sum(c["correct"] for c in flat)
    incorrect_automatic = sum(c["incorrect_automatic"] for c in flat)
    review_count = sum(c["is_review"] for c in flat)
    decided = total - review_count
    decided_correct = sum(c["correct"] for c in flat if not c["is_review"])
    true_abstentions = sum(c["true_abstention"] for c in flat)
    false_abstentions = sum(c["false_abstention"] for c in flat)
    unsafe_review_cases = sum(c["unsafe_classification_of_review_case"] for c in flat)
    ground_truth_review_total = sum(c["ground_truth_review_only"] for c in flat)
    predicted_review_total = review_count
    category_only = [c for c in flat if not c["ground_truth_review_only"]]

    agreement_cases = [c for c in flat if c.get("agreement") is not None]
    agree = [c for c in agreement_cases if c["agreement"] in ("agree", "agree_classify")]
    disagree = [c for c in agreement_cases if c not in agree]

    # Item 18 scopes voluntary-review analysis to E2/E3: only those candidates
    # carry an explicit ``decision`` field distinguishing "the model chose
    # review" from "Python fell back to review after an invalid/omitted
    # response". E0 has no such field (review can be a genuine category
    # value or a fallback, indistinguishably), and E1 has no decision
    # field at all (only whether pass 1 independently resolved a category).
    explicit_review_applicable = condition in ("E2", "E3")
    if explicit_review_applicable:
        voluntary_review = [c for c in flat if c.get("pass1_decision") == "review"]
        voluntary_review_correct = sum(
            c["ground_truth_review_only"] for c in voluntary_review
        )
        voluntary_misclassification_of_ambiguous = sum(
            1
            for c in flat
            if c["ground_truth_review_only"] and c.get("pass1_decision") == "classify"
        )
    else:
        voluntary_review = []
        voluntary_review_correct = None
        voluntary_misclassification_of_ambiguous = None

    return {
        "runs": len(condition_runs),
        "total_files_scored": total,
        "overall": {
            "strict_category_accuracy": correct / total if total else 0.0,
            "incorrect_automatic_decision_rate": incorrect_automatic / total if total else 0.0,
            "review_rate": review_count / total if total else 0.0,
            "decision_rate": decided / total if total else 0.0,
            "accuracy_on_decided": decided_correct / decided if decided else 0.0,
        },
        "raw_counts": {
            "correct_automatic": decided_correct,
            "incorrect_automatic": incorrect_automatic,
            "review": review_count,
        },
        "unsafe_automation_rate": incorrect_automatic / total if total else 0.0,
        "automation_coverage": decided / total if total else 0.0,
        "abstention": {
            "true_abstentions": true_abstentions,
            "false_abstentions": false_abstentions,
            "unsafe_classifications_of_ground_truth_review": unsafe_review_cases,
            "safe_reviews_of_ground_truth_review": true_abstentions,
            "agreements_between_passes": len(agree),
            "disagreements_between_passes": len(disagree),
            "agreement_accuracy": (
                sum(c["correct"] for c in agree) / len(agree) if agree else None
            ),
            "disagreement_accuracy": (
                sum(c["correct"] for c in disagree) / len(disagree) if disagree else None
            ),
        },
        "review_quality": {
            "review_recall": (
                true_abstentions / ground_truth_review_total
                if ground_truth_review_total
                else None
            ),
            "review_precision": (
                true_abstentions / predicted_review_total if predicted_review_total else None
            ),
        },
        "category_only": {
            "n": len(category_only),
            "category_accuracy": (
                sum(c["correct"] for c in category_only) / len(category_only)
                if category_only
                else None
            ),
            "false_review_rate": (
                sum(c["is_review"] for c in category_only) / len(category_only)
                if category_only
                else None
            ),
            "incorrect_category_rate": (
                sum(c["incorrect_automatic"] for c in category_only) / len(category_only)
                if category_only
                else None
            ),
        },
        "explicit_review": {
            "applicable": explicit_review_applicable,
            "voluntary_review_count": (
                len(voluntary_review) if explicit_review_applicable else None
            ),
            "voluntary_review_matches_ground_truth_ambiguity": voluntary_review_correct,
            "voluntary_misclassification_of_ambiguous_files": (
                voluntary_misclassification_of_ambiguous
            ),
        },
        "cost_scenarios": {
            name: _cost(flat, weights) for name, weights in COST_SCENARIOS.items()
        },
    }


def _stability(condition_runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_file: dict[str, list[dict[str, Any]]] = {}
    for run in condition_runs:
        for case in run["cases"]:
            by_file.setdefault(case["filename"], []).append(case)
    buckets = Counter()
    unstable_files: dict[str, list[str]] = {
        "unstable_category": [],
        "unstable_review_classify_decision": [],
    }
    for filename, cases in by_file.items():
        predictions = [c["predicted"] for c in cases]
        review_flags = {c["is_review"] for c in cases}
        if len(review_flags) > 1:
            buckets["unstable_review_classify_decision"] += 1
            unstable_files["unstable_review_classify_decision"].append(filename)
            continue
        if len(set(predictions)) > 1:
            buckets["unstable_category"] += 1
            unstable_files["unstable_category"].append(filename)
            continue
        if cases[0]["is_review"]:
            buckets["consistently_reviewed"] += 1
        elif cases[0]["correct"]:
            buckets["consistently_correct"] += 1
        else:
            buckets["consistently_wrong"] += 1
    return {"counts": dict(buckets), "unstable_files": unstable_files}


def aggregate(
    runs: list[dict[str, Any]], expected: dict[str, list[str]], *, review_directory: str
) -> dict[str, Any]:
    scored_runs = []
    for run in runs:
        cases = score_run(run, expected, review_directory=review_directory)
        scored_runs.append({**run, "cases": cases})
    summary = {}
    stability = {}
    for condition in CONDITIONS:
        condition_runs = [r for r in scored_runs if r["condition"] == condition]
        if not condition_runs:
            continue
        summary[condition] = _condition_summary(condition_runs)
        stability[condition] = _stability(condition_runs)
    return {"summary": summary, "stability": stability}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("MODEL_ID", DEFAULT_MODEL_ID))
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--think", dest="think", action="store_true")
    thinking.add_argument("--no-think", dest="think", action="store_false")
    parser.set_defaults(think=False)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--fixture", type=Path, default=PROJECT_ROOT / "evals/calibration/fixture"
    )
    parser.add_argument(
        "--expected", type=Path, default=PROJECT_ROOT / "evals/calibration/expected.yaml"
    )
    parser.add_argument(
        "--fixture-manifest",
        type=Path,
        default=PROJECT_ROOT / "evals/calibration/fixture_manifest.json",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "evals/results")
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def run_experiment(args: argparse.Namespace) -> Path:
    source = _git_source()
    fixture = args.fixture.resolve(strict=True)
    expected_path = args.expected.resolve(strict=True)
    fixture_manifest = _verify_dataset_manifest(
        args.fixture_manifest.resolve(strict=True), fixture, expected_path
    )
    expected = _load_expected(expected_path)

    schedule = [
        list(COUNTERBALANCED_SCHEDULE[index % len(COUNTERBALANCED_SCHEDULE)])
        for index in range(args.repetitions)
    ]
    model = _model_manifest(args.model, args.think)
    experiment_id = args.experiment_id or (
        f"structured-calibration-{datetime.now(timezone.utc).strftime('%Y%m%d')}-"
        f"{source['short_commit']}"
    )
    output = args.output_root / experiment_id
    if output.exists():
        raise FileExistsError(f"experiment output already exists: {output}")

    manifest = _experiment_manifest(
        source=source, model=model, fixture_manifest=fixture_manifest,
        repetitions=args.repetitions, schedule=schedule, timeout=args.timeout,
    )
    manifest["experiment_id"] = experiment_id

    with tempfile.TemporaryDirectory(prefix=f"{experiment_id}-") as temporary_name:
        temporary = Path(temporary_name)
        staging = temporary / "artifacts"
        staging.mkdir()
        staged_fixture = temporary / "fixture"
        _stage_fixture(fixture, fixture_manifest, staged_fixture)
        shutil.copyfile(args.fixture_manifest, staging / "fixture_manifest.json")

        warmup = run_in_subprocess(_warmup_worker, (args.model, args.think), timeout=args.timeout)
        manifest["execution"]["warmup_result"] = warmup
        if warmup.get("status") != "ok":
            raise RuntimeError("model warmup failed")

        runs: list[dict[str, Any]] = []
        raw_path = staging / "raw_runs.jsonl"
        aborted = False
        for repetition, order in enumerate(schedule, 1):
            for position, condition in enumerate(order, 1):
                started = time.perf_counter()
                metrics = run_in_subprocess(
                    _calibration_worker,
                    (str(staged_fixture), condition, args.model, args.think),
                    timeout=args.timeout,
                )
                status = metrics.get("status", "error")
                if status == "ok":
                    for field in SECURITY_ZERO_FIELDS:
                        value = int(metrics["telemetry"].get(field, 0) or 0)
                        if value:
                            raise RuntimeError(
                                f"calibration cycle is metadata-only but {field}={value}"
                            )
                run = {
                    "experiment_id": experiment_id,
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

        final_model = _model_manifest(args.model, args.think)
        unchanged = final_model.get("digest") == model.get("digest")
        manifest["execution"]["model_identity_check"] = "unchanged" if unchanged else "changed during run"
        if not unchanged:
            aborted = True

        complete = not aborted and len(runs) == args.repetitions * len(CONDITIONS)
        result = {"experiment_id": experiment_id, "complete": complete, "run_count": len(runs)}
        if complete:
            result.update(aggregate(runs, expected, review_directory=load_rules().review_directory))
        (staging / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report = (
            _render_report(experiment_id=experiment_id, manifest=manifest, result=result)
            if complete
            else f"# {experiment_id}\n\nStopped after {len(runs)} scheduled runs.\n"
        )
        (staging / "report.md").write_text(report, encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, output)
    return output


def _render_report(*, experiment_id: str, manifest: dict[str, Any], result: dict[str, Any]) -> str:
    lines = [
        f"# E0/E1/E2/E3 abstention calibration — {experiment_id}",
        "",
        "> development-calibration fixture. The consumed 41-file Holdout was not used.",
        "",
        f"- Commit: `{manifest['source']['commit']}`",
        f"- Model: `{manifest['model']['identifier']}` digest `{manifest['model'].get('digest')}`",
        f"- Repetitions: {manifest['execution']['repetitions']} per condition",
        "",
        "## Main results",
        "",
        "| Condition | Accuracy | Unsafe automation | Automation coverage | "
        "Review rate | Accuracy decided | Review recall | Review precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, label in CONDITIONS.items():
        if condition not in result["summary"]:
            continue
        s = result["summary"][condition]
        rq = s["review_quality"]
        recall_text = "n/a" if rq["review_recall"] is None else f"{rq['review_recall']:.1%}"
        precision_text = (
            "n/a" if rq["review_precision"] is None else f"{rq['review_precision']:.1%}"
        )
        lines.append(
            f"| {condition} — {label} | {s['overall']['strict_category_accuracy']:.1%} | "
            f"{s['unsafe_automation_rate']:.1%} | {s['automation_coverage']:.1%} | "
            f"{s['overall']['review_rate']:.1%} | {s['overall']['accuracy_on_decided']:.1%} | "
            f"{recall_text} | {precision_text} |"
        )
    lines.extend(["", "## Cost scenarios (lower is better)", "", "| Condition | " + " | ".join(COST_SCENARIOS) + " |", "|---|" + "---:|" * len(COST_SCENARIOS)])
    for condition in CONDITIONS:
        if condition not in result["summary"]:
            continue
        costs = result["summary"][condition]["cost_scenarios"]
        lines.append(f"| {condition} | " + " | ".join(str(costs[name]) for name in COST_SCENARIOS) + " |")
    lines.extend(["", "## Stability across repetitions", ""])
    for condition in CONDITIONS:
        if condition not in result.get("stability", {}):
            continue
        lines.append(f"- {condition}: {result['stability'][condition]['counts']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    output = run_experiment(parse_args(argv))
    print(f"Calibration artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
