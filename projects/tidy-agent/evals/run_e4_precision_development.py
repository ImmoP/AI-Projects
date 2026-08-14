"""Frozen, counterbalanced E3/E4-current/E4-refined precision-refinement
Development comparison.

This is a **new** harness, deliberately not an edit of
``evals/run_post_holdout_development.py`` (the E3/E4/E5 harness that already
produced a committed historical result): section 35's requirement that E5's
"historical reproducibility must remain intact" applies equally to the
harness that produced it, so that file is left byte-for-byte untouched by
this Development cycle. E5 is not included here; see its retirement note in
``evals/post_holdout_candidates.py``.

Metadata-only: no ``--read-contents``, no ``--allow-remote-content``, no
peek tool is ever constructed. This does not touch either consumed Holdout
-- only the four reusable Development fixtures (``evals/calibration``,
``evals/boundary_calibration``, ``evals/veto_precision_calibration``,
``evals/e3_error_calibration``) are read here.

``evals/e3_error_calibration`` was added after the corrected 2026-08-12
E4-precision Development cycle (see
``evals/results/e4-precision-development-20260812-b6095d49ee1d/``), which
observed only 1 unique E3 automatic error across the other three fixtures
-- below the frozen minimum of
``MINIMUM_E3_ERRORS_FOR_ROBUST_VETO_ESTIMATE`` (3). It targets a
different, complementary question from ``evals/veto_precision_calibration``
(which asks "can E4-refined avoid false vetoes on correct E3 decisions?"):
"when metadata is semantically deceptive but still plausible enough for E3
to automate, does E3 make correlated wrong automatic decisions -- and can
the already-frozen veto strategies catch them?" It does not guarantee
observing >=3 unique E3 automatic errors in any future live run; it only
increases the probability. See
``evals/e3_error_calibration/README.md`` and ``build_fixture.py`` for the
fixture's stress-family design. This module extension adds no candidate
change of any kind -- E3, E4-current, and E4-refined remain exactly as
frozen in ``evals/post_holdout_candidates.py``.

Call-accounting design (item 27): E4-current and E4-refined both begin from
the exact frozen E3 result and add no model call of their own -- calling
``run_e4``/``run_e4_refined`` independently per condition would each
re-invoke ``run_e3`` from scratch, redundantly repeating the identical
(deterministic, temperature 0) two-pass call three times per repetition.
This harness instead calls ``run_e3`` **once per repetition** and derives
all three conditions' results from that single call via the pure functions
``apply_conflict_veto`` (E4-current) and ``apply_refined_veto``
(E4-refined) -- a 3x reduction in live model calls relative to calling each
candidate's ``run_e4``-style entry point independently, with no change to
E3's own behaviour and no change to how any condition is scored: each
condition still receives its own correctly-scoped final-categories mapping
and per-file detail, exactly as if it had been computed independently
(which, at temperature 0, it deterministically would be -- confirmed
empirically in the prior E3/E4/E5 live run, where every condition/fixture
combination had zero unstable files across five repetitions).

Unique-file vs. repeated-observation aggregation (bugfix, post-dating the
2026-08-12 completed Development run recorded under
``evals/results/e4-precision-development-20260812-b6095d49ee1d/``, which
exposed this defect and is preserved there unmodified): every fixture is
run for multiple repetitions to measure deterministic stability and
protocol reliability, **not** to multiply the semantic sample size (item
14 of that run's brief). Repeating the same (fixture, filename) decision
five times must count as one semantic file, not five. Earlier, this
harness's own summary (and, transitively, ``evidence_strength_note`` /
``evaluate_success_criteria``) computed primary metrics -- including the
combined unique E3 automatic-error count that gates the frozen
evidence-strength rule -- straight from the flattened, un-deduplicated,
repetition-multiplied case list. On that completed run this silently
inflated 1 true unique E3 automatic error into a reported count of 5,
flipping the evidence-strength verdict from the correct ``underpowered``
to an incorrect ``not underpowered``. That misreading was caught and
documented after the fact in that run's ``report.md`` /
``summary_unique_files.json`` without altering the original artifacts;
this module now fixes the aggregation itself so the defect cannot recur in
any future run.

The fix: every case produced by ``score_run`` is tagged with an explicit
``(fixture, filename)`` identity (``case_identity``) rather than relying on
implicit string-namespacing conventions, and ``aggregate_unique_files``
deduplicates repeated per-repetition observations down to one semantic
record per identity (``aggregate_repeated_observations`` keeps the
un-deduplicated view available separately, for stability/reproducibility
diagnostics only). ``_condition_summary``/``aggregate`` now report both
views side by side under clearly named ``unique_file_metrics`` (primary;
use this for every semantic conclusion and candidate-decision input) and
``repeated_observation_metrics`` (diagnostic only; never feed this into
``evidence_strength_note``, ``compute_evidence_strength``, or
``evaluate_success_criteria``). ``evaluate_success_criteria`` was renamed
to take ``unique_e3_automatic_errors`` (not the old, ambiguous
``combined_e3_errors``) precisely so a repetition-summed count cannot be
passed to it without the mismatch being visible at the call site.
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

from evals.post_holdout_candidates import (  # noqa: E402
    apply_conflict_veto,
    apply_refined_veto,
    run_e3,
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
    "E3": "production comparator: explicit abstention + two-pass agreement gate (unchanged)",
    "E4-current": "frozen E4 from the prior Development cycle (comparator; unmodified)",
    "E4-refined": "precision-oriented deterministic veto (hard vs. soft conflicts; this cycle)",
}
# Nominal presentation order only (item 26/8): E4-current and E4-refined add
# no independent model call (see module docstring), so there is no real
# execution-order effect between conditions within one repetition left to
# counterbalance. The rotation is retained for continuity with the prior
# harness's reporting shape and so per-repetition output ordering is itself
# frozen and reproducible, not because it changes what gets measured.
COUNTERBALANCED_SCHEDULE = (
    ("E3", "E4-current", "E4-refined"),
    ("E4-current", "E4-refined", "E3"),
    ("E4-refined", "E3", "E4-current"),
)
COST_SCENARIOS = {
    "safety_heavy": {"incorrect_automatic": 10, "review": 1, "correct": 0},
    "balanced": {"incorrect_automatic": 5, "review": 1, "correct": 0},
    "coverage_heavy": {"incorrect_automatic": 3, "review": 1, "correct": 0},
}
SECURITY_ZERO_FIELDS = ("peek_requests_authorized", "content_unavailable")

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
    "veto_precision_calibration": {
        "fixture": PROJECT_ROOT / "evals/veto_precision_calibration/fixture",
        "expected": PROJECT_ROOT / "evals/veto_precision_calibration/expected.yaml",
        "manifest": PROJECT_ROOT / "evals/veto_precision_calibration/fixture_manifest.json",
    },
    "e3_error_calibration": {
        "fixture": PROJECT_ROOT / "evals/e3_error_calibration/fixture",
        "expected": PROJECT_ROOT / "evals/e3_error_calibration/expected.yaml",
        "manifest": PROJECT_ROOT / "evals/e3_error_calibration/fixture_manifest.json",
    },
}

# Item 31, frozen before any live inference: below this many unique E3
# automatic errors observed across the combined Development fixtures,
# veto precision/recall ratios must be reported as underpowered, not
# overinterpreted as a stable estimate.
MINIMUM_E3_ERRORS_FOR_ROBUST_VETO_ESTIMATE = 3

# Item 32, frozen before any live inference. Every criterion must hold for
# E4-refined to be considered promising; do not change after seeing results.
SUCCESS_CRITERIA_DESCRIPTIONS = (
    "1. unsafe automation no worse than E4-current on the combined Development set",
    "2. accuracy on decided no worse than E4-current",
    "3. review recall no worse than E4-current by more than one unique case",
    "4. automation coverage not more than 3 percentage points below E4-current combined",
    "5. veto precision materially higher than E4-current (underpowered, not passed, if fewer "
    f"than {MINIMUM_E3_ERRORS_FOR_ROBUST_VETO_ESTIMATE} unique E3 automatic errors)",
    "6. improvement not confined to only one Development fixture",
    "7. no fixture shows an obvious catastrophic false-veto pattern",
)


# --- Per-repetition worker: one shared E3 call, three derived conditions ---


def _e4_precision_worker(
    result_queue: Any,
    fixture: str,
    model_id: str | None,
    think: bool | None,
) -> None:
    """Compute E3, E4-current, and E4-refined from a single shared E3 call."""
    started = time.perf_counter()
    try:
        rules = load_rules()
        moves, unresolved = classify_directory(Path(fixture), rules)
        if moves:
            raise RuntimeError(
                "precision-development fixtures must be fully unresolved by "
                f"deterministic rules; {len(moves)} file(s) resolved unexpectedly"
            )
        metadata = metadata_for_names(fixture, unresolved)
        sources = [str(item["name"]) for item in metadata]
        real_categories = list(rules.categories)
        review_directory = rules.review_directory
        model = build_model(model_id, think=think)

        e3_final, e3_detail, telemetry = run_e3(
            model, metadata, real_categories, review_directory=review_directory
        )
        e3_detail_by_source = {item["filename"]: item for item in e3_detail}

        current_veto = apply_conflict_veto(e3_final, sources, review_directory=review_directory)
        refined_veto = apply_refined_veto(e3_final, sources, review_directory=review_directory)

        conditions: dict[str, dict[str, Any]] = {
            "E3": {
                "final": dict(e3_final),
                "detail": list(e3_detail),
            },
            "E4-current": {
                "final": {s: current_veto[s].final for s in sources},
                "detail": [
                    {
                        "filename": s,
                        "e3_category": e3_final.get(s, review_directory),
                        "veto_applicable": current_veto[s].applicable,
                        "matched_category_cue_families": current_veto[s].matched_category_cue_families,
                        "conflict_detected": current_veto[s].conflict_detected,
                        "veto_reason_code": current_veto[s].veto_reason_code,
                        "final": current_veto[s].final,
                    }
                    for s in sources
                ],
            },
            "E4-refined": {
                "final": {s: refined_veto[s].final for s in sources},
                "detail": [
                    {
                        "filename": s,
                        "e3_category": e3_final.get(s, review_directory),
                        "veto_applicable": refined_veto[s].applicable,
                        "category_support": refined_veto[s].category_support,
                        "high_specificity_support": refined_veto[s].high_specificity_support,
                        "conflict_tier": refined_veto[s].conflict_tier,
                        "veto_reason_code": refined_veto[s].veto_reason_code,
                        "final": refined_veto[s].final,
                    }
                    for s in sources
                ],
            },
        }
        result_queue.put(
            {
                "status": "ok",
                "sources": sources,
                "conditions": conditions,
                "telemetry": telemetry,
                "shared_e3_call": True,
                "model_calls_this_repetition": telemetry.get("classification_requests"),
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
            "live E4-precision development evaluation requires a clean Git "
            "worktree; commit or remove all changes first"
        )
    commit = _git("rev-parse", "HEAD")
    script = Path(__file__).resolve()
    return {
        "commit": commit,
        "short_commit": commit[:12],
        "dirty": False,
        "branch": _git("branch", "--show-current"),
        "evaluation_script": "evals/run_e4_precision_development.py",
        "evaluation_script_sha256": _sha256(script),
        "post_holdout_candidates_sha256": _sha256(
            PROJECT_ROOT / "evals/post_holdout_candidates.py"
        ),
    }


# --- Scoring (mirrors evals/run_post_holdout_development.py's shape) -------


def _score_case(
    filename: str, allowed: list[str], predicted: str, *, review_directory: str
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
    return {
        "n": n,
        "correct_automatic": correct_automatic,
        "wrong_automatic": wrong_automatic,
        "false_review": false_review,
        "correct_automation_rate": correct_automatic / n if n else None,
        "wrong_category_rate": wrong_automatic / n if n else None,
        "false_review_rate": false_review / n if n else None,
    }


def _review_subset_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    subset = [c for c in cases if c["ground_truth_review_only"]]
    n = len(subset)
    correctly_reviewed = sum(c["is_review"] for c in subset)
    incorrectly_automated = sum(1 for c in subset if not c["is_review"])
    return {
        "n": n,
        "correctly_reviewed": correctly_reviewed,
        "incorrectly_automated": incorrectly_automated,
        "review_recall": correctly_reviewed / n if n else None,
    }


def _e3_gate_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
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


def _veto_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Shared E4-current/E4-refined veto-quality metrics (item 29). Both
    candidates' detail records use the identical field names
    (``veto_applicable``, ``e3_category``), by design, so one function
    scores either."""
    presented = [c for c in cases if c.get("veto_applicable") is True]
    accepted = [c for c in presented if not c["is_review"]]
    vetoed = [c for c in presented if c["is_review"]]
    true_positive_vetoes = sum(1 for c in vetoed if c["e3_category"] not in c["allowed"])
    false_positive_vetoes = sum(1 for c in vetoed if c["e3_category"] in c["allowed"])
    e3_errors_presented = sum(1 for c in presented if c["e3_category"] not in c["allowed"])
    unsafe_errors_surviving = sum(1 for c in accepted if c["e3_category"] not in c["allowed"])
    reason_codes = Counter(c.get("veto_reason_code") for c in presented)
    return {
        "e3_automatic_candidates": len(presented),
        "accepted": len(accepted),
        "vetoed": len(vetoed),
        "true_positive_vetoes": true_positive_vetoes,
        "false_positive_vetoes": false_positive_vetoes,
        "e3_automatic_errors_surviving": unsafe_errors_surviving,
        "veto_precision": true_positive_vetoes / len(vetoed) if vetoed else None,
        "veto_recall_for_e3_automatic_errors": (
            true_positive_vetoes / e3_errors_presented if e3_errors_presented else None
        ),
        "coverage_loss_relative_to_e3": len(vetoed),
        "reason_code_distribution": dict(reason_codes),
    }


# --- Unique-file identity and deduplication (bugfix) -----------------------
#
# A repetition exists to measure deterministic stability/protocol
# reliability, never to multiply the semantic sample size. Every function
# below this point that feeds a candidate-decision or evidence-strength
# calculation MUST consume the output of `aggregate_unique_files`, never
# `aggregate_repeated_observations` directly.


def case_identity(case: dict[str, Any]) -> tuple[str, str]:
    """Stable unique-file identity: ``(fixture, filename)``.

    Never deduplicate by filename alone: the same basename in two
    different Development fixtures (or, hypothetically, a future
    non-namespaced caller) must remain two distinct semantic cases. Every
    case dict must have ``fixture`` populated (``aggregate`` does this
    immediately after ``score_run``) before this is called.
    """
    return (case.get("fixture"), case["filename"])


def aggregate_repeated_observations(condition_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every per-repetition, per-file scored case for one condition.

    Valid for stability/reproducibility/protocol diagnostics only. Its raw
    counts are inflated by the repetition count and must never be read as
    semantic sample size or fed into candidate-decision logic -- doing so
    was the root cause of the historical unique-file counting bug (see the
    module docstring).
    """
    return [case for run in condition_runs for case in run["cases"]]


def deduplicate_to_unique_files(
    observations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collapse repeated per-repetition observations into one semantic
    record per ``case_identity`` (fixture, filename).

    Fully deterministic repetitions of the same file collapse trivially
    (every repetition agrees, so any one of them is representative). If a
    file's ``predicted`` value differs across repetitions (instability),
    the record from the LOWEST repetition number is used as the
    documented, fixed representative for primary unique-file semantic
    metrics -- this is a disclosed policy, not a silent majority vote, and
    the harness never infers a "winning" prediction from vote counts. Every
    such file is additionally returned in ``unstable_filenames`` so no
    report can present unique-file metrics for it without that caveat
    being visible; callers surface this as ``unstable_unique_files``
    alongside the per-repetition ``stability`` block, which retains the
    full observation history.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in observations:
        groups.setdefault(case_identity(case), []).append(case)

    unique_cases: list[dict[str, Any]] = []
    unstable_filenames: list[str] = []
    for identity, group in groups.items():
        group_sorted = sorted(group, key=lambda c: c.get("repetition") or 0)
        predictions = {c["predicted"] for c in group_sorted}
        representative = dict(group_sorted[0])
        representative["observed_repetitions"] = len(group_sorted)
        representative["unstable"] = len(predictions) > 1
        if representative["unstable"]:
            unstable_filenames.append(identity[1])
        unique_cases.append(representative)
    return unique_cases, unstable_filenames


def aggregate_unique_files(
    condition_runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """The primary semantic dataset for one condition: one record per
    unique ``(fixture, filename)`` identity, regardless of repetition
    count. Every candidate-decision/evidence-strength input must come from
    here."""
    return deduplicate_to_unique_files(aggregate_repeated_observations(condition_runs))


def _metrics_block(condition: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Shared metrics computation, used identically for the unique-file
    view and the repeated-observation view -- the only difference between
    the two is which case list (deduplicated or not) is passed in."""
    block: dict[str, Any] = {
        "primary": _primary_metrics(cases),
        "real_category_subset": _real_category_subset_metrics(cases),
        "review_subset": _review_subset_metrics(cases),
        "cost_scenarios": {name: _cost(cases, weights) for name, weights in COST_SCENARIOS.items()},
    }
    if condition == "E3":
        block["e3_gate_analysis"] = _e3_gate_metrics(cases)
    if condition in ("E4-current", "E4-refined"):
        block["veto_analysis"] = _veto_metrics(cases)
    return block


def _condition_summary(condition: str, condition_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute BOTH the unique-file (primary, semantic) metrics and the
    repeated-observation (diagnostic-only) metrics for one condition.

    Primary candidate-decision logic must always read ``unique_file_metrics``:
    each unique (fixture, filename) identity contributes exactly one
    semantic record regardless of repetition count.
    ``repeated_observation_metrics`` retains the un-deduplicated,
    repetition-multiplied view for stability/reproducibility diagnostics
    only and must never be used for candidate selection or
    evidence-strength thresholds -- conflating the two was the root cause
    of the historical counting bug.
    """
    repeated_cases = aggregate_repeated_observations(condition_runs)
    unique_cases, unstable_unique_files = deduplicate_to_unique_files(repeated_cases)
    return {
        "runs": len(condition_runs),
        "unique_file_metrics": _metrics_block(condition, unique_cases),
        "repeated_observation_metrics": _metrics_block(condition, repeated_cases),
        "unstable_unique_files": sorted(set(unstable_unique_files)),
    }


def _stability(condition_runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_file: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in condition_runs:
        for case in run["cases"]:
            by_file.setdefault(case_identity(case), []).append(case)
    buckets = Counter()
    unstable_files: list[str] = []
    for identity, cases in by_file.items():
        predictions = {c["predicted"] for c in cases}
        if len(predictions) > 1:
            buckets["unstable"] += 1
            unstable_files.append(identity[1])
        elif cases[0]["is_review"]:
            buckets["consistently_reviewed"] += 1
        elif cases[0]["correct"]:
            buckets["consistently_correct"] += 1
        else:
            buckets["consistently_wrong"] += 1
    return {"unique_file_count": len(by_file), "counts": dict(buckets), "unstable_files": unstable_files}


def aggregate(
    runs: list[dict[str, Any]], expected: dict[str, list[str]], *, review_directory: str
) -> dict[str, Any]:
    """Returns ``{"unique_file_metrics": {...}, "repeated_observation_metrics": {...},
    "stability": {...}}`` per condition. See the module docstring and
    ``_condition_summary`` for why these two metrics views are kept
    explicitly separate: `unique_file_metrics` is the primary semantic
    result, `repeated_observation_metrics` is diagnostic-only.
    """
    scored_runs = []
    for run in runs:
        cases = score_run(run["metrics"], expected, review_directory=review_directory)
        for case in cases:
            case["fixture"] = run.get("fixture")
            case["repetition"] = run.get("repetition")
        scored_runs.append({**run, "cases": cases})
    unique_file_metrics: dict[str, Any] = {}
    repeated_observation_metrics: dict[str, Any] = {}
    stability: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_runs = [r for r in scored_runs if r["condition"] == condition]
        if not condition_runs:
            continue
        condition_summary = _condition_summary(condition, condition_runs)
        unique_file_metrics[condition] = {
            "runs": condition_summary["runs"],
            **condition_summary["unique_file_metrics"],
        }
        repeated_observation_metrics[condition] = {
            "runs": condition_summary["runs"],
            **condition_summary["repeated_observation_metrics"],
        }
        stability[condition] = _stability(condition_runs)
        stability[condition]["unstable_unique_files"] = condition_summary["unstable_unique_files"]
    return {
        "unique_file_metrics": unique_file_metrics,
        "repeated_observation_metrics": repeated_observation_metrics,
        "stability": stability,
    }


# --- e3_error_calibration-specific diagnostics (items 35/38/41/42/43) ------
#
# These are new-fixture-only diagnostics that need per-file stress-family
# metadata sourced from evals/e3_error_calibration/build_fixture.py's own
# CASES data -- never sent to a model, never entered in expected.yaml, and
# meaningless for the other three Development fixtures (which carry no
# family tags). Every function here consumes already-scored, already
# unique-file-deduplicated case lists (the same shape `aggregate` produces
# internally) -- none of them run inference or fabricate a prediction; no
# real E3/E4 numbers exist until a future live run actually produces them.


def e3_error_calibration_family_lookup() -> dict[str, str]:
    """filename -> primary stress family, sourced only from
    ``evals/e3_error_calibration/build_fixture.py``'s ``CASES`` tuple.
    Local/lazy import so fixtures that don't need this module never pay
    its import cost, and so this module is never imported at harness
    import time (item 44: family tags must never reach a model prompt --
    they are Python-side evaluator metadata only, exactly like
    ``CASES``'s rationale strings)."""
    from evals.e3_error_calibration.build_fixture import CASES

    return {name: family for name, _category, _rationale, family, _tags in CASES}


def stress_family_metrics(
    e3_unique_cases: list[dict[str, Any]], family_by_file: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Item 35: per-stress-family breakdown of E3's unique-file automatic
    decisions -- files, automatic decisions, reviews, correct/incorrect
    automatic, automatic error rate, accuracy on decided. Shows which
    semantic structures actually defeat E3. ``e3_unique_cases`` must
    already be unique-file-deduplicated E3 cases (item 12)."""
    by_family: dict[str, list[dict[str, Any]]] = {}
    for case in e3_unique_cases:
        family = family_by_file.get(case["filename"])
        if family is None:
            continue
        by_family.setdefault(family, []).append(case)

    result: dict[str, dict[str, Any]] = {}
    for family, cases in by_family.items():
        automatic = [c for c in cases if not c["is_review"]]
        errors = [c for c in automatic if not c["correct"]]
        automatic_n = len(automatic)
        result[family] = {
            "files": len(cases),
            "automatic_decisions": automatic_n,
            "reviews": len(cases) - automatic_n,
            "correct_automatic": automatic_n - len(errors),
            "incorrect_automatic": len(errors),
            "automatic_error_rate": (len(errors) / automatic_n) if automatic_n else None,
            "accuracy_on_decided": ((automatic_n - len(errors)) / automatic_n) if automatic_n else None,
        }
    return result


def e3_error_density(e3_primary: dict[str, Any]) -> float | None:
    """Item 41: unique E3 automatic errors / unique E3 automatic
    decisions -- distinct from ``unsafe_automation_rate``, which divides
    by ALL unique files (including reviewed ones), not just automated
    ones. ``e3_primary`` must be a ``_primary_metrics``-shaped unique-file
    block (e.g. ``unique_file_metrics["E3"]["primary"]``)."""
    raw = e3_primary["raw_counts"]
    automatic = raw["correct_automatic"] + raw["incorrect_automatic"]
    return (raw["incorrect_automatic"] / automatic) if automatic else None


def error_family_capture_matrix(
    e3_unique_cases: list[dict[str, Any]],
    e4_current_unique_cases: list[dict[str, Any]],
    e4_refined_unique_cases: list[dict[str, Any]],
    family_by_file: dict[str, str],
) -> dict[str, dict[str, int]]:
    """Item 38: per-stress-family E3-automatic-error capture/false-positive
    matrix -- for each family, how many unique E3 automatic errors
    occurred, how many each E4 variant caught (routed back to review), and
    how many additional unique false-positive vetoes each variant added on
    files E3 got right. All three case lists must already be unique-file
    cases for the same fixture (item 18). Do not infer safety from
    aggregate precision alone (item 38) -- this table is the intended
    complement."""
    current_by_file = {c["filename"]: c for c in e4_current_unique_cases}
    refined_by_file = {c["filename"]: c for c in e4_refined_unique_cases}

    matrix: dict[str, dict[str, int]] = {
        family: {
            "e3_automatic_errors": 0,
            "e4_current_caught": 0,
            "e4_refined_caught": 0,
            "e4_current_false_positive": 0,
            "e4_refined_false_positive": 0,
        }
        for family in sorted(set(family_by_file.values()))
    }
    for e3_case in e3_unique_cases:
        family = family_by_file.get(e3_case["filename"])
        if family is None or e3_case["is_review"]:
            continue
        row = matrix[family]
        current_case = current_by_file.get(e3_case["filename"])
        refined_case = refined_by_file.get(e3_case["filename"])
        if not e3_case["correct"]:
            row["e3_automatic_errors"] += 1
            if current_case is not None and current_case["is_review"]:
                row["e4_current_caught"] += 1
            if refined_case is not None and refined_case["is_review"]:
                row["e4_refined_caught"] += 1
        else:
            if current_case is not None and current_case["is_review"]:
                row["e4_current_false_positive"] += 1
            if refined_case is not None and refined_case["is_review"]:
                row["e4_refined_false_positive"] += 1
    return matrix


def review_escape_rescue(
    e3_unique_cases: list[dict[str, Any]], e4_unique_cases: list[dict[str, Any]]
) -> dict[str, int]:
    """Item 42: for ground-truth-``_ToReview`` files, how many did E3
    correctly review vs incorrectly automate (an ambiguous case escaping
    E3's own abstention mechanism), and of those E3 incorrectly automated,
    how many did this E4 condition rescue back to review via its veto.
    Both case lists must already be unique-file cases for the same
    fixture."""
    e4_by_file = {c["filename"]: c for c in e4_unique_cases}
    subset = [c for c in e3_unique_cases if c["ground_truth_review_only"]]
    incorrectly_automated = [c for c in subset if not c["is_review"]]
    rescued = sum(
        1
        for c in incorrectly_automated
        if (e4_by_file.get(c["filename"]) or {}).get("is_review")
    )
    return {
        "n": len(subset),
        "e3_correctly_reviewed": sum(1 for c in subset if c["is_review"]),
        "e3_incorrectly_automated": len(incorrectly_automated),
        "rescued": rescued,
    }


def _scored_unique_cases_by_condition(
    runs: list[dict[str, Any]], expected: dict[str, list[str]], *, review_directory: str
) -> dict[str, list[dict[str, Any]]]:
    """Shared helper: unique-file cases per condition for one fixture's raw
    runs, via the identical scoring/dedup path ``aggregate`` uses
    internally (``score_run`` then ``aggregate_unique_files``). Kept
    separate from ``aggregate`` itself so the new-fixture-only diagnostics
    below cannot accidentally regress ``aggregate``'s already-tested
    contract (item 37)."""
    scored_runs = []
    for run in runs:
        cases = score_run(run["metrics"], expected, review_directory=review_directory)
        for case in cases:
            case["fixture"] = run.get("fixture")
            case["repetition"] = run.get("repetition")
        scored_runs.append({**run, "cases": cases})
    result: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        condition_runs = [r for r in scored_runs if r["condition"] == condition]
        if not condition_runs:
            continue
        unique_cases, _unstable = aggregate_unique_files(condition_runs)
        result[condition] = unique_cases
    return result


def compute_e3_error_calibration_diagnostics(
    runs: list[dict[str, Any]], expected: dict[str, list[str]], *, review_directory: str
) -> dict[str, Any]:
    """Items 35/38/41/42/43 combined: stress-family metrics, E3 error
    density, the real-category wrong-vs-review breakdown (already produced
    generically by ``_real_category_subset_metrics``), the error-family
    capture matrix, and review escape/rescue -- all for
    ``e3_error_calibration`` specifically. No real predictions are
    computed by calling this function in isolation; it only aggregates
    whatever ``runs`` a caller supplies (from a real live run, or from a
    FakeModel-driven offline test)."""
    by_condition = _scored_unique_cases_by_condition(runs, expected, review_directory=review_directory)
    family_by_file = e3_error_calibration_family_lookup()
    e3_cases = by_condition.get("E3", [])

    diagnostics: dict[str, Any] = {
        "stress_family_metrics": stress_family_metrics(e3_cases, family_by_file),
        "e3_error_density": e3_error_density(_primary_metrics(e3_cases)),
        "real_category_subset": _real_category_subset_metrics(e3_cases),
    }
    if "E4-current" in by_condition and "E4-refined" in by_condition:
        diagnostics["error_family_capture_matrix"] = error_family_capture_matrix(
            e3_cases, by_condition["E4-current"], by_condition["E4-refined"], family_by_file,
        )
        diagnostics["review_escape"] = {
            "E4-current": review_escape_rescue(e3_cases, by_condition["E4-current"]),
            "E4-refined": review_escape_rescue(e3_cases, by_condition["E4-refined"]),
        }
    return diagnostics


# --- Predefined evidence-strength and success-criteria rules (items 31-32) -


def evidence_strength_note(unique_combined_e3_summary: dict[str, Any]) -> str | None:
    """Item 31, frozen before live inference. Uses unique E3 automatic
    errors (``raw_counts.incorrect_automatic``) on the combined
    Development-fixtures E3 result.

    ``unique_combined_e3_summary`` MUST be a ``unique_file_metrics["E3"]``
    condition block (i.e. already deduplicated by ``aggregate``/
    ``_condition_summary``) -- never a ``repeated_observation_metrics``
    block, whose raw counts are inflated by the repetition count. Passing
    the repeated-observation block here was the exact defect that made the
    2026-08-12 Development run report 5 "unique" E3 errors when only 1
    unique file was actually wrong (see the module docstring); the
    parameter name is deliberately explicit about which block is required
    so that mistake is visible at the call site.
    """
    unique_errors = unique_combined_e3_summary["primary"]["raw_counts"]["incorrect_automatic"]
    if unique_errors < MINIMUM_E3_ERRORS_FOR_ROBUST_VETO_ESTIMATE:
        return (
            f"insufficient correlated-error events to estimate veto precision/recall "
            f"robustly ({unique_errors} unique E3 automatic error(s) observed combined, "
            f"below the frozen minimum of {MINIMUM_E3_ERRORS_FOR_ROBUST_VETO_ESTIMATE})"
        )
    return None


def compute_evidence_strength(unique_combined_e3_summary: dict[str, Any]) -> dict[str, Any]:
    """Structured, machine-readable companion to ``evidence_strength_note``:
    the exact unique-file E3 error count, the frozen threshold, and the
    resulting ``underpowered`` flag, all in one place so downstream code
    never has to re-derive or misinterpret them. Same input contract as
    ``evidence_strength_note`` -- must be a ``unique_file_metrics["E3"]``
    block."""
    unique_errors = unique_combined_e3_summary["primary"]["raw_counts"]["incorrect_automatic"]
    return {
        "unique_e3_automatic_errors": unique_errors,
        "minimum_required": MINIMUM_E3_ERRORS_FOR_ROBUST_VETO_ESTIMATE,
        "underpowered": unique_errors < MINIMUM_E3_ERRORS_FOR_ROBUST_VETO_ESTIMATE,
        "note": evidence_strength_note(unique_combined_e3_summary),
    }


def evaluate_success_criteria(
    *,
    unique_e3_automatic_errors: int,
    e4_current_combined: dict[str, Any],
    e4_refined_combined: dict[str, Any],
    per_fixture_current: dict[str, dict[str, Any]],
    per_fixture_refined: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Item 32, frozen before live inference. All seven criteria must hold
    for E4-refined to be considered promising; criterion 5 is explicitly
    ``"UNDERPOWERED"`` (not ``"PASS"``) below the frozen minimum unique-file
    error count.

    ``unique_e3_automatic_errors`` MUST be the deduplicated unique-file
    count (e.g. from ``compute_evidence_strength``), never a
    repetition-summed raw count -- passing the wrong one was the historical
    bug this harness now guards against structurally via this parameter's
    explicit name (replacing the old, ambiguous ``combined_e3_errors``).
    ``e4_current_combined``/``e4_refined_combined``/``per_fixture_current``/
    ``per_fixture_refined`` must likewise be ``unique_file_metrics``
    condition blocks, never ``repeated_observation_metrics``.
    """
    cur_primary = e4_current_combined["primary"]
    ref_primary = e4_refined_combined["primary"]
    underpowered = unique_e3_automatic_errors < MINIMUM_E3_ERRORS_FOR_ROBUST_VETO_ESTIMATE

    def _delta(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else a - b

    review_recall_delta_unique_cases = None
    cur_recall = e4_current_combined["review_subset"]["review_recall"]
    ref_recall = e4_refined_combined["review_subset"]["review_recall"]
    review_n = e4_current_combined["review_subset"]["n"]
    if cur_recall is not None and ref_recall is not None and review_n:
        review_recall_delta_unique_cases = round((cur_recall - ref_recall) * review_n)

    if underpowered:
        criterion_5: Any = "UNDERPOWERED"
    elif (
        e4_refined_combined["veto_analysis"]["veto_precision"] is not None
        and e4_current_combined["veto_analysis"]["veto_precision"] is not None
        and e4_refined_combined["veto_analysis"]["veto_precision"]
        > e4_current_combined["veto_analysis"]["veto_precision"]
    ):
        criterion_5 = "PASS"
    else:
        criterion_5 = "FAIL"

    criteria = {
        "1_unsafe_automation_no_worse": (
            ref_primary["unsafe_automation_rate"] is not None
            and cur_primary["unsafe_automation_rate"] is not None
            and ref_primary["unsafe_automation_rate"] <= cur_primary["unsafe_automation_rate"]
        ),
        "2_accuracy_on_decided_no_worse": (
            ref_primary["accuracy_on_decided"] is not None
            and cur_primary["accuracy_on_decided"] is not None
            and ref_primary["accuracy_on_decided"] >= cur_primary["accuracy_on_decided"]
        ),
        "3_review_recall_not_worse_by_more_than_one_case": (
            review_recall_delta_unique_cases is not None and review_recall_delta_unique_cases <= 1
        ),
        "4_coverage_not_more_than_3pp_below": (
            ref_primary["automation_coverage"] is not None
            and cur_primary["automation_coverage"] is not None
            and (cur_primary["automation_coverage"] - ref_primary["automation_coverage"]) <= 0.03
        ),
        "5_veto_precision_materially_higher": criterion_5,
        "6_improvement_not_confined_to_one_fixture": {
            fixture: _delta(
                per_fixture_current[fixture]["veto_analysis"]["veto_precision"],
                per_fixture_refined[fixture]["veto_analysis"]["veto_precision"],
            )
            for fixture in per_fixture_current
        },
        "7_no_catastrophic_false_veto_pattern": {
            fixture: per_fixture_refined[fixture]["veto_analysis"]["false_positive_vetoes"]
            for fixture in per_fixture_refined
        },
    }
    return {
        "underpowered": underpowered,
        "unique_e3_automatic_errors": unique_e3_automatic_errors,
        "review_recall_delta_unique_cases": review_recall_delta_unique_cases,
        "criteria": criteria,
        "descriptions": SUCCESS_CRITERIA_DESCRIPTIONS,
    }


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
        "--fixtures", nargs="+", choices=list(FIXTURES), default=list(FIXTURES),
        help="which Development fixtures to run (all four by default, reported "
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
    per_fixture: dict[str, dict[str, Any]],
    expected_by_fixture: dict[str, dict[str, list[str]]],
    runs_by_fixture: dict[str, list[dict[str, Any]]],
    *,
    review_directory: str,
) -> dict[str, Any]:
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
                f"{fixture_name}::{s}" for s in run["metrics"].get("sources", [])
            ]
            namespaced["metrics"]["final"] = {
                f"{fixture_name}::{s}": category for s, category in run["metrics"].get("final", {}).items()
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
        f"e4-precision-development-{datetime.now(timezone.utc).strftime('%Y%m%d')}-"
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
            "scheduled_positions_per_fixture": args.repetitions * len(CONDITIONS),
            "condition_presentation_order": schedule,
            "call_accounting": (
                "One shared run_e3 call per repetition (2 model calls); "
                "E4-current and E4-refined are derived from that same call "
                "via apply_conflict_veto/apply_refined_veto with zero "
                "additional model calls. Expected total model calls per "
                "fixture = repetitions * 2, not repetitions * len(CONDITIONS) * 2."
            ),
            "model_lifecycle": "warm",
            "warmup": "one discarded model request before all measured runs",
            "standard_timeout_seconds": args.timeout,
            "failed_run_policy": "record failure and stop; never substitute model; no reruns",
            "minimum_e3_errors_for_robust_veto_estimate": MINIMUM_E3_ERRORS_FOR_ROBUST_VETO_ESTIMATE,
            "success_criteria": SUCCESS_CRITERIA_DESCRIPTIONS,
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
                started = time.perf_counter()
                shared = run_in_subprocess(
                    _e4_precision_worker,
                    (str(staged_fixture), args.model, args.think),
                    timeout=args.timeout,
                )
                status = shared.get("status", "error")
                if status == "ok":
                    for field in SECURITY_ZERO_FIELDS:
                        value = int(shared["telemetry"].get(field, 0) or 0)
                        if value:
                            raise RuntimeError(
                                f"E4-precision development cycle is metadata-only but "
                                f"{field}={value}"
                            )
                elapsed = time.perf_counter() - started
                if status == "ok":
                    for position, condition in enumerate(order, 1):
                        cond_data = shared["conditions"][condition]
                        run = {
                            "experiment_id": experiment_id,
                            "fixture": fixture_name,
                            "repetition": repetition,
                            "sequence_position": position,
                            "condition": condition,
                            "status": "ok",
                            "shared_e3_call": True,
                            "total_run_latency_seconds": elapsed,
                            "metrics": {
                                "status": "ok",
                                "sources": shared["sources"],
                                "final": cond_data["final"],
                                "detail": cond_data["detail"],
                                "telemetry": shared["telemetry"],
                            },
                        }
                        runs.append(run)
                        with raw_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(run, ensure_ascii=False, sort_keys=True) + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                else:
                    run = {
                        "experiment_id": experiment_id,
                        "fixture": fixture_name,
                        "repetition": repetition,
                        "condition": None,
                        "status": status,
                        "error": shared.get("error"),
                        "total_run_latency_seconds": elapsed,
                    }
                    runs.append(run)
                    with raw_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(run, ensure_ascii=False, sort_keys=True) + "\n")
                    aborted = True
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
            len([r for r in runs_by_fixture.get(name, []) if r["status"] == "ok"]) == expected_run_count
            for name in args.fixtures
        )
        result: dict[str, Any] = {"experiment_id": experiment_id, "complete": complete}
        if complete:
            review_directory = load_rules().review_directory
            result["per_fixture"] = {
                name: aggregate(runs_by_fixture[name], expected_by_fixture[name], review_directory=review_directory)
                for name in args.fixtures
            }
            if "e3_error_calibration" in args.fixtures:
                # Items 35/38/41/42/43: new-fixture-only diagnostics, kept
                # under a dedicated top-level key so they are never hidden
                # inside a combined-only result (item 31 of the fixture brief).
                result["e3_error_calibration_diagnostics"] = compute_e3_error_calibration_diagnostics(
                    runs_by_fixture["e3_error_calibration"],
                    expected_by_fixture["e3_error_calibration"],
                    review_directory=review_directory,
                )
            if len(args.fixtures) > 1:
                result["combined"] = _combined(
                    result["per_fixture"], expected_by_fixture, runs_by_fixture,
                    review_directory=review_directory,
                )
                # Primary/decision inputs MUST come from unique_file_metrics (item
                # 3/12/23): repeated_observation_metrics is repetition-multiplied
                # and diagnostic-only -- feeding it here was the historical bug.
                combined_unique = result["combined"]["unique_file_metrics"]
                result["evidence_strength"] = compute_evidence_strength(combined_unique["E3"])
                result["evidence_strength_note"] = result["evidence_strength"]["note"]
                if "E4-current" in combined_unique and "E4-refined" in combined_unique:
                    result["success_criteria_evaluation"] = evaluate_success_criteria(
                        unique_e3_automatic_errors=result["evidence_strength"]["unique_e3_automatic_errors"],
                        e4_current_combined=combined_unique["E4-current"],
                        e4_refined_combined=combined_unique["E4-refined"],
                        per_fixture_current={
                            name: result["per_fixture"][name]["unique_file_metrics"]["E4-current"]
                            for name in args.fixtures
                        },
                        per_fixture_refined={
                            name: result["per_fixture"][name]["unique_file_metrics"]["E4-refined"]
                            for name in args.fixtures
                        },
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
    print(f"E4-precision development artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
