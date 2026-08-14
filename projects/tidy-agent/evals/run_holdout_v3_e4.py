"""One-time live runner for Holdout v3, the frozen E3 -> E4-current pipeline.

Built directly from the Holdout v3 construction specification and the
narrow, already-frozen production interfaces in ``src/tidy/classification.py``
and ``evals/post_holdout_candidates.py`` -- no prior Holdout runner was read
or used as a template for this file.

This module supports exactly one pipeline: frozen E3 followed by frozen
E4-current (``evals.post_holdout_candidates.run_e4``). There is no
candidate-selection option, no E4-refined branch, no E5 branch, and no
alternative experimental mode. E4-current adds zero additional model calls
beyond E3's own two classification passes.

Consumption semantics: Holdout v3 (``evals/holdout_v3/``) is *unconsumed*
until the first measured model request that includes Holdout-v3 fixture
metadata is issued. At that instant Holdout v3 becomes **permanently
consumed** -- this remains true even if the run subsequently fails.
Before that instant, integrity or infrastructure problems may be repaired
freely and re-attempted; nothing has been spent yet. After that instant,
the Holdout must not be rerun, and any partial result artifacts must be
preserved rather than discarded.

No warmup uses Holdout-v3 metadata. This runner performs no warmup at all;
if a caller needs one, it must use synthetic data outside this module and
must not touch ``evals/holdout_v3/fixture``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from evals.post_holdout_candidates import run_e4

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_DIR = Path(__file__).resolve().parent / "holdout_v3"
FIXTURE_DIR = HOLDOUT_DIR / "fixture"
EXPECTED_PATH = HOLDOUT_DIR / "expected.yaml"
MANIFEST_PATH = HOLDOUT_DIR / "fixture_manifest.json"
CANDIDATE_SELECTION_PATH = HOLDOUT_DIR / "candidate_selection.json"
CODE_PINS_PATH = HOLDOUT_DIR / "code_pins.json"
CONSUMED_MARKER_PATH = HOLDOUT_DIR / "CONSUMED.json"
RESULTS_ROOT = REPO_ROOT / "evals" / "results"

REVIEW_DIRECTORY = "_ToReview"
REAL_CATEGORIES = ("Documents", "Code", "Images", "Archives", "Installers")
SELECTED_CANDIDATE = "E4-current"

EXPECTED_MODEL_ID = "ollama_chat/qwen3.5:4b"
EXPECTED_MODEL_DIGEST = "2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd"
EXPECTED_QUANTIZATION = "Q4_K_M"
EXPECTED_TEMPERATURE = 0
EXPECTED_THINKING_ENABLED = False
EXPECTED_NUM_CTX = 8192


class HoldoutIntegrityError(RuntimeError):
    """Raised strictly before any measured request; Holdout v3 stays unconsumed."""


class ModelIdentityError(RuntimeError):
    """Raised strictly before any measured request; Holdout v3 stays unconsumed."""


class ResultDirectoryExistsError(RuntimeError):
    """Raised strictly before any measured request; Holdout v3 stays unconsumed."""


class HoldoutAlreadyConsumedError(RuntimeError):
    """Holdout v3 was already consumed by an earlier run; it cannot be rerun."""


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    digest: str
    quantization: str
    temperature: float
    thinking_enabled: bool
    num_ctx: int


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _normalized(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


# ---------------------------------------------------------------------------
# Pre-measured-request integrity gates
# ---------------------------------------------------------------------------


def verify_code_pins() -> dict[str, str]:
    """Verify pinned code hashes. Must pass before any measured request."""
    pins = json.loads(CODE_PINS_PATH.read_text(encoding="utf-8"))["sha256"]
    actual: dict[str, str] = {}
    for rel_path, expected_hash in pins.items():
        digest = _sha256_file(REPO_ROOT / rel_path)
        actual[rel_path] = digest
        if digest != expected_hash:
            raise HoldoutIntegrityError(
                f"code hash mismatch for {rel_path}: expected {expected_hash}, got {digest}"
            )
    return actual


def verify_fixture_hash() -> tuple[str, str]:
    """Recompute fixture/ground-truth hashes and compare to the frozen manifest."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    names = [p.name for p in FIXTURE_DIR.iterdir() if p.is_file()]
    if len(names) != manifest["total_files"]:
        raise HoldoutIntegrityError(
            f"fixture file count mismatch: expected {manifest['total_files']}, got {len(names)}"
        )
    listing = "\n".join(sorted(names)).encode("utf-8")
    dataset_digest = _sha256_bytes(listing)
    if dataset_digest != manifest["dataset_sha256"]:
        raise HoldoutIntegrityError(
            f"fixture dataset hash mismatch: expected {manifest['dataset_sha256']}, got {dataset_digest}"
        )
    ground_truth_digest = _sha256_file(EXPECTED_PATH)
    if ground_truth_digest != manifest["ground_truth_sha256"]:
        raise HoldoutIntegrityError(
            "ground-truth hash mismatch: expected "
            f"{manifest['ground_truth_sha256']}, got {ground_truth_digest}"
        )
    return dataset_digest, ground_truth_digest


def verify_model_identity(identity: ModelIdentity) -> None:
    """Must pass before any measured request. Mismatch means STOP -- no substitution."""
    mismatches = []
    if identity.model_id != EXPECTED_MODEL_ID:
        mismatches.append(f"model_id: expected {EXPECTED_MODEL_ID!r}, got {identity.model_id!r}")
    if identity.digest != EXPECTED_MODEL_DIGEST:
        mismatches.append(f"digest: expected {EXPECTED_MODEL_DIGEST!r}, got {identity.digest!r}")
    if identity.quantization != EXPECTED_QUANTIZATION:
        mismatches.append(
            f"quantization: expected {EXPECTED_QUANTIZATION!r}, got {identity.quantization!r}"
        )
    if identity.temperature != EXPECTED_TEMPERATURE:
        mismatches.append(
            f"temperature: expected {EXPECTED_TEMPERATURE!r}, got {identity.temperature!r}"
        )
    if identity.thinking_enabled != EXPECTED_THINKING_ENABLED:
        mismatches.append(
            "thinking_enabled: expected "
            f"{EXPECTED_THINKING_ENABLED!r}, got {identity.thinking_enabled!r}"
        )
    if identity.num_ctx != EXPECTED_NUM_CTX:
        mismatches.append(f"num_ctx: expected {EXPECTED_NUM_CTX!r}, got {identity.num_ctx!r}")
    if mismatches:
        raise ModelIdentityError("; ".join(mismatches))


def is_consumed() -> bool:
    return CONSUMED_MARKER_PATH.exists()


def _read_consumption_record() -> dict[str, Any] | None:
    if not CONSUMED_MARKER_PATH.exists():
        return None
    return json.loads(CONSUMED_MARKER_PATH.read_text(encoding="utf-8"))


def _mark_consumed(*, timestamp: str, result_dir_name: str, model_identity: ModelIdentity) -> None:
    """Written synchronously, immediately before the first measured request is
    issued -- so an interrupted process still leaves Holdout v3 consumed."""
    CONSUMED_MARKER_PATH.write_text(
        json.dumps(
            {
                "consumed": True,
                "first_measured_request_timestamp": timestamp,
                "result_dir": result_dir_name,
                "selected_candidate": SELECTED_CANDIDATE,
                "model_digest": model_identity.digest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Loading fixture + ground truth (evaluator-only data stays out of the prompt)
# ---------------------------------------------------------------------------


def load_ground_truth() -> dict[str, dict[str, Any]]:
    doc = yaml.safe_load(EXPECTED_PATH.read_text(encoding="utf-8"))
    return {case["filename"]: case for case in doc["cases"]}


def load_fixture_metadata() -> list[dict[str, str]]:
    """Production-approved metadata only. Every dict has exactly one key,
    ``name`` -- no ground truth, rationale, stratum, language, tags, or
    instruction-like flag ever reaches this structure, so it cannot reach a
    model prompt built from it."""
    names = sorted(p.name for p in FIXTURE_DIR.iterdir() if p.is_file())
    return [{"name": name} for name in names]


# ---------------------------------------------------------------------------
# Result directory
# ---------------------------------------------------------------------------


def create_result_directory(freeze_tag: str) -> Path:
    date_str = time.strftime("%Y%m%d")
    result_dir = RESULTS_ROOT / f"holdout-v3-e4-current-{date_str}-{freeze_tag}"
    if result_dir.exists():
        raise ResultDirectoryExistsError(f"result directory already exists: {result_dir.name}")
    result_dir.mkdir(parents=True)
    return result_dir


# ---------------------------------------------------------------------------
# Metrics (items 36-43)
# ---------------------------------------------------------------------------


def _decision_bucket(predicted: str, expected: str) -> str:
    if predicted == REVIEW_DIRECTORY:
        return "review"
    if predicted == expected:
        return "correct_automatic"
    return "incorrect_automatic"


def _primary_metrics(sources: Sequence[str], final: Mapping[str, str], ground_truth: Mapping[str, Any]) -> dict[str, Any]:
    total = len(sources)
    correct_automatic = incorrect_automatic = review = 0
    for source in sources:
        expected = ground_truth[source]["expected_outcome"]
        bucket = _decision_bucket(final[source], expected)
        if bucket == "correct_automatic":
            correct_automatic += 1
        elif bucket == "incorrect_automatic":
            incorrect_automatic += 1
        else:
            review += 1
    automatic = correct_automatic + incorrect_automatic
    predicted_review_and_expected_review = sum(
        1
        for source in sources
        if final[source] == REVIEW_DIRECTORY and ground_truth[source]["expected_outcome"] == REVIEW_DIRECTORY
    )
    expected_review_total = sum(
        1 for source in sources if ground_truth[source]["expected_outcome"] == REVIEW_DIRECTORY
    )
    predicted_review_total = review
    unsafe_automation = sum(
        1
        for source in sources
        if final[source] != REVIEW_DIRECTORY and ground_truth[source]["expected_outcome"] == REVIEW_DIRECTORY
    )
    return {
        "total_unique_files": total,
        "correct_automatic": correct_automatic,
        "incorrect_automatic": incorrect_automatic,
        "review": review,
        "strict_accuracy": (correct_automatic / total) if total else None,
        "unsafe_automation_count": unsafe_automation,
        "unsafe_automation_rate": (unsafe_automation / total) if total else None,
        "automation_coverage": (automatic / total) if total else None,
        "review_rate": (predicted_review_total / total) if total else None,
        "accuracy_on_decided": (correct_automatic / automatic) if automatic else None,
        "review_recall": (
            predicted_review_and_expected_review / expected_review_total
            if expected_review_total
            else None
        ),
        "review_precision": (
            predicted_review_and_expected_review / predicted_review_total
            if predicted_review_total
            else None
        ),
    }


def _real_category_subset_metrics(sources: Sequence[str], final: Mapping[str, str], ground_truth: Mapping[str, Any]) -> dict[str, Any]:
    real_sources = [s for s in sources if ground_truth[s]["expected_outcome"] != REVIEW_DIRECTORY]
    n = len(real_sources)
    correct_automatic = wrong_automatic = false_review = 0
    for source in real_sources:
        expected = ground_truth[source]["expected_outcome"]
        predicted = final[source]
        if predicted == REVIEW_DIRECTORY:
            false_review += 1
        elif predicted == expected:
            correct_automatic += 1
        else:
            wrong_automatic += 1
    return {
        "n": n,
        "correct_automatic": correct_automatic,
        "wrong_automatic": wrong_automatic,
        "false_review": false_review,
        "correct_automation_rate": correct_automatic / n if n else None,
        "wrong_category_rate": wrong_automatic / n if n else None,
        "false_review_rate": false_review / n if n else None,
    }


def _review_subset_metrics(sources: Sequence[str], final: Mapping[str, str], ground_truth: Mapping[str, Any]) -> dict[str, Any]:
    review_sources = [s for s in sources if ground_truth[s]["expected_outcome"] == REVIEW_DIRECTORY]
    n = len(review_sources)
    correctly_reviewed = sum(1 for s in review_sources if final[s] == REVIEW_DIRECTORY)
    incorrectly_automated = n - correctly_reviewed
    return {
        "n": n,
        "correctly_reviewed": correctly_reviewed,
        "incorrectly_automated": incorrectly_automated,
        "review_recall": correctly_reviewed / n if n else None,
    }


def _e3_internal_diagnostics(detail: Sequence[Mapping[str, Any]], ground_truth: Mapping[str, Any]) -> dict[str, Any]:
    total = len(detail)
    correct_automatic = incorrect_automatic = review = unsafe_automation = 0
    for item in detail:
        source = item["filename"]
        expected = ground_truth[source]["expected_outcome"]
        e3_category = item["e3_category"]
        if e3_category == REVIEW_DIRECTORY:
            review += 1
        elif e3_category == expected:
            correct_automatic += 1
        else:
            incorrect_automatic += 1
        if e3_category != REVIEW_DIRECTORY and expected == REVIEW_DIRECTORY:
            unsafe_automation += 1
    coverage = (correct_automatic + incorrect_automatic) / total if total else None
    return {
        "e3_correct_automatic": correct_automatic,
        "e3_incorrect_automatic": incorrect_automatic,
        "e3_review": review,
        "e3_unsafe_automation": unsafe_automation,
        "e3_coverage": coverage,
    }


def _e3_gate_diagnostics(detail: Sequence[Mapping[str, Any]], ground_truth: Mapping[str, Any], telemetry: Mapping[str, Any]) -> dict[str, Any]:
    buckets = {
        "classify_classify_same": [],
        "classify_classify_different": [],
        "classify_review": [],
        "review_classify": [],
        "review_review": [],
        "invalid": [],
    }
    agreement_to_bucket = {
        "agree_classify": "classify_classify_same",
        "disagree_classify": "classify_classify_different",
        "both_invalid": "invalid",
    }
    for item in detail:
        agreement = item["e3_agreement"]
        if agreement in agreement_to_bucket:
            buckets[agreement_to_bucket[agreement]].append(item)
        elif agreement == "review_involved":
            p1, p2 = item["e3_pass1_decision"], item["e3_pass2_decision"]
            if p1 == "classify" and p2 == "review":
                buckets["classify_review"].append(item)
            elif p1 == "review" and p2 == "classify":
                buckets["review_classify"].append(item)
            else:
                buckets["review_review"].append(item)
        else:
            buckets["invalid"].append(item)

    same = buckets["classify_classify_same"]
    correct = sum(1 for item in same if item["e3_category"] == ground_truth[item["filename"]]["expected_outcome"])
    incorrect = len(same) - correct

    return {
        "classify_classify_same_count": len(same),
        "classify_classify_same_correct": correct,
        "classify_classify_same_incorrect": incorrect,
        "classify_classify_same_error_rate": (incorrect / len(same)) if same else None,
        "classify_classify_different_count": len(buckets["classify_classify_different"]),
        "classify_review_count": len(buckets["classify_review"]),
        "review_classify_count": len(buckets["review_classify"]),
        "review_review_count": len(buckets["review_review"]),
        "invalid_count": len(buckets["invalid"]),
        "telemetry_gate_counts": {
            key: telemetry.get(key)
            for key in (
                "gate_classify_same_category_count",
                "gate_classify_different_category_count",
                "gate_classify_then_review_count",
                "gate_review_then_classify_count",
                "gate_review_review_count",
                "gate_invalid_count",
            )
        },
    }


def _e4_diagnostics(detail: Sequence[Mapping[str, Any]], ground_truth: Mapping[str, Any]) -> dict[str, Any]:
    automatic_candidates = [item for item in detail if item["e3_category"] != REVIEW_DIRECTORY]
    accepted = [item for item in automatic_candidates if item["final"] == item["e3_category"]]
    vetoed = [item for item in automatic_candidates if item["final"] != item["e3_category"]]

    def _e3_wrong(item: Mapping[str, Any]) -> bool:
        return item["e3_category"] != ground_truth[item["filename"]]["expected_outcome"]

    true_positive_vetoes = sum(1 for item in vetoed if _e3_wrong(item))
    false_positive_vetoes = len(vetoed) - true_positive_vetoes
    e3_errors_surviving = sum(1 for item in accepted if _e3_wrong(item))
    total_e3_errors_among_automatic = true_positive_vetoes + e3_errors_surviving

    reason_codes: dict[str, int] = {}
    for item in detail:
        reason_codes[item["veto_reason_code"]] = reason_codes.get(item["veto_reason_code"], 0) + 1

    return {
        "e3_automatic_candidates": len(automatic_candidates),
        "accepted": len(accepted),
        "vetoed": len(vetoed),
        "true_positive_vetoes": true_positive_vetoes,
        "false_positive_vetoes": false_positive_vetoes,
        "e3_errors_surviving": e3_errors_surviving,
        "veto_precision": (true_positive_vetoes / len(vetoed)) if vetoed else None,
        "veto_recall": (
            true_positive_vetoes / total_e3_errors_among_automatic
            if total_e3_errors_among_automatic
            else None
        ),
        "fp_tp_ratio": (
            false_positive_vetoes / true_positive_vetoes if true_positive_vetoes else None
        ),
        "coverage_loss": (len(vetoed) / len(automatic_candidates)) if automatic_candidates else None,
        "reason_code_distribution": reason_codes,
    }


PRIMARY_STRATA = (
    "ordinary_realistic",
    "contextual_relational",
    "distractor_rich_resolvable",
    "insufficient_metadata",
    "latent_dual_role",
    "container_artifact_role_ambiguity",
)


def _stratum_diagnostics(sources: Sequence[str], final: Mapping[str, str], ground_truth: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for stratum in PRIMARY_STRATA:
        stratum_sources = [s for s in sources if ground_truth[s]["primary_stratum"] == stratum]
        result[stratum] = _primary_metrics(stratum_sources, final, ground_truth)
    return result


def _instruction_like_diagnostics(sources: Sequence[str], final: Mapping[str, str], ground_truth: Mapping[str, Any]) -> dict[str, Any]:
    subset = [s for s in sources if ground_truth[s]["instruction_like"]]
    correct_automatic = incorrect_automatic = review = 0
    for source in subset:
        bucket = _decision_bucket(final[source], ground_truth[source]["expected_outcome"])
        if bucket == "correct_automatic":
            correct_automatic += 1
        elif bucket == "incorrect_automatic":
            incorrect_automatic += 1
        else:
            review += 1
    return {
        "n": len(subset),
        "correct_automatic": correct_automatic,
        "incorrect_automatic": incorrect_automatic,
        "review": review,
    }


def build_report(
    final: Mapping[str, str],
    detail: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    sources: Sequence[str],
) -> dict[str, Any]:
    if len(set(sources)) != len(sources):
        raise HoldoutIntegrityError("duplicate source filenames in fixture metadata")
    return {
        "primary_metrics": _primary_metrics(sources, final, ground_truth),
        "real_category_subset": _real_category_subset_metrics(sources, final, ground_truth),
        "review_subset": _review_subset_metrics(sources, final, ground_truth),
        "e3_internal_diagnostics": _e3_internal_diagnostics(detail, ground_truth),
        "e3_gate_diagnostics": _e3_gate_diagnostics(detail, ground_truth, telemetry),
        "e4_diagnostics": _e4_diagnostics(detail, ground_truth),
        "stratum_diagnostics": _stratum_diagnostics(sources, final, ground_truth),
        "instruction_like_diagnostic": _instruction_like_diagnostics(sources, final, ground_truth),
    }


# ---------------------------------------------------------------------------
# Result artifacts (item 45-46)
# ---------------------------------------------------------------------------


def _write_result_artifacts(
    result_dir: Path,
    *,
    metadata: Sequence[Mapping[str, str]],
    final: Mapping[str, str],
    detail: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
    report: Mapping[str, Any],
    consumed: bool,
    timestamp: str,
    run_complete: bool,
    model_identity: ModelIdentity,
    code_hashes: Mapping[str, str],
    fixture_hashes: tuple[str, str],
) -> None:
    manifest = {
        "consumed": consumed,
        "first_measured_request_timestamp": timestamp,
        "run_complete": run_complete,
        "selected_candidate": SELECTED_CANDIDATE,
        "code_hashes": dict(code_hashes),
        "fixture_dataset_sha256": fixture_hashes[0],
        "fixture_ground_truth_sha256": fixture_hashes[1],
        "model_digest": model_identity.digest,
        "model_id": model_identity.model_id,
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (result_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with (result_dir / "raw_run.jsonl").open("w", encoding="utf-8") as f:
        for item in detail:
            f.write(json.dumps({"filename": item["filename"], "final": item["final"]}) + "\n")

    per_file_evidence = [
        {
            "filename": item["filename"],
            "e3_category": item["e3_category"],
            "e3_agreement": item["e3_agreement"],
            "veto_reason_code": item["veto_reason_code"],
            "final": item["final"],
        }
        for item in detail
    ]
    (result_dir / "per_file_evidence.json").write_text(
        json.dumps(per_file_evidence, indent=2) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# Holdout v3 -- E4-current report",
        "",
        f"Consumed: {consumed}",
        f"Run complete: {run_complete}",
        f"First measured request: {timestamp}",
        "",
        "## Primary metrics",
        "```json",
        json.dumps(report["primary_metrics"], indent=2),
        "```",
    ]
    (result_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _write_partial_failure_artifacts(result_dir: Path, timestamp: str, model_identity: ModelIdentity, exc: Exception) -> None:
    manifest = {
        "consumed": True,
        "first_measured_request_timestamp": timestamp,
        "run_complete": False,
        "selected_candidate": SELECTED_CANDIDATE,
        "model_digest": model_identity.digest,
        "failure": str(exc),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_holdout_v3(model: Any, *, freeze_tag: str, model_identity: ModelIdentity) -> Path:
    """Exactly one evaluation of the 120-file Holdout v3 fixture.

    Consumes Holdout v3 on the first measured request (immediately before it
    is issued). Never mutates fixture files. Never reruns a consumed
    Holdout. Pipeline is fixed: frozen E3 + frozen E4-current.
    """
    if is_consumed():
        raise HoldoutAlreadyConsumedError(
            "Holdout v3 was already consumed; it cannot be rerun on this or any candidate."
        )

    code_hashes = verify_code_pins()
    fixture_hashes = verify_fixture_hash()
    verify_model_identity(model_identity)

    result_dir = create_result_directory(freeze_tag)

    metadata = load_fixture_metadata()
    for item in metadata:
        if set(item) != {"name"}:
            raise HoldoutIntegrityError("fixture metadata must carry only 'name'")
    ground_truth = load_ground_truth()
    sources = [m["name"] for m in metadata]
    if set(sources) != set(ground_truth):
        raise HoldoutIntegrityError("fixture/ground-truth source mismatch")

    fixture_snapshot_before = sorted(p.name for p in FIXTURE_DIR.iterdir() if p.is_file())

    started = time.time()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    _mark_consumed(timestamp=timestamp, result_dir_name=result_dir.name, model_identity=model_identity)

    try:
        final, detail, telemetry = run_e4(
            model, metadata, list(REAL_CATEGORIES), review_directory=REVIEW_DIRECTORY
        )
        fixture_snapshot_after = sorted(p.name for p in FIXTURE_DIR.iterdir() if p.is_file())
        if fixture_snapshot_after != fixture_snapshot_before:
            raise HoldoutIntegrityError("fixture directory was mutated during evaluation")
        report = build_report(final, detail, telemetry, ground_truth, sources)
        _write_result_artifacts(
            result_dir,
            metadata=metadata,
            final=final,
            detail=detail,
            telemetry=telemetry,
            report=report,
            consumed=True,
            timestamp=timestamp,
            run_complete=True,
            model_identity=model_identity,
            code_hashes=code_hashes,
            fixture_hashes=fixture_hashes,
        )
    except Exception as exc:
        _write_partial_failure_artifacts(result_dir, timestamp, model_identity, exc)
        raise
    return result_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """No candidate-selection option: this CLI supports exactly one pipeline
    (frozen E3 + frozen E4-current). Factored out so tests can inspect the
    parser's option surface without invoking ``main``."""
    parser = argparse.ArgumentParser(
        description="One-time live Holdout v3 evaluation of the frozen E3 -> E4-current pipeline."
    )
    parser.add_argument("--freeze-tag", required=True, help="Freeze tag used in the result directory name.")
    parser.add_argument("--confirm-consumes-holdout-v3", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.confirm_consumes_holdout_v3:
        raise SystemExit("Refusing to run without --confirm-consumes-holdout-v3")
    raise SystemExit(
        "This CLI entry point requires a live model client (Ollama qwen3.5:4b) to be "
        "wired in by the caller; run_holdout_v3() must be invoked programmatically with "
        "a real model object and its verified ModelIdentity. Not implemented as a bare "
        "CLI to avoid accidental live invocation."
    )


if __name__ == "__main__":
    raise SystemExit(main())
