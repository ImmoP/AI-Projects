"""The single permitted live Holdout v5 evaluation runner. Pipeline: E4-batched.

Frozen under ``evals/holdout_v5_protocol.md``. Holdout v5 evaluates the
**revised** system version ``E4-batched`` (batched E3 transport + unchanged
E4-current veto) -- it is *not* a rerun of the frozen ``E3 -> E4-current``
system that Holdout v4 consumed, and it is *not* a second attempt on Holdout
v4. Holdout v4 remains consumed and ``PARTIAL_INCONCLUSIVE`` forever; nothing
here reads, reuses, or reinterprets any Holdout v4 artifact.

This module mirrors ``evals/run_holdout_v4_e4.py``'s protections (frozen
candidate pin, code pins, fixture/ground-truth hashes, authoring-freeze marker,
blind historical-overlap audit, exactly-one consumption guard, fail-closed
validity rule, durable atomic persistence) but is a *separate* runner for a
*separate* Holdout and candidate. The frozen v4 runner is not imported or
modified.

Before any measured provider request this verifies, in order: frozen candidate
selection, fixture/ground-truth hashes vs the frozen manifest, sha256 of every
pinned source file, the blind historical-overlap audit (exact overlap must be
zero), live model identity vs the frozen pin, and that the local
``CONSUMED.json`` marker does not yet exist. Any failure raises/reports before
any dispatch and before ``CONSUMED.json`` is written, leaving Holdout v5
unconsumed. ``CONSUMED.json`` is written durably, atomically, immediately
before the first measured provider request; after that write Holdout v5 is
permanently consumed regardless of what happens later -- no rerun path.

CLI (run as a module from the project root so ``evals`` resolves as a package)::

    python -m evals.run_holdout_v5_e4 --preflight
    python -m evals.run_holdout_v5_e4 --run [--run-label LABEL]

``--preflight`` performs only non-consuming checks and never sends a Holdout
v5 filename to any model. ``--run`` repeats every preflight check and then runs
the single measured Holdout v5 evaluation. Running with neither flag prints
help and exits without touching any Holdout v5 state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tidy.agent import build_model, endpoint_is_local, resolved_model_endpoint
from tidy.classification import (
    DEFAULT_CLASSIFICATION_BATCH_SIZE,
    chunk_classification_metadata,
)

from evals import one_time_eval_runtime as runtime
from evals.e4_batched import CANDIDATE_DESIGNATION, CANDIDATE_PIPELINE, run_e4_batched
from evals.holdout_v5 import blind_audit

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_DIR = Path(__file__).resolve().parent / "holdout_v5"
FIXTURE_DIR = HOLDOUT_DIR / "fixture"
GROUND_TRUTH_PATH = HOLDOUT_DIR / "ground_truth.json"
CODE_PINS_PATH = HOLDOUT_DIR / "code_pins.json"
CONSUMED_PATH = HOLDOUT_DIR / "CONSUMED.json"
CONSUMPTION_RECORD_PATH = HOLDOUT_DIR / "consumption_record.json"
CANDIDATE_SELECTION_PATH = HOLDOUT_DIR / "candidate_selection.json"
AUTHORING_FROZEN_PATH = HOLDOUT_DIR / "AUTHORING_FROZEN.json"
FIXTURE_MANIFEST_PATH = HOLDOUT_DIR / "fixture_manifest.json"
RESULTS_ROOT = REPO_ROOT / "evals" / "results"

REVIEW_DIRECTORY = "_ToReview"
REAL_CATEGORIES = ("Documents", "Code", "Images", "Archives", "Installers")
PIPELINE_LABEL = "E4-batched(holdout-v5)"
EXPECTED_BATCH_SIZE = DEFAULT_CLASSIFICATION_BATCH_SIZE

# The revised transport changes *how* requests are grouped, not the model. For
# comparability with the system v4 tried to evaluate, the same local model and
# decoding configuration are pinned.
EXPECTED_MODEL_ID = "ollama_chat/qwen3.5:4b"
EXPECTED_MODEL_DIGEST = "2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd"
EXPECTED_QUANTIZATION = "Q4_K_M"
EXPECTED_TEMPERATURE = 0
EXPECTED_THINKING_ENABLED = False
EXPECTED_NUM_CTX = 8192

ModelIdentity = runtime.ModelIdentity


def expected_model_identity() -> ModelIdentity:
    return ModelIdentity(
        model_id=EXPECTED_MODEL_ID,
        digest=EXPECTED_MODEL_DIGEST,
        quantization=EXPECTED_QUANTIZATION,
        temperature=EXPECTED_TEMPERATURE,
        thinking_enabled=EXPECTED_THINKING_ENABLED,
        num_ctx=EXPECTED_NUM_CTX,
    )


def expected_provider_request_count(source_count: int, batch_size: int = EXPECTED_BATCH_SIZE) -> int:
    """E3's fixed two-pass protocol, batched: 2 requests per batch."""
    return 2 * len(chunk_classification_metadata([{}] * source_count, batch_size))


class CodePinMismatchError(RuntimeError):
    """Raised before any measured request when a pinned source file's sha256
    no longer matches ``code_pins.json``. No substitution is attempted."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_code_pins(repo_root: Path = REPO_ROOT, pins_path: Path = CODE_PINS_PATH) -> None:
    pins = json.loads(pins_path.read_text(encoding="utf-8"))["pins"]
    mismatches = []
    for rel_path, expected_sha in pins.items():
        actual = _file_sha256(repo_root / rel_path)
        if actual != expected_sha:
            mismatches.append(f"{rel_path}: expected {expected_sha}, got {actual}")
    if mismatches:
        raise CodePinMismatchError("; ".join(mismatches))


def _fixture_hash(names: Sequence[str]) -> str:
    listing = "\n".join(sorted(unicodedata.normalize("NFC", n) for n in names)).encode("utf-8")
    return hashlib.sha256(listing).hexdigest()


def load_holdout_metadata(fixture_dir: Path = FIXTURE_DIR) -> list[dict[str, str]]:
    """Production-approved metadata only: every dict carries exactly ``name``."""
    names = sorted(p.name for p in fixture_dir.iterdir() if p.is_file())
    return [{"name": name} for name in names]


def load_ground_truth(ground_truth_path: Path = GROUND_TRUTH_PATH) -> dict[str, str]:
    """Evaluator-only expected outcomes. Never sent to the model."""
    records = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    return {r["filename"]: r["expected_outcome"] for r in records}


class CandidateSelectionError(RuntimeError):
    """Raised before any measured request when the frozen candidate selection
    is missing, not closed, or does not designate the E4-batched candidate."""


def verify_candidate_selection(path: Path = CANDIDATE_SELECTION_PATH) -> dict[str, Any]:
    """Fail closed unless the frozen selection designates E4-batched, is
    closed, and has not already run inference."""
    try:
        selection = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CandidateSelectionError(f"cannot read candidate_selection.json: {exc}") from exc
    if selection.get("selected_candidate") != CANDIDATE_DESIGNATION:
        raise CandidateSelectionError(
            f"selected_candidate must be {CANDIDATE_DESIGNATION!r}, got "
            f"{selection.get('selected_candidate')!r}"
        )
    if selection.get("candidate_pipeline") != CANDIDATE_PIPELINE:
        raise CandidateSelectionError(
            f"candidate_pipeline must be {CANDIDATE_PIPELINE!r}, got "
            f"{selection.get('candidate_pipeline')!r}"
        )
    if selection.get("selection_closed") is not True:
        raise CandidateSelectionError("candidate selection is not closed")
    if selection.get("holdout_v5_inference_performed") is not False:
        raise CandidateSelectionError("candidate selection already records inference performed")
    return selection


class HoldoutV5RunFailed(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Precommitted scoring metrics and the fail-closed validity rule
# ---------------------------------------------------------------------------
#
# The metric *definitions* below are precommitted in
# ``evals/holdout_v5_protocol.md`` before any case is evaluated. The primary
# portfolio metrics are automation precision, automation rate, and review
# recall; overall exact-decision accuracy is explicitly secondary to the
# safety metrics. These are computed from run output only after inference.


def _compute_scoring(
    sources: Sequence[str],
    final: Mapping[str, str],
    ground_truth: Mapping[str, str],
) -> dict[str, Any]:
    total = len(sources)
    auto = [s for s in sources if final.get(s) != REVIEW_DIRECTORY]
    review = [s for s in sources if final.get(s) == REVIEW_DIRECTORY]
    correct_automatic = [s for s in auto if final.get(s) == ground_truth.get(s)]
    incorrect_automatic = [s for s in auto if final.get(s) != ground_truth.get(s)]
    gt_review = [s for s in sources if ground_truth.get(s) == REVIEW_DIRECTORY]
    routed_to_review = [s for s in gt_review if final.get(s) == REVIEW_DIRECTORY]

    confusion: dict[str, dict[str, int]] = {}
    for s in sources:
        expected = ground_truth.get(s)
        predicted = final.get(s)
        if expected is None or predicted is None:
            continue
        confusion.setdefault(str(expected), {}).setdefault(str(predicted), 0)
        confusion[str(expected)][str(predicted)] += 1

    exact = sum(1 for s in sources if final.get(s) == ground_truth.get(s))
    return {
        "expected_source_count": total,
        "auto_count": len(auto),
        "review_count": len(review),
        "automation_rate": (len(auto) / total) if total else None,
        "correct_automatic": len(correct_automatic),
        "incorrect_automatic": len(incorrect_automatic),
        "automation_precision": (len(correct_automatic) / len(auto)) if auto else None,
        "ground_truth_review_count": len(gt_review),
        "correctly_routed_to_review": len(routed_to_review),
        "review_recall": (len(routed_to_review) / len(gt_review)) if gt_review else None,
        "matches_ground_truth_count": exact,
        "overall_exact_accuracy": (exact / total) if total else None,
        "confusion_by_category": confusion,
    }


def _evaluate_validity(
    sources: Sequence[str],
    final: Mapping[str, str],
    detail: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
    proxy: "runtime.JournalingModelProxy",
    expected_requests: int,
) -> dict[str, Any]:
    """Precommitted fail-closed validity rule (see holdout_v5_protocol.md).

    ``complete_valid`` requires zero provider errors, zero parse failures, zero
    schema-validation failures, zero batch-validation failures, zero
    length-truncation responses, every expected provider response received, and
    every source resolved by both the E3 gate and the E4 veto. Any transport
    defect marks the run ``partial_inconclusive`` -- the batching contains the
    blast radius (a failed batch's own sources fail closed to review and are
    reported), but the run is still reported transparently as not fully valid.
    """
    provider_errors = int(telemetry.get("provider_errors", 0))
    parse_failures = int(telemetry.get("parse_failures", 0))
    schema_failures = int(telemetry.get("schema_validation_failures", 0))
    batch_failures = int(telemetry.get("batch_validation_failures", 0))
    length_finish = int(telemetry.get("length_finish_responses", 0))
    all_required_provider_responses_received = (
        proxy.dispatch_attempted == expected_requests
        and proxy.dispatch_returned == expected_requests
        and proxy.dispatch_failed == 0
    )
    gate_resolved = int(telemetry.get("gate_final_automatic_count", 0)) + int(
        telemetry.get("gate_final_review_count", 0)
    )
    e3_gate_completed = gate_resolved == len(sources)
    e4_completed = set(final) == set(sources) and all(
        d.get("final") is not None for d in detail
    )
    evaluation_valid = (
        provider_errors == 0
        and parse_failures == 0
        and schema_failures == 0
        and batch_failures == 0
        and length_finish == 0
        and all_required_provider_responses_received
        and e3_gate_completed
        and e4_completed
    )
    return {
        "provider_errors": provider_errors,
        "parse_failures": parse_failures,
        "schema_failures": schema_failures,
        "batch_validation_failures": batch_failures,
        "length_finish_responses": length_finish,
        "all_required_provider_responses_received": all_required_provider_responses_received,
        "E3_gate_completed": e3_gate_completed,
        "E4_current_completed": e4_completed,
        "result_persistence_completed": False,
        "evaluation_valid": evaluation_valid,
        "evaluation_status": "pending_persistence" if evaluation_valid else "partial_inconclusive",
    }


def _persist_results(
    result_dir: Path,
    *,
    detail: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
    proxy: "runtime.JournalingModelProxy",
    scoring: Mapping[str, Any],
) -> None:
    raw_run_lines = "".join(
        json.dumps({"filename": item["filename"], "final": item["final"]}) + "\n"
        for item in detail
    )
    runtime.atomic_write_text(result_dir / "raw_run.jsonl", raw_run_lines)
    runtime.atomic_write_json(result_dir / "per_file_evidence.json", list(detail))
    summary = {
        "telemetry": dict(telemetry),
        "dispatch": {
            "attempted": proxy.dispatch_attempted,
            "returned": proxy.dispatch_returned,
            "failed": proxy.dispatch_failed,
        },
        "scoring": dict(scoring),
    }
    runtime.atomic_write_json(result_dir / "summary.json", summary)


def run_holdout_v5(
    model: Any,
    *,
    run_label: str,
    expected_identity: ModelIdentity,
    actual_identity: ModelIdentity,
    fixture_dir: Path = FIXTURE_DIR,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    code_pins_path: Path = CODE_PINS_PATH,
    candidate_selection_path: Path = CANDIDATE_SELECTION_PATH,
    consumed_path: Path = CONSUMED_PATH,
    consumption_record_path: Path = CONSUMPTION_RECORD_PATH,
    results_root: Path = RESULTS_ROOT,
    repo_root: Path = REPO_ROOT,
    batch_size: int = EXPECTED_BATCH_SIZE,
) -> Path:
    """Run the single permitted live Holdout v5 evaluation.

    STOPs unconsumed -- no ``consumed_path`` written, no result directory left
    behind -- on candidate-selection failure, model-identity mismatch, code-pin
    mismatch, or a result-directory collision; all four checks happen before
    ``create_fresh_result_directory`` and before ``consumed_path`` is written.
    Every other failure after that point still leaves Holdout v5 permanently
    consumed (see module docstring).
    """
    verify_candidate_selection(candidate_selection_path)
    runtime.verify_model_identity(expected_identity, actual_identity)
    verify_code_pins(repo_root, code_pins_path)

    date_str = time.strftime("%Y%m%d")
    result_dir = runtime.create_fresh_result_directory(
        results_root, f"holdout-v5-e4-{date_str}-{run_label}"
    )
    recorder = runtime.LifecycleRecorder(result_dir)

    metadata = load_holdout_metadata(fixture_dir)
    sources = [m["name"] for m in metadata]
    fixture_hash = _fixture_hash(sources)
    expected_requests = expected_provider_request_count(len(sources), batch_size)
    phase_labels = tuple(
        f"batch{index}_pass{pass_number}"
        for index in range(len(chunk_classification_metadata(metadata, batch_size)))
        for pass_number in (1, 2)
    )

    recorder.record_prepared(
        run_id=run_label,
        pipeline=PIPELINE_LABEL,
        model_identity_expected=expected_identity.as_dict(),
        fixture_hash=fixture_hash,
    )

    fixture_snapshot_before = sorted(p.name for p in fixture_dir.iterdir() if p.is_file())

    try:
        # Consumption boundary: durable, atomic, and the last thing written
        # before the first measured provider request. Everything above this
        # line may still leave Holdout v5 unconsumed on failure; nothing below
        # it may.
        runtime.atomic_write_json(
            consumed_path,
            {
                "consumed": True,
                "run_id": run_label,
                "pipeline": PIPELINE_LABEL,
                "candidate": CANDIDATE_DESIGNATION,
                "batch_size": batch_size,
                "fixture_hash": fixture_hash,
                "result_dir": result_dir.name,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

        proxy = runtime.JournalingModelProxy(model, recorder, phase_labels=phase_labels)
        final, detail, telemetry = run_e4_batched(
            proxy, metadata, list(REAL_CATEGORIES), review_directory=REVIEW_DIRECTORY,
            batch_size=batch_size,
        )

        fixture_snapshot_after = sorted(p.name for p in fixture_dir.iterdir() if p.is_file())
        if fixture_snapshot_after != fixture_snapshot_before:
            raise HoldoutV5RunFailed("holdout_v5 fixture directory was mutated during the run")

        recorder.record_state("scoring_complete")

        ground_truth = load_ground_truth(ground_truth_path)
        scoring = _compute_scoring(sources, final, ground_truth)
        validity = _evaluate_validity(
            sources, final, detail, telemetry, proxy, expected_requests
        )
        _persist_results(
            result_dir, detail=detail, telemetry=telemetry, proxy=proxy, scoring=scoring
        )
        recorder.record_state("persisted")

        validity["result_persistence_completed"] = True
        validity["evaluation_status"] = (
            "complete_valid" if validity["evaluation_valid"] else "partial_inconclusive"
        )
        runtime.atomic_write_json(result_dir / "evaluation_status.json", validity)

        # Permanent, committed-after-the-fact record of the single consumption.
        runtime.atomic_write_json(
            consumption_record_path,
            {
                "consumed": True,
                "run_id": run_label,
                "pipeline": PIPELINE_LABEL,
                "candidate": CANDIDATE_DESIGNATION,
                "candidate_pipeline": CANDIDATE_PIPELINE,
                "batch_size": batch_size,
                "fixture_hash": fixture_hash,
                "result_dir": result_dir.name,
                "consumption_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "evaluation_valid": validity["evaluation_valid"],
                "evaluation_status": validity["evaluation_status"],
                "note": (
                    "This is the permanent, committed record that Holdout v5 has "
                    "been consumed. The runtime consumption gate is the local, "
                    "gitignored CONSUMED.json marker, not this file."
                ),
            },
        )
        recorder.record_state("complete")
    except Exception as exc:
        recorder.record_state("partial_failed", error=str(exc))
        raise
    return result_dir


# ---------------------------------------------------------------------------
# CLI: live model identity resolution (no evaluation traffic)
# ---------------------------------------------------------------------------
#
# Everything below only ever talks to Ollama's local registry endpoint
# (``/api/tags`` -- a model-listing lookup, not a generation request) or reads
# back the configuration of a model object this module built itself. No Holdout
# v5 filename or fixture content is ever constructed or sent by any function
# below.

DEFAULT_RUN_LABEL = "dev"  # overridden by --run-label; pin the real SHA at freeze time


class LiveIdentityError(RuntimeError):
    """Raised when the live model identity cannot be established confidently.
    Callers must treat this as preflight FAIL / STOP unconsumed -- never
    substitute a guessed or expected identity in its place."""


def _ollama_tag(model_id: str) -> str:
    """Strip the litellm ``ollama_chat/``/``ollama/`` provider prefix."""
    for prefix in ("ollama_chat/", "ollama/"):
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return model_id


def fetch_ollama_tags(endpoint: str, *, timeout: float = 5.0) -> list[dict[str, Any]]:
    """Query the local Ollama registry's model list. Raises ``LiveIdentityError``
    on any connectivity or protocol problem; never raises for "model absent"."""
    url = endpoint.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LiveIdentityError(f"Ollama endpoint unreachable at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LiveIdentityError(f"Ollama endpoint returned invalid JSON at {url}: {exc}") from exc
    models = payload.get("models")
    if not isinstance(models, list):
        raise LiveIdentityError(f"Ollama endpoint returned an unexpected /api/tags shape at {url}")
    return models


def find_ollama_model_entry(tags: list[dict[str, Any]], tag: str) -> dict[str, Any] | None:
    for entry in tags:
        name = entry.get("model") or entry.get("name")
        if name == tag:
            return entry
    return None


def build_frozen_eval_model() -> Any:
    """Build the one permitted evaluation model via the production factory
    (``tidy.agent.build_model``), with the frozen evaluation configuration
    explicitly enforced. No alternate candidate/model path exists here."""
    return build_model(EXPECTED_MODEL_ID, think=EXPECTED_THINKING_ENABLED)


def resolve_actual_identity(model: Any) -> ModelIdentity:
    """Read back the exact configuration ``model`` was built with, and query
    the live Ollama registry for the installed model's digest and quantization.
    Raises ``LiveIdentityError`` on any gap rather than substituting an
    expected/guessed value. Gating here is reachability plus an exact
    digest/quantization/model_id match, not strict loopback: Holdout v5 never
    sends file content (only filenames, no peek tool), so the loopback-only
    rationale does not apply to this filename-only evaluation traffic. Endpoint
    locality is still surfaced by preflight as a non-gating diagnostic field.
    """
    model_id = getattr(model, "model_id", None)
    if not isinstance(model_id, str) or not model_id:
        raise LiveIdentityError("model object exposes no model_id")
    kwargs = getattr(model, "kwargs", None) or {}
    temperature = kwargs.get("temperature")
    thinking_enabled = kwargs.get("think")
    num_ctx = kwargs.get("num_ctx")
    if temperature is None or thinking_enabled is None or num_ctx is None:
        raise LiveIdentityError(
            "model object is missing an explicit temperature/think/num_ctx configuration"
        )
    _, endpoint = resolved_model_endpoint(model_id)
    if not endpoint:
        raise LiveIdentityError(f"no endpoint resolved for local model_id {model_id!r}")
    tag = _ollama_tag(model_id)
    tags = fetch_ollama_tags(endpoint)
    entry = find_ollama_model_entry(tags, tag)
    if entry is None:
        raise LiveIdentityError(f"model {tag!r} is not installed in the local Ollama registry at {endpoint}")
    digest = entry.get("digest")
    quantization = (entry.get("details") or {}).get("quantization_level")
    if not digest or not quantization:
        raise LiveIdentityError(f"Ollama registry entry for {tag!r} is missing digest/quantization_level")
    return ModelIdentity(
        model_id=model_id,
        digest=digest,
        quantization=quantization,
        temperature=float(temperature),
        thinking_enabled=bool(thinking_enabled),
        num_ctx=int(num_ctx),
    )


# ---------------------------------------------------------------------------
# CLI: preflight (non-consuming)
# ---------------------------------------------------------------------------


@dataclass
class PreflightResult:
    passed: bool
    checks: dict[str, bool]
    errors: dict[str, str]
    result_dir_path: Path


# Reported for transparency but never gates PASS/FAIL -- see
# resolve_actual_identity's docstring for why strict loopback is not required
# for Holdout v5's filename-only evaluation traffic.
_DIAGNOSTIC_ONLY_CHECKS = frozenset({"endpoint_is_local"})

_REQUIRED_ARTIFACTS = (
    "fixture",
    "ground_truth.json",
    "AUTHORING_FROZEN.json",
    "candidate_selection.json",
    "code_pins.json",
    "fixture_manifest.json",
)


def _recompute_fixture_hashes(fixture_dir: Path, ground_truth_path: Path) -> tuple[str, str]:
    names = [p.name for p in fixture_dir.iterdir() if p.is_file()]
    dataset_sha256 = _fixture_hash(names)
    ground_truth_sha256 = hashlib.sha256(ground_truth_path.read_bytes()).hexdigest()
    return dataset_sha256, ground_truth_sha256


def run_preflight(
    *,
    run_label: str = DEFAULT_RUN_LABEL,
    fixture_dir: Path = FIXTURE_DIR,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    code_pins_path: Path = CODE_PINS_PATH,
    candidate_selection_path: Path = CANDIDATE_SELECTION_PATH,
    consumed_path: Path = CONSUMED_PATH,
    results_root: Path = RESULTS_ROOT,
    repo_root: Path = REPO_ROOT,
    holdout_dir: Path = HOLDOUT_DIR,
    build_model_fn: Any = build_frozen_eval_model,
) -> PreflightResult:
    """Run every non-consuming pre-flight check independently (a failure in one
    check never skips the rest) and return an aggregate PASS/FAIL report. Never
    sends a Holdout v5 filename anywhere, never issues a classification request,
    never creates ``consumed_path`` or a measured result directory. Every
    required frozen artifact must exist and validate before PASS."""
    checks: dict[str, bool] = {}
    errors: dict[str, str] = {}

    checks["holdout_not_consumed"] = not consumed_path.exists()

    # Every required frozen artifact must exist and the fixture must be
    # non-empty; the runner refuses to run against a missing/placeholder
    # dataset. This is what makes "no fake cases" enforceable.
    missing = [name for name in _REQUIRED_ARTIFACTS if not (holdout_dir / name).exists()]
    fixture_files = (
        [p for p in fixture_dir.iterdir() if p.is_file()] if fixture_dir.is_dir() else []
    )
    checks["required_artifacts_present"] = not missing and len(fixture_files) > 0
    if missing:
        errors["required_artifacts_present"] = "missing: " + ", ".join(missing)
    elif not fixture_files:
        errors["required_artifacts_present"] = "fixture directory is empty"

    try:
        verify_candidate_selection(candidate_selection_path)
        checks["candidate_selection_valid"] = True
    except Exception as exc:
        checks["candidate_selection_valid"] = False
        errors["candidate_selection_valid"] = str(exc)

    try:
        dataset_sha256, ground_truth_sha256 = _recompute_fixture_hashes(fixture_dir, ground_truth_path)
        frozen = json.loads((holdout_dir / "AUTHORING_FROZEN.json").read_text(encoding="utf-8"))
        checks["fixture_and_ground_truth_hashes_match_frozen"] = (
            dataset_sha256 == frozen["dataset_sha256"]
            and ground_truth_sha256 == frozen["ground_truth_sha256"]
        )
    except Exception as exc:
        checks["fixture_and_ground_truth_hashes_match_frozen"] = False
        errors["fixture_and_ground_truth_hashes_match_frozen"] = str(exc)

    try:
        audit = blind_audit.run_blind_audit(repo_root)
        checks["blind_audit_exact_overlap_zero"] = audit["exact_historical_overlap_count"] == 0
        # Lexical near-similarity is diagnostic only and never gates (protocol).
    except Exception as exc:
        checks["blind_audit_exact_overlap_zero"] = False
        errors["blind_audit_exact_overlap_zero"] = str(exc)

    try:
        verify_code_pins(repo_root, code_pins_path)
        checks["code_pins_match"] = True
    except Exception as exc:
        checks["code_pins_match"] = False
        errors["code_pins_match"] = str(exc)

    expected = expected_model_identity()
    model = None
    try:
        model = build_model_fn()
        resolved_id, endpoint = resolved_model_endpoint(EXPECTED_MODEL_ID)
        checks["configured_model_id_matches"] = resolved_id == expected.model_id
    except Exception as exc:
        checks["configured_model_id_matches"] = False
        errors["configured_model_id_matches"] = str(exc)
        endpoint = None

    checks["endpoint_is_local"] = bool(model is not None and endpoint_is_local(EXPECTED_MODEL_ID))

    tags: list[dict[str, Any]] | None = None
    if endpoint:
        try:
            tags = fetch_ollama_tags(endpoint)
            checks["local_ollama_endpoint_reachable"] = True
        except LiveIdentityError as exc:
            checks["local_ollama_endpoint_reachable"] = False
            errors["local_ollama_endpoint_reachable"] = str(exc)
    else:
        checks["local_ollama_endpoint_reachable"] = False

    entry = find_ollama_model_entry(tags, _ollama_tag(EXPECTED_MODEL_ID)) if tags is not None else None
    checks["expected_model_installed_locally"] = entry is not None
    digest = entry.get("digest") if entry else None
    quantization = (entry.get("details") or {}).get("quantization_level") if entry else None
    checks["actual_digest_matches"] = digest == expected.digest
    checks["actual_quantization_matches"] = quantization == expected.quantization

    kwargs = getattr(model, "kwargs", None) or {} if model is not None else {}
    temperature = kwargs.get("temperature")
    thinking_enabled = kwargs.get("think")
    num_ctx = kwargs.get("num_ctx")
    checks["temperature_matches"] = (
        temperature is not None and float(temperature) == float(expected.temperature)
    )
    checks["thinking_disabled_matches"] = (
        thinking_enabled is not None and bool(thinking_enabled) == expected.thinking_enabled
    )
    checks["num_ctx_matches"] = num_ctx is not None and int(num_ctx) == expected.num_ctx

    date_str = time.strftime("%Y%m%d")
    result_dir_path = results_root / f"holdout-v5-e4-{date_str}-{run_label}"
    checks["result_directory_available"] = not result_dir_path.exists()

    passed = all(v for k, v in checks.items() if k not in _DIAGNOSTIC_ONLY_CHECKS)
    return PreflightResult(passed=passed, checks=checks, errors=errors, result_dir_path=result_dir_path)


def format_preflight_report(result: PreflightResult) -> str:
    lines = ["Holdout v5 preflight report", "=" * 32]
    for name, value in result.checks.items():
        status = "PASS" if value else "FAIL"
        suffix = " (diagnostic only, does not gate)" if name in _DIAGNOSTIC_ONLY_CHECKS else ""
        lines.append(f"[{status}] {name}{suffix}")
        if not value and name in result.errors:
            lines.append(f"       {result.errors[name]}")
    lines.append("")
    lines.append(f"planned result directory: {result.result_dir_path}")
    lines.append("")
    lines.append("OVERALL: " + ("PASS" if result.passed else "FAIL"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: argparse entry point
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run_holdout_v5_e4",
        description=(
            "Pre-consumption CLI for the single permitted live Holdout v5 "
            "evaluation (E4-batched). Supported invocation is "
            "`python -m evals.run_holdout_v5_e4`; direct-path execution is not "
            "supported."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Run only non-consuming checks and print a PASS/FAIL report. Does not consume Holdout v5.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Repeat preflight, then run the single measured Holdout v5 evaluation. Consumes Holdout v5.",
    )
    parser.add_argument(
        "--run-label",
        default=DEFAULT_RUN_LABEL,
        help=f"Result-directory run label (default: {DEFAULT_RUN_LABEL!r}).",
    )
    return parser


def _cmd_run(run_label: str) -> int:
    preflight = run_preflight(run_label=run_label)
    print(format_preflight_report(preflight))
    if not preflight.passed:
        print(
            "\nPreflight failed; STOPPING before consumption. Holdout v5 remains unconsumed.",
            file=sys.stderr,
        )
        return 1
    model = build_frozen_eval_model()
    actual_identity = resolve_actual_identity(model)
    expected_identity = expected_model_identity()
    print(f"\nResult directory: {preflight.result_dir_path}")
    result_dir = run_holdout_v5(
        model,
        run_label=run_label,
        expected_identity=expected_identity,
        actual_identity=actual_identity,
    )
    print(f"Run complete. Results at: {result_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.preflight and not args.run:
        parser.print_help()
        return 0
    if args.preflight:
        result = run_preflight(run_label=args.run_label)
        print(format_preflight_report(result))
        return 0 if result.passed else 1
    return _cmd_run(args.run_label)


if __name__ == "__main__":
    raise SystemExit(main())





