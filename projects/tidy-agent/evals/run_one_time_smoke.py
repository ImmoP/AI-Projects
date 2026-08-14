"""Synthetic, rerunnable smoke test of the real one-time-eval lifecycle.

This is infrastructure hardening, not an evaluation. It exercises the exact
stages a future one-time Holdout runner needs to survive process
interruption -- lifecycle marker, request-start journal, response journal,
atomic persistence, completion state -- using the synthetic, non-evidentiary
120-file fixture in ``evals/synthetic_one_time_smoke/fixture``. It never
touches ``evals/holdout_v3/`` or any other Holdout's consumption state, and
it may be rerun freely.

The classification path is real and unmocked in live trials: the frozen
``evals.post_holdout_candidates.run_e4`` (E3's two passes, unchanged, plus
the frozen E4-current deterministic veto) is called exactly as-is. Only the
``model`` object passed into it is wrapped in
``evals.one_time_eval_runtime.JournalingModelProxy`` to journal each
dispatch/response -- no change to any frozen candidate or production file.
"""

from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals import one_time_eval_runtime as runtime
from evals.post_holdout_candidates import run_e4

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = Path(__file__).resolve().parent / "synthetic_one_time_smoke"
SYNTHETIC_FIXTURE_DIR = SYNTHETIC_DIR / "fixture"
SYNTHETIC_MANIFEST_PATH = SYNTHETIC_DIR / "fixture_manifest.json"
RESULTS_ROOT = REPO_ROOT / "evals" / "results"

REVIEW_DIRECTORY = "_ToReview"
REAL_CATEGORIES = ("Documents", "Code", "Images", "Archives", "Installers")
PIPELINE_LABEL = "E3+E4-current(synthetic-smoke)"

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


def _fixture_hash(names: Sequence[str]) -> str:
    listing = "\n".join(sorted(unicodedata.normalize("NFC", n) for n in names)).encode("utf-8")
    return hashlib.sha256(listing).hexdigest()


def load_synthetic_metadata(fixture_dir: Path = SYNTHETIC_FIXTURE_DIR) -> list[dict[str, str]]:
    """Production-approved metadata only: every dict carries exactly ``name``."""
    names = sorted(p.name for p in fixture_dir.iterdir() if p.is_file())
    return [{"name": name} for name in names]


def load_intended_labels(manifest_path: Path = SYNTHETIC_MANIFEST_PATH) -> dict[str, str]:
    """Sanity-diagnostic-only labels. Never sent to the model, never used to
    tune E3/E4, and not scientific evidence of anything."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {c["filename"]: c["intended_category"] for c in manifest["cases"]}


def _sanity_diagnostics(
    sources: Sequence[str],
    final: Mapping[str, str],
    intended: Mapping[str, str],
) -> dict[str, Any]:
    """Infrastructure/schema-validation diagnostics only -- see module and
    README docstrings: never used for candidate selection or tuning."""
    auto = sum(1 for s in sources if final.get(s) != REVIEW_DIRECTORY)
    review = sum(1 for s in sources if final.get(s) == REVIEW_DIRECTORY)
    matches_intended = sum(1 for s in sources if final.get(s) == intended.get(s))
    returned = set(final)
    expected_set = set(sources)
    return {
        "note": (
            "Synthetic classification metrics are infrastructure diagnostics "
            "only and provide no evidence of real-world accuracy or "
            "generalization."
        ),
        "expected_source_count": len(sources),
        "returned_unique_sources": len(returned),
        "missing_sources": sorted(expected_set - returned),
        "invented_sources": sorted(returned - expected_set),
        "duplicate_sources": len(sources) - len(set(sources)),
        "auto": auto,
        "review": review,
        "matches_intended_label_count": matches_intended,
    }


def _validation_diagnostics(detail: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    invalid_categories = [
        item["filename"]
        for item in detail
        if item["final"] != REVIEW_DIRECTORY and item["final"] not in REAL_CATEGORIES
    ]
    return {
        "invalid_categories": invalid_categories,
        "invalid_category_count": len(invalid_categories),
    }


class SmokeTrialFailed(RuntimeError):
    pass


def run_smoke_trial(
    model: Any,
    *,
    run_label: str,
    expected_identity: ModelIdentity,
    actual_identity: ModelIdentity,
    fixture_dir: Path = SYNTHETIC_FIXTURE_DIR,
    manifest_path: Path = SYNTHETIC_MANIFEST_PATH,
    results_root: Path = RESULTS_ROOT,
) -> Path:
    """Run exactly one synthetic smoke trial. Rerunnable; never touches any
    Holdout's consumption state. Mirrors the lifecycle a future one-time
    Holdout runner must survive, using the real, unmocked E3+E4-current path."""
    runtime.verify_model_identity(expected_identity, actual_identity)

    date_str = time.strftime("%Y%m%d")
    result_dir = runtime.create_fresh_result_directory(
        results_root, f"synthetic-one-time-smoke-{date_str}-{run_label}"
    )
    recorder = runtime.LifecycleRecorder(result_dir)

    metadata = load_synthetic_metadata(fixture_dir)
    sources = [m["name"] for m in metadata]
    fixture_hash = _fixture_hash(sources)

    recorder.record_prepared(
        run_id=run_label,
        pipeline=PIPELINE_LABEL,
        model_identity_expected=expected_identity.as_dict(),
        fixture_hash=fixture_hash,
    )

    fixture_snapshot_before = sorted(p.name for p in fixture_dir.iterdir() if p.is_file())

    try:
        proxy = runtime.JournalingModelProxy(model, recorder, phase_labels=("pass1", "pass2"))
        final, detail, telemetry = run_e4(
            proxy, metadata, list(REAL_CATEGORIES), review_directory=REVIEW_DIRECTORY
        )

        fixture_snapshot_after = sorted(p.name for p in fixture_dir.iterdir() if p.is_file())
        if fixture_snapshot_after != fixture_snapshot_before:
            raise SmokeTrialFailed("synthetic fixture directory was mutated during the trial")

        recorder.record_state("scoring_complete")

        intended = load_intended_labels(manifest_path)
        sanity = _sanity_diagnostics(sources, final, intended)
        validation = _validation_diagnostics(detail)

        _persist_results(
            result_dir,
            detail=detail,
            telemetry=telemetry,
            sanity=sanity,
            validation=validation,
            proxy=proxy,
        )
        recorder.record_state("persisted")
        recorder.record_state("complete")
    except Exception as exc:
        recorder.record_state("partial_failed", error=str(exc))
        raise
    return result_dir


def _persist_results(
    result_dir: Path,
    *,
    detail: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
    sanity: Mapping[str, Any],
    validation: Mapping[str, Any],
    proxy: "runtime.JournalingModelProxy",
) -> None:
    raw_run_lines = "".join(
        json.dumps({"filename": item["filename"], "final": item["final"]}) + "\n" for item in detail
    )
    runtime.atomic_write_text(result_dir / "raw_run.jsonl", raw_run_lines)
    runtime.atomic_write_json(result_dir / "per_file_evidence.json", list(detail))

    summary = {
        "sanity_diagnostics": dict(sanity),
        "validation_diagnostics": dict(validation),
        "telemetry": dict(telemetry),
        "dispatch": {
            "attempted": proxy.dispatch_attempted,
            "returned": proxy.dispatch_returned,
            "failed": proxy.dispatch_failed,
        },
    }
    runtime.atomic_write_json(result_dir / "summary.json", summary)

    report_lines = [
        "# Synthetic one-time-eval smoke trial",
        "",
        "> Synthetic classification metrics are infrastructure diagnostics "
        "only and provide no evidence of real-world accuracy or "
        "generalization.",
        "",
        "## Dispatch",
        "```json",
        json.dumps(summary["dispatch"], indent=2),
        "```",
        "## Sanity diagnostics",
        "```json",
        json.dumps(sanity, indent=2),
        "```",
    ]
    runtime.atomic_write_text(result_dir / "report.md", "\n".join(report_lines) + "\n")
