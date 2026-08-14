"""The single permitted live Holdout v4 evaluation runner. Pipeline: E3 -> E4-current only.

Frozen under ``evals/final_portfolio_holdout_protocol.md``
(``final_protocol_version = 1``) and ``evals/holdout_v4/candidate_selection.json``
(``selection_closed = true``). There is no candidate parameter and no
Development mode: this module runs exactly one frozen pipeline against
exactly one frozen fixture. The frozen ``evals.post_holdout_candidates.run_e4``
(E3's two passes, unchanged, plus the frozen E4-current deterministic veto)
is called exactly as-is; only the ``model`` object is wrapped in
``evals.one_time_eval_runtime.JournalingModelProxy`` to journal each
dispatch/response, mirroring ``evals/run_one_time_smoke.py``.

Before any measured provider request this module verifies, in order: the
live model identity against the frozen pin, then the sha256 of every
pinned source file against ``evals/holdout_v4/code_pins.json``. Either
mismatch raises before any dispatch and before ``CONSUMED.json`` is
written, leaving Holdout v4 unconsumed. ``CONSUMED.json`` is written
durably, atomically, immediately before the first measured provider
request; after that write Holdout v4 is permanently consumed regardless of
what happens later in the run -- there is no rerun path.

A run is ``complete_valid`` only if there were zero provider errors, zero
parse failures, zero schema-validation failures, every expected provider
response was received, the E3 agreement gate resolved every source, E4's
veto resolved every source, and results were durably persisted. The
production classification backend's fallback-to-review-on-provider-failure
behaviour is untouched and out of scope here; for this Holdout, any
provider/parse/schema/infrastructure failure after consumption is reported
as ``partial_inconclusive`` -- a protocol/infrastructure failure, not model
abstention -- and is not eligible for rerun.

CLI (supported invocation only -- run as a module, from the repository
root, so ``evals`` resolves as a package)::

    python -m evals.run_holdout_v4_e4 --preflight
    python -m evals.run_holdout_v4_e4 --run [--run-label LABEL]

Direct-path execution (``python evals/run_holdout_v4_e4.py``) is not
supported: it fails at the top-level ``from evals import ...`` before any
Holdout code runs, and no path-manipulation workaround is added here.
``--preflight`` and ``--run`` are mutually exclusive; running with neither
flag prints help and exits without touching any Holdout v4 state.
``--preflight`` performs only non-consuming checks (fixture/ground-truth
hashes, code pins, live Ollama model identity, result-directory
availability) and never sends a Holdout filename to any model. ``--run``
repeats every preflight check and then delegates, unmodified, to
``run_holdout_v4`` above.
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

from evals import one_time_eval_runtime as runtime
from evals.post_holdout_candidates import run_e4

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_DIR = Path(__file__).resolve().parent / "holdout_v4"
FIXTURE_DIR = HOLDOUT_DIR / "fixture"
GROUND_TRUTH_PATH = HOLDOUT_DIR / "ground_truth.json"
CODE_PINS_PATH = HOLDOUT_DIR / "code_pins.json"
CONSUMED_PATH = HOLDOUT_DIR / "CONSUMED.json"
RESULTS_ROOT = REPO_ROOT / "evals" / "results"

REVIEW_DIRECTORY = "_ToReview"
REAL_CATEGORIES = ("Documents", "Code", "Images", "Archives", "Installers")
PIPELINE_LABEL = "E3+E4-current(holdout-v4)"

EXPECTED_MODEL_ID = "ollama_chat/qwen3.5:4b"
EXPECTED_MODEL_DIGEST = "2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd"
EXPECTED_QUANTIZATION = "Q4_K_M"
EXPECTED_TEMPERATURE = 0
EXPECTED_THINKING_ENABLED = False
EXPECTED_NUM_CTX = 8192
EXPECTED_PROVIDER_REQUEST_COUNT = 2  # E3's fixed two-pass protocol: pass1, pass2

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


class HoldoutV4RunFailed(RuntimeError):
    pass


_VALID_GATE_AGREEMENTS = frozenset(
    {"agree_classify", "disagree_classify", "review_involved", "both_invalid"}
)


def _evaluate_validity(
    sources: Sequence[str],
    final: Mapping[str, str],
    detail: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
    proxy: "runtime.JournalingModelProxy",
) -> dict[str, Any]:
    provider_errors = int(telemetry.get("provider_errors", 0))
    parse_failures = int(telemetry.get("parse_failures", 0))
    schema_failures = int(telemetry.get("schema_validation_failures", 0))
    all_required_provider_responses_received = (
        proxy.dispatch_attempted == EXPECTED_PROVIDER_REQUEST_COUNT
        and proxy.dispatch_returned == EXPECTED_PROVIDER_REQUEST_COUNT
        and proxy.dispatch_failed == 0
    )
    # evals.run_structured_calibration.run_e3 (the Development harness E4
    # builds on) calls merge_agreement_gate directly and never routes through
    # StructuredClassifier.classify_with_agreement_gate, so the production
    # gate_final_*/pass*_* telemetry counters in tidy.classification stay at
    # zero here by construction -- they are not a signal in this harness.
    # Gate completion is instead read from ``detail`` (run_e4's own per-source
    # record of Python's deterministic merge_agreement_gate outcome, plus its
    # local veto), which covers every source unconditionally by construction
    # of merge_agreement_gate itself; a short count or an unrecognized
    # agreement label would mean a source was silently dropped upstream.
    e3_gate_completed = len(detail) == len(sources) and all(
        d.get("e3_agreement") in _VALID_GATE_AGREEMENTS for d in detail
    )
    e4_current_completed = set(final) == set(sources) and all(
        d.get("final") is not None for d in detail
    )
    evaluation_valid = (
        provider_errors == 0
        and parse_failures == 0
        and schema_failures == 0
        and all_required_provider_responses_received
        and e3_gate_completed
        and e4_current_completed
    )
    return {
        "provider_errors": provider_errors,
        "parse_failures": parse_failures,
        "schema_failures": schema_failures,
        "all_required_provider_responses_received": all_required_provider_responses_received,
        "E3_gate_completed": e3_gate_completed,
        "E4_current_completed": e4_current_completed,
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
        json.dumps({"filename": item["filename"], "final": item["final"]}) + "\n" for item in detail
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


def run_holdout_v4(
    model: Any,
    *,
    run_label: str,
    expected_identity: ModelIdentity,
    actual_identity: ModelIdentity,
    fixture_dir: Path = FIXTURE_DIR,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    code_pins_path: Path = CODE_PINS_PATH,
    consumed_path: Path = CONSUMED_PATH,
    results_root: Path = RESULTS_ROOT,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Run the single permitted live Holdout v4 evaluation.

    STOPs unconsumed -- no ``consumed_path`` written, no result directory
    left behind -- on model-identity mismatch, code-pin mismatch, or a
    result-directory collision; all three checks happen before
    ``create_fresh_result_directory`` and before ``consumed_path`` is
    written. Every other failure after that point still leaves Holdout v4
    permanently consumed (see module docstring).
    """
    runtime.verify_model_identity(expected_identity, actual_identity)
    verify_code_pins(repo_root, code_pins_path)

    date_str = time.strftime("%Y%m%d")
    result_dir = runtime.create_fresh_result_directory(
        results_root, f"holdout-v4-e4-{date_str}-{run_label}"
    )
    recorder = runtime.LifecycleRecorder(result_dir)

    metadata = load_holdout_metadata(fixture_dir)
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
        # Consumption boundary: durable, atomic, and the last thing written
        # before the first measured provider request. Everything above this
        # line may still leave Holdout v4 unconsumed on failure; nothing
        # below it may.
        runtime.atomic_write_json(
            consumed_path,
            {
                "consumed": True,
                "run_id": run_label,
                "pipeline": PIPELINE_LABEL,
                "fixture_hash": fixture_hash,
                "result_dir": result_dir.name,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

        proxy = runtime.JournalingModelProxy(model, recorder, phase_labels=("pass1", "pass2"))
        final, detail, telemetry = run_e4(
            proxy, metadata, list(REAL_CATEGORIES), review_directory=REVIEW_DIRECTORY
        )

        fixture_snapshot_after = sorted(p.name for p in fixture_dir.iterdir() if p.is_file())
        if fixture_snapshot_after != fixture_snapshot_before:
            raise HoldoutV4RunFailed("holdout_v4 fixture directory was mutated during the run")

        recorder.record_state("scoring_complete")

        ground_truth = load_ground_truth(ground_truth_path)
        matches_ground_truth = sum(1 for s in sources if final.get(s) == ground_truth.get(s))
        scoring = {
            "expected_source_count": len(sources),
            "auto_count": sum(1 for s in sources if final.get(s) != REVIEW_DIRECTORY),
            "review_count": sum(1 for s in sources if final.get(s) == REVIEW_DIRECTORY),
            "matches_ground_truth_count": matches_ground_truth,
        }

        validity = _evaluate_validity(sources, final, detail, telemetry, proxy)

        _persist_results(
            result_dir, detail=detail, telemetry=telemetry, proxy=proxy, scoring=scoring
        )
        recorder.record_state("persisted")

        validity["result_persistence_completed"] = True
        validity["evaluation_status"] = (
            "complete_valid" if validity["evaluation_valid"] else "partial_inconclusive"
        )
        runtime.atomic_write_json(result_dir / "evaluation_status.json", validity)

        # The frozen ClassificationBackend.request() (src/tidy/classification.py)
        # already catches every provider exception internally and degrades
        # that pass to a fallback rather than re-raising (see
        # evals/run_one_time_smoke.py and tests/test_run_one_time_smoke.py's
        # "degrades gracefully through real E3" tests for the discovery).
        # The lifecycle state therefore reflects whether the run process
        # itself completed end-to-end -- it always does, fail-safe, once
        # persistence lands -- never the scientific validity of the result;
        # only ``evaluation_status.json`` carries that (item 31/32).
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
# (``/api/tags`` -- a model-listing lookup, not a generation request) or
# reads back the configuration of a model object this module built itself.
# No Holdout v4 filename or fixture content is ever constructed or sent by
# any function below.

DEFAULT_RUN_LABEL = "39423af"  # short SHA of the frozen preflight-fix commit


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
    the live Ollama registry (wherever ``.env``/``API_BASE`` points it) for
    the installed model's digest and quantization. Raises
    ``LiveIdentityError`` on any gap rather than substituting an
    expected/guessed value -- callers must not fall back to
    ``expected_model_identity()`` on failure.

    Gating here is reachability plus an exact digest/quantization/model_id
    match, not strict loopback: Holdout v4 never sends file content (only
    filenames, no peek tool -- see ``classify_with_agreement_gate``), so the
    loopback-only rationale ``tidy.agent.endpoint_is_local`` exists for
    (whether to authorize sending file *excerpts* to a model) does not apply
    to this evaluation traffic. Whether the resolved endpoint is a strict
    loopback address is still surfaced by preflight as a non-gating
    diagnostic field.
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
# resolve_actual_identity's docstring for why strict loopback is not
# required for Holdout v4's filename-only evaluation traffic.
_DIAGNOSTIC_ONLY_CHECKS = frozenset({"endpoint_is_local"})


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
    consumed_path: Path = CONSUMED_PATH,
    results_root: Path = RESULTS_ROOT,
    repo_root: Path = REPO_ROOT,
    holdout_dir: Path = HOLDOUT_DIR,
    build_model_fn: Any = build_frozen_eval_model,
) -> PreflightResult:
    """Run every non-consuming pre-flight check independently (a failure in
    one check never skips the rest) and return an aggregate PASS/FAIL
    report. Never sends a Holdout v4 filename anywhere, never issues a
    classification request, never creates ``consumed_path`` or a measured
    result directory."""
    checks: dict[str, bool] = {}
    errors: dict[str, str] = {}

    checks["holdout_not_consumed"] = not consumed_path.exists()

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

    # Diagnostic only (see resolve_actual_identity's docstring): Holdout v4
    # sends only filenames, never file content, so strict loopback is not
    # required for this evaluation traffic. Reachability plus an exact
    # digest/quantization/model_id match (below) is the real gate.
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
    result_dir_path = results_root / f"holdout-v4-e4-{date_str}-{run_label}"
    checks["result_directory_available"] = not result_dir_path.exists()

    passed = all(v for k, v in checks.items() if k not in _DIAGNOSTIC_ONLY_CHECKS)
    return PreflightResult(passed=passed, checks=checks, errors=errors, result_dir_path=result_dir_path)


def format_preflight_report(result: PreflightResult) -> str:
    lines = ["Holdout v4 preflight report", "=" * 32]
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
        prog="python -m evals.run_holdout_v4_e4",
        description=(
            "Pre-consumption CLI for the single permitted live Holdout v4 "
            "evaluation (frozen E3 -> E4-current). Supported invocation is "
            "`python -m evals.run_holdout_v4_e4`; direct-path execution is "
            "not supported."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Run only non-consuming checks and print a PASS/FAIL report. Does not consume Holdout v4.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Repeat preflight, then run the single measured Holdout v4 evaluation. Consumes Holdout v4.",
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
            "\nPreflight failed; STOPPING before consumption. Holdout v4 remains unconsumed.",
            file=sys.stderr,
        )
        return 1
    model = build_frozen_eval_model()
    actual_identity = resolve_actual_identity(model)
    expected_identity = expected_model_identity()
    print(f"\nResult directory: {preflight.result_dir_path}")
    result_dir = run_holdout_v4(
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
