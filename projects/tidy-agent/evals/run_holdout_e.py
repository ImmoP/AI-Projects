"""Run the frozen, selected E configuration exactly once against the locked holdout.

This is not a new evaluation design. It reuses the same ``run_evaluation`` core
used by the frozen A/E/C development diagnostic (``run_structured_aec.py``)
with condition E's exact parameters (``read_contents=False``,
``metadata_control=True``, no remote content authorization), run a single time
against ``evals/holdout`` instead of ``evals/dev``. A and C are intentionally
absent: the holdout is not a venue for re-deciding between conditions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from tidy.agent import DEFAULT_MODEL_ID  # noqa: E402
from tidy.content_parser import (  # noqa: E402
    MAX_DOCX_BYTES,
    MAX_PDF_BYTES,
    PARSER_TIMEOUT_SECONDS,
)
from tidy.rules import load_rules  # noqa: E402
from tidy.tools import (  # noqa: E402
    MAX_PEEK_BYTES,
    MAX_PEEK_CHARS,
    MAX_TASK_PEEKS,
    MAX_TEXT_FILE_BYTES,
)

from evals.run_evals import (  # noqa: E402
    REVIEW_DIRECTORY,
    UNSCORED_PREDICTIONS,
    _warmup_worker,
    run_evaluation,
    run_in_subprocess,
)
from evals.run_structured_abcd import (  # noqa: E402
    _git,
    _installed_versions,
    _model_manifest,
    _sha256,
    _stage_fixture,
    _verify_dataset_manifest,
)

E_CONDITION = {
    "label": "Two-pass metadata control",
    "content": False,
    "metadata_control": True,
}
EXPECTED_MODEL_IDENTIFIER = "ollama_chat/qwen3.5:4b"
EXPECTED_MODEL_DIGEST = (
    "2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd"
)


def _relative_to_repo(path: Path, toplevel: Path) -> str:
    return str(path.resolve().relative_to(toplevel))


def _git_source_allow_prior_results() -> dict[str, Any]:
    """Same freeze check as the development scripts, with two documented carve-outs.

    1. ``evals/results/`` accumulates prior, already-written, uncommitted
       experiment output between sessions (this project intentionally tracks
       those reports in Git, but nothing here is permitted to run ``git add``
       or ``git commit``). An untracked prior report in that directory is
       measurement history, not implementation drift.
    2. This script (``evals/run_holdout_e.py``, plus its bytecode cache) is
       itself new: no single-condition, single-execution holdout runner
       existed before this task, and the requested artifact set
       (holdout_manifest.json / raw_run.jsonl / error_analysis.json) does not
       match any prior script's output shape. It is a thin orchestration
       wrapper around the already-frozen ``run_evaluation()`` call used
       verbatim (same parameters) by condition E inside
       ``run_structured_aec.py``; it adds no new prompt, schema, rule,
       decision threshold, or classification logic. Its own sha256 is
       recorded below for audit, and every *other* tracked file must still be
       byte-identical to the frozen commit — a modification to any existing
       file (src/, config/, evals/run_evals.py, evals/run_structured_*.py,
       fixtures, ground truth) still blocks the run.
    """
    toplevel = Path(_git("rev-parse", "--show-toplevel"))
    project_prefix = _relative_to_repo(PROJECT_ROOT, toplevel)
    results_prefix = f"{project_prefix}/evals/results/"
    harness_path = f"{project_prefix}/evals/run_holdout_e.py"
    harness_cache_prefix = f"{project_prefix}/evals/__pycache__/run_holdout_e."

    tracked_modifications = _git("diff", "--name-only", "HEAD")
    if tracked_modifications:
        raise RuntimeError(
            "holdout requires zero modifications to any tracked file since "
            f"the frozen commit; found: {tracked_modifications.splitlines()}"
        )

    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    offending = [
        line
        for line in status.splitlines()
        if not line[3:].startswith(results_prefix)
        and line[3:] != harness_path
        and not line[3:].startswith(harness_cache_prefix)
    ]
    if offending:
        raise RuntimeError(
            "holdout requires a clean Git worktree outside evals/results/ and "
            f"this new harness script; found: {offending}"
        )
    commit = _git("rev-parse", "HEAD")
    script = Path(__file__).resolve()
    return {
        "commit": commit,
        "short_commit": commit[:12],
        "dirty": False,
        "worktree_exception": (
            "untracked prior reports under evals/results/ and the new "
            "evals/run_holdout_e.py harness itself permitted; zero "
            "modifications to any other tracked file, and zero other "
            "untracked files, required"
        ),
        "branch": _git("branch", "--show-current"),
        "evaluation_script": "evals/run_holdout_e.py",
        "evaluation_script_sha256": _sha256(script),
        "evaluation_script_is_new_harness": (
            "no single-condition holdout runner existed before this task; "
            "wraps run_evaluation() with condition E's exact frozen "
            "parameters, unchanged from run_structured_aec.py"
        ),
    }


def _reject_existing_holdout_predictions(output_root: Path) -> None:
    if not output_root.is_dir():
        return
    existing = sorted(
        p.name
        for p in output_root.iterdir()
        if p.is_dir() and "holdout" in p.name.lower()
    )
    if existing:
        raise RuntimeError(
            "refusing to run: prior holdout prediction directories already "
            f"exist under {output_root}: {existing}"
        )


def _wilson_interval(successes: int, total: int, *, z: float = 1.959963985) -> tuple[float, float]:
    """95% Wilson score interval; safe at small n unlike a naive normal interval."""
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _outcome_label(case: dict[str, Any]) -> str:
    predicted = case["predicted"]
    if predicted in UNSCORED_PREDICTIONS:
        return f"run_incomplete_{predicted.lower()}"
    if case["mode"] == "rule":
        return "correct_rule" if case["correct"] else "rule_mismatch"
    if predicted == REVIEW_DIRECTORY:
        fallback = case.get("fallback", "")
        if fallback == "omitted":
            return "review_fallback_omitted"
        if fallback == "invalid":
            return "review_fallback_invalid"
        if fallback == "no-proposal":
            return "review_fallback_no_proposal"
        return "review_intentional"
    return "correct_automatic" if case["correct"] else "incorrect_automatic"


def _development_reference(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    condition_e = summary["conditions"]["E"]
    metrics = condition_e["metrics"]
    try:
        source_report = str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        source_report = path.name
    return {
        "source_report": source_report,
        "run_count": condition_e["runs"],
        "strict_category_accuracy": metrics["strict_category_accuracy"]["mean"],
        "decision_rate": metrics["decision_rate"]["mean"],
        "accuracy_on_decided": metrics["accuracy_on_decided"]["mean"],
        "review_rate": metrics["review_rate"]["mean"],
        "incorrect_decision_rate": metrics["incorrect_decision_rate"]["mean"],
        "total_run_latency_seconds": metrics["total_run_latency_seconds"]["mean"],
        "total_tokens": metrics["total_tokens"]["mean"],
    }


def _mechanism_and_reliability(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cases = metrics.get("cases", [])
    rule_cases = [c for c in cases if c["mode"] == "rule"]
    agent_cases = [c for c in cases if c["mode"] == "agent"]
    review_cases = [c for c in agent_cases if c["predicted"] == REVIEW_DIRECTORY]
    decided_cases = [
        c
        for c in agent_cases
        if c["predicted"] != REVIEW_DIRECTORY and c["predicted"] not in UNSCORED_PREDICTIONS
    ]
    incorrect_decided = [c for c in decided_cases if not c["correct"]]
    correct_decided = [c for c in decided_cases if c["correct"]]

    mechanism = {
        "total_files": len(cases),
        "deterministic_rule_resolved": len(rule_cases),
        "reached_model_classification": len(agent_cases),
        "ended_in_review": len(review_cases),
        "decided_by_model": len(decided_cases),
        "correct_automatic": len(correct_decided),
        "incorrect_automatic": len(incorrect_decided),
    }

    reliability_fields = (
        "classification_requests",
        "native_schema_responses",
        "json_object_responses",
        "plain_json_responses",
        "parse_failures",
        "schema_validation_failures",
        "provider_errors",
        "incomplete_responses",
        "duplicate_source_responses",
        "invented_source_responses",
        "invented_category_responses",
        "fallback_to_review_count",
    )
    reliability = {
        field: int(metrics.get(field, 0) or 0) for field in reliability_fields
    }
    return mechanism, reliability


def _security_assertions(metrics: dict[str, Any]) -> dict[str, Any]:
    checked_zero_fields = (
        "peek_calls",
        "peek_readable",
        "peek_nonempty",
        "peek_bytes_read",
        "peek_chars_returned",
        "peek_requests_authorized",
        "content_unavailable",
    )
    values = {field: int(metrics.get(field, 0) or 0) for field in checked_zero_fields}
    violations = {field: value for field, value in values.items() if value != 0}
    if violations:
        raise RuntimeError(f"metadata control attempted content access: {violations}")
    return {
        "zero_content_peeks": True,
        "zero_file_content_reads": True,
        "zero_parser_invocations": True,
        "checked_fields": values,
        "read_contents_flag": bool(metrics.get("read_contents", False)),
        "metadata_control_flag": bool(metrics.get("metadata_control", False)),
    }


def _experiment_manifest(
    *,
    source: dict[str, Any],
    model: dict[str, Any],
    holdout_manifest: dict[str, Any],
    development_reference: dict[str, Any],
    timeout: float,
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
        "expected_model": {
            "identifier": EXPECTED_MODEL_IDENTIFIER,
            "digest": EXPECTED_MODEL_DIGEST,
        },
        "tidy_configuration": {
            "rules": [
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": _sha256(path),
                }
                for path in rule_paths
            ],
            "categories": list(rules.categories),
            "review_directory": rules.review_directory,
            "condition": E_CONDITION,
            "remote_content_authorized": False,
            "max_task_peeks": MAX_TASK_PEEKS,
            "max_peek_chars": MAX_PEEK_CHARS,
            "max_plain_text_read_bytes": MAX_PEEK_BYTES,
            "max_text_file_bytes": MAX_TEXT_FILE_BYTES,
            "max_pdf_bytes": MAX_PDF_BYTES,
            "max_docx_bytes": MAX_DOCX_BYTES,
            "parser_timeout_seconds": PARSER_TIMEOUT_SECONDS,
        },
        "dataset": {
            "designation": "locked holdout — single evaluation",
            "holdout_manifest": "evals/holdout/fixture_manifest.json",
            "holdout_dataset_sha256": holdout_manifest["dataset_sha256"],
            "holdout_file_count": len(holdout_manifest["files"]),
            "ground_truth_sha256": holdout_manifest["ground_truth"]["sha256"],
        },
        "development_reference": development_reference,
        "execution": {
            "repetitions": 1,
            "scheduled_runs": 1,
            "model_lifecycle": "warm",
            "warmup": "one discarded model request before the measured run",
            "standard_timeout_seconds": timeout,
            "failed_run_policy": (
                "record failure and stop; never substitute model; never rerun"
            ),
            "selection_policy": (
                "E was preselected from development results before this run; "
                "A and C are not executed here and this run cannot change that "
                "selection"
            ),
        },
    }


def run_holdout(args: argparse.Namespace) -> Path:
    source = _git_source_allow_prior_results()
    fixture = args.fixture.resolve(strict=True)
    expected_path = args.expected.resolve(strict=True)
    holdout_manifest = _verify_dataset_manifest(
        args.fixture_manifest.resolve(strict=True), fixture, expected_path
    )

    output_root = args.output_root
    _reject_existing_holdout_predictions(output_root)

    development_reference = _development_reference(
        args.development_summary.resolve(strict=True)
    )

    model = _model_manifest(args.model, args.think)
    if model["identifier"] != EXPECTED_MODEL_IDENTIFIER:
        raise RuntimeError(
            "model identity changed: expected "
            f"{EXPECTED_MODEL_IDENTIFIER!r}, got {model['identifier']!r}"
        )
    if model.get("digest") != EXPECTED_MODEL_DIGEST:
        raise RuntimeError(
            "model digest changed: expected "
            f"{EXPECTED_MODEL_DIGEST!r}, got {model.get('digest')!r}"
        )

    experiment_id = args.experiment_id or (
        f"structured-e-holdout-{datetime.now(timezone.utc).strftime('%Y%m%d')}-"
        f"{source['short_commit']}"
    )
    output = output_root / experiment_id
    if output.exists():
        raise FileExistsError(f"holdout output already exists: {output}")

    manifest = _experiment_manifest(
        source=source,
        model=model,
        holdout_manifest=holdout_manifest,
        development_reference=development_reference,
        timeout=args.timeout,
    )
    manifest["experiment_id"] = experiment_id

    with tempfile.TemporaryDirectory(prefix=f"{experiment_id}-") as temporary_name:
        temporary = Path(temporary_name)
        staging = temporary / "artifacts"
        staging.mkdir()
        staged_fixture = temporary / "fixture"
        _stage_fixture(fixture, holdout_manifest, staged_fixture)
        shutil.copyfile(args.fixture_manifest, staging / "holdout_manifest.json")

        warmup = run_in_subprocess(
            _warmup_worker, (args.model, args.think), timeout=args.timeout
        )
        manifest["execution"]["warmup_result"] = warmup
        if warmup.get("status") != "ok":
            raise RuntimeError("model warmup failed")

        started = time.perf_counter()
        try:
            metrics = run_evaluation(
                fixture=staged_fixture,
                expected_path=expected_path,
                output=staging / "e-holdout.md",
                model_id=args.model,
                think=args.think,
                timeout=args.timeout,
                use_agent=True,
                group=False,
                read_contents=False,
                allow_remote_content=False,
                metadata_control=True,
                perform_warmup=False,
                warmup_result=warmup,
            )
            run_status = metrics.get("status", "error")
        except Exception as exc:
            metrics = {"status": "error", "error": type(exc).__name__}
            run_status = "error"
        if int(metrics.get("provider_errors", 0) or 0):
            run_status = "provider_error"
        total_latency = time.perf_counter() - started

        run = {
            "experiment_id": experiment_id,
            "condition": "E",
            "condition_label": E_CONDITION["label"],
            "status": run_status,
            "total_run_latency_seconds": total_latency,
            "metrics": metrics,
        }
        raw_path = staging / "raw_run.jsonl"
        raw_path.write_text(
            json.dumps(run, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        security = None
        if run_status == "ok":
            security = _security_assertions(metrics)

        final_model = _model_manifest(args.model, args.think)
        identity_unchanged = final_model.get("digest") == model.get("digest")
        manifest["execution"]["model_identity_check"] = (
            "unchanged" if identity_unchanged else "changed during run"
        )
        complete = run_status == "ok" and identity_unchanged

        mechanism: dict[str, Any] = {}
        reliability: dict[str, Any] = {}
        error_analysis: list[dict[str, Any]] = []
        headline: dict[str, Any] = {}
        comparison: dict[str, Any] = {}
        confidence: dict[str, Any] = {}

        if complete:
            mechanism, reliability = _mechanism_and_reliability(metrics)
            cases = metrics.get("cases", [])
            for case in cases:
                label = _outcome_label(case)
                if label in ("correct_automatic", "correct_rule"):
                    continue
                error_analysis.append(
                    {
                        "filename": case["filename"],
                        "ground_truth": case["allowed"],
                        "predicted": case["predicted"],
                        "mechanism": case["mode"],
                        "outcome": label,
                    }
                )
            total = len(cases)
            correct = int(metrics.get("overall_correct", 0) or 0)
            decided = mechanism["decided_by_model"]
            decided_correct = mechanism["correct_automatic"]
            headline = {
                "total_files": total,
                "correct": correct,
                "incorrect_automatic": mechanism["incorrect_automatic"],
                "review": mechanism["ended_in_review"],
                "strict_category_accuracy": float(metrics.get("overall_accuracy", 0) or 0),
                "incorrect_decision_rate": (
                    mechanism["incorrect_automatic"] / total if total else 0.0
                ),
                "review_rate": (
                    mechanism["ended_in_review"] / total if total else 0.0
                ),
                "decision_rate": float(metrics.get("decision_rate", 0) or 0),
                "accuracy_on_decided": float(metrics.get("decided_accuracy", 0) or 0),
                "classification_latency_seconds": float(
                    metrics.get("class_latency_seconds", 0) or 0
                ),
                "total_run_latency_seconds": total_latency,
                "total_tokens": int(metrics.get("class_input_tokens", 0) or 0)
                + int(metrics.get("class_completion_tokens", 0) or 0),
                "class_input_tokens": int(metrics.get("class_input_tokens", 0) or 0),
                "class_completion_tokens": int(
                    metrics.get("class_completion_tokens", 0) or 0
                ),
                "classification_requests": int(
                    metrics.get("classification_requests", 0) or 0
                ),
            }
            accuracy_ci = _wilson_interval(correct, total)
            decided_ci = _wilson_interval(decided_correct, decided) if decided else (0.0, 0.0)
            confidence = {
                "strict_category_accuracy_wilson_95ci": accuracy_ci,
                "accuracy_on_decided_wilson_95ci": decided_ci,
                "note": (
                    f"{total} files; treat any single-holdout point estimate "
                    "as imprecise, not as a significance test"
                ),
            }
            comparison = {
                metric: headline_value - development_reference[metric]
                for metric, headline_value in (
                    ("strict_category_accuracy", headline["strict_category_accuracy"]),
                    ("incorrect_decision_rate", headline["incorrect_decision_rate"]),
                    ("review_rate", headline["review_rate"]),
                    ("decision_rate", headline["decision_rate"]),
                    ("accuracy_on_decided", headline["accuracy_on_decided"]),
                )
                if metric in development_reference
            }

        result = {
            "experiment_id": experiment_id,
            "complete": complete,
            "run_status": run_status,
            "headline": headline,
            "mechanism_coverage": mechanism,
            "structured_output_reliability": reliability,
            "confidence_intervals": confidence,
            "development_vs_holdout": comparison,
            "security": security,
        }
        (staging / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "error_analysis.json").write_text(
            json.dumps(error_analysis, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report = _render_report(
            experiment_id=experiment_id,
            manifest=manifest,
            result=result,
            error_analysis=error_analysis,
            development_reference=development_reference,
        )
        (staging / "report.md").write_text(report, encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, output)
    return output


def _render_report(
    *,
    experiment_id: str,
    manifest: dict[str, Any],
    result: dict[str, Any],
    error_analysis: list[dict[str, Any]],
    development_reference: dict[str, Any],
) -> str:
    lines = [
        f"# E — locked holdout evaluation — {experiment_id}",
        "",
        "> Single execution. E was preselected from development results before "
        "this run. A and C were not executed on the holdout.",
        "",
        "## Provenance",
        "",
        f"- Commit: `{manifest['source']['commit']}` (clean at experiment start)",
        f"- Model: `{manifest['model']['identifier']}`",
        f"- Digest: `{manifest['model']['digest']}`",
        f"- Holdout files: {manifest['dataset']['holdout_file_count']}",
        f"- Run status: `{result['run_status']}`",
        f"- Complete: {result['complete']}",
        "",
    ]
    if not result["complete"]:
        lines.append(
            "Run did not complete cleanly; see summary.json/raw_run.jsonl for "
            "the recorded failure. No further metrics were derived."
        )
        return "\n".join(lines) + "\n"

    headline = result["headline"]
    mechanism = result["mechanism_coverage"]
    reliability = result["structured_output_reliability"]
    ci = result["confidence_intervals"]
    comparison = result["development_vs_holdout"]

    lines.extend(
        [
            "## Results",
            "",
            f"- Total files: {headline['total_files']}",
            f"- Correct: {headline['correct']}",
            f"- Incorrect (automatic): {headline['incorrect_automatic']}",
            f"- `_ToReview`: {headline['review']}",
            f"- Strict category accuracy: {headline['strict_category_accuracy']:.1%} "
            f"(Wilson 95% CI {ci['strict_category_accuracy_wilson_95ci'][0]:.1%}"
            f"–{ci['strict_category_accuracy_wilson_95ci'][1]:.1%})",
            f"- Incorrect-decision rate: {headline['incorrect_decision_rate']:.1%}",
            f"- Review rate: {headline['review_rate']:.1%}",
            f"- Decision rate: {headline['decision_rate']:.1%}",
            f"- Accuracy on decided: {headline['accuracy_on_decided']:.1%} "
            f"(Wilson 95% CI {ci['accuracy_on_decided_wilson_95ci'][0]:.1%}"
            f"–{ci['accuracy_on_decided_wilson_95ci'][1]:.1%})",
            "",
            "## Mechanism coverage",
            "",
            f"- Deterministic-rule resolved: {mechanism['deterministic_rule_resolved']}",
            f"- Reached model classification: {mechanism['reached_model_classification']}",
            f"- Ended in `_ToReview`: {mechanism['ended_in_review']}",
            "",
            "## Structured-output reliability",
            "",
        ]
    )
    if any(reliability.get(f, 0) for f in reliability if f != "classification_requests"):
        for field, value in reliability.items():
            lines.append(f"- {field}: {value}")
    else:
        lines.append(
            f"- No protocol failures. {reliability.get('classification_requests', 0)} "
            "classification request(s), 0 parse/schema/provider/incomplete/"
            "duplicate/invented failures."
        )

    lines.extend(
        [
            "",
            "## Performance",
            "",
            f"- Total run latency: {headline['total_run_latency_seconds']:.1f} s",
            f"- Classification latency: {headline['classification_latency_seconds']:.1f} s",
            f"- Tokens: {headline['total_tokens']} "
            f"(input {headline['class_input_tokens']}, "
            f"completion {headline['class_completion_tokens']})",
            f"- Classification requests: {headline['classification_requests']}",
            "",
            "## Development vs holdout (E only, not combined)",
            "",
            f"- Development reference: {development_reference['run_count']} runs, "
            f"source `{development_reference['source_report']}`",
            "| Metric | Development | Holdout | Holdout − Development |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, delta in comparison.items():
        dev_value = development_reference[metric]
        holdout_value = headline[metric]
        lines.append(
            f"| {metric.replace('_', ' ')} | {dev_value:.1%} | {holdout_value:.1%} | "
            f"{delta:+.1%} |"
        )

    security = result.get("security") or {}
    lines.extend(
        [
            "",
            "## Security verification",
            "",
            f"- Zero content peeks: {security.get('zero_content_peeks')}",
            f"- Zero file-content reads: {security.get('zero_file_content_reads')}",
            f"- Zero parser invocations: {security.get('zero_parser_invocations')}",
            f"- Checked fields: {security.get('checked_fields')}",
            "- No filesystem mutation: the organization plan was evaluated, not executed.",
            "",
            "## Error analysis",
            "",
            f"{len(error_analysis)} file(s) not cleanly correct-and-automatic. "
            "See error_analysis.json for the full list (filename, ground truth, "
            "predicted, mechanism, outcome label — no file contents).",
            "",
            "| File | Ground truth | Predicted | Mechanism | Outcome |",
            "|---|---|---|---|---|",
        ]
    )
    for item in error_analysis:
        lines.append(
            f"| `{item['filename']}` | {', '.join(item['ground_truth'])} | "
            f"{item['predicted']} | {item['mechanism']} | {item['outcome']} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("MODEL_ID", DEFAULT_MODEL_ID))
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--think", dest="think", action="store_true")
    thinking.add_argument("--no-think", dest="think", action="store_false")
    parser.set_defaults(think=False)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--fixture", type=Path, default=PROJECT_ROOT / "evals/holdout/fixture"
    )
    parser.add_argument(
        "--expected", type=Path, default=PROJECT_ROOT / "evals/holdout/expected.yaml"
    )
    parser.add_argument(
        "--fixture-manifest",
        type=Path,
        default=PROJECT_ROOT / "evals/holdout/fixture_manifest.json",
    )
    parser.add_argument(
        "--development-summary",
        type=Path,
        default=PROJECT_ROOT
        / "evals/results/structured-aec-20260811-fc9d5fe9654d/summary.json",
    )
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "evals/results"
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    output = run_holdout(parse_args(argv))
    print(f"Holdout artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
