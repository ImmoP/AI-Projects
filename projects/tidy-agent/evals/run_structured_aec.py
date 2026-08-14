"""Run the frozen A/E/C diagnostic for selection, second-pass, and content effects."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
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

CONDITIONS = {
    "A": {
        "label": "One-pass metadata",
        "content": False,
        "metadata_control": False,
    },
    "E": {
        "label": "Two-pass metadata control",
        "content": False,
        "metadata_control": True,
    },
    "C": {
        "label": "Two-pass metadata + content",
        "content": True,
        "metadata_control": False,
    },
}
COUNTERBALANCED_SCHEDULE = (
    ("A", "E", "C"),
    ("E", "C", "A"),
    ("C", "A", "E"),
    ("A", "E", "C"),
    ("E", "C", "A"),
)
COUNT_FIELDS = (
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
    "peek_candidates_total",
    "peek_candidates_eligible",
    "peek_candidates_empty",
    "peek_candidates_oversized",
    "peek_candidates_unsupported",
    "peek_requests_model",
    "peek_requests_unique",
    "peek_requests_authorized",
    "peek_requests_control_selected",
    "peek_requests_rejected",
    "peek_requests_deduplicated",
    "peek_calls",
    "peek_readable",
    "peek_nonempty",
    "content_unavailable",
    "peek_source_bytes_considered",
    "peek_bytes_read",
    "peek_chars_returned",
    "peek_parser_skipped",
    "peek_parser_timeouts",
    "peek_parser_errors",
    "peek_phase_input_tokens",
    "peek_phase_completion_tokens",
    "final_classification_input_tokens",
    "final_classification_completion_tokens",
)


def _git_source() -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "live diagnostic requires a clean Git worktree; commit all intended "
            "changes before inference"
        )
    commit = _git("rev-parse", "HEAD")
    script = Path(__file__).resolve()
    return {
        "commit": commit,
        "short_commit": commit[:12],
        "dirty": False,
        "branch": _git("branch", "--show-current"),
        "evaluation_script": "evals/run_structured_aec.py",
        "evaluation_script_sha256": _sha256(script),
    }


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values) if values else 0.0,
        "maximum": max(values) if values else 0.0,
        "individual_values": values,
    }


def _metric_values(run: dict[str, Any]) -> dict[str, float]:
    metrics = run["metrics"]
    authorized = int(metrics.get("peek_requests_authorized", 0) or 0)
    nonempty = int(metrics.get("peek_nonempty", 0) or 0)
    return {
        "strict_category_accuracy": float(metrics.get("overall_accuracy", 0) or 0),
        "decision_rate": float(metrics.get("decision_rate", 0) or 0),
        "accuracy_on_decided": float(metrics.get("decided_accuracy", 0) or 0),
        "review_rate": float(metrics.get("review_rate", 0) or 0),
        "incorrect_decision_rate": 1
        - float(metrics.get("overall_accuracy", 0) or 0),
        "total_run_latency_seconds": float(
            run.get("total_run_latency_seconds", 0) or 0
        ),
        "classification_latency_seconds": float(
            metrics.get("class_latency_seconds", 0) or 0
        ),
        "total_tokens": float(
            int(metrics.get("class_input_tokens", 0) or 0)
            + int(metrics.get("class_completion_tokens", 0) or 0)
        ),
        "authorized_peeks": float(authorized),
        "nonempty_peeks": float(nonempty),
        "characters_delivered": float(
            int(metrics.get("peek_chars_returned", 0) or 0)
        ),
        "informative_peek_rate": nonempty / authorized if authorized else 0.0,
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = [run for run in runs if run["condition"] == condition]
        values = [_metric_values(run) for run in selected]
        result[condition] = {
            "label": CONDITIONS[condition]["label"],
            "runs": len(selected),
            "statuses": [run["status"] for run in selected],
            "metrics": {
                key: _stats([item[key] for item in values])
                for key in values[0]
            }
            if values
            else {},
            "totals": {
                field: sum(int(run["metrics"].get(field, 0) or 0) for run in selected)
                for field in COUNT_FIELDS
            },
            "requested_source_frequency": dict(
                sorted(
                    Counter(
                        source
                        for run in selected
                        for source in run["metrics"].get(
                            "peek_requested_sources", []
                        )
                    ).items()
                )
            ),
            "authorized_source_frequency": dict(
                sorted(
                    Counter(
                        source
                        for run in selected
                        for source in run["metrics"].get("peeked_sources", [])
                    ).items()
                )
            ),
        }
    return result


def _outcome(case: dict[str, Any]) -> str:
    if case["predicted"] == "_ToReview":
        return "review"
    return "correct" if case["correct"] else "wrong"


def _case_map(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["filename"]: case for case in run["metrics"].get("cases", [])}


def _paired_analysis(
    runs: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]]]:
    by_key = {(run["repetition"], run["condition"]): run for run in runs}
    transitions: dict[str, Counter[str]] = {
        "A_vs_E": Counter(),
        "E_vs_C": Counter(),
    }
    evidence: list[dict[str, Any]] = []
    for repetition in sorted({run["repetition"] for run in runs}):
        a = by_key.get((repetition, "A"))
        e = by_key.get((repetition, "E"))
        c = by_key.get((repetition, "C"))
        if not a or not e or not c:
            continue
        maps = {key: _case_map(run) for key, run in (("A", a), ("E", e), ("C", c))}
        c_file_metrics = c["metrics"].get("peek_file_metrics", {})
        for filename in maps["A"]:
            states = {key: _outcome(maps[key][filename]) for key in maps}
            for left, right in (("A", "E"), ("E", "C")):
                label = (
                    "unchanged"
                    if states[left] == states[right]
                    else f"{states[left]} → {states[right]}"
                )
                transitions[f"{left}_vs_{right}"][label] += 1
            file_metric = c_file_metrics.get(filename, {})
            if (
                maps["A"][filename]["predicted"]
                != maps["E"][filename]["predicted"]
                or maps["E"][filename]["predicted"]
                != maps["C"][filename]["predicted"]
                or filename in e["metrics"].get("peek_requested_sources", [])
                or filename in c["metrics"].get("peek_requested_sources", [])
            ):
                evidence.append(
                    {
                        "repetition": repetition,
                        "filename": filename,
                        "ground_truth": maps["C"][filename]["allowed"],
                        "A": maps["A"][filename]["predicted"],
                        "E": maps["E"][filename]["predicted"],
                        "C": maps["C"][filename]["predicted"],
                        "A_outcome": states["A"],
                        "E_outcome": states["E"],
                        "C_outcome": states["C"],
                        "E_requested": filename
                        in e["metrics"].get("peek_requested_sources", []),
                        "C_requested": filename
                        in c["metrics"].get("peek_requested_sources", []),
                        "C_authorized": filename
                        in c["metrics"].get("peeked_sources", []),
                        "C_readable": bool(file_metric.get("readable", False)),
                        "C_nonempty": bool(file_metric.get("nonempty", False)),
                        "C_bytes_read": int(file_metric.get("bytes_read", 0) or 0),
                        "C_chars_returned": int(
                            file_metric.get("chars_returned", 0) or 0
                        ),
                    }
                )
    return (
        {key: dict(sorted(value.items())) for key, value in transitions.items()},
        evidence,
    )


def _content_evidence_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    changed = [
        item
        for item in evidence
        if item["C_nonempty"] and item["E_outcome"] != item["C_outcome"]
    ]
    improvements = [
        item
        for item in changed
        if (item["E_outcome"], item["C_outcome"])
        in {
            ("wrong", "correct"),
            ("review", "correct"),
            ("wrong", "review"),
        }
    ]
    harms = [
        item
        for item in changed
        if (item["E_outcome"], item["C_outcome"])
        in {
            ("correct", "wrong"),
            ("correct", "review"),
            ("review", "wrong"),
        }
    ]
    return {
        "nonempty_content_changed_count": len(changed),
        "nonempty_content_improved_count": len(improvements),
        "nonempty_content_harmed_count": len(harms),
        "improved_files": sorted({item["filename"] for item in improvements}),
        "harmed_files": sorted({item["filename"] for item in harms}),
    }


def _experiment_manifest(
    *,
    source: dict[str, Any],
    model: dict[str, Any],
    fixture_manifest: dict[str, Any],
    holdout_manifest: dict[str, Any],
    repetitions: int,
    schedule: list[list[str]],
    timeout: float,
    allow_remote_content: bool,
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
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": _sha256(path),
                }
                for path in rule_paths
            ],
            "categories": list(rules.categories),
            "review_directory": rules.review_directory,
            "conditions": CONDITIONS,
            "remote_content_authorized": allow_remote_content,
            "max_task_peeks": MAX_TASK_PEEKS,
            "max_peek_chars": MAX_PEEK_CHARS,
            "max_plain_text_read_bytes": MAX_PEEK_BYTES,
            "max_text_file_bytes": MAX_TEXT_FILE_BYTES,
            "max_pdf_bytes": MAX_PDF_BYTES,
            "max_docx_bytes": MAX_DOCX_BYTES,
            "parser_timeout_seconds": PARSER_TIMEOUT_SECONDS,
            "phase_one_metadata": [
                "source",
                "extension",
                "size_bytes",
                "content_reader",
            ],
            "candidate_filter": [
                "zero-byte excluded",
                "known unsupported binary extension excluded",
                "format-size limit enforced before selection",
            ],
            "informative_peek_rate_definition": (
                "successful peeks returning at least one character / "
                "authorized peek requests"
            ),
        },
        "dataset": {
            "designation": "development benchmark",
            "fixture_manifest": "evals/dev/fixture_manifest.json",
            "fixture_dataset_sha256": fixture_manifest["dataset_sha256"],
            "fixture_file_count": len(fixture_manifest["files"]),
            "ground_truth_sha256": fixture_manifest["ground_truth"]["sha256"],
            "holdout_manifest": "evals/holdout/fixture_manifest.json",
            "holdout_dataset_sha256": holdout_manifest["dataset_sha256"],
            "holdout_executed": False,
        },
        "execution": {
            "repetitions": repetitions,
            "scheduled_runs": repetitions * len(CONDITIONS),
            "condition_order": schedule,
            "model_lifecycle": "warm",
            "warmup": "one discarded model request before all measured runs",
            "standard_timeout_seconds": timeout,
            "failed_run_policy": "record failure and stop; never substitute model",
            "control_E": (
                "same structured selection and final classification schemas as C; "
                "selected sources receive metadata-control markers; no peek tool, "
                "file read, or parser"
            ),
        },
    }


def _format_stat(item: dict[str, Any], *, percent: bool = False) -> str:
    factor = 100 if percent else 1
    suffix = "%" if percent else ""
    return (
        f"{item['mean'] * factor:.1f} ± "
        f"{item['standard_deviation'] * factor:.1f}{suffix}"
    )


def _render_report(
    *,
    experiment_id: str,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    transitions: dict[str, dict[str, int]],
    evidence: list[dict[str, Any]],
    content_evidence: dict[str, Any],
) -> str:
    lines = [
        f"# A/E/C content diagnostic — {experiment_id}",
        "",
        "> Development benchmark only. The locked holdout was not executed.",
        "",
        "## Provenance",
        "",
        f"- Commit: `{manifest['source']['commit']}` (clean at experiment start)",
        f"- Model: `{manifest['model']['identifier']}`",
        f"- Digest: `{manifest['model']['digest']}`",
        f"- Structured output: `{manifest['model']['structured_output_mode']}`",
        f"- Repetitions: {manifest['execution']['repetitions']} per condition; "
        "one discarded warmup",
        f"- Development files: {manifest['dataset']['fixture_file_count']}",
        "",
        "## Main results",
        "",
        "| Condition | Accuracy | Decision rate | Accuracy decided | Review | "
        "Incorrect | Latency | Tokens | Authorized peeks | Non-empty peeks | "
        "Characters |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = summary[condition]["metrics"]
        lines.append(
            f"| {condition} — {CONDITIONS[condition]['label']} | "
            f"{_format_stat(metrics['strict_category_accuracy'], percent=True)} | "
            f"{_format_stat(metrics['decision_rate'], percent=True)} | "
            f"{_format_stat(metrics['accuracy_on_decided'], percent=True)} | "
            f"{_format_stat(metrics['review_rate'], percent=True)} | "
            f"{_format_stat(metrics['incorrect_decision_rate'], percent=True)} | "
            f"{_format_stat(metrics['total_run_latency_seconds'])} s | "
            f"{_format_stat(metrics['total_tokens'])} | "
            f"{_format_stat(metrics['authorized_peeks'])} | "
            f"{_format_stat(metrics['nonempty_peeks'])} | "
            f"{_format_stat(metrics['characters_delivered'])} |"
        )

    lines.extend(
        [
            "",
            "## Candidate selection",
            "",
            "Phase 1 received relative source, normalized extension, byte size, and "
            "reader type only. Zero-byte, pre-known unsupported, and over-limit "
            "files were excluded in Python before model selection.",
        ]
    )
    for condition in ("E", "C"):
        totals = summary[condition]["totals"]
        rate = (
            totals["peek_nonempty"] / totals["peek_requests_authorized"]
            if totals["peek_requests_authorized"]
            else 0.0
        )
        lines.append(
            f"- {condition}: candidates {totals['peek_candidates_eligible']}/"
            f"{totals['peek_candidates_total']} across runs; empty filtered "
            f"{totals['peek_candidates_empty']}; requested "
            f"{totals['peek_requests_model']}; authorized "
            f"{totals['peek_requests_authorized']}; non-empty "
            f"{totals['peek_nonempty']}; informative-peek rate {rate:.1%}; "
            f"characters delivered {totals['peek_chars_returned']}."
        )
        lines.append(
            f"- {condition} requested sources: "
            f"{summary[condition]['requested_source_frequency'] or 'none'}."
        )

    lines.extend(["", "## Causal comparisons", ""])
    for key, value in transitions.items():
        rendered = ", ".join(f"{name}: {count}" for name, count in value.items())
        lines.append(f"- {key.replace('_', ' ')}: {rendered}.")
    a_accuracy = summary["A"]["metrics"]["strict_category_accuracy"]["mean"]
    e_accuracy = summary["E"]["metrics"]["strict_category_accuracy"]["mean"]
    c_accuracy = summary["C"]["metrics"]["strict_category_accuracy"]["mean"]
    lines.extend(
        [
            f"- Second-pass/control effect E−A: {e_accuracy - a_accuracy:+.1%}.",
            f"- Actual-content effect C−E: {c_accuracy - e_accuracy:+.1%}.",
            f"- Non-empty delivered content changed "
            f"{content_evidence['nonempty_content_changed_count']} paired file "
            f"outcomes, improved "
            f"{content_evidence['nonempty_content_improved_count']}, "
            f"and harmed {content_evidence['nonempty_content_harmed_count']}.",
            f"- Demonstrably improved files: {content_evidence['improved_files'] or 'none'}.",
            f"- Demonstrably harmed files: {content_evidence['harmed_files'] or 'none'}.",
            "",
            "## Per-file evidence",
            "",
            "| Rep | File | Ground truth | A | E | C | E requested | C "
            "requested/authorized | Non-empty | Bytes/chars |",
            "|---:|---|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in evidence:
        lines.append(
            f"| {item['repetition']} | `{item['filename']}` | "
            f"{', '.join(item['ground_truth'])} | {item['A']} | {item['E']} | "
            f"{item['C']} | {'yes' if item['E_requested'] else 'no'} | "
            f"{'yes' if item['C_requested'] else 'no'}/"
            f"{'yes' if item['C_authorized'] else 'no'} | "
            f"{'yes' if item['C_nonempty'] else 'no'} | "
            f"{item['C_bytes_read']}/{item['C_chars_returned']} |"
        )

    harms = content_evidence["nonempty_content_harmed_count"]
    improvements = content_evidence["nonempty_content_improved_count"]
    if c_accuracy > e_accuracy and improvements and not harms:
        recommendation = (
            "Keep A as the privacy-preserving default and retain C as an optional "
            "content mode only; actual content showed benefit beyond E."
        )
    elif e_accuracy > a_accuracy and abs(c_accuracy - e_accuracy) < 1e-12:
        recommendation = (
            "Prefer a metadata self-review/control design over content access; E "
            "captured the gain and C added none."
        )
    elif c_accuracy <= e_accuracy:
        recommendation = (
            "Keep A as default; content did not outperform the metadata control. "
            "Further development is needed before content can be justified."
        )
    else:
        recommendation = (
            "Keep A as default pending more development evidence; the content "
            "tradeoff is not yet decisive."
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            recommendation,
            "",
            "## Safety and holdout",
            "",
            "- E constructed no peek tool, read no file content, and invoked no parser.",
            f"- C never exceeded {MAX_TASK_PEEKS} authorized peeks per run.",
            "- Persisted artifacts contain per-file counts but no excerpts, "
            "absolute paths, raw endpoints, credentials, or chain-of-thought.",
            "- The 41-file holdout remained hash-locked and was not executed.",
            "- Holdout readiness depends on accepting this result without another "
            "prompt/schema/rule/model change; otherwise freeze and rerun development first.",
            "",
            "The pre-change root-cause table is in "
            "[content_selection_root_cause.md](../../content_selection_root_cause.md).",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(args: argparse.Namespace) -> Path:
    source = _git_source()
    fixture = args.fixture.resolve(strict=True)
    expected_path = args.expected.resolve(strict=True)
    fixture_manifest = _verify_dataset_manifest(
        args.fixture_manifest.resolve(strict=True),
        fixture,
        expected_path,
    )
    holdout_manifest_path = args.holdout_manifest.resolve(strict=True)
    holdout_manifest = _verify_dataset_manifest(
        holdout_manifest_path,
        holdout_manifest_path.parent / "fixture",
        holdout_manifest_path.parent / "expected.yaml",
    )
    schedule = [
        list(COUNTERBALANCED_SCHEDULE[index % len(COUNTERBALANCED_SCHEDULE)])
        for index in range(args.repetitions)
    ]
    model = _model_manifest(args.model, args.think)
    if not model["endpoint_local"] and not args.allow_remote_content:
        raise RuntimeError("remote endpoint requires --allow-remote-content for C")
    experiment_id = args.experiment_id or (
        f"structured-aec-{datetime.now(timezone.utc).strftime('%Y%m%d')}-"
        f"{source['short_commit']}"
    )
    output = args.output_root / experiment_id
    if output.exists():
        raise FileExistsError(f"experiment output already exists: {output}")
    manifest = _experiment_manifest(
        source=source,
        model=model,
        fixture_manifest=fixture_manifest,
        holdout_manifest=holdout_manifest,
        repetitions=args.repetitions,
        schedule=schedule,
        timeout=args.timeout,
        allow_remote_content=args.allow_remote_content,
    )
    manifest["experiment_id"] = experiment_id

    with tempfile.TemporaryDirectory(prefix=f"{experiment_id}-") as temporary_name:
        temporary = Path(temporary_name)
        staging = temporary / "artifacts"
        staging.mkdir()
        staged_fixture = temporary / "fixture"
        _stage_fixture(fixture, fixture_manifest, staged_fixture)
        shutil.copyfile(args.fixture_manifest, staging / "fixture_manifest.json")
        warmup = run_in_subprocess(
            _warmup_worker,
            (args.model, args.think),
            timeout=args.timeout,
        )
        manifest["execution"]["warmup_result"] = warmup
        if warmup.get("status") != "ok":
            raise RuntimeError("model warmup failed")

        runs: list[dict[str, Any]] = []
        raw_path = staging / "raw_runs.jsonl"
        aborted = False
        for repetition, order in enumerate(schedule, 1):
            for position, condition in enumerate(order, 1):
                configuration = CONDITIONS[condition]
                started = time.perf_counter()
                try:
                    metrics = run_evaluation(
                        fixture=staged_fixture,
                        expected_path=expected_path,
                        output=staging / f"{condition.lower()}-rep{repetition}.md",
                        model_id=args.model,
                        think=args.think,
                        timeout=args.timeout,
                        use_agent=True,
                        group=False,
                        read_contents=configuration["content"],
                        allow_remote_content=args.allow_remote_content,
                        metadata_control=configuration["metadata_control"],
                        perform_warmup=False,
                        warmup_result=warmup,
                    )
                except Exception as exc:
                    metrics = {"status": "error", "error": type(exc).__name__}
                run_status = metrics.get("status", "error")
                if int(metrics.get("provider_errors", 0) or 0):
                    run_status = "provider_error"
                run = {
                    "experiment_id": experiment_id,
                    "repetition": repetition,
                    "sequence_position": position,
                    "condition": condition,
                    "condition_label": configuration["label"],
                    "status": run_status,
                    "total_run_latency_seconds": time.perf_counter() - started,
                    "metrics": metrics,
                }
                runs.append(run)
                with raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(run, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                if run_status != "ok":
                    aborted = True
                    break
                if condition == "E" and any(
                    int(metrics.get(field, 0) or 0)
                    for field in (
                        "peek_calls",
                        "peek_bytes_read",
                        "peek_chars_returned",
                    )
                ):
                    raise RuntimeError("metadata control attempted content access")
                if int(metrics.get("peek_calls", 0) or 0) > MAX_TASK_PEEKS:
                    raise RuntimeError("peek budget exceeded")
            if aborted:
                break

        final_model = _model_manifest(args.model, args.think)
        unchanged = final_model.get("digest") == model.get("digest")
        manifest["execution"]["model_identity_check"] = (
            "unchanged" if unchanged else "changed during run"
        )
        if not unchanged:
            aborted = True
        summary = _aggregate(runs)
        transitions, evidence = _paired_analysis(runs)
        content_evidence = _content_evidence_summary(evidence)
        result = {
            "experiment_id": experiment_id,
            "complete": not aborted and len(runs) == args.repetitions * 3,
            "run_count": len(runs),
            "condition_order": schedule,
            "conditions": summary,
            "paired_transitions": transitions,
            "content_evidence": content_evidence,
        }
        (staging / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "per_file_evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report = (
            _render_report(
                experiment_id=experiment_id,
                manifest=manifest,
                summary=summary,
                transitions=transitions,
                evidence=evidence,
                content_evidence=content_evidence,
            )
            if result["complete"]
            else f"# {experiment_id}\n\nStopped after {len(runs)} scheduled runs.\n"
        )
        (staging / "report.md").write_text(report, encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, output)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("MODEL_ID", DEFAULT_MODEL_ID))
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--think", dest="think", action="store_true")
    thinking.add_argument("--no-think", dest="think", action="store_false")
    parser.set_defaults(think=False)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--allow-remote-content", action="store_true")
    parser.add_argument("--experiment-id")
    parser.add_argument("--fixture", type=Path, default=PROJECT_ROOT / "evals/fixture")
    parser.add_argument(
        "--expected",
        type=Path,
        default=PROJECT_ROOT / "evals/expected.yaml",
    )
    parser.add_argument(
        "--fixture-manifest",
        type=Path,
        default=PROJECT_ROOT / "evals/dev/fixture_manifest.json",
    )
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        default=PROJECT_ROOT / "evals/holdout/fixture_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "evals/results",
    )
    args = parser.parse_args(argv)
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    output = run_experiment(parse_args(argv))
    print(f"Experiment artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
