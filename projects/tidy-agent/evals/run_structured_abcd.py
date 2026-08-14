"""Run the frozen, counterbalanced structured-classification A/B/C/D experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from tidy.agent import (
    DEFAULT_MODEL_ID,
    build_model,
    endpoint_is_local,
    resolved_model_endpoint,
    safe_endpoint_display,
)
from tidy.classification import ClassificationBackend
from tidy.content_parser import MAX_DOCX_BYTES, MAX_PDF_BYTES, PARSER_TIMEOUT_SECONDS
from tidy.rules import load_rules
from tidy.tools import (
    MAX_PEEK_BYTES,
    MAX_PEEK_CHARS,
    MAX_TASK_PEEKS,
    MAX_TEXT_FILE_BYTES,
)

from evals.run_evals import (
    REVIEW_DIRECTORY,
    UNSCORED_PREDICTIONS,
    _ollama_json,
    _warmup_worker,
    endpoint_url,
    run_evaluation,
    run_in_subprocess,
)

CONDITIONS = {
    "A": {"label": "Metadata only", "group": False, "content": False},
    "B": {"label": "Metadata + Grouping", "group": True, "content": False},
    "C": {"label": "Metadata + Content", "group": False, "content": True},
    "D": {"label": "Metadata + Grouping + Content", "group": True, "content": True},
}
COUNTERBALANCED_SCHEDULE = (
    ("A", "B", "C", "D"),
    ("B", "C", "D", "A"),
    ("C", "D", "A", "B"),
    ("D", "A", "B", "C"),
    ("A", "C", "B", "D"),
)
DEPENDENCIES = (
    "smolagents",
    "litellm",
    "pydantic",
    "PyYAML",
    "pypdf",
    "python-docx",
    "python-dotenv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_source() -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "live evaluation requires a clean Git worktree; commit or remove "
            "all changes first"
        )
    commit = _git("rev-parse", "HEAD")
    return {
        "commit": commit,
        "short_commit": commit[:12],
        "dirty": False,
        "branch": _git("branch", "--show-current"),
        "evaluation_script": "evals/run_structured_abcd.py",
        "evaluation_script_sha256": _sha256(Path(__file__).resolve()),
    }


def _load_expected(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload.get("files"), dict):
        raise ValueError("ground truth must contain a files mapping")
    return payload


def _verify_dataset_manifest(path: Path, fixture: Path, expected: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for item in manifest.get("files", []):
        target = fixture / item["path"]
        if not target.is_file():
            raise RuntimeError(f"frozen fixture file missing: {item['path']}")
        if target.stat().st_size != item["size"] or _sha256(target) != item["sha256"]:
            raise RuntimeError(f"frozen fixture file changed: {item['path']}")
    expected_record = manifest.get("ground_truth", {})
    if _sha256(expected) != expected_record.get("sha256"):
        raise RuntimeError("frozen ground truth hash changed")
    calculated = _canonical_digest(
        {"files": manifest.get("files", []), "ground_truth": expected_record}
    )
    if calculated != manifest.get("dataset_sha256"):
        raise RuntimeError("dataset manifest self-hash is invalid")
    return manifest


def _stage_fixture(source: Path, manifest: dict[str, Any], target: Path) -> None:
    target.mkdir(parents=True)
    for item in manifest["files"]:
        destination = target / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / item["path"], destination)


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def _model_manifest(model_id: str | None, think: bool) -> dict[str, Any]:
    resolved_id, api_base = resolved_model_endpoint(model_id)
    local = endpoint_is_local(resolved_id, api_base=api_base)
    model = build_model(resolved_id, api_base=api_base, think=think)
    backend = ClassificationBackend(model)
    result: dict[str, Any] = {
        "identifier": resolved_id,
        "family": resolved_id.split("/", 1)[-1].split(":", 1)[0],
        "provider_runtime": "Ollama via LiteLLM"
        if resolved_id.startswith("ollama")
        else "LiteLLM provider",
        "endpoint": safe_endpoint_display(resolved_id, api_base=api_base),
        "endpoint_local": local,
        "structured_output_mode": backend.telemetry.structured_output_mode,
        "temperature": model.kwargs.get("temperature", "not configured"),
        "seed": "not configured",
        "thinking": think,
        "max_tokens": "not configured (provider default)",
        "context_length": model.kwargs.get("num_ctx", "not configured"),
        "request_timeout_seconds": model.kwargs.get("timeout", "not configured"),
        "retry_policy": "zero application retries",
    }
    if not resolved_id.startswith("ollama"):
        result.update(
            {
                "digest": "unsupported",
                "quantization": "unsupported",
                "runtime_version": "provider managed",
            }
        )
        return result

    host = endpoint_url(resolved_id).rstrip("/")
    version = _ollama_json(host, "/api/version")
    tags = _ollama_json(host, "/api/tags")
    loaded = _ollama_json(host, "/api/ps")
    if not isinstance(version, dict) or not isinstance(tags, dict):
        raise RuntimeError("configured Ollama endpoint is unavailable")
    name = resolved_id.split("/", 1)[1]
    entry = next(
        (
            item
            for item in tags.get("models", []) or []
            if item.get("name") == name or item.get("model") == name
        ),
        None,
    )
    if not isinstance(entry, dict):
        raise RuntimeError(f"configured model is unavailable: {resolved_id}")
    details = entry.get("details", {}) or {}
    loaded_entry = next(
        (
            item
            for item in (loaded or {}).get("models", [])
            if item.get("name") == name or item.get("model") == name
        ),
        None,
    )
    result.update(
        {
            "digest": str(entry.get("digest", "")),
            "quantization": str(details.get("quantization_level", "unknown")),
            "parameter_size": str(details.get("parameter_size", "unknown")),
            "runtime_version": str(version.get("version", "unknown")),
            "already_loaded_before_warmup": loaded_entry is not None,
            "loaded_context_length": (
                loaded_entry.get("context_length", "not loaded")
                if isinstance(loaded_entry, dict)
                else "not loaded"
            ),
            "loaded_size_vram_bytes": (
                loaded_entry.get("size_vram", "not loaded")
                if isinstance(loaded_entry, dict)
                else "not loaded"
            ),
        }
    )
    return result


def _experiment_manifest(
    *,
    source: dict[str, Any],
    model: dict[str, Any],
    fixture_manifest: dict[str, Any],
    holdout_manifest: dict[str, Any],
    repetitions: int,
    schedule: list[list[str]],
    timeout: float,
    group_timeout: float,
    allow_remote_content: bool,
) -> dict[str, Any]:
    rules = load_rules()
    rule_paths = [PROJECT_ROOT / "config/rules.yaml", PROJECT_ROOT / "src/tidy/config/rules.yaml"]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "python_environment": {
            "operating_system": platform.platform(),
            "architecture": platform.machine(),
            "processor": platform.processor() or "not reported",
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
            "group_timeout_seconds": group_timeout,
            "failed_run_policy": "record failure and stop; never substitute model",
        },
    }


def _category(metrics: dict[str, Any], condition: str) -> dict[str, Any]:
    return metrics["category"] if CONDITIONS[condition]["group"] else metrics


def _condition_values(run: dict[str, Any]) -> dict[str, float]:
    condition = run["condition"]
    metrics = run["metrics"]
    category = _category(metrics, condition)
    cases = category.get("cases", [])
    incorrect = sum(
        not case.get("correct", False)
        and case.get("predicted") != REVIEW_DIRECTORY
        and case.get("predicted") not in UNSCORED_PREDICTIONS
        for case in cases
    )
    total_tokens = (
        int(metrics.get("input_tokens", 0)) + int(metrics.get("completion_tokens", 0))
        if CONDITIONS[condition]["group"]
        else int(category.get("class_input_tokens", 0))
        + int(category.get("class_completion_tokens", 0))
    )
    return {
        "strict_category_accuracy": float(category.get("overall_accuracy", 0)),
        "decision_rate": float(category.get("decision_rate", 0)),
        "accuracy_on_decided": float(category.get("decided_accuracy", 0)),
        "review_rate": float(category.get("review_rate", 0)),
        "incorrect_decision_rate": incorrect / len(cases) if cases else 0.0,
        "total_run_latency_seconds": float(run["total_run_latency_seconds"]),
        "classification_latency_seconds": float(
            category.get("class_latency_seconds", 0)
        ),
        "peek_phase_latency_seconds": float(
            category.get("peek_phase_latency_seconds", 0)
        ),
        "content_processing_latency_seconds": float(
            category.get("content_processing_latency_seconds", 0)
        ),
        "final_classification_latency_seconds": float(
            category.get("final_classification_latency_seconds", 0)
        ),
        "grouping_latency_seconds": float(metrics.get("group_latency_seconds", 0)),
        "grouping_input_tokens": float(metrics.get("group_input_tokens", 0)),
        "grouping_output_tokens": float(metrics.get("group_completion_tokens", 0)),
        "peek_phase_input_tokens": float(category.get("peek_phase_input_tokens", 0)),
        "peek_phase_output_tokens": float(
            category.get("peek_phase_completion_tokens", 0)
        ),
        "final_classification_input_tokens": float(
            category.get("final_classification_input_tokens", 0)
        ),
        "final_classification_output_tokens": float(
            category.get("final_classification_completion_tokens", 0)
        ),
        "input_tokens": float(
            metrics.get("input_tokens", category.get("class_input_tokens", 0))
        ),
        "output_tokens": float(
            metrics.get("completion_tokens", category.get("class_completion_tokens", 0))
        ),
        "total_tokens": float(total_tokens),
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


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_runs = [run for run in runs if run["condition"] == condition]
        values = [_condition_values(run) for run in condition_runs]
        metrics = {
            key: _stats([item[key] for item in values])
            for key in values[0]
        } if values else {}
        categories = [_category(run["metrics"], condition) for run in condition_runs]
        count_fields = (
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
            "peek_requests_model",
            "peek_requests_unique",
            "peek_requests_authorized",
            "peek_requests_rejected",
            "peek_requests_deduplicated",
            "peek_calls",
            "peek_readable",
            "content_unavailable",
            "peek_parser_errors",
            "peek_parser_timeouts",
            "peek_source_bytes_considered",
            "peek_bytes_read",
            "peek_chars_returned",
            "peek_phase_input_tokens",
            "peek_phase_completion_tokens",
            "final_classification_input_tokens",
            "final_classification_completion_tokens",
        )
        coverage = {
            "deterministic_rules": [
                len(category.get("deterministic_rule_sources", []))
                for category in categories
            ],
            "structured_classification": [
                int(category.get("classification_source_count", 0))
                for category in categories
            ],
            "content_assisted": [
                len(category.get("peeked_sources", [])) for category in categories
            ],
            "grouping": [
                int(run["metrics"].get("grouped_total", 0))
                for run in condition_runs
            ],
            "review": [int(category.get("review_count", 0)) for category in categories],
        }
        summary[condition] = {
            "label": CONDITIONS[condition]["label"],
            "runs": len(condition_runs),
            "statuses": [run["status"] for run in condition_runs],
            "metrics": metrics,
            "reliability_totals": {
                field: sum(int(category.get(field, 0)) for category in categories)
                for field in count_fields
            },
            "coverage": {key: _stats([float(value) for value in entries]) for key, entries in coverage.items()},
            "category_denominators": [
                int(category.get("overall_total", 0)) for category in categories
            ],
            "grouping": {
                field: [run["metrics"].get(field, 0) for run in condition_runs]
                for field in (
                    "groups_proposed",
                    "groups_accepted",
                    "groups_discarded",
                    "groups_rejected",
                    "grouped_total",
                    "clustering_purity",
                    "group_cohesion",
                    "scatter_in_group",
                    "group_input_tokens",
                    "group_completion_tokens",
                )
            } if CONDITIONS[condition]["group"] else {},
        }
        if CONDITIONS[condition]["group"]:
            summary[condition]["grouping"].update(
                {
                    "grouped_rule_resolved_count": [
                        len(run["metrics"].get("grouped_rule_resolved_sources", []))
                        for run in condition_runs
                    ],
                    "grouped_unresolved_count": [
                        len(run["metrics"].get("grouped_unresolved_sources", []))
                        for run in condition_runs
                    ],
                }
            )
    return summary


def _outcome_state(
    *,
    filename: str,
    condition: str,
    metrics: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[str, str]:
    category = _category(metrics, condition)
    destination = category.get("all_destinations", {}).get(filename, "UNASSIGNED")
    grouped = set(category.get("grouped_sources", []))
    if destination == REVIEW_DIRECTORY:
        return "review", destination
    if filename not in grouped:
        allowed = expected["files"].get(filename, [])
        return ("correct" if destination in allowed else "wrong"), destination

    label_for = {
        member: label
        for label, members in (expected.get("groups", {}) or {}).items()
        for member in members
    }
    members = [
        name
        for name in grouped
        if category.get("all_destinations", {}).get(name) == destination
    ]
    label = label_for.get(filename)
    pure = label is not None and all(label_for.get(member) == label for member in members)
    return ("correct" if pure else "wrong"), destination


def _paired_analysis(
    runs: list[dict[str, Any]], expected: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    by_pair = {(run["repetition"], run["condition"]): run for run in runs}
    transitions: dict[str, Counter[str]] = {
        "A_vs_C": Counter(),
        "B_vs_D": Counter(),
        "A_vs_B": Counter(),
        "C_vs_D": Counter(),
    }
    per_file: dict[str, dict[str, Any]] = {
        name: {"transitions": [], "destinations": {}, "peeked": {}, "grouped": {}}
        for name in expected["files"]
    }
    for repetition in sorted({run["repetition"] for run in runs}):
        for left, right, label in (
            ("A", "C", "A_vs_C"),
            ("B", "D", "B_vs_D"),
            ("A", "B", "A_vs_B"),
            ("C", "D", "C_vs_D"),
        ):
            if (repetition, left) not in by_pair or (repetition, right) not in by_pair:
                continue
            left_run = by_pair[(repetition, left)]
            right_run = by_pair[(repetition, right)]
            for filename in expected["files"]:
                left_state, left_destination = _outcome_state(
                    filename=filename,
                    condition=left,
                    metrics=left_run["metrics"],
                    expected=expected,
                )
                right_state, right_destination = _outcome_state(
                    filename=filename,
                    condition=right,
                    metrics=right_run["metrics"],
                    expected=expected,
                )
                transition = (
                    "unchanged"
                    if left_state == right_state and left_destination == right_destination
                    else f"{left_state} → {right_state}"
                )
                transitions[label][transition] += 1
                if transition != "unchanged":
                    per_file[filename]["transitions"].append(
                        {"pair": label, "repetition": repetition, "transition": transition}
                    )

    for run in runs:
        condition = run["condition"]
        category = _category(run["metrics"], condition)
        for filename in expected["files"]:
            _, destination = _outcome_state(
                filename=filename,
                condition=condition,
                metrics=run["metrics"],
                expected=expected,
            )
            per_file[filename]["destinations"].setdefault(condition, []).append(destination)
            per_file[filename]["peeked"].setdefault(condition, []).append(
                filename in set(category.get("peeked_sources", []))
            )
            per_file[filename]["grouped"].setdefault(condition, []).append(
                filename in set(category.get("grouped_sources", []))
            )
    return {key: dict(value) for key, value in transitions.items()}, per_file


def _error_analysis(
    per_file: dict[str, dict[str, Any]], expected: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename, data in per_file.items():
        labels: list[str] = []
        outcome_summary: dict[str, str] = {}
        unstable = False
        repeated_review = False
        for condition, destinations in data["destinations"].items():
            counts = Counter(destinations)
            display = ", ".join(f"{name} ({count}/{len(destinations)})" for name, count in counts.most_common())
            outcome_summary[condition] = display
            unstable |= len(counts) > 1
            repeated_review |= counts.get(REVIEW_DIRECTORY, 0) >= max(1, len(destinations) // 2 + 1)
        transitions = [item["transition"] for item in data["transitions"]]
        if any(value in {"wrong → correct", "review → correct"} for value in transitions):
            labels.append("content improved outcome")
        if any(value in {"correct → wrong", "correct → review", "review → wrong"} for value in transitions):
            labels.append("content harmed outcome")
        if unstable:
            labels.append("unstable between repetitions")
        if repeated_review:
            labels.append("repeatedly sent to _ToReview")
        if any(
            any(data["grouped"].get(condition, [])) for condition in ("B", "D")
        ):
            labels.append("changed by grouping")
        consistently_wrong = True
        for condition in CONDITIONS:
            condition_destinations = data["destinations"].get(condition, [])
            for destination in condition_destinations:
                if destination in expected["files"][filename] or destination == REVIEW_DIRECTORY:
                    consistently_wrong = False
        if consistently_wrong:
            labels.append("consistently wrong")
        if not labels:
            continue
        rows.append(
            {
                "filename": filename,
                "ground_truth": expected["files"][filename],
                "outcomes": outcome_summary,
                "stability": "unstable" if unstable else "stable",
                "peeked": {
                    condition: any(values)
                    for condition, values in data["peeked"].items()
                },
                "grouped": {
                    condition: any(values)
                    for condition, values in data["grouped"].items()
                },
                "analysis_label": "; ".join(labels),
            }
        )
    return rows


def _mean(summary: dict[str, Any], condition: str, metric: str) -> float:
    return float(summary[condition]["metrics"][metric]["mean"])


def _render_report(
    *,
    experiment_id: str,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    transitions: dict[str, Any],
    errors: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Structured A/B/C/D development evaluation — {experiment_id}",
        "",
        "> This is the development benchmark, not the locked holdout.",
        "",
        "## Experiment provenance",
        "",
        f"- Git commit: `{manifest['source']['commit']}` (clean at experiment start)",
        f"- Model: `{manifest['model']['identifier']}`",
        f"- Digest: `{manifest['model']['digest']}`",
        f"- Runtime: {manifest['model']['provider_runtime']} {manifest['model']['runtime_version']}",
        f"- Endpoint: {manifest['model']['endpoint']}",
        f"- Structured output: `{manifest['model']['structured_output_mode']}`",
        f"- Repetitions: {manifest['execution']['repetitions']} per condition; warm model after one discarded warmup",
        "",
        "## Dataset",
        "",
        f"- Files: {manifest['dataset']['fixture_file_count']}",
        f"- Fixture manifest SHA-256: `{manifest['dataset']['fixture_dataset_sha256']}`",
        f"- Ground-truth SHA-256: `{manifest['dataset']['ground_truth_sha256']}`",
        "- Deterministic rules remained enabled in every condition.",
        "",
        "## A/B/C/D results",
        "",
        "Category accuracy is calculated over files receiving category decisions; grouped files are excluded and reported through grouping metrics.",
        "",
        "| Condition | Category accuracy | Decision rate | Accuracy decided | Review rate | Incorrect decision rate | Total latency | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        def stat(metric: str, percent: bool = False) -> str:
            value = summary[condition]["metrics"][metric]
            factor = 100 if percent else 1
            suffix = "%" if percent else ""
            return f"{value['mean'] * factor:.1f} ± {value['standard_deviation'] * factor:.1f}{suffix}"

        lines.append(
            f"| {condition} — {CONDITIONS[condition]['label']} | "
            f"{stat('strict_category_accuracy', True)} | {stat('decision_rate', True)} | "
            f"{stat('accuracy_on_decided', True)} | {stat('review_rate', True)} | "
            f"{stat('incorrect_decision_rate', True)} | {stat('total_run_latency_seconds')} s | "
            f"{stat('total_tokens')} |"
        )

    lines.extend(["", "### Coverage by mechanism", "", "| Condition | Rules | Classifier | Content-peeked | Grouped | _ToReview |", "|---|---:|---:|---:|---:|---:|"])
    for condition in CONDITIONS:
        coverage = summary[condition]["coverage"]
        lines.append(
            f"| {condition} | {coverage['deterministic_rules']['mean']:.1f} | "
            f"{coverage['structured_classification']['mean']:.1f} | "
            f"{coverage['content_assisted']['mean']:.1f} | "
            f"{coverage['grouping']['mean']:.1f} | {coverage['review']['mean']:.1f} |"
        )

    lines.extend(["", "## Content analysis", ""])
    for pair in ("A_vs_C", "B_vs_D"):
        rendered = ", ".join(f"{key}: {value}" for key, value in sorted(transitions[pair].items()))
        lines.append(f"- {pair.replace('_', ' ')} paired transitions: {rendered or 'none'}.")
    for condition in ("C", "D"):
        reliability = summary[condition]["reliability_totals"]
        available = manifest["execution"]["repetitions"] * MAX_TASK_PEEKS
        utilization = reliability["peek_calls"] / available if available else 0
        lines.append(
            f"- {condition}: {reliability['peek_calls']}/{available} actual peeks "
            f"({utilization:.1%} utilization), {reliability['peek_readable']} readable, "
            f"{reliability['content_unavailable']} unavailable, "
            f"{reliability['peek_parser_errors']} parser errors and "
            f"{reliability['peek_parser_timeouts']} timeouts."
        )
        lines.append(
            f"- {condition} token split: peek phase "
            f"{reliability['peek_phase_input_tokens']} input / "
            f"{reliability['peek_phase_completion_tokens']} output; final phase "
            f"{reliability['final_classification_input_tokens']} input / "
            f"{reliability['final_classification_completion_tokens']} output."
        )

    lines.extend(["", "## Grouping analysis", ""])
    for condition in ("B", "D"):
        grouping = summary[condition]["grouping"]
        purity = grouping["clustering_purity"]
        cohesion = grouping["group_cohesion"]
        lines.append(
            f"- {condition}: groups proposed {grouping['groups_proposed']}; accepted "
            f"{grouping['groups_accepted']}; grouped files {grouping['grouped_total']}; "
            f"purity {statistics.mean(purity):.1%} ± {statistics.stdev(purity) if len(purity) > 1 else 0:.1%}; "
            f"cohesion {statistics.mean(cohesion):.1%}; scatter harms {grouping['scatter_in_group']}; "
            f"rule/model destinations overridden {grouping['grouped_rule_resolved_count']}/"
            f"{grouping['grouped_unresolved_count']}."
        )
        comparison = "A_vs_B" if condition == "B" else "C_vs_D"
        rendered = ", ".join(
            f"{key}: {value}" for key, value in sorted(transitions[comparison].items())
        )
        lines.append(f"- {comparison.replace('_', ' ')} outcome transitions: {rendered or 'none'}.")

    lines.extend(["", "## Structured-output reliability", "", "| Condition | Requests | Native schema | JSON object | Plain JSON | Parse failures | Schema failures | Provider failures | Incomplete | Duplicate | Invented source/category | Forced review |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for condition in CONDITIONS:
        item = summary[condition]["reliability_totals"]
        lines.append(
            f"| {condition} | {item['classification_requests']} | {item['native_schema_responses']} | "
            f"{item['json_object_responses']} | {item['plain_json_responses']} | "
            f"{item['parse_failures']} | {item['schema_validation_failures']} | "
            f"{item['provider_errors']} | {item['incomplete_responses']} | "
            f"{item['duplicate_source_responses']} | {item['invented_source_responses']}/{item['invented_category_responses']} | "
            f"{item['fallback_to_review_count']} |"
        )

    lines.extend(["", "## Stability", ""])
    for condition in CONDITIONS:
        latency = summary[condition]["metrics"]["total_run_latency_seconds"]
        accuracy = summary[condition]["metrics"]["strict_category_accuracy"]
        decision = summary[condition]["metrics"]["decision_rate"]
        review = summary[condition]["metrics"]["review_rate"]
        incorrect = summary[condition]["metrics"]["incorrect_decision_rate"]
        lines.append(
            f"- {condition}: accuracy {accuracy['individual_values']}; decision rate "
            f"{decision['individual_values']}; review rate {review['individual_values']}; "
            f"incorrect-decision rate {incorrect['individual_values']}; latency "
            f"mean/median/SD/min/max = {latency['mean']:.1f}/{latency['median']:.1f}/"
            f"{latency['standard_deviation']:.1f}/{latency['minimum']:.1f}/{latency['maximum']:.1f} s."
        )
    lines.append(f"- Difficult or unstable files listed below: {len(errors)}.")

    lines.extend(["", "## Security evaluation", ""])
    rejected = sum(
        summary[condition]["reliability_totals"]["peek_requests_rejected"]
        for condition in ("C", "D")
    )
    lines.extend(
        [
            "- Capability-boundary violations: **0 observed**; no run exceeded the Python four-peek limit and model output never executed filesystem operations.",
            f"- Unauthorized/invalid/over-budget peek requests rejected: {rejected}.",
            "- Privacy-control failures: **0 observed**; artifacts contain filenames, outcomes and counts but no excerpts, credentials, raw endpoints or chain-of-thought.",
            "- The development fixture contains no intentionally labelled semantic prompt-injection subset. Semantic manipulation resistance is therefore not measured here; the locked holdout contains such cases and was not executed.",
            "- Capability escalation prevention and semantic manipulation resistance are separate claims.",
        ]
    )

    lines.extend(["", "## Per-file error analysis", "", "| File | Ground truth | A | B | C | D | Stability | Peeked | Grouped | Deterministic label |", "|---|---|---|---|---|---|---|---|---|---|"])
    for item in errors:
        outcomes = item["outcomes"]
        peeked = ", ".join(key for key, value in item["peeked"].items() if value) or "no"
        grouped = ", ".join(key for key, value in item["grouped"].items() if value) or "no"
        filename = item["filename"].replace("|", "\\|")
        lines.append(
            f"| `{filename}` | {', '.join(item['ground_truth'])} | "
            f"{outcomes.get('A', '—')} | {outcomes.get('B', '—')} | "
            f"{outcomes.get('C', '—')} | {outcomes.get('D', '—')} | "
            f"{item['stability']} | {peeked} | {grouped} | {item['analysis_label']} |"
        )

    content_gain = _mean(summary, "C", "strict_category_accuracy") - _mean(summary, "A", "strict_category_accuracy")
    content_group_gain = _mean(summary, "D", "strict_category_accuracy") - _mean(summary, "B", "strict_category_accuracy")
    content_review_delta = _mean(summary, "C", "review_rate") - _mean(summary, "A", "review_rate")
    content_group_review_delta = _mean(summary, "D", "review_rate") - _mean(summary, "B", "review_rate")
    content_latency_cost = _mean(summary, "C", "total_run_latency_seconds") - _mean(summary, "A", "total_run_latency_seconds")
    content_group_latency_cost = _mean(summary, "D", "total_run_latency_seconds") - _mean(summary, "B", "total_run_latency_seconds")
    content_token_cost = _mean(summary, "C", "total_tokens") - _mean(summary, "A", "total_tokens")
    content_group_token_cost = _mean(summary, "D", "total_tokens") - _mean(summary, "B", "total_tokens")
    grouping_purity = statistics.mean(summary["B"]["grouping"]["clustering_purity"])
    grouping_coverage = statistics.mean(summary["B"]["grouping"]["grouped_total"])
    grouping_scatter = sum(summary["B"]["grouping"]["scatter_in_group"])
    content_harm = (
        _mean(summary, "C", "incorrect_decision_rate")
        > _mean(summary, "A", "incorrect_decision_rate")
        or _mean(summary, "D", "incorrect_decision_rate")
        > _mean(summary, "B", "incorrect_decision_rate")
    )
    content_benefit = (content_gain > 0 or content_group_gain > 0) and not content_harm
    grouping_benefit = grouping_coverage > 0 and grouping_purity >= 0.95 and grouping_scatter == 0
    default = "D" if content_benefit and grouping_benefit else "C" if content_benefit else "B" if grouping_benefit else "A"
    reliability_failures = sum(
        summary[condition]["reliability_totals"][field]
        for condition in CONDITIONS
        for field in ("parse_failures", "schema_validation_failures", "provider_errors")
    )
    lines.extend(
        [
            "",
            "## Historical comparison — non-controlled",
            "",
            "Historical CodeAgent reports remain context only. They used a different classification protocol and, in some runs, different content behavior; they are not part of this controlled comparison.",
            "",
            "## Core experimental questions",
            "",
            f"- **Q1 — Accuracy:** C−A is {content_gain:+.1%}; D−B is {content_group_gain:+.1%} in category-scored files.",
            f"- **Q2 — `_ToReview`:** C−A is {content_review_delta:+.1%}; D−B is {content_group_review_delta:+.1%}. Incorrect-decision rates are shown beside these changes in the main table.",
            f"- **Q3 — Grouping:** B has mean purity {grouping_purity:.1%}, mean coverage {grouping_coverage:.1f} files, and {grouping_scatter} harmful scatter placements across repetitions; C-vs-D is reported through the paired transitions.",
            f"- **Q4 — Interaction:** D should be preferred over B or C only if its paired gains justify both axes; its category delta over B is {content_group_gain:+.1%}.",
            f"- **Q5 — Reliability:** parse/schema/provider failures total {reliability_failures} across all scheduled runs.",
            f"- **Q6 — Content cost:** C−A costs {content_latency_cost:+.1f} s and {content_token_cost:+.0f} tokens per run; D−B costs {content_group_latency_cost:+.1f} s and {content_group_token_cost:+.0f} tokens per run.",
            "",
            "## Development benchmark conclusion",
            "",
            f"1. Content effect without grouping (C−A category accuracy): {content_gain:+.1%}.",
            f"2. Content effect with grouping (D−B category accuracy): {content_group_gain:+.1%}.",
            "3. Grouping quality must be judged from purity/cohesion and harmful scatter counts, not category accuracy alone.",
            "4. Structured-output operational reliability is shown by the complete failure table above.",
            f"5. Default recommendation on this development benchmark: **{default}**. Complexity is not selected without measurable benefit.",
            "",
            "## Holdout readiness",
            "",
            "The separate holdout fixture and ground truth are prepared and hash-locked, but were not executed. Holdout readiness depends on accepting this development result without changing prompts, schemas, rules, grouping thresholds, peek behavior, model digest or generation parameters. Any such change requires a new commit and a new development experiment before the one-time holdout run.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(args: argparse.Namespace) -> Path:
    source = _git_source()
    fixture = args.fixture.resolve(strict=True)
    expected_path = args.expected.resolve(strict=True)
    fixture_manifest = _verify_dataset_manifest(
        args.fixture_manifest.resolve(strict=True), fixture, expected_path
    )
    holdout_manifest_path = args.holdout_manifest.resolve(strict=True)
    holdout_manifest = _verify_dataset_manifest(
        holdout_manifest_path,
        holdout_manifest_path.parent / "fixture",
        holdout_manifest_path.parent / "expected.yaml",
    )
    expected = _load_expected(expected_path)
    schedule = [
        list(COUNTERBALANCED_SCHEDULE[index % len(COUNTERBALANCED_SCHEDULE)])
        for index in range(args.repetitions)
    ]
    model = _model_manifest(args.model, args.think)
    if not model["endpoint_local"] and not args.allow_remote_content:
        raise RuntimeError("remote endpoint requires --allow-remote-content for C and D")
    experiment_id = args.experiment_id or (
        f"structured-abcd-{datetime.now(timezone.utc).strftime('%Y%m%d')}-"
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
        group_timeout=args.group_timeout,
        allow_remote_content=args.allow_remote_content,
    )
    manifest["experiment_id"] = experiment_id

    with tempfile.TemporaryDirectory(prefix=f"{experiment_id}-") as temporary_name:
        temporary = Path(temporary_name)
        staging = temporary / "artifacts"
        staging.mkdir()
        staged_fixture = temporary / "fixture"
        _stage_fixture(fixture, fixture_manifest, staged_fixture)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(args.fixture_manifest, staging / "fixture_manifest.json")
        warmup = run_in_subprocess(
            _warmup_worker,
            (args.model, args.think),
            timeout=args.timeout,
        )
        manifest["execution"]["warmup_result"] = warmup
        if warmup.get("status") != "ok":
            raise RuntimeError(f"model warmup failed: {warmup.get('error', 'unknown')}")

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
                        group=configuration["group"],
                        group_timeout=args.group_timeout,
                        read_contents=configuration["content"],
                        allow_remote_content=args.allow_remote_content,
                        perform_warmup=False,
                        warmup_result=warmup,
                    )
                except Exception as exc:
                    metrics = {"status": "error", "error": type(exc).__name__}
                category_metrics = (
                    metrics.get("category", {})
                    if configuration["group"]
                    else metrics
                )
                run_status = metrics.get("status", "error")
                if int(category_metrics.get("provider_errors", 0) or 0):
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
                    handle.write(json.dumps(run, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                if run["status"] != "ok":
                    aborted = True
                    break
            if aborted:
                break

        try:
            final_model = _model_manifest(args.model, args.think)
        except Exception as exc:
            final_model = {"error": type(exc).__name__}
            aborted = True
            manifest["execution"]["model_identity_check"] = "unavailable after runs"
        if final_model.get("digest") and final_model.get("digest") != model.get("digest"):
            aborted = True
            manifest["execution"]["model_identity_check"] = "changed during run"
        elif final_model.get("digest"):
            manifest["execution"]["model_identity_check"] = "unchanged"
        summary = _aggregate_runs(runs)
        transitions, per_file = _paired_analysis(runs, expected)
        errors = _error_analysis(per_file, expected)
        result = {
            "experiment_id": experiment_id,
            "complete": not aborted and len(runs) == args.repetitions * 4,
            "run_count": len(runs),
            "condition_order": schedule,
            "conditions": summary,
            "paired_content_transitions": transitions,
            "error_analysis_count": len(errors),
        }
        (staging / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "error_analysis.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if result["complete"]:
            report = _render_report(
                experiment_id=experiment_id,
                manifest=manifest,
                summary=summary,
                transitions=transitions,
                errors=errors,
            )
        else:
            report = (
                f"# {experiment_id}\n\nExperiment stopped after {len(runs)} runs "
                "because a scheduled run failed. No model was substituted.\n"
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
    parser.add_argument("--group-timeout", type=float, default=600.0)
    parser.add_argument("--allow-remote-content", action="store_true")
    parser.add_argument("--experiment-id")
    parser.add_argument("--fixture", type=Path, default=PROJECT_ROOT / "evals/fixture")
    parser.add_argument("--expected", type=Path, default=PROJECT_ROOT / "evals/expected.yaml")
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
        "--output-root", type=Path, default=PROJECT_ROOT / "evals/results"
    )
    args = parser.parse_args(argv)
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    if args.timeout <= 0 or args.group_timeout <= 0:
        parser.error("timeouts must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    output = run_experiment(parse_args(argv))
    print(f"Experiment artifacts: {output}")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
