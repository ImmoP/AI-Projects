"""Run isolated, incremental filename evaluations without an LLM judge."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import queue
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from dotenv import load_dotenv  # noqa: E402
from tidy.agent import (  # noqa: E402
    DEFAULT_MODEL_ID,
    build_group_task,
    build_legacy_group_task,
    build_model,
    endpoint_is_local,
    ensure_content_authorized,
    resolved_model_endpoint,
    safe_endpoint_display,
)
from tidy.classification import build_peek_candidates  # noqa: E402
from tidy.cli import build_combined_plan  # noqa: E402
from tidy.rules import classify_directory, load_rules  # noqa: E402
from tidy.tools import metadata_for_names  # noqa: E402

LOGGER = logging.getLogger(__name__)

REVIEW_DIRECTORY = "_ToReview"
# Predictions that are not a category decision at all: the file was never
# scored, so it belongs in no accuracy numerator or denominator.
UNSCORED_PREDICTIONS = frozenset({"EXCLUDED", "TIMEOUT", "ERROR", "UNASSIGNED"})


@dataclass(frozen=True)
class MemoryMetrics:
    invalid_plan_entries: int
    correction_rounds: int
    steps: int
    latency_seconds: float
    completion_tokens: int
    input_tokens: int = 0
    peek_calls: int = 0
    peek_readable: int = 0


@dataclass(frozen=True)
class GroupingExpectation:
    groups: dict[str, tuple[str, ...]]
    scatter: tuple[str, ...]

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(
            [filename for members in self.groups.values() for filename in members]
            + list(self.scatter)
        )


def _json_objects(text: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text):
        start = text.find("{", position)
        if start < 0:
            return
        try:
            payload, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            position = start + 1
            continue
        position = start + consumed
        if isinstance(payload, dict):
            yield payload


def memory_metrics(memory: Any | None) -> MemoryMetrics:
    """Derive validation, step, latency, and token metrics from AgentMemory."""
    if memory is None:
        return MemoryMetrics(0, 0, 0, 0.0, 0)

    proposal_calls = 0
    peek_calls = 0
    peek_readable = 0
    invalid_entries = 0
    invalid_rounds = 0
    steps = 0
    latency = 0.0
    completion_tokens = 0
    input_tokens = 0
    seen_feedback: set[str] = set()

    for step in getattr(memory, "steps", []):
        if getattr(step, "step_number", None) is not None:
            steps += 1
        timing = getattr(step, "timing", None)
        start = getattr(timing, "start_time", None)
        end = getattr(timing, "end_time", None)
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            latency += max(0.0, end - start)

        usage = getattr(step, "token_usage", None)
        output_tokens = getattr(usage, "output_tokens", 0)
        if isinstance(output_tokens, int):
            completion_tokens += output_tokens
        step_input_tokens = getattr(usage, "input_tokens", 0)
        if isinstance(step_input_tokens, int):
            input_tokens += step_input_tokens

        code_action = getattr(step, "code_action", None)
        if isinstance(code_action, str):
            proposal_calls += len(re.findall(r"\bpropose_plan\s*\(", code_action))
            peek_calls += len(re.findall(r"\bpeek_file\s*\(", code_action))
        for tool_call in getattr(step, "tool_calls", None) or []:
            name = getattr(tool_call, "name", None)
            if name == "propose_plan":
                proposal_calls += 1
            elif name == "peek_file":
                peek_calls += 1

        observations = getattr(step, "observations", None)
        if not isinstance(observations, str):
            continue
        for payload in _json_objects(observations):
            if {"ok", "readable", "status"} <= payload.keys():
                peek_readable += bool(payload.get("readable"))
            if not {"ok", "moves", "errors", "allowed_categories"} <= payload.keys():
                continue
            fingerprint = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            if fingerprint in seen_feedback:
                continue
            seen_feedback.add(fingerprint)
            errors = payload.get("errors", [])
            if payload.get("ok") is False and isinstance(errors, list):
                invalid_rounds += 1
                invalid_entries += len(
                    {
                        error.get("index", -1)
                        for error in errors
                        if isinstance(error, dict)
                    }
                )

    corrections = invalid_rounds
    if corrections == 0 and proposal_calls > 1:
        corrections = proposal_calls - 1
    return MemoryMetrics(
        invalid_entries,
        corrections,
        steps,
        latency,
        completion_tokens,
        input_tokens,
        peek_calls,
        peek_readable,
    )


def _redacted_host(base: str) -> str:
    """Name a private endpoint without publishing its address.

    Reports are committed to a public repository, and a tailnet or LAN address
    identifies someone's machine while adding nothing to reproducibility — the
    digest, quantization and placement already say what answered. Loopback is
    kept verbatim because it identifies nobody.
    """
    # Keep this compatibility helper safe even when passed userinfo, a query,
    # or a public endpoint. Reports need locality, never a dialable address.
    return safe_endpoint_display(api_base=base)


def endpoint_url(model_id: str | None) -> str:
    """Return the address to connect to. Never put this in a report."""
    _, endpoint = resolved_model_endpoint(model_id)
    return endpoint or "provider default"


def resolved_endpoint(model_id: str | None) -> str:
    """Report which server answered, with a private address redacted.

    Kept strictly apart from :func:`endpoint_url`: this value is for reading,
    that one is for connecting. Using this one to open a socket is how the
    redaction first broke every run.
    """
    return safe_endpoint_display(model_id)


def _safe_error(exc: BaseException) -> str:
    """Keep attacker-controlled exception text out of generated reports."""
    return type(exc).__name__


def _ollama_json(host: str, route: str, payload: dict[str, Any] | None = None) -> Any:
    """One small GET/POST against a local Ollama, with the IPv4 fallback."""
    hosts = [host, host.replace("localhost", "127.0.0.1")] if "localhost" in host else [host]
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    for candidate in hosts:
        request = urllib.request.Request(
            f"{candidate}{route}",
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            continue
    return None


def model_provenance(model_id: str | None) -> dict[str, str]:
    """Record which weights answered and how they were loaded.

    The tag alone does not identify a run: the same tag can be served by another
    host, and the same host can load it onto the GPU or entirely into CPU memory.
    Both change the numerics, so both are printed rather than assumed constant.
    """
    load_dotenv()
    resolved = model_id or os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
    provenance: dict[str, str] = {"tag": resolved}
    if not resolved.startswith("ollama"):
        return provenance
    host = endpoint_url(model_id).rstrip("/")
    name = resolved.split("/", 1)[1] if "/" in resolved else resolved
    loaded = _ollama_json(host, "/api/ps") or {}
    for entry in loaded.get("models", []) or []:
        if entry.get("name") != name and entry.get("model") != name:
            continue
        digest = str(entry.get("digest", ""))
        details = entry.get("details", {}) or {}
        vram = entry.get("size_vram")
        provenance.update(
            {
                "digest": digest[:16],
                "quantization": str(details.get("quantization_level", "")),
                "parameter_size": str(details.get("parameter_size", "")),
                "context_length": str(entry.get("context_length", "")),
                "placement": (
                    "CPU only (size_vram=0)"
                    if vram == 0
                    else f"GPU/VRAM {vram} bytes"
                    if isinstance(vram, int)
                    else "unknown"
                ),
            }
        )
        return provenance
    tags = _ollama_json(host, "/api/tags") or {}
    for entry in tags.get("models", []) or []:
        if entry.get("name") == name:
            provenance["digest"] = str(entry.get("digest", ""))[:16]
            provenance["placement"] = "not loaded at report time"
    return provenance


def _provenance_lines(provenance: dict[str, str]) -> list[str]:
    if not provenance:
        return []
    ordered = [
        ("digest", "Model digest"),
        ("quantization", "Quantization"),
        ("parameter_size", "Parameter size"),
        ("context_length", "Context length"),
        ("placement", "Model placement"),
    ]
    return [
        f"- {label}: `{provenance[key]}`" for key, label in ordered if provenance.get(key)
    ]


def ground_truth_fingerprint(path: Path) -> str:
    """Identify the ground truth a report was measured against."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{path.name} sha256:{digest}"


def _archive_existing(output: Path) -> Path | None:
    """Move an existing report aside instead of overwriting it.

    A repeated run with the same flags writes to the same path, which silently
    destroyed one batch of reports before. The previous file is kept under its
    own modification time so the record survives even when the link stays
    stable.
    """
    if not output.exists():
        return None
    stamp = datetime.fromtimestamp(output.stat().st_mtime, tz=timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    archived = output.with_name(f"{output.stem}-superseded-{stamp}{output.suffix}")
    if archived.exists():
        return archived
    output.replace(archived)
    LOGGER.info("previous report kept as %s", archived)
    return archived


def _load_expected(path: Path) -> dict[str, list[str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("expected.yaml must contain a 'files' mapping")
    expected: dict[str, list[str]] = {}
    for name, allowed in files.items():
        if not isinstance(name, str) or not isinstance(allowed, list) or not allowed:
            raise ValueError(f"invalid expected entry for {name!r}")
        expected[name] = [str(category) for category in allowed]
    return expected


def _load_grouping_expected(path: Path) -> GroupingExpectation:
    """Load ground-truth relationships, never destination folder names."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_groups = payload.get("groups")
    raw_scatter = payload.get("scatter", [])
    if not isinstance(raw_groups, dict) or not raw_groups:
        raise ValueError("expected.yaml must contain a non-empty 'groups' mapping")
    if not isinstance(raw_scatter, list):
        raise ValueError("expected.yaml 'scatter' must be a list")

    groups: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for label, raw_files in raw_groups.items():
        if not isinstance(label, str) or not label:
            raise ValueError("group labels must be non-empty strings")
        if (
            not isinstance(raw_files, list)
            or len(raw_files) < 3
            or any(not isinstance(filename, str) or not filename for filename in raw_files)
        ):
            raise ValueError(f"group {label!r} must contain at least three filenames")
        duplicates = seen.intersection(raw_files)
        if duplicates:
            raise ValueError(f"files appear more than once: {', '.join(sorted(duplicates))}")
        seen.update(raw_files)
        groups[label] = tuple(raw_files)

    scatter: list[str] = []
    for filename in raw_scatter:
        if not isinstance(filename, str) or not filename:
            raise ValueError("scatter entries must be non-empty strings")
        if filename in seen:
            raise ValueError(f"file appears in a group and scatter: {filename}")
        if filename in scatter:
            raise ValueError(f"duplicate scatter file: {filename}")
        scatter.append(filename)
    return GroupingExpectation(groups, tuple(scatter))


def _warmup_worker(
    result_queue: multiprocessing.Queue,
    model_id: str | None,
    think: bool | None,
) -> None:
    started = time.perf_counter()
    try:
        model = build_model(model_id, think=think)
        response = model.generate(
            [{"role": "user", "content": [{"type": "text", "text": "Reply only: ready"}]}]
        )
        usage = getattr(response, "token_usage", None)
        result_queue.put(
            {
                "status": "ok",
                "latency_seconds": time.perf_counter() - started,
                "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
            }
        )
    except Exception as exc:  # isolated in a child; converted to report data
        result_queue.put(
            {
                "status": "error",
                "latency_seconds": time.perf_counter() - started,
                "error": _safe_error(exc),
            }
        )


def _plan_worker(
    result_queue: multiprocessing.Queue,
    fixture: str,
    model_id: str | None,
    think: bool | None,
    use_agent: bool,
    group: bool,
    read_contents: bool,
    allow_remote_content: bool,
    metadata_control: bool,
    classification_mode: str = "single_pass",
) -> None:
    """Build one complete plan so clustering is evaluated with global context."""
    started = time.perf_counter()
    try:
        initial_rule_moves, initial_unresolved = classify_directory(
            Path(fixture), load_rules()
        )
        bundle = build_combined_plan(
            fixture,
            model_id=model_id,
            think=think,
            use_agent=use_agent,
            group=group,
            read_contents=read_contents,
            allow_remote_content=allow_remote_content,
            metadata_control=metadata_control,
            classification_mode=classification_mode,
        )
        group_metrics = memory_metrics(
            getattr(bundle.group_agent, "memory", None)
        )
        classification_metrics = bundle.classification_metrics
        classifier_active = bundle.classifier is not None
        group_entries = bundle.grouping.entries if bundle.grouping else ()
        grouped_sources = set(bundle.grouping.grouped_files) if bundle.grouping else set()
        rule_resolved_sources = {move["source"] for move in initial_rule_moves}
        unresolved_sources = set(initial_unresolved)
        result_queue.put(
            {
                "status": "ok",
                "invalid_folder_names": (
                    bundle.grouping.invalid_folder_names if bundle.grouping else 0
                ),
                "grouped_sources": sorted(bundle.grouping.grouped_files)
                if bundle.grouping
                else [],
                # The executor drops clusters below the minimum size and rejects
                # unsafe ones. Keeping the proposal separate from the accepted
                # result is what lets the report tell a model's clustering
                # decision apart from the executor's filtering.
                "group_folders": [
                    entry.folder_name for entry in group_entries if entry.status == "accepted"
                ],
                "proposed_group_members": sorted(
                    {filename for entry in group_entries for filename in entry.files}
                ),
                "discarded_group_members": sorted(
                    {
                        filename
                        for entry in group_entries
                        if entry.status == "discarded"
                        for filename in entry.files
                    }
                ),
                "groups_proposed": len(group_entries),
                "groups_accepted": sum(
                    entry.status == "accepted" for entry in group_entries
                ),
                "groups_discarded": sum(
                    entry.status == "discarded" for entry in group_entries
                ),
                "groups_rejected": sum(
                    entry.status == "rejected" for entry in group_entries
                ),
                "grouped_rule_resolved_sources": sorted(
                    grouped_sources & rule_resolved_sources
                ),
                "grouped_unresolved_sources": sorted(
                    grouped_sources & unresolved_sources
                ),
                "deterministic_rule_sources": sorted(
                    move["source"]
                    for move in bundle.moves
                    if move.get("origin") == "rule"
                ),
                "classification_source_count": bundle.unresolved_count,
                "peeked_sources": list(bundle.peeked_sources),
                "peek_requested_sources": list(bundle.peek_requested_sources),
                "moves": [
                    {
                        key: move[key]
                        for key in ("source", "destination", "origin", "fallback")
                        if key in move
                    }
                    for move in bundle.moves
                ],
                "omitted_sources": list(bundle.omitted_sources),
                "invalid_sources": list(bundle.invalid_sources),
                "unproposed_sources": list(bundle.unproposed_sources),
                "class_latency_seconds": float(
                    classification_metrics.get("latency_seconds", 0) or 0
                ),
                "class_input_tokens": int(
                    classification_metrics.get("input_tokens", 0) or 0
                ),
                "class_completion_tokens": int(
                    classification_metrics.get("completion_tokens", 0) or 0
                ),
                "class_steps": int(
                    classification_metrics.get("classification_requests", 0) or 0
                ),
                **bundle.peek_metrics,
                **classification_metrics,
                "class_invalid_plan_entries": int(
                    classification_metrics.get("schema_validation_failures", 0) or 0
                ),
                "class_correction_rounds": 0,
                "agent_runs": int(bundle.group_agent is not None) + int(classifier_active),
                "latency_seconds": group_metrics.latency_seconds
                + float(classification_metrics.get("latency_seconds", 0) or 0)
                or (time.perf_counter() - started),
                "steps": group_metrics.steps
                + int(classification_metrics.get("classification_requests", 0) or 0),
                "completion_tokens": group_metrics.completion_tokens
                + int(classification_metrics.get("completion_tokens", 0) or 0),
                "input_tokens": group_metrics.input_tokens
                + int(classification_metrics.get("input_tokens", 0) or 0),
                "group_latency_seconds": group_metrics.latency_seconds,
                "group_input_tokens": group_metrics.input_tokens,
                "group_completion_tokens": group_metrics.completion_tokens,
                "group_steps": group_metrics.steps,
            }
        )
    except Exception as exc:  # isolated in a child; converted to report data
        result_queue.put(
            {
                "status": "error",
                "moves": [],
                "invalid_folder_names": 0,
                "grouped_sources": [],
                "group_folders": [],
                "proposed_group_members": [],
                "discarded_group_members": [],
                "omitted_sources": [],
                "invalid_sources": [],
                "unproposed_sources": [],
                "groups_proposed": 0,
                "groups_accepted": 0,
                "groups_discarded": 0,
                "groups_rejected": 0,
                "grouped_rule_resolved_sources": [],
                "grouped_unresolved_sources": [],
                "deterministic_rule_sources": [],
                "classification_source_count": 0,
                "peeked_sources": [],
                "peek_requested_sources": [],
                "agent_runs": 0,
                "latency_seconds": time.perf_counter() - started,
                "error": _safe_error(exc),
            }
        )


def run_in_subprocess(
    worker: Callable[..., None],
    args: tuple[Any, ...],
    *,
    timeout: float,
) -> dict[str, Any]:
    """Run one worker with a hard wall-clock timeout and serializable result."""
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=worker, args=(result_queue, *args))
    started = time.perf_counter()
    process.start()
    process.join(timeout)
    elapsed = time.perf_counter() - started
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():  # pragma: no cover - defensive OS fallback
            process.kill()
            process.join(5)
        result_queue.close()
        return {"status": "timeout", "latency_seconds": elapsed}

    try:
        payload = result_queue.get(timeout=1)
    except queue.Empty:
        payload = {
            "status": "error",
            "error": f"worker exited with code {process.exitcode} without a result",
        }
    finally:
        result_queue.close()
    payload.setdefault("latency_seconds", elapsed)
    return payload


def _delayed_result_worker(
    result_queue: multiprocessing.Queue,
    delay: float,
) -> None:
    """Small deterministic worker used to regression-test timeout isolation."""
    time.sleep(delay)
    result_queue.put({"status": "ok"})


def _aggregate(cases: list[dict[str, Any]], *, totals: dict[str, int]) -> dict[str, Any]:
    correct = sum(case["correct"] for case in cases)
    unknown_cases = [case for case in cases if case["unresolved"]]
    unknown_correct = sum(case["correct"] for case in unknown_cases)
    agent_attempts = [case for case in unknown_cases if case["mode"] == "agent"]
    assigned = [case for case in cases if case["predicted"] not in {"EXCLUDED", "TIMEOUT", "ERROR"}]

    def average(field: str) -> float:
        if not agent_attempts:
            return 0.0
        return sum(float(case.get(field, 0)) for case in agent_attempts) / len(agent_attempts)

    # `unknown_accuracy` accepts `_ToReview` alongside a content category for
    # most ambiguous files, so it rewards not deciding just as much as deciding
    # correctly. Splitting the two is what makes the agent comparable with the
    # rules-only baseline, which never decides and still scores well.
    decided = [
        case
        for case in unknown_cases
        if case["predicted"] != REVIEW_DIRECTORY
        and case["predicted"] not in UNSCORED_PREDICTIONS
    ]
    decided_correct = sum(case["correct"] for case in decided)
    strict = [case for case in unknown_cases if len(case["allowed"]) == 1]
    strict_correct = sum(case["correct"] for case in strict)

    return {
        "overall_accuracy": correct / totals["all"] if totals["all"] else 0.0,
        "overall_correct": correct,
        "overall_total": totals["all"],
        "unknown_accuracy": unknown_correct / totals["unknown"] if totals["unknown"] else 0.0,
        "unknown_correct": unknown_correct,
        "unknown_total": totals["unknown"],
        "decision_rate": len(decided) / totals["unknown"] if totals["unknown"] else 0.0,
        "decided_count": len(decided),
        "decided_accuracy": decided_correct / len(decided) if decided else 0.0,
        "decided_correct": decided_correct,
        "abstained_count": sum(
            case["predicted"] == REVIEW_DIRECTORY for case in unknown_cases
        ),
        "strict_total": len(strict),
        "strict_correct": strict_correct,
        "strict_accuracy": strict_correct / len(strict) if strict else 0.0,
        "review_rate": (
            sum(case["predicted"] == REVIEW_DIRECTORY for case in assigned) / len(assigned)
            if assigned
            else 0.0
        ),
        "review_count": sum(case["predicted"] == REVIEW_DIRECTORY for case in assigned),
        "assigned_count": len(assigned),
        # An omitted file is one the model never named at all; an invalid
        # fallback is one it named but assigned unusably. Both end in
        # ``_ToReview/`` and are therefore reported as separate columns.
        "omitted_count": sum(case.get("fallback") == "omitted" for case in cases),
        "omitted_files": [
            case.get("filename", "") for case in cases if case.get("fallback") == "omitted"
        ],
        "invalid_fallback_count": sum(
            case.get("fallback") == "invalid" for case in cases
        ),
        "no_proposal_count": sum(
            case.get("fallback") == "no-proposal" for case in cases
        ),
        "invalid_plan_entries": sum(case.get("invalid_plan_entries", 0) for case in agent_attempts),
        "correction_rounds": sum(case.get("correction_rounds", 0) for case in agent_attempts),
        "average_steps": average("steps"),
        "average_latency_seconds": average("latency_seconds"),
        "average_completion_tokens": average("completion_tokens"),
        "completion_tokens": sum(case.get("completion_tokens", 0) for case in agent_attempts),
        "timeouts": sum(case["status"] == "timeout" for case in cases),
        "errors": sum(case["status"] == "error" for case in cases),
        "completed": len(cases),
        "cases": cases,
    }


def _category_metrics(
    expected_categories: dict[str, list[str]],
    unresolved_names: list[str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Score fixed-category placement for every file that was not grouped.

    Grouped files are excluded rather than counted wrong: a semantic group folder
    is a deliberate alternative to a fixed category, not a misclassification.
    """
    moves = payload.get("moves", []) if payload.get("status") == "ok" else []
    destinations = {
        move["source"]: Path(move["destination"]).parts[0]
        for move in moves
        if isinstance(move, dict)
        and isinstance(move.get("source"), str)
        and isinstance(move.get("destination"), str)
        and Path(move["destination"]).parts
    }
    grouped = set(payload.get("grouped_sources", []) or [])
    omitted = set(payload.get("omitted_sources", []) or [])
    invalid = set(payload.get("invalid_sources", []) or [])
    unproposed = set(payload.get("unproposed_sources", []) or [])
    unresolved = set(unresolved_names)

    cases: list[dict[str, Any]] = []
    for filename, allowed in expected_categories.items():
        if filename in grouped:
            continue
        predicted = destinations.get(filename, "UNASSIGNED")
        is_unresolved = filename in unresolved
        cases.append(
            {
                "filename": filename,
                "allowed": allowed,
                "predicted": predicted,
                "correct": predicted in allowed,
                "unresolved": is_unresolved,
                "mode": "agent" if is_unresolved else "rule",
                "status": payload.get("status", "error"),
                "fallback": (
                    "omitted"
                    if filename in omitted
                    else "invalid"
                    if filename in invalid
                    else "no-proposal" if filename in unproposed else ""
                ),
            }
        )

    # Every file counted by `grouped_excluded` is listed here with the folder it
    # went to. The count and the table are the same quantity, so a report can no
    # longer claim a number its own tables contradict.
    excluded_cases = [
        {
            "filename": filename,
            "destination": destinations.get(filename, "UNASSIGNED"),
            "unresolved": filename in unresolved,
        }
        for filename in expected_categories
        if filename in grouped
    ]

    evaluated_unresolved = [case for case in cases if case["unresolved"]]
    metrics = _aggregate(
        cases,
        totals={"all": len(cases), "unknown": len(evaluated_unresolved)},
    )
    metrics.update(
        {
            "status": payload.get("status", "error"),
            "error": payload.get("error", ""),
            "agent_runs": int(payload.get("agent_runs", 0) or 0),
            # Scored files that were grouped. Files in a group folder that carry
            # no category ground truth are not part of this number; the
            # clustering section reports those separately.
            "grouped_excluded": len(excluded_cases),
            "excluded_cases": excluded_cases,
            "unresolved_total_before_grouping": len(unresolved),
            "invalid_plan_entries": int(payload.get("class_invalid_plan_entries", 0) or 0),
            "correction_rounds": int(payload.get("class_correction_rounds", 0) or 0),
            "class_steps": int(payload.get("class_steps", 0) or 0),
            "class_latency_seconds": float(payload.get("class_latency_seconds", 0) or 0),
            "class_input_tokens": int(payload.get("class_input_tokens", 0) or 0),
            "class_completion_tokens": int(
                payload.get("class_completion_tokens", 0) or 0
            ),
            "peek_calls": int(payload.get("peek_calls", 0) or 0),
            "peek_readable": int(payload.get("peek_readable", 0) or 0),
            "peek_nonempty": int(payload.get("peek_nonempty", 0) or 0),
            "peek_unique_files": int(payload.get("peek_unique_files", 0) or 0),
            "peek_source_bytes_considered": int(
                payload.get("peek_source_bytes_considered", 0) or 0
            ),
            "peek_bytes_read": int(payload.get("peek_bytes_read", 0) or 0),
            "peek_chars_returned": int(
                payload.get("peek_chars_returned", 0) or 0
            ),
            "peek_parser_skipped": int(
                payload.get("peek_parser_skipped", 0) or 0
            ),
            "peek_parser_timeouts": int(
                payload.get("peek_parser_timeouts", 0) or 0
            ),
            "peek_parser_errors": int(
                payload.get("peek_parser_errors", 0) or 0
            ),
            "peek_file_metrics": dict(payload.get("peek_file_metrics", {}) or {}),
            "classification_backend": payload.get(
                "classification_backend", "rules_only"
            ),
            "structured_output_mode": payload.get(
                "structured_output_mode", "none"
            ),
            "classification_requests": int(
                payload.get("classification_requests", 0) or 0
            ),
            "peek_phase_requests": int(payload.get("peek_phase_requests", 0) or 0),
            "final_classification_requests": int(
                payload.get("final_classification_requests", 0) or 0
            ),
            "parse_failures": int(payload.get("parse_failures", 0) or 0),
            "schema_validation_failures": int(
                payload.get("schema_validation_failures", 0) or 0
            ),
            "fallback_to_review_count": int(
                payload.get("fallback_to_review_count", 0) or 0
            ),
            "peek_requests_model": int(payload.get("peek_requests_model", 0) or 0),
            "peek_requests_authorized": int(
                payload.get("peek_requests_authorized", 0) or 0
            ),
            "peek_requests_rejected": int(
                payload.get("peek_requests_rejected", 0) or 0
            ),
            "peek_requests_deduplicated": int(
                payload.get("peek_requests_deduplicated", 0) or 0
            ),
            "peek_requests_control_selected": int(
                payload.get("peek_requests_control_selected", 0) or 0
            ),
            "peek_requests_unique": int(
                payload.get("peek_requests_unique", 0) or 0
            ),
            "peek_candidates_total": int(
                payload.get("peek_candidates_total", 0) or 0
            ),
            "peek_candidates_eligible": int(
                payload.get("peek_candidates_eligible", 0) or 0
            ),
            "peek_candidates_empty": int(
                payload.get("peek_candidates_empty", 0) or 0
            ),
            "peek_candidates_oversized": int(
                payload.get("peek_candidates_oversized", 0) or 0
            ),
            "peek_candidates_unsupported": int(
                payload.get("peek_candidates_unsupported", 0) or 0
            ),
            "content_unavailable": int(
                payload.get("content_unavailable", 0) or 0
            ),
            "provider_errors": int(payload.get("provider_errors", 0) or 0),
            "incomplete_responses": int(
                payload.get("incomplete_responses", 0) or 0
            ),
            "duplicate_source_responses": int(
                payload.get("duplicate_source_responses", 0) or 0
            ),
            "invented_source_responses": int(
                payload.get("invented_source_responses", 0) or 0
            ),
            "invented_category_responses": int(
                payload.get("invented_category_responses", 0) or 0
            ),
            "native_schema_responses": int(
                payload.get("native_schema_responses", 0) or 0
            ),
            "json_object_responses": int(
                payload.get("json_object_responses", 0) or 0
            ),
            "plain_json_responses": int(
                payload.get("plain_json_responses", 0) or 0
            ),
            "peek_phase_latency_seconds": float(
                payload.get("peek_phase_latency_seconds", 0) or 0
            ),
            "final_classification_latency_seconds": float(
                payload.get("final_classification_latency_seconds", 0) or 0
            ),
            "content_processing_latency_seconds": float(
                payload.get("content_processing_latency_seconds", 0) or 0
            ),
            "peek_phase_input_tokens": int(
                payload.get("peek_phase_input_tokens", 0) or 0
            ),
            "peek_phase_completion_tokens": int(
                payload.get("peek_phase_completion_tokens", 0) or 0
            ),
            "final_classification_input_tokens": int(
                payload.get("final_classification_input_tokens", 0) or 0
            ),
            "final_classification_completion_tokens": int(
                payload.get("final_classification_completion_tokens", 0) or 0
            ),
            "deterministic_rule_sources": list(
                payload.get("deterministic_rule_sources", []) or []
            ),
            "classification_source_count": int(
                payload.get("classification_source_count", 0) or 0
            ),
            "peeked_sources": list(payload.get("peeked_sources", []) or []),
            "peek_requested_sources": list(
                payload.get("peek_requested_sources", []) or []
            ),
            "all_destinations": destinations,
            "grouped_sources": sorted(grouped),
        }
    )
    return metrics


def _mode_header(
    *,
    group: bool,
    read_contents: bool,
    metadata_control: bool = False,
) -> list[str]:
    """Name the run mode and its metric family unambiguously.

    Two reports are only comparable when these three lines are identical; the
    metric families themselves differ between modes.
    """
    axes = (
        f"`--{'group' if group else 'no-group'}`, content reading "
        f"`{read_contents}`, metadata control `{metadata_control}`"
    )
    if group:
        return [
            "- Run mode: **grouping** (" + axes + ")",
            "- Metric family: **clustering metrics + category accuracy for "
            "ungrouped files**",
            "- Comparable only with other reports whose run mode line is identical.",
        ]
    return [
        "- Run mode: **fixed categories only** (" + axes + ")",
        "- Metric family: **category accuracy only**",
        "- No grouping ran, so no clustering metric is defined in this mode; "
        "clustering rows are omitted rather than reported as 0 or N/A.",
        "- Comparable only with other reports whose run mode line is identical.",
    ]


def _run_header(
    *,
    title: str,
    model: str,
    think: bool | None,
    timeout: float,
    group_timeout: float,
    group: bool,
    read_contents: bool,
    metadata_control: bool = False,
    status: str,
    warmup: dict[str, Any] | None,
    ground_truth: str = "",
    endpoint: str = "",
    provenance: dict[str, str] | None = None,
) -> list[str]:
    think_label = "model default" if think is None else str(think)
    lines = [
        f"# {title}",
        "",
        f"- Status: **{status}**",
        *_mode_header(
            group=group,
            read_contents=read_contents,
            metadata_control=metadata_control,
        ),
        f"- Model: `{model}`",
        f"- Think: `{think_label}`",
        f"- Standard run timeout: {timeout:.1f} s",
    ]
    if group:
        lines.append(f"- Clustering run timeout: {group_timeout:.1f} s")
    if ground_truth:
        lines.append(f"- Ground truth: `{ground_truth}`")
    if endpoint:
        lines.append(f"- Endpoint: `{endpoint}`")
    lines.extend(_provenance_lines(provenance or {}))
    lines.append(f"- Evaluated: {datetime.now(timezone.utc).isoformat()}")
    if warmup:
        lines.append(
            f"- Warm-up: {warmup['status']} in {warmup.get('latency_seconds', 0):.3f} s"
        )
    return lines


def _escaped(name: str) -> str:
    return name.replace("|", "\\|")


def _rate(share: float, correct: int, total: int) -> str:
    """Format a rate, but never print 0.0% for an empty denominator."""
    if not total:
        return f"n/a (0/{total})"
    return f"{share:.1%} ({correct}/{total})"


def _content_rows(metrics: dict[str, Any]) -> list[str]:
    """Content-free reach/resource telemetry, printed only when it ran."""
    if not metrics.get("read_contents"):
        return []
    authorized = int(metrics.get("peek_requests_authorized", 0) or 0)
    nonempty = int(metrics.get("peek_nonempty", 0) or 0)
    informative_rate = nonempty / authorized if authorized else 0.0
    return [
        f"| Phase-1 candidates eligible / unresolved | "
        f"{metrics.get('peek_candidates_eligible', 0)}/"
        f"{metrics.get('peek_candidates_total', 0)} |",
        f"| Empty candidates filtered before Phase 1 | "
        f"{metrics.get('peek_candidates_empty', 0)} |",
        f"| Oversized/unsupported candidates filtered before Phase 1 | "
        f"{metrics.get('peek_candidates_oversized', 0)}/"
        f"{metrics.get('peek_candidates_unsupported', 0)} |",
        f"| Peek filenames requested by model | {metrics.get('peek_requests_model', 0)} |",
        f"| Unique valid peek filenames requested | {metrics.get('peek_requests_unique', 0)} |",
        f"| Unique peek requests authorized by application | {metrics.get('peek_requests_authorized', 0)} |",
        f"| Peek requests rejected | {metrics.get('peek_requests_rejected', 0)} |",
        f"| Duplicate peek requests deduplicated | {metrics.get('peek_requests_deduplicated', 0)} |",
        f"| Content peeks attempted | {metrics.get('peek_calls', 0)} |",
        f"| Unique authorized files peeked | {metrics.get('peek_unique_files', 0)} |",
        f"| Peeks that returned text | {metrics.get('peek_readable', 0)} |",
        f"| Peeks that returned non-empty text | {nonempty} |",
        f"| Informative peek rate (non-empty / authorized) | "
        f"{informative_rate:.1%} ({nonempty}/{authorized}) |",
        f"| Authorized peeks with unavailable content | {metrics.get('content_unavailable', 0)} |",
        f"| Source bytes considered | {metrics.get('peek_source_bytes_considered', 0)} |",
        f"| Bytes actually read | {metrics.get('peek_bytes_read', 0)} |",
        f"| Characters returned to model | {metrics.get('peek_chars_returned', 0)} |",
        f"| Peek-request phase latency | {metrics.get('peek_phase_latency_seconds', 0):.3f} s |",
        f"| Content extraction/processing latency | {metrics.get('content_processing_latency_seconds', 0):.3f} s |",
        f"| Final-classification phase latency | {metrics.get('final_classification_latency_seconds', 0):.3f} s |",
        f"| Parser inputs skipped | {metrics.get('peek_parser_skipped', 0)} |",
        f"| Parser timeouts | {metrics.get('peek_parser_timeouts', 0)} |",
        f"| Parser errors | {metrics.get('peek_parser_errors', 0)} |",
        f"| Model endpoint locality | "
        f"{'local' if metrics.get('endpoint_local') else 'remote'} |",
        f"| Unresolved files peek_file may open | "
        f"{metrics.get('peek_eligible', 0)}/{metrics['unknown_total']} |",
    ]


def _category_metric_rows(metrics: dict[str, Any], *, scope: str) -> list[str]:
    """Render the category metric rows shared by both run modes."""
    return [
        "| Metric | Result |",
        "|---|---:|",
        f"| Category accuracy, {scope} eligible files | {metrics['overall_accuracy']:.1%} ({metrics['overall_correct']}/{metrics['overall_total']}) |",
        f"| Category accuracy, {scope} unresolved files (`_ToReview` accepted) | {_rate(metrics['unknown_accuracy'], metrics['unknown_correct'], metrics['unknown_total'])} |",
        f"| Decision rate, unresolved files | {_rate(metrics['decision_rate'], metrics['decided_count'], metrics['unknown_total'])} |",
        f"| Accuracy on decided files only | {_rate(metrics['decided_accuracy'], metrics['decided_correct'], metrics['decided_count'])} |",
        f"| Abstentions (`_ToReview`), unresolved files | {metrics['abstained_count']}/{metrics['unknown_total']} |",
        f"| Accuracy, strict subset (exactly one accepted category) | {_rate(metrics['strict_accuracy'], metrics['strict_correct'], metrics['strict_total'])} |",
        f"| Files omitted by the model | {metrics['omitted_count']}/{metrics['unknown_total']} |",
        f"| Invalid-assignment fallbacks | {metrics['invalid_fallback_count']}/{metrics['unknown_total']} |",
        f"| Files without any usable proposal | {metrics['no_proposal_count']}/{metrics['unknown_total']} |",
        f"| `_ToReview/` rate, all eligible files | {metrics['review_rate']:.1%} ({metrics['review_count']}/{metrics['assigned_count']}) |",
        f"| Invalid plan entries | {metrics['invalid_plan_entries']} |",
        f"| Correction rounds | {metrics['correction_rounds']} |",
        f"| Classification steps | {metrics['class_steps']} |",
        f"| Classification latency | {metrics['class_latency_seconds']:.3f} s |",
        f"| Classification input tokens | {metrics['class_input_tokens']} |",
        f"| Classification completion tokens | {metrics['class_completion_tokens']} |",
        f"| Classification backend | {metrics.get('classification_backend', 'rules_only')} |",
        f"| Structured output mode | {metrics.get('structured_output_mode', 'none')} |",
        f"| Classification model requests | {metrics.get('classification_requests', 0)} |",
        f"| Peek-phase model requests | {metrics.get('peek_phase_requests', 0)} |",
        f"| Final-classification model requests | {metrics.get('final_classification_requests', 0)} |",
        f"| Strict JSON parse failures | {metrics.get('parse_failures', 0)} |",
        f"| Schema validation failures | {metrics.get('schema_validation_failures', 0)} |",
        f"| Provider failures | {metrics.get('provider_errors', 0)} |",
        f"| Incomplete structured responses | {metrics.get('incomplete_responses', 0)} |",
        f"| Duplicate-source responses | {metrics.get('duplicate_source_responses', 0)} |",
        f"| Invented-source responses | {metrics.get('invented_source_responses', 0)} |",
        f"| Invented-category responses | {metrics.get('invented_category_responses', 0)} |",
        f"| Native-schema responses | {metrics.get('native_schema_responses', 0)} |",
        f"| JSON-object responses | {metrics.get('json_object_responses', 0)} |",
        f"| Strict plain-JSON responses | {metrics.get('plain_json_responses', 0)} |",
        f"| Structured fallbacks to `_ToReview` | {metrics.get('fallback_to_review_count', 0)} |",
        *_content_rows(metrics),
    ]


def _rubric_note() -> list[str]:
    """State plainly which row rewards abstention and which one does not."""
    return [
        "",
        "`_ToReview` is an accepted answer for most ambiguous filenames, so the "
        "unresolved-accuracy row credits abstention exactly as much as a correct "
        "decision — a model that always abstained would score well on it. "
        "**Decision rate** and **accuracy on decided files** separate the two and "
        "are the rows to compare against the rules-only baseline, which decides "
        "nothing.",
    ]


def _assignment_table(cases: list[dict[str, Any]]) -> list[str]:
    """Render one row per scored file so every rate can be recounted by hand."""
    lines = [
        "",
        "## Assignments",
        "",
        "| File | Mode | Predicted | Accepted | Correct | Fallback |",
        "|---|---|---|---|---|---|",
    ]
    for case in cases:
        accepted = ", ".join(f"`{item}`" for item in case["allowed"])
        result = "PASS" if case["correct"] else "FAIL"
        lines.append(
            f"| `{_escaped(case['filename'])}` | {case['mode']} | `{case['predicted']}` | "
            f"{accepted} | {result} | {case['fallback'] or '—'} |"
        )
    return lines


def render_category_markdown(
    metrics: dict[str, Any],
    *,
    model: str,
    think: bool | None,
    timeout: float,
    group_timeout: float,
    read_contents: bool,
    warmup: dict[str, Any] | None,
    metadata_control: bool = False,
    ground_truth: str = "",
    endpoint: str = "",
    provenance: dict[str, str] | None = None,
) -> str:
    """Render the no-grouping report: category accuracy, no clustering metrics."""
    lines = _run_header(
        title="tidy-agent evaluation — category mode",
        model=model,
        think=think,
        timeout=timeout,
        group_timeout=group_timeout,
        group=False,
        read_contents=read_contents,
        metadata_control=metadata_control,
        status=metrics["status"],
        warmup=warmup,
        ground_truth=ground_truth,
        endpoint=endpoint,
        provenance=provenance,
    )
    lines.append("- Judge: deterministic expected-category comparison (no LLM judge)")
    lines.append("")
    lines.extend(_category_metric_rows(metrics, scope="all"))
    lines.append(f"| Agent runs | {metrics['agent_runs']} |")
    lines.extend(_rubric_note())
    if metrics["omitted_files"]:
        lines.extend(
            [
                "",
                "Omitted by the model (deterministic `_ToReview/` fallback): "
                + ", ".join(f"`{name}`" for name in metrics["omitted_files"]),
            ]
        )
    lines.extend(_assignment_table(metrics["cases"]))
    if metrics.get("error"):
        lines.extend(["", f"Error: `{metrics['error']}`"])
    return "\n".join(lines) + "\n"


def _purity_numerator(
    clusters: dict[str, list[str]],
    labels: dict[str, str],
) -> int:
    """Sum the largest ground-truth label per predicted cluster."""
    numerator = 0
    for members in clusters.values():
        counts: dict[str, int] = {}
        for filename in members:
            label = labels[filename]
            counts[label] = counts.get(label, 0) + 1
        if counts:
            numerator += max(counts.values())
    return numerator


def _grouping_metrics(
    expected: GroupingExpectation,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Score clustering decisions, keeping them apart from category placement.

    A file that ends up in ``Documents/`` was not clustered: the extension rules
    put it there. Only an accepted semantic group folder is a clustering
    decision, so every rate below that claims to measure clustering counts group
    folders only. The destination-level figures are reported too, under names
    that say what they include, because co-location in a fixed category folder
    is a real property of the end state — just not a clustering error.
    """
    ok = payload.get("status") == "ok"
    moves = payload.get("moves", []) if ok else []
    destinations = {
        move["source"]: Path(move["destination"]).parts[0]
        for move in moves
        if isinstance(move, dict)
        and isinstance(move.get("source"), str)
        and isinstance(move.get("destination"), str)
        and Path(move["destination"]).parts
    }
    grouped_sources = set(payload.get("grouped_sources", []) or []) if ok else set()
    proposed_members = set(payload.get("proposed_group_members", []) or []) if ok else set()
    discarded_members = (
        set(payload.get("discarded_group_members", []) or []) if ok else set()
    )
    labels = {
        filename: label
        for label, filenames in expected.groups.items()
        for filename in filenames
    }
    labels.update({filename: f"scatter:{filename}" for filename in expected.scatter})
    predicted_clusters: dict[str, list[str]] = {}
    group_clusters: dict[str, list[str]] = {}
    for filename in expected.files:
        destination = destinations.get(filename, f"UNASSIGNED:{filename}")
        predicted_clusters.setdefault(destination, []).append(filename)
        if filename in grouped_sources:
            group_clusters.setdefault(destination, []).append(filename)

    destination_purity_numerator = _purity_numerator(predicted_clusters, labels) if ok else 0
    destination_purity = (
        destination_purity_numerator / len(expected.files) if expected.files else 0.0
    )
    grouped_evaluated = [name for name in expected.files if name in grouped_sources]
    group_purity_numerator = _purity_numerator(group_clusters, labels) if ok else 0
    group_purity = (
        group_purity_numerator / len(grouped_evaluated) if grouped_evaluated else 0.0
    )

    cohesive_groups: dict[str, bool] = {}
    for label, members in expected.groups.items():
        folders = {destinations.get(filename) for filename in members}
        cohesive_groups[label] = (
            None not in folders
            and len(folders) == 1
            and all(filename in grouped_sources for filename in members)
        )

    # Three different things, three different names. Only the first is a
    # clustering error; the second is what the executor's minimum-size filter
    # caught; the third is a consequence of the extension rules.
    scatter_in_group = [name for name in expected.scatter if name in grouped_sources]
    scatter_in_proposed_group = [
        name for name in expected.scatter if name in proposed_members
    ]
    scatter_in_discarded_group = [
        name for name in expected.scatter if name in discarded_members
    ]
    scatter_sharing_category = [
        filename
        for filename in expected.scatter
        if filename not in grouped_sources
        and len(predicted_clusters.get(destinations.get(filename, ""), [])) > 1
    ]
    cases = [
        {
            "filename": filename,
            "expected": labels[filename],
            "destination": destinations.get(filename, "UNASSIGNED"),
            "placed_by": "group" if filename in grouped_sources else "category",
        }
        for filename in expected.files
    ]
    agent_runs = int(payload.get("agent_runs", 0) or 0)
    return {
        "status": payload.get("status", "error"),
        "clustering_purity": group_purity,
        "purity_correct": group_purity_numerator,
        "grouped_evaluated": len(grouped_evaluated),
        "destination_purity": destination_purity,
        "destination_purity_correct": destination_purity_numerator,
        "total_files": len(expected.files),
        "group_members_total": len(expected.files) - len(expected.scatter),
        "grouped_without_ground_truth": max(
            0, len(grouped_sources) - len(grouped_evaluated)
        ),
        "grouped_total": len(grouped_sources),
        "cohesive_groups": sum(cohesive_groups.values()),
        "total_groups": len(cohesive_groups),
        "group_cohesion": (
            sum(cohesive_groups.values()) / len(cohesive_groups)
            if cohesive_groups
            else 0.0
        ),
        "scatter_in_group": len(scatter_in_group),
        "scatter_in_group_files": scatter_in_group,
        "scatter_in_proposed_group": len(scatter_in_proposed_group),
        "scatter_in_proposed_group_files": scatter_in_proposed_group,
        "scatter_in_discarded_group": len(scatter_in_discarded_group),
        "scatter_sharing_category": len(scatter_sharing_category),
        "scatter_sharing_category_files": scatter_sharing_category,
        "scatter_total": len(expected.scatter),
        "invalid_folder_names": int(payload.get("invalid_folder_names", 0) or 0),
        "average_latency_seconds": (
            float(payload.get("latency_seconds", 0.0)) / agent_runs
            if agent_runs
            else 0.0
        ),
        "agent_runs": agent_runs,
        "steps": int(payload.get("steps", 0) or 0),
        "completion_tokens": int(payload.get("completion_tokens", 0) or 0),
        "input_tokens": int(payload.get("input_tokens", 0) or 0),
        "group_latency_seconds": float(payload.get("group_latency_seconds", 0) or 0),
        "group_input_tokens": int(payload.get("group_input_tokens", 0) or 0),
        "group_completion_tokens": int(
            payload.get("group_completion_tokens", 0) or 0
        ),
        "group_steps": int(payload.get("group_steps", 0) or 0),
        "groups_proposed": int(payload.get("groups_proposed", 0) or 0),
        "groups_accepted": int(payload.get("groups_accepted", 0) or 0),
        "groups_discarded": int(payload.get("groups_discarded", 0) or 0),
        "groups_rejected": int(payload.get("groups_rejected", 0) or 0),
        "grouped_rule_resolved_sources": list(
            payload.get("grouped_rule_resolved_sources", []) or []
        ),
        "grouped_unresolved_sources": list(
            payload.get("grouped_unresolved_sources", []) or []
        ),
        "error": payload.get("error", ""),
        "cases": cases,
    }


def _group_prompt_metrics(fixture: Path, model_id: str | None) -> dict[str, Any]:
    """Measure old and compact clustering task payloads with one tokenizer."""
    rules = load_rules()
    rule_moves, unresolved = classify_directory(fixture, rules)
    candidate_names = [move["source"] for move in rule_moves] + unresolved
    metadata = metadata_for_names(fixture, candidate_names)
    legacy = build_legacy_group_task(metadata)
    compact = build_group_task(metadata)
    resolved_model = model_id or os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
    try:
        from litellm import token_counter

        legacy_tokens = token_counter(model=resolved_model, text=legacy)
        compact_tokens = token_counter(model=resolved_model, text=compact)
        method = f"LiteLLM token_counter ({resolved_model})"
    except Exception as exc:
        legacy_tokens = 0
        compact_tokens = 0
        method = f"unavailable: {type(exc).__name__}"
    return {
        "legacy_prompt_chars": len(legacy),
        "compact_prompt_chars": len(compact),
        "legacy_prompt_tokens": legacy_tokens,
        "compact_prompt_tokens": compact_tokens,
        "prompt_token_reduction": (
            1 - compact_tokens / legacy_tokens if legacy_tokens else 0.0
        ),
        "prompt_measurement_method": method,
    }


def render_grouping_markdown(
    metrics: dict[str, Any],
    *,
    model: str,
    think: bool | None,
    timeout: float,
    group_timeout: float,
    read_contents: bool,
    warmup: dict[str, Any] | None,
    ground_truth: str = "",
    endpoint: str = "",
    provenance: dict[str, str] | None = None,
) -> str:
    """Render the grouping report: clustering metrics plus ungrouped accuracy."""
    category = metrics["category"]
    lines = _run_header(
        title="tidy-agent evaluation — grouping mode",
        model=model,
        think=think,
        timeout=timeout,
        group_timeout=group_timeout,
        group=True,
        read_contents=read_contents,
        status=metrics["status"],
        warmup=warmup,
        ground_truth=ground_truth,
        endpoint=endpoint,
        provenance=provenance,
    )
    lines.append(
        "- Judge: deterministic group-membership and expected-category "
        "comparison (no LLM judge)"
    )
    lines.extend(
        [
            "",
            "### Clustering metrics",
            "",
            "Only an accepted semantic group folder counts as a clustering "
            "decision. Files the extension rules placed in a fixed category are "
            "treated as ungrouped, not as a cluster, so rows about clustering "
            "cannot be moved by rule placements. Purity alone is maximised by "
            "grouping nothing and must be read together with group cohesion.",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Clustering purity, files in group folders | {_rate(metrics['clustering_purity'], metrics['purity_correct'], metrics['grouped_evaluated'])} |",
            f"| Ground-truth files placed in a group folder | {metrics['grouped_evaluated']}/{metrics['total_files']} |",
            f"| Clustering ground truth | {metrics['group_members_total']} group members + {metrics['scatter_total']} scatter files |",
            f"| Files in a group folder without clustering ground truth | {metrics['grouped_without_ground_truth']} |",
            f"| Destination purity, all evaluated files (includes fixed category folders) | {metrics['destination_purity']:.1%} ({metrics['destination_purity_correct']}/{metrics['total_files']}) |",
            f"| Fully co-located expected groups | {metrics['group_cohesion']:.1%} ({metrics['cohesive_groups']}/{metrics['total_groups']}) |",
            f"| Scatter files in an accepted group folder | {metrics['scatter_in_group']}/{metrics['scatter_total']} |",
            f"| Scatter files in a proposed cluster (before executor filtering) | {metrics['scatter_in_proposed_group']}/{metrics['scatter_total']} |",
            f"| …of those, dropped by the minimum-cluster-size filter | {metrics['scatter_in_discarded_group']}/{metrics['scatter_total']} |",
            f"| Scatter files sharing a fixed category folder (not a clustering error) | {metrics['scatter_sharing_category']}/{metrics['scatter_total']} |",
            f"| Invalid proposed folder names | {metrics['invalid_folder_names']} |",
            f"| Groups proposed / accepted / discarded / rejected | {metrics['groups_proposed']} / {metrics['groups_accepted']} / {metrics['groups_discarded']} / {metrics['groups_rejected']} |",
            f"| Rule-resolved files overridden by grouping | {len(metrics['grouped_rule_resolved_sources'])} |",
            f"| Unresolved files handled by grouping | {len(metrics['grouped_unresolved_sources'])} |",
            f"| Legacy clustering task tokens | {metrics['legacy_prompt_tokens']} |",
            f"| Compact clustering task tokens | {metrics['compact_prompt_tokens']} |",
            f"| Task token reduction | {metrics['prompt_token_reduction']:.1%} |",
            f"| Legacy/compact task characters | {metrics['legacy_prompt_chars']}/{metrics['compact_prompt_chars']} |",
            f"| Actual clustering input tokens | {metrics['group_input_tokens']} |",
            f"| Actual clustering completion tokens | {metrics['group_completion_tokens']} |",
            f"| Clustering steps | {metrics['group_steps']} |",
            f"| Clustering latency | {metrics['group_latency_seconds']:.3f} s |",
            f"| All agent input tokens | {metrics['input_tokens']} |",
            f"| Average agent latency | {metrics['average_latency_seconds']:.3f} s |",
            f"| Agent runs | {metrics['agent_runs']} |",
            f"| Agent steps | {metrics['steps']} |",
            f"| Completion tokens | {metrics['completion_tokens']} |",
            f"| Prompt measurement | {metrics['prompt_measurement_method']} |",
            "",
            "### Category metrics for ungrouped files",
            "",
            f"{category['grouped_excluded']} scored file(s) were placed in a "
            "semantic group folder and are excluded here; they are listed under "
            "*Files excluded from category scoring* below. That count covers "
            "files with category ground truth"
            + (
                f", the same set as the {metrics['grouped_evaluated']} file(s) "
                "with clustering ground truth counted above."
                if metrics["grouped_without_ground_truth"] == 0
                else f", which is not the same set as the {metrics['grouped_evaluated']} "
                "file(s) with clustering ground truth counted above."
            ),
            "",
        ]
    )
    lines.extend(_category_metric_rows(category, scope="ungrouped"))
    lines.extend(_rubric_note())
    if category["omitted_files"]:
        lines.extend(
            [
                "",
                "Omitted by the model (deterministic `_ToReview/` fallback): "
                + ", ".join(f"`{name}`" for name in category["omitted_files"]),
            ]
        )
    lines.extend(_assignment_table(category["cases"]))
    lines.extend(
        [
            "",
            "## Files excluded from category scoring",
            "",
            "| File | Group folder | Unresolved by rules |",
            "|---|---|---|",
        ]
    )
    for excluded in category["excluded_cases"]:
        lines.append(
            f"| `{_escaped(excluded['filename'])}` | `{excluded['destination']}` | "
            f"{'yes' if excluded['unresolved'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Final placement",
            "",
            "| File | Expected membership | Destination folder | Placed by |",
            "|---|---|---|---|",
        ]
    )
    for case in metrics["cases"]:
        lines.append(
            f"| `{_escaped(case['filename'])}` | `{case['expected']}` | "
            f"`{case['destination']}` | {case['placed_by']} |"
        )
    if metrics["error"]:
        lines.extend(["", f"Error: `{metrics['error']}`"])
    return "\n".join(lines) + "\n"


def run_evaluation(
    *,
    fixture: Path,
    expected_path: Path,
    output: Path,
    model_id: str | None,
    think: bool | None,
    timeout: float,
    use_agent: bool,
    group: bool = False,
    group_timeout: float = 600.0,
    read_contents: bool = False,
    allow_remote_content: bool = False,
    metadata_control: bool = False,
    classification_mode: str = "single_pass",
    perform_warmup: bool = True,
    warmup_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if read_contents and metadata_control:
        raise ValueError("content reading and metadata control are mutually exclusive")
    if read_contents and use_agent:
        ensure_content_authorized(
            model_id,
            allow_remote_content=allow_remote_content,
        )
    expected = _load_grouping_expected(expected_path)
    expected_categories = _load_expected(expected_path)
    missing = [
        name
        for name in {*expected.files, *expected_categories}
        if not (fixture / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing fixture files: {', '.join(sorted(missing))}")
    if group and not use_agent:
        raise ValueError("grouping evaluation requires the agent")

    _, unresolved_names = classify_directory(fixture, load_rules())
    model_label = "rules-only" if not use_agent else (model_id or "MODEL_ID/default Ollama")
    warmup = (
        run_in_subprocess(_warmup_worker, (model_id, think), timeout=timeout)
        if use_agent and perform_warmup
        else warmup_result
    )
    provenance = model_provenance(model_id) if use_agent else {}
    payload = run_in_subprocess(
        _plan_worker,
        (
            str(fixture),
            model_id,
            think,
            use_agent,
            group,
            read_contents,
            allow_remote_content,
            metadata_control,
            classification_mode,
        ),
        timeout=group_timeout if group else timeout,
    )
    category = _category_metrics(expected_categories, unresolved_names, payload)
    # Record how far the content axis can reach after the same cheap metadata
    # filter used by production Phase 1. This does not open candidate files.
    category["read_contents"] = read_contents
    category["metadata_control"] = metadata_control
    category["endpoint_local"] = endpoint_is_local(model_id) if use_agent else True
    unresolved_metadata = metadata_for_names(fixture, unresolved_names)
    category["peek_eligible"] = len(build_peek_candidates(unresolved_metadata))

    # The two modes measure different things, so each renders only the metric
    # family its run can actually produce.
    if group:
        metrics = _grouping_metrics(expected, payload)
        metrics.update(_group_prompt_metrics(fixture, model_id))
        metrics["category"] = category
        report = render_grouping_markdown(
            metrics,
            model=model_label,
            think=think,
            timeout=timeout,
            group_timeout=group_timeout,
            read_contents=read_contents,
            warmup=warmup,
            ground_truth=ground_truth_fingerprint(expected_path),
            endpoint=resolved_endpoint(model_id) if use_agent else "",
            provenance=provenance,
        )
    else:
        metrics = category
        report = render_category_markdown(
            metrics,
            model=model_label,
            think=think,
            timeout=timeout,
            group_timeout=group_timeout,
            read_contents=read_contents,
            metadata_control=metadata_control,
            warmup=warmup,
            ground_truth=ground_truth_fingerprint(expected_path),
            endpoint=resolved_endpoint(model_id) if use_agent else "",
            provenance=provenance,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    _archive_existing(output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(output)
    return metrics


def reset_model_state(
    model_id: str | None,
    *,
    api_base: str | None = None,
    timeout: float = 60.0,
) -> dict[str, str]:
    """Unload a local Ollama model so the next repetition starts cold.

    Repetitions against one warm process share the loaded weights and whatever
    server-side state came with them, and at temperature 0 they reproduce each
    other exactly. That makes the spread across repetitions 0 by construction
    while the same configuration measured in a later session differed by seven
    points. Forcing a reload between runs is the closest reproducible stand-in
    for a session boundary; without it the spread must be labelled as a lower
    bound instead of reported as variance.
    """
    load_dotenv()
    resolved = model_id or os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
    if not resolved.startswith("ollama"):
        return {"status": "skipped", "detail": f"no reset defined for {resolved}"}
    host = (api_base or endpoint_url(model_id)).rstrip("/")
    name = resolved.split("/", 1)[1] if "/" in resolved else resolved
    payload = json.dumps({"model": name, "keep_alive": 0}).encode("utf-8")
    # ``localhost`` can resolve to ::1 while Ollama listens on IPv4 only, which
    # urllib reports as ENETUNREACH instead of falling back. HTTP clients that do
    # fall back keep working, so without this retry the reset would silently
    # degrade every repetition to a warm one.
    hosts = [host]
    if "localhost" in host:
        hosts.append(host.replace("localhost", "127.0.0.1"))
    failures: list[str] = []
    for candidate in hosts:
        request = urllib.request.Request(
            f"{candidate}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            failures.append(type(exc).__name__)
            continue
        return {"status": "unloaded", "detail": f"{name} unloaded via keep_alive=0"}
    return {"status": "failed", "detail": "; ".join(failures)}


def _headline_values(metrics: dict[str, Any], *, group: bool) -> dict[str, tuple[float, str]]:
    """Pick the few metrics a repeated run should be summarised by."""
    category = metrics["category"] if group else metrics
    rows: dict[str, tuple[float, str]] = {}
    if group:
        rows["Clustering purity, files in group folders"] = (
            metrics["clustering_purity"],
            _rate(
                metrics["clustering_purity"],
                metrics["purity_correct"],
                metrics["grouped_evaluated"],
            ),
        )
        rows["Fully co-located expected groups"] = (
            metrics["group_cohesion"],
            _rate(
                metrics["group_cohesion"],
                metrics["cohesive_groups"],
                metrics["total_groups"],
            ),
        )
        rows["Ground-truth files placed in a group folder"] = (
            float(metrics["grouped_evaluated"]),
            f"{metrics['grouped_evaluated']}/{metrics['total_files']}",
        )
        rows["Scatter files in an accepted group folder"] = (
            float(metrics["scatter_in_group"]),
            f"{metrics['scatter_in_group']}/{metrics['scatter_total']}",
        )
        rows["Scatter files in a proposed cluster"] = (
            float(metrics["scatter_in_proposed_group"]),
            f"{metrics['scatter_in_proposed_group']}/{metrics['scatter_total']}",
        )
    rows["Category accuracy, unresolved files (`_ToReview` accepted)"] = (
        category["unknown_accuracy"],
        _rate(
            category["unknown_accuracy"],
            category["unknown_correct"],
            category["unknown_total"],
        ),
    )
    rows["Decision rate, unresolved files"] = (
        category["decision_rate"],
        _rate(
            category["decision_rate"],
            category["decided_count"],
            category["unknown_total"],
        ),
    )
    rows["Accuracy on decided files only"] = (
        category["decided_accuracy"],
        _rate(
            category["decided_accuracy"],
            category["decided_correct"],
            category["decided_count"],
        ),
    )
    rows["Files omitted by the model"] = (
        float(category["omitted_count"]),
        f"{category['omitted_count']}/{category['unknown_total']}",
    )
    rows["Classification latency"] = (
        category["class_latency_seconds"],
        f"{category['class_latency_seconds']:.1f} s",
    )
    return rows


def render_repeat_summary(
    runs: list[dict[str, Any]],
    *,
    group: bool,
    model: str,
    think: bool | None,
    timeout: float,
    group_timeout: float,
    reports: list[Path],
    read_contents: bool = False,
    ground_truth: str = "",
    endpoint: str = "",
    provenance: dict[str, str] | None = None,
    resets: list[dict[str, str]] | None = None,
) -> str:
    """Render one range per metric across repeated runs of the same mode.

    A single run cannot separate a mode difference from run-to-run variation, so
    repeated runs are reported as a spread over the individual values, never
    collapsed into one number.
    """
    think_label = "model default" if think is None else str(think)
    lines = [
        "# tidy-agent evaluation — repeated runs",
        "",
        f"- Runs: **{len(runs)}**",
        *_mode_header(group=group, read_contents=read_contents),
        f"- Model: `{model}`",
        f"- Think: `{think_label}`",
        f"- Standard run timeout: {timeout:.1f} s",
    ]
    if group:
        lines.append(f"- Clustering run timeout: {group_timeout:.1f} s")
    if ground_truth:
        lines.append(f"- Ground truth: `{ground_truth}`")
    if endpoint:
        lines.append(f"- Endpoint: `{endpoint}`")
    lines.extend(_provenance_lines(provenance or {}))
    cold = bool(resets) and all(reset["status"] == "unloaded" for reset in resets)
    isolation = (
        "model unloaded before every run (`keep_alive=0`), so each repetition "
        "reloads the weights"
        if cold
        else "none — every repetition ran against the same warm model process"
    )
    range_label = "Range (cold-start runs)" if cold else "Range (within-session, warm)"
    lines.extend(
        [
            f"- Evaluated: {datetime.now(timezone.utc).isoformat()}",
            "- Statuses: " + ", ".join(f"run {i}: {m['status']}" for i, m in enumerate(runs, 1)),
            f"- Repetition isolation: {isolation}",
        ]
    )
    if resets and not cold:
        lines.append(
            "- Reset results: "
            + ", ".join(f"run {i}: {reset['status']}" for i, reset in enumerate(resets, 1))
        )
    lines.extend(
        [
            "",
            "**How far this range reaches.** At temperature 0 repetitions "
            "against one warm process reproduce each other exactly, so a spread "
            "of zero there is a property of the setup, not a measurement of "
            "stability."
            + (
                " These runs reload the model between repetitions, which "
                "removes the warm-process part of that objection but still "
                "shares one machine, one server build, and one sitting."
                if cold
                else " This run did not reset the model between repetitions, so "
                "the range below is a **lower bound of variance** and must be "
                "labelled as such wherever it is quoted."
            ),
            "",
            "**Measured between-session delta.** The same configuration "
            "(`qwen3.5:4b`, `--no-think`, category mode) scored 75.0% (21/28) "
            "unresolved accuracy on 2026-08-10 at 17:44 UTC and 82.1% (23/28) "
            "at 20:52 UTC — 7.1 points apart with identical flags and code. "
            "That difference is itself a measurement and outranks any "
            "within-session range printed below.",
            "",
            "A range that overlaps another mode's range is not evidence of a "
            "difference between the modes.",
            "",
        ]
    )
    per_run = [_headline_values(metrics, group=group) for metrics in runs]
    if len(runs) > 1 and all(
        len({run[label][1] for run in per_run}) == 1
        for label in per_run[0]
        if label != "Classification latency"
    ):
        lines.extend(
            [
                "**Every repetition in this batch produced identical values.** "
                "Reloading the model between runs did not reproduce the "
                "between-session delta either, so whatever varies is not the "
                "warm process alone — most likely load-time conditions this "
                "harness does not control. The range stays a lower bound.",
                "",
            ]
        )
    headers = " | ".join(f"run {index}" for index in range(1, len(runs) + 1))
    lines.append(f"| Metric | {headers} | {range_label} |")
    lines.append("|---|" + "---:|" * (len(runs) + 1))
    for label in per_run[0]:
        values = [run[label][0] for run in per_run]
        displays = [run[label][1] for run in per_run]
        spread = "identical" if min(values) == max(values) else f"{min(values):g} – {max(values):g}"
        lines.append(f"| {label} | " + " | ".join(displays) + f" | {spread} |")
    lines.extend(
        [
            "",
            "Individual reports:",
            "",
            *(f"- [`{path.name}`]({path.name})" for path in reports),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="LiteLLM model id (overrides MODEL_ID)")
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="evaluate deterministic rules plus _ToReview fallback",
    )
    grouping = parser.add_mutually_exclusive_group()
    grouping.add_argument("--group", dest="group", action="store_true")
    grouping.add_argument("--no-group", dest="group", action="store_false")
    parser.set_defaults(group=False)
    reading = parser.add_mutually_exclusive_group()
    reading.add_argument(
        "--read-contents",
        dest="read_contents",
        action="store_true",
        help="let structured classification peek into unresolved files",
    )
    reading.add_argument(
        "--no-read-contents", dest="read_contents", action="store_false"
    )
    parser.set_defaults(read_contents=False)
    parser.add_argument(
        "--allow-remote-content",
        action="store_true",
        help="authorize excerpts to be sent to a non-local model endpoint",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--think", dest="think", action="store_true")
    thinking.add_argument("--no-think", dest="think", action="store_false")
    parser.set_defaults(think=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="wall-clock timeout per warm-up/run in seconds (default: 120)",
    )
    parser.add_argument(
        "--group-timeout",
        type=float,
        default=600.0,
        help="wall-clock timeout for the complete clustering run (default: 600)",
    )
    parser.add_argument("--fixture", type=Path, default=Path(__file__).parent / "fixture")
    parser.add_argument(
        "--expected", type=Path, default=Path(__file__).parent / "expected.yaml"
    )
    parser.add_argument("--output", type=Path, help="Markdown output path")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat the same mode N times and add a range summary (default: 1)",
    )
    reset = parser.add_mutually_exclusive_group()
    reset.add_argument(
        "--reset-between-runs",
        dest="reset_between_runs",
        action="store_true",
        help="unload the local model before every repetition (default with --repeat)",
    )
    reset.add_argument(
        "--no-reset-between-runs",
        dest="reset_between_runs",
        action="store_false",
        help="keep the warm model process; the reported range is a lower bound",
    )
    parser.set_defaults(reset_between_runs=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than zero")
    if args.group_timeout <= 0:
        raise SystemExit("--group-timeout must be greater than zero")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least one")
    output = args.output or (
        Path(__file__).parent
        / "results"
        / f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.md"
    )

    # Repetitions are only worth reporting as a range if they are independent of
    # each other, so isolate them by default whenever more than one runs.
    reset_between_runs = (
        args.repeat > 1 and not args.no_agent
        if args.reset_between_runs is None
        else args.reset_between_runs
    )

    runs: list[dict[str, Any]] = []
    reports: list[Path] = []
    resets: list[dict[str, str]] = []
    for index in range(1, args.repeat + 1):
        run_output = (
            output
            if args.repeat == 1
            else output.with_name(f"{output.stem}-run{index}{output.suffix}")
        )
        if reset_between_runs:
            reset = reset_model_state(args.model)
            resets.append(reset)
            LOGGER.info("model reset before run %d: %s", index, reset)
            if reset["status"] == "failed":
                LOGGER.warning(
                    "run %d starts warm; the reported range is a lower bound", index
                )
        LOGGER.info("run %d/%d -> %s", index, args.repeat, run_output)
        runs.append(
            run_evaluation(
                fixture=args.fixture,
                expected_path=args.expected,
                output=run_output,
                model_id=args.model,
                think=args.think,
                timeout=args.timeout,
                use_agent=not args.no_agent,
                group=args.group,
                group_timeout=args.group_timeout,
                read_contents=args.read_contents,
                allow_remote_content=args.allow_remote_content,
            )
        )
        reports.append(run_output)

    if args.repeat == 1:
        print(output.read_text(encoding="utf-8"), end="")
        print(f"\nSaved: {output}")
    else:
        summary = render_repeat_summary(
            runs,
            group=args.group,
            model="rules-only" if args.no_agent else (args.model or "MODEL_ID/default Ollama"),
            think=args.think,
            timeout=args.timeout,
            group_timeout=args.group_timeout,
            reports=reports,
            read_contents=args.read_contents,
            ground_truth=ground_truth_fingerprint(args.expected),
            endpoint="rules-only" if args.no_agent else resolved_endpoint(args.model),
            provenance={} if args.no_agent else model_provenance(args.model),
            resets=resets,
        )
        _archive_existing(output)
        output.write_text(summary, encoding="utf-8")
        print(summary, end="")
        print(f"\nSaved: {output} (+ {len(reports)} run reports)")
    return 0 if all(metrics["status"] == "ok" for metrics in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
