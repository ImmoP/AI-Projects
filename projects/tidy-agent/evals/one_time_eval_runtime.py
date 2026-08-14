"""Generic, reusable infrastructure for one-time live evaluation runners.

Extracted after the Holdout v3 incident: a process was interrupted between
writing its consumption marker and issuing the first measured provider
request, leaving an empty result directory with no way to distinguish
"marker written but dispatch never started" from "provider request in
flight" from "provider response returned". Nothing here is specific to any
Holdout, fixture, or candidate -- it operates only on a result directory, a
run id, and generic phase/ordinal labels, so it can back a synthetic smoke
runner today and a future independent Holdout runner later without either
one importing the other.

Nothing in this module performs model inference, candidate classification,
or fixture construction. It has no import dependency on
``evals/run_holdout_v3_e4.py`` or any Holdout artifact, by design.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

LIFECYCLE_STATES = (
    "prepared",
    "request_started",
    "response_received",
    "scoring_complete",
    "persisted",
    "complete",
    "partial_failed",
)


class ResultDirectoryExistsError(RuntimeError):
    """A fresh result directory was requested but the path already exists."""


# ---------------------------------------------------------------------------
# Atomic persistence primitives
# ---------------------------------------------------------------------------


def _fsync_dir(dir_path: Path) -> None:
    """Best-effort directory fsync. Not all platforms support this."""
    try:
        fd = os.open(str(dir_path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write-temp, flush, fsync, atomic rename, best-effort directory fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    with open(tmp_path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    _fsync_dir(path.parent)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def append_jsonl(path: Path, obj: Any) -> None:
    """Append one flushed-and-fsynced line. A crash mid-write can only ever
    truncate the final line; readers must tolerate that (see ``read_jsonl``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as handle:
        handle.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse every complete line; silently drop one truncated trailing line."""
    if not path.exists():
        return []
    lines = path.read_bytes().split(b"\n")
    records: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(json.loads(line.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return records


def create_fresh_result_directory(root: Path, name: str) -> Path:
    result_dir = root / name
    if result_dir.exists():
        raise ResultDirectoryExistsError(f"result directory already exists: {name}")
    result_dir.mkdir(parents=True)
    return result_dir


# ---------------------------------------------------------------------------
# Lifecycle recorder
# ---------------------------------------------------------------------------


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class LifecycleRecorder:
    """Journals lifecycle-state transitions and per-request/response events
    for one run, durably, so a postmortem can always answer: was a request
    ever attempted, did it get a response, and did final results land.

    ``manifest.json`` is written exactly once, at :meth:`record_prepared`,
    before any provider dispatch -- this directly ensures the result
    directory is never empty if interruption happens after that point.
    ``lifecycle.json`` is rewritten atomically on every state transition.
    ``dispatch_journal.jsonl`` is appended to (flushed + fsynced) once per
    request-start and once per response-received event.
    """

    def __init__(self, result_dir: Path) -> None:
        self.result_dir = result_dir
        self.manifest_path = result_dir / "manifest.json"
        self.lifecycle_path = result_dir / "lifecycle.json"
        self.journal_path = result_dir / "dispatch_journal.jsonl"
        self._doc: dict[str, Any] = {}

    def record_prepared(
        self,
        *,
        run_id: str,
        pipeline: str,
        model_identity_expected: dict[str, Any],
        fixture_hash: str,
    ) -> None:
        timestamp = _utc_timestamp()
        self._doc = {
            "run_id": run_id,
            "pipeline": pipeline,
            "model_identity_expected": model_identity_expected,
            "fixture_hash": fixture_hash,
            "start_timestamp": timestamp,
            "state": "prepared",
            "state_history": [{"state": "prepared", "timestamp": timestamp}],
        }
        atomic_write_json(
            self.manifest_path,
            {
                "run_id": run_id,
                "pipeline": pipeline,
                "model_identity_expected": model_identity_expected,
                "fixture_hash": fixture_hash,
                "start_timestamp": timestamp,
                "lifecycle_state": "prepared",
            },
        )
        atomic_write_json(self.lifecycle_path, self._doc)

    def _advance(self, state: str, **extra: Any) -> None:
        if state not in LIFECYCLE_STATES:
            raise ValueError(f"unknown lifecycle state: {state}")
        self._doc["state"] = state
        self._doc.setdefault("state_history", []).append(
            {"state": state, "timestamp": _utc_timestamp(), **extra}
        )
        atomic_write_json(self.lifecycle_path, self._doc)

    def record_state(self, state: str, **extra: Any) -> None:
        self._advance(state, **extra)

    def record_request_started(self, *, phase: str, ordinal: int) -> None:
        append_jsonl(
            self.journal_path,
            {
                "event": "request_started",
                "phase": phase,
                "ordinal": ordinal,
                "timestamp": _utc_timestamp(),
                "fixture_hash": self._doc.get("fixture_hash"),
            },
        )
        self._advance("request_started", phase=phase, ordinal=ordinal)

    def record_response_received(
        self,
        *,
        phase: str,
        ordinal: int,
        parse_status: str,
        latency_seconds: float,
        input_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        append_jsonl(
            self.journal_path,
            {
                "event": "response_received",
                "phase": phase,
                "ordinal": ordinal,
                "timestamp": _utc_timestamp(),
                "parse_status": parse_status,
                "latency_seconds": latency_seconds,
                "input_tokens": input_tokens,
                "completion_tokens": completion_tokens,
            },
        )
        self._advance("response_received", phase=phase, ordinal=ordinal)

    def journal_entries(self) -> list[dict[str, Any]]:
        return read_jsonl(self.journal_path)


# ---------------------------------------------------------------------------
# Provider-call journaling proxy
# ---------------------------------------------------------------------------


class JournalingModelProxy:
    """Wraps a real model so every ``generate`` call is journaled immediately
    before dispatch and immediately after response -- without requiring any
    change to the classifier/candidate code that calls ``model.generate``.

    On a provider exception, no response event is recorded (there was no
    response); the exception propagates so the caller's own lifecycle
    handling can record ``partial_failed``. Non-``generate`` attribute
    access passes through to the wrapped model unchanged, so callers that
    inspect ``model.model_id`` or ``model.structured_output_mode`` keep
    working transparently.
    """

    def __init__(
        self,
        model: Any,
        recorder: LifecycleRecorder,
        *,
        phase_labels: Sequence[str] | None = None,
    ) -> None:
        self._model = model
        self._recorder = recorder
        self._phase_labels = tuple(phase_labels) if phase_labels else ()
        self._ordinal = 0
        self.dispatch_attempted = 0
        self.dispatch_returned = 0
        self.dispatch_failed = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def _phase_for(self, ordinal: int) -> str:
        if ordinal <= len(self._phase_labels):
            return self._phase_labels[ordinal - 1]
        return f"call{ordinal}"

    def generate(self, messages: Any, **kwargs: Any) -> Any:
        self._ordinal += 1
        ordinal = self._ordinal
        phase = self._phase_for(ordinal)
        self._recorder.record_request_started(phase=phase, ordinal=ordinal)
        self.dispatch_attempted += 1
        started = time.perf_counter()
        try:
            response = self._model.generate(messages, **kwargs)
        except Exception:
            self.dispatch_failed += 1
            raise
        elapsed = time.perf_counter() - started
        self.dispatch_returned += 1
        usage = getattr(response, "token_usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
        completion_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
        self._recorder.record_response_received(
            phase=phase,
            ordinal=ordinal,
            parse_status="received",
            latency_seconds=elapsed,
            input_tokens=int(input_tokens) if input_tokens is not None else None,
            completion_tokens=int(completion_tokens) if completion_tokens is not None else None,
        )
        return response


# ---------------------------------------------------------------------------
# Generic model-identity verification (self-contained; no Holdout coupling)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    digest: str
    quantization: str
    temperature: float
    thinking_enabled: bool
    num_ctx: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "digest": self.digest,
            "quantization": self.quantization,
            "temperature": self.temperature,
            "thinking_enabled": self.thinking_enabled,
            "num_ctx": self.num_ctx,
        }


class ModelIdentityError(RuntimeError):
    """Raised before any measured request when the live model identity does
    not match the expected identity. No substitution is ever attempted."""


def verify_model_identity(expected: ModelIdentity, actual: ModelIdentity) -> None:
    mismatches = [
        f"{field}: expected {getattr(expected, field)!r}, got {getattr(actual, field)!r}"
        for field in ("model_id", "digest", "quantization", "temperature", "thinking_enabled", "num_ctx")
        if getattr(expected, field) != getattr(actual, field)
    ]
    if mismatches:
        raise ModelIdentityError("; ".join(mismatches))
