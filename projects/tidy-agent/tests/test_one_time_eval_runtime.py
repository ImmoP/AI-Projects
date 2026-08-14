"""Unit tests for the generic one-time-eval runtime primitives: atomic
writes, the lifecycle recorder, the journaling model proxy, and model
identity verification. No Holdout, fixture, or candidate code is involved.
"""
from __future__ import annotations

import json

import pytest
from evals import one_time_eval_runtime as runtime


def test_atomic_write_json_roundtrip(tmp_path):
    path = tmp_path / "sub" / "data.json"
    runtime.atomic_write_json(path, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(path.read_text()) == {"a": 1, "b": [1, 2, 3]}


def test_atomic_write_leaves_no_temp_files(tmp_path):
    path = tmp_path / "data.json"
    runtime.atomic_write_json(path, {"x": 1})
    runtime.atomic_write_json(path, {"x": 2})
    entries = list(tmp_path.iterdir())
    assert entries == [path]
    assert json.loads(path.read_text()) == {"x": 2}


def test_append_jsonl_and_read_jsonl(tmp_path):
    path = tmp_path / "journal.jsonl"
    runtime.append_jsonl(path, {"n": 1})
    runtime.append_jsonl(path, {"n": 2})
    assert runtime.read_jsonl(path) == [{"n": 1}, {"n": 2}]


def test_read_jsonl_tolerates_truncated_trailing_line(tmp_path):
    path = tmp_path / "journal.jsonl"
    runtime.append_jsonl(path, {"n": 1})
    with open(path, "ab") as handle:
        handle.write(b'{"n": 2, "trunc')  # simulate a crash mid-write
    assert runtime.read_jsonl(path) == [{"n": 1}]


def test_read_jsonl_missing_file_returns_empty(tmp_path):
    assert runtime.read_jsonl(tmp_path / "missing.jsonl") == []


def test_create_fresh_result_directory(tmp_path):
    result_dir = runtime.create_fresh_result_directory(tmp_path, "run-1")
    assert result_dir.is_dir()


def test_create_fresh_result_directory_collision_rejected(tmp_path):
    runtime.create_fresh_result_directory(tmp_path, "run-1")
    with pytest.raises(runtime.ResultDirectoryExistsError):
        runtime.create_fresh_result_directory(tmp_path, "run-1")


# ---------------------------------------------------------------------------
# LifecycleRecorder
# ---------------------------------------------------------------------------


def _identity_dict():
    return {"model_id": "m", "digest": "d", "quantization": "q", "temperature": 0, "thinking_enabled": False, "num_ctx": 8192}


def test_record_prepared_writes_manifest_and_lifecycle_before_any_dispatch(tmp_path):
    recorder = runtime.LifecycleRecorder(tmp_path)
    recorder.record_prepared(
        run_id="r1", pipeline="p", model_identity_expected=_identity_dict(), fixture_hash="abc"
    )
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "lifecycle.json").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["run_id"] == "r1"
    assert manifest["lifecycle_state"] == "prepared"
    assert manifest["fixture_hash"] == "abc"
    lifecycle = json.loads((tmp_path / "lifecycle.json").read_text())
    assert lifecycle["state"] == "prepared"


def test_lifecycle_state_history_accumulates(tmp_path):
    recorder = runtime.LifecycleRecorder(tmp_path)
    recorder.record_prepared(run_id="r1", pipeline="p", model_identity_expected=_identity_dict(), fixture_hash="abc")
    recorder.record_state("scoring_complete")
    recorder.record_state("persisted")
    recorder.record_state("complete")
    lifecycle = json.loads((tmp_path / "lifecycle.json").read_text())
    assert [s["state"] for s in lifecycle["state_history"]] == [
        "prepared", "scoring_complete", "persisted", "complete",
    ]
    assert lifecycle["state"] == "complete"


def test_lifecycle_rejects_unknown_state(tmp_path):
    recorder = runtime.LifecycleRecorder(tmp_path)
    recorder.record_prepared(run_id="r1", pipeline="p", model_identity_expected=_identity_dict(), fixture_hash="abc")
    with pytest.raises(ValueError):
        recorder.record_state("not_a_real_state")


def test_request_and_response_journaling(tmp_path):
    recorder = runtime.LifecycleRecorder(tmp_path)
    recorder.record_prepared(run_id="r1", pipeline="p", model_identity_expected=_identity_dict(), fixture_hash="abc")
    recorder.record_request_started(phase="pass1", ordinal=1)
    entries = recorder.journal_entries()
    assert len(entries) == 1
    assert entries[0]["event"] == "request_started"
    assert entries[0]["fixture_hash"] == "abc"

    recorder.record_response_received(
        phase="pass1", ordinal=1, parse_status="received", latency_seconds=1.5,
        input_tokens=10, completion_tokens=5,
    )
    entries = recorder.journal_entries()
    assert len(entries) == 2
    assert entries[1]["event"] == "response_received"
    assert entries[1]["input_tokens"] == 10

    lifecycle = json.loads((tmp_path / "lifecycle.json").read_text())
    assert lifecycle["state"] == "response_received"


def test_partial_failed_preserves_prior_history(tmp_path):
    recorder = runtime.LifecycleRecorder(tmp_path)
    recorder.record_prepared(run_id="r1", pipeline="p", model_identity_expected=_identity_dict(), fixture_hash="abc")
    recorder.record_request_started(phase="pass1", ordinal=1)
    recorder.record_state("partial_failed", error="boom")
    lifecycle = json.loads((tmp_path / "lifecycle.json").read_text())
    assert lifecycle["state"] == "partial_failed"
    states = [s["state"] for s in lifecycle["state_history"]]
    assert states == ["prepared", "request_started", "partial_failed"]
    assert lifecycle["state_history"][-1]["error"] == "boom"
    assert "complete" not in states


# ---------------------------------------------------------------------------
# JournalingModelProxy
# ---------------------------------------------------------------------------


class _Usage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Response:
    def __init__(self, content, usage=None):
        self.content = content
        self.token_usage = usage


class _StubModel:
    model_id = "stub/model"
    structured_output_mode = "plain_json"

    def __init__(self, responses=None, fail_on_ordinal=None):
        self.calls = 0
        self.responses = responses or []
        self.fail_on_ordinal = fail_on_ordinal

    def generate(self, messages, **kwargs):
        self.calls += 1
        if self.fail_on_ordinal == self.calls:
            raise RuntimeError("simulated provider failure")
        return self.responses[self.calls - 1]


def test_journaling_proxy_passthrough_attributes(tmp_path):
    recorder = runtime.LifecycleRecorder(tmp_path)
    recorder.record_prepared(run_id="r1", pipeline="p", model_identity_expected=_identity_dict(), fixture_hash="abc")
    stub = _StubModel(responses=[_Response("{}", _Usage(10, 2))])
    proxy = runtime.JournalingModelProxy(stub, recorder, phase_labels=("pass1", "pass2"))
    assert proxy.model_id == "stub/model"
    assert proxy.structured_output_mode == "plain_json"


def test_journaling_proxy_records_request_and_response_per_call(tmp_path):
    recorder = runtime.LifecycleRecorder(tmp_path)
    recorder.record_prepared(run_id="r1", pipeline="p", model_identity_expected=_identity_dict(), fixture_hash="abc")
    stub = _StubModel(responses=[_Response("{}", _Usage(10, 2)), _Response("{}", _Usage(20, 4))])
    proxy = runtime.JournalingModelProxy(stub, recorder, phase_labels=("pass1", "pass2"))

    proxy.generate([])
    proxy.generate([])

    entries = recorder.journal_entries()
    assert [e["event"] for e in entries] == [
        "request_started", "response_received", "request_started", "response_received",
    ]
    assert entries[0]["phase"] == "pass1"
    assert entries[2]["phase"] == "pass2"
    assert entries[1]["input_tokens"] == 10
    assert entries[3]["input_tokens"] == 20
    assert proxy.dispatch_attempted == 2
    assert proxy.dispatch_returned == 2
    assert proxy.dispatch_failed == 0


def test_journaling_proxy_records_request_started_but_no_response_on_failure(tmp_path):
    recorder = runtime.LifecycleRecorder(tmp_path)
    recorder.record_prepared(run_id="r1", pipeline="p", model_identity_expected=_identity_dict(), fixture_hash="abc")
    stub = _StubModel(responses=[], fail_on_ordinal=1)
    proxy = runtime.JournalingModelProxy(stub, recorder, phase_labels=("pass1", "pass2"))

    with pytest.raises(RuntimeError):
        proxy.generate([])

    entries = recorder.journal_entries()
    assert [e["event"] for e in entries] == ["request_started"]
    assert proxy.dispatch_attempted == 1
    assert proxy.dispatch_returned == 0
    assert proxy.dispatch_failed == 1


def test_journaling_proxy_second_call_failure_leaves_first_response_intact(tmp_path):
    recorder = runtime.LifecycleRecorder(tmp_path)
    recorder.record_prepared(run_id="r1", pipeline="p", model_identity_expected=_identity_dict(), fixture_hash="abc")
    stub = _StubModel(responses=[_Response("{}", _Usage(10, 2))], fail_on_ordinal=2)
    proxy = runtime.JournalingModelProxy(stub, recorder, phase_labels=("pass1", "pass2"))

    proxy.generate([])
    with pytest.raises(RuntimeError):
        proxy.generate([])

    entries = recorder.journal_entries()
    assert [e["event"] for e in entries] == [
        "request_started", "response_received", "request_started",
    ]


# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------


def _identity(**overrides):
    base = dict(model_id="ollama_chat/qwen3.5:4b", digest="abc123", quantization="Q4_K_M",
                temperature=0, thinking_enabled=False, num_ctx=8192)
    base.update(overrides)
    return runtime.ModelIdentity(**base)


def test_model_identity_match_passes():
    runtime.verify_model_identity(_identity(), _identity())


def test_model_identity_digest_mismatch_raises():
    with pytest.raises(runtime.ModelIdentityError):
        runtime.verify_model_identity(_identity(), _identity(digest="different"))


def test_model_identity_reports_all_mismatches():
    with pytest.raises(runtime.ModelIdentityError) as exc_info:
        runtime.verify_model_identity(_identity(), _identity(digest="different", num_ctx=4096))
    message = str(exc_info.value)
    assert "digest" in message
    assert "num_ctx" in message
