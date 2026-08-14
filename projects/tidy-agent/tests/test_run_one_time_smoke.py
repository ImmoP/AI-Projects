"""Failure-injection and lifecycle tests for the synthetic smoke runner.

FakeModel/mocks only -- these tests never call a real provider and never
touch evals/holdout_v3 or any other Holdout's consumption state. An
autouse guard fails loudly if a test ever does.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from evals import one_time_eval_runtime as runtime
from evals import run_one_time_smoke as smoke

REPO_ROOT = Path(smoke.__file__).resolve().parent.parent
REAL_HOLDOUT_V3_CONSUMED = REPO_ROOT / "evals" / "holdout_v3" / "CONSUMED.json"
PROTECTED_FILES = [
    REPO_ROOT / "src" / "tidy" / "classification.py",
    REPO_ROOT / "src" / "tidy" / "cli.py",
    REPO_ROOT / "config" / "rules.yaml",
    REPO_ROOT / "evals" / "post_holdout_candidates.py",
    REPO_ROOT / "evals" / "run_holdout_v3_e4.py",
]


def _hash_all(paths):
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


@pytest.fixture(autouse=True)
def _guard_holdout_v3_and_production():
    before_consumed = REAL_HOLDOUT_V3_CONSUMED.read_bytes() if REAL_HOLDOUT_V3_CONSUMED.exists() else None
    before_hashes = _hash_all(PROTECTED_FILES)
    yield
    after_consumed = REAL_HOLDOUT_V3_CONSUMED.read_bytes() if REAL_HOLDOUT_V3_CONSUMED.exists() else None
    assert after_consumed == before_consumed, "Holdout v3 consumption record was mutated by a smoke test"
    assert _hash_all(PROTECTED_FILES) == before_hashes, "a protected production/candidate file was mutated"


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.token_usage = None


class RecordingFakeModel:
    model_id = "fake/test-model"

    def __init__(self, decision_fn=None, fail_on_ordinal=None):
        self.calls = 0
        self.fail_on_ordinal = fail_on_ordinal
        self.decision_fn = decision_fn or self._default_decision

    @staticmethod
    def _default_decision(name):
        if "document" in name:
            return {"source": name, "decision": "classify", "category": "Documents"}
        if "image" in name:
            return {"source": name, "decision": "classify", "category": "Images"}
        return {"source": name, "decision": "review", "category": None}

    def generate(self, messages, **kwargs):
        self.calls += 1
        if self.fail_on_ordinal == self.calls:
            raise RuntimeError(f"simulated provider failure on call {self.calls}")
        text = messages[0]["content"][0]["text"]
        start = text.index("<FILENAME_DATA>") + len("<FILENAME_DATA>")
        end = text.index("</FILENAME_DATA>")
        names = [n for n in text[start:end].strip().split("\n") if n]
        decisions = [self.decision_fn(n) for n in names]
        return FakeResponse(json.dumps({"decisions": decisions}))


CASES = [
    {"filename": "synthetic_document_case_001_test_doc", "intended_category": "Documents"},
    {"filename": "synthetic_image_case_002_test_img", "intended_category": "Images"},
    {"filename": "synthetic_review_case_003_test_unresolved", "intended_category": "_ToReview"},
]


@pytest.fixture
def isolated_smoke(tmp_path):
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    for c in CASES:
        (fixture_dir / c["filename"]).write_bytes(b"")
    manifest_path = tmp_path / "fixture_manifest.json"
    manifest_path.write_text(json.dumps({"cases": CASES}), encoding="utf-8")
    results_root = tmp_path / "results"
    results_root.mkdir()
    return {"fixture_dir": fixture_dir, "manifest_path": manifest_path, "results_root": results_root}


def _identity():
    return smoke.expected_model_identity()


def _run(model, isolated_smoke, run_label):
    return smoke.run_smoke_trial(
        model,
        run_label=run_label,
        expected_identity=_identity(),
        actual_identity=_identity(),
        fixture_dir=isolated_smoke["fixture_dir"],
        manifest_path=isolated_smoke["manifest_path"],
        results_root=isolated_smoke["results_root"],
    )


def _lifecycle_state(result_dir):
    return json.loads((result_dir / "lifecycle.json").read_text())


def _find_result_dir(results_root, run_label):
    matches = list(results_root.glob(f"synthetic-one-time-smoke-*-{run_label}"))
    assert len(matches) == 1
    return matches[0]


# ---------------------------------------------------------------------------
# Happy path / completion requirements
# ---------------------------------------------------------------------------


def test_happy_path_reaches_complete_with_nonempty_result_dir(isolated_smoke):
    model = RecordingFakeModel()
    result_dir = _run(model, isolated_smoke, "happy")
    assert model.calls == 2
    lifecycle = _lifecycle_state(result_dir)
    assert lifecycle["state"] == "complete"
    expected_files = {
        "manifest.json", "lifecycle.json", "dispatch_journal.jsonl",
        "raw_run.jsonl", "per_file_evidence.json", "summary.json", "report.md",
    }
    present = {p.name for p in result_dir.iterdir()}
    assert expected_files <= present
    for name in expected_files:
        assert (result_dir / name).stat().st_size > 0


def test_e4_adds_zero_model_calls(isolated_smoke):
    model = RecordingFakeModel()
    _run(model, isolated_smoke, "e4zero")
    assert model.calls == 2  # exactly E3's two passes; E4-current is a local veto, no model call


def test_no_fixture_mutation(isolated_smoke):
    before = {p.name: p.stat().st_size for p in isolated_smoke["fixture_dir"].iterdir()}
    _run(RecordingFakeModel(), isolated_smoke, "nomut")
    after = {p.name: p.stat().st_size for p in isolated_smoke["fixture_dir"].iterdir()}
    assert before == after


def test_rerunnable_smoke_never_writes_holdout_marker(isolated_smoke):
    _run(RecordingFakeModel(), isolated_smoke, "rerun1")
    _run(RecordingFakeModel(), isolated_smoke, "rerun2")
    assert not (isolated_smoke["results_root"] / "CONSUMED.json").exists()


# ---------------------------------------------------------------------------
# Result-directory collision
# ---------------------------------------------------------------------------


def test_result_directory_collision_rejected(isolated_smoke):
    import time

    date_str = time.strftime("%Y%m%d")
    pre_existing = isolated_smoke["results_root"] / f"synthetic-one-time-smoke-{date_str}-collide"
    pre_existing.mkdir()
    model = RecordingFakeModel()
    with pytest.raises(runtime.ResultDirectoryExistsError):
        _run(model, isolated_smoke, "collide")
    assert model.calls == 0


# ---------------------------------------------------------------------------
# Model identity mismatch stops before any dispatch
# ---------------------------------------------------------------------------


def test_model_identity_mismatch_stops_before_dispatch(isolated_smoke):
    bad_identity = runtime.ModelIdentity(
        model_id=smoke.EXPECTED_MODEL_ID, digest="deadbeef",
        quantization=smoke.EXPECTED_QUANTIZATION, temperature=smoke.EXPECTED_TEMPERATURE,
        thinking_enabled=smoke.EXPECTED_THINKING_ENABLED, num_ctx=smoke.EXPECTED_NUM_CTX,
    )
    model = RecordingFakeModel()
    with pytest.raises(runtime.ModelIdentityError):
        smoke.run_smoke_trial(
            model, run_label="badid", expected_identity=_identity(), actual_identity=bad_identity,
            fixture_dir=isolated_smoke["fixture_dir"], manifest_path=isolated_smoke["manifest_path"],
            results_root=isolated_smoke["results_root"],
        )
    assert model.calls == 0
    assert not list(isolated_smoke["results_root"].iterdir())


# ---------------------------------------------------------------------------
# Item 22: failure after result-directory creation but before dispatch
# ---------------------------------------------------------------------------


def test_failure_before_dispatch(isolated_smoke, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated pre-dispatch failure")

    monkeypatch.setattr(smoke, "run_e4", _boom)
    model = RecordingFakeModel()
    with pytest.raises(RuntimeError):
        _run(model, isolated_smoke, "predispatch")

    assert model.calls == 0
    result_dir = _find_result_dir(isolated_smoke["results_root"], "predispatch")
    lifecycle = _lifecycle_state(result_dir)
    assert lifecycle["state"] == "partial_failed"
    assert lifecycle["state_history"][-1]["error"] == "simulated pre-dispatch failure"
    assert not runtime.read_jsonl(result_dir / "dispatch_journal.jsonl")
    assert (result_dir / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Items 23/24: failure after a request-start journal but before its response
#
# Discovery: the frozen ClassificationBackend.request() (src/tidy/
# classification.py) already catches every provider exception internally
# and degrades that pass to a fallback rather than re-raising -- so, run
# through the real, unmodified E3 path, a mid-dispatch provider failure
# never aborts the trial; it silently degrades the affected pass to
# _ToReview and the run still reaches "complete". This is frozen,
# deliberate, fail-safe production behavior, not a smoke-infrastructure
# bug, and it must not be changed here. The precise "request_started
# recorded, no matching response_received" signature this item asks for is
# therefore tested in isolation at the JournalingModelProxy level (see
# tests/test_one_time_eval_runtime.py::test_journaling_proxy_records_
# request_started_but_no_response_on_failure and
# ::test_journaling_proxy_second_call_failure_leaves_first_response_intact),
# which is the layer that actually performs the journaling. These two
# integration tests instead confirm the fully-integrated real-path
# consequence: no false "success", the provider error is visible in
# telemetry, and the journal is still asymmetric for the failed ordinal.
# ---------------------------------------------------------------------------


def test_failure_during_first_provider_call_degrades_gracefully_through_real_e3(isolated_smoke):
    model = RecordingFakeModel(fail_on_ordinal=1)
    result_dir = _run(model, isolated_smoke, "duringcall1")  # does not raise -- see discovery note above

    assert model.calls == 2
    lifecycle = _lifecycle_state(result_dir)
    assert lifecycle["state"] == "complete"
    entries = runtime.read_jsonl(result_dir / "dispatch_journal.jsonl")
    assert [(e["event"], e["phase"]) for e in entries] == [
        ("request_started", "pass1"),
        ("request_started", "pass2"),
        ("response_received", "pass2"),
    ]
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["telemetry"]["provider_errors"] == 1
    # pass1 was entirely invalid, so the agreement gate cannot automate anything
    assert summary["sanity_diagnostics"]["auto"] == 0


def test_failure_during_second_provider_call_degrades_gracefully_through_real_e3(isolated_smoke):
    model = RecordingFakeModel(fail_on_ordinal=2)
    result_dir = _run(model, isolated_smoke, "duringcall2")  # does not raise -- see discovery note above

    assert model.calls == 2
    lifecycle = _lifecycle_state(result_dir)
    assert lifecycle["state"] == "complete"
    entries = runtime.read_jsonl(result_dir / "dispatch_journal.jsonl")
    assert [(e["event"], e["phase"]) for e in entries] == [
        ("request_started", "pass1"),
        ("response_received", "pass1"),
        ("request_started", "pass2"),
    ]
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["telemetry"]["provider_errors"] == 1
    assert summary["sanity_diagnostics"]["auto"] == 0


# ---------------------------------------------------------------------------
# Item 25: failure after model responses but before final report persistence
# ---------------------------------------------------------------------------


def test_failure_before_persistence_preserves_raw_evidence(isolated_smoke, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated persistence failure")

    monkeypatch.setattr(smoke, "_persist_results", _boom)
    model = RecordingFakeModel()
    with pytest.raises(RuntimeError):
        _run(model, isolated_smoke, "prepersist")

    result_dir = _find_result_dir(isolated_smoke["results_root"], "prepersist")
    lifecycle = _lifecycle_state(result_dir)
    assert lifecycle["state"] == "partial_failed"
    states = [s["state"] for s in lifecycle["state_history"]]
    assert "scoring_complete" in states
    assert "complete" not in states

    entries = runtime.read_jsonl(result_dir / "dispatch_journal.jsonl")
    assert [e["event"] for e in entries] == [
        "request_started", "response_received", "request_started", "response_received",
    ]
    # final artifacts were never written -- persistence failed before they landed
    assert not (result_dir / "summary.json").exists()
    assert not (result_dir / "raw_run.jsonl").exists()
