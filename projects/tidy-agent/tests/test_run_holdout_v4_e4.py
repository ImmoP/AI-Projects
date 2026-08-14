"""Failure-injection and lifecycle tests for the frozen Holdout v4 runner.

FakeModel/mocks only -- these tests never call a real provider and never
touch the real evals/holdout_v4/CONSUMED.json (which must not exist) or any
protected production/candidate file. An autouse guard fails loudly if a
test ever does. All fixtures and ground truth here are synthetic and
isolated to tmp_path; none of the real Holdout v4 case material is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from evals import one_time_eval_runtime as runtime
from evals import run_holdout_v4_e4 as v4runner

REPO_ROOT = Path(v4runner.__file__).resolve().parent.parent
REAL_HOLDOUT_V4_CONSUMED = REPO_ROOT / "evals" / "holdout_v4" / "CONSUMED.json"
PROTECTED_FILES = [
    REPO_ROOT / "src" / "tidy" / "classification.py",
    REPO_ROOT / "src" / "tidy" / "cli.py",
    REPO_ROOT / "config" / "rules.yaml",
    REPO_ROOT / "evals" / "post_holdout_candidates.py",
    REPO_ROOT / "evals" / "one_time_eval_runtime.py",
    REPO_ROOT / "evals" / "run_holdout_v4_e4.py",
    REPO_ROOT / "evals" / "holdout_v3" / "CONSUMED.json",
]


def _hash_all(paths):
    return {p: (p.read_bytes() if p.exists() else None) for p in paths}


@pytest.fixture(autouse=True)
def _guard_holdout_v4_and_production():
    assert not REAL_HOLDOUT_V4_CONSUMED.exists(), "Holdout v4 must remain unconsumed before tests run"
    before = _hash_all(PROTECTED_FILES)
    yield
    assert not REAL_HOLDOUT_V4_CONSUMED.exists(), "a test wrote the real Holdout v4 consumption marker"
    assert _hash_all(PROTECTED_FILES) == before, "a protected production/candidate/runtime file was mutated"


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.token_usage = None


class FaultInjectingFakeModel:
    """Explicit classify/review schema producer with configurable faults.

    ``fail_on_ordinal``: raise instead of returning (exercises the frozen
    ClassificationBackend's internal catch-and-fallback).
    ``malformed_json_on_ordinal``: return unparseable text (parse failure).
    ``bad_schema_on_ordinal``: return valid JSON with the wrong shape
    (schema-validation failure).
    """

    model_id = "fake/test-model"

    def __init__(self, *, fail_on_ordinal=None, malformed_json_on_ordinal=None, bad_schema_on_ordinal=None):
        self.calls = 0
        self.fail_on_ordinal = fail_on_ordinal
        self.malformed_json_on_ordinal = malformed_json_on_ordinal
        self.bad_schema_on_ordinal = bad_schema_on_ordinal

    @staticmethod
    def _decide(name):
        if "doc" in name:
            return {"source": name, "decision": "classify", "category": "Documents"}
        if "img" in name:
            return {"source": name, "decision": "classify", "category": "Images"}
        return {"source": name, "decision": "review", "category": None}

    def generate(self, messages, **kwargs):
        self.calls += 1
        ordinal = self.calls
        if self.fail_on_ordinal == ordinal:
            raise RuntimeError(f"simulated provider failure on call {ordinal}")
        text = messages[0]["content"][0]["text"]
        start = text.index("<FILENAME_DATA>") + len("<FILENAME_DATA>")
        end = text.index("</FILENAME_DATA>")
        names = [n for n in text[start:end].strip().split("\n") if n]
        if self.malformed_json_on_ordinal == ordinal:
            return FakeResponse("{not valid json")
        if self.bad_schema_on_ordinal == ordinal:
            return FakeResponse(json.dumps({"wrong_top_level_key": []}))
        decisions = [self._decide(n) for n in names]
        return FakeResponse(json.dumps({"decisions": decisions}))


CASES = [
    {"filename": "isolated_doc_case_alpha", "expected_outcome": "Documents"},
    {"filename": "isolated_img_case_beta", "expected_outcome": "Images"},
    {"filename": "isolated_review_case_gamma", "expected_outcome": "_ToReview"},
]


@pytest.fixture
def sandbox(tmp_path):
    repo_root = tmp_path / "repo"
    fixture_dir = repo_root / "evals" / "holdout_v4" / "fixture"
    fixture_dir.mkdir(parents=True)
    for c in CASES:
        (fixture_dir / c["filename"]).write_bytes(b"")
    ground_truth_path = repo_root / "evals" / "holdout_v4" / "ground_truth.json"
    ground_truth_path.write_text(json.dumps(CASES), encoding="utf-8")

    pinned_rel_paths = [
        "src/tidy/classification.py",
        "evals/post_holdout_candidates.py",
        "evals/one_time_eval_runtime.py",
        "evals/run_holdout_v4_e4.py",
    ]
    pins = {}
    for rel in pinned_rel_paths:
        real_path = REPO_ROOT / rel
        content = real_path.read_bytes()
        sandbox_path = repo_root / rel
        sandbox_path.parent.mkdir(parents=True, exist_ok=True)
        sandbox_path.write_bytes(content)
        pins[rel] = hashlib.sha256(content).hexdigest()
    code_pins_path = repo_root / "evals" / "holdout_v4" / "code_pins.json"
    code_pins_path.write_text(json.dumps({"final_protocol_version": 1, "pins": pins}), encoding="utf-8")

    consumed_path = repo_root / "evals" / "holdout_v4" / "CONSUMED.json"
    results_root = repo_root / "evals" / "results"
    results_root.mkdir(parents=True)
    return {
        "repo_root": repo_root,
        "fixture_dir": fixture_dir,
        "ground_truth_path": ground_truth_path,
        "code_pins_path": code_pins_path,
        "consumed_path": consumed_path,
        "results_root": results_root,
    }


def _identity():
    return v4runner.expected_model_identity()


def _run(model, sandbox, run_label):
    return v4runner.run_holdout_v4(
        model,
        run_label=run_label,
        expected_identity=_identity(),
        actual_identity=_identity(),
        fixture_dir=sandbox["fixture_dir"],
        ground_truth_path=sandbox["ground_truth_path"],
        code_pins_path=sandbox["code_pins_path"],
        consumed_path=sandbox["consumed_path"],
        results_root=sandbox["results_root"],
        repo_root=sandbox["repo_root"],
    )


def _status(result_dir):
    return json.loads((result_dir / "evaluation_status.json").read_text())


def _lifecycle(result_dir):
    return json.loads((result_dir / "lifecycle.json").read_text())


# ---------------------------------------------------------------------------
# Success -> complete_valid
# ---------------------------------------------------------------------------


def test_success_yields_complete_valid_and_consumes(sandbox):
    model = FaultInjectingFakeModel()
    result_dir = _run(model, sandbox, "happy")
    status = _status(result_dir)
    assert status["evaluation_valid"] is True
    assert status["evaluation_status"] == "complete_valid"
    assert status["provider_errors"] == 0
    assert status["parse_failures"] == 0
    assert status["schema_failures"] == 0
    assert status["all_required_provider_responses_received"] is True
    assert status["E3_gate_completed"] is True
    assert status["E4_current_completed"] is True
    assert status["result_persistence_completed"] is True
    assert _lifecycle(result_dir)["state"] == "complete"
    assert sandbox["consumed_path"].exists()


# ---------------------------------------------------------------------------
# Provider failure hidden by the frozen backend's own fallback
# ---------------------------------------------------------------------------


def test_provider_failure_hidden_by_backend_fallback_yields_partial_inconclusive(sandbox):
    model = FaultInjectingFakeModel(fail_on_ordinal=1)
    result_dir = _run(model, sandbox, "providerfail")  # does not raise: caught inside ClassificationBackend
    status = _status(result_dir)
    assert status["provider_errors"] >= 1
    assert status["evaluation_valid"] is False
    assert status["evaluation_status"] == "partial_inconclusive"
    # the frozen backend's fail-safe design still reaches lifecycle "complete"
    assert _lifecycle(result_dir)["state"] == "complete"
    assert sandbox["consumed_path"].exists()  # consumption already happened before dispatch


def test_provider_failure_leaves_a_response_missing(sandbox):
    model = FaultInjectingFakeModel(fail_on_ordinal=2)
    result_dir = _run(model, sandbox, "missingresponse")
    status = _status(result_dir)
    assert status["all_required_provider_responses_received"] is False
    assert status["evaluation_status"] == "partial_inconclusive"


# ---------------------------------------------------------------------------
# Parse failure
# ---------------------------------------------------------------------------


def test_parse_failure_yields_partial_inconclusive(sandbox):
    model = FaultInjectingFakeModel(malformed_json_on_ordinal=1)
    result_dir = _run(model, sandbox, "parsefail")
    status = _status(result_dir)
    assert status["parse_failures"] >= 1
    assert status["evaluation_valid"] is False
    assert status["evaluation_status"] == "partial_inconclusive"


# ---------------------------------------------------------------------------
# Schema failure
# ---------------------------------------------------------------------------


def test_schema_failure_yields_partial_inconclusive(sandbox):
    model = FaultInjectingFakeModel(bad_schema_on_ordinal=1)
    result_dir = _run(model, sandbox, "schemafail")
    status = _status(result_dir)
    assert status["schema_failures"] >= 1
    assert status["evaluation_valid"] is False
    assert status["evaluation_status"] == "partial_inconclusive"


# ---------------------------------------------------------------------------
# Model mismatch -> STOP unconsumed
# ---------------------------------------------------------------------------


def test_model_identity_mismatch_stops_unconsumed(sandbox):
    bad_identity = runtime.ModelIdentity(
        model_id=v4runner.EXPECTED_MODEL_ID, digest="deadbeef",
        quantization=v4runner.EXPECTED_QUANTIZATION, temperature=v4runner.EXPECTED_TEMPERATURE,
        thinking_enabled=v4runner.EXPECTED_THINKING_ENABLED, num_ctx=v4runner.EXPECTED_NUM_CTX,
    )
    model = FaultInjectingFakeModel()
    with pytest.raises(runtime.ModelIdentityError):
        v4runner.run_holdout_v4(
            model, run_label="badid", expected_identity=_identity(), actual_identity=bad_identity,
            fixture_dir=sandbox["fixture_dir"], ground_truth_path=sandbox["ground_truth_path"],
            code_pins_path=sandbox["code_pins_path"], consumed_path=sandbox["consumed_path"],
            results_root=sandbox["results_root"], repo_root=sandbox["repo_root"],
        )
    assert model.calls == 0
    assert not sandbox["consumed_path"].exists()
    assert not list(sandbox["results_root"].iterdir())


# ---------------------------------------------------------------------------
# Code pin mismatch -> STOP unconsumed
# ---------------------------------------------------------------------------


def test_code_pin_mismatch_stops_unconsumed(sandbox):
    pins = json.loads(sandbox["code_pins_path"].read_text())
    pins["pins"]["src/tidy/classification.py"] = "0" * 64
    sandbox["code_pins_path"].write_text(json.dumps(pins), encoding="utf-8")
    model = FaultInjectingFakeModel()
    with pytest.raises(v4runner.CodePinMismatchError):
        _run(model, sandbox, "pinmismatch")
    assert model.calls == 0
    assert not sandbox["consumed_path"].exists()
    assert not list(sandbox["results_root"].iterdir())


# ---------------------------------------------------------------------------
# Result collision -> STOP unconsumed
# ---------------------------------------------------------------------------


def test_result_directory_collision_stops_unconsumed(sandbox):
    import time

    date_str = time.strftime("%Y%m%d")
    pre_existing = sandbox["results_root"] / f"holdout-v4-e4-{date_str}-collide"
    pre_existing.mkdir()
    model = FaultInjectingFakeModel()
    with pytest.raises(runtime.ResultDirectoryExistsError):
        _run(model, sandbox, "collide")
    assert model.calls == 0
    assert not sandbox["consumed_path"].exists()


def test_no_fixture_mutation(sandbox):
    before = {p.name: p.stat().st_size for p in sandbox["fixture_dir"].iterdir()}
    _run(FaultInjectingFakeModel(), sandbox, "nomut")
    after = {p.name: p.stat().st_size for p in sandbox["fixture_dir"].iterdir()}
    assert before == after


# ---------------------------------------------------------------------------
# CLI: no-argument invocation must not consume
# ---------------------------------------------------------------------------


def test_cli_no_arguments_does_not_consume_and_returns_zero(capsys):
    exit_code = v4runner.main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "usage" in captured.out.lower()
    assert not REAL_HOLDOUT_V4_CONSUMED.exists()


def test_cli_help_exits_zero_and_does_not_consume(capsys):
    with pytest.raises(SystemExit) as exc_info:
        v4runner.main(["--help"])
    assert exc_info.value.code == 0
    assert not REAL_HOLDOUT_V4_CONSUMED.exists()


def test_cli_help_works_through_python_dash_m(tmp_path):
    """Proves the originally reported defect is fixed: `python -m
    evals.run_holdout_v4_e4 --help` must succeed (module import resolves),
    while direct-path execution is deliberately left unsupported."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "evals.run_holdout_v4_e4", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    assert not REAL_HOLDOUT_V4_CONSUMED.exists()


def test_direct_path_execution_remains_unsupported(tmp_path):
    """No path-manipulation workaround was added: running the file directly
    (not as a module) must still fail exactly as originally reported."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "evals" / "run_holdout_v4_e4.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr
    assert "evals" in result.stderr


def test_preflight_and_run_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        v4runner.main(["--preflight", "--run"])


# ---------------------------------------------------------------------------
# CLI: --preflight (mocked provider/model metadata, real run_preflight logic)
# ---------------------------------------------------------------------------


class FakeModelObj:
    def __init__(self, model_id, *, temperature=0.0, think=False, num_ctx=8192):
        self.model_id = model_id
        self.kwargs = {"temperature": temperature, "think": think, "num_ctx": num_ctx}


def _write_authoring_frozen(sandbox):
    dataset_sha256, ground_truth_sha256 = v4runner._recompute_fixture_hashes(
        sandbox["fixture_dir"], sandbox["ground_truth_path"]
    )
    holdout_dir = sandbox["repo_root"] / "evals" / "holdout_v4"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    (holdout_dir / "AUTHORING_FROZEN.json").write_text(
        json.dumps({"dataset_sha256": dataset_sha256, "ground_truth_sha256": ground_truth_sha256}),
        encoding="utf-8",
    )
    return holdout_dir


def _preflight(sandbox, holdout_dir, *, build_model_fn, run_label="preflight-test"):
    return v4runner.run_preflight(
        run_label=run_label,
        fixture_dir=sandbox["fixture_dir"],
        ground_truth_path=sandbox["ground_truth_path"],
        code_pins_path=sandbox["code_pins_path"],
        consumed_path=sandbox["consumed_path"],
        results_root=sandbox["results_root"],
        repo_root=sandbox["repo_root"],
        holdout_dir=holdout_dir,
        build_model_fn=build_model_fn,
    )


def _patch_ollama(monkeypatch, *, entries, local=True, endpoint="http://127.0.0.1:11434"):
    monkeypatch.setattr(v4runner, "resolved_model_endpoint", lambda model_id: (model_id, endpoint))
    monkeypatch.setattr(v4runner, "endpoint_is_local", lambda model_id: local)
    monkeypatch.setattr(v4runner, "fetch_ollama_tags", lambda ep, timeout=5.0: entries)


def _matching_entry():
    return {
        "model": "qwen3.5:4b",
        "digest": v4runner.EXPECTED_MODEL_DIGEST,
        "details": {"quantization_level": v4runner.EXPECTED_QUANTIZATION},
    }


def test_preflight_pass_with_exact_expected_identity(sandbox, monkeypatch):
    holdout_dir = _write_authoring_frozen(sandbox)
    _patch_ollama(monkeypatch, entries=[_matching_entry()])
    model = FakeModelObj(v4runner.EXPECTED_MODEL_ID)
    result = _preflight(sandbox, holdout_dir, build_model_fn=lambda: model)
    assert result.passed is True
    assert result.checks["actual_digest_matches"] is True
    assert result.checks["actual_quantization_matches"] is True
    assert result.checks["expected_model_installed_locally"] is True
    assert not sandbox["consumed_path"].exists()
    assert not list(sandbox["results_root"].iterdir())


def test_preflight_fails_on_digest_mismatch(sandbox, monkeypatch):
    holdout_dir = _write_authoring_frozen(sandbox)
    bad_entry = _matching_entry()
    bad_entry["digest"] = "0" * 64
    _patch_ollama(monkeypatch, entries=[bad_entry])
    model = FakeModelObj(v4runner.EXPECTED_MODEL_ID)
    result = _preflight(sandbox, holdout_dir, build_model_fn=lambda: model)
    assert result.passed is False
    assert result.checks["actual_digest_matches"] is False


def test_preflight_fails_when_model_missing_locally(sandbox, monkeypatch):
    holdout_dir = _write_authoring_frozen(sandbox)
    _patch_ollama(monkeypatch, entries=[])  # no models installed
    model = FakeModelObj(v4runner.EXPECTED_MODEL_ID)
    result = _preflight(sandbox, holdout_dir, build_model_fn=lambda: model)
    assert result.passed is False
    assert result.checks["expected_model_installed_locally"] is False
    assert result.checks["actual_digest_matches"] is False


def test_preflight_fails_on_code_pin_mismatch(sandbox, monkeypatch):
    holdout_dir = _write_authoring_frozen(sandbox)
    _patch_ollama(monkeypatch, entries=[_matching_entry()])
    pins = json.loads(sandbox["code_pins_path"].read_text())
    pins["pins"]["src/tidy/classification.py"] = "0" * 64
    sandbox["code_pins_path"].write_text(json.dumps(pins), encoding="utf-8")
    model = FakeModelObj(v4runner.EXPECTED_MODEL_ID)
    result = _preflight(sandbox, holdout_dir, build_model_fn=lambda: model)
    assert result.passed is False
    assert result.checks["code_pins_match"] is False


def test_preflight_fails_when_already_consumed(sandbox, monkeypatch):
    holdout_dir = _write_authoring_frozen(sandbox)
    _patch_ollama(monkeypatch, entries=[_matching_entry()])
    sandbox["consumed_path"].write_text(json.dumps({"consumed": True}), encoding="utf-8")
    model = FakeModelObj(v4runner.EXPECTED_MODEL_ID)
    result = _preflight(sandbox, holdout_dir, build_model_fn=lambda: model)
    assert result.passed is False
    assert result.checks["holdout_not_consumed"] is False


def test_preflight_never_calls_model_or_creates_artifacts(sandbox, monkeypatch):
    holdout_dir = _write_authoring_frozen(sandbox)
    _patch_ollama(monkeypatch, entries=[_matching_entry()])
    model = FakeModelObj(v4runner.EXPECTED_MODEL_ID)
    assert not hasattr(model, "generate")  # preflight must never even need a generate() method
    _preflight(sandbox, holdout_dir, build_model_fn=lambda: model)
    assert not sandbox["consumed_path"].exists()
    assert not list(sandbox["results_root"].iterdir())


def test_preflight_reports_endpoint_locality_as_diagnostic_only(sandbox, monkeypatch):
    """A reachable non-loopback endpoint (e.g. a private-network Ollama host)
    with exact digest/quantization/model match still PASSes overall -- only
    strict-loopback reachability is diagnostic, not gating."""
    holdout_dir = _write_authoring_frozen(sandbox)
    _patch_ollama(monkeypatch, entries=[_matching_entry()], local=False, endpoint="http://100.64.0.5:11434")
    model = FakeModelObj(v4runner.EXPECTED_MODEL_ID)
    result = _preflight(sandbox, holdout_dir, build_model_fn=lambda: model)
    assert result.checks["endpoint_is_local"] is False
    assert result.passed is True


# ---------------------------------------------------------------------------
# CLI: --run orchestration (mocked collaborators; never touches real paths)
# ---------------------------------------------------------------------------


def _canned_preflight(passed, result_dir_path=Path("/tmp/does-not-matter")):
    return v4runner.PreflightResult(passed=passed, checks={}, errors={}, result_dir_path=result_dir_path)


def test_run_delegates_exactly_once_to_run_holdout_v4(monkeypatch):
    calls = []

    def fake_run_holdout_v4(model, *, run_label, expected_identity, actual_identity):
        calls.append((model, run_label, expected_identity, actual_identity))
        return Path("/tmp/fake-result-dir")

    monkeypatch.setattr(v4runner, "run_preflight", lambda run_label: _canned_preflight(True))
    monkeypatch.setattr(v4runner, "build_frozen_eval_model", lambda: FakeModelObj(v4runner.EXPECTED_MODEL_ID))
    monkeypatch.setattr(v4runner, "resolve_actual_identity", lambda model: v4runner.expected_model_identity())
    monkeypatch.setattr(v4runner, "run_holdout_v4", fake_run_holdout_v4)

    exit_code = v4runner._cmd_run("orchestration-test")
    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0][1] == "orchestration-test"
    assert not REAL_HOLDOUT_V4_CONSUMED.exists()


def test_run_preflight_failure_prevents_delegation(monkeypatch):
    calls = []
    monkeypatch.setattr(v4runner, "run_preflight", lambda run_label: _canned_preflight(False))
    monkeypatch.setattr(v4runner, "build_frozen_eval_model", lambda: FakeModelObj(v4runner.EXPECTED_MODEL_ID))
    monkeypatch.setattr(v4runner, "resolve_actual_identity", lambda model: v4runner.expected_model_identity())
    monkeypatch.setattr(v4runner, "run_holdout_v4", lambda *a, **k: calls.append(1) or Path("/tmp/x"))

    exit_code = v4runner._cmd_run("should-not-run")
    assert exit_code == 1
    assert calls == []
    assert not REAL_HOLDOUT_V4_CONSUMED.exists()


def test_run_cli_flag_invokes_cmd_run_exactly_once(monkeypatch):
    calls = []
    monkeypatch.setattr(v4runner, "run_preflight", lambda run_label: _canned_preflight(True))
    monkeypatch.setattr(v4runner, "build_frozen_eval_model", lambda: FakeModelObj(v4runner.EXPECTED_MODEL_ID))
    monkeypatch.setattr(v4runner, "resolve_actual_identity", lambda model: v4runner.expected_model_identity())
    monkeypatch.setattr(
        v4runner, "run_holdout_v4",
        lambda model, *, run_label, expected_identity, actual_identity: (calls.append(run_label) or Path("/tmp/x")),
    )
    exit_code = v4runner.main(["--run", "--run-label", "cli-orchestration"])
    assert exit_code == 0
    assert calls == ["cli-orchestration"]
    assert not REAL_HOLDOUT_V4_CONSUMED.exists()
