"""Runner tests for evals/run_holdout_v3_e4.py, using FakeModel/mocks only.

Every test that exercises ``run_holdout_v3`` monkeypatches the module's path
constants onto an isolated ``tmp_path`` fixture (a handful of synthetic
files, never evals/holdout_v3/fixture). No real model is contacted and the
real Holdout v3 is never touched -- an autouse guard fails loudly if it
ever is.

Holdout v3 was consumed for real by an interrupted live run (see
``evals/results/holdout-v3-e4-current-20260813-3bf30aaf2665/``); that
consumption is permanent and is documented forever in the committed
``evals/holdout_v3/consumption_record.json``. ``evals/holdout_v3/CONSUMED.json``
itself, by contrast, is the local runtime marker the real runner checks
before permitting a run -- gitignored, present only on a machine that
actually ran it (this one), and never committed, so a fresh checkout does
not have it. The guard below checks that the local marker's *content* is
never mutated by these tests (tolerating either "absent" or "present and
unchanged", rather than asserting either state), since "these tests must
never touch the real marker" is what actually matters here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from evals import run_holdout_v3_e4 as runner

REPO_ROOT = Path(runner.__file__).resolve().parent.parent
REAL_CONSUMED_MARKER = REPO_ROOT / "evals" / "holdout_v3" / "CONSUMED.json"


def _real_marker_snapshot() -> bytes | None:
    return REAL_CONSUMED_MARKER.read_bytes() if REAL_CONSUMED_MARKER.exists() else None


@pytest.fixture(autouse=True)
def _guard_real_holdout_v3():
    before = _real_marker_snapshot()
    yield
    assert _real_marker_snapshot() == before, "real Holdout v3 consumption record was mutated by this test"


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.token_usage = None


class RecordingFakeModel:
    model_id = "fake/test-model"

    def __init__(self, decision_fn=None):
        self.calls = 0
        self.seen_prompts: list[str] = []
        self.decision_fn = decision_fn or self._default_decision

    @staticmethod
    def _default_decision(name):
        lower = name.lower()
        if "doc" in lower:
            return {"source": name, "decision": "classify", "category": "Documents"}
        if "img" in lower:
            return {"source": name, "decision": "classify", "category": "Images"}
        return {"source": name, "decision": "review", "category": None}

    def generate(self, messages, **kwargs):
        self.calls += 1
        text = messages[0]["content"][0]["text"]
        self.seen_prompts.append(text)
        start = text.index("<FILENAME_DATA>") + len("<FILENAME_DATA>")
        end = text.index("</FILENAME_DATA>")
        names = [n for n in text[start:end].strip().split("\n") if n]
        decisions = [self.decision_fn(n) for n in names]
        return FakeResponse(json.dumps({"decisions": decisions}))


CASES = [
    {"filename": "sample_doc_report", "expected_outcome": "Documents", "primary_stratum": "ordinary_realistic",
     "language": "en", "secondary_tags": [], "instruction_like": False,
     "rationale": "EVALUATOR_ONLY_RATIONALE_MARKER"},
    {"filename": "sample_img_snapshot", "expected_outcome": "Images", "primary_stratum": "ordinary_realistic",
     "language": "en", "secondary_tags": [], "instruction_like": False,
     "rationale": "EVALUATOR_ONLY_RATIONALE_MARKER"},
    {"filename": "sample_unclassifiable_thing", "expected_outcome": "_ToReview",
     "primary_stratum": "insufficient_metadata", "language": "en", "secondary_tags": [],
     "instruction_like": False, "rationale": "EVALUATOR_ONLY_RATIONALE_MARKER"},
]


def valid_identity() -> "runner.ModelIdentity":
    return runner.ModelIdentity(
        model_id=runner.EXPECTED_MODEL_ID,
        digest=runner.EXPECTED_MODEL_DIGEST,
        quantization=runner.EXPECTED_QUANTIZATION,
        temperature=runner.EXPECTED_TEMPERATURE,
        thinking_enabled=runner.EXPECTED_THINKING_ENABLED,
        num_ctx=runner.EXPECTED_NUM_CTX,
    )


@pytest.fixture
def isolated_holdout(tmp_path, monkeypatch):
    holdout_dir = tmp_path / "holdout_v3"
    fixture_dir = holdout_dir / "fixture"
    fixture_dir.mkdir(parents=True)
    names = [c["filename"] for c in CASES]
    for n in names:
        (fixture_dir / n).write_bytes(b"")

    expected_path = holdout_dir / "expected.yaml"
    expected_path.write_text(yaml.safe_dump({"cases": CASES}, allow_unicode=True), encoding="utf-8")

    dataset_listing = "\n".join(sorted(names)).encode("utf-8")
    manifest = {
        "total_files": len(names),
        "dataset_sha256": hashlib.sha256(dataset_listing).hexdigest(),
        "ground_truth_sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest(),
    }
    manifest_path = holdout_dir / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    real_files = {
        "src/tidy/classification.py": REPO_ROOT / "src" / "tidy" / "classification.py",
        "evals/post_holdout_candidates.py": REPO_ROOT / "evals" / "post_holdout_candidates.py",
        "evals/run_holdout_v3_e4.py": REPO_ROOT / "evals" / "run_holdout_v3_e4.py",
    }
    pins = {rel: hashlib.sha256(p.read_bytes()).hexdigest() for rel, p in real_files.items()}
    code_pins_path = holdout_dir / "code_pins.json"
    code_pins_path.write_text(json.dumps({"sha256": pins}), encoding="utf-8")

    results_root = tmp_path / "results"
    results_root.mkdir()
    consumed_marker = holdout_dir / "CONSUMED.json"

    monkeypatch.setattr(runner, "HOLDOUT_DIR", holdout_dir)
    monkeypatch.setattr(runner, "FIXTURE_DIR", fixture_dir)
    monkeypatch.setattr(runner, "EXPECTED_PATH", expected_path)
    monkeypatch.setattr(runner, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(runner, "CODE_PINS_PATH", code_pins_path)
    monkeypatch.setattr(runner, "CONSUMED_MARKER_PATH", consumed_marker)
    monkeypatch.setattr(runner, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(runner, "REPO_ROOT", REPO_ROOT)

    return {
        "holdout_dir": holdout_dir,
        "fixture_dir": fixture_dir,
        "manifest_path": manifest_path,
        "code_pins_path": code_pins_path,
        "consumed_marker": consumed_marker,
        "results_root": results_root,
        "names": names,
    }


# ---------------------------------------------------------------------------
# Candidate freeze
# ---------------------------------------------------------------------------


def test_selected_candidate_is_e4_current():
    assert runner.SELECTED_CANDIDATE == "E4-current"


def test_no_alternate_candidate_cli_option():
    parser = runner.build_arg_parser()
    option_strings = {s for action in parser._actions for s in action.option_strings}
    assert not any("candidate" in s.lower() for s in option_strings)
    assert option_strings == {"-h", "--help", "--freeze-tag", "--confirm-consumes-holdout-v3"}


def test_e4_refined_and_e5_never_referenced_in_runner_source():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "run_e4_refined" not in source
    assert "run_e5" not in source
    assert not hasattr(runner, "run_e4_refined")
    assert not hasattr(runner, "run_e5")


# ---------------------------------------------------------------------------
# E3 two-pass / zero extra E4 calls
# ---------------------------------------------------------------------------


def test_e3_two_pass_and_e4_adds_zero_model_calls(isolated_holdout):
    model = RecordingFakeModel()
    result_dir = runner.run_holdout_v3(model, freeze_tag="t1", model_identity=valid_identity())
    assert model.calls == 2
    assert (result_dir / "summary.json").exists()


# ---------------------------------------------------------------------------
# Pre-request integrity gates
# ---------------------------------------------------------------------------


def test_code_hash_mismatch_stops_before_request(isolated_holdout):
    pins_path = isolated_holdout["code_pins_path"]
    pins = json.loads(pins_path.read_text())
    pins["sha256"]["src/tidy/classification.py"] = "0" * 64
    pins_path.write_text(json.dumps(pins))

    model = RecordingFakeModel()
    with pytest.raises(runner.HoldoutIntegrityError):
        runner.run_holdout_v3(model, freeze_tag="t2", model_identity=valid_identity())
    assert model.calls == 0
    assert not runner.is_consumed()


def test_fixture_hash_mismatch_stops_before_request(isolated_holdout):
    manifest_path = isolated_holdout["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    model = RecordingFakeModel()
    with pytest.raises(runner.HoldoutIntegrityError):
        runner.run_holdout_v3(model, freeze_tag="t3", model_identity=valid_identity())
    assert model.calls == 0
    assert not runner.is_consumed()


def test_model_digest_mismatch_stops_before_request(isolated_holdout):
    bad_identity = runner.ModelIdentity(
        model_id=runner.EXPECTED_MODEL_ID,
        digest="deadbeef",
        quantization=runner.EXPECTED_QUANTIZATION,
        temperature=runner.EXPECTED_TEMPERATURE,
        thinking_enabled=runner.EXPECTED_THINKING_ENABLED,
        num_ctx=runner.EXPECTED_NUM_CTX,
    )
    model = RecordingFakeModel()
    with pytest.raises(runner.ModelIdentityError):
        runner.run_holdout_v3(model, freeze_tag="t4", model_identity=bad_identity)
    assert model.calls == 0
    assert not runner.is_consumed()


def test_result_directory_collision_rejected(isolated_holdout, monkeypatch):
    import time

    date_str = time.strftime("%Y%m%d")
    pre_existing = isolated_holdout["results_root"] / f"holdout-v3-e4-current-{date_str}-t5"
    pre_existing.mkdir()

    model = RecordingFakeModel()
    with pytest.raises(runner.ResultDirectoryExistsError):
        runner.run_holdout_v3(model, freeze_tag="t5", model_identity=valid_identity())
    assert model.calls == 0
    assert not runner.is_consumed()


# ---------------------------------------------------------------------------
# Fixture immutability
# ---------------------------------------------------------------------------


def test_no_fixture_mutation(isolated_holdout):
    fixture_dir = isolated_holdout["fixture_dir"]
    before = {p.name: p.stat().st_size for p in fixture_dir.iterdir()}
    model = RecordingFakeModel()
    runner.run_holdout_v3(model, freeze_tag="t6", model_identity=valid_identity())
    after = {p.name: p.stat().st_size for p in fixture_dir.iterdir()}
    assert before == after


# ---------------------------------------------------------------------------
# Consumption lifecycle
# ---------------------------------------------------------------------------


def test_consumed_lifecycle_begins_on_first_measured_request(isolated_holdout):
    assert not runner.is_consumed()
    model = RecordingFakeModel()
    runner.run_holdout_v3(model, freeze_tag="t7", model_identity=valid_identity())
    assert runner.is_consumed()
    record = json.loads(isolated_holdout["consumed_marker"].read_text())
    assert record["consumed"] is True
    assert "first_measured_request_timestamp" in record


def test_rerun_refused_after_consumption(isolated_holdout):
    model = RecordingFakeModel()
    runner.run_holdout_v3(model, freeze_tag="t8", model_identity=valid_identity())
    with pytest.raises(runner.HoldoutAlreadyConsumedError):
        runner.run_holdout_v3(model, freeze_tag="t8b", model_identity=valid_identity())


def test_post_request_failure_records_consumed_partial(isolated_holdout, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure after the measured request was issued")

    original_run_e4 = runner.run_e4

    def _spy_run_e4(model, metadata, real_categories, *, review_directory):
        model.generate([{"role": "user", "content": [{"type": "text", "text": "x <FILENAME_DATA></FILENAME_DATA>"}]}])
        raise RuntimeError("simulated post-request failure")

    monkeypatch.setattr(runner, "run_e4", _spy_run_e4)

    model = RecordingFakeModel()
    with pytest.raises(RuntimeError):
        runner.run_holdout_v3(model, freeze_tag="t9", model_identity=valid_identity())

    assert runner.is_consumed()
    record = json.loads(isolated_holdout["consumed_marker"].read_text())
    assert record["consumed"] is True

    result_dirs = list(isolated_holdout["results_root"].glob("holdout-v3-e4-current-*-t9"))
    assert len(result_dirs) == 1
    manifest = json.loads((result_dirs[0] / "manifest.json").read_text())
    assert manifest["consumed"] is True
    assert manifest["run_complete"] is False


def test_pre_request_failure_leaves_unconsumed(isolated_holdout):
    pins_path = isolated_holdout["code_pins_path"]
    pins = json.loads(pins_path.read_text())
    pins["sha256"]["evals/post_holdout_candidates.py"] = "0" * 64
    pins_path.write_text(json.dumps(pins))

    model = RecordingFakeModel()
    with pytest.raises(runner.HoldoutIntegrityError):
        runner.run_holdout_v3(model, freeze_tag="t10", model_identity=valid_identity())
    assert not runner.is_consumed()
    assert model.calls == 0


# ---------------------------------------------------------------------------
# Metadata isolation (item 44) -- prompts never carry evaluator-only fields
# ---------------------------------------------------------------------------


def test_load_fixture_metadata_carries_only_name(isolated_holdout):
    metadata = runner.load_fixture_metadata()
    for item in metadata:
        assert set(item) == {"name"}


def test_evaluator_fields_never_enter_prompt(isolated_holdout):
    model = RecordingFakeModel()
    runner.run_holdout_v3(model, freeze_tag="t11", model_identity=valid_identity())
    forbidden_strings = [
        "EVALUATOR_ONLY_RATIONALE_MARKER",
        "primary_stratum",
        "insufficient_metadata",
        "ordinary_realistic",
        "instruction_like",
        "rationale",
        "secondary_tags",
    ]
    for prompt in model.seen_prompts:
        for forbidden in forbidden_strings:
            assert forbidden not in prompt, forbidden


# ---------------------------------------------------------------------------
# Denominator (against the REAL, now-permanently-consumed fixture -- read-only)
# ---------------------------------------------------------------------------


def test_unique_denominator_exactly_120_on_real_fixture():
    """Read-only: the real fixture is unaffected by consumption -- it still
    has 120 unique files. Holdout v3 itself is permanently consumed (see the
    module docstring); that fact is asserted here against the committed
    ``consumption_record.json``, not ``runner.is_consumed()`` (the local
    runtime marker), since a fresh checkout does not carry the latter."""
    metadata = runner.load_fixture_metadata()
    names = [m["name"] for m in metadata]
    assert len(names) == 120
    assert len(set(names)) == 120
    record = json.loads(
        (runner.HOLDOUT_DIR / "consumption_record.json").read_text(encoding="utf-8")
    )
    assert record["consumed"] is True
