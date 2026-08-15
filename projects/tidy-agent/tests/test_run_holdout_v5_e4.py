"""Failure-injection and lifecycle tests for the Holdout v5 runner.

Mocks only -- these tests never call a real provider and never touch the real
``evals/holdout_v5/`` artifacts or the real ``CONSUMED.json`` marker. An autouse
guard fails loudly if a test ever does. All fixtures, ground truth, pins, and
candidate selection here are synthetic and isolated to ``tmp_path``; none of the
real Holdout v5 case material exists yet, and none is created by these tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import run_holdout_v5_e4 as v5
from evals import one_time_eval_runtime as runtime

REPO_ROOT = Path(v5.__file__).resolve().parent.parent
REAL_V5_CONSUMED = REPO_ROOT / "evals" / "holdout_v5" / "CONSUMED.json"
REAL_HOLDOUT_V5_DIR = REPO_ROOT / "evals" / "holdout_v5"
PROTECTED_FILES = [
    REPO_ROOT / "src" / "tidy" / "classification.py",
    REPO_ROOT / "evals" / "e4_batched.py",
    REPO_ROOT / "evals" / "run_holdout_v5_e4.py",
    REPO_ROOT / "evals" / "post_holdout_candidates.py",
    REPO_ROOT / "evals" / "one_time_eval_runtime.py",
]


def _hash_all(paths):
    return {p: (p.read_bytes() if p.exists() else None) for p in paths}


@pytest.fixture(autouse=True)
def _guard_holdout_v5_and_production():
    # The v5 design must not create fake cases, so these tests assert the real
    # directory has no fixture/consumed marker before and after.
    assert not REAL_V5_CONSUMED.exists(), "Holdout v5 must remain unconsumed before tests run"
    assert not (REAL_HOLDOUT_V5_DIR / "fixture").exists(), "no v5 fixture may be authored here"
    assert not (REAL_HOLDOUT_V5_DIR / "ground_truth.json").exists()
    before = _hash_all(PROTECTED_FILES)
    yield
    assert not REAL_V5_CONSUMED.exists(), "a test wrote the real Holdout v5 consumption marker"
    assert not (REAL_HOLDOUT_V5_DIR / "fixture").exists()
    assert _hash_all(PROTECTED_FILES) == before, "a protected file was mutated"


class ReviewOnlyFake:
    """Content-free: returns a valid review decision for every batch source."""
    structured_output_mode = "json_schema"

    def generate(self, messages, **kwargs):
        text = messages[0]["content"][0]["text"]
        start = text.index("<FILENAME_DATA>") + len("<FILENAME_DATA>")
        end = text.index("</FILENAME_DATA>")
        names = [n for n in text[start:end].strip().split("\n") if n]
        return SimpleNamespace(
            content=json.dumps({
                "decisions": [
                    {"source": n, "decision": "review", "category": None} for n in names
                ]
            }),
            token_usage=SimpleNamespace(input_tokens=4, output_tokens=2),
        )


@pytest.fixture
def sandbox(tmp_path):
    sb = {
        "holdout": tmp_path / "holdout_v5",
        "results": tmp_path / "evals" / "results",
    }
    sb["holdout"].mkdir(parents=True)
    sb["results"].mkdir(parents=True)
    fixture = sb["holdout"] / "fixture"
    fixture.mkdir()
    names = [f"case_{i:03d}" for i in range(5)]
    for n in names:
        (fixture / n).write_bytes(b"")
    ground_truth = {n: "_ToReview" for n in names}
    sb["ground_truth_path"] = sb["holdout"] / "ground_truth.json"
    sb["ground_truth_path"].write_text(json.dumps(ground_truth), encoding="utf-8")
    sb["fixture_dir"] = fixture
    sb["names"] = names
    sb["candidate_selection_path"] = sb["holdout"] / "candidate_selection.json"
    sb["candidate_selection_path"].write_text(json.dumps({
        "selected_candidate": v5.CANDIDATE_DESIGNATION,
        "candidate_pipeline": v5.CANDIDATE_PIPELINE,
        "selection_closed": True,
        "holdout_v5_inference_performed": False,
    }), encoding="utf-8")
    sb["authoring_frozen_path"] = sb["holdout"] / "AUTHORING_FROZEN.json"
    sb["authoring_frozen_path"].write_text(json.dumps({
        "authoring_frozen": True,
        "case_count": len(names),
        "dataset_sha256": v5._fixture_hash(names),
        "ground_truth_sha256": hashlib.sha256(sb["ground_truth_path"].read_bytes()).hexdigest(),
        "historical_audit_performed": True,
        "model_inference_performed": False,
    }), encoding="utf-8")
    sb["code_pins_path"] = sb["holdout"] / "code_pins.json"
    pins = {
        "src/tidy/classification.py": hashlib.sha256((REPO_ROOT / "src/tidy/classification.py").read_bytes()).hexdigest(),
        "evals/e4_batched.py": hashlib.sha256((REPO_ROOT / "evals/e4_batched.py").read_bytes()).hexdigest(),
    }
    sb["code_pins_path"].write_text(json.dumps({"pins": pins}), encoding="utf-8")
    sb["consumed_path"] = sb["holdout"] / "CONSUMED.json"
    sb["consumption_record_path"] = sb["holdout"] / "consumption_record.json"
    sb["fixture_manifest_path"] = sb["holdout"] / "fixture_manifest.json"
    sb["fixture_manifest_path"].write_text(json.dumps({"frozen": True}), encoding="utf-8")
    return sb


def _identity(**overrides):
    base = {
        "model_id": v5.EXPECTED_MODEL_ID,
        "digest": v5.EXPECTED_MODEL_DIGEST,
        "quantization": v5.EXPECTED_QUANTIZATION,
        "temperature": v5.EXPECTED_TEMPERATURE,
        "thinking_enabled": v5.EXPECTED_THINKING_ENABLED,
        "num_ctx": v5.EXPECTED_NUM_CTX,
    }
    base.update(overrides)
    return v5.ModelIdentity(**base)


def _run(sandbox, model, *, expected=None, actual=None):
    return v5.run_holdout_v5(
        model,
        run_label="sandbox-test",
        expected_identity=expected or _identity(),
        actual_identity=actual or _identity(),
        fixture_dir=sandbox["fixture_dir"],
        ground_truth_path=sandbox["ground_truth_path"],
        code_pins_path=sandbox["code_pins_path"],
        candidate_selection_path=sandbox["candidate_selection_path"],
        consumed_path=sandbox["consumed_path"],
        consumption_record_path=sandbox["consumption_record_path"],
        results_root=sandbox["results"],
        repo_root=REPO_ROOT,
        batch_size=20,
    )