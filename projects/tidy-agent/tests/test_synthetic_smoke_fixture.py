"""Integrity tests for the synthetic, non-evidentiary smoke fixture.

Read-only: never touches evals/holdout_v3 or any other Holdout artifact.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = REPO_ROOT / "evals" / "synthetic_one_time_smoke"
FIXTURE_DIR = SMOKE_DIR / "fixture"
MANIFEST_PATH = SMOKE_DIR / "fixture_manifest.json"

REAL_CATEGORIES = ("Documents", "Code", "Images", "Archives", "Installers")
REVIEW_LABEL = "_ToReview"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fixture_names() -> list[str]:
    return [p.name for p in FIXTURE_DIR.iterdir() if p.is_file()]


def test_exactly_120_files(fixture_names):
    assert len(fixture_names) == 120


def test_manifest_marks_non_evidentiary(manifest):
    assert manifest["purpose"] == "infrastructure_smoke_only"
    assert manifest["evaluation_evidence"] is False
    assert manifest["rerunnable"] is True


def test_distribution_20_per_family(manifest):
    counts = {cat: 0 for cat in REAL_CATEGORIES}
    review_count = 0
    for case in manifest["cases"]:
        if case["intended_category"] == REVIEW_LABEL:
            review_count += 1
        else:
            counts[case["intended_category"]] += 1
    assert counts == {cat: 20 for cat in REAL_CATEGORIES}
    assert review_count == 20


def test_all_files_zero_byte():
    for path in FIXTURE_DIR.iterdir():
        if path.is_file():
            assert path.stat().st_size == 0, path.name


def test_all_files_extensionless():
    for path in FIXTURE_DIR.iterdir():
        if path.is_file():
            assert Path(path.name).suffix == "", path.name


def test_all_names_nfc_normalized(fixture_names):
    for name in fixture_names:
        assert name == unicodedata.normalize("NFC", name), name


def test_all_names_cross_platform_safe(fixture_names):
    forbidden = set('<>:"/\\|?*')
    for name in fixture_names:
        assert not (forbidden & set(name)), name
        assert not name.endswith(" ") and not name.endswith("."), name


def test_no_collision(fixture_names):
    keys = [unicodedata.normalize("NFC", n).casefold() for n in fixture_names]
    assert len(keys) == len(set(keys))


def test_names_use_synthetic_template(fixture_names):
    for name in fixture_names:
        assert name.startswith("synthetic_"), name
        assert "_case_" in name


def test_manifest_hash_matches_live_fixture(manifest, fixture_names):
    listing = "\n".join(sorted(fixture_names)).encode("utf-8")
    assert hashlib.sha256(listing).hexdigest() == manifest["dataset_sha256"]


def test_fixture_and_manifest_agree(manifest, fixture_names):
    assert set(fixture_names) == {c["filename"] for c in manifest["cases"]}


def test_never_touches_holdout_v3():
    """This module and its fixture must have no path overlap with Holdout v3."""
    holdout_v3_dir = REPO_ROOT / "evals" / "holdout_v3"
    assert not str(SMOKE_DIR).startswith(str(holdout_v3_dir))
    assert not str(FIXTURE_DIR).startswith(str(holdout_v3_dir))
