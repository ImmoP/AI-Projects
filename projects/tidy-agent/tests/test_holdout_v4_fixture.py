"""Offline structural integrity tests for the frozen Holdout v4 fixture.

No model inference, no historical fixture access -- these only check the
already-frozen ``evals/holdout_v4/`` artifacts against the final portfolio
holdout protocol's required structural composition (see
``evals/final_portfolio_holdout_protocol.md`` and
``evals/holdout_v4/fixture_manifest.json``).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_DIR = REPO_ROOT / "evals" / "holdout_v4"
FIXTURE_DIR = HOLDOUT_DIR / "fixture"
GROUND_TRUTH_PATH = HOLDOUT_DIR / "ground_truth.json"

REAL_CATEGORIES = ("Documents", "Code", "Images", "Archives", "Installers")
REVIEW = "_ToReview"
REAL_STRATA = ("ordinary_realistic", "relational_semantics", "distractor_resolvable")
REVIEW_STRATA = ("insufficient_metadata", "latent_dual_role", "container_role_ambiguity")
MARKERS = ("unclear", "maybe", "unknown", "unsure", "ambiguous")


@pytest.fixture(scope="module")
def records():
    return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fixture_files():
    return sorted(p for p in FIXTURE_DIR.iterdir() if p.is_file())


def test_total_case_count(records):
    assert len(records) == 150


def test_fixture_directory_matches_ground_truth(records, fixture_files):
    assert {p.name for p in fixture_files} == {r["filename"] for r in records}
    assert len(fixture_files) == 150


def test_real_and_review_split(records):
    real = [r for r in records if r["expected_outcome"] != REVIEW]
    review = [r for r in records if r["expected_outcome"] == REVIEW]
    assert len(real) == 90
    assert len(review) == 60


def test_real_category_distribution(records):
    for cat in REAL_CATEGORIES:
        assert sum(1 for r in records if r["expected_outcome"] == cat) == 18


def test_real_strata_distribution(records):
    for stratum in REAL_STRATA:
        assert sum(
            1 for r in records if r["expected_outcome"] != REVIEW and r["primary_stratum"] == stratum
        ) == 30


def test_real_strata_per_category_exactly_six(records):
    for cat in REAL_CATEGORIES:
        for stratum in REAL_STRATA:
            count = sum(
                1
                for r in records
                if r["expected_outcome"] == cat and r["primary_stratum"] == stratum
            )
            assert count == 6, (cat, stratum, count)


def test_review_strata_distribution(records):
    for stratum in REVIEW_STRATA:
        assert sum(
            1 for r in records if r["expected_outcome"] == REVIEW and r["primary_stratum"] == stratum
        ) == 20


def test_review_marker_free_minimum(records):
    review = [r for r in records if r["expected_outcome"] == REVIEW]
    marker_free = sum(1 for r in review if r["ambiguity_marker_free"] is True)
    assert marker_free >= 50
    # Independently re-derive from the raw filename text, not the stored flag.
    rederived = sum(
        1 for r in review if not any(m in r["filename"].lower() for m in MARKERS)
    )
    assert rederived >= 50
    assert rederived == marker_free


def test_non_english_minimum_and_language_diversity(records):
    non_english = [r for r in records if r["language"] != "en"]
    assert len(non_english) >= 75
    languages = {r["language"] for r in records}
    assert "en" in languages
    assert len(languages) >= 8


def test_no_single_non_english_language_exceeds_twenty_percent(records):
    from collections import Counter

    counts = Counter(r["language"] for r in records if r["language"] != "en")
    cap = 0.20 * len(records)
    for lang, count in counts.items():
        assert count <= cap, (lang, count, cap)


def test_morphology_minimum(records):
    assert sum(1 for r in records if r["morphology"]) >= 25


def test_instruction_like_exact_split(records):
    instr = [r for r in records if r["instruction_like"]]
    assert len(instr) == 12
    assert sum(1 for r in instr if r["expected_outcome"] != REVIEW) == 5
    assert sum(1 for r in instr if r["expected_outcome"] == REVIEW) == 7


def test_short_and_long_name_counts_and_mix(records):
    short = [r for r in records if r["short_name"]]
    long_ = [r for r in records if r["long_name"]]
    assert len(short) == 15
    assert len(long_) == 15
    assert not ({r["filename"] for r in short} & {r["filename"] for r in long_})
    assert any(r["expected_outcome"] != REVIEW for r in short)
    assert any(r["expected_outcome"] == REVIEW for r in short)
    assert any(r["expected_outcome"] != REVIEW for r in long_)
    assert any(r["expected_outcome"] == REVIEW for r in long_)


def test_short_and_long_are_the_extremes_by_length(records):
    ranked = sorted(records, key=lambda r: (len(r["filename"]), r["filename"].casefold()))
    expected_short = {r["filename"] for r in ranked[:15]}
    expected_long = {r["filename"] for r in ranked[-15:]}
    assert {r["filename"] for r in records if r["short_name"]} == expected_short
    assert {r["filename"] for r in records if r["long_name"]} == expected_long


def test_all_files_zero_byte(fixture_files):
    for p in fixture_files:
        assert p.stat().st_size == 0, p.name


def test_all_files_extensionless(fixture_files):
    for p in fixture_files:
        assert "." not in p.name, p.name


def test_all_filenames_nfc_normalized(fixture_files):
    for p in fixture_files:
        assert unicodedata.normalize("NFC", p.name) == p.name, p.name


_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$", re.IGNORECASE)
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def test_all_filenames_windows_safe(fixture_files):
    for p in fixture_files:
        name = p.name
        assert not _BAD_CHARS.search(name), name
        assert not name.endswith(" ") and not name.endswith("."), name
        assert not _RESERVED.match(name), name


def test_casefold_and_nfc_collision_free(fixture_files):
    normalized = [unicodedata.normalize("NFC", p.name).casefold() for p in fixture_files]
    assert len(set(normalized)) == len(normalized)


def test_deterministic_extension_rule_cannot_resolve_any_case(fixture_files):
    # config/rules.yaml maps categories by dotted extension; every Holdout v4
    # filename is extensionless by construction, so none can match.
    for p in fixture_files:
        assert "." not in p.name, p.name


def test_authoring_frozen_artifact_matches_fixture(records):
    frozen = json.loads((HOLDOUT_DIR / "AUTHORING_FROZEN.json").read_text(encoding="utf-8"))
    assert frozen["authoring_frozen"] is True
    assert frozen["case_count"] == 150 == len(records)
    assert frozen["historical_audit_performed"] is False
    assert frozen["model_inference_performed"] is False


def test_no_holdout_v4_consumption_marker_exists():
    assert not (HOLDOUT_DIR / "CONSUMED.json").exists()
