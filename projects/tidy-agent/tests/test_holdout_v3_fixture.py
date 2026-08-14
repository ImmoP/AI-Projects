"""Fixture-integrity tests for the independently-authored Holdout v3.

Read-only: these tests only inspect ``evals/holdout_v3/`` metadata (file
names, sizes, and the evaluator-only ground truth); they never call the
runner and never issue a model request, so Holdout v3 stays unconsumed.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_DIR = REPO_ROOT / "evals" / "holdout_v3"
FIXTURE_DIR = HOLDOUT_DIR / "fixture"
EXPECTED_PATH = HOLDOUT_DIR / "expected.yaml"
MANIFEST_PATH = HOLDOUT_DIR / "fixture_manifest.json"

REAL_CATEGORIES = ("Documents", "Code", "Images", "Archives", "Installers")
REVIEW_DIRECTORY = "_ToReview"
REAL_STRATA = ("ordinary_realistic", "contextual_relational", "distractor_rich_resolvable")
REVIEW_STRATA = ("insufficient_metadata", "latent_dual_role", "container_artifact_role_ambiguity")

# Independently-authored general ambiguity-marker vocabulary, used only to
# audit that the review set is not a trivial explicit-uncertainty benchmark.
# Never exposed to the model; never derived from evals/post_holdout_candidates.py.
GENERAL_AMBIGUITY_MARKERS = frozenset(
    unicodedata.normalize("NFC", w).casefold()
    for w in {
        "or", "either", "unsure", "unclear", "uncertain", "undetermined",
        "unknown", "maybe", "possibly", "ambiguous", "vs",
        "oder", "unklar", "unsicher", "ungewiss", "vielleicht",
        "peut-etre", "incertain",
        "quiza", "incierto", "tal vez",
        "forse", "incerto",
        "talvez", "incerto",
        "misschien", "onduidelijk",
        "niejasne", "moze", "niepewne",
    }
)


def _has_marker(name: str) -> bool:
    tokens = set(unicodedata.normalize("NFC", name).casefold().replace("-", "_").split("_"))
    return bool(tokens & GENERAL_AMBIGUITY_MARKERS)


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    doc = yaml.safe_load(EXPECTED_PATH.read_text(encoding="utf-8"))
    return doc["cases"]


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fixture_names() -> list[str]:
    return [p.name for p in FIXTURE_DIR.iterdir() if p.is_file()]


def test_exactly_120_files_on_disk(fixture_names):
    assert len(fixture_names) == 120


def test_exactly_120_cases_in_ground_truth(cases):
    assert len(cases) == 120


def test_fixture_and_ground_truth_agree(cases, fixture_names):
    assert set(fixture_names) == {c["filename"] for c in cases}


def test_75_real_45_review(cases):
    real = [c for c in cases if c["expected_outcome"] != REVIEW_DIRECTORY]
    review = [c for c in cases if c["expected_outcome"] == REVIEW_DIRECTORY]
    assert len(real) == 75
    assert len(review) == 45


def test_15_per_real_category(cases):
    counts = {cat: 0 for cat in REAL_CATEGORIES}
    for c in cases:
        if c["expected_outcome"] in counts:
            counts[c["expected_outcome"]] += 1
    assert counts == {cat: 15 for cat in REAL_CATEGORIES}


def test_real_strata_25_each(cases):
    counts = {s: 0 for s in REAL_STRATA}
    for c in cases:
        if c["primary_stratum"] in counts:
            counts[c["primary_stratum"]] += 1
    assert counts == {s: 25 for s in REAL_STRATA}


def test_review_strata_15_each(cases):
    counts = {s: 0 for s in REVIEW_STRATA}
    for c in cases:
        if c["primary_stratum"] in counts:
            counts[c["primary_stratum"]] += 1
    assert counts == {s: 15 for s in REVIEW_STRATA}


def test_five_per_real_category_per_real_stratum(cases):
    for stratum in REAL_STRATA:
        counts = {cat: 0 for cat in REAL_CATEGORIES}
        for c in cases:
            if c["primary_stratum"] == stratum:
                counts[c["expected_outcome"]] += 1
        assert counts == {cat: 5 for cat in REAL_CATEGORIES}, stratum


def test_marker_free_review_count_at_least_36(cases):
    review = [c for c in cases if c["expected_outcome"] == REVIEW_DIRECTORY]
    marker_free = sum(1 for c in review if not _has_marker(c["filename"]))
    assert marker_free >= 36


def test_multilingual_at_least_40(cases):
    non_english = sum(1 for c in cases if c["language"] != "en")
    assert non_english >= 40


def test_multilingual_several_languages_no_single_dominant(cases):
    langs: dict[str, int] = {}
    for c in cases:
        if c["language"] != "en":
            langs[c["language"]] = langs.get(c["language"], 0) + 1
    total = sum(langs.values())
    assert len(langs) >= 4
    assert max(langs.values()) / total < 0.5


def test_morphology_at_least_15(cases):
    count = sum(1 for c in cases if "morphology" in c["secondary_tags"])
    assert count >= 15


def test_instruction_like_exactly_10(cases):
    subset = [c for c in cases if c["instruction_like"]]
    assert len(subset) == 10


def test_instruction_like_split_4_real_6_review(cases):
    subset = [c for c in cases if c["instruction_like"]]
    real = sum(1 for c in subset if c["expected_outcome"] != REVIEW_DIRECTORY)
    review = sum(1 for c in subset if c["expected_outcome"] == REVIEW_DIRECTORY)
    assert (real, review) == (4, 6)


def test_all_files_zero_byte():
    for path in FIXTURE_DIR.iterdir():
        if path.is_file():
            assert path.stat().st_size == 0, path.name


def test_all_files_extensionless():
    for path in FIXTURE_DIR.iterdir():
        if path.is_file():
            assert Path(path.name).suffix == "", path.name


def test_deterministic_rule_unresolved(fixture_names):
    rules = yaml.safe_load((REPO_ROOT / "config" / "rules.yaml").read_text(encoding="utf-8"))
    all_extensions = {ext for exts in rules["categories"].values() for ext in exts}
    for name in fixture_names:
        assert Path(name).suffix.lower() not in all_extensions


def test_all_names_nfc_normalized(fixture_names):
    for name in fixture_names:
        assert name == unicodedata.normalize("NFC", name), name


def test_all_names_windows_safe(fixture_names):
    forbidden = set('<>:"/\\|?*')
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    for name in fixture_names:
        assert not (forbidden & set(name)), name
        assert not name.endswith(" ") and not name.endswith("."), name
        assert name.upper() not in reserved, name
        assert 0 < len(name) <= 200, name


def test_no_casefold_or_nfc_collision(fixture_names):
    keys = [unicodedata.normalize("NFC", n).casefold() for n in fixture_names]
    assert len(keys) == len(set(keys))


def test_manifest_matches_recomputed_values(cases, manifest, fixture_names):
    assert manifest["total_files"] == 120
    assert manifest["real_category_count"] == 75
    assert manifest["review_count"] == 45
    assert manifest["morphology_count"] >= 15
    assert manifest["multilingual_count"] >= 40
    assert manifest["marker_free_review_count"] >= 36
    assert manifest["instruction_like_count"] == 10


def test_manifest_hashes_match_live_files(manifest, fixture_names):
    import hashlib

    listing = "\n".join(sorted(fixture_names)).encode("utf-8")
    assert hashlib.sha256(listing).hexdigest() == manifest["dataset_sha256"]
    assert hashlib.sha256(EXPECTED_PATH.read_bytes()).hexdigest() == manifest["ground_truth_sha256"]


def test_evaluator_fields_never_shaped_like_model_input(cases):
    """The ground-truth artifact carries evaluator-only fields; confirm the
    schema is what production input-building code must never receive."""
    for c in cases:
        assert set(c) == {
            "filename", "expected_outcome", "primary_stratum", "language",
            "secondary_tags", "instruction_like", "rationale",
        }
