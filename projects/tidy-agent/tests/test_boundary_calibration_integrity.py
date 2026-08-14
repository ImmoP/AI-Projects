"""Offline integrity checks for the post-Holdout-v2 boundary-calibration
development fixture.

No Ollama/network dependency: every test operates on the fixture files,
``expected.yaml``, and ``fixture_manifest.json`` already on disk, or on the
pure ``classify_directory``/``load_rules`` functions already exercised
elsewhere in this suite. Nothing here runs model inference, and nothing
here inspects the consumed ``evals/holdout/`` or ``evals/holdout_v2/``
fixtures -- this file only ever reads paths under
``evals/boundary_calibration/``.
"""

from __future__ import annotations

import string
import unicodedata
from collections import Counter
from pathlib import Path

from evals.boundary_calibration.build_fixture import CASES
from evals.run_evals import _load_expected
from evals.run_structured_abcd import _verify_dataset_manifest
from tidy.rules import classify_directory, load_rules

ROOT = Path(__file__).parents[1] / "evals" / "boundary_calibration"
FIXTURE = ROOT / "fixture"
EXPECTED = ROOT / "expected.yaml"
MANIFEST = ROOT / "fixture_manifest.json"

_WINDOWS_INVALID_CHARS = set('<>:"/\\|?*')
_RESERVED_BASENAMES = (
    {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
    | {"CON", "PRN", "AUX", "NUL"}
)
_PRODUCTION_CATEGORIES = set(load_rules().categories) | {load_rules().review_directory}


def test_fixture_count_matches_manifest_and_cases() -> None:
    on_disk = {path.name for path in FIXTURE.iterdir() if path.is_file()}
    assert len(on_disk) == 66
    assert len(CASES) == 66
    assert on_disk == {name for name, _category, _rationale in CASES}


def test_manifest_hashes_match_the_fixture_and_ground_truth() -> None:
    manifest = _verify_dataset_manifest(MANIFEST, FIXTURE, EXPECTED)
    assert len(manifest["files"]) == 66
    assert manifest["designation"] == "post-holdout-v2 development boundary-calibration"
    assert manifest["model_inference_performed"] is False


def test_ground_truth_has_exactly_one_label_per_fixture_file() -> None:
    expected = _load_expected(EXPECTED)
    fixture_names = {path.name for path in FIXTURE.iterdir() if path.is_file()}

    assert set(expected) == fixture_names
    for name, allowed in expected.items():
        assert isinstance(allowed, list) and len(allowed) == 1, name


def test_composition_matches_documented_distribution() -> None:
    review_directory = load_rules().review_directory
    real = [category for _name, category, _rationale in CASES if category != review_directory]
    review = [category for _name, category, _rationale in CASES if category == review_directory]

    assert len(real) == 40
    assert len(review) == 26
    assert Counter(real) == {
        "Documents": 12,
        "Archives": 8,
        "Images": 7,
        "Code": 7,
        "Installers": 6,
    }


def test_no_missing_or_duplicate_source_in_cases() -> None:
    names = [name for name, _category, _rationale in CASES]
    assert len(names) == len(set(names))


def test_every_non_review_label_is_a_production_category() -> None:
    review_directory = load_rules().review_directory
    for name, category, _rationale in CASES:
        assert category in _PRODUCTION_CATEGORIES, name
        if category != review_directory:
            assert category in load_rules().categories, name


def test_every_case_bypasses_deterministic_rules(tmp_path: Path) -> None:
    """Every boundary-calibration file must reach unresolved (E3/E4/E5)."""
    rules = load_rules()

    moves, unresolved = classify_directory(FIXTURE, rules)

    assert moves == []
    assert set(unresolved) == {name for name, _category, _rationale in CASES}


def test_no_case_matches_an_exclude_pattern() -> None:
    rules = load_rules()
    for name, _category, _rationale in CASES:
        assert not rules.is_excluded(name), name


def test_all_filenames_are_cross_platform_safe() -> None:
    for name, _category, _rationale in CASES:
        assert not (set(name) & _WINDOWS_INVALID_CHARS), name
        assert all(ord(char) >= 32 for char in name), name
        assert not name.endswith((".", " ")), name
        assert name == name.strip(), name
        stem = name.split(".", 1)[0].upper()
        assert stem not in _RESERVED_BASENAMES, name
        assert len(name.encode("utf-8")) <= 255, name
        assert "/" not in name and "\\" not in name


def test_all_filenames_are_nfc_normalized() -> None:
    for name, _category, _rationale in CASES:
        assert unicodedata.normalize("NFC", name) == name, name


def test_no_casefold_or_nfc_collisions_between_any_two_filenames() -> None:
    names = [name for name, _category, _rationale in CASES]
    folded = [unicodedata.normalize("NFC", name).casefold() for name in names]
    assert len(set(folded)) == len(folded)


def test_no_filename_is_purely_ascii_control_or_whitespace() -> None:
    for name, _category, _rationale in CASES:
        assert name.strip(string.whitespace) == name
        assert len(name) > 0


def test_fixture_files_are_empty_placeholders() -> None:
    for path in FIXTURE.iterdir():
        if path.is_file():
            assert path.stat().st_size == 0, path.name


def test_boundary_calibration_module_never_imports_either_holdout() -> None:
    """The docstring may *name* the excluded paths (documenting the
    independence protocol); it must never *import* from them or reference
    them anywhere outside that documentation.
    """
    source = Path(__file__).parents[1].joinpath(
        "evals", "boundary_calibration", "build_fixture.py"
    ).read_text(encoding="utf-8")
    forbidden_imports = (
        "import evals.holdout",
        "from evals.holdout import",
        "from evals.holdout.",
        "from evals.holdout_v2 import",
        "from evals.holdout_v2.",
        "import evals.holdout_v2",
    )
    for needle in forbidden_imports:
        assert needle not in source, needle
    docstring_end = source.index('"""', source.index('"""') + 3) + 3
    code_body = source[docstring_end:]
    for needle in ("evals/holdout/", "evals/holdout_v2/", "evals.holdout"):
        assert needle not in code_body, needle
