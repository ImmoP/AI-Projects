"""Offline integrity checks for the E3 automatic-error Development stress
fixture.

No Ollama/network dependency: every test operates on the fixture files,
``expected.yaml``, and ``fixture_manifest.json`` already on disk, or on the
pure ``classify_directory``/``load_rules`` functions already exercised
elsewhere in this suite. Nothing here runs model inference, and nothing
here inspects the consumed ``evals/holdout/`` or ``evals/holdout_v2/``
fixtures -- this file only ever reads paths under
``evals/e3_error_calibration/``.
"""

from __future__ import annotations

import string
import unicodedata
from collections import Counter
from pathlib import Path

from evals.e3_error_calibration.build_fixture import CASES, EXPLICIT_AMBIGUITY_MARKERS
from evals.run_evals import _load_expected
from evals.run_structured_abcd import _verify_dataset_manifest
from tidy.rules import classify_directory, load_rules

ROOT = Path(__file__).parents[1] / "evals" / "e3_error_calibration"
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
_REAL_FAMILIES = {
    "subject_vs_artifact",
    "tool_vs_output",
    "container_lexical_trap",
    "installer_driver_trap",
    "media_document_trap",
}
_REVIEW_FAMILIES = {"latent_dual_role", "latent_container_content", "dominant_cue_ambiguity"}


def test_fixture_count_matches_manifest_and_cases() -> None:
    on_disk = {path.name for path in FIXTURE.iterdir() if path.is_file()}
    assert len(on_disk) == 72
    assert len(CASES) == 72
    assert on_disk == {name for name, *_ in CASES}


def test_manifest_hashes_match_the_fixture_and_ground_truth() -> None:
    manifest = _verify_dataset_manifest(MANIFEST, FIXTURE, EXPECTED)
    assert len(manifest["files"]) == 72
    assert manifest["designation"] == "E3 automatic-error development stress fixture"
    assert manifest["model_inference_performed"] is False


def test_ground_truth_has_exactly_one_label_per_fixture_file() -> None:
    expected = _load_expected(EXPECTED)
    fixture_names = {path.name for path in FIXTURE.iterdir() if path.is_file()}

    assert set(expected) == fixture_names
    for name, allowed in expected.items():
        assert isinstance(allowed, list) and len(allowed) == 1, name


def test_composition_matches_documented_48_24_split() -> None:
    review_directory = load_rules().review_directory
    real = [c for c in CASES if c[1] != review_directory]
    review = [c for c in CASES if c[1] == review_directory]

    assert len(real) == 48
    assert len(review) == 24


def test_real_category_distribution_matches_documented_counts() -> None:
    review_directory = load_rules().review_directory
    real_categories = [category for _name, category, *_rest in CASES if category != review_directory]
    assert Counter(real_categories) == {
        "Documents": 10,
        "Code": 10,
        "Images": 10,
        "Archives": 9,
        "Installers": 9,
    }


def test_real_category_family_minimums_are_met() -> None:
    review_directory = load_rules().review_directory
    real_families = [family for _n, category, _r, family, _t in CASES if category != review_directory]
    counts = Counter(real_families)
    assert set(counts) == _REAL_FAMILIES
    assert counts["subject_vs_artifact"] >= 16
    assert counts["tool_vs_output"] >= 10
    assert counts["container_lexical_trap"] >= 8
    assert counts["installer_driver_trap"] >= 8
    assert counts["media_document_trap"] >= 6
    assert sum(counts.values()) == 48


def test_container_lexical_trap_family_avoids_archives_ground_truth() -> None:
    """Item 14: the container/package vocabulary family exists specifically
    to show that word alone can't be trusted -- it must not be dominated by
    genuinely-Archives cases."""
    for name, category, _rationale, family, _tags in CASES:
        if family == "container_lexical_trap":
            assert category != "Archives", name


def test_installer_driver_trap_family_avoids_installers_ground_truth() -> None:
    """Item 15: same principle for the driver/setup/deployment vocabulary
    family -- it must not be dominated by genuinely-Installers cases."""
    for name, category, _rationale, family, _tags in CASES:
        if family == "installer_driver_trap":
            assert category != "Installers", name


def test_media_document_trap_family_includes_both_directions() -> None:
    """Item 16: some real Images cases must carry misleading non-image
    vocabulary, AND some non-Images cases must carry misleading media
    vocabulary -- the trap must not be purely one-directional."""
    media_family = [c for c in CASES if c[3] == "media_document_trap"]
    categories = {c[1] for c in media_family}
    assert "Images" in categories
    assert categories - {"Images"}  # at least one non-Images category present


def test_review_family_counts_match_documented_distribution() -> None:
    review_directory = load_rules().review_directory
    review_families = [family for _n, category, _r, family, _t in CASES if category == review_directory]
    counts = Counter(review_families)
    assert set(counts) == _REVIEW_FAMILIES
    assert counts["latent_dual_role"] == 10
    assert counts["latent_container_content"] == 6
    assert counts["dominant_cue_ambiguity"] == 8
    assert sum(counts.values()) == 24


def test_no_missing_or_duplicate_source_in_cases() -> None:
    names = [name for name, *_ in CASES]
    assert len(names) == len(set(names))


def test_every_non_review_label_is_a_production_category() -> None:
    review_directory = load_rules().review_directory
    for name, category, *_rest in CASES:
        assert category in _PRODUCTION_CATEGORIES, name
        if category != review_directory:
            assert category in load_rules().categories, name


def test_every_case_bypasses_deterministic_rules() -> None:
    """Every e3_error_calibration fixture file must reach unresolved
    (E3/E4-current/E4-refined)."""
    rules = load_rules()

    moves, unresolved = classify_directory(FIXTURE, rules)

    assert moves == []
    assert set(unresolved) == {name for name, *_ in CASES}


def test_no_case_matches_an_exclude_pattern() -> None:
    rules = load_rules()
    for name, *_rest in CASES:
        assert not rules.is_excluded(name), name


def test_all_filenames_are_cross_platform_safe() -> None:
    for name, *_rest in CASES:
        assert not (set(name) & _WINDOWS_INVALID_CHARS), name
        assert all(ord(char) >= 32 for char in name), name
        assert not name.endswith((".", " ")), name
        assert name == name.strip(), name
        stem = name.split(".", 1)[0].upper()
        assert stem not in _RESERVED_BASENAMES, name
        assert len(name.encode("utf-8")) <= 255, name
        assert "/" not in name and "\\" not in name


def test_all_filenames_are_nfc_normalized() -> None:
    for name, *_rest in CASES:
        assert unicodedata.normalize("NFC", name) == name, name


def test_no_casefold_or_nfc_collisions_between_any_two_filenames() -> None:
    names = [name for name, *_rest in CASES]
    folded = [unicodedata.normalize("NFC", name).casefold() for name in names]
    assert len(set(folded)) == len(folded)


def test_no_filename_is_purely_ascii_control_or_whitespace() -> None:
    for name, *_rest in CASES:
        assert name.strip(string.whitespace) == name
        assert len(name) > 0


def test_fixture_files_are_empty_placeholders() -> None:
    for path in FIXTURE.iterdir():
        if path.is_file():
            assert path.stat().st_size == 0, path.name


def test_no_filename_has_an_extension() -> None:
    for name, *_rest in CASES:
        assert "." not in name, name


def test_at_least_20_of_24_review_cases_have_no_explicit_ambiguity_marker() -> None:
    """Item 23: the fixture must not accidentally become another
    review-marker detection benchmark."""
    review_directory = load_rules().review_directory
    review_cases = [c for c in CASES if c[1] == review_directory]
    marker_free = 0
    for name, _category, _rationale, _family, _tags in review_cases:
        tokens = {token.lower() for token in name.split("_")}
        if not (tokens & EXPLICIT_AMBIGUITY_MARKERS):
            marker_free += 1
    assert marker_free >= 20
    assert len(review_cases) == 24


def test_at_least_18_of_72_files_are_multilingual() -> None:
    tagged = sum(1 for c in CASES if "multilingual" in c[4])
    assert tagged >= 18


def test_at_least_8_files_have_compound_morphology_stress() -> None:
    tagged = sum(1 for c in CASES if "compound_morphology" in c[4])
    assert tagged >= 8


def test_no_filename_duplicates_existing_development_fixtures() -> None:
    """Item 27/28: fresh concepts only -- no literal filename reuse from
    any of the three prior Development fixtures. (Neither Holdout fixture
    is imported or read here or anywhere in this module -- see the
    module-source guard test below -- so only Development fixtures are
    checked directly, per item 28's explicit restriction.)
    """
    from evals.boundary_calibration.build_fixture import CASES as BOUNDARY_CASES
    from evals.calibration.build_fixture import FILENAMES as CALIBRATION_FILENAMES
    from evals.veto_precision_calibration.build_fixture import CASES as VETO_PRECISION_CASES

    new_names = {name for name, *_rest in CASES}
    assert new_names.isdisjoint(CALIBRATION_FILENAMES)
    assert new_names.isdisjoint({name for name, _c, _r in BOUNDARY_CASES})
    assert new_names.isdisjoint({name for name, _c, _r in VETO_PRECISION_CASES})


def test_e3_error_calibration_module_never_imports_either_holdout() -> None:
    """The docstring may *name* the excluded paths (documenting the
    independence protocol); it must never *import* from them or reference
    them anywhere outside that documentation.
    """
    source = Path(__file__).parents[1].joinpath(
        "evals", "e3_error_calibration", "build_fixture.py"
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
