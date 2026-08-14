"""Regression pin for E4-current, taken at the start of the E4-refined cycle.

E4-current (``evaluate_ambiguity_veto`` / ``run_e4`` in
``evals/post_holdout_candidates.py``) is the frozen comparator for this
Development cycle and must not be silently redefined while E4-refined is
developed alongside it. This file exists specifically to keep the two
genuinely distinct: each test below drives E4-current on one of the exact
filenames used to validate E4-refined's design in
``tests/test_e4_refined.py`` and asserts E4-current's *original* (looser)
behavior, so any accidental edit to E4-current's rules would fail here even
if it happened to still pass E4-refined's own tests.

These are Development-derived filenames (freshly constructed to exercise
generic patterns observed in the previous live Development run's aggregate
false/true-positive counts), never Holdout-derived.
"""

from __future__ import annotations

from evals.post_holdout_candidates import (
    AMBIGUITY_MARKERS,
    CATEGORY_STRONG_CUES,
    CONTAINER_CUES,
    GENERIC_NEVER_CUE_WORDS,
    evaluate_ambiguity_veto,
)

REVIEW = "_ToReview"


def test_e4_current_vocabulary_is_the_original_five_category_set() -> None:
    assert set(CATEGORY_STRONG_CUES) == {"Images", "Documents", "Archives", "Code", "Installers"}
    assert "backup" in CATEGORY_STRONG_CUES["Archives"]  # unlike E4-refined, never moved out
    assert "sicherung" in CATEGORY_STRONG_CUES["Archives"]


def test_e4_current_still_vetoes_on_bare_two_category_cue_co_occurrence() -> None:
    """The exact pattern item 9 of the E4-refined cycle identifies as
    E4-current's main source of false vetoes: two strong cues, no other
    structure. E4-current must keep doing this -- that is the whole point
    of comparing it against E4-refined in the next live experiment."""
    outcome = evaluate_ambiguity_veto(
        "bild_bearbeitung_batch_prozess_programm", "Code", review_directory=REVIEW
    )
    assert outcome.veto_reason_code == "MULTI_CATEGORY_STRONG_CUES"
    assert outcome.final == REVIEW


def test_e4_current_still_vetoes_container_plus_alternative_without_specificity_tiering() -> None:
    outcome = evaluate_ambiguity_veto(
        "vertrag_archiv_nummer_uebersicht_dokument", "Documents", review_directory=REVIEW
    )
    # E4-current has no soft tier: this is a hard veto, unlike E4-refined's
    # MULTI_CUE_SOFT_CONFLICT on the identical filename (see test_e4_refined.py).
    assert outcome.veto_reason_code == "MULTI_CATEGORY_STRONG_CUES"
    assert outcome.final == REVIEW


def test_e4_current_ambiguity_marker_rule_needs_no_second_category() -> None:
    """E4-current's Rule C fires on any ambiguity marker alone, regardless of
    whether a second category has any lexical support -- unlike E4-refined's
    Family A, which requires genuine two-sided evidence (item 9)."""
    outcome = evaluate_ambiguity_veto("unklar_wert_pruefen", "Documents", review_directory=REVIEW)
    assert outcome.veto_reason_code == "AMBIGUITY_MARKER_WITH_CLAIM"
    assert outcome.final == REVIEW


def test_e4_current_never_produces_a_soft_conflict_tier() -> None:
    """E4-current's VetoOutcome has no notion of a soft/telemetry-only
    conflict -- every detected conflict is a hard veto. This is the
    structural difference the E4-refined cycle exists to address."""
    outcome = evaluate_ambiguity_veto(
        "bild_bearbeitung_batch_prozess_programm", "Code", review_directory=REVIEW
    )
    assert not hasattr(outcome, "conflict_tier")
    assert outcome.conflict_detected is True


def test_e4_current_container_cues_never_included_backup_or_sicherung() -> None:
    # Confirms the container-word set (already excluding "package"/"bundle"
    # from category cues) was never touched by this cycle.
    assert CONTAINER_CUES == frozenset(
        w.casefold() for w in ("package", "packages", "bundle", "bundled", "paket", "pakete", "gebündelt")
    )


def test_e4_current_ambiguity_markers_and_never_cue_words_are_unchanged() -> None:
    assert AMBIGUITY_MARKERS == frozenset(
        w.casefold()
        for w in ("or", "unclear", "unsure", "either", "uncertain", "undetermined", "oder", "unklar", "unsicher", "ungewiss")
    )
    assert "backup" not in GENERIC_NEVER_CUE_WORDS  # E4-current never classified it as generic-noise
