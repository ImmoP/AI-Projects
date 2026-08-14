"""Offline, deterministic tests for the E4-refined precision-oriented veto.

No Ollama dependency: every test drives ``evaluate_refined_veto`` /
``apply_refined_veto`` / ``run_e4_refined`` directly, the latter through a
``FakeModel`` identical in spirit to ``tests/test_post_holdout_candidates.py``.
Nothing here touches ``evals/holdout`` or ``evals/holdout_v2`` -- no fixture
path, filename, or label from either is read, and a structural test at the
bottom asserts the module source never references either directory
(shared with the existing E4-current/E5 guard in
``tests/test_post_holdout_candidates.py``).

E4-refined is deliberately never given exact-filename special cases: every
test below exercises a *freshly constructed* filename chosen to exercise
one design rule, not a literal string copied from any Development or
Holdout fixture.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from types import SimpleNamespace

from evals.post_holdout_candidates import (
    E4R_CATEGORY_CUES,
    E4R_CONTAINER_CUES,
    E4R_EXPLICIT_DISJUNCTION_MARKERS,
    E4R_EXPLICIT_UNCERTAINTY_MARKERS,
    E4R_GENERIC_NEVER_CUE_WORDS,
    E4R_HIGH_SPECIFICITY_CUES,
    apply_refined_veto,
    evaluate_refined_veto,
    run_e4_refined,
)

REVIEW = "_ToReview"
REAL_CATEGORIES = ["Documents", "Code", "Images", "Archives", "Installers"]


class FakeModel:
    structured_output_mode = "json_schema"

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        response = self.responses.pop(0)
        return SimpleNamespace(
            content=response,
            token_usage=SimpleNamespace(input_tokens=5, output_tokens=2),
        )


def _classify_pair(*pairs: tuple[str, str]) -> str:
    items = ",".join(
        f'{{"source":"{s}","decision":"classify","category":"{c}"}}' for s, c in pairs
    )
    return f'{{"decisions":[{items}]}}'


# --- Vocabulary hygiene -------------------------------------------------


def test_vocabulary_categories_match_production_categories() -> None:
    assert set(E4R_CATEGORY_CUES) == set(REAL_CATEGORIES)
    assert set(E4R_HIGH_SPECIFICITY_CUES) == set(REAL_CATEGORIES)


def test_high_specificity_is_a_subset_of_category_support() -> None:
    for category in REAL_CATEGORIES:
        assert E4R_HIGH_SPECIFICITY_CUES[category] <= E4R_CATEGORY_CUES[category]


def test_generic_never_cue_words_absent_from_every_cue_set() -> None:
    all_cues: set[str] = set(E4R_CONTAINER_CUES) | set(E4R_EXPLICIT_UNCERTAINTY_MARKERS) | set(
        E4R_EXPLICIT_DISJUNCTION_MARKERS
    )
    for words in E4R_CATEGORY_CUES.values():
        all_cues |= set(words)
    for words in E4R_HIGH_SPECIFICITY_CUES.values():
        all_cues |= set(words)
    # "backup"/"sicherung" are a documented, deliberate dual case (see module
    # comment): generic on their own, but eligible container evidence when
    # combined with independent structure. Every other never-cue word must
    # be wholly absent.
    unexpected = (all_cues - {"backup", "sicherung"}) & E4R_GENERIC_NEVER_CUE_WORDS
    assert unexpected == set()
    assert "backup" in E4R_CONTAINER_CUES and "backup" not in set().union(
        *E4R_CATEGORY_CUES.values()
    )


# --- Preserve cases (hard negatives) ----------------------------------------


def test_document_about_image_topic_is_preserved() -> None:
    # A report whose subject is photography; the file itself is a document.
    outcome = evaluate_refined_veto(
        "kamera_workshop_teilnehmer_zusammenfassung_bericht", "Documents", review_directory=REVIEW
    )
    assert outcome.final == "Documents"
    assert outcome.conflict_tier in {"none", "soft"}


def test_code_for_media_processing_is_preserved() -> None:
    # A script whose purpose is processing images; the file itself is Code.
    outcome = evaluate_refined_veto(
        "bild_konvertierung_stapel_skript_python", "Code", review_directory=REVIEW
    )
    assert outcome.final == "Code"
    assert outcome.conflict_tier in {"none", "soft"}


def test_installer_documentation_semantics_is_preserved() -> None:
    # Documentation describing an installer, not the installer itself.
    outcome = evaluate_refined_veto(
        "installation_anleitung_schritt_fuer_schritt_dokument", "Documents", review_directory=REVIEW
    )
    assert outcome.final == "Documents"
    assert outcome.conflict_tier in {"none", "soft"}


def test_archive_management_source_code_is_preserved() -> None:
    # Source code that manages/creates archives is itself Code, not Archives.
    outcome = evaluate_refined_veto(
        "archiv_verwaltung_backend_modul_quellcode", "Code", review_directory=REVIEW
    )
    assert outcome.final == "Code"
    assert outcome.conflict_tier in {"none", "soft"}


def test_weak_generic_cue_alone_is_preserved() -> None:
    outcome = evaluate_refined_veto("vertrag_entwurf_neu_kopie", "Documents", review_directory=REVIEW)
    assert outcome.final == "Documents"
    assert outcome.veto_reason_code == "NO_CONFLICT"


def test_generic_workflow_words_never_justify_review() -> None:
    outcome = evaluate_refined_veto(
        "quellcode_projekt_alt_final_version_kopie", "Code", review_directory=REVIEW
    )
    assert outcome.final == "Code"
    assert outcome.conflict_tier in {"none", "soft"}


def test_two_cue_families_alone_is_soft_not_hard() -> None:
    """The central design change: cue co-occurrence alone must never veto."""
    outcome = evaluate_refined_veto(
        "vertrag_archiv_nummer_uebersicht_dokument", "Documents", review_directory=REVIEW
    )
    assert outcome.final == "Documents"  # preserved
    assert outcome.conflict_tier == "soft"
    assert outcome.veto_reason_code in {"MULTI_CUE_SOFT_CONFLICT", "CONTAINER_WORD_SOFT_SIGNAL"}


# --- Veto cases (hard conflicts) --------------------------------------------


def test_explicit_disjunction_with_two_sided_evidence_reviews() -> None:
    outcome = evaluate_refined_veto(
        "praesentation_folien_oder_bild_zusammenstellung", "Images", review_directory=REVIEW
    )
    assert outcome.final == REVIEW
    assert outcome.veto_reason_code == "EXPLICIT_CATEGORY_AMBIGUITY"
    assert outcome.conflict_tier == "hard"


def test_explicit_uncertainty_with_two_sided_evidence_reviews() -> None:
    outcome = evaluate_refined_veto(
        "vertrag_oder_quellcode_unklar_typ", "Code", review_directory=REVIEW
    )
    assert outcome.final == REVIEW
    assert outcome.veto_reason_code == "EXPLICIT_CATEGORY_AMBIGUITY"


def test_predicted_category_unsupported_with_strong_alternative_reviews() -> None:
    outcome = evaluate_refined_veto("vertrag_dokument_2026", "Code", review_directory=REVIEW)
    assert outcome.final == REVIEW
    assert outcome.veto_reason_code == "PREDICTED_CATEGORY_UNSUPPORTED"
    assert outcome.conflict_tier == "hard"


def test_predicted_category_unsupported_requires_high_specificity_alternative() -> None:
    # A single *moderate* alternative cue must not be enough for Family B.
    outcome = evaluate_refined_veto("bericht_2026", "Code", review_directory=REVIEW)
    assert outcome.veto_reason_code != "PREDICTED_CATEGORY_UNSUPPORTED"


def test_container_content_ambiguity_reviews() -> None:
    # e3_category has a *moderate* cue of its own ("programm") so Family B
    # (no support at all) does not preempt this; combined with a container
    # word and a genuinely high-specificity alternative ("vertrag"), only
    # Family C's own-high-specificity-anchor gap applies.
    outcome = evaluate_refined_veto("programm_backup_vertrag_wichtig", "Code", review_directory=REVIEW)
    assert outcome.final == REVIEW
    assert outcome.veto_reason_code == "CONTAINER_CONTENT_AMBIGUITY"
    assert outcome.conflict_tier == "hard"


def test_family_b_preempts_when_predicted_category_has_no_support_at_all() -> None:
    # A predicted category with *zero* support (not even moderate) and a
    # container word alongside a strong alternative is still correctly
    # routed to review -- just via the more severe Family B, since an
    # unsupported prediction is stronger evidence than a container word.
    outcome = evaluate_refined_veto("foto_sammlung_paket", "Archives", review_directory=REVIEW)
    assert outcome.final == REVIEW
    assert outcome.veto_reason_code == "PREDICTED_CATEGORY_UNSUPPORTED"


def test_container_word_alone_does_not_trigger_hard_conflict() -> None:
    outcome = evaluate_refined_veto("archiv_paket_gesamt", "Archives", review_directory=REVIEW)
    assert outcome.final == "Archives"
    assert outcome.conflict_tier in {"none", "soft"}


# --- Safety -----------------------------------------------------------------


def test_not_applicable_when_e3_already_reviews() -> None:
    outcome = evaluate_refined_veto("irrelevant_name", REVIEW, review_directory=REVIEW)
    assert outcome.applicable is False
    assert outcome.conflict_tier == "not_applicable"
    assert outcome.final == REVIEW


def test_run_e4_refined_invalid_e3_result_remains_review() -> None:
    metadata = [{"name": "one", "size_bytes": 0}]
    model = FakeModel(
        "not-json",
        _classify_pair(("one", "Documents")),
    )
    final, detail, _telemetry = run_e4_refined(model, metadata, REAL_CATEGORIES, review_directory=REVIEW)
    assert final["one"] == REVIEW
    assert detail[0]["conflict_tier"] == "not_applicable"


def test_final_category_is_always_e3_category_or_review_never_invented() -> None:
    for name, e3_cat in [
        ("vertrag_dokument_2026", "Code"),
        ("foto_sammlung_paket", "Archives"),
        ("archiv_paket_gesamt", "Archives"),
        ("bild_bearbeitung_batch_prozess_programm", "Code"),
    ]:
        outcome = evaluate_refined_veto(name, e3_cat, review_directory=REVIEW)
        assert outcome.final in {e3_cat, REVIEW}


def test_no_path_or_separator_ever_appears_in_output() -> None:
    outcome = evaluate_refined_veto("../../etc/passwd_vertrag", "Documents", review_directory=REVIEW)
    assert "/" not in outcome.final
    assert ".." not in outcome.final


def test_arbitrary_unknown_source_does_not_crash_apply_refined_veto() -> None:
    e3_final = {"a": "Documents"}
    outcomes = apply_refined_veto(e3_final, ["a", "ghost_not_in_e3_final"], review_directory=REVIEW)
    assert outcomes["ghost_not_in_e3_final"].e3_category == REVIEW
    assert outcomes["ghost_not_in_e3_final"].final == REVIEW


def test_deterministic_repeated_calls_agree() -> None:
    first = evaluate_refined_veto("vertrag_archiv_nummer_uebersicht_dokument", "Documents", review_directory=REVIEW)
    second = evaluate_refined_veto("vertrag_archiv_nummer_uebersicht_dokument", "Documents", review_directory=REVIEW)
    assert first == second


def test_run_e4_refined_adds_no_model_call_beyond_e3() -> None:
    metadata = [{"name": "one", "size_bytes": 0}, {"name": "two", "size_bytes": 0}]
    model = FakeModel(
        _classify_pair(("one", "Documents"), ("two", "Code")),
        _classify_pair(("one", "Documents"), ("two", "Code")),
    )
    _final, _detail, _telemetry = run_e4_refined(model, metadata, REAL_CATEGORIES, review_directory=REVIEW)
    assert len(model.calls) == 2  # identical to run_e3's own call count


# --- Unicode / multilingual --------------------------------------------------


def test_unicode_nfd_and_nfc_forms_agree() -> None:
    nfc_name = "vertrag_büro_dokument"
    nfd_name = unicodedata.normalize("NFD", nfc_name)
    assert nfc_name != nfd_name

    nfc_outcome = evaluate_refined_veto(nfc_name, "Documents", review_directory=REVIEW)
    nfd_outcome = evaluate_refined_veto(nfd_name, "Documents", review_directory=REVIEW)
    assert nfc_outcome.veto_reason_code == nfd_outcome.veto_reason_code
    assert nfc_outcome.final == nfd_outcome.final


def test_casefold_handles_case_variation() -> None:
    lower = evaluate_refined_veto("vertrag_dokument_2026", "Code", review_directory=REVIEW)
    upper = evaluate_refined_veto("VERTRAG_DOKUMENT_2026", "Code", review_directory=REVIEW)
    assert lower.veto_reason_code == upper.veto_reason_code == "PREDICTED_CATEGORY_UNSUPPORTED"


def test_multilingual_cue_families_trigger_symmetrically() -> None:
    german = evaluate_refined_veto("vertrag_dokument_2026", "Code", review_directory=REVIEW)
    english = evaluate_refined_veto("contract_document_2026", "Code", review_directory=REVIEW)
    assert german.veto_reason_code == english.veto_reason_code == "PREDICTED_CATEGORY_UNSUPPORTED"


def test_compound_words_without_separators_are_not_decomposed() -> None:
    """Documented limitation: E4-refined deliberately does not perform
    substring/compound decomposition (see module design notes) to avoid
    reintroducing false-positive risk. An unseparated German compound must
    therefore behave identically to an unrelated, cue-free filename."""
    compound = evaluate_refined_veto("reisebericht_2026", "Code", review_directory=REVIEW)
    unrelated = evaluate_refined_veto("xyzabc_2026", "Code", review_directory=REVIEW)
    assert compound.veto_reason_code == unrelated.veto_reason_code == "NO_CONFLICT"
    assert compound.final == "Code"


# --- Structural guards --------------------------------------------------


_MODULE_PATH = Path(__file__).parents[1].joinpath("evals", "post_holdout_candidates.py")
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")
_MODULE_DOCSTRING_END = _MODULE_SOURCE.index('"""', _MODULE_SOURCE.index('"""') + 3) + 3
_MODULE_CODE_BODY = _MODULE_SOURCE[_MODULE_DOCSTRING_END:]


def test_e4_refined_adds_no_model_call_reference() -> None:
    # Structural: the E4-refined section of the module must never construct
    # a prompt or invoke a backend request of its own.
    refined_section = _MODULE_SOURCE[_MODULE_SOURCE.index("# E4-refined -- precision") :]
    assert "backend.request(" not in refined_section
    assert "build_verifier_prompt(" not in refined_section
    assert "build_explicit_abstention_prompt(" not in refined_section


def test_module_still_never_references_either_holdout_directory() -> None:
    assert "evals/holdout" not in _MODULE_CODE_BODY
    assert "holdout_v2" not in _MODULE_CODE_BODY
    assert "evals.holdout" not in _MODULE_CODE_BODY
