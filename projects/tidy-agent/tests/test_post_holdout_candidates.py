"""Offline, deterministic tests for the post-Holdout-v2 E4/E5 candidates.

No Ollama dependency: every test drives the pure validation/merge/veto
functions directly with hand-built JSON, or a ``FakeModel`` identical in
spirit to ``tests/test_calibration_candidates.py``. Nothing here touches
``evals/holdout`` or ``evals/holdout_v2`` -- no fixture path, filename, or
label from either is read, and a structural test at the bottom of this file
asserts the candidate module's source never references either directory.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from types import SimpleNamespace

from evals.post_holdout_candidates import (
    AMBIGUITY_MARKERS,
    CATEGORY_STRONG_CUES,
    CONTAINER_CUES,
    GENERIC_NEVER_CUE_WORDS,
    VERIFIER_JSON_SCHEMA,
    ValidatedVerifierResponse,
    VerifierDecision,
    apply_conflict_veto,
    build_verifier_prompt,
    evaluate_ambiguity_veto,
    merge_classifier_verifier,
    run_e4,
    run_e5,
    validate_verifier_response,
)
from tidy.classification import validate_explicit_abstention_response

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


# --- E4: cue vocabulary hygiene ---------------------------------------------


def test_cue_vocabulary_contains_only_documented_generic_semantics() -> None:
    all_cues: set[str] = set(CONTAINER_CUES) | set(AMBIGUITY_MARKERS)
    for words in CATEGORY_STRONG_CUES.values():
        all_cues |= set(words)
    assert all_cues & GENERIC_NEVER_CUE_WORDS == set()


def test_cue_vocabulary_categories_match_production_categories() -> None:
    assert set(CATEGORY_STRONG_CUES) == set(REAL_CATEGORIES)


# --- E4: veto rule mechanics -------------------------------------------------


def test_veto_no_conflict_preserves_e3_category() -> None:
    outcome = evaluate_ambiguity_veto("bericht_final_kopie", "Documents", review_directory=REVIEW)

    assert outcome.conflict_detected is False
    assert outcome.veto_reason_code == "NO_CONFLICT"
    assert outcome.final == "Documents"


def test_veto_strong_cues_for_two_categories_reviews() -> None:
    outcome = evaluate_ambiguity_veto("foto_archiv_paket", "Archives", review_directory=REVIEW)

    assert outcome.conflict_detected is True
    assert outcome.veto_reason_code == "MULTI_CATEGORY_STRONG_CUES"
    assert outcome.final == REVIEW
    assert set(outcome.matched_category_cue_families) == {"Images", "Archives"}


def test_veto_container_content_conflict() -> None:
    # "paket" (weak container word) + a strong Images cue, while Archives
    # (E3's own category) has no strong cue of its own.
    outcome = evaluate_ambiguity_veto("urlaub_fotos_paket", "Archives", review_directory=REVIEW)

    assert outcome.conflict_detected is True
    assert outcome.veto_reason_code == "CONTAINER_CONTENT_CONFLICT"
    assert outcome.final == REVIEW


def test_veto_ambiguity_marker_with_claim() -> None:
    outcome = evaluate_ambiguity_veto("unklar_kategorie_bericht", "Documents", review_directory=REVIEW)

    assert outcome.conflict_detected is True
    assert outcome.veto_reason_code == "AMBIGUITY_MARKER_WITH_CLAIM"
    assert outcome.final == REVIEW


def test_veto_weak_generic_word_does_not_trigger() -> None:
    outcome = evaluate_ambiguity_veto("bericht_final_neu_kopie", "Documents", review_directory=REVIEW)

    assert outcome.conflict_detected is False
    assert outcome.veto_reason_code == "NO_CONFLICT"


def test_veto_plain_container_word_alone_does_not_trigger() -> None:
    # A container word with no competing content cue anywhere must not veto.
    outcome = evaluate_ambiguity_veto("archiv_paket", "Archives", review_directory=REVIEW)

    assert outcome.conflict_detected is False
    assert outcome.veto_reason_code == "NO_CONFLICT"


def test_veto_not_applicable_when_e3_already_reviews() -> None:
    outcome = evaluate_ambiguity_veto("anything_at_all", REVIEW, review_directory=REVIEW)

    assert outcome.applicable is False
    assert outcome.conflict_detected is False
    assert outcome.veto_reason_code == "NOT_APPLICABLE_E3_REVIEW"
    assert outcome.final == REVIEW


def test_veto_is_deterministic() -> None:
    first = evaluate_ambiguity_veto("foto_archiv_paket", "Archives", review_directory=REVIEW)
    second = evaluate_ambiguity_veto("foto_archiv_paket", "Archives", review_directory=REVIEW)

    assert first == second


def test_veto_unicode_normalization_nfd_and_nfc_agree() -> None:
    nfc_name = "bericht_büro"
    nfd_name = unicodedata.normalize("NFD", nfc_name)
    assert nfc_name != nfd_name  # sanity: genuinely different byte forms

    nfc_outcome = evaluate_ambiguity_veto(nfc_name, "Documents", review_directory=REVIEW)
    nfd_outcome = evaluate_ambiguity_veto(nfd_name, "Documents", review_directory=REVIEW)

    assert nfc_outcome.veto_reason_code == nfd_outcome.veto_reason_code == "NO_CONFLICT"


def test_veto_casefold_handles_case_variation() -> None:
    lower = evaluate_ambiguity_veto("foto_archiv_paket", "Archives", review_directory=REVIEW)
    upper = evaluate_ambiguity_veto("FOTO_ARCHIV_PAKET", "Archives", review_directory=REVIEW)
    mixed = evaluate_ambiguity_veto("Foto_Archiv_Paket", "Archives", review_directory=REVIEW)

    assert lower.veto_reason_code == upper.veto_reason_code == mixed.veto_reason_code
    assert upper.veto_reason_code == "MULTI_CATEGORY_STRONG_CUES"


def test_veto_multilingual_cue_families_trigger_symmetrically() -> None:
    german = evaluate_ambiguity_veto("foto_archiv_paket", "Archives", review_directory=REVIEW)
    english = evaluate_ambiguity_veto("photo_archive_package", "Archives", review_directory=REVIEW)

    assert german.veto_reason_code == english.veto_reason_code == "MULTI_CATEGORY_STRONG_CUES"


def test_veto_no_arbitrary_path_handling() -> None:
    # Defensive: a filename containing separators must never crash the
    # tokenizer or leak a path-like value into the veto outcome.
    outcome = evaluate_ambiguity_veto("../../etc/passwd_bericht", "Documents", review_directory=REVIEW)

    assert outcome.final in {"Documents", REVIEW}
    assert "/" not in outcome.final
    assert ".." not in outcome.final


def test_veto_has_no_exact_filename_exceptions() -> None:
    # Two unrelated filenames sharing the same token composition must
    # receive identical veto behaviour -- nothing keys off a specific name.
    one = evaluate_ambiguity_veto("foto_archiv_paket", "Archives", review_directory=REVIEW)
    other = evaluate_ambiguity_veto("archiv_foto_paket", "Archives", review_directory=REVIEW)

    assert one.veto_reason_code == other.veto_reason_code


# --- E4: apply_conflict_veto / run_e4 orchestration --------------------------


def test_apply_conflict_veto_skips_files_e3_already_reviewed() -> None:
    e3_final = {"a": "Documents", "b": REVIEW}
    outcomes = apply_conflict_veto(e3_final, ["a", "b"], review_directory=REVIEW)

    assert outcomes["b"].applicable is False
    assert outcomes["b"].final == REVIEW


def test_run_e4_preserves_e3_category_when_no_conflict_and_reviews_on_conflict() -> None:
    metadata = [
        {"name": "bericht_final_kopie", "size_bytes": 0},
        {"name": "foto_archiv_paket", "size_bytes": 0},
        {"name": "some_review_case", "size_bytes": 0},
    ]
    # Build a realistic two-pass E3 response covering all three sources.
    model = FakeModel(
        '{"decisions":['
        '{"source":"bericht_final_kopie","decision":"classify","category":"Documents"},'
        '{"source":"foto_archiv_paket","decision":"classify","category":"Archives"},'
        '{"source":"some_review_case","decision":"review","category":null}]}',
        '{"decisions":['
        '{"source":"bericht_final_kopie","decision":"classify","category":"Documents"},'
        '{"source":"foto_archiv_paket","decision":"classify","category":"Archives"},'
        '{"source":"some_review_case","decision":"review","category":null}]}',
    )

    final, detail, telemetry = run_e4(model, metadata, REAL_CATEGORIES, review_directory=REVIEW)

    assert final["bericht_final_kopie"] == "Documents"
    assert final["foto_archiv_paket"] == REVIEW  # vetoed
    assert final["some_review_case"] == REVIEW  # untouched, not applicable
    assert len(model.calls) == 2  # E4 adds no model call beyond E3's two

    by_name = {item["filename"]: item for item in detail}
    assert by_name["foto_archiv_paket"]["veto_reason_code"] == "MULTI_CATEGORY_STRONG_CUES"
    assert by_name["some_review_case"]["veto_reason_code"] == "NOT_APPLICABLE_E3_REVIEW"
    assert by_name["bericht_final_kopie"]["veto_reason_code"] == "NO_CONFLICT"


def test_run_e4_invalid_e3_result_remains_review() -> None:
    metadata = [{"name": "one", "size_bytes": 0}]
    # Pass 1 malformed -> E3's own gate already resolves "one" to review.
    model = FakeModel(
        "not-json",
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
    )

    final, detail, _telemetry = run_e4(model, metadata, REAL_CATEGORIES, review_directory=REVIEW)

    assert final["one"] == REVIEW
    assert detail[0]["veto_reason_code"] == "NOT_APPLICABLE_E3_REVIEW"


# --- E5 verifier: schema and validation -------------------------------------


def test_verifier_schema_has_no_numeric_confidence_field() -> None:
    properties = VERIFIER_JSON_SCHEMA["properties"]["decisions"]["items"]["properties"]
    assert set(properties) == {"source", "decision", "category"}
    assert properties["decision"]["enum"] == ["accept", "review"]
    assert VERIFIER_JSON_SCHEMA["properties"]["decisions"]["items"]["additionalProperties"] is False


def test_verifier_accept_same_category_is_valid() -> None:
    result = validate_verifier_response(
        '{"decisions":[{"source":"one","decision":"accept","category":"Documents"}]}',
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    assert result.decisions["one"] == VerifierDecision("one", "accept", "Documents")
    assert result.invalid_sources == ()


def test_verifier_review_is_valid() -> None:
    result = validate_verifier_response(
        '{"decisions":[{"source":"one","decision":"review","category":null}]}',
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    assert result.decisions["one"] == VerifierDecision("one", "review", None)
    assert result.invalid_sources == ()


def test_verifier_accept_different_category_is_rejected() -> None:
    result = validate_verifier_response(
        '{"decisions":[{"source":"one","decision":"accept","category":"Code"}]}',
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    assert "one" not in result.decisions
    assert result.invalid_sources == ("one",)
    assert result.invalid_reasons == {"one": "category_mismatch"}


def test_verifier_accept_null_category_is_rejected() -> None:
    result = validate_verifier_response(
        '{"decisions":[{"source":"one","decision":"accept","category":null}]}',
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    assert "one" not in result.decisions
    assert result.invalid_reasons == {"one": "missing_category_for_accept"}


def test_verifier_review_with_category_is_rejected() -> None:
    result = validate_verifier_response(
        '{"decisions":[{"source":"one","decision":"review","category":"Documents"}]}',
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    assert "one" not in result.decisions
    assert result.invalid_reasons == {"one": "category_present_for_review"}


def test_verifier_invalid_decision_enum_is_rejected() -> None:
    result = validate_verifier_response(
        '{"decisions":[{"source":"one","decision":"maybe","category":null}]}',
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    assert "one" not in result.decisions
    assert result.invalid_reasons == {"one": "invalid_decision_enum"}


def test_verifier_missing_source_is_omitted() -> None:
    result = validate_verifier_response(
        '{"decisions":[]}',
        {"one": "Documents", "two": "Code"},
        REAL_CATEGORIES,
    )

    assert result.omitted_sources == ("one", "two")


def test_verifier_duplicate_source_is_rejected() -> None:
    result = validate_verifier_response(
        '{"decisions":['
        '{"source":"one","decision":"accept","category":"Documents"},'
        '{"source":"one","decision":"accept","category":"Documents"}]}',
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    assert "one" not in result.decisions
    assert result.invalid_sources == ("one",)
    assert result.telemetry["duplicate_source_responses"] == 1


def test_verifier_invented_source_is_recorded_and_ignored() -> None:
    result = validate_verifier_response(
        '{"decisions":[{"source":"ghost","decision":"accept","category":"Documents"}]}',
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    assert result.unknown_sources == ("ghost",)
    assert result.omitted_sources == ("one",)


def test_verifier_invented_category_on_accept_is_rejected() -> None:
    result = validate_verifier_response(
        '{"decisions":[{"source":"one","decision":"accept","category":"SecretFolder"}]}',
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    assert "one" not in result.decisions
    assert result.invalid_reasons == {"one": "invalid_category_for_accept"}
    assert result.telemetry["invented_category_responses"] == 1


def test_verifier_malformed_body_falls_back_to_review() -> None:
    result = validate_verifier_response(
        "not-json",
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    assert result.decisions == {}
    assert result.invalid_sources == ("one",)


# --- E5 full state machine ---------------------------------------------------


def _classifier_result(raw: str):
    return validate_explicit_abstention_response(raw, ["one"], REAL_CATEGORIES)


def test_state_machine_classify_x_accept_x_gives_x() -> None:
    classifier = _classifier_result(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}'
    )
    verifier = validate_verifier_response(
        '{"decisions":[{"source":"one","decision":"accept","category":"Documents"}]}',
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    outcome = merge_classifier_verifier(classifier, verifier, ["one"], review_directory=REVIEW)

    assert outcome["one"].final == "Documents"
    assert outcome["one"].reason_code == "VERIFIER_ACCEPT"


def test_state_machine_classify_x_verifier_review_gives_review() -> None:
    classifier = _classifier_result(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}'
    )
    verifier = validate_verifier_response(
        '{"decisions":[{"source":"one","decision":"review","category":null}]}',
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    outcome = merge_classifier_verifier(classifier, verifier, ["one"], review_directory=REVIEW)

    assert outcome["one"].final == REVIEW
    assert outcome["one"].reason_code == "VERIFIER_REVIEW"


def test_state_machine_verifier_attempts_different_category_gives_review() -> None:
    classifier = _classifier_result(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}'
    )
    # The strict validator already rejects this into invalid_sources.
    verifier = validate_verifier_response(
        '{"decisions":[{"source":"one","decision":"accept","category":"Code"}]}',
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    outcome = merge_classifier_verifier(classifier, verifier, ["one"], review_directory=REVIEW)

    assert outcome["one"].final == REVIEW
    assert outcome["one"].reason_code == "VERIFIER_INVALID"


def test_state_machine_classifier_review_gives_review_without_calling_verifier() -> None:
    classifier = _classifier_result(
        '{"decisions":[{"source":"one","decision":"review","category":null}]}'
    )
    # No verifier entry exists at all -- the verifier must never be asked.
    verifier = ValidatedVerifierResponse({}, (), (), (), {}, {})

    outcome = merge_classifier_verifier(classifier, verifier, ["one"], review_directory=REVIEW)

    assert outcome["one"].final == REVIEW
    assert outcome["one"].reason_code == "CLASSIFIER_REVIEW"
    assert outcome["one"].verifier_decision is None


def test_state_machine_classifier_invalid_gives_review() -> None:
    classifier = _classifier_result("not-json")
    verifier = ValidatedVerifierResponse({}, (), (), (), {}, {})

    outcome = merge_classifier_verifier(classifier, verifier, ["one"], review_directory=REVIEW)

    assert outcome["one"].final == REVIEW
    assert outcome["one"].reason_code == "CLASSIFIER_INVALID"


def test_state_machine_verifier_invalid_gives_review() -> None:
    classifier = _classifier_result(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}'
    )
    verifier = validate_verifier_response(
        "not-json",
        {"one": "Documents"},
        REAL_CATEGORIES,
    )

    outcome = merge_classifier_verifier(classifier, verifier, ["one"], review_directory=REVIEW)

    assert outcome["one"].final == REVIEW
    assert outcome["one"].reason_code == "VERIFIER_INVALID"


# --- E5 orchestration (run_e5) ----------------------------------------------


def test_run_e5_calls_classifier_once_then_verifier_once() -> None:
    metadata = [{"name": "one", "size_bytes": 0}, {"name": "two", "size_bytes": 0}]
    model = FakeModel(
        '{"decisions":['
        '{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"two","decision":"review","category":null}]}',
        '{"decisions":[{"source":"one","decision":"accept","category":"Documents"}]}',
    )

    final, detail, telemetry = run_e5(model, metadata, REAL_CATEGORIES, review_directory=REVIEW)

    assert len(model.calls) == 2
    assert final == {"one": "Documents", "two": REVIEW}
    assert telemetry["classification_requests"] == 2


def test_run_e5_skips_verifier_call_when_nothing_is_eligible() -> None:
    metadata = [{"name": "one", "size_bytes": 0}]
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"review","category":null}]}',
    )

    final, _detail, _telemetry = run_e5(model, metadata, REAL_CATEGORIES, review_directory=REVIEW)

    assert len(model.calls) == 1  # no verifier call
    assert final == {"one": REVIEW}


def test_run_e5_does_not_reverse_source_order_for_verifier() -> None:
    metadata = [{"name": "a", "size_bytes": 0}, {"name": "b", "size_bytes": 0}]
    model = FakeModel(
        '{"decisions":['
        '{"source":"a","decision":"classify","category":"Documents"},'
        '{"source":"b","decision":"classify","category":"Code"}]}',
        '{"decisions":['
        '{"source":"a","decision":"accept","category":"Documents"},'
        '{"source":"b","decision":"accept","category":"Code"}]}',
    )

    run_e5(model, metadata, REAL_CATEGORIES, review_directory=REVIEW)

    verifier_prompt = model.calls[1]["messages"][0]["content"][0]["text"]
    block = verifier_prompt.split("<VERIFICATION_CANDIDATE_DATA>")[1].split(
        "</VERIFICATION_CANDIDATE_DATA>"
    )[0]
    assert block.index('"a"') < block.index('"b"')  # original order preserved


def test_run_e5_verifier_receives_proposal_but_not_ground_truth() -> None:
    metadata = [{"name": "one", "size_bytes": 0}]
    model = FakeModel(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}',
        '{"decisions":[{"source":"one","decision":"accept","category":"Documents"}]}',
    )

    run_e5(model, metadata, REAL_CATEGORIES, review_directory=REVIEW)

    verifier_prompt = model.calls[1]["messages"][0]["content"][0]["text"]
    assert '"proposed_category":"Documents"' in verifier_prompt
    assert "ground_truth" not in verifier_prompt
    assert "expected" not in verifier_prompt.lower()


def test_verifier_prompt_never_requests_numeric_confidence() -> None:
    prompt = build_verifier_prompt(["one"], {"one": "Documents"}, REAL_CATEGORIES)

    # The prompt explicitly forbids a confidence/score/probability output;
    # it must never *request* one, so the schema stays free of that field
    # (already pinned by test_verifier_schema_has_no_numeric_confidence_field).
    assert "do not output a numeric confidence" in prompt.lower()
    assert '"confidence"' not in prompt
    assert '"score"' not in prompt
    assert '"probability"' not in prompt


def test_verifier_prompt_is_structurally_different_from_classifier_prompt() -> None:
    from tidy.classification import build_explicit_abstention_prompt

    classifier_prompt = build_explicit_abstention_prompt(
        [{"name": "one"}], REAL_CATEGORIES
    )
    verifier_prompt = build_verifier_prompt(["one"], {"one": "Documents"}, REAL_CATEGORIES)

    assert classifier_prompt != verifier_prompt
    assert "verifier" in verifier_prompt.lower()
    assert "verifier" not in classifier_prompt.lower()
    assert "<FILENAME_DATA>" in classifier_prompt
    assert "<VERIFICATION_CANDIDATE_DATA>" in verifier_prompt


# --- Structural guards: metadata-only, no Holdout reference -----------------


_MODULE_PATH = Path(__file__).parents[1].joinpath("evals", "post_holdout_candidates.py")
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")
# The module's own docstring *names* production symbols and Holdout paths
# purely to document that this module leaves them untouched (see the
# analogous split in tests/test_holdout_v2_integrity.py). Only the code body
# after the docstring is scanned for accidental imports/calls/references.
_MODULE_DOCSTRING_END = _MODULE_SOURCE.index('"""', _MODULE_SOURCE.index('"""') + 3) + 3
_MODULE_CODE_BODY = _MODULE_SOURCE[_MODULE_DOCSTRING_END:]


def test_module_never_imports_content_grouping_or_cli_surfaces() -> None:
    forbidden_substrings = (
        "tidy.executor",
        "tidy.tools",
        "tidy.content_parser",
        "tidy.cli",
        "propose_groups",
        "peek_file",
        "PeekSession",
        "allow_remote_content",
        "read_contents=True",
    )
    for needle in forbidden_substrings:
        assert needle not in _MODULE_CODE_BODY, needle


def test_module_never_references_either_holdout_directory() -> None:
    assert "evals/holdout" not in _MODULE_CODE_BODY
    assert "holdout_v2" not in _MODULE_CODE_BODY
    assert "evals.holdout" not in _MODULE_CODE_BODY


def test_module_does_not_import_or_call_production_default_directly() -> None:
    # E4/E5 are eval-only: they never call build_combined_plan / the CLI
    # dispatch that selects the production default for a real run.
    assert "build_combined_plan" not in _MODULE_CODE_BODY
    assert "classify_with_agreement_gate" not in _MODULE_CODE_BODY
