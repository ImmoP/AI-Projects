"""Offline, deterministic tests for the E1/E2/E3 abstention candidates.

No Ollama dependency: every test drives the pure validation/merge functions
directly with hand-built JSON, or a ``FakeModel`` identical in spirit to
``tests/test_classification.py``. Nothing here touches the consumed 41-file
Holdout, and none of these tests reference its fixture path or filenames
(required item 24) — the calibration fixture used elsewhere in this cycle is
the separate, reusable ``development-calibration`` set.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from evals.calibration_candidates import (
    EXPLICIT_ABSTENTION_JSON_SCHEMA,
    build_explicit_abstention_prompt,
    merge_agreement_gate,
    merge_disagreement_abstention,
    resolve_explicit_abstention,
    reverse_pass_order,
    validate_explicit_abstention_response,
)
from tidy.classification import (
    StructuredClassifier,
    validate_classification_response,
)

SOURCES = ["one", "two"]
CATEGORIES_WITH_REVIEW = ["Documents", "Code", "_ToReview"]
REAL_CATEGORIES = ["Documents", "Code"]


def _pass(raw: str):
    return validate_classification_response(raw, SOURCES, CATEGORIES_WITH_REVIEW)


def _abstention_pass(raw: str):
    return validate_explicit_abstention_response(raw, SOURCES, REAL_CATEGORIES)


# --- E1: deterministic disagreement abstention ------------------------------


def test_e1_both_passes_same_valid_category_is_accepted() -> None:
    pass1 = _pass('{"decisions":[{"source":"one","category":"Documents"}]}')
    pass2 = _pass('{"decisions":[{"source":"one","category":"Documents"}]}')

    outcome = merge_disagreement_abstention(
        pass1, pass2, ["one"], review_directory="_ToReview"
    )

    assert outcome["one"].final == "Documents"
    assert outcome["one"].agreement == "agree"


def test_e1_passes_disagree_on_different_categories_reviews() -> None:
    pass1 = _pass('{"decisions":[{"source":"one","category":"Documents"}]}')
    pass2 = _pass('{"decisions":[{"source":"one","category":"Code"}]}')

    outcome = merge_disagreement_abstention(
        pass1, pass2, ["one"], review_directory="_ToReview"
    )

    assert outcome["one"].final == "_ToReview"
    assert outcome["one"].agreement == "disagree"


def test_e1_category_then_review_reviews() -> None:
    pass1 = _pass('{"decisions":[{"source":"one","category":"Documents"}]}')
    pass2 = _pass('{"decisions":[{"source":"one","category":"_ToReview"}]}')

    outcome = merge_disagreement_abstention(
        pass1, pass2, ["one"], review_directory="_ToReview"
    )

    assert outcome["one"].final == "_ToReview"
    assert outcome["one"].agreement == "disagree"


def test_e1_review_then_category_reviews() -> None:
    pass1 = _pass('{"decisions":[{"source":"one","category":"_ToReview"}]}')
    pass2 = _pass('{"decisions":[{"source":"one","category":"Documents"}]}')

    outcome = merge_disagreement_abstention(
        pass1, pass2, ["one"], review_directory="_ToReview"
    )

    assert outcome["one"].final == "_ToReview"
    assert outcome["one"].agreement == "disagree"


def test_e1_both_review_reviews() -> None:
    pass1 = _pass('{"decisions":[{"source":"one","category":"_ToReview"}]}')
    pass2 = _pass('{"decisions":[{"source":"one","category":"_ToReview"}]}')

    outcome = merge_disagreement_abstention(
        pass1, pass2, ["one"], review_directory="_ToReview"
    )

    assert outcome["one"].final == "_ToReview"
    assert outcome["one"].agreement == "agree"  # agreeing to review is still review


def test_e1_invalid_first_pass_reviews() -> None:
    pass1 = _pass("not-json")
    pass2 = _pass('{"decisions":[{"source":"one","category":"Documents"}]}')

    outcome = merge_disagreement_abstention(
        pass1, pass2, ["one"], review_directory="_ToReview"
    )

    assert outcome["one"].final == "_ToReview"
    assert outcome["one"].agreement == "pass1_invalid"
    assert outcome["one"].pass1_valid is False


def test_e1_invalid_second_pass_reviews() -> None:
    pass1 = _pass('{"decisions":[{"source":"one","category":"Documents"}]}')
    pass2 = _pass("not-json")

    outcome = merge_disagreement_abstention(
        pass1, pass2, ["one"], review_directory="_ToReview"
    )

    assert outcome["one"].final == "_ToReview"
    assert outcome["one"].agreement == "pass2_invalid"
    assert outcome["one"].pass2_valid is False


# --- E2: explicit structured abstention -------------------------------------


def test_e2_valid_explicit_classify() -> None:
    result = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}'
    )

    assert result.decisions["one"].decision == "classify"
    assert result.decisions["one"].category == "Documents"
    assert result.invalid_sources == ()


def test_e2_valid_explicit_review() -> None:
    result = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"review","category":null}]}'
    )

    assert result.decisions["one"].decision == "review"
    assert result.decisions["one"].category is None
    assert result.invalid_sources == ()


def test_e2_classify_without_category_is_rejected() -> None:
    result = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"classify","category":null}]}'
    )

    assert "one" not in result.decisions
    assert result.invalid_sources == ("one",)


def test_e2_review_with_category_is_rejected() -> None:
    result = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"review","category":"Documents"}]}'
    )

    assert "one" not in result.decisions
    assert result.invalid_sources == ("one",)


def test_e2_invalid_decision_enum_is_rejected() -> None:
    result = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"maybe","category":null}]}'
    )

    assert "one" not in result.decisions
    assert result.invalid_sources == ("one",)


def test_e2_unknown_category_is_rejected() -> None:
    result = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"classify","category":"SecretFolder"}]}'
    )

    assert "one" not in result.decisions
    assert result.invalid_sources == ("one",)
    assert result.telemetry["invented_category_responses"] == 1


def test_e2_invalid_reasons_break_down_the_specific_combinations() -> None:
    classify_without_category = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"classify","category":null}]}'
    )
    review_with_category = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"review","category":"Documents"}]}'
    )
    bad_enum = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"maybe","category":null}]}'
    )

    assert classify_without_category.invalid_reasons == {
        "one": "missing_category_for_classify"
    }
    assert review_with_category.invalid_reasons == {"one": "category_present_for_review"}
    assert bad_enum.invalid_reasons == {"one": "invalid_decision_enum"}


def test_e2_missing_source_falls_back_to_review() -> None:
    result = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}'
    )
    assert result.omitted_sources == ("two",)

    resolved = resolve_explicit_abstention(result, SOURCES, review_directory="_ToReview")

    assert resolved == {"one": "Documents", "two": "_ToReview"}


def test_e2_duplicate_source_is_rejected() -> None:
    result = _abstention_pass(
        '{"decisions":['
        '{"source":"one","decision":"classify","category":"Documents"},'
        '{"source":"one","decision":"classify","category":"Code"}]}'
    )

    assert "one" not in result.decisions
    assert result.invalid_sources == ("one",)
    assert result.telemetry["duplicate_source_responses"] == 1


def test_e2_schema_requires_source_decision_and_category_keys() -> None:
    schema = EXPLICIT_ABSTENTION_JSON_SCHEMA["properties"]["decisions"]["items"]
    assert set(schema["required"]) == {"source", "decision", "category"}
    assert schema["properties"]["decision"]["enum"] == ["classify", "review"]
    assert schema["additionalProperties"] is False


# --- E3: explicit abstention + agreement gate -------------------------------


def test_e3_both_classify_same_category_is_accepted() -> None:
    pass1 = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}'
    )
    pass2 = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}'
    )

    outcome = merge_agreement_gate(pass1, pass2, ["one"], review_directory="_ToReview")

    assert outcome["one"].final == "Documents"
    assert outcome["one"].agreement == "agree_classify"


def test_e3_classify_disagreement_reviews() -> None:
    pass1 = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}'
    )
    pass2 = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"classify","category":"Code"}]}'
    )

    outcome = merge_agreement_gate(pass1, pass2, ["one"], review_directory="_ToReview")

    assert outcome["one"].final == "_ToReview"
    assert outcome["one"].agreement == "disagree_classify"


def test_e3_classify_review_mismatch_reviews() -> None:
    pass1 = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"classify","category":"Documents"}]}'
    )
    pass2 = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"review","category":null}]}'
    )

    outcome = merge_agreement_gate(pass1, pass2, ["one"], review_directory="_ToReview")

    assert outcome["one"].final == "_ToReview"
    assert outcome["one"].agreement == "review_involved"

    # order-swapped mirror: review then classify also reviews
    reversed_outcome = merge_agreement_gate(
        pass2, pass1, ["one"], review_directory="_ToReview"
    )
    assert reversed_outcome["one"].final == "_ToReview"


def test_e3_both_review_reviews() -> None:
    pass1 = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"review","category":null}]}'
    )
    pass2 = _abstention_pass(
        '{"decisions":[{"source":"one","decision":"review","category":null}]}'
    )

    outcome = merge_agreement_gate(pass1, pass2, ["one"], review_directory="_ToReview")

    assert outcome["one"].final == "_ToReview"
    assert outcome["one"].agreement == "review_involved"


# --- Pass-order helper -------------------------------------------------------


def test_reverse_pass_order_is_deterministic_and_content_preserving() -> None:
    metadata = [{"name": "a"}, {"name": "b"}, {"name": "c"}]

    reversed_once = reverse_pass_order(metadata)
    reversed_twice = reverse_pass_order(reversed_once)

    assert reversed_once == ({"name": "c"}, {"name": "b"}, {"name": "a"})
    assert reversed_twice == tuple(metadata)


def test_explicit_abstention_prompt_never_offers_review_directory_as_category() -> None:
    prompt = build_explicit_abstention_prompt([{"name": "one"}], REAL_CATEGORIES)

    assert "_ToReview" not in prompt
    assert "Documents" in prompt and "Code" in prompt


# --- Regression: production surfaces this cycle must not touch -------------


class FakeModel:
    structured_output_mode = "json_schema"

    def __init__(self, *responses):
        self.responses = list(responses)

    def generate(self, messages, **kwargs):
        response = self.responses.pop(0)
        return SimpleNamespace(
            content=response,
            token_usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        )


def test_e0_metadata_control_behaviour_is_unchanged() -> None:
    """E0 is untouched production code; this pins its existing two-call shape."""
    model = FakeModel(
        '{"requests":[{"source":"one"}]}',
        '{"decisions":[{"source":"one","category":"Documents"},'
        '{"source":"two","category":"Code"}]}',
    )
    classifier = StructuredClassifier(model)
    metadata = [
        {"name": "one", "size_bytes": 10},
        {"name": "two", "size_bytes": 20},
    ]

    result = classifier.classify(
        metadata, CATEGORIES_WITH_REVIEW, metadata_control=True
    )

    assert result.categories == {"one": "Documents", "two": "Code"}
    assert result.telemetry["classification_requests"] == 2
    assert result.telemetry["peek_requests_authorized"] == 0
    assert result.telemetry["content_unavailable"] == 0


_CALIBRATION_SOURCE = Path(__file__).parents[1].joinpath(
    "evals", "calibration_candidates.py"
).read_text(encoding="utf-8")


def test_calibration_module_never_imports_grouping_or_content_or_executor() -> None:
    """Structural guard for items 20-23: rules/executor/grouping/content/CLI

    are provably untouched by this module's imports. The corresponding
    source files themselves are additionally unmodified, which
    ``git diff --stat`` on this change set confirms directly.
    """
    forbidden_substrings = (
        "tidy.executor",
        "tidy.tools",
        "tidy.content_parser",
        "tidy.cli",
        "tidy.rules",
        "propose_groups",
        "peek_file",
        "PeekSession",
        "allow_remote_content",
        "read_contents",
    )
    for needle in forbidden_substrings:
        assert needle not in _CALIBRATION_SOURCE, needle


def test_calibration_module_reads_no_holdout_path_or_fixture() -> None:
    """Item 24: no new evaluation code in this module references the consumed

    Holdout's directory or dataset.
    """
    assert "holdout" not in _CALIBRATION_SOURCE.lower()
