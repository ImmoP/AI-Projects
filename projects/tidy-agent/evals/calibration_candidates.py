"""Evaluation-only abstention candidates E1/E2/E3, compared against baseline E0.

E0 is the existing production two-pass metadata control
(``StructuredClassifier.classify(..., metadata_control=True)``) and is
imported unchanged; nothing here alters it.

Design note on independent passes. E0's two model calls are already
structurally different (a peek-candidate-selection call, then a final
classification call); reusing that shape for a second *classification*
opinion would mean literally the same prompt twice at temperature 0, which a
warm local model reproduces deterministically and would make disagreement
vacuous. Instead, E1 and E3 obtain a genuine second opinion by presenting the
identical instructions and schema with the file list in reversed order for
pass 2 (``reverse_pass_order``). This adds no new wording, asks the model
nothing about its own uncertainty, and exploits a known, well-documented
model behaviour (order/position sensitivity) as the deliberate, Python-
controlled source of two independent samples. Every abstention decision below
is computed by plain Python from already-validated per-pass output; the model
is never asked whether disagreement matters.

E1 — deterministic disagreement abstention. Both passes reuse the *existing*
production classification schema and prompt (``CLASSIFICATION_JSON_SCHEMA``,
``build_classification_prompt``), where ``category`` may already be the
review directory itself. ``merge_disagreement_abstention`` accepts only a
source both passes independently, validly resolved to the identical
category; every other combination (disagreement, one or both passes invalid,
omitted, or duplicated) becomes ``_ToReview``.

E2 — explicit structured abstention. A new schema separates the abstention
signal from the category: ``decision`` is ``"classify"`` or ``"review"``, and
``category`` is required only for ``"classify"`` and must be null/absent for
``"review"``. The review directory never appears as a category value here;
it is expressed exclusively through ``decision``.

E3 — explicit abstention + agreement gate. Both passes use E2's schema
(again with reversed order for pass 2); Python accepts a category only when
both passes explicitly classify into the same category.

E3 was subsequently selected and integrated into production
(``tidy.classification.StructuredClassifier.classify_with_agreement_gate``).
The schema, prompt, validator, and gate it and E2 share are therefore
imported from there rather than redefined here, so this module and
production cannot silently diverge into two different E3 state machines.
Only E1 (not selected, not integrated) still lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from tidy.classification import (  # noqa: F401 -- re-exported for eval-script imports
    EXPLICIT_ABSTENTION_JSON_SCHEMA,
    AbstentionDecision,
    AgreementGateOutcome,
    ClassificationTelemetry,
    ValidatedAbstentionClassification,
    ValidatedClassification,
    build_explicit_abstention_prompt,
    merge_agreement_gate,
    reverse_pass_order,
    validate_explicit_abstention_response,
)


def resolve_explicit_abstention(
    pass_result: ValidatedAbstentionClassification,
    sources: Sequence[str],
    *,
    review_directory: str,
) -> dict[str, str]:
    """E2 standalone resolution: valid ``classify`` wins, everything else reviews."""
    resolved: dict[str, str] = {}
    for source in sources:
        decision = pass_result.decisions.get(source)
        if decision is not None and decision.decision == "classify" and decision.category:
            resolved[source] = decision.category
        else:
            resolved[source] = review_directory
    return resolved


@dataclass(frozen=True)
class DisagreementOutcome:
    source: str
    final: str
    pass1_valid: bool
    pass2_valid: bool
    pass1_category: str | None
    pass2_category: str | None
    agreement: Literal[
        "agree", "disagree", "pass1_invalid", "pass2_invalid", "both_invalid"
    ]


def merge_disagreement_abstention(
    pass1: ValidatedClassification,
    pass2: ValidatedClassification,
    sources: Sequence[str],
    *,
    review_directory: str,
) -> dict[str, DisagreementOutcome]:
    """E1 state table: accept only when both independently valid passes agree.

    | pass 1            | pass 2            | final        |
    |--------------------|--------------------|--------------|
    | category X         | category X         | X            |
    | category X         | category Y (!=X)    | `_ToReview`  |
    | category X         | `_ToReview`         | `_ToReview`  |
    | `_ToReview`         | category X         | `_ToReview`  |
    | `_ToReview`         | `_ToReview`         | `_ToReview`  |
    | invalid/omitted    | (anything)          | `_ToReview`  |
    | (anything)          | invalid/omitted    | `_ToReview`  |

    "Invalid/omitted" covers a malformed pass, a duplicated source, or a
    source the pass never mentioned — ``validate_classification_response``
    already resolves all three to "not present in ``.categories``" before
    this function ever runs, so no duplicate-specific branch is needed here.
    """
    results: dict[str, DisagreementOutcome] = {}
    for source in sources:
        cat1 = pass1.categories.get(source)
        cat2 = pass2.categories.get(source)
        valid1 = cat1 is not None
        valid2 = cat2 is not None
        if valid1 and valid2:
            if cat1 == cat2:
                final = cat1
                agreement: Any = "agree"
            else:
                final = review_directory
                agreement = "disagree"
        elif valid1 and not valid2:
            final = review_directory
            agreement = "pass2_invalid"
        elif valid2 and not valid1:
            final = review_directory
            agreement = "pass1_invalid"
        else:
            final = review_directory
            agreement = "both_invalid"
        results[source] = DisagreementOutcome(
            source, final, valid1, valid2, cat1, cat2, agreement
        )
    return results


# AgreementGateOutcome and merge_agreement_gate (E3's state table) now live in
# tidy.classification and are imported above; see the module docstring.
