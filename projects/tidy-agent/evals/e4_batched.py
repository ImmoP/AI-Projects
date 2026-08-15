"""E4-batched: the revised evaluation candidate for the future Holdout v5.

``E4-batched`` is the *same classification policy* as the candidate evaluated
inconclusively by Holdout v4 (``E3 -> E4-current``), with exactly one change:
the structured-output transport. The two E3 passes are now issued as
deterministic bounded batches with strict per-batch source-set validation
(:meth:`tidy.classification.StructuredClassifier.classify_with_agreement_gate_batched`)
instead of one monolithic long-list request per pass. The E4-current
deterministic ambiguity/conflict veto (``apply_conflict_veto``) is reused
**unchanged** from ``evals/post_holdout_candidates.py`` -- no cue vocabulary,
threshold, category set, or veto criterion is touched.

Nothing here modifies the frozen production E3 default
(``StructuredClassifier.classify_with_agreement_gate``) or the frozen
E4-current functions; this module only composes them. Holdout v4 remains
consumed and ``PARTIAL_INCONCLUSIVE``; this candidate exists so a *revised*
system version can later be evaluated once on a genuinely untouched Holdout
v5. It is evaluation-only and is not wired into the ``tidy`` CLI.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tidy.classification import (
    DEFAULT_CLASSIFICATION_BATCH_SIZE,
    StructuredClassifier,
)

from evals.post_holdout_candidates import apply_conflict_veto

# Candidate designation for the revised system. "E4-batched" = batched E3
# transport + unchanged E4-current veto; it is intentionally *not* named like a
# new classification strategy because the classification policy is unchanged.
CANDIDATE_DESIGNATION = "E4-batched"
CANDIDATE_PIPELINE = "E3-batched -> E4-current"


def run_e4_batched(
    model: Any,
    metadata: Sequence[Mapping[str, Any]],
    real_categories: Sequence[str],
    *,
    review_directory: str,
    batch_size: int = DEFAULT_CLASSIFICATION_BATCH_SIZE,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """Run batched E3, then the unchanged E4-current veto.

    Returns ``(final, detail, telemetry)`` mirroring the shape of
    ``evals.post_holdout_candidates.run_e4`` so downstream scoring/persistence
    stays familiar: ``final`` maps every source to a real category or
    ``review_directory``; ``detail`` is one content-free record per source;
    ``telemetry`` is the batched gate's telemetry snapshot (including the
    per-batch reliability diagnostics).
    """
    sources = [str(item["name"]) for item in metadata]
    classifier = StructuredClassifier(model)
    result = classifier.classify_with_agreement_gate_batched(
        metadata,
        real_categories,
        review_directory=review_directory,
        batch_size=batch_size,
    )
    invalid = set(result.invalid_sources)
    # Every source resolves to either its agreed category or the review
    # directory; a source whose batch failed strict validation in a pass is
    # absent from ``result.categories`` and therefore maps to review (fail
    # closed). No prediction is manufactured for an invalid source.
    e3_final = {source: result.categories.get(source, review_directory) for source in sources}
    veto = apply_conflict_veto(e3_final, sources, review_directory=review_directory)
    final = {source: veto[source].final for source in sources}
    detail = [
        {
            "filename": source,
            "e3_category": e3_final[source],
            "e3_batch_failed": source in invalid,
            "veto_applicable": veto[source].applicable,
            "matched_category_cue_families": veto[source].matched_category_cue_families,
            "conflict_detected": veto[source].conflict_detected,
            "veto_reason_code": veto[source].veto_reason_code,
            "final": veto[source].final,
        }
        for source in sources
    ]
    return final, detail, result.telemetry
