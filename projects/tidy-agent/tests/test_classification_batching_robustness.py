"""Development-only robustness check for the batched classification transport.

This is **Development evidence**, not a Holdout result. It exercises the
batched E3 + E4-current path against real *Development* fixture **filenames
only** (calibration / boundary_calibration / veto_precision_calibration) with a
mocked model, verifying the transport invariants at realistic scale: no source
omissions, exact request/response cardinality, deterministic merging, and that
every request stays within the bounded batch size. No fixture file content, no
labels, no ground truth, and no Holdout v4 material is read; the mock returns a
content-free ``review`` decision for every source so no classification
semantics are under test here -- only transport integrity.

These numbers must never be advertised as final generalization metrics; they
are calibration-device evidence for the batching logic only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tidy.classification import DEFAULT_CLASSIFICATION_BATCH_SIZE, StructuredClassifier
from evals.e4_batched import run_e4_batched

REAL_CATEGORIES = ["Documents", "Code"]
REVIEW = "_ToReview"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEV_FIXTURE_DIRS = (
    PROJECT_ROOT / "evals" / "calibration" / "fixture",
    PROJECT_ROOT / "evals" / "boundary_calibration" / "fixture",
    PROJECT_ROOT / "evals" / "veto_precision_calibration" / "fixture",
)


def _fixture_names():
    """Filename strings only -- never content, labels, or ground truth."""
    names = []
    for d in DEV_FIXTURE_DIRS:
        if d.is_dir():
            names.extend(sorted(p.name for p in d.iterdir() if p.is_file()))
    return sorted(set(names))


class ContentFreeReviewModel:
    """Returns a valid review decision for every source in a batch. No network."""

    structured_output_mode = "json_schema"

    def __init__(self):
        self.calls = []  # (names, batch_size)

    def generate(self, messages, **kwargs):
        text = messages[0]["content"][0]["text"]
        start = text.index("<FILENAME_DATA>") + len("<FILENAME_DATA>")
        end = text.index("</FILENAME_DATA>")
        names = [n for n in text[start:end].strip().split("\n") if n]
        self.calls.append(names)
        decisions = [
            {"source": n, "decision": "review", "category": None} for n in names
        ]
        return SimpleNamespace(
            content=json.dumps({"decisions": decisions}),
            token_usage=SimpleNamespace(input_tokens=3, output_tokens=1),
        )


def test_batched_transport_on_development_fixtures_is_lossless():
    names = _fixture_names()
    assert len(names) > 50, "timed fixture corpus unexpectedly small"
    model = ContentFreeReviewModel()
    metadata = [{"name": n} for n in names]

    result = StructuredClassifier(model).classify_with_agreement_gate_batched(
        metadata, REAL_CATEGORIES, review_directory=REVIEW, batch_size=20
    )

    # Every source appears exactly once; nothing omitted, nothing invalid.
    assert set(result.categories) == set(names)
    assert result.invalid_sources == ()
    assert result.telemetry["batch_validation_failures"] == 0

    # Bounding: no single request ever exceeds the conservative batch size.
    request_sizes = [len(c) for c in model.calls]
    assert max(request_sizes) <= DEFAULT_CLASSIFICATION_BATCH_SIZE
    num_batches = len(_num_batches(len(names), DEFAULT_CLASSIFICATION_BATCH_SIZE))
    assert len(model.calls) == 2 * num_batches

    # Determinism: two independent runs over the same names agree exactly.
    second = StructuredClassifier(ContentFreeReviewModel()).classify_with_agreement_gate_batched(
        metadata, REAL_CATEGORIES, review_directory=REVIEW, batch_size=20
    )
    assert second.categories == result.categories


def _num_batches(n, batch_size):
    return [batch_size] * (n // batch_size) + ([n % batch_size] if n % batch_size else [])


def test_e4_batched_runs_end_to_end_on_development_fixture_names(tmp_path):
    # A full E3-batched + E4-current pass over the aggregated dev filenames
    # completes with every source resolved, confirming reliality changes do not
    # break the eval pipeline shape. (Development evidence only.)
    names = _fixture_names()
    final, detail, telemetry = run_e4_batched(
        ContentFreeReviewModel(), [{"name": n} for n in names],
        list(REAL_CATEGORIES), review_directory=REVIEW, batch_size=20,
    )
    assert set(final) == set(names)
    assert all(v == REVIEW for v in final.values())  # every source survived as review
    assert len(detail) == len(names)
    assert telemetry["batch_validation_failures"] == 0
    assert telemetry["request_diagnostics"]  # per-batch diagnostics persisted