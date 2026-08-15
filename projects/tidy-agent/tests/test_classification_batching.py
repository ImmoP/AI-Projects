"""Offline contract tests for the batched structured-classification transport.

Covers the reliability contract added for the future Holdout v5 candidate
``E4-batched``: deterministic bounded batching, strict per-batch source-set
validation (set + cardinality equality), fail-closed handling of every
structural defect, E3 agreement-gate semantic preservation, E4-current veto
semantic preservation, and optional provider finish/token metadata.

Every test drives ``StructuredClassifier.classify_with_agreement_gate_batched``
or ``evals.e4_batched.run_e4_batched`` with a mocked model. No Ollama or network
call is ever made. No Holdout v4 case material is used -- all filenames are
synthetic.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tidy.classification import (
    DEFAULT_CLASSIFICATION_BATCH_SIZE,
    ClassificationTelemetry,
    StructuredClassifier,
    chunk_classification_metadata,
    validate_abstention_batch,
)
from evals.e4_batched import run_e4_batched
from evals.post_holdout_candidates import run_e4

REAL_CATEGORIES = ["Documents", "Code"]
REVIEW = "_ToReview"


def _metadata(names):
    return [{"name": n} for n in names]


def _response(content, *, finish_reason=None, with_usage=True, with_raw=False):
    """Minimal provider-like response. ``finish_reason=None`` + ``with_raw``
    False model a provider that surfaces no finish metadata (optionality)."""
    ns = SimpleNamespace(content=content)
    ns.token_usage = (
        SimpleNamespace(input_tokens=5, output_tokens=2) if with_usage else None
    )
    if with_raw:
        ns.raw = SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason)])
    elif finish_reason is not None:
        ns.finish_reason = finish_reason
    return ns


class BatchFakeModel:
    """Parse each batch's FILENAME_DATA and return a configurable response.

    ``decider(source, pass_number)`` returns a decision dict per source
    (default: classify everything as Documents). ``transform(items, names,
    pass_number, call_index)`` may alter the item list before serialization.
    ``finish_for(call_index)`` may return a finish reason. ``raise_on`` is a set
    of call indices that raise a provider error. Call parity: odd = pass 1 of a
    batch, even = pass 2.
    """

    structured_output_mode = "json_schema"

    def __init__(self, decider=None, transform=None, finish_for=None, raise_on=frozenset(), with_usage=True):
        self.decider = decider or (lambda source, pass_number: {
            "source": source, "decision": "classify", "category": "Documents"
        })
        self.transform = transform
        self.finish_for = finish_for or (lambda call_index: None)
        self.raise_on = set(raise_on)
        self.with_usage = with_usage
        self.calls = []  # list of (names_in_order, pass_number)

    def generate(self, messages, **kwargs):
        text = messages[0]["content"][0]["text"]
        start = text.index("<FILENAME_DATA>") + len("<FILENAME_DATA>")
        end = text.index("</FILENAME_DATA>")
        names = [n for n in text[start:end].strip().split("\n") if n]
        call_index = len(self.calls) + 1
        pass_number = 1 if call_index % 2 == 1 else 2
        self.calls.append((names, pass_number))
        if call_index in self.raise_on:
            raise RuntimeError(f"simulated provider failure on call {call_index}")
        items = [self.decider(name, pass_number) for name in names]
        if self.transform is not None:
            items = self.transform(items, names, pass_number, call_index)
        return _response(
            json.dumps({"decisions": items}),
            finish_reason=self.finish_for(call_index),
            with_usage=self.with_usage,
        )


# --- A. Complete response: expected N sources -> exactly N accepted ----------


def test_complete_response_all_sources_accepted():
    names = [f"file{i}" for i in range(5)]
    model = BatchFakeModel()
    result = StructuredClassifier(model).classify_with_agreement_gate_batched(
        _metadata(names), REAL_CATEGORIES, review_directory=REVIEW, batch_size=20
    )
    assert set(result.categories) == set(names)
    assert all(v == "Documents" for v in result.categories.values())
    assert result.invalid_sources == ()
    assert result.telemetry["batch_validation_failures"] == 0
    assert len(model.calls) == 2  # single batch: pass1 + pass2


# --- B/C/D/E/F. Strict per-batch source-set validation fails closed ---------


def _drop_last(items, names, pass_number, call_index):
    return items[:-1]


def _drop_middle(items, names, pass_number, call_index):
    return [it for i, it in enumerate(items) if i != len(items) // 2]


def _add_hallucinated(items, names, pass_number, call_index):
    return items + [{"source": "invented_extra", "decision": "classify", "category": "Code"}]


def _duplicate_first(items, names, pass_number, call_index):
    return items + [dict(items[0])]


def _malformed_item(items, names, pass_number, call_index):
    broken = [dict(items[0])]
    broken[0].pop("category", None)  # classify without a category -> malformed
    return broken + items[1:]


@pytest.mark.parametrize(
    "transform",
    [_drop_last, _drop_middle, _add_hallucinated, _duplicate_first, _malformed_item],
    ids=["missing_last", "missing_middle", "extra_source", "duplicate_source", "malformed_item"],
)
def test_partial_or_invalid_batch_fails_closed(transform):
    names = [f"file{i}" for i in range(5)]
    model = BatchFakeModel(transform=transform)
    result = StructuredClassifier(model).classify_with_agreement_gate_batched(
        _metadata(names), REAL_CATEGORIES, review_directory=REVIEW, batch_size=20
    )
    # Every source in the defective batch is failed closed to review; none is
    # automatically classified from a partial/invalid response.
    assert result.categories == {}
    assert set(result.invalid_sources) == set(names)
    assert result.telemetry["batch_validation_failures"] == 2  # both passes rejected


def test_missing_last_item_fail_closed_unit():
    telemetry = ClassificationTelemetry()
    raw = json.dumps({"decisions": [
        {"source": "a", "decision": "classify", "category": "Documents"},
        {"source": "b", "decision": "classify", "category": "Code"},
    ]})
    clean, result = validate_abstention_batch(raw, ["a", "b", "c"], REAL_CATEGORIES, telemetry)
    assert clean is False
    assert result.omitted_sources == ("c",)


def test_extra_and_duplicate_fail_closed_unit():
    telemetry = ClassificationTelemetry()
    raw = json.dumps({"decisions": [
        {"source": "a", "decision": "classify", "category": "Documents"},
        {"source": "a", "decision": "classify", "category": "Documents"},
        {"source": "b", "decision": "classify", "category": "Code"},
        {"source": "zz", "decision": "classify", "category": "Code"},
    ]})
    clean, _ = validate_abstention_batch(raw, ["a", "b"], REAL_CATEGORIES, telemetry)
    assert clean is False


# --- G. Deterministic batching ----------------------------------------------


def test_chunking_is_deterministic_in_membership_and_order():
    items = _metadata([f"f{i:02d}" for i in range(50)])
    first = chunk_classification_metadata(items, 20)
    second = chunk_classification_metadata(items, 20)
    assert [[str(m["name"]) for m in b] for b in first] == [
        [str(m["name"]) for m in b] for b in second
    ]
    assert [len(b) for b in first] == [20, 20, 10]


# --- H. Batch boundary sizes ------------------------------------------------


@pytest.mark.parametrize(
    "n,expected_sizes",
    [
        (DEFAULT_CLASSIFICATION_BATCH_SIZE, [20]),
        (DEFAULT_CLASSIFICATION_BATCH_SIZE + 1, [20, 1]),
        (2 * DEFAULT_CLASSIFICATION_BATCH_SIZE + 1, [20, 20, 1]),
    ],
)
def test_batch_boundaries(n, expected_sizes):
    names = [f"f{i:02d}" for i in range(n)]
    model = BatchFakeModel()
    boundaries = chunk_classification_metadata(_metadata(names), DEFAULT_CLASSIFICATION_BATCH_SIZE)
    assert [len(b) for b in boundaries] == expected_sizes
    result = StructuredClassifier(model).classify_with_agreement_gate_batched(
        _metadata(names), REAL_CATEGORIES, review_directory=REVIEW,
        batch_size=DEFAULT_CLASSIFICATION_BATCH_SIZE,
    )
    # Two passes per batch.
    assert len(model.calls) == 2 * len(expected_sizes)
    assert set(result.categories) == set(names)
    assert result.telemetry["classification_batches"] == 2 * len(expected_sizes)


# --- I. E3 agreement across batches merges correctly by source ---------------


def test_agreement_across_batches_merges_by_source():
    # 45 sources across 3 batches. decider: doc* -> Documents, code* -> Code,
    # review* -> review. Both passes use the same decider, so every source
    # agrees across passes and the gate resolves it deterministically.
    names = [f"doc{i:02d}" for i in range(20)] + [f"code{i:02d}" for i in range(20)] + [f"review{i:02d}" for i in range(5)]
    model = BatchFakeModel(decider=lambda source, pass_number: (
        {"source": source, "decision": "classify", "category": "Documents"}
        if source.startswith("doc")
        else {"source": source, "decision": "classify", "category": "Code"}
        if source.startswith("code")
        else {"source": source, "decision": "review", "category": None}
    ))
    result = StructuredClassifier(model).classify_with_agreement_gate_batched(
        _metadata(names), REAL_CATEGORIES, review_directory=REVIEW, batch_size=20
    )
    for i in range(20):
        assert result.categories[f"doc{i:02d}"] == "Documents"
        assert result.categories[f"code{i:02d}"] == "Code"
    for i in range(5):
        assert result.categories[f"review{i:02d}"] == REVIEW
    assert set(result.categories) == set(names)
    assert result.invalid_sources == ()


# --- J. One disagreement in one batch affects only that source ---------------


def test_single_disagreement_does_not_corrupt_neighbours():
    names = [f"f{i:02d}" for i in range(45)]  # f10 sits in batch 0 (index 10)
    def decider(source, pass_number):
        if source == "f10":
            # pass1 says Documents, pass2 says Code -> E3 disagreement -> review
            return {"source": source, "decision": "classify",
                    "category": "Documents" if pass_number == 1 else "Code"}
        return {"source": source, "decision": "classify", "category": "Documents"}
    model = BatchFakeModel(decider=decider)
    result = StructuredClassifier(model).classify_with_agreement_gate_batched(
        _metadata(names), REAL_CATEGORIES, review_directory=REVIEW, batch_size=20
    )
    assert result.categories["f10"] == REVIEW  # disagreement routed to review
    for n in names:
        if n != "f10":
            assert result.categories[n] == "Documents"  # neighbours uncorrupted
    assert set(result.categories) == set(names)


# --- K. One entire model batch invalid fails only that batch ----------------


def test_one_whole_batch_invalid_fails_closed_contained_to_that_batch():
    names = [f"f{i:02d}" for i in range(45)]
    # Batch 1 is the second batch: its pass1 = call 3, pass2 = call 4.
    def transform(items, names_list, pass_number, call_index):
        if call_index in (3, 4):
            return items[:-1]  # drop the final source -> missing-source defect
        return items
    model = BatchFakeModel(transform=transform)
    result = StructuredClassifier(model).classify_with_agreement_gate_batched(
        _metadata(names), REAL_CATEGORIES, review_directory=REVIEW, batch_size=20
    )
    batch1 = names[20:40]  # sources 20..39 belong to the invalid batch
    batch0_and_2 = names[:20] + names[40:]
    for n in batch1:
        assert n in result.invalid_sources  # failed closed
    for n in batch0_and_2:
        assert result.categories[n] == "Documents"  # other batches unaffected
    assert result.telemetry["batch_validation_failures"] == 2  # pass1 + pass2 of batch 1


# --- L. Order perturbation is real while membership is preserved -------------


def test_pass2_order_is_perturbed_but_membership_is_equivalent():
    names = [f"f{i:02d}" for i in range(45)]
    model = BatchFakeModel()
    StructuredClassifier(model).classify_with_agreement_gate_batched(
        _metadata(names), REAL_CATEGORIES, review_directory=REVIEW, batch_size=20
    )
    # calls: batch0 pass1 (idx0), batch0 pass2 (idx1), batch1 pass1 (idx2), ...
    # Pass 2 reverses within the same membership (a real order perturbation).
    for batch_index in range(3):
        pass1_names = model.calls[2 * batch_index][0]
        pass2_names = model.calls[2 * batch_index + 1][0]
        assert set(pass1_names) == set(pass2_names)          # identical membership
        assert pass2_names == list(reversed(pass1_names))     # order actually changed


# --- M. E4-current output is unchanged for identical E3 predictions ---------


def test_e4_current_output_unchanged_for_identical_e3_predictions():
    # E3 auto-classifies a few filenames as Documents; E4-current's deterministic
    # veto then runs unchanged on top. The monolithic run_e4 and the batched
    # run_e4_batched (which collapses to one batch here) must produce the exact
    # same final output, proving E4-batched reuses E4-current untouched.
    names = ["doc_report_final", "vertrag_archiv_nummer_dokument", "simple_notes"]
    decider = lambda source, pass_number: {
        "source": source, "decision": "classify", "category": "Documents"
    }
    final_mono, _, _ = run_e4(
        BatchFakeModel(decider=decider), _metadata(names),
        list(REAL_CATEGORIES), review_directory=REVIEW,
    )
    final_batch, detail_batch, _ = run_e4_batched(
        BatchFakeModel(decider=decider), _metadata(names),
        list(REAL_CATEGORIES), review_directory=REVIEW, batch_size=20,
    )
    assert final_mono == final_batch
    # Spot-check that the frozen E4-current veto semantics survive the batched
    # pipeline: a filename carrying two category cues triggers a veto to review.
    by_source = {d["filename"]: d for d in detail_batch}
    assert by_source["vertrag_archiv_nummer_dokument"]["veto_reason_code"] == "MULTI_CATEGORY_STRONG_CUES"
    assert by_source["vertrag_archiv_nummer_dokument"]["final"] == REVIEW
    assert by_source["doc_report_final"]["final"] == "Documents"


# --- N. Provider finish/token metadata optionality ---------------------------


def test_provider_without_finish_or_token_metadata_still_works():
    # with_usage=False => response has token_usage=None and no finish reason.
    names = [f"file{i}" for i in range(5)]
    model = BatchFakeModel(with_usage=False)
    result = StructuredClassifier(model).classify_with_agreement_gate_batched(
        _metadata(names), REAL_CATEGORIES, review_directory=REVIEW, batch_size=20
    )
    assert set(result.categories) == set(names)
    assert result.invalid_sources == ()
    assert result.telemetry["batch_validation_failures"] == 0
    # Every request diagnostic still recorded with finish_reason=None.
    assert len(result.telemetry["request_diagnostics"]) == 2
    assert all(d["finish_reason"] is None for d in result.telemetry["request_diagnostics"])


# --- O. Length/truncation finish reason cannot be a valid complete result ---


def test_length_finish_reason_fails_batch_closed_even_with_complete_looking_content():
    names = [f"file{i}" for i in range(5)]
    # Content is complete JSON, but the provider reports an output-length stop.
    model = BatchFakeModel(finish_for=lambda call_index: "length")
    result = StructuredClassifier(model).classify_with_agreement_gate_batched(
        _metadata(names), REAL_CATEGORIES, review_directory=REVIEW, batch_size=20
    )
    assert result.categories == {}            # nothing auto-classified
    assert set(result.invalid_sources) == set(names)  # failed closed
    assert result.telemetry["length_finish_responses"] == 2
    assert result.telemetry["batch_validation_failures"] == 2


def test_length_finish_reason_surfaces_in_request_diagnostics():
    model = BatchFakeModel(finish_for=lambda call_index: "length")
    result = StructuredClassifier(model).classify_with_agreement_gate_batched(
        _metadata(["a", "b"]), REAL_CATEGORIES, review_directory=REVIEW, batch_size=20
    )
    assert set(d["finish_reason"] for d in result.telemetry["request_diagnostics"]) == {"length"}


# --- Reliability diagnostics --------------------------------------------------


def test_provider_error_fails_batch_closed():
    names = [f"file{i}" for i in range(5)]
    model = BatchFakeModel(raise_on={1, 2})  # both passes of the single batch raise
    result = StructuredClassifier(model).classify_with_agreement_gate_batched(
        _metadata(names), REAL_CATEGORIES, review_directory=REVIEW, batch_size=20
    )
    assert result.categories == {}
    assert set(result.invalid_sources) == set(names)
    assert result.telemetry["provider_errors"] == 2
    assert result.telemetry["batch_validation_failures"] == 2


def test_request_diagnostics_record_batch_size_and_schema_status():
    names = [f"f{i:02d}" for i in range(45)]
    def transform(items, names_list, pass_number, call_index):
        return items[:-1] if call_index in (3, 4) else items
    model = BatchFakeModel(transform=transform)
    result = StructuredClassifier(model).classify_with_agreement_gate_batched(
        _metadata(names), REAL_CATEGORIES, review_directory=REVIEW, batch_size=20
    )
    diag = result.telemetry["request_diagnostics"]
    assert len(diag) == 6
    assert diag[0]["batch_index"] == 0 and diag[0]["pass"] == 1
    assert diag[2]["batch_index"] == 1 and diag[2]["schema_ok"] is False
    assert diag[2]["expected_item_count"] == 20 and diag[2]["batch_size"] == 20
    assert diag[2]["returned_item_count"] == 19  # one item dropped
