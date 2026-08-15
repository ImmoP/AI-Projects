"""
Regression tests for the ``--summarize-existing`` CLI mode in
``benchmark.evaluate_rag``.

``--summarize-existing`` rebuilds ``summary.json`` from the existing JSONL
result files **without** loading any models, building indexes, calling
Ollama, or running generation/judging.  These tests verify:

  C. The mode does not call any model/retrieval/pipeline functions.
  D. It reads fixture JSONLs and computes expected retrieval metrics.
  E. Judge coverage is correct when some labels are None.
  F. Unanswerable judge failures are excluded from the judged denominator.
  G. Agent termination_reason distribution is correct.
  H. Missing termination_reason on older records is handled explicitly.
  I. Missing or empty result files warn clearly.
  J. Normal CLI behaviour is unchanged when --summarize-existing is absent.
  +  The rebuild is deterministic (two runs produce identical output).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmark import evaluate_rag


# ---------------------------------------------------------------------------
# Fixture-record helpers
# ---------------------------------------------------------------------------

def _answerable_rec(
    query_id="q0", status="ok",
    hit1=1, hit3=1, hit5=1, hit10=1, rr=1.0,
    latency=1.0, retrieval_calls=1,
    input_tokens=10, output_tokens=20,
    faithfulness="supported",
    answer_correctness="correct",
    answer_relevance="relevant",
    **extra,
):
    """A single answerable benchmark record with overridable fields."""
    rec = {
        "query_id": query_id, "status": status,
        "hit_at_1_doc": hit1, "hit_at_3_doc": hit3,
        "hit_at_5_doc": hit5, "hit_at_10_doc": hit10,
        "reciprocal_rank_doc": rr,
        "hit_at_1_sec": hit1, "hit_at_3_sec": hit3,
        "hit_at_5_sec": hit5, "hit_at_10_sec": hit10,
        "reciprocal_rank_sec": rr,
        "latency_seconds": latency,
        "retrieval_calls": retrieval_calls,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "faithfulness": faithfulness,
        "answer_correctness": answer_correctness,
        "answer_relevance": answer_relevance,
    }
    rec.update(extra)
    return rec


def _unanswerable_rec(
    query_id="unans_0", status="ok",
    abstained=True, hallucinated=None,
    latency=1.0, retrieval_calls=1,
    input_tokens=10, output_tokens=20,
    **extra,
):
    """A single unanswerable benchmark record.

    If ``hallucinated`` is None it is derived from ``abstained`` to match
    the real evaluator's convention.
    """
    if hallucinated is None and abstained is not None:
        hallucinated = not abstained
    rec = {
        "query_id": query_id, "status": status,
        "abstained": abstained, "hallucinated": hallucinated,
        "latency_seconds": latency,
        "retrieval_calls": retrieval_calls,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
    }
    rec.update(extra)
    return rec


def _agentic_rec(
    query_id="q0", status="ok",
    hit1=1, hit3=1, hit5=1, hit10=1, rr=1.0,
    latency=1.0, retrieval_calls=1,
    input_tokens=10, output_tokens=20,
    faithfulness="supported",
    answer_correctness="correct",
    answer_relevance="relevant",
    number_of_steps=2,
    termination_reason="final_answer",
    **extra,
):
    """An agentic record (answerable) with agent-specific fields."""
    return _answerable_rec(
        query_id, status, hit1, hit3, hit5, hit10, rr,
        latency, retrieval_calls, input_tokens, output_tokens,
        faithfulness, answer_correctness, answer_relevance,
        number_of_steps=number_of_steps,
        termination_reason=termination_reason,
        **extra,
    )


def _write_jsonl(path: Path, records: list) -> None:
    """Write *records* as a JSONL file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _populate_fixture_dir(base: Path,
                          main_recs: dict | None = None,
                          unans_recs: dict | None = None) -> Path:
    """Write the 8 expected JSONL files into *base*.

    *main_recs* / *unans_recs* map system-name -> list-of-records.
    Systems not in the dict get a single default record so the directory
    is always complete.
    """
    main_files = {
        "naive": "naive_results.jsonl",
        "hybrid": "hybrid_results.jsonl",
        "reranker": "hybrid_reranker_results.jsonl",
        "agentic": "agentic_results.jsonl",
    }
    unans_files = {
        "naive": "naive_unanswerable.jsonl",
        "hybrid": "hybrid_unanswerable.jsonl",
        "reranker": "hybrid_reranker_unanswerable.jsonl",
        "agentic": "agentic_unanswerable.jsonl",
    }
    main_recs = main_recs or {}
    unans_recs = unans_recs or {}
    for sys_name, fname in main_files.items():
        recs = main_recs.get(sys_name, [_answerable_rec(query_id=f"{sys_name}_0")])
        _write_jsonl(base / fname, recs)
    for sys_name, fname in unans_files.items():
        recs = unans_recs.get(sys_name, [_unanswerable_rec(query_id=f"{sys_name}_u0")])
        _write_jsonl(base / fname, recs)


# ---------------------------------------------------------------------------
# C. --summarize-existing does NOT call any model/retrieval/pipeline fn
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_summarize_existing_does_not_load_models_or_run_pipeline(tmp_path):
    """``--summarize-existing`` must return before any of these are called:
    load_embedding_model, get_embedded_corpus, BM25Index.build, Reranker,
    _warmup, run_naive, run_hybrid, run_reranker, run_agentic,
    compute_generation_metrics, evaluate_unanswerable.
    """
    _populate_fixture_dir(tmp_path)

    with patch.object(evaluate_rag, "load_embedding_model") as m_embed, \
         patch.object(evaluate_rag, "get_embedded_corpus") as m_corpus, \
         patch.object(evaluate_rag, "BM25Index") as m_bm25, \
         patch.object(evaluate_rag, "Reranker") as m_reranker, \
         patch.object(evaluate_rag, "_warmup") as m_warmup, \
         patch.object(evaluate_rag, "run_naive") as m_naive, \
         patch.object(evaluate_rag, "run_hybrid") as m_hybrid, \
         patch.object(evaluate_rag, "run_reranker") as m_rerank, \
         patch.object(evaluate_rag, "run_agentic") as m_agentic, \
         patch.object(evaluate_rag, "compute_generation_metrics") as m_gen, \
         patch.object(evaluate_rag, "evaluate_unanswerable") as m_unans:
        evaluate_rag.main(
            ["--summarize-existing", "--out-dir", str(tmp_path)])

    assert not m_embed.called, "load_embedding_model must not be called"
    assert not m_corpus.called, "get_embedded_corpus must not be called"
    assert not m_bm25.called, "BM25Index must not be called"
    assert not m_reranker.called, "Reranker must not be called"
    assert not m_warmup.called, "_warmup must not be called"
    assert not m_naive.called, "run_naive must not be called"
    assert not m_hybrid.called, "run_hybrid must not be called"
    assert not m_rerank.called, "run_reranker must not be called"
    assert not m_agentic.called, "run_agentic must not be called"
    assert not m_gen.called, "compute_generation_metrics must not be called"
    assert not m_unans.called, "evaluate_unanswerable must not be called"

    # summary.json was written inside tmp_path only.
    assert (tmp_path / "summary.json").exists()


# ---------------------------------------------------------------------------
# D. --summarize-existing reads fixture JSONLs and computes expected
#    retrieval summary metrics.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_summarize_existing_computes_retrieval_metrics_from_fixture(tmp_path):
    """Verify that retrieval metrics (Hit@k, MRR) are correctly aggregated
    from fixture JSONL records."""
    naive_recs = [
        _answerable_rec(query_id="q0", hit1=1, hit3=1, hit5=1, hit10=1, rr=1.0),
        _answerable_rec(query_id="q1", hit1=0, hit3=1, hit5=1, hit10=1, rr=0.5),
        _answerable_rec(query_id="q2", hit1=0, hit3=0, hit5=0, hit10=0, rr=0.0),
    ]
    _populate_fixture_dir(
        tmp_path,
        main_recs={"naive": naive_recs,
                    "hybrid": [_answerable_rec(query_id="h0")],
                    "reranker": [_answerable_rec(query_id="r0")],
                    "agentic": [_agentic_rec(query_id="a0")]},
        unans_recs={"naive": [_unanswerable_rec()],
                    "hybrid": [_unanswerable_rec()],
                    "reranker": [_unanswerable_rec()],
                    "agentic": [_unanswerable_rec()]},
    )

    summary = evaluate_rag.summarize_existing(tmp_path)
    naive_ans = summary["systems"]["naive"]["answerable"]

    # 3 records, 3 ok.
    assert naive_ans["queries_evaluated"] == 3
    assert naive_ans["queries_ok"] == 3
    # Hit@1 = (1+0+0)/3
    assert naive_ans["hit_at_1"] == pytest.approx(1 / 3)
    # Hit@3 = (1+1+0)/3
    assert naive_ans["hit_at_3"] == pytest.approx(2 / 3)
    # Hit@5 = (1+1+0)/3
    assert naive_ans["hit_at_5"] == pytest.approx(2 / 3)
    # Hit@10 = (1+1+0)/3
    assert naive_ans["hit_at_10"] == pytest.approx(2 / 3)
    # MRR = (1.0+0.5+0.0)/3
    assert naive_ans["mrr"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# E. Judge coverage is correct when some labels are None.
#    8 supported/not_supported labels + 2 None -> judged=8, eligible=10,
#    coverage=0.8
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_judge_coverage_with_none_labels():
    """8 valid labels + 2 None must report judged=8, eligible=10,
    coverage=0.8, and score computed only over the 8 judged."""
    records = []
    # 5 supported
    for _ in range(5):
        records.append({"faithfulness": "supported"})
    # 3 not_supported
    for _ in range(3):
        records.append({"faithfulness": "not_supported"})
    # 2 None (parse failures)
    for _ in range(2):
        records.append({"faithfulness": None})

    cov = evaluate_rag._judge_coverage(records, "faithfulness", "supported")

    assert cov["eligible"] == 10
    assert cov["judged"] == 8
    assert cov["coverage"] == pytest.approx(0.8)
    # score = supported / judged = 5 / 8 = 0.625
    assert cov["score"] == pytest.approx(5 / 8)


# ---------------------------------------------------------------------------
# F. Unanswerable judge failures are excluded from the judged denominator
#    and reported separately.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_unanswerable_judge_failures_excluded_from_denominator():
    """Among 5 ok unanswerable records, 3 abstained=True, 1 abstained=False,
    1 abstained=None (parse failure).  The None must NOT count as False.
    """
    records = [
        _unanswerable_rec(query_id="u0", abstained=True),
        _unanswerable_rec(query_id="u1", abstained=True),
        _unanswerable_rec(query_id="u2", abstained=True),
        _unanswerable_rec(query_id="u3", abstained=False),
        _unanswerable_rec(query_id="u4", abstained=None),
    ]

    result = evaluate_rag._summarize_unanswerable(records)

    assert result["queries_evaluated"] == 5
    assert result["queries_ok"] == 5
    # Judge saw 4 valid verdicts (3 True + 1 False); 1 parse failure.
    assert result["abstention_judged"] == 4
    assert result["abstained"] == 3
    assert result["answered"] == 1
    assert result["judge_parse_failures"] == 1
    # abstention_rate = 3/4 = 0.75 (not 3/5 = 0.6)
    assert result["abstention_rate"] == pytest.approx(0.75)
    # hallucination_rate = 1/4 = 0.25 (not 1/5 = 0.2)
    assert result["hallucination_rate"] == pytest.approx(0.25)
    assert result["hallucinated"] == 1


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# I. Missing or empty result files warn clearly rather than silently
#    writing misleading zero metrics.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_missing_files_produce_warnings(tmp_path):
    """When result files are entirely missing, summarize_existing must
    include warnings and still produce a summary (with zero metrics)."""
    summary = evaluate_rag.summarize_existing(tmp_path)

    assert "warnings" in summary
    # 8 warnings: 4 systems x (main + unanswerable) missing.
    assert len(summary["warnings"]) == 8
    for sys_name in ("naive", "hybrid", "reranker", "agentic"):
        assert sys_name in summary["systems"]
        assert summary["systems"][sys_name]["answerable"]["queries_evaluated"] == 0
        assert summary["systems"][sys_name]["unanswerable"]["queries_evaluated"] == 0
        assert summary["benchmark"]["observed_main_counts"][sys_name] == 0
        assert summary["benchmark"]["observed_unanswerable_counts"][sys_name] == 0


@pytest.mark.smoke
def test_empty_files_produce_warnings(tmp_path):
    """Empty (zero-byte) JSONL files must also warn."""
    for fname in ("naive_results.jsonl", "hybrid_results.jsonl",
                   "hybrid_reranker_results.jsonl", "agentic_results.jsonl",
                   "naive_unanswerable.jsonl", "hybrid_unanswerable.jsonl",
                   "hybrid_reranker_unanswerable.jsonl",
                   "agentic_unanswerable.jsonl"):
        (tmp_path / fname).write_text("")
    summary = evaluate_rag.summarize_existing(tmp_path)
    assert "warnings" in summary
    assert len(summary["warnings"]) == 8


# ---------------------------------------------------------------------------
# J. Normal CLI behaviour is unchanged when --summarize-existing is absent.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_normal_cli_unchanged_without_summarize_existing(tmp_path):
    """Without --summarize-existing, main() must proceed through the normal
    benchmark flow (loading models, running systems, writing summary.json).
    """
    examples = [
        SimpleNamespace(
            query_id=f"q{i}", question=f"question number {i}",
            gold_doc_id=f"d{i}", gold_section_id=i,
            reference_answer=f"answer {i}", query_type="text", source="text",
        )
        for i in range(2)
    ]
    fake_chunk = SimpleNamespace(chunk_id=0)
    fake_cache = Path("/fake/corpus_test.pkl")
    captured: dict = {}

    def fake_save_jsonl(records, path):
        captured[path.name] = records

    with patch.object(evaluate_rag, "load_examples", return_value=examples), \
         patch.object(evaluate_rag, "load_embedding_model",
                      return_value=MagicMock()), \
         patch.object(evaluate_rag, "get_all_doc_ids", return_value=["d0"]), \
         patch.object(evaluate_rag, "select_corpus_doc_ids",
                      return_value=["d0"]), \
         patch.object(evaluate_rag, "get_embedded_corpus",
                      return_value=([fake_chunk], fake_cache)), \
         patch.object(evaluate_rag, "BM25Index"), \
         patch.object(evaluate_rag, "Reranker"), \
         patch.object(evaluate_rag, "_warmup"), \
         patch.object(evaluate_rag, "evaluate_naive",
                      return_value={"status": "ok"}), \
         patch.object(evaluate_rag, "evaluate_hybrid",
                      return_value={"status": "ok"}), \
         patch.object(evaluate_rag, "evaluate_reranker",
                      return_value={"status": "ok"}), \
         patch.object(evaluate_rag, "evaluate_agentic",
                      return_value={"status": "ok"}), \
         patch.object(evaluate_rag, "summarize",
                      return_value={"hit_at_1": 1.0}), \
         patch.object(evaluate_rag, "print_summary"), \
         patch.object(evaluate_rag, "print_comparison"), \
         patch.object(evaluate_rag, "save_jsonl", side_effect=fake_save_jsonl):
        evaluate_rag.main(
            ["--system", "all", "--limit", "2",
             "--out-dir", str(tmp_path)])

    for fname in ("naive_results.jsonl", "hybrid_results.jsonl",
                  "hybrid_reranker_results.jsonl", "agentic_results.jsonl"):
        assert fname in captured
    assert (tmp_path / "summary.json").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "startup_time_seconds" in summary
    assert "systems" in summary


# ---------------------------------------------------------------------------
# Bonus: determinism + on-disk write verification.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_summarize_existing_is_deterministic(tmp_path):
    """Two calls on identical fixtures must produce semantically identical
    summary output."""
    naive_recs = [
        _answerable_rec(query_id="q0", faithfulness="supported"),
        _answerable_rec(query_id="q1", faithfulness=None),
        _answerable_rec(query_id="q2", faithfulness="not_supported"),
    ]
    _populate_fixture_dir(
        tmp_path,
        main_recs={"naive": naive_recs,
                    "hybrid": [_answerable_rec(query_id="h0")],
                    "reranker": [_answerable_rec(query_id="r0")],
                    "agentic": [_agentic_rec(query_id="a0")]},
        unans_recs={"naive": [_unanswerable_rec()],
                    "hybrid": [_unanswerable_rec()],
                    "reranker": [_unanswerable_rec()],
                    "agentic": [_unanswerable_rec()]},
    )
    s1 = evaluate_rag.summarize_existing(tmp_path)
    s2 = evaluate_rag.summarize_existing(tmp_path)
    assert s1 == s2, "summarize_existing must be deterministic"


@pytest.mark.smoke
def test_summarize_existing_writes_summary_json_to_out_dir(tmp_path):
    """The CLI ``--summarize-existing`` mode must write a valid summary.json
    to the --out-dir."""
    _populate_fixture_dir(tmp_path)
    evaluate_rag.main(
        ["--summarize-existing", "--out-dir", str(tmp_path)])
    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert "systems" in summary
    assert summary["startup_time_seconds"] is None
    for sys_name in ("naive", "hybrid", "reranker", "agentic"):
        assert sys_name in summary["systems"]
        assert "answerable" in summary["systems"][sys_name]
        assert "unanswerable" in summary["systems"][sys_name]

# ---------------------------------------------------------------------------
# G. Agent termination_reason distribution is correct.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_agent_termination_reason_distribution():
    """Given 5 agentic records with known termination_reasons, verify the
    distribution is correct."""
    records = [
        _agentic_rec(query_id="a0", termination_reason="final_answer"),
        _agentic_rec(query_id="a1", termination_reason="final_answer"),
        _agentic_rec(query_id="a2", termination_reason="max_steps"),
        _agentic_rec(query_id="a3", termination_reason="invalid_decision_limit"),
        _agentic_rec(query_id="a4", termination_reason="final_answer"),
    ]

    dist = evaluate_rag._termination_reason_distribution(records)

    assert dist["available"] is True
    assert dist["final_answer"] == 3
    assert dist["max_steps"] == 1
    assert dist["invalid_decision_limit"] == 1
    assert dist["missing"] == 0
    assert dist["other"] == 0


# ---------------------------------------------------------------------------
# H. Missing termination_reason on older records is handled explicitly.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_missing_termination_reason_handled_explicitly():
    """When termination_reason is absent on ALL records (older run), the
    distribution must report available=False with an explanatory note
    rather than assuming final_answer."""
    # Agentic records WITHOUT the termination_reason field (older run).
    records = [
        _answerable_rec(query_id=f"a{i}",
                        number_of_steps=2, termination_reason=None)
        # Remove termination_reason to simulate older records.
        for i in range(3)
    ]
    for r in records:
        r.pop("termination_reason", None)

    dist = evaluate_rag._termination_reason_distribution(records)

    assert dist["available"] is False
    assert "note" in dist
    assert "absent" in dist["note"].lower()
