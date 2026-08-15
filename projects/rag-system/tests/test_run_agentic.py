"""
Regression tests for the run_agentic() wrapper in benchmark.evaluate_rag.

Guards against the original bug: run_agentic() was a stub that returned None
(no loop, no evaluate_agentic() call), so ``recs = run_agentic(...)`` was
None and ``save_jsonl(recs, ...)`` crashed with
``TypeError: 'NoneType' object is not iterable``.

Verified:
  A. run_agentic() returns a list, never None.
  B. evaluate_agentic() is invoked once per example.
  C. records are returned in the same order as the subset.
  D. the wrapper forwards embedded_chunks, model, top_k, ollama_model,
     base_url and max_steps.
  E. a mocked --system agentic run reaches save_jsonl with an iterable.
Plus an optional --system all regression proving the system loop passes
through naive/hybrid/reranker/agentic without any wrapper returning None.

evaluate_agentic() itself is mocked throughout -- the agent is not exercised.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmark import evaluate_rag


def _examples(n: int = 3) -> list:
    return [
        SimpleNamespace(
            query_id=f"q{i}",
            question=f"question number {i} about cats and dogs",
            gold_doc_id=f"d{i}",
            gold_section_id=i,
            reference_answer=f"answer {i}",
            query_type="text",
            source="text",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# A. run_agentic() returns a list, never None
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_run_agentic_returns_list_not_none():
    with patch.object(evaluate_rag, "evaluate_agentic",
                      return_value={"status": "ok"}):
        recs = evaluate_rag.run_agentic(
            _examples(2), [], object(), top_k=5,
            ollama_model="m", base_url="u", max_steps=3,
        )
    assert recs is not None
    assert isinstance(recs, list)
    assert len(recs) == 2


# ---------------------------------------------------------------------------
# B. evaluate_agentic() is invoked once per example
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_run_agentic_calls_evaluate_agentic_once_per_example():
    examples = _examples(3)
    with patch.object(evaluate_rag, "evaluate_agentic") as m_eval:
        m_eval.return_value = {"status": "ok"}
        recs = evaluate_rag.run_agentic(
            examples, [], object(), top_k=5,
            ollama_model="m", base_url="u", max_steps=3,
        )
    assert m_eval.call_count == 3
    assert len(recs) == 3


# ---------------------------------------------------------------------------
# C. records are returned in the same order as the subset
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_run_agentic_preserves_subset_order():
    examples = _examples(3)

    def fake_eval(ex, *_args, **_kwargs):
        return {"status": "ok", "query_id": ex.query_id}

    with patch.object(evaluate_rag, "evaluate_agentic", side_effect=fake_eval):
        recs = evaluate_rag.run_agentic(
            examples, [], object(), top_k=5,
            ollama_model="m", base_url="u", max_steps=3,
        )
    assert [r["query_id"] for r in recs] == [e.query_id for e in examples]


# ---------------------------------------------------------------------------
# D. the wrapper forwards embedded_chunks, model, top_k, ollama_model,
#    base_url and max_steps correctly
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_run_agentic_forwards_arguments():
    examples = _examples(1)
    chunks = object()   # sentinel for identity check
    model = object()    # sentinel for identity check

    with patch.object(evaluate_rag, "evaluate_agentic") as m_eval:
        m_eval.return_value = {"status": "ok"}
        evaluate_rag.run_agentic(
            examples, chunks, model, top_k=7,
            ollama_model="llama", base_url="http://x:1234", max_steps=9,
        )

    m_eval.assert_called_once()
    args, kwargs = m_eval.call_args
    # Positional: example, embedded_chunks, model
    assert args[0] is examples[0]
    assert args[1] is chunks
    assert args[2] is model
    # Keyword: top_k, ollama_model, base_url, max_steps
    assert kwargs == {
        "top_k": 7, "ollama_model": "llama",
        "base_url": "http://x:1234", "max_steps": 9,
    }


# ---------------------------------------------------------------------------
# E. a mocked --system agentic run reaches save_jsonl with an iterable
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_system_agentic_path_reaches_save_jsonl_with_iterable(tmp_path):
    examples = _examples(2)
    fake_chunk = SimpleNamespace(chunk_id=0)
    fake_cache = Path("/fake/corpus_test.pkl")
    captured: dict = {}

    def fake_save_jsonl(records, path):
        captured["records"] = records
        captured["path"] = path

    with patch.object(evaluate_rag, "load_examples", return_value=examples), \
         patch.object(evaluate_rag, "load_embedding_model",
                      return_value=MagicMock()), \
         patch.object(evaluate_rag, "get_all_doc_ids", return_value=["d0"]), \
         patch.object(evaluate_rag, "select_corpus_doc_ids",
                      return_value=["d0"]), \
         patch.object(evaluate_rag, "get_embedded_corpus",
                      return_value=([fake_chunk], fake_cache)), \
         patch.object(evaluate_rag, "_warmup"), \
         patch.object(evaluate_rag, "evaluate_agentic",
                      return_value={"status": "ok"}), \
         patch.object(evaluate_rag, "summarize", return_value={}), \
         patch.object(evaluate_rag, "print_summary"), \
         patch.object(evaluate_rag, "save_jsonl", side_effect=fake_save_jsonl):
        evaluate_rag.main(
            ["--system", "agentic", "--limit", "2",
             "--out-dir", str(tmp_path)])

    recs = captured["records"]
    assert recs is not None
    assert isinstance(recs, list)
    assert len(recs) == 2
    assert "agentic_results.jsonl" in str(captured["path"])


# ---------------------------------------------------------------------------
# Optional: --system all loop yields iterable records for every system
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_system_all_loop_produces_iterable_records_for_each_system(tmp_path):
    examples = _examples(2)
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
         patch.object(evaluate_rag, "summarize", return_value={}), \
         patch.object(evaluate_rag, "print_summary"), \
         patch.object(evaluate_rag, "print_comparison"), \
         patch.object(evaluate_rag, "save_jsonl", side_effect=fake_save_jsonl):
        evaluate_rag.main(
            ["--system", "all", "--limit", "2",
             "--out-dir", str(tmp_path)])

    # Every wrapper must hand save_jsonl a non-None iterable (not None).
    for fname in ("naive_results.jsonl", "hybrid_results.jsonl",
                  "hybrid_reranker_results.jsonl", "agentic_results.jsonl"):
        assert fname in captured, f"save_jsonl not reached for {fname}"
        recs = captured[fname]
        assert recs is not None, f"{fname}: records is None"
        assert isinstance(recs, list), f"{fname}: not a list"
        assert len(recs) == 2, f"{fname}: expected 2 records, got {len(recs)}"