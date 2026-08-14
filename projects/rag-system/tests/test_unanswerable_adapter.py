"""
Regression tests for the unanswerable-evaluation adapter in
benchmark.evaluate_rag.

Guards against the original crash:

    [unans:naive] unans_001
    AttributeError: 'NoneType' object has no attribute 'get'
    at: if rec.get("status") == "ok":

Root cause: ``_run_unans()`` dispatched on ``run_fn.__name__`` with substring
matching but had (1) NO naive branch (so run_naive fell through and returned
None) and (2) passed the single Example object ``ex`` to the batch ``run_*``
wrappers, which expect an iterable ``subset`` and return a *list*.

Verified here:

  A. Naive:  run_naive receives ``[ex]``; _run_unans returns its first record
             dict; never None.
  B. Hybrid: receives ``[ex]``; BM25 + hybrid params forwarded; one record.
  C. Reranker: receives ``[ex]``; BM25 + reranker + candidate params forwarded;
               one record.
  D. Agentic: receives ``[ex]``; top_k / model / base_url / max_steps forwarded;
              one record.
  E. An unsupported run function raises ValueError instead of returning None.
     A wrapper that returns an empty list also raises ValueError.
  F. evaluate_unanswerable() can process one mocked unanswerable query for each
     of the four systems without ``rec`` being None.
  G. ``--unanswerable-only`` skips the answerable query loop and generation
     judging, never passes the main ``*_results.jsonl`` files to save_jsonl,
     and writes only the ``*_unanswerable.jsonl`` files.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmark import evaluate_rag


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ex(query_id: str = "unans_001",
             question: str = "What is the exact GPT-4 parameter count?"):
    """A single example object shaped like the one evaluate_unanswerable builds."""
    return SimpleNamespace(
        query_id=query_id,
        question=question,
        reference_answer="",
        gold_doc_id="",
        gold_section_id=0,
        query_type="unanswerable",
        source="unanswerable",
    )


def _rcfg() -> dict:
    """A retrieval-config dict identical to the one main() builds from args."""
    return {
        "top_k": 5,
        "dense_k": 50,
        "bm25_k": 50,
        "rrf_k": 60,
        "candidate_k": 50,
        "max_steps": 5,
    }


def _stub_run(return_record: dict, name: str):
    """A stand-in for a run_* wrapper.

    Mimics the real batch wrappers: takes an *iterable* ``subset``, returns a
    *list* of records.  Captures its call args so tests can assert that the
    subset was ``[ex]`` (not the bare object) and that system-specific kwargs
    were forwarded unchanged.
    """
    captured: dict = {}

    def wrapper(subset, *args, **kwargs):
        captured["subset"] = subset
        captured["args"] = args
        captured["kwargs"] = kwargs
        return [return_record]

    wrapper.__name__ = name  # exact-name dispatch in _run_unans relies on this
    wrapper.captured = captured
    return wrapper


# ---------------------------------------------------------------------------
# A. Naive: run_naive receives [ex]; _run_unans returns its first record
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_run_unans_naive_receives_list_and_returns_record():
    ex = _make_ex()
    chunks, model = object(), object()
    rcfg = _rcfg()
    record = {"status": "ok", "answer": "I cannot answer this question."}
    stub = _stub_run(record, "run_naive")

    rec = evaluate_rag._run_unans(
        ex, stub, chunks, model, None, None, rcfg, "qwen2.5:3b", "http://x:11434")

    # The wrapper must receive [ex] (a one-element list), not the bare object.
    assert stub.captured["subset"] == [ex]
    assert stub.captured["subset"][0] is ex
    # Positional args forwarded unchanged.
    assert stub.captured["args"] == (chunks, model)
    # Naive-specific kwargs forwarded unchanged.
    assert stub.captured["kwargs"] == {
        "top_k": rcfg["top_k"], "ollama_model": "qwen2.5:3b",
        "base_url": "http://x:11434",
    }
    # _run_unans returns the single record dict -- never None, never the list.
    assert rec is record
    assert isinstance(rec, dict)


# ---------------------------------------------------------------------------
# B. Hybrid: receives [ex]; BM25 + hybrid params forwarded; one record
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_run_unans_hybrid_receives_list_and_forwards_params():
    ex = _make_ex()
    chunks, model, bm25 = object(), object(), object()
    rcfg = _rcfg()
    record = {"status": "ok", "answer": "x"}
    stub = _stub_run(record, "run_hybrid")

    rec = evaluate_rag._run_unans(
        ex, stub, chunks, model, bm25, None, rcfg, "qwen", "http://x")

    assert stub.captured["subset"] == [ex]
    assert stub.captured["args"] == (chunks, model, bm25)
    assert stub.captured["kwargs"] == {
        "top_k": rcfg["top_k"], "dense_k": rcfg["dense_k"],
        "bm25_k": rcfg["bm25_k"], "rrf_k": rcfg["rrf_k"],
        "ollama_model": "qwen", "base_url": "http://x",
    }
    assert rec is record


# ---------------------------------------------------------------------------
# C. Reranker: receives [ex]; BM25 + reranker + candidate params forwarded
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_run_unans_reranker_receives_list_and_forwards_params():
    ex = _make_ex()
    chunks, model, bm25, reranker = object(), object(), object(), object()
    rcfg = _rcfg()
    record = {"status": "ok", "answer": "x"}
    stub = _stub_run(record, "run_reranker")

    rec = evaluate_rag._run_unans(
        ex, stub, chunks, model, bm25, reranker, rcfg, "qwen", "http://x")

    assert stub.captured["subset"] == [ex]
    assert stub.captured["args"] == (chunks, model, bm25, reranker)
    assert stub.captured["kwargs"] == {
        "top_k": rcfg["top_k"], "candidate_k": rcfg["candidate_k"],
        "dense_k": rcfg["dense_k"], "bm25_k": rcfg["bm25_k"],
        "rrf_k": rcfg["rrf_k"],
        "ollama_model": "qwen", "base_url": "http://x",
    }
    assert rec is record


# ---------------------------------------------------------------------------
# D. Agentic: receives [ex]; top_k / model / base_url / max_steps forwarded
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_run_unans_agentic_receives_list_and_forwards_params():
    ex = _make_ex()
    chunks, model = object(), object()
    rcfg = _rcfg()
    record = {"status": "ok", "answer": "x"}
    stub = _stub_run(record, "run_agentic")

    rec = evaluate_rag._run_unans(
        ex, stub, chunks, model, None, None, rcfg, "qwen", "http://x")

    assert stub.captured["subset"] == [ex]
    assert stub.captured["args"] == (chunks, model)
    assert stub.captured["kwargs"] == {
        "top_k": rcfg["top_k"], "ollama_model": "qwen",
        "base_url": "http://x", "max_steps": rcfg["max_steps"],
    }
    assert rec is record


# ---------------------------------------------------------------------------
# E. Unsupported run function raises ValueError (no silent None return)
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_run_unans_unsupported_run_function_raises_valueerror():
    ex = _make_ex()

    def unknown(subset, *args, **kwargs):
        return [{"status": "ok"}]

    unknown.__name__ = "run_totally_unknown"

    with pytest.raises(ValueError, match="Unsupported unanswerable run function"):
        evaluate_rag._run_unans(
            ex, unknown, object(), object(), None, None, _rcfg(), "m", "u")


@pytest.mark.smoke
def test_run_unans_empty_record_list_raises_valueerror():
    ex = _make_ex()

    def empty(subset, *args, **kwargs):
        return []  # wrapper unexpectedly returned nothing

    empty.__name__ = "run_naive"

    with pytest.raises(ValueError, match="empty record list"):
        evaluate_rag._run_unans(
            ex, empty, object(), object(), None, None, _rcfg(), "m", "u")


# ---------------------------------------------------------------------------
# F. evaluate_unanswerable() returns a non-None record for each system
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.parametrize(
    "sys_name,wrapper_name",
    [
        ("naive", "run_naive"),
        ("hybrid", "run_hybrid"),
        ("reranker", "run_reranker"),
        ("agentic", "run_agentic"),
    ],
)
def test_evaluate_unanswerable_returns_non_none_record_per_system(sys_name, wrapper_name):
    uq = [{
        "query_id": "unans_001",
        "question": "What is the exact GPT-4 parameter count?",
    }]
    record = {"status": "ok", "answer": "I cannot answer this question."}
    stub = _stub_run(record, wrapper_name)

    # judge_abstention and the httpx client are patched out so no network is
    # touched; the point is that evaluate_unanswerable never sees rec is None.
    with patch.object(evaluate_rag, "judge_abstention", return_value=True), \
         patch("httpx.Client"):
        recs = evaluate_rag.evaluate_unanswerable(
            uq, stub, sys_name, object(), object(), None, None,
            _rcfg(), "qwen2.5:3b", "http://x:11434")

    assert len(recs) == 1
    rec = recs[0]
    assert rec is not None
    assert rec["status"] == "ok"
    assert rec["abstained"] is True
    assert rec["hallucinated"] is False
    # The stub wrapper received [ex] (proves the list-wrap fix end to end).
    assert stub.captured["subset"][0].question == uq[0]["question"]



# ---------------------------------------------------------------------------
# G. --unanswerable-only: skips answerable loop + generation judge, writes
#    only *_unanswerable.jsonl, never touches *_results.jsonl
# ---------------------------------------------------------------------------

def _examples(n: int = 2) -> list:
    return [
        SimpleNamespace(
            query_id=f"q{i}", question=f"question number {i}",
            gold_doc_id=f"d{i}", gold_section_id=i,
            reference_answer=f"answer {i}", query_type="text", source="text",
        )
        for i in range(n)
    ]


@pytest.mark.smoke
def test_unanswerable_only_skips_main_loop_and_generation_and_writes_unanswerable():
    examples = _examples(2)
    fake_chunk = SimpleNamespace(chunk_id=0)
    fake_cache = Path("/fake/corpus_test.pkl")
    captured: dict = {}

    def fake_save_jsonl(records, path):
        captured[path.name] = records

    with patch.object(evaluate_rag, "load_examples", return_value=examples), \
         patch.object(evaluate_rag, "load_embedding_model",
                      return_value=MagicMock()) as m_model, \
         patch.object(evaluate_rag, "get_all_doc_ids", return_value=["d0"]), \
         patch.object(evaluate_rag, "select_corpus_doc_ids",
                      return_value=["d0"]), \
         patch.object(evaluate_rag, "get_embedded_corpus",
                      return_value=([fake_chunk], fake_cache)) as m_corpus, \
         patch.object(evaluate_rag, "BM25Index"), \
         patch.object(evaluate_rag, "Reranker"), \
         patch.object(evaluate_rag, "_warmup"), \
         patch.object(evaluate_rag, "run_naive") as m_run_naive, \
         patch.object(evaluate_rag, "run_hybrid") as m_run_hybrid, \
         patch.object(evaluate_rag, "run_reranker") as m_run_reranker, \
         patch.object(evaluate_rag, "run_agentic") as m_run_agentic, \
         patch.object(evaluate_rag, "compute_generation_metrics") as m_gen, \
         patch.object(evaluate_rag, "evaluate_unanswerable",
                      return_value=[{"status": "ok", "abstained": True}]) as m_unans, \
         patch.object(evaluate_rag, "save_jsonl", side_effect=fake_save_jsonl), \
         patch.object(evaluate_rag, "summarize"), \
         patch.object(evaluate_rag, "print_summary"), \
         patch.object(evaluate_rag, "print_comparison"):
        evaluate_rag.main(
            ["--system", "all", "--corpus-docs", "0", "--unanswerable-only"])

    # The embedding model + cached corpus are still loaded normally.
    assert m_model.called
    assert m_corpus.called

    # The normal answerable query loop must NOT run for any system.
    m_run_naive.assert_not_called()
    m_run_hybrid.assert_not_called()
    m_run_reranker.assert_not_called()
    m_run_agentic.assert_not_called()

    # The generation LLM-as-a-Judge must NOT run.
    m_gen.assert_not_called()

    # The unanswerable stage ran once per system (4 systems for "all").
    assert m_unans.call_count == 4

    # The four main *_results.jsonl files must never be passed to save_jsonl.
    for main_fname in ("naive_results.jsonl", "hybrid_results.jsonl",
                       "hybrid_reranker_results.jsonl", "agentic_results.jsonl"):
        assert main_fname not in captured, (
            f"{main_fname} must not be written in --unanswerable-only mode")

    # Only the four *_unanswerable.jsonl files are written.
    expected = {"naive_unanswerable.jsonl", "hybrid_unanswerable.jsonl",
                "hybrid_reranker_unanswerable.jsonl", "agentic_unanswerable.jsonl"}
    assert set(captured.keys()) == expected, (
        f"--unanswerable-only wrote unexpected files: {set(captured.keys())}")
    for fname in expected:
        assert isinstance(captured[fname], list)
        assert len(captured[fname]) == 1


@pytest.mark.smoke
def test_unanswerable_only_absent_runs_normal_loop():
    """When --unanswerable-only is absent, normal CLI behavior is unchanged:
    the answerable run_* wrappers ARE called (so we are not accidentally
    short-circuiting the benchmark)."""
    examples = _examples(1)
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
        evaluate_rag.main(["--system", "all", "--limit", "1"])

    # Normal mode writes the four main result files (proves the loop ran).
    for fname in ("naive_results.jsonl", "hybrid_results.jsonl",
                  "hybrid_reranker_results.jsonl", "agentic_results.jsonl"):
        assert fname in captured

