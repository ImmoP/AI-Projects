"""
Evaluators for all four RAG variants.

Variants:
  A  naive    - dense retrieval only
  B  hybrid   - dense + BM25 fused with RRF
  C  reranker - hybrid retrieval + cross-encoder reranking
  D  agentic  - ReAct loop with dense retrieval

All reuse the existing src.retriever, src.generator and src.agent modules.
"""
from __future__ import annotations

import time

from src.agent.react_agent import run_agent
from src.generator import generate_answer
from src.retriever import retrieve

from benchmark.metrics import compute_metrics


def _base_record(example):
    return {
        "query_id": example.query_id,
        "question": example.question,
        "reference_answer": example.reference_answer,
        "gold_doc_id": example.gold_doc_id,
        "gold_section_id": example.gold_section_id,
        "query_type": example.query_type,
        "source": example.source,
    }


def _attach_metrics(record, retrieved, gold_doc_id, gold_section_id):
    m_doc = compute_metrics(retrieved, gold_doc_id, level="document")
    m_sec = compute_metrics(retrieved, gold_doc_id, gold_section_id, level="section")
    for m, suffix in ((m_doc, "_doc"), (m_sec, "_sec")):
        for k in ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
                  "reciprocal_rank", "first_hit_rank"):
            record[f"{k}{suffix}"] = getattr(m, k)
    record["retrieved_chunk_ids"] = [r.chunk_id for r in retrieved]
    record["retrieved_sources"] = [r.source for r in retrieved]
    return record


def _error_record(example, error):
    rec = _base_record(example)
    rec.update({"status": "error", "error": str(error)[:500], "answer": "",
                 "retrieval_calls": 0, "latency_seconds": None})
    return rec


def _generation_tokens(gen):
    """Safely extract token counts from a GenerationResult or AgentState."""
    return getattr(gen, "input_tokens", None), getattr(gen, "output_tokens", None)


# ---------------------------------------------------------------------------
# A. Naive RAG (dense retrieval only)
# ---------------------------------------------------------------------------

def evaluate_naive(example, embedded_chunks, model, top_k=5, ollama_model="qwen2.5:3b", base_url="http://localhost:11434"):
    """Dense-vector retrieval + generation."""
    start = time.perf_counter()
    try:
        retrieved = retrieve(query=example.question, embedded_chunks=embedded_chunks, model=model, top_k=top_k)
        generation = generate_answer(query=example.question, chunks=retrieved, model=ollama_model, base_url=base_url)
    except Exception as exc:
        rec = _error_record(example, exc)
        rec["latency_seconds"] = time.perf_counter() - start
        return rec
    rec = _base_record(example)
    return _finalise_record(rec, retrieved, example, generation, start)


# ---------------------------------------------------------------------------
# B. Hybrid RAG (dense + BM25 + RRF)
# ---------------------------------------------------------------------------

def evaluate_hybrid(example, embedded_chunks, model, bm25_index, top_k=5, dense_k=50, bm25_k=50, rrf_k=60, ollama_model="qwen2.5:3b", base_url="http://localhost:11434"):
    """Hybrid dense+BM25 retrieval with RRF fusion + generation."""
    from benchmark.hybrid import hybrid_retrieve
    start = time.perf_counter()
    try:
        retrieved = hybrid_retrieve(example.question, embedded_chunks, model, bm25_index, top_k=top_k, dense_k=dense_k, bm25_k=bm25_k, rrf_k=rrf_k)
        generation = generate_answer(query=example.question, chunks=retrieved, model=ollama_model, base_url=base_url)
    except Exception as exc:
        rec = _error_record(example, exc)
        rec["latency_seconds"] = time.perf_counter() - start
        return rec
    rec = _base_record(example)
    return _finalise_record(rec, retrieved, example, generation, start)


# ---------------------------------------------------------------------------
# C. Hybrid + Reranker
# ---------------------------------------------------------------------------

def evaluate_reranker(example, embedded_chunks, model, bm25_index, reranker, top_k=5, candidate_k=50, dense_k=50, bm25_k=50, rrf_k=60, ollama_model="qwen2.5:3b", base_url="http://localhost:11434"):
    """Hybrid retrieval -> cross-encoder rerank -> generation."""
    from benchmark.hybrid import hybrid_retrieve
    start = time.perf_counter()
    try:
        candidates = hybrid_retrieve(example.question, embedded_chunks, model, bm25_index, top_k=candidate_k, dense_k=dense_k, bm25_k=bm25_k, rrf_k=rrf_k)
        retrieved = reranker.rerank(example.question, candidates, top_k=top_k)
        generation = generate_answer(query=example.question, chunks=retrieved, model=ollama_model, base_url=base_url)
    except Exception as exc:
        rec = _error_record(example, exc)
        rec["latency_seconds"] = time.perf_counter() - start
        return rec
    rec = _base_record(example)
    rec = _finalise_record(rec, retrieved, example, generation, start)
    rec["candidate_count"] = len(candidates)
    return rec


def _finalise_record(rec, retrieved, example, generation, start_time, retrieval_calls=1):
    """Populate the common per-query record fields after a successful run."""
    _attach_metrics(rec, retrieved, example.gold_doc_id, example.gold_section_id)
    answer = generation.answer if hasattr(generation, "answer") else str(generation or "")
    itok, otok = _generation_tokens(generation)
    rec.update({
        "status": "ok", "answer": answer,
        "retrieval_calls": retrieval_calls,
        "latency_seconds": time.perf_counter() - start_time,
        "input_tokens": itok, "output_tokens": otok,
        "total_tokens": (itok + otok) if (itok is not None and otok is not None) else None,
        "token_usage_complete": (itok is not None and otok is not None),
    })
    return rec


# ---------------------------------------------------------------------------
# D. Agentic RAG (existing ReAct loop)
# ---------------------------------------------------------------------------

def _reconstruct_agent_retrieval(state, embedded_chunks, model, top_k):
    per_call = []
    aggregated = []
    seen_ids = set()
    for query in state.queries:
        results = retrieve(query=query, embedded_chunks=embedded_chunks, model=model, top_k=top_k)
        per_call.append(results)
        for r in results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                aggregated.append(r)
    return per_call, aggregated


def evaluate_agentic(example, embedded_chunks, model, top_k=5, ollama_model="qwen2.5:3b", base_url="http://localhost:11434", max_steps=5):
    start = time.perf_counter()
    try:
        state = run_agent(question=example.question, embedded_chunks=embedded_chunks, model=model, ollama_model=ollama_model, max_steps=max_steps, top_k=top_k, ollama_base_url=base_url)
        per_call, aggregated = _reconstruct_agent_retrieval(state, embedded_chunks, model, top_k)
    except Exception as exc:
        rec = _error_record(example, exc)
        rec["latency_seconds"] = time.perf_counter() - start
        return rec

    rec = _base_record(example)
    _attach_metrics(rec, aggregated, example.gold_doc_id, example.gold_section_id)

    gold_found_call = None
    for idx, results in enumerate(per_call, start=1):
        if any(r.source == example.gold_doc_id for r in results):
            gold_found_call = idx
            break

    hit_step_limit = bool(state.final_answer) and (not state.observations or state.observations[-1].action != "final_answer")

    rec.update({
        "status": "ok",
        "answer": state.final_answer or "",
        "number_of_steps": state.step,
        "retrieval_calls": len(state.queries),
        "queries_generated": list(state.queries),
        "gold_found_at_call": gold_found_call,
        "hit_step_limit": hit_step_limit,
        "termination_reason": state.termination_reason,
        "latency_seconds": time.perf_counter() - start,
        "input_tokens": getattr(state, "input_tokens", 0),
        "output_tokens": getattr(state, "output_tokens", 0),
        "total_tokens": (getattr(state, "input_tokens", 0) + getattr(state, "output_tokens", 0)) or None,
        # Agentic token counts include decision + final-generation calls.
        # Ollama exposes prompt_eval_count/eval_count, so this is complete.
        "token_usage_complete": True,
    })
    return rec


# Backward-compatible aliases
evaluate_classical = evaluate_naive
