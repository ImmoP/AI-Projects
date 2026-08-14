"""
Hybrid Retrieval — dense vector retrieval + BM25 lexical retrieval fused with
Reciprocal Rank Fusion (RRF).

Standard RRF formula:

    RRF_score(d) = sum over retrieval systems i of 1 / (k + rank_i(d))

A conventional constant ``k = 60`` is used (not tuned on the eval set).

The dense retrieval reuses the existing ``src.retriever.retrieve`` and BM25
uses ``benchmark.bm25`` over the same chunks, so no retrieval logic is
duplicated.
"""

from __future__ import annotations

from src.retriever import RetrievalResult, retrieve

from benchmark.bm25 import BM25Index


def _rrf_fuse(
    ranked_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> dict[int, float]:
    """Compute RRF scores keyed by chunk_id from several ranked lists."""
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + 1.0 / (
                k + rank
            )
    return fused


def hybrid_retrieve(
    query: str,
    embedded_chunks: list,
    model,
    bm25_index: BM25Index,
    top_k: int = 5,
    dense_k: int = 50,
    bm25_k: int = 50,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """
    Run dense + BM25 retrieval in parallel, fuse with RRF, return top_k.

    Parameters
    ----------
    query : str
        The search query.
    embedded_chunks : list
        The embedded vector corpus (same chunks BM25 indexes).
    model : SentenceTransformer
        Embedding model for dense retrieval.
    bm25_index : BM25Index
        Pre-built BM25 index over the same chunks.
    top_k : int
        Number of final fused results to return.
    dense_k : int
        Number of dense candidates to consider.
    bm25_k : int
        Number of BM25 candidates to consider.
    rrf_k : int
        RRF fusion constant (conventional default 60).
    """
    if not embedded_chunks:
        return []

    dense_results = retrieve(
        query=query,
        embedded_chunks=embedded_chunks,
        model=model,
        top_k=dense_k,
    )
    bm25_results = bm25_index.search(query, top_k=bm25_k)

    fused = _rrf_fuse([dense_results, bm25_results], k=rrf_k)

    # Reconstruct ordered results from the fused scores.
    ordered_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]
    by_id = {r.chunk_id: r for r in dense_results}
    by_id.update({r.chunk_id: r for r in bm25_results})

    results = []
    for chunk_id in ordered_ids:
        r = by_id[chunk_id]
        # Overwrite the score with the RRF score.
        results.append(
            RetrievalResult(
                text=r.text,
                source=r.source,
                page=r.page,
                chunk_id=r.chunk_id,
                score=fused[chunk_id],
            )
        )
    return results