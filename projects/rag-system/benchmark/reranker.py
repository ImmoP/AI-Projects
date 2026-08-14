"""
Reranker module for the "Hybrid + Reranker" RAG variant.

After hybrid retrieval produces a candidate set, a cross-encoder scores each
(query, chunk) pair and the top-k are kept for generation.  The reranker only
operates on the initial candidate set (never on the whole corpus).

Uses ``sentence_transformers.CrossEncoder``.  The default model (``cross-encoder/ms-marco-MiniLM-L-6-v2``) is a small English
cross-encoder.
"""

from __future__ import annotations

from src.retriever import RetrievalResult

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Thin wrapper around a sentence-transformers cross-encoder."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Score candidates with the cross-encoder and return the top_k.

        Candidates retain their original metadata; only the ``score`` field is
        replaced with the cross-encoder relevance score.
        """
        if not candidates:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores and re-rank.
        scored = []
        for c, s in zip(candidates, scores):
            scored.append(
                RetrievalResult(
                    text=c.text,
                    source=c.source,
                    page=c.page,
                    chunk_id=c.chunk_id,
                    score=float(s),
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]