"""
Okapi BM25 lexical retrieval (self-contained, no external dependency).

Used by the Hybrid RAG variant.  BM25 runs *in parallel* with the dense
vector retriever and the two rankings are fused with Reciprocal Rank Fusion
(see ``hybrid.py``).

The index is built over the *exact same chunks* as the vector index, so both
ranking signals come from identical searchable units.

Formula (standard Okapi BM25):

    score(q, d) = sum over query terms t of
        IDF(t) * ( f(t,d) * (k1 + 1) )
        / ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )

with

    IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )

where N is the number of chunks, n(t) the number of chunks containing t,
f(t,d) the term frequency, |d| the chunk length and avgdl the mean length.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from src.retriever import RetrievalResult

# Tokenizer: lowercase and keep word characters (letters/digits/underscore).
_TOKEN_RE = re.compile(r"[\w]+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization used for both indexing and querying."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """
    A small Okapi BM25 index over a list of chunks.

    Construction is O(total words).  ``search`` scores every chunk against a
    query and returns the top-k as ``RetrievalResult`` objects (the score is
    the BM25 score).  Chunk metadata (source, page, chunk_id, text) is
    preserved so the results can be fused with dense retrieval.
    """

    chunk_ids: list[int]
    sources: list[str]
    pages: list[int]
    texts: list[str]
    doc_lengths: list[int]
    avgdl: float
    # term -> (number of chunks containing it, per-chunk term frequencies)
    term_stats: dict[str, tuple[int, list[int]]]
    k1: float
    b: float

    @classmethod
    def build(cls, chunks, k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        """Build an index from chunk-like objects (need .text and metadata)."""
        doc_terms: list[list[str]] = []
        doc_lengths: list[int] = []

        for chunk in chunks:
            terms = tokenize(chunk.text)
            doc_terms.append(terms)
            doc_lengths.append(len(terms))

        avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
        n_docs = len(doc_terms)

        # term -> per-chunk term frequencies (computed once, O(total words)).
        term_lists: dict[str, list[int]] = {}
        for doc_idx, terms in enumerate(doc_terms):
            local: dict[str, int] = {}
            for term in terms:
                local[term] = local.get(term, 0) + 1
            for term, tf in local.items():
                term_lists.setdefault(term, [0] * n_docs)[doc_idx] = tf

        term_stats = {
            term: (sum(1 for v in lst if v > 0), lst)
            for term, lst in term_lists.items()
        }

        return cls(
            chunk_ids=[c.chunk_id for c in chunks],
            sources=[c.source for c in chunks],
            pages=[c.page for c in chunks],
            texts=[c.text for c in chunks],
            doc_lengths=doc_lengths,
            avgdl=avgdl,
            term_stats=term_stats,
            k1=k1,
            b=b,
        )

    def _idf(self, term: str) -> float:
        n_t = self.term_stats.get(term, (0, []))[0]
        n = len(self.chunk_ids)
        return math.log((n - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def score(self, query: str) -> list[float]:
        """Return an unnormalised BM25 score for every chunk."""
        n = len(self.chunk_ids)
        scores = [0.0] * n
        query_terms = set(tokenize(query))
        if not query_terms:
            return scores

        for term in query_terms:
            stats = self.term_stats.get(term)
            if stats is None:
                continue
            n_t, tf_list = stats
            idf = self._idf(term)
            if idf <= 0.0:
                continue
            for i in range(n):
                tf = tf_list[i]
                if tf == 0:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[i] / self.avgdl
                )
                scores[i] += idf * (tf * (self.k1 + 1)) / denom

        return scores

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return the top-k chunks by BM25 score as RetrievalResult objects."""
        scores = self.score(query)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for i in ranked:
            results.append(
                RetrievalResult(
                    text=self.texts[i],
                    source=self.sources[i],
                    page=self.pages[i],
                    chunk_id=self.chunk_ids[i],
                    score=scores[i],
                )
            )
        return results
