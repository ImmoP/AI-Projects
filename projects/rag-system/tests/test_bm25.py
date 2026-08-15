"""
Tests for benchmark.bm25 -- the sparse inverted-index Okapi BM25 implementation.

Covers (per the task spec):

  A. Ranking parity: the sparse BM25 scores are *exactly equal* to a
     mathematically equivalent dense reference implementation (the old,
     non-scalable representation) on a small deterministic fixture, and
     relevant documents rank correctly.
  B. Query terms absent from the corpus contribute nothing.
  C. Repeated query terms preserve the existing ``set(tokenize(query))``
     behaviour (duplicates are not double-counted).
  D. ``--prepare-corpus`` exits before BM25 construction
     (``BM25Index.build`` is never called).
  E. Existing RAG tests -- exercised by the rest of the suite
     (test_retrieval.py, test_chunking.py, test_pipeline_e2e.py).

The dense reference in ``_dense_bm25_scores`` is implemented here purely as a
test oracle; it is the representation that does NOT scale, deliberately kept
out of the production code.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmark.bm25 import BM25Index, tokenize
from src.chunking import TextChunk
from src.retriever import RetrievalResult


def _chunks() -> list[TextChunk]:
    """A tiny, hand-crafted corpus with known term frequencies."""
    return [
        TextChunk(text="cat cat cat dog", source="animals.md", page=1, chunk_id=0),
        TextChunk(text="dog dog bird", source="animals.md", page=2, chunk_id=1),
        TextChunk(text="fish bird fish", source="wildlife.md", page=1, chunk_id=2),
        TextChunk(text="the quick brown fox", source="story.md", page=1, chunk_id=3),
    ]


def _dense_bm25_scores(chunks, query, k1=1.5, b=0.75):
    """Reference dense Okapi BM25 -- a dense term x doc tf vector per term.

    This mirrors the *old* representation the index used before the sparse
    fix: ``term -> [tf]*n_docs`` (length-N list, mostly zeros).  It is
    reimplemented here independently so we can prove the sparse index is
    mathematically equivalent -- the float operations and their order are
    identical, so we assert exact equality.
    """
    docs = [tokenize(c.text) for c in chunks]
    n = len(docs)
    doc_lengths = [len(d) for d in docs]
    avgdl = sum(doc_lengths) / n if n else 0.0

    # term -> dense per-chunk tf list (length n).  This is the representation
    # that does NOT scale; it is here only as an oracle.
    term_lists: dict[str, list[int]] = {}
    for doc_idx, terms in enumerate(docs):
        local: dict[str, int] = {}
        for term in terms:
            local[term] = local.get(term, 0) + 1
        for term, tf in local.items():
            term_lists.setdefault(term, [0] * n)[doc_idx] = tf

    scores = [0.0] * n
    for term in set(tokenize(query)):
        if term not in term_lists:
            continue
        tf_list = term_lists[term]
        n_t = sum(1 for v in tf_list if v > 0)
        idf = math.log((n - n_t + 0.5) / (n_t + 0.5) + 1.0)
        if idf <= 0.0:
            continue
        for i in range(n):
            tf = tf_list[i]
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * doc_lengths[i] / avgdl)
            scores[i] += idf * (tf * (k1 + 1)) / denom
    return scores


# ---------------------------------------------------------------------------
# A. Ranking parity (sparse == dense) + relevant docs rank first
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.parametrize(
    "query",
    [
        "cat",
        "dog",
        "bird",
        "fish",
        "fox quick",          # two terms, same chunk
        "cat dog bird fish",  # multi-term, multi-doc
    ],
)
def test_bm25_sparse_matches_dense_reference(query):
    chunks = _chunks()
    index = BM25Index.build(chunks)
    sparse = index.score(query)
    dense = _dense_bm25_scores(chunks, query)

    assert len(sparse) == len(dense) == len(chunks)
    # Exact equality: identical float operations performed in the same order.
    assert sparse == dense


@pytest.mark.smoke
def test_bm25_relevant_document_ranks_first():
    index = BM25Index.build(_chunks())

    # "cat" only appears in chunk 0 (tf=3) -> it must rank first.
    top = index.search("cat", top_k=1)
    assert top[0].chunk_id == 0

    # "dog" appears in chunk 0 (tf=1) and chunk 1 (tf=2); chunk 1 has higher
    # tf and is shorter, so it should rank above chunk 0.
    ranked = index.search("dog", top_k=2)
    assert ranked[0].chunk_id == 1
    assert ranked[1].chunk_id == 0


@pytest.mark.smoke
def test_bm25_non_matching_chunks_score_zero():
    index = BM25Index.build(_chunks())
    scores = index.score("cat")
    assert scores[0] > 0.0                       # only chunk 0 has "cat"
    assert all(s == 0.0 for i, s in enumerate(scores) if i != 0)


@pytest.mark.smoke
def test_bm25_score_list_length_matches_corpus():
    chunks = _chunks()
    index = BM25Index.build(chunks)
    # Every query returns one score per chunk (fusion code relies on this).
    assert len(index.score("cat dog bird fish fox")) == len(chunks)
    assert len(index.score("zzzznonexistent")) == len(chunks)
    assert len(index.score("")) == len(chunks)


# ---------------------------------------------------------------------------
# B. Terms absent from the corpus produce no contribution
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_absent_query_term_contributes_nothing():
    index = BM25Index.build(_chunks())

    # "zebra" is not in the corpus at all -> all-zero scores.
    assert index.score("zebra") == [0.0] * len(_chunks())

    # Mixing one present term and one absent term must equal the score of
    # just the present term (the absent term adds nothing).
    only_cat = index.score("cat")
    cat_and_zebra = index.score("cat zebra")
    assert cat_and_zebra == only_cat


# ---------------------------------------------------------------------------
# C. Repeated query terms preserve set(tokenize(query)) behaviour
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_repeated_query_terms_are_not_double_counted():
    index = BM25Index.build(_chunks())
    once = index.score("cat")
    repeated = index.score("cat cat cat")
    assert repeated == once


# ---------------------------------------------------------------------------
# Metadata + RetrievalResult preservation
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_search_preserves_metadata_and_result_type():
    chunks = _chunks()
    index = BM25Index.build(chunks)
    results = index.search("dog", top_k=2)

    assert all(isinstance(r, RetrievalResult) for r in results)

    by_id = {c.chunk_id: c for c in chunks}
    for r in results:
        src = by_id[r.chunk_id]
        assert r.source == src.source
        assert r.page == src.page
        assert r.text == src.text

    # Sorted by score descending.
    assert [r.score for r in results] == sorted(
        (r.score for r in results), reverse=True
    )


# ---------------------------------------------------------------------------
# Sparse data-structure verification (documents the fix)
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_term_stats_uses_sparse_postings():
    index = BM25Index.build(_chunks())

    # "cat" occurs only in chunk 0 with tf=3.
    n_t, postings = index.term_stats["cat"]
    assert n_t == 1
    assert postings == [(0, 3)]  # only (doc_idx, tf) where term occurs

    # "dog" occurs in chunk 0 (tf=1) and chunk 1 (tf=2).
    n_t, postings = index.term_stats["dog"]
    assert n_t == 2
    assert postings == [(0, 1), (1, 2)]

    # No term stores a dense length-N vector any more.
    for df, plist in index.term_stats.values():
        assert all(tf > 0 for _doc_idx, tf in plist)
        assert len(plist) == df


# ---------------------------------------------------------------------------
# D. --prepare-corpus exits before BM25 construction
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_prepare_corpus_exits_before_bm25_construction(tmp_path):
    """
    With --prepare-corpus, main() must return immediately after the embedded
    corpus is loaded/cached and must NOT, for the default --system "all":

      - build the BM25 index          (BM25Index.build)
      - load the reranker              (Reranker)
      - run warm-up                    (_warmup)
      - run any evaluation/generation   (run_naive, run_hybrid,
                                         run_reranker, run_agentic)

    The default --system is "all" (=> need_bm25 is True AND every run_*
    branch is exercised), so ALL of these WOULD be called if the
    short-circuit were missing -- this makes every negative assertion
    meaningful rather than trivially true.

    Uses --out-dir tmp_path so no artefacts are written to
    config.RESULTS_DIR.
    """
    from benchmark import evaluate_rag

    fake_example = SimpleNamespace(
        query_id="q0", question="ignored", gold_doc_id="d0"
    )
    fake_chunk = SimpleNamespace(chunk_id=0)
    fake_cache = Path("/fake/corpus_test.pkl")

    with patch.object(evaluate_rag, "load_examples",
                      return_value=[fake_example]) as m_examples, \
         patch.object(evaluate_rag, "load_embedding_model",
                      return_value=MagicMock()) as m_model, \
         patch.object(evaluate_rag, "get_all_doc_ids",
                      return_value=["d0"]) as m_all_ids, \
         patch.object(evaluate_rag, "select_corpus_doc_ids",
                      return_value=["d0"]) as m_select, \
         patch.object(evaluate_rag, "get_embedded_corpus",
                      return_value=([fake_chunk], fake_cache)) as m_corpus, \
         patch.object(evaluate_rag, "BM25Index") as m_bm25, \
         patch.object(evaluate_rag, "Reranker") as m_reranker, \
         patch.object(evaluate_rag, "_warmup") as m_warmup, \
         patch.object(evaluate_rag, "run_naive") as m_run_naive, \
         patch.object(evaluate_rag, "run_hybrid") as m_run_hybrid, \
         patch.object(evaluate_rag, "run_reranker") as m_run_reranker, \
         patch.object(evaluate_rag, "run_agentic") as m_run_agentic:
        ret = evaluate_rag.main(
            ["--prepare-corpus", "--out-dir", str(tmp_path)])

    # Corpus loading happened -- proves execution reached the short-circuit.
    assert m_corpus.called
    assert m_model.called

    # --prepare-corpus must short-circuit before ALL of the following.
    m_bm25.build.assert_not_called()        # no BM25 index built
    m_reranker.assert_not_called()          # no reranker model loaded
    m_warmup.assert_not_called()            # no warm-up
    m_run_naive.assert_not_called()         # no naive generation/eval
    m_run_hybrid.assert_not_called()        # no hybrid generation/eval
    m_run_reranker.assert_not_called()      # no reranker generation/eval
    m_run_agentic.assert_not_called()       # no agentic generation/eval

    assert ret is None


