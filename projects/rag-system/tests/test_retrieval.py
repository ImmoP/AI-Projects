"""
Smoke tests for src.retriever, using the fixture corpus and the
deterministic HashingEmbedder from conftest.py instead of a real
sentence-transformers model. Checks that retrieval finds the right
document, not that scores are "good" -- no model quality claims here.
"""

import pytest
from src.chunking import chunk_pages
from src.embeddings import embed_chunks
from src.retriever import retrieve


@pytest.fixture
def embedded_corpus(corpus_pages, fake_embedder):
    chunks = chunk_pages(corpus_pages, chunk_size=1000, overlap=0)
    return embed_chunks(chunks, fake_embedder)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "query,expected_source",
    [
        ("How do I grind coffee beans for espresso brewing?", "coffee_brewing.md"),
        ("What boots are best for a mountain ridge hike?", "mountain_hiking.md"),
        ("How does a B-tree index speed up SQL queries?", "database_indexing.md"),
        ("How did Renaissance painters use perspective and pigment?", "renaissance_painting.md"),
        ("How does a photovoltaic panel produce electricity?", "solar_panels.md"),
    ],
)
def test_query_retrieves_expected_document_top1(embedded_corpus, fake_embedder, query, expected_source):
    results = retrieve(query=query, embedded_chunks=embedded_corpus, model=fake_embedder, top_k=1)

    assert len(results) == 1
    assert results[0].source == expected_source


@pytest.mark.smoke
def test_results_are_sorted_by_score_descending(embedded_corpus, fake_embedder):
    results = retrieve(
        query="coffee espresso grind water",
        embedded_chunks=embedded_corpus,
        model=fake_embedder,
        top_k=5,
    )

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.smoke
def test_retrieve_respects_top_k(embedded_corpus, fake_embedder):
    results = retrieve(query="mountain trail hike", embedded_chunks=embedded_corpus, model=fake_embedder, top_k=2)
    assert len(results) == 2


@pytest.mark.smoke
def test_retrieve_with_no_chunks_returns_empty(fake_embedder):
    assert retrieve(query="anything", embedded_chunks=[], model=fake_embedder, top_k=5) == []
