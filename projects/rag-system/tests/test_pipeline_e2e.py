"""
End-to-end smoke test for RagPipeline: does the full
chunk -> embed -> retrieve -> generate flow run and produce a well-formed
result? Uses the fake embedder and fake generator from conftest.py, both
injected through existing/added constructor parameters -- no monkeypatching
of src internals, no network access, no Ollama server required.

This checks that the pipeline runs and its output is well-formed, not that
the answer is a *good* answer.
"""

import pytest
from src.chunking import chunk_pages
from src.embeddings import embed_chunks
from src.generator import GenerationResult
from src.rag import RagPipeline

CORPUS_SOURCES = {
    "coffee_brewing.md",
    "mountain_hiking.md",
    "database_indexing.md",
    "renaissance_painting.md",
    "solar_panels.md",
}


@pytest.fixture
def loaded_pipeline(corpus_pages, fake_embedder, fake_generator):
    chunks = chunk_pages(corpus_pages, chunk_size=1000, overlap=0)
    embedded_chunks = embed_chunks(chunks, fake_embedder)

    pipeline = RagPipeline(
        pages=corpus_pages,
        chunks=chunks,
        embedded_chunks=embedded_chunks,
        model=fake_embedder,
        generate_fn=fake_generator,
        top_k=3,
        use_cache=False,
    )
    pipeline._is_loaded = True
    return pipeline


@pytest.mark.smoke
def test_pipeline_runs_end_to_end(loaded_pipeline):
    result = loaded_pipeline.answer("How do I grind coffee beans for espresso brewing?")

    assert isinstance(result, GenerationResult)
    assert isinstance(result.answer, str)
    assert result.answer.strip()


@pytest.mark.smoke
def test_pipeline_sources_reference_existing_chunks(loaded_pipeline):
    result = loaded_pipeline.answer("How does a photovoltaic panel produce electricity?")

    assert result.sources, "expected at least one source"
    known_chunk_keys = {(c.source, c.page) for c in loaded_pipeline.embedded_chunks}
    for source in result.sources:
        assert {"source", "page", "score"} <= source.keys()
        assert source["source"] in CORPUS_SOURCES
        assert (source["source"], source["page"]) in known_chunk_keys


@pytest.mark.smoke
def test_pipeline_top1_source_matches_query_topic(loaded_pipeline):
    result = loaded_pipeline.answer("How does a B-tree index speed up SQL queries?")
    assert result.sources[0]["source"] == "database_indexing.md"


@pytest.mark.smoke
def test_pipeline_requires_load_before_answer(corpus_pages, fake_embedder, fake_generator):
    pipeline = RagPipeline(model=fake_embedder, generate_fn=fake_generator)
    with pytest.raises(RuntimeError):
        pipeline.answer("anything")


@pytest.mark.smoke
def test_pipeline_rejects_empty_query(loaded_pipeline):
    with pytest.raises(ValueError):
        loaded_pipeline.answer("   ")
