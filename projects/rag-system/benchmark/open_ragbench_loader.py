"""
Open RAGBench loader.

Reads the locally downloaded Open RAGBench ``pdf/arxiv`` split, filters it to
text-only questions, and builds the embedded corpus that is shared by *both*
the Classical and the Agentic RAG pipelines.

The corpus is built from the benchmark's JSON ``corpus/`` files.  Each paper
is composed of ``sections``; each section is turned into a ``DocumentPage``
with::

    source = paper id (e.g. "2401.01872v2")
    page   = section_id

so that a retrieved chunk's ``(source, page)`` can be matched against the
benchmark's ``gold_doc_id`` + ``gold_section_id``.  This gives a real
section-level mapping (it is not a fabricated approximation).

The embedded corpus is cached to disk (pickle) keyed by the exact set of
corpus documents plus the chunking/embedding settings, so repeated runs are
fast and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer
from src.chunking import chunk_pages
from src.embeddings import embed_chunks
from src.ingest import DocumentPage

from benchmark.config import (
    ANSWERS_PATH,
    CACHE_DIR,
    CORPUS_DIR,
    QRELS_PATH,
    QUERIES_PATH,
)


@dataclass
class RagbenchExample:
    """One normalised benchmark example (evaluation-only data)."""

    query_id: str
    question: str
    reference_answer: str
    gold_doc_id: str
    gold_section_id: int
    query_type: str
    source: str


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_queries(path: Path = QUERIES_PATH) -> dict:
    """Return ``{query_id: {"query", "type", "source"}}``."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_qrels(path: Path = QRELS_PATH) -> dict:
    """Return ``{query_id: {"doc_id", "section_id"}}``."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_answers(path: Path = ANSWERS_PATH) -> dict:
    """Return ``{query_id: reference_answer}``."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_examples(
    queries_path: Path = QUERIES_PATH,
    qrels_path: Path = QRELS_PATH,
    answers_path: Path = ANSWERS_PATH,
    source_filter: str = "text",
) -> list[RagbenchExample]:
    """
    Load and normalise the benchmark, keeping only questions whose
    ``source`` equals ``source_filter`` (default ``"text"``).

    Rows where a required field is missing are skipped (defensively) so a
    single malformed entry does not break the whole loader.
    """
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    answers = load_answers(answers_path)

    examples: list[RagbenchExample] = []

    for query_id, q in queries.items():
        if q.get("source") != source_filter:
            continue

        rel = qrels.get(query_id)
        if rel is None or "doc_id" not in rel or "section_id" not in rel:
            continue

        answer = answers.get(query_id, "")
        if not answer.strip():
            # A question without a reference answer cannot be answer-evaluated.
            continue

        examples.append(
            RagbenchExample(
                query_id=query_id,
                question=q.get("query", "").strip(),
                reference_answer=answer,
                gold_doc_id=str(rel["doc_id"]),
                gold_section_id=int(rel["section_id"]),
                query_type=q.get("type", ""),
                source=q.get("source", source_filter),
            )
        )

    return examples


# ---------------------------------------------------------------------------
# Corpus construction (shared by both pipelines)
# ---------------------------------------------------------------------------

def get_all_doc_ids(corpus_dir: Path = CORPUS_DIR) -> list[str]:
    """Return the sorted list of all paper ids in the corpus."""
    return [p.stem for p in sorted(corpus_dir.glob("*.json"))]


def select_corpus_doc_ids(
    examples: list[RagbenchExample],
    all_doc_ids: list[str],
    corpus_docs: int = 100,
    seed: int = 42,
) -> list[str]:
    """
    Deterministically choose which corpus documents to index.

    The gold documents referenced by ``examples`` are always included.  If
    ``corpus_docs`` is greater than the number of gold documents, additional
    distractor documents (hard negatives) are sampled with ``seed`` to reach
    the budget.  If ``corpus_docs`` is ``0`` (see ``FULL_CORPUS``) or larger
    than the whole corpus, every document is used.

    Returns a sorted list of doc ids.  The same selection is used for both
    RAG systems, guaranteeing a fair comparison.
    """
    gold_docs = {ex.gold_doc_id for ex in examples}
    required = set(gold_docs)

    if corpus_docs == 0 or corpus_docs >= len(all_doc_ids):
        return sorted(all_doc_ids)

    distractors = [d for d in all_doc_ids if d not in required]
    rng = random.Random(seed)  # deterministic shuffle
    rng.shuffle(distractors)

    take = max(corpus_docs - len(required), 0)
    selected = required | set(distractors[:take])
    return sorted(selected)


def _doc_to_pages(doc: dict, doc_id: str) -> list[DocumentPage]:
    """Turn one corpus JSON paper into one DocumentPage per section."""
    pages = []
    for section in doc.get("sections", []):
        text = (section.get("text") or "").strip()
        if not text:
            continue
        pages.append(
            DocumentPage(
                text=text,
                source=doc_id,
                page=int(section.get("section_id", 0)),
            )
        )
    return pages


def build_embedded_corpus(
    doc_ids: list[str],
    model: SentenceTransformer,
    corpus_dir: Path = CORPUS_DIR,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list:
    """
    Build the embedded corpus for ``doc_ids`` using the existing chunking and
    embedding modules (no re-implementation).

    Returns ``list[EmbeddedChunk]`` with globally-unique ``chunk_id`` values.
    """
    pages: list[DocumentPage] = []
    for doc_id in doc_ids:
        path = corpus_dir / f"{doc_id}.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        pages.extend(_doc_to_pages(doc, doc_id))

    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    return embed_chunks(chunks, model)


def _corpus_cache_key(
    doc_ids: list[str],
    chunk_size: int,
    overlap: int,
    embedding_model_name: str,
) -> str:
    """Stable hash of everything that determines the embedded corpus."""
    payload = "|".join(
        [
            json.dumps(sorted(doc_ids), separators=(",", ":")),
            str(chunk_size),
            str(overlap),
            embedding_model_name,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def get_embedded_corpus(
    doc_ids: list[str],
    model: SentenceTransformer,
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    chunk_size: int = 500,
    overlap: int = 100,
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
) -> tuple[list, Path]:
    """
    Return the embedded corpus for ``doc_ids``, building and caching it if
    necessary.  Returns ``(embedded_chunks, cache_path_used)``.
    """
    if not doc_ids:
        return [], Path("")

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _corpus_cache_key(doc_ids, chunk_size, overlap, embedding_model_name)
    cache_path = cache_dir / f"corpus_{key}.pkl"

    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as fh:
            embedded = pickle.load(fh)
        print(f"[corpus] loaded {len(embedded)} chunks from cache: {cache_path.name}")
        return embedded, cache_path

    print(f"[corpus] building embeddings for {len(doc_ids)} documents ...")
    embedded = build_embedded_corpus(
        doc_ids, model, chunk_size=chunk_size, overlap=overlap
    )
    with open(cache_path, "wb") as fh:
        pickle.dump(embedded, fh)
    print(f"[corpus] cached {len(embedded)} chunks to {cache_path.name}")
    return embedded, cache_path
