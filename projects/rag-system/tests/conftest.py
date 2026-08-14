"""
Shared fixtures for the rag-system test suite.

Provides a deterministic hashing embedder and a scripted generator so the
pipeline can be exercised end-to-end without a network connection, without
downloading sentence-transformers weights, and without a running Ollama
server. Both are injected via existing constructor parameters
(embed_chunks/retrieve already take a `model`, RagPipeline takes
`generate_fn`) -- nothing here monkeypatches internals of the src modules.
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
from src.generator import GenerationResult
from src.ingest import DocumentPage

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CORPUS_DIR = FIXTURES_DIR / "corpus"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common function words excluded from vectorization -- without this, short
# queries like "How does a panel produce electricity?" are dominated by
# "how"/"does"/"a" bucket collisions instead of the actual topic words.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "how", "what", "why", "when", "where", "who", "which", "does", "do",
    "did", "to", "of", "in", "on", "at", "by", "for", "with", "as", "and",
    "or", "but", "not", "no", "so", "if", "than", "that", "this", "it",
    "its", "their", "they", "you", "your", "we", "our", "i", "he", "she",
    "his", "her", "from", "into", "can", "will", "would", "should",
}


class HashingEmbedder:
    """
    Deterministic bag-of-words hashing "embedder" for tests.

    Not a real semantic embedding model: it hashes each word into a fixed
    bucket of a fixed-size vector, so documents that share vocabulary end
    up with high cosine similarity. Exposes the same `encode(...)` surface
    as sentence_transformers.SentenceTransformer, which is what
    embed_chunks() and retrieve() call.
    """

    def __init__(self, dims: int = 256):
        self.dims = dims

    def _vectorize(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dims, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            if token in _STOPWORDS:
                continue
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dims
            vector[bucket] += 1.0
        return vector

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        single_input = isinstance(texts, str)
        inputs = [texts] if single_input else list(texts)

        vectors = np.stack([self._vectorize(t) for t in inputs])

        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

        return vectors[0] if single_input else vectors


@pytest.fixture
def fake_embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def corpus_pages() -> list[DocumentPage]:
    """Load the synthetic fixture corpus as DocumentPage objects (page=1 each)."""
    pages = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        pages.append(DocumentPage(text=path.read_text(encoding="utf-8"), source=path.name, page=1))
    return pages


def fake_generate_answer(query, chunks, **_kwargs) -> GenerationResult:
    """Scripted stand-in for generator.generate_answer -- no LLM call."""
    if not chunks:
        return GenerationResult(
            answer="No relevant documents were retrieved to answer this question.",
            sources=[],
        )

    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            sources.append({"source": chunk.source, "page": chunk.page, "score": chunk.score})

    answer = f"Fake answer for '{query}', grounded in {len(chunks)} chunk(s) from {sources[0]['source']}."
    return GenerationResult(answer=answer, sources=sources, input_tokens=0, output_tokens=0)


@pytest.fixture
def fake_generator():
    return fake_generate_answer
