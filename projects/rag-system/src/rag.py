"""
RAG Orchestrator - the main entry point for the Retrieval-Augmented Generation pipeline.

Ties together the full flow:
  PDFs -> Ingest -> Chunk -> Embed -> Retrieve -> Generate

Usage (CLI):
    python -m src.rag "What is multi-head attention?"
    python -m src.rag "Explain the transformer architecture" --model llama3.2 --top-k 5

Usage (Python):
    from src.rag import run_rag_pipeline
    result = run_rag_pipeline("What is attention?")
    print(result.answer)
    print(result.sources)
"""

import argparse
import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.chunking import chunk_pages
from src.embeddings import embed_chunks, load_embedding_model
from src.generator import DEFAULT_MODEL, OLLAMA_BASE_URL, GenerationResult, generate_answer
from src.ingest import load_pdfs
from src.retriever import retrieve

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
CHUNKS_CACHE = CACHE_DIR / "chunks.pkl"
EMBEDDINGS_CACHE = CACHE_DIR / "embeddings.pkl"


@dataclass
class RagPipeline:
    """
    Holds all state for a RAG pipeline run.
    Loads/embeds documents once so multiple queries can reuse them.
    """

    pages: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    embedded_chunks: list = field(default_factory=list)
    model: SentenceTransformer | None = None

    chunk_size: int = 500
    overlap: int = 100
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    ollama_model: str = DEFAULT_MODEL
    ollama_base_url: str = OLLAMA_BASE_URL
    top_k: int = 5
    temperature: float = 0.1
    max_tokens: int = 1024
    use_cache: bool = True

    # Injectable generation step. Defaults to the real Ollama-backed
    # generator; tests substitute a fake to run the pipeline without a
    # live LLM. Must match generate_answer's signature.
    generate_fn: Callable[..., GenerationResult] = generate_answer

    _is_loaded: bool = False

    def load_and_embed(self, pdf_dir: str = "pdfs") -> None:
        """Load PDFs, chunk them, and compute embeddings (with optional caching)."""
        if self.use_cache and self._try_load_from_cache():
            self._is_loaded = True
            return

        print("Loading PDFs...")
        self.pages = load_pdfs(pdf_dir)
        print(f"  Found {len(self.pages)} pages")

        print("Chunking...")
        self.chunks = chunk_pages(self.pages, chunk_size=self.chunk_size, overlap=self.overlap)
        print(f"  Created {len(self.chunks)} chunks")

        print("Loading embedding model...")
        self.model = load_embedding_model(self.embedding_model_name)

        print("Embedding chunks...")
        self.embedded_chunks = embed_chunks(self.chunks, self.model)
        print(f"  Created {len(self.embedded_chunks)} embeddings")

        if self.use_cache:
            self._save_to_cache()

        self._is_loaded = True

    def answer(self, query: str) -> GenerationResult:
        """Run retrieval + generation for a single query."""
        if not self._is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load_and_embed() first.")
        if not query.strip():
            raise ValueError("Query must not be empty")

        retrieved_chunks = retrieve(
            query=query,
            embedded_chunks=self.embedded_chunks,
            model=self.model,
            top_k=self.top_k,
        )
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks")

        generation = self.generate_fn(
            query=query,
            chunks=retrieved_chunks,
            model=self.ollama_model,
            base_url=self.ollama_base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return generation

    def _cache_paths(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CHUNKS_CACHE, EMBEDDINGS_CACHE

    def _try_load_from_cache(self) -> bool:
        chunks_path, emb_path = self._cache_paths()
        if not chunks_path.exists() or not emb_path.exists():
            return False
        try:
            print("Loading from cache...")
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            with open(emb_path, "rb") as f:
                self.embedded_chunks = pickle.load(f)
            print(f"  Loaded {len(self.chunks)} chunks and {len(self.embedded_chunks)} embeddings")
            print("Loading embedding model...")
            self.model = load_embedding_model(self.embedding_model_name)
            return True
        except Exception as exc:
            print(f"  Cache load failed ({exc}), re-processing...")
            return False

    def _save_to_cache(self) -> None:
        chunks_path, emb_path = self._cache_paths()
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(emb_path, "wb") as f:
            pickle.dump(self.embedded_chunks, f)
        print(f"Saved {len(self.chunks)} chunks and {len(self.embedded_chunks)} embeddings to cache")


def run_rag_pipeline(query: str, **kwargs) -> GenerationResult:
    """One-shot convenience function."""
    valid_keys = {k for k in kwargs if hasattr(RagPipeline, k)}
    pipe_kwargs = {k: kwargs[k] for k in valid_keys}
    pipeline = RagPipeline(**pipe_kwargs)
    pdf_dir = kwargs.get("pdf_dir", "pdfs")
    pipeline.load_and_embed(pdf_dir)
    return pipeline.answer(query)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Pipeline: answer questions from your PDF library.")
    parser.add_argument("query", type=str, help="The question to answer")
    parser.add_argument("--pdf-dir", type=str, default="pdfs")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    pipeline = RagPipeline(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        ollama_model=args.model,
        top_k=args.top_k,
        temperature=args.temperature,
        use_cache=not args.no_cache,
    )
    pipeline.load_and_embed(args.pdf_dir)
    result = pipeline.answer(args.query)

    if args.json:
        print(json.dumps({"query": args.query, "answer": result.answer, "sources": result.sources}, indent=2))
    else:
        print()
        print("=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(result.answer)
        print()
        print("=" * 80)
        print("SOURCES:")
        for src in result.sources:
            print(f"  - {src['source']}, p. {src['page']} (score: {src['score']:.3f})")


if __name__ == "__main__":
    main()
