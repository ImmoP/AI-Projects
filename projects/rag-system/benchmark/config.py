"""
Shared configuration for the RAG benchmark.

Centralises paths and default hyper-parameters so that the classical and
agentic pipelines are evaluated with identical settings.  Everything here
can be overridden from the command line (see ``evaluate_rag.py``).
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
# Resolve relative to this file so the benchmark works from any CWD.
BENCHMARK_ROOT = Path(__file__).resolve().parent
OPEN_RAGBENCH_DIR = BENCHMARK_ROOT / "open_ragbench" / "pdf" / "arxiv"
QUERIES_PATH = OPEN_RAGBENCH_DIR / "queries.json"
QRELS_PATH = OPEN_RAGBENCH_DIR / "qrels.json"
ANSWERS_PATH = OPEN_RAGBENCH_DIR / "answers.json"
CORPUS_DIR = OPEN_RAGBENCH_DIR / "corpus"

# Output locations
RESULTS_DIR = BENCHMARK_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"

# --- RAG defaults (shared by both systems) -----------------------------
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 5
DEFAULT_MAX_STEPS = 5
DEFAULT_SEED = 42

# --- Hybrid Search (Variant B) ----------------------------------------
# Conventional RRF fusion constant (not tuned on the eval set).
DEFAULT_RRF_K = 60
# How many candidates to pull from each retriever before fusing.
DEFAULT_CANDIDATE_K = 50
# Okapi BM25 parameters (standard defaults).
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75

# --- Reranker (Variant C) ---------------------------------------------
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_FINAL_K = 5

# --- Unanswerable queries (hallucination test) ------------------------
UNANSWERABLE_QUERIES_PATH = BENCHMARK_ROOT / "unanswerable_queries.json"

# --- Ollama (generation) ----------------------------------------------
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# --- Open RAGBench -----------------------------------------------------
# Only evaluate text-only questions initially (image/table questions need
# multimodal support that the current pipeline does not provide).
TEXT_ONLY_SOURCE = "text"

# Total corpus document budget used by default.  The gold documents of the
# evaluated queries are always included; the remainder are deterministic
# "distractor" documents (hard negatives) sampled with a fixed seed so the
# experiment is reproducible.  Set to 0 to use the full 1000-doc corpus.
DEFAULT_CORPUS_DOCS = 100
FULL_CORPUS = 0  # sentinel meaning "use every document"