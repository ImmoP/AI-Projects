"""
Manual demo script for the Stage 1 ReAct agent.

This is NOT a test. It needs a running local Ollama server and a real PDF
corpus on disk -- neither of which CI has. See tests/ for the automated,
offline test suite.

Usage (run from the project root):
    python -m examples.demo_agentic "Why is multi-head attention useful?"
    python -m examples.demo_agentic "..." --pdf-dir pdfs \
        --source-filter "Attention Is All You Need" --model qwen2.5:3b
"""

import argparse
import time

from src.agent.react_agent import run_agent
from src.chunking import chunk_pages
from src.embeddings import embed_chunks, load_embedding_model
from src.ingest import load_pdfs


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual demo of the Stage 1 ReAct agent.")
    parser.add_argument("question", type=str, help="The question to ask.")
    parser.add_argument("--pdf-dir", type=str, default="pdfs", help="Directory containing the PDF corpus.")
    parser.add_argument(
        "--source-filter",
        type=str,
        default=None,
        help="Only load pages whose source filename contains this substring.",
    )
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="Ollama model name.")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print("DEMO: Agentic RAG - Stage 1")
    print("=" * 60)

    print()
    print("[1/4] Loading documents...")
    pages = load_pdfs(args.pdf_dir)
    if args.source_filter:
        pages = [p for p in pages if args.source_filter in p.source]
    print(f"  {len(pages)} pages loaded")

    print()
    print("[2/4] Chunking...")
    chunks = chunk_pages(pages, chunk_size=500, overlap=100)
    print(f"  {len(chunks)} chunks")

    print()
    print("[3/4] Loading embedding model and embedding chunks...")
    model = load_embedding_model()
    embedded = embed_chunks(chunks, model)
    print(f"  {len(embedded)} embeddings")

    print()
    print("[4/4] Running agent...")
    print(f"  Question: {args.question}")
    print()

    t_start = time.time()
    state = run_agent(
        question=args.question,
        embedded_chunks=embedded,
        model=model,
        ollama_model=args.model,
        max_steps=args.max_steps,
        top_k=args.top_k,
    )
    elapsed = time.time() - t_start

    print()
    print("=" * 60)
    print("AGENT EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Steps executed: {state.step}")
    print(f"Retrieval queries: {len(state.queries)}")
    for i, q in enumerate(state.queries):
        print(f"  Step {i + 1}) retrieve query=\"{q}\"")
    print()
    print("FINAL ANSWER:")
    print("-" * 60)
    print(state.final_answer or "(no answer produced)")
    print()
    print("Observations stored in agent state:")
    for obs in state.observations:
        print(f"  Step {obs.step}: action={obs.action}, query=\"{obs.query[:50] if obs.query else ''}\"")
    print()
    print("Done")


if __name__ == "__main__":
    main()
