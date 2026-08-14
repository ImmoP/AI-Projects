"""
Manual demo script for the classical RAG pipeline.

This is NOT a test. It needs a running local Ollama server and a real PDF
corpus on disk -- neither of which CI has. See tests/ for the automated,
offline test suite.

Usage (run from the project root):
    python -m examples.demo_rag "What is multi-head attention?"
    python -m examples.demo_rag "Explain the transformer architecture" \
        --pdf-dir pdfs --model qwen3:8b --top-k 5
"""

import argparse

from src.generator import DEFAULT_MODEL
from src.rag import RagPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual demo of the classical RAG pipeline.")
    parser.add_argument("query", type=str, help="The question to ask.")
    parser.add_argument("--pdf-dir", type=str, default="pdfs", help="Directory containing the PDF corpus.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    pipeline = RagPipeline(
        ollama_model=args.model,
        top_k=args.top_k,
        temperature=args.temperature,
        use_cache=False,
    )

    pipeline.load_and_embed(args.pdf_dir)

    print()
    print("QUERY:")
    print(args.query)
    print()

    result = pipeline.answer(args.query)

    print()
    print("=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(result.answer)
    print()
    print("=" * 80)
    print("SOURCES:")
    for source in result.sources:
        print(f"  - {source['source']} (score: {source['score']:.3f})")
    print()
    print("Done")


if __name__ == "__main__":
    main()
