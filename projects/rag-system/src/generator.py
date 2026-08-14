"""
Generator module — the "G" in RAG.

Takes the retrieved document chunks and the user query,
constructs a prompt, and sends it to an LLM (via Ollama)
to produce a grounded, citation-aware answer.
"""

from dataclasses import dataclass, field

import httpx

from src.retriever import RetrievalResult

# Default Ollama settings — change via environment or constructor args
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"

# System prompt for the LLM — instructs it to answer grounded in the context
SYSTEM_PROMPT = """You are a helpful research assistant. You answer questions based *only* on the
provided context (retrieved passages from academic papers).

Instructions:
1. Answer concisely using only the context below.
2. If the context does not contain enough information to answer the question, say
   "I cannot find sufficient information in the provided context to answer this question."
3. Cite the source document and page number for each fact you use, like: (Source.pdf, p. 5)
4. Do NOT make up facts or use outside knowledge.
5. Format your answer in clear paragraphs."""


@dataclass
class GenerationResult:
    """The final output of the RAG pipeline."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    # sources is a list of {"source": str, "page": int, "score": float}
    # so the caller can display them alongside the answer
    # Token usage surfaced from the Ollama response (None if unavailable).
    input_tokens: int | None = None
    output_tokens: int | None = None


def _format_context(chunks: list[RetrievalResult]) -> str:
    """Turn a list of retrieved chunks into a numbered context block for the prompt."""

    lines = []

    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{idx}] (Source: {chunk.source}, Page: {chunk.page}, "
            f"Relevance: {chunk.score:.3f})"
        )
        lines.append(chunk.text)
        lines.append("")  # blank line between chunks

    return "\n".join(lines)


def _build_prompt(
    query: str,
    chunks: list[RetrievalResult],
) -> list[dict]:
    """
    Build the message list for the Ollama chat API.

    Returns a list of dicts with "role" and "content" keys,
    suitable for the /api/chat endpoint.
    """

    context = _format_context(chunks)

    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer (with citations):"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def generate_answer(
    query: str,
    chunks: list[RetrievalResult],
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> GenerationResult:
    """
    Send the query + retrieved chunks to an Ollama LLM and return the answer.

    Parameters
    ----------
    query : str
        The user's original question.
    chunks : list[RetrievalResult]
        The top-k chunks retrieved from the vector store.
    model : str
        Ollama model name (e.g. "llama3.2", "mistral", "phi3").
    base_url : str
        Base URL of the running Ollama instance.
    temperature : float
        LLM temperature — lower = more deterministic.
    max_tokens : int
        Maximum number of tokens in the generated answer.

    Returns
    -------
    GenerationResult containing the answer text and source metadata.
    """

    if not query.strip():
        raise ValueError("Query must not be empty")

    if not chunks:
        return GenerationResult(
            answer="No relevant documents were retrieved to answer this question.",
            sources=[],
        )

    messages = _build_prompt(query, chunks)

    # Prepare the request payload for Ollama's /api/chat endpoint
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    # Send request to Ollama
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    message = data["message"]
    # Some Ollama models (e.g. Qwen3, DeepSeek) output reasoning in a
    # "thinking" field. If the visible "content" is empty, fall back to
    # the "thinking" text so the user still sees the model's output.
    answer_text = (message.get("content") or message.get("thinking") or "").strip()

    # Token usage from the Ollama response (may be absent on some servers).
    input_tokens = data.get("prompt_eval_count")
    output_tokens = data.get("eval_count")

    # Collect source information for the caller
    seen_sources = set()
    sources = []

    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append(
                {
                    "source": chunk.source,
                    "page": chunk.page,
                    "score": chunk.score,
                }
            )

    return GenerationResult(answer=answer_text, sources=sources,
                            input_tokens=input_tokens, output_tokens=output_tokens)


if __name__ == "__main__":
    # Simple test — run the full pipeline up to generation
    from src.chunking import chunk_pages
    from src.embeddings import embed_chunks, load_embedding_model
    from src.ingest import load_pdfs
    from src.retriever import retrieve

    print("Loading PDFs...")
    pages = load_pdfs("pdfs")

    print("Chunking...")
    chunks = chunk_pages(pages, chunk_size=500, overlap=100)

    print("Embedding...")
    model = load_embedding_model()
    embedded_chunks = embed_chunks(chunks, model)

    query = "What is multi-head attention and why is it useful?"

    print(f"\nQuery: {query}")
    print("Retrieving...")

    results = retrieve(
        query=query,
        embedded_chunks=embedded_chunks,
        model=model,
        top_k=5,
    )

    print(f"Retrieved {len(results)} chunks")
    print("Generating answer via Ollama...\n")

    generation = generate_answer(
        query=query,
        chunks=results,
        model=DEFAULT_MODEL,
    )

    print("=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(generation.answer)
    print()
    print("=" * 80)
    print("SOURCES:")
    for src in generation.sources:
        print(f"  - {src['source']}, p. {src['page']} (score: {src['score']:.3f})")
