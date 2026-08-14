"""
Agent tools — expose existing system capabilities as callable tools.

Each tool is a plain function that the agent loop can invoke.
Tools do not hold state; they receive input and return a formatted
string observation that the agent can reason about.
"""

from sentence_transformers import SentenceTransformer

from src.retriever import RetrievalResult, retrieve


def retrieve_tool(
    query: str,
    embedded_chunks: list,
    model: SentenceTransformer,
    top_k: int = 5,
) -> str:
    """
    Perform a dense retrieval search.

    Internally calls `retriever.retrieve()` — the same function used
    by the Classical RAG pipeline. No duplicate logic.

    Parameters
    ----------
    query : str
        The search query.
    embedded_chunks : list
        Pre-computed embedded chunks from the document corpus.
    model : SentenceTransformer
        The embedding model.
    top_k : int
        Number of chunks to return.

    Returns
    -------
    str
        A formatted string listing the retrieved chunks with their
        source, page, score, and text.
    """

    if not query.strip():
        return "Error: empty query."

    results: list[RetrievalResult] = retrieve(
        query=query,
        embedded_chunks=embedded_chunks,
        model=model,
        top_k=top_k,
    )

    if not results:
        return "No relevant documents found."

    lines = []
    for i, chunk in enumerate(results, start=1):
        lines.append(
            f"[{i}] Source: {chunk.source}, Page: {chunk.page}, "
            f"Score: {chunk.score:.4f}"
        )
        lines.append(f"    Text: {chunk.text[:500]}")
        lines.append("")

    return "\n".join(lines).strip()
