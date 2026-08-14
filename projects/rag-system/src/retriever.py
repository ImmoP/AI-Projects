"""
This code implements the actual retrieval step. The query is first converted into an embedding vector.
Cosine similarity is then calculated between the query embedding and the stored document embeddings.
The documents are ranked according to their similarity scores, and the top-k most similar results are returned.
"""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.embeddings import EmbeddedChunk

# Data structure to hold the results of a retrieval operation.
# Each result contains the text of the chunk, the source document,
# the page number, the chunk ID, and a similarity score indicating
# how similar the chunk is to the query.

@dataclass
class RetrievalResult:
    text: str
    source: str
    page: int
    chunk_id: int
    score: float  # How similar is this chunk to the query. Higher is better.


# Convert the user query into a vector

def embed_query(
    query: str,
    model: SentenceTransformer
) -> np.ndarray:
    """
    Convert a user query into an embedding vector.
    """

    if not query.strip():
        raise ValueError("Query must not be empty")

    # Convert query into an embedding vector using the provided model (normalized)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    return np.asarray(
        query_embedding,
        dtype=np.float32
    )


# Calculate the cosine similarity between two vectors

def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> float:
    """
    Calculate the cosine similarity between two vectors.
    """

    # Convert both vectors to numpy arrays of type float32

    vector_a = np.asarray(vector_a, dtype=np.float32)
    vector_b = np.asarray(vector_b, dtype=np.float32)

    # Calculate the vector lengths

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    similarity = np.dot(vector_a, vector_b) / denominator

    return float(similarity)


# Receives query, document chunks, and embedding model,
# returns the top-k most similar chunks to the query

def retrieve(
    query: str,
    embedded_chunks: list[EmbeddedChunk],
    model: SentenceTransformer,
    top_k: int = 5
) -> list[RetrievalResult]:
    """
    Find the chunks that are most semantically similar
    to the user query.
    """

    if not embedded_chunks:
        return []

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    # Embed the query into a vector using the provided model

    query_embedding = embed_query(
        query,
        model
    )

    # Result list to hold the retrieval results

    results = []

    for chunk in embedded_chunks:

        # Calculate the cosine similarity between the query embedding
        # and the chunk embedding

        similarity = cosine_similarity(
            query_embedding,
            chunk.embedding
        )

        # Create result: Now each chunk has its retrieval score attached

        results.append(
            RetrievalResult(
                text=chunk.text,
                source=chunk.source,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                score=similarity
            )
        )

    # Sort the results by score in descending order

    results.sort(
        key=lambda result: result.score,
        reverse=True
    )

    # Return the top-k results

    return results[:top_k]


# Testing block to demonstrate the retrieval process.
# It loads PDFs, chunks them, embeds the chunks,
# and then retrieves the most relevant chunks for a given query.

if __name__ == "__main__":
    from src.chunking import chunk_pages
    from src.embeddings import embed_chunks, load_embedding_model
    from src.ingest import load_pdfs

    pages = load_pdfs("pdfs")

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=100
    )

    model = load_embedding_model()

    embedded_chunks = embed_chunks(
        chunks,
        model
    )

    query = "Why is multi-head attention useful?"

    results = retrieve(
        query=query,
        embedded_chunks=embedded_chunks,
        model=model,
        top_k=5
    )

    print()
    print(f"Query: {query}")
    print()

    for rank, result in enumerate(results, start=1):

        print(f"Result {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {result.source}")
        print(f"Page: {result.page}")
        print(f"Chunk ID: {result.chunk_id}")
        print()
        print(result.text[:500])
        print()
        print("-" * 80)
        print()