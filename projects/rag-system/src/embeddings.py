# We now convert every text chunk into a numerical vector that represents it's semantic meaning


from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from src.chunking import TextChunk


# Data typ structur for chunks after embedding
@dataclass
class EmbeddedChunk:
    text: str
    source: str
    page: int
    chunk_id: int
    embedding: list[float]

# loading embedding model and embedding chunks
def load_embedding_model(
    model_name: str = "BAAI/bge-small-en-v1.5"
) -> SentenceTransformer:
    """
    Load and return the sentence-transformers embedding model.
    """

    model = SentenceTransformer(model_name)

    return model

# embedding chunks
def embed_chunks(
    chunks: list[TextChunk],
    model: SentenceTransformer
) -> list[EmbeddedChunk]:
    """
    Convert TextChunk objects into EmbeddedChunk objects.

    Each chunk receives a numerical embedding vector while
    preserving its original text and metadata.
    """

    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append(
            EmbeddedChunk(
                text=chunk.text,
                source=chunk.source,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                embedding=embedding.tolist()
            )
        )

    return embedded_chunks


if __name__ == "__main__":
    from src.chunking import chunk_pages
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

    print(f"Loaded {len(pages)} pages")
    print(f"Created {len(chunks)} chunks")
    print(f"Created {len(embedded_chunks)} embeddings")

    if embedded_chunks:
        first_chunk = embedded_chunks[0]

        print()
        print("First embedded chunk:")
        print(f"Source: {first_chunk.source}")
        print(f"Page: {first_chunk.page}")
        print(f"Chunk ID: {first_chunk.chunk_id}")
        print(f"Embedding dimensions: {len(first_chunk.embedding)}")
        print(f"First 10 values: {first_chunk.embedding[:10]}")