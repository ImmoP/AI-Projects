# takes the structured pages and splits them into smaller overlapping pieces that are easier to retrieve 
from dataclasses import dataclass

from src.ingest import DocumentPage


# structure of one chunck
@dataclass
class TextChunk:
    text: str  # chunk text
    source: str # PDF filename
    page: int # PDF page
    chunk_id: int # unique chunk number

# takes a list of pages and returns a list of chunks
def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 500, # each chunk is up to 500 words
    overlap: int = 100 # neighbouring chunks share 100 words
) -> list[TextChunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must not be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    chunk_id = 0

    for page in pages:
        words = page.text.split()

        start = 0

        while start < len(words):
            end = start + chunk_size

            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        source=page.source,
                        page=page.page,
                        chunk_id=chunk_id
                    )
                )

                chunk_id += 1

            start += chunk_size - overlap

    return chunks

# test code
if __name__ == "__main__":
    from src.ingest import load_pdfs

    pages = load_pdfs("pdfs")

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=100
    )

    print(f"Loaded {len(pages)} pages")
    print(f"Created {len(chunks)} chunks")

    if chunks:
        print()
        print("First chunk:")
        print(chunks[0])