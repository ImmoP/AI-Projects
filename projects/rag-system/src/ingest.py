"""
Entry point for documents into our RAG system.
 Ingestion means taking the raw data from source and converting it into a strcutured format that the rest of the pipeline can process
It extracts the text page by page and adds source and page number into a list.
 
 """
from dataclasses import dataclass
from pathlib import Path

import pymupdf


# define structured object for each pdf
@dataclass
class DocumentPage:
    text: str  # actual text on the page
    source: str # PDF filename
    page: int  # page number


# Load all PDFs from a directory and convert pages into DocumentPage objects
def load_pdfs(pdf_dir: str) -> list[DocumentPage]:
    pages = []

    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        document = pymupdf.open(pdf_path)

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            pages.append(
                DocumentPage(
                    text=text,
                    source=pdf_path.name,
                    page=page_number
                )
            )

        document.close()

    return pages