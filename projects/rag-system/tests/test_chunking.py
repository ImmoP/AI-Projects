"""
Smoke tests for src.chunking: does splitting produce the expected number
and boundaries of chunks? No embedding model, no I/O involved -- chunk_pages
is a pure function.
"""

import pytest
from src.chunking import chunk_pages
from src.ingest import DocumentPage


def _page(word_count: int, source: str = "doc.pdf", page: int = 1) -> DocumentPage:
    text = " ".join(f"w{i}" for i in range(word_count))
    return DocumentPage(text=text, source=source, page=page)


@pytest.mark.smoke
def test_chunk_count_and_boundaries():
    # 250 words, chunk_size=100, overlap=20 -> step=80 -> chunks at 0, 80, 160, 240
    page = _page(250)

    chunks = chunk_pages([page], chunk_size=100, overlap=20)

    assert len(chunks) == 4
    assert [c.chunk_id for c in chunks] == [0, 1, 2, 3]

    assert chunks[0].text.split() == [f"w{i}" for i in range(0, 100)]
    assert chunks[1].text.split() == [f"w{i}" for i in range(80, 180)]
    assert chunks[2].text.split() == [f"w{i}" for i in range(160, 250)]
    assert chunks[3].text.split() == [f"w{i}" for i in range(240, 250)]

    for chunk in chunks:
        assert chunk.source == "doc.pdf"
        assert chunk.page == 1


@pytest.mark.smoke
def test_neighbouring_chunks_share_the_overlap():
    page = _page(250)
    chunks = chunk_pages([page], chunk_size=100, overlap=20)

    # last 20 words of chunk 0 == first 20 words of chunk 1
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]


@pytest.mark.smoke
def test_chunk_id_continues_across_pages():
    pages = [_page(50, source="a.pdf", page=1), _page(50, source="b.pdf", page=1)]

    chunks = chunk_pages(pages, chunk_size=50, overlap=0)

    assert len(chunks) == 2
    assert [c.chunk_id for c in chunks] == [0, 1]
    assert chunks[0].source == "a.pdf"
    assert chunks[1].source == "b.pdf"


@pytest.mark.smoke
def test_empty_pages_yield_no_chunks():
    assert chunk_pages([], chunk_size=100, overlap=20) == []


@pytest.mark.smoke
@pytest.mark.parametrize(
    "chunk_size,overlap",
    [
        (0, 0),
        (-10, 0),
        (100, -1),
        (100, 100),
        (100, 150),
    ],
)
def test_invalid_chunk_size_or_overlap_raises(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([_page(10)], chunk_size=chunk_size, overlap=overlap)
