"""
Smoke test for the tokenizer wrapper: does encode -> decode reproduce the
original text exactly? tiktoken's GPT-2 BPE encoding is byte-level, so this
holds for arbitrary text, not just "clean" ASCII.
"""

import pytest
from src.llm_from_scratch.tokenizer import text_to_token_ids, token_ids_to_text


@pytest.mark.smoke
@pytest.mark.parametrize(
    "text",
    [
        "Hello, world!",
        "The lighthouse at Cold Point stood on a spit of black rock.",
        "Punctuation, casing, and   irregular   whitespace -- all of it.",
        "Unicode: café, naïve, Zürich, привет, 日本語.",
        "",
    ],
)
def test_encode_decode_roundtrip(tokenizer, text):
    token_ids = text_to_token_ids(text, tokenizer)
    decoded = token_ids_to_text(token_ids, tokenizer)
    assert decoded == text


@pytest.mark.smoke
def test_roundtrip_on_mini_corpus(tokenizer, mini_corpus_text):
    token_ids = text_to_token_ids(mini_corpus_text, tokenizer)
    decoded = token_ids_to_text(token_ids, tokenizer)
    assert decoded == mini_corpus_text
