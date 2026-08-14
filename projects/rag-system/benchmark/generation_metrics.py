"""
Generation evaluation (Phase 6) — LLM-as-a-Judge.

Judges are kept strictly separate from the RAG systems.  They are only run
*after* answers are produced and operate on the recorded answer + retrieved
context + (for correctness only) the reference answer.

Judgements are binary where appropriate:

* Faithfulness          : is EVERY factual claim in the answer supported by
                          the provided context?  -> supported / not_supported
* Answer Relevance      : does the answer actually address the question?
                          -> relevant / not_relevant
* Answer Correctness    : is the answer correct w.r.t. the reference answer?
                          -> correct / incorrect

Nothing here is used during retrieval or answer generation, so there is no
benchmark leakage.
"""

from __future__ import annotations

import re

import httpx

from benchmark.config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL

# ---------------------------------------------------------------------------
# Low-level Ollama chat
# ---------------------------------------------------------------------------

def _ollama_chat(
    system: str,
    user: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    temperature: float = 0.0,
    max_tokens: int = 256,
    client: httpx.Client | None = None,
) -> str:
    """Send a chat request to Ollama and return the response text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    close = client is None
    client = client or httpx.Client(timeout=120.0)
    try:
        response = client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
    finally:
        if close:
            client.close()

    msg = data["message"]
    return (msg.get("content") or msg.get("thinking") or "").strip()

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

FAITHFULNESS_SYSTEM = (
    "You are a strict, impartial evaluation judge. You only judge whether a "
    "generated answer is fully supported by a provided context. You never "
    "use outside knowledge to fill gaps."
)

FAITHFULNESS_USER = """
You are given a context, a question, and an answer.

Determine whether EVERY factual claim in the answer is supported by the
provided context.

1. Identify the individual factual claims.
2. Check each claim against the context.
3. Briefly justify the assessment.
4. Return exactly one final label:

VERDICT: supported
or
VERDICT: not_supported

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}
"""

RELEVANCE_SYSTEM = (
    "You are a strict, impartial evaluation judge for answer relevance. "
    "You decide whether an answer actually addresses the question asked."
)

RELEVANCE_USER = """
Determine whether the answer is relevant to (directly addresses) the question.

Return exactly one final label:

VERDICT: relevant
or
VERDICT: not_relevant

QUESTION:
{question}

ANSWER:
{answer}
"""

CORRECTNESS_SYSTEM = (
    "You are a strict, impartial evaluation judge for answer correctness. "
    "Judge whether an answer is correct given a reference (gold) answer."
)

CORRECTNESS_USER = """
Determine whether the answer is correct compared to the reference answer.
The answer may be a paraphrase; it is correct if it conveys the same key
facts as the reference without contradicting it.

Return exactly one final label:

VERDICT: correct
or
VERDICT: incorrect

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

CANDIDATE ANSWER:
{candidate}
"""


# ---------------------------------------------------------------------------
# Judge entry points
# ---------------------------------------------------------------------------

def _format_context(retrieved_sources_texts: list) -> str:
    """Join retrieved chunk texts into a compact numbered context block."""
    if not retrieved_sources_texts:
        return "(no retrieved context)"
    lines = [f"[{i}] {text}" for i, text in enumerate(retrieved_sources_texts, 1)]
    return "\n".join(lines)


def judge_faithfulness(
    question: str,
    context_texts: list[str],
    answer: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    client: httpx.Client | None = None,
) -> str | None:
    """Return ``"supported"``/``"not_supported"``/``None`` (parse failure)."""
    user = FAITHFULNESS_USER.format(
        context=_format_context(context_texts), question=question, answer=answer
    )
    text = _ollama_chat(
        FAITHFULNESS_SYSTEM, user, model=model, base_url=base_url, client=client
    )
    return _parse_verdict(text, "supported", "not_supported")


def judge_relevance(
    question: str,
    answer: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    client: httpx.Client | None = None,
) -> str | None:
    """Return ``"relevant"``/``"not_relevant"``/``None``."""
    user = RELEVANCE_USER.format(question=question, answer=answer)
    text = _ollama_chat(
        RELEVANCE_SYSTEM, user, model=model, base_url=base_url, client=client
    )
    return _parse_verdict(text, "relevant", "not_relevant")


def judge_correctness(
    question: str,
    reference: str,
    candidate: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    client: httpx.Client | None = None,
) -> str | None:
    """Return ``"correct"``/``"incorrect"``/``None``."""
    user = CORRECTNESS_USER.format(
        question=question, reference=reference, candidate=candidate
    )
    text = _ollama_chat(
        CORRECTNESS_SYSTEM, user, model=model, base_url=base_url, client=client
    )
    return _parse_verdict(text, "correct", "incorrect")

def _parse_verdict(text: str, positive: str, negative: str) -> str | None:
    """Extract ``VERDICT: <label>`` from a judge response (case-insensitive)."""
    match = re.search(r"VERDICT\s*:\s*(\w+)", text, re.IGNORECASE)
    if not match:
        return None
    label = match.group(1).strip().lower()
    if label == positive.lower():
        return positive
    if label == negative.lower():
        return negative
    if label.replace("-", "_") == negative.replace("-", "_").replace(" ", "_"):
        return negative
    return None