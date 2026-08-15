"""
Regression tests for the ReAct agent retry / termination control flow.

Guards against the production bug (query unans_014) where the ReAct loop ran
effectively forever:

    --- Step 1 ---
      Agent: unknown action 'unknown'. Asking again.
    --- Step 1 ---
      Agent: unknown action 'unknown'. Asking again.
    ... (hundreds of times)

Root cause: the loop was bounded only by ``state.step < MAX_STEPS``, but
``state.step`` is *not* incremented for malformed/unknown decisions nor for
``retrieve`` actions with an empty query.  A model that kept emitting
unparseable JSON therefore never advanced ``state.step`` and never hit the
``MAX_STEPS`` guard.

The fix introduces a separate, bounded *consecutive* invalid-decision counter
(``MAX_INVALID_DECISION_RETRIES``) and a ``termination_reason`` field on
``AgentState``.  These tests mock ``_call_llm`` (and the retrieval/generation
I/O) so no network or real model is required:

  A. Repeated unknown actions are bounded.
  B. Empty retrieve queries are bounded.
  C. A valid action resets the consecutive invalid counter.
  D. A normal retrieve -> final_answer run is unchanged.
  E. Token accounting includes invalid decision attempts.
  F. Existing MAX_STEPS behaviour remains bounded (termination_reason="max_steps").
  G. Invalid-limit fallback reuses already-retrieved evidence.
  H. evaluate_agentic() exposes termination_reason on the benchmark record.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.agent.react_agent import MAX_INVALID_DECISION_RETRIES, run_agent
from src.agent.state import AgentState
from src.generator import GenerationResult
from src.retriever import RetrievalResult

# The exact unanswerable query that triggered the infinite loop in production.
QUESTION = (
    "What is the peak GPU memory usage reported for serving a 70B parameter "
    "model in the paper 'Efficient Memory Management for Large Language Model "
    "Serving' (vLLM)?"
)

FALLBACK_NO_EVIDENCE = "Unable to produce a valid tool decision after repeated attempts."


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

def _chunk(chunk_id=0, source="paper.pdf", page=1, score=0.9,
           text="evidence chunk text"):
    return RetrievalResult(text=text, source=source, page=page,
                           chunk_id=chunk_id, score=score)


def _gen(answer="Generated final answer.", in_tok=10, out_tok=20):
    return GenerationResult(answer=answer, input_tokens=in_tok,
                            output_tokens=out_tok)


def _run(scripts, *, max_steps=5, retrieve_results=None, gen_result=None,
         retrieve_tool_text="observation text"):
    """Run run_agent with ``_call_llm`` scripted and all I/O mocked.

    ``scripts`` is either a list of ``(llm_text, in_tok, out_tok)`` tuples
    consumed in order, or a callable returning such a tuple (for an "always"
    response).  retrieve_tool, generate_answer and retriever.retrieve are
    replaced with deterministic stubs.  Returns ``(state, llm_mock,
    retrieve_tool_mock, generate_mock, retrieve_mock)``.
    """
    retrieve_results = [_chunk()] if retrieve_results is None else retrieve_results
    gen_result = _gen() if gen_result is None else gen_result
    with patch("src.agent.react_agent._call_llm", side_effect=scripts) as llm_mock, \
         patch("src.agent.react_agent.retrieve_tool",
               return_value=retrieve_tool_text) as rt_mock, \
         patch("src.agent.react_agent.generate_answer",
               return_value=gen_result) as gen_mock, \
         patch("src.retriever.retrieve",
               return_value=retrieve_results) as retrieve_mock:
        state = run_agent(
            question=QUESTION, embedded_chunks=[], model=object(),
            max_steps=max_steps, top_k=5, ollama_model="m", ollama_base_url="u",
        )
    return state, llm_mock, rt_mock, gen_mock, retrieve_mock


# ---------------------------------------------------------------------------
# A. Repeated unknown actions are bounded
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_repeated_unknown_actions_are_bounded():
    # Malformed text -> _parse_tool_call returns {"action": "unknown", ...}.
    state, llm_mock, *_ = _run(lambda *a, **k: ("garbage response", 5, 7))

    assert llm_mock.call_count == MAX_INVALID_DECISION_RETRIES
    assert state.step == 0  # invalid decisions never count as a real step
    assert state.termination_reason == "invalid_decision_limit"
    assert state.final_answer == FALLBACK_NO_EVIDENCE


# ---------------------------------------------------------------------------
# B. Empty retrieve queries are bounded
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_empty_retrieve_queries_are_bounded():
    empty_retrieve = '{"action": "retrieve", "query": "   "}'
    state, llm_mock, *_ = _run(lambda *a, **k: (empty_retrieve, 5, 7))

    assert llm_mock.call_count == MAX_INVALID_DECISION_RETRIES
    assert state.step == 0
    assert state.termination_reason == "invalid_decision_limit"
    assert state.final_answer == FALLBACK_NO_EVIDENCE


# ---------------------------------------------------------------------------
# C. A valid action resets the consecutive invalid counter
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_valid_action_resets_invalid_counter():
    unknown = ("garbage response", 1, 1)
    retrieve = ('{"action": "retrieve", "query": "q"}', 1, 1)
    final = ('{"action": "final_answer"}', 1, 1)
    # unknown, unknown, valid retrieve, unknown, unknown, final_answer
    scripts = [unknown, unknown, retrieve, unknown, unknown, final]

    state, llm_mock, rt_mock, gen_mock, _ = _run(scripts)

    # The two invalids *after* the valid retrieve only reach 2/3, so the loop
    # must NOT terminate early; it reaches final_answer.
    assert llm_mock.call_count == 6
    assert state.termination_reason == "final_answer"
    assert state.final_answer == "Generated final answer."
    assert state.step == 2  # one retrieve + one final_answer
    assert rt_mock.call_count == 1  # the valid retrieve ran once


# ---------------------------------------------------------------------------
# D. Normal retrieve -> final_answer run is unchanged
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_normal_retrieve_then_final_answer_unchanged():
    retrieve = ('{"action": "retrieve", "query": "example query"}', 2, 3)
    final = ('{"action": "final_answer"}', 4, 5)
    gen = _gen(answer="final answer text", in_tok=10, out_tok=20)

    state, llm_mock, rt_mock, gen_mock, _ = _run([retrieve, final], gen_result=gen)

    assert rt_mock.call_count == 1
    assert gen_mock.call_count == 1  # generation happened via the final_answer path
    assert state.step == 2
    assert state.termination_reason == "final_answer"
    assert state.final_answer == "final answer text"
    # Token accounting: both decision calls + the final generation.
    assert state.input_tokens == 2 + 4 + 10
    assert state.output_tokens == 3 + 5 + 20


# ---------------------------------------------------------------------------
# E. Token accounting includes invalid decision attempts
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_token_accounting_includes_invalid_attempts():
    # Every decision attempt (even malformed ones) must accrue token usage.
    state, llm_mock, *_ = _run(lambda *a, **k: ("garbage response", 7, 11))

    assert llm_mock.call_count == MAX_INVALID_DECISION_RETRIES
    assert state.input_tokens == 7 * MAX_INVALID_DECISION_RETRIES
    assert state.output_tokens == 11 * MAX_INVALID_DECISION_RETRIES
    assert state.termination_reason == "invalid_decision_limit"


# ---------------------------------------------------------------------------
# F. Existing MAX_STEPS behaviour remains bounded
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_max_steps_behaviour_remains_bounded():
    # Always retrieve validly until the step budget is exhausted.
    retrieve = '{"action": "retrieve", "query": "q"}'
    gen = _gen(answer="max-steps fallback answer", in_tok=0, out_tok=0)

    state, llm_mock, rt_mock, gen_mock, _ = _run(
        lambda *a, **k: (retrieve, 1, 1), max_steps=2, gen_result=gen,
    )

    assert llm_mock.call_count == 2  # two real retrieve steps
    assert rt_mock.call_count == 2
    assert state.step == 2
    assert state.termination_reason == "max_steps"
    # Existing fallback generation from accumulated evidence still runs.
    assert gen_mock.call_count == 1
    assert state.final_answer == "max-steps fallback answer"


# ---------------------------------------------------------------------------
# G. Invalid-limit fallback reuses already-retrieved evidence
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_invalid_limit_fallback_uses_existing_evidence():
    retrieve = ('{"action": "retrieve", "query": "q"}', 1, 1)
    unknown = ("garbage response", 1, 1)
    gen = _gen(answer="evidence-based fallback answer")

    # valid retrieve, then three consecutive unknowns -> invalid limit.
    state, llm_mock, rt_mock, gen_mock, retrieve_mock = _run(
        [retrieve, unknown, unknown, unknown], gen_result=gen,
    )

    assert llm_mock.call_count == 4
    assert state.step == 1  # only the valid retrieve counted as a step
    assert state.termination_reason == "invalid_decision_limit"
    # The fallback must generate from the previously retrieved evidence rather
    # than hanging / fabricating.
    assert gen_mock.call_count == 1
    assert retrieve_mock.call_count >= 1  # re-ran the accumulated query
    assert state.final_answer == "evidence-based fallback answer"
    # Must NOT report this as a MAX_STEPS condition.
    assert state.termination_reason != "max_steps"


# ---------------------------------------------------------------------------
# H. evaluate_agentic() exposes termination_reason
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_evaluate_agentic_exposes_termination_reason():
    from benchmark.evaluators import evaluate_agentic

    ex = SimpleNamespace(
        query_id="unans_014", question=QUESTION, reference_answer="",
        gold_doc_id="", gold_section_id=0, query_type="unanswerable",
        source="unanswerable",
    )
    fake_state = AgentState(
        question=QUESTION, final_answer=FALLBACK_NO_EVIDENCE,
        termination_reason="invalid_decision_limit",
    )

    with patch("benchmark.evaluators.run_agent", return_value=fake_state):
        rec = evaluate_agentic(ex, [], object(), top_k=5,
                               ollama_model="m", base_url="u", max_steps=3)

    assert rec["status"] == "ok"
    assert "termination_reason" in rec
    assert rec["termination_reason"] == "invalid_decision_limit"