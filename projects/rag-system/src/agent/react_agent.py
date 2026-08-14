"""
ReAct Agent - minimal agentic retrieval loop.

The agent iteratively decides whether to retrieve more information
or produce a final answer. It uses structured JSON tool calls so that
the decision process is observable and parsable.

Stage 1 supports exactly two actions:
  - "retrieve"      - perform a dense retrieval search
  - "final_answer"  - generate the final answer from accumulated evidence

The agent reuses retriever.py and generator.py from the Classical RAG
pipeline.  It does not duplicate any retrieval or generation logic.
"""

import json
import re

import httpx

from src.agent.state import AgentObservation, AgentState
from src.agent.tools import retrieve_tool

AGENT_SYSTEM_PROMPT = """You are a research assistant that answers questions using a library of
academic PDFs. You have access to two tools:

1.  retrieve(query: str)
    - Use this to search your PDF library.
    - The query should be a short search phrase (3-8 words).
    - Returns relevant passages with source and page number.

2.  final_answer()
    - Use this when you have enough evidence to answer the question.
    - You will then be given all retrieved passages to produce the answer.

Rules:
- You may retrieve MULTIPLE times with different queries.
- Each retrieval costs a step, so be thoughtful.
- After retrieval you will see what was found.
- If the results are not sufficient, retrieve again with a better query.

On each step, output EXACTLY ONE JSON object with no other text:

For retrieval:
{{
    "action": "retrieve",
    "query": "your search query here"
}}

For answering:
{{
    "action": "final_answer"
}}
"""

from src.generator import DEFAULT_MODEL, OLLAMA_BASE_URL, generate_answer


def _parse_tool_call(text: str) -> dict:
    """Extract a JSON tool call from the LLM's response text."""

    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return {"action": "unknown", "error": f"could not parse: {text[:100]}"}


def _call_llm(messages, model=DEFAULT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1):
    """
    Send a chat request to Ollama.

    Returns ``(text, input_tokens, output_tokens)`` so the agent loop can
    accumulate token usage.  Token counts are None when the server omits them.
    """

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 512},
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

    msg = data["message"]
    text = (msg.get("content") or msg.get("thinking") or "").strip()
    return text, data.get("prompt_eval_count"), data.get("eval_count")



def run_agent(question, embedded_chunks, model,
              ollama_model=DEFAULT_MODEL, max_steps=5, top_k=5,
              ollama_base_url=OLLAMA_BASE_URL):
    """
    Run the Stage 1 ReAct agent loop.

    Parameters
    ----------
    question : str
        The user's question.
    embedded_chunks : list
        Pre-computed embedded chunks.
    model : SentenceTransformer
        The embedding model.
    ollama_model : str
        Ollama model name for the agent's decisions.
    max_steps : int
        Maximum number of retrieval steps.
    top_k : int
        Number of chunks per retrieval call.
    ollama_base_url : str
        Base URL of the Ollama server.

    Returns
    -------
    AgentState
        The full agent state including observations and final answer.
    """

    state = AgentState(question=question, MAX_STEPS=max_steps)

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nWhat is your first step?"},
    ]

    while state.step < state.MAX_STEPS:

        print(f"\n--- Step {state.step + 1} ---")

        llm_response, tok_in, tok_out = _call_llm(messages, model=ollama_model, base_url=ollama_base_url)
        state.input_tokens += int(tok_in or 0)
        state.output_tokens += int(tok_out or 0)
        decision = _parse_tool_call(llm_response)
        action = decision.get("action", "unknown")

        if action == "retrieve":
            query = decision.get("query", "").strip()

            if not query:
                print("  Agent: invalid retrieve - no query. Skipping.")
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "user", "content": "Your retrieve action had no query. Please provide a query."})
                continue

            print("  Action: retrieve")
            print(f"  Query: {query}")

            observation_text = retrieve_tool(
                query=query,
                embedded_chunks=embedded_chunks,
                model=model,
                top_k=top_k,
            )

            state.step += 1
            state.queries.append(query)
            state.observations.append(AgentObservation(
                step=state.step, action="retrieve", query=query, result=observation_text,
            ))

            print(f"  Observation: {len(observation_text)} chars retrieved")

            messages.append({"role": "assistant", "content": llm_response})
            messages.append({
                "role": "user",
                "content": (
                    f"Retrieval results for query '{query}':\n"
                    f"{observation_text}\n\n"
                    "Do you need more information or can you answer now?\n"
                    "Respond with a JSON action."
                ),
            })

        elif action == "final_answer":
            print("  Action: final_answer")

            # Re-retrieve with all queries to get proper objects for generator
            from src.retriever import retrieve as _retrieve

            retrieved = []
            for q in state.queries:
                qr = _retrieve(query=q, embedded_chunks=embedded_chunks, model=model, top_k=top_k)
                retrieved.extend(qr)

            # Deduplicate by chunk_id
            seen_ids = set()
            unique_chunks = []
            for r in retrieved:
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    unique_chunks.append(r)

            if not unique_chunks:
                state.final_answer = "No relevant documents were retrieved to answer this question."
            else:
                generation = generate_answer(
                    query=question,
                    chunks=unique_chunks,
                    model=ollama_model,
                    base_url=ollama_base_url,
                )
                state.final_answer = generation.answer
                state.input_tokens += int(generation.input_tokens or 0)
                state.output_tokens += int(generation.output_tokens or 0)

            state.step += 1
            state.observations.append(AgentObservation(
                step=state.step, action="final_answer", query="", result=state.final_answer or "",
            ))
            print(f"  Answer produced ({len(state.final_answer or '')} chars)")
            break

        else:
            print(f"  Agent: unknown action '{action}'. Asking again.")
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({
                "role": "user",
                "content": (
                    f"I don't understand action '{action}'. "
                    'Please respond with either:\n'
                    '{"action": "retrieve", "query": "..."}\n'
                    'or\n'
                    '{"action": "final_answer"}'
                ),
            })

    if state.final_answer is None:
        print(f"\n  Max steps ({state.MAX_STEPS}) reached without final_answer.")
        from src.retriever import retrieve as _retrieve
        all_retrieved = []
        for q in state.queries:
            qr = _retrieve(query=q, embedded_chunks=embedded_chunks, model=model, top_k=top_k)
            all_retrieved.extend(qr)

        if all_retrieved:
            generation = generate_answer(query=question, chunks=all_retrieved, model=ollama_model, base_url=ollama_base_url)
            state.final_answer = generation.answer
            state.input_tokens += int(generation.input_tokens or 0)
            state.output_tokens += int(generation.output_tokens or 0)
        else:
            state.final_answer = "Maximum steps reached without sufficient evidence to answer."

    return state


