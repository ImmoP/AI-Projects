"""
Agent state — working memory for one agent run.

Tracks what the agent has done so far: which retrieval queries were
made, what observations came back, and what the final answer is.
"""

from dataclasses import dataclass, field


@dataclass
class AgentObservation:
    """
    A single tool execution record.

    The agent's state stores a list of these so the execution flow
    can be inspected after the run.
    """

    step: int
    action: str  # "retrieve" or "final_answer"
    query: str  # the retrieval query used (empty for final_answer)
    result: str  # tool output formatted as text


@dataclass
class AgentState:
    """
    Complete working memory of one agent run.

    Attributes
    ----------
    question : str
        The original user question.
    step : int
        Current step counter (0-indexed).
    queries : list[str]
        All retrieval queries the agent has made so far.
    observations : list[AgentObservation]
        Sequential record of tool executions.
    final_answer : str | None
        The generated final answer, or None if not yet produced.
    all_chunks : list[dict]
        All chunks retrieved across all retrieval steps, accumulated so
        the generator has the full evidence base at final_answer time.
    """

    question: str
    step: int = 0
    queries: list[str] = field(default_factory=list)
    observations: list[AgentObservation] = field(default_factory=list)
    final_answer: str | None = None
    all_chunks: list = field(default_factory=list)
    # Token usage accumulated across the agent's ReAct/decision LLM calls and
    # the final generation.  These are inference tokens (judge/eval tokens are
    # kept separate and are not added here).
    input_tokens: int = 0
    output_tokens: int = 0
    # Why the agent loop terminated.  One of:
    #   "final_answer"            - the agent emitted a final_answer action
    #   "max_steps"               - the step budget (MAX_STEPS) was exhausted
    #   "invalid_decision_limit"  - too many consecutive malformed/unknown
    #                               decisions (see MAX_INVALID_DECISION_RETRIES)
    # None while the run is still in progress / not yet set.
    termination_reason: str | None = None

    MAX_STEPS: int = 5
