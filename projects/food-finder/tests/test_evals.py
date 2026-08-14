"""Unit tests for deterministic evaluation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from evals.run_evals import arguments_match, extract_tool_calls, run_case
from smolagents import Model
from smolagents.models import ChatMessage, MessageRole


class ScriptedModel(Model):
    """Return one known CodeAgent action without contacting a model server."""

    def generate(self, messages, **kwargs) -> ChatMessage:
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=(
                "```python\n"
                'result = search_restaurants(city="Berlin", cuisine="Japanese")\n'
                "final_answer(result)\n"
                "```"
            ),
        )


def test_extracts_literal_tool_call_from_agent_memory() -> None:
    memory = SimpleNamespace(
        steps=[
            SimpleNamespace(
                code_action=(
                    "results = search_restaurants(city='Berlin', cuisine='Japanese', "
                    "min_reviews=100)\nfinal_answer(results)"
                )
            )
        ]
    )

    assert extract_tool_calls(memory) == [
        {
            "name": "search_restaurants",
            "arguments": {
                "city": "Berlin",
                "cuisine": "Japanese",
                "min_reviews": 100,
            },
        }
    ]


def test_argument_matching_is_case_insensitive_and_subset_based() -> None:
    expected = {"city": "  BERLIN ", "cuisine": "Japanese"}
    observed = {"city": "berlin", "cuisine": "japanese", "min_rating": 4.0}

    assert arguments_match(expected, observed)


@pytest.mark.smoke
def test_offline_case_runs_without_google_api_call() -> None:
    case = {
        "id": "offline_end_to_end",
        "prompt": "Find Japanese restaurants in Berlin.",
        "tool_needed": True,
        "expected_tool": "search_restaurants",
        "expected_arguments": {"city": "Berlin", "cuisine": "Japanese"},
    }

    result = run_case(
        case,
        model=ScriptedModel(model_id="scripted"),
        max_steps=2,
        offline=True,
    )

    assert result.observed_tools == ["search_restaurants"]
    assert result.tool_decision_correct
    assert result.arguments_correct
    assert result.steps == 1
    assert result.error is None
