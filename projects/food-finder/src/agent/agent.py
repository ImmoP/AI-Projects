"""Factory for the Food Finder agent."""

from __future__ import annotations

from smolagents import CodeAgent
from smolagents.models import Model

from .tools import search_restaurants

AGENT_INSTRUCTIONS = """
You are a restaurant recommendation agent.

Rules:
1. Use search_restaurants when the user asks for restaurant recommendations.
2. Never invent restaurant names, ratings, review counts, addresses, or prices.
3. Pass the requested city and cuisine to the tool.
4. Judge the best restaurants using both rating and number of reviews.
5. Prefer reliable ratings backed by many reviews.
6. Present recommendations clearly and include the restaurant name, rating,
   review count, address, price level (when available), and Google Maps link.
7. Briefly explain why the top restaurants were selected.
""".strip()


def build_agent(
    model: Model,
    *,
    max_steps: int = 5,
    verbosity_level: int = 1,
) -> CodeAgent:
    """Build a Food Finder agent around any smolagents-compatible model."""
    return CodeAgent(
        model=model,
        tools=[search_restaurants],
        instructions=AGENT_INSTRUCTIONS,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
    )
