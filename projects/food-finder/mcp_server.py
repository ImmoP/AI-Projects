"""Expose Food Finder's existing restaurant search through MCP over stdio."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from src.agent.tools import search_restaurants as search_restaurants_tool

mcp = FastMCP(
    "food-finder",
    instructions="Search Google Places for well-reviewed restaurants.",
)


@mcp.tool(name="search_restaurants")
def search_restaurants(
    city: str,
    cuisine: str,
    min_rating: float = 4.0,
    min_reviews: int = 50,
) -> str:
    """Search and rank restaurants by city, cuisine, rating, and review count."""
    return search_restaurants_tool(
        city=city,
        cuisine=cuisine,
        min_rating=min_rating,
        min_reviews=min_reviews,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
