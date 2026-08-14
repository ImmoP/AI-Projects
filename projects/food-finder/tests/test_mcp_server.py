"""Tests for the MCP adapter around the shared tool implementation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from mcp_server import search_restaurants


@pytest.mark.smoke
@patch("mcp_server.search_restaurants_tool")
def test_mcp_adapter_delegates_to_shared_tool(mock_tool) -> None:
    mock_tool.return_value = '[{"name": "Shared result"}]'

    result = search_restaurants(
        city="Munich",
        cuisine="Vietnamese",
        min_rating=4.3,
        min_reviews=80,
    )

    assert result == '[{"name": "Shared result"}]'
    mock_tool.assert_called_once_with(
        city="Munich",
        cuisine="Vietnamese",
        min_rating=4.3,
        min_reviews=80,
    )
