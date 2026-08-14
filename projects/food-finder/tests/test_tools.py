"""Tests for the Google Places restaurant search tool."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

from src.agent.tools import search_restaurants


def _response(*, ok: bool = True, status_code: int = 200, payload: dict | None = None) -> Mock:
    response = Mock()
    response.ok = ok
    response.status_code = status_code
    response.text = "upstream error"
    response.json.return_value = payload or {"places": []}
    return response


@patch("src.agent.tools.requests.post")
def test_success(mock_post: Mock, monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-key")
    mock_post.return_value = _response(
        payload={
            "places": [
                {
                    "displayName": {"text": "Pasta House"},
                    "formattedAddress": "Main Street 1",
                    "rating": 4.7,
                    "userRatingCount": 320,
                    "priceLevel": "PRICE_LEVEL_MODERATE",
                    "primaryType": "italian_restaurant",
                    "googleMapsUri": "https://maps.example/pasta-house",
                }
            ]
        }
    )

    result = json.loads(
        search_restaurants(city="Frankfurt", cuisine="Italian", min_reviews=100)
    )

    assert result == [
        {
            "name": "Pasta House",
            "rating": 4.7,
            "reviews": 320,
            "address": "Main Street 1",
            "price_level": "PRICE_LEVEL_MODERATE",
            "type": "italian_restaurant",
            "google_maps": "https://maps.example/pasta-house",
        }
    ]
    request = mock_post.call_args
    assert request.kwargs["json"]["textQuery"] == "Italian restaurants in Frankfurt"
    assert request.kwargs["timeout"] == 20


@patch("src.agent.tools.requests.post")
def test_missing_api_key(mock_post: Mock, monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)

    result = json.loads(search_restaurants(city="Berlin", cuisine="Japanese"))

    assert result == {
        "ok": False,
        "error": {
            "code": "missing_api_key",
            "message": "GOOGLE_PLACES_API_KEY is not configured.",
        },
    }
    mock_post.assert_not_called()


@patch("src.agent.tools.requests.post")
def test_http_error(mock_post: Mock, monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-key")
    mock_post.return_value = _response(ok=False, status_code=429)

    result = json.loads(search_restaurants(city="Hamburg", cuisine="Thai"))

    assert result["ok"] is False
    assert result["error"] == {
        "code": "places_api_error",
        "message": "Google Places API returned an error.",
        "status_code": 429,
        "details": "upstream error",
    }


@patch("src.agent.tools.requests.post")
def test_filters_results_below_min_reviews(mock_post: Mock, monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-key")
    mock_post.return_value = _response(
        payload={
            "places": [
                {
                    "displayName": {"text": "Popular Place"},
                    "rating": 4.6,
                    "userRatingCount": 250,
                },
                {
                    "displayName": {"text": "New Place"},
                    "rating": 5.0,
                    "userRatingCount": 12,
                },
            ]
        }
    )

    result = json.loads(
        search_restaurants(city="Cologne", cuisine="Greek", min_reviews=100)
    )

    assert [restaurant["name"] for restaurant in result] == ["Popular Place"]
