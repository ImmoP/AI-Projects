"""Tools used by the Food Finder agent."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv
from smolagents import tool

load_dotenv()

LOGGER = logging.getLogger(__name__)
PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_FIELD_MASK = ",".join(
    (
        "places.displayName",
        "places.formattedAddress",
        "places.rating",
        "places.userRatingCount",
        "places.priceLevel",
        "places.googleMapsUri",
        "places.primaryType",
    )
)


def _error_response(
    code: str,
    message: str,
    *,
    status_code: int | None = None,
    details: str | None = None,
) -> str:
    """Return every tool error with the same JSON envelope."""
    error: dict[str, Any] = {"code": code, "message": message}
    if status_code is not None:
        error["status_code"] = status_code
    if details:
        error["details"] = details
    return json.dumps({"ok": False, "error": error}, ensure_ascii=False, indent=2)


@tool
def search_restaurants(
    city: str,
    cuisine: str,
    min_rating: float = 4.0,
    min_reviews: int = 50,
) -> str:
    """Search for highly rated restaurants of a specific cuisine in a city.

    Restaurants are filtered by minimum rating and minimum number of reviews,
    then sorted by rating and review count in descending order.

    Args:
        city: City in which to search, for example "Frankfurt am Main".
        cuisine: Type of cuisine, for example "Italian", "Greek", or "Japanese".
        min_rating: Minimum Google rating from 0.0 to 5.0.
        min_reviews: Minimum number of Google user reviews required.

    Returns:
        A JSON string containing matching restaurants or a structured error.
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        LOGGER.error("Google Places API key is not configured")
        return _error_response(
            "missing_api_key",
            "GOOGLE_PLACES_API_KEY is not configured.",
        )

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": PLACES_FIELD_MASK,
    }
    body = {
        "textQuery": f"{cuisine} restaurants in {city}",
        "includedType": "restaurant",
        "minRating": min_rating,
        "pageSize": 20,
        "languageCode": "de",
    }

    LOGGER.info(
        "Searching restaurants in %s (cuisine=%s, min_rating=%s, min_reviews=%s)",
        city,
        cuisine,
        min_rating,
        min_reviews,
    )
    try:
        response = requests.post(
            PLACES_TEXT_SEARCH_URL,
            headers=headers,
            json=body,
            timeout=20,
        )
    except requests.RequestException as exc:
        LOGGER.exception("Request to Google Places failed")
        return _error_response(
            "request_failed",
            "Request to Google Places failed.",
            details=str(exc),
        )

    LOGGER.debug("Google Places returned HTTP %s", response.status_code)
    if not response.ok:
        LOGGER.error("Google Places API error (%s): %s", response.status_code, response.text)
        return _error_response(
            "places_api_error",
            "Google Places API returned an error.",
            status_code=response.status_code,
            details=response.text,
        )

    try:
        places = response.json().get("places", [])
    except (requests.JSONDecodeError, AttributeError) as exc:
        LOGGER.exception("Google Places returned invalid JSON")
        return _error_response(
            "invalid_response",
            "Google Places API returned an invalid response.",
            details=str(exc),
        )

    restaurants = []
    for place in places:
        rating = place.get("rating", 0)
        review_count = place.get("userRatingCount", 0)
        if review_count < min_reviews:
            continue

        restaurants.append(
            {
                "name": place.get("displayName", {}).get("text"),
                "rating": rating,
                "reviews": review_count,
                "address": place.get("formattedAddress"),
                "price_level": place.get("priceLevel"),
                "type": place.get("primaryType"),
                "google_maps": place.get("googleMapsUri"),
            }
        )

    restaurants.sort(
        key=lambda restaurant: (restaurant["rating"], restaurant["reviews"]),
        reverse=True,
    )
    LOGGER.info("Returning %d restaurant matches", len(restaurants))
    return json.dumps(restaurants, ensure_ascii=False, indent=2)
