"""
CORE v3 API client.

Requires a free API key from https://core.ac.uk/services/api.
Aggregates institutional repositories and OA content.

Docs: https://api.core.ac.uk/docs/v3
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.core.ac.uk"


@dataclass
class COREWork:
    """A single work from the CORE API."""

    core_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    download_url: str | None = None


class COREClient:
    """
    Client for the CORE v3 API.

    Auth via Bearer token. Rate limits vary by plan; we enforce
    a 2.0s pause between requests for the free tier.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._last_request_time: float = 0.0
        self._min_interval: float = 2.0

    def search_works(self, query: str, limit: int = 5) -> list[COREWork]:
        """
        Search CORE works by keyword query.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of COREWork results
        """
        params: dict[str, Any] = {
            "q": query,
            "limit": min(limit, 50),
        }

        data = self._request(f"{BASE_URL}/v3/search/works", params)
        if data is None:
            return []

        works = []
        for item in data.get("results", []) or []:
            try:
                works.append(self._parse_work(item))
            except Exception as e:
                logger.debug(f"Failed to parse CORE work: {e}")
                continue

        return works

    def _request(self, url: str, params: dict[str, Any]) -> dict | None:
        """Make a GET request with rate limiting and Bearer auth."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            self._last_request_time = time.monotonic()
            resp = httpx.get(url, params=params, headers=headers, timeout=30.0, follow_redirects=True)

            if resp.status_code == 429:
                logger.warning("CORE rate limit hit; backing off 10s")
                time.sleep(10.0)
                return self._request(url, params)

            resp.raise_for_status()
            return resp.json()

        except httpx.HTTPStatusError as e:
            logger.warning(f"CORE HTTP error: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"CORE request error: {e}")
            return None

    def _parse_work(self, data: dict) -> COREWork:
        """Parse a CORE API response dict into COREWork."""
        # Extract authors
        authors = []
        for author in data.get("authors", []) or []:
            if isinstance(author, dict):
                name = author.get("name", "")
            else:
                name = str(author)
            if name:
                authors.append(name)

        # Extract year from yearPublished
        year = data.get("yearPublished")
        if year is not None:
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = None

        return COREWork(
            core_id=str(data.get("id", "")),
            title=data.get("title", ""),
            authors=authors,
            year=year,
            abstract=data.get("abstract"),
            doi=data.get("doi"),
            download_url=data.get("downloadUrl"),
        )
