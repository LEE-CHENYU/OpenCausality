"""
Semantic Scholar Graph API client.

Free tier (~100 requests per 5 minutes, no API key required).
Uses httpx for HTTP requests.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEARCH_FIELDS = "paperId,title,authors,year,abstract,citationCount,externalIds"
PAPER_FIELDS = SEARCH_FIELDS


@dataclass
class S2Author:
    """Semantic Scholar author."""

    name: str
    author_id: str = ""


@dataclass
class S2Paper:
    """Semantic Scholar paper."""

    paper_id: str
    title: str
    authors: list[S2Author]
    year: int | None
    abstract: str | None
    citation_count: int
    doi: str | None
    external_ids: dict[str, str] = field(default_factory=dict)


class SemanticScholarClient:
    """
    Client for the Semantic Scholar Graph API.

    Free tier: ~100 requests per 5 minutes.
    We enforce a 3-second pause between requests to stay well within limits.
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key
        self._last_request_time: float = 0.0
        self._min_interval: float = 3.0  # seconds between requests

    def search_papers(
        self,
        query: str,
        limit: int = 5,
        year_range: str | None = None,
    ) -> list[S2Paper]:
        """
        Search for papers by keyword query.

        Args:
            query: Search query string
            limit: Maximum number of results (max 100)
            year_range: Optional year filter, e.g. "2010-2024"

        Returns:
            List of S2Paper results
        """
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": SEARCH_FIELDS,
        }
        if year_range:
            params["year"] = year_range

        data = self._request(f"{BASE_URL}/paper/search", params)
        if data is None:
            return []

        papers = []
        for item in data.get("data", []):
            try:
                papers.append(self._parse_paper(item))
            except Exception as e:
                logger.debug(f"Failed to parse paper: {e}")
                continue

        return papers

    def get_paper(self, paper_id: str) -> S2Paper | None:
        """
        Get a single paper by Semantic Scholar paper ID or DOI.

        Args:
            paper_id: S2 paper ID or DOI (e.g. "DOI:10.1234/...")

        Returns:
            S2Paper or None if not found
        """
        params = {"fields": PAPER_FIELDS}
        data = self._request(f"{BASE_URL}/paper/{paper_id}", params)
        if data is None:
            return None
        try:
            return self._parse_paper(data)
        except Exception as e:
            logger.warning(f"Failed to parse paper {paper_id}: {e}")
            return None

    def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _request(self, url: str, params: dict[str, Any]) -> dict | None:
        """
        Make a GET request with rate limiting and error handling.

        Returns:
            Parsed JSON dict or None on failure
        """
        self._rate_limit()

        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        try:
            self._last_request_time = time.monotonic()
            resp = httpx.get(url, params=params, headers=headers, timeout=30.0)

            if resp.status_code == 429:
                logger.warning("Semantic Scholar rate limit hit; backing off 10s")
                time.sleep(10.0)
                return self._request(url, params)  # retry once

            resp.raise_for_status()
            return resp.json()

        except httpx.HTTPStatusError as e:
            logger.warning(f"Semantic Scholar HTTP error: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"Semantic Scholar request error: {e}")
            return None

    def _parse_paper(self, data: dict) -> S2Paper:
        """Parse API response dict into S2Paper."""
        authors = []
        for a in data.get("authors", []) or []:
            authors.append(S2Author(
                name=a.get("name", ""),
                author_id=str(a.get("authorId", "")),
            ))

        ext_ids = data.get("externalIds") or {}
        doi = ext_ids.get("DOI") or data.get("doi")

        return S2Paper(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            authors=authors,
            year=data.get("year"),
            abstract=data.get("abstract"),
            citation_count=data.get("citationCount", 0) or 0,
            doi=doi,
            external_ids=ext_ids,
        )
