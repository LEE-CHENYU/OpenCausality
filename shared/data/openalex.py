"""
OpenAlex Works API client.

No authentication required. Providing a `mailto` email enables the
polite pool (faster rate limits).

Docs: https://docs.openalex.org/api-entities/works
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"


@dataclass
class OpenAlexWork:
    """A single work from the OpenAlex API."""

    openalex_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    cited_by_count: int = 0
    doi: str | None = None
    oa_url: str | None = None


class OpenAlexClient:
    """
    Client for the OpenAlex Works API.

    Polite pool: include `mailto` for faster responses (~10 req/s).
    Without it, the limit is ~1 req/s.
    We enforce a 0.5s pause between requests to be safe.
    """

    def __init__(self, mailto: str | None = None):
        self._mailto = mailto
        self._last_request_time: float = 0.0
        self._min_interval: float = 0.5

    def search_works(self, query: str, limit: int = 5) -> list[OpenAlexWork]:
        """
        Search OpenAlex works by keyword query.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of OpenAlexWork results
        """
        params: dict[str, Any] = {
            "search": query,
            "per_page": min(limit, 50),
        }
        if self._mailto:
            params["mailto"] = self._mailto

        data = self._request(f"{BASE_URL}/works", params)
        if data is None:
            return []

        works = []
        for item in data.get("results", []):
            try:
                works.append(self._parse_work(item))
            except Exception as e:
                logger.debug(f"Failed to parse OpenAlex work: {e}")
                continue

        return works

    def _request(self, url: str, params: dict[str, Any]) -> dict | None:
        """Make a GET request with rate limiting."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        try:
            self._last_request_time = time.monotonic()
            resp = httpx.get(url, params=params, timeout=30.0)

            if resp.status_code == 429:
                logger.warning("OpenAlex rate limit hit; backing off 5s")
                time.sleep(5.0)
                return self._request(url, params)

            resp.raise_for_status()
            return resp.json()

        except httpx.HTTPStatusError as e:
            logger.warning(f"OpenAlex HTTP error: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"OpenAlex request error: {e}")
            return None

    def _parse_work(self, data: dict) -> OpenAlexWork:
        """Parse an OpenAlex API response dict into OpenAlexWork."""
        # Extract author names
        authors = []
        for authorship in data.get("authorships", []) or []:
            author = authorship.get("author", {}) or {}
            name = author.get("display_name", "")
            if name:
                authors.append(name)

        # Extract DOI (strip https://doi.org/ prefix)
        doi_raw = data.get("doi") or ""
        doi = doi_raw.replace("https://doi.org/", "").strip() or None

        # Extract OA URL
        oa_url = None
        best_oa = data.get("best_oa_location") or {}
        if best_oa:
            oa_url = best_oa.get("pdf_url") or best_oa.get("landing_page_url")

        # Reconstruct abstract from inverted index
        abstract = self._reconstruct_abstract(data.get("abstract_inverted_index"))

        return OpenAlexWork(
            openalex_id=data.get("id", ""),
            title=data.get("title", ""),
            authors=authors,
            year=data.get("publication_year"),
            abstract=abstract,
            cited_by_count=data.get("cited_by_count", 0) or 0,
            doi=doi,
            oa_url=oa_url,
        )

    def _reconstruct_abstract(self, inverted_index: dict | None) -> str | None:
        """
        Reconstruct plain-text abstract from OpenAlex inverted index.

        The inverted index maps each word to a list of positions:
        {"word": [0, 5], "another": [1]} -> "word another ... word"
        """
        if not inverted_index:
            return None

        # Build position -> word mapping
        positions: dict[int, str] = {}
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions[pos] = word

        if not positions:
            return None

        # Reconstruct in order
        max_pos = max(positions.keys())
        words = [positions.get(i, "") for i in range(max_pos + 1)]
        return " ".join(w for w in words if w)
