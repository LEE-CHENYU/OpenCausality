"""
Unpaywall DOI -> Open Access PDF resolver.

Requires an email address for API access (free, no key needed).

Docs: https://unpaywall.org/products/api
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.unpaywall.org/v2"


@dataclass
class UnpaywallResult:
    """Result of an Unpaywall DOI lookup."""

    doi: str
    is_oa: bool
    pdf_url: str | None = None
    oa_status: str = "closed"  # gold, green, hybrid, bronze, closed


class UnpaywallClient:
    """
    Client for the Unpaywall REST API.

    Rate limit: ~100k requests/day. We enforce 0.2s between requests.
    """

    def __init__(self, email: str):
        self._email = email
        self._last_request_time: float = 0.0
        self._min_interval: float = 0.2

    def resolve_doi(self, doi: str) -> UnpaywallResult | None:
        """
        Resolve a single DOI to its Open Access status.

        Args:
            doi: DOI string (e.g. "10.1016/j.eneco.2023.107033")

        Returns:
            UnpaywallResult or None if lookup failed
        """
        if not doi:
            return None

        params: dict[str, Any] = {"email": self._email}
        data = self._request(f"{BASE_URL}/{doi}", params)
        if data is None:
            return None

        # Extract best OA PDF URL
        pdf_url = None
        best_oa = data.get("best_oa_location") or {}
        if best_oa:
            pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url")

        return UnpaywallResult(
            doi=doi,
            is_oa=data.get("is_oa", False),
            pdf_url=pdf_url,
            oa_status=data.get("oa_status", "closed") or "closed",
        )

    def resolve_batch(self, dois: list[str]) -> dict[str, UnpaywallResult]:
        """
        Resolve multiple DOIs to their Open Access status.

        Args:
            dois: List of DOI strings

        Returns:
            Dict mapping DOI -> UnpaywallResult (only successful lookups)
        """
        results: dict[str, UnpaywallResult] = {}
        for doi in dois:
            result = self.resolve_doi(doi)
            if result is not None:
                results[doi] = result
        return results

    def _request(self, url: str, params: dict[str, Any]) -> dict | None:
        """Make a GET request with rate limiting."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        try:
            self._last_request_time = time.monotonic()
            resp = httpx.get(url, params=params, timeout=30.0)

            if resp.status_code == 404:
                logger.debug(f"Unpaywall: DOI not found in {url}")
                return None

            if resp.status_code == 429:
                logger.warning("Unpaywall rate limit hit; backing off 5s")
                time.sleep(5.0)
                return self._request(url, params)

            resp.raise_for_status()
            return resp.json()

        except httpx.HTTPStatusError as e:
            logger.warning(f"Unpaywall HTTP error: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"Unpaywall request error: {e}")
            return None
