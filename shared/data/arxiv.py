"""
arXiv API client.

No authentication required. Returns Atom XML.
Uses stdlib xml.etree.ElementTree for parsing (no extra dependencies).

Docs: https://info.arxiv.org/help/api/index.html
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "http://export.arxiv.org/api"

# Atom / arXiv namespaces
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass
class ArXivPaper:
    """A single paper from the arXiv API."""

    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    pdf_url: str | None = None
    categories: list[str] = field(default_factory=list)


class ArXivClient:
    """
    Client for the arXiv API.

    Rate limit guidance: ~1 request per 3 seconds.
    We enforce a 1.5s pause between requests.
    """

    def __init__(self):
        self._last_request_time: float = 0.0
        self._min_interval: float = 1.5

    def search_papers(self, query: str, limit: int = 5) -> list[ArXivPaper]:
        """
        Search arXiv for papers matching a query.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of ArXivPaper results
        """
        params: dict[str, Any] = {
            "search_query": f"all:{query}",
            "max_results": min(limit, 50),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        xml_text = self._request(f"{BASE_URL}/query", params)
        if xml_text is None:
            return []

        return self._parse_feed(xml_text)

    def _request(self, url: str, params: dict[str, Any]) -> str | None:
        """Make a GET request with rate limiting. Returns raw XML text."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        try:
            self._last_request_time = time.monotonic()
            resp = httpx.get(url, params=params, timeout=30.0)

            if resp.status_code == 429:
                logger.warning("arXiv rate limit hit; backing off 10s")
                time.sleep(10.0)
                return self._request(url, params)

            resp.raise_for_status()
            return resp.text

        except httpx.HTTPStatusError as e:
            logger.warning(f"arXiv HTTP error: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"arXiv request error: {e}")
            return None

    def _parse_feed(self, xml_text: str) -> list[ArXivPaper]:
        """Parse Atom XML feed into a list of ArXivPaper."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning(f"arXiv XML parse error: {e}")
            return []

        papers = []
        for entry in root.findall("atom:entry", NS):
            try:
                papers.append(self._parse_entry(entry))
            except Exception as e:
                logger.debug(f"Failed to parse arXiv entry: {e}")
                continue

        return papers

    def _parse_entry(self, entry: ET.Element) -> ArXivPaper:
        """Parse a single Atom entry into ArXivPaper."""
        # ID: e.g. "http://arxiv.org/abs/2301.12345v1"
        raw_id = (entry.findtext("atom:id", "", NS) or "").strip()
        arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

        # Title (may have newlines)
        title = (entry.findtext("atom:title", "", NS) or "").strip()
        title = " ".join(title.split())  # normalize whitespace

        # Authors
        authors = []
        for author_el in entry.findall("atom:author", NS):
            name = (author_el.findtext("atom:name", "", NS) or "").strip()
            if name:
                authors.append(name)

        # Abstract / summary
        abstract = (entry.findtext("atom:summary", "", NS) or "").strip()
        abstract = " ".join(abstract.split()) if abstract else None

        # Published year
        published = entry.findtext("atom:published", "", NS) or ""
        year = None
        if published and len(published) >= 4:
            try:
                year = int(published[:4])
            except ValueError:
                pass

        # DOI (from arxiv namespace)
        doi = None
        doi_el = entry.find("arxiv:doi", NS)
        if doi_el is not None and doi_el.text:
            doi = doi_el.text.strip()

        # PDF URL from links
        pdf_url = None
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf" or (
                link.get("type") == "application/pdf"
            ):
                pdf_url = link.get("href")
                break
        # Fallback: construct from arxiv_id
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        # Categories
        categories = []
        for cat in entry.findall("atom:category", NS):
            term = cat.get("term", "")
            if term:
                categories.append(term)
        # Also check arxiv:primary_category
        prim_cat = entry.find("arxiv:primary_category", NS)
        if prim_cat is not None:
            term = prim_cat.get("term", "")
            if term and term not in categories:
                categories.insert(0, term)

        return ArXivPaper(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            doi=doi,
            pdf_url=pdf_url,
            categories=categories,
        )
