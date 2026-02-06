"""
PaperScout Agent: Search literature for edge justification.

PaperScout finds citations that support or challenge proposed
DAG edges. Output goes to specs/candidates/citations/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A single citation supporting or challenging an edge."""

    doi: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    excerpt: str = ""
    relevance: str = ""  # "supports", "challenges", "contextual"
    edge_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "doi": self.doi,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "excerpt": self.excerpt,
            "relevance": self.relevance,
            "edge_id": self.edge_id,
        }


@dataclass
class CitationBundle:
    """Collection of citations for an edge or set of edges."""

    edge_id: str
    citations: list[Citation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "citations": [c.to_dict() for c in self.citations],
        }


class PaperScout:
    """
    PaperScout agent: finds literature support for DAG edges.

    Output goes to quarantined location (specs/candidates/citations/).
    """

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("specs/candidates/citations")
        self.bundles: list[CitationBundle] = []

    def add_bundle(self, bundle: CitationBundle) -> None:
        """Add a citation bundle."""
        self.bundles.append(bundle)
        logger.info(
            f"PaperScout: {len(bundle.citations)} citations for {bundle.edge_id}"
        )

    def save_bundles(self) -> Path:
        """Save all citation bundles to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "citation_bundle.json"

        data = [b.to_dict() for b in self.bundles]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path
