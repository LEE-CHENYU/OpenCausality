"""
PaperScout Agent: Search literature for edge justification.

PaperScout finds citations that support or challenge proposed
DAG edges using the Semantic Scholar API.
Output goes to the configured citations directory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A single citation supporting or challenging an edge."""

    doi: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    excerpt: str = ""
    relevance: str = ""  # "supporting", "challenging", "methodological"
    edge_id: str = ""
    paper_id: str = ""          # Semantic Scholar paper ID
    citation_count: int = 0     # for relevance weighting
    search_query: str = ""      # query that found this paper

    def to_dict(self) -> dict[str, Any]:
        return {
            "doi": self.doi,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "excerpt": self.excerpt,
            "relevance": self.relevance,
            "edge_id": self.edge_id,
            "paper_id": self.paper_id,
            "citation_count": self.citation_count,
            "search_query": self.search_query,
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
    PaperScout agent: finds literature support for DAG edges
    using the Semantic Scholar Graph API.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        s2_client: Any | None = None,
    ):
        self.output_dir = output_dir or Path("outputs/agentic/citations")
        self.bundles: list[CitationBundle] = []
        self.search_status: dict[str, str] = {}  # edge_id -> status

        # Lazy-init Semantic Scholar client
        self._s2_client = s2_client

    @property
    def s2_client(self) -> Any:
        if self._s2_client is None:
            from shared.data.semantic_scholar import SemanticScholarClient
            self._s2_client = SemanticScholarClient()
        return self._s2_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_bundle(self, bundle: CitationBundle) -> None:
        """Add a citation bundle (legacy compatibility)."""
        self.bundles.append(bundle)
        logger.info(
            f"PaperScout: {len(bundle.citations)} citations for {bundle.edge_id}"
        )

    def search_for_edge(
        self,
        edge_id: str,
        from_node_name: str,
        from_node_desc: str,
        to_node_name: str,
        to_node_desc: str,
        limit: int = 5,
    ) -> CitationBundle:
        """
        Search Semantic Scholar for papers relevant to a single edge.

        Args:
            edge_id: Edge identifier
            from_node_name: Treatment node human name
            from_node_desc: Treatment node description (unused for now)
            to_node_name: Outcome node human name
            to_node_desc: Outcome node description (unused for now)
            limit: Max papers to fetch

        Returns:
            CitationBundle for this edge
        """
        query = self._build_search_query(
            from_node_name, from_node_desc, to_node_name, to_node_desc,
        )
        logger.info(f"PaperScout: searching for '{query}' (edge={edge_id})")

        try:
            papers = self.s2_client.search_papers(query, limit=limit)
        except Exception as e:
            logger.warning(f"PaperScout: search failed for {edge_id}: {e}")
            self.search_status[edge_id] = "FAILED"
            return CitationBundle(edge_id=edge_id)

        citations: list[Citation] = []
        for paper in papers:
            relevance = self._categorize_citation(paper, from_node_name, to_node_name)
            citation = self._s2_paper_to_citation(paper, edge_id, relevance, query)
            citations.append(citation)

        bundle = CitationBundle(edge_id=edge_id, citations=citations)
        self.bundles.append(bundle)
        self.search_status[edge_id] = "SEARCHED"

        logger.info(
            f"PaperScout: {len(citations)} papers for {edge_id} "
            f"(supporting={sum(1 for c in citations if c.relevance == 'supporting')}, "
            f"challenging={sum(1 for c in citations if c.relevance == 'challenging')}, "
            f"methodological={sum(1 for c in citations if c.relevance == 'methodological')})"
        )
        return bundle

    def search_all_edges(self, dag: Any) -> dict[str, CitationBundle]:
        """
        Search literature for all edges in a DAG.

        Args:
            dag: DAGSpec with edges and nodes

        Returns:
            dict mapping edge_id -> CitationBundle
        """
        node_map = {n.id: n for n in dag.nodes}
        results: dict[str, CitationBundle] = {}

        for edge in dag.edges:
            from_node = node_map.get(edge.from_node)
            to_node = node_map.get(edge.to_node)

            if from_node is None or to_node is None:
                self.search_status[edge.id] = "SKIPPED"
                continue

            bundle = self.search_for_edge(
                edge_id=edge.id,
                from_node_name=from_node.name,
                from_node_desc=getattr(from_node, "description", ""),
                to_node_name=to_node.name,
                to_node_desc=getattr(to_node, "description", ""),
            )
            results[edge.id] = bundle

        return results

    def save_bundles(self) -> Path:
        """Save all citation bundles to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "citation_bundle.json"

        data = [b.to_dict() for b in self.bundles]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"PaperScout: saved {len(self.bundles)} bundles to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_search_query(
        self,
        from_name: str,
        from_desc: str,
        to_name: str,
        to_desc: str,
    ) -> str:
        """Build a Semantic Scholar search query from node names."""
        # Convert underscores to spaces for readability
        treatment = from_name.replace("_", " ")
        outcome = to_name.replace("_", " ")
        return f"{treatment} {outcome} causal effect"

    def _categorize_citation(
        self,
        paper: Any,
        from_name: str,
        to_name: str,
    ) -> str:
        """
        Categorize a paper as supporting, challenging, or methodological.

        Uses keyword heuristics on the abstract.
        """
        abstract = (paper.abstract or "").lower()

        # Challenging: evidence against the relationship
        challenging_keywords = [
            "no significant", "no evidence", "insignificant",
            "fails to", "cannot reject", "no effect",
            "no relationship", "no causal",
        ]
        for kw in challenging_keywords:
            if kw in abstract:
                return "challenging"

        # Methodological: about methods rather than the specific relationship
        method_keywords = [
            "local projections", "impulse response", "structural var",
            "identification strategy", "instrumental variable",
            "difference-in-differences", "regression discontinuity",
            "bayesian", "monte carlo", "simulation",
        ]
        for kw in method_keywords:
            if kw in abstract:
                return "methodological"

        # Default: supporting
        return "supporting"

    def _s2_paper_to_citation(
        self,
        paper: Any,
        edge_id: str,
        relevance: str,
        query: str,
    ) -> Citation:
        """Convert an S2Paper to a Citation dataclass."""
        return Citation(
            doi=paper.doi or "",
            title=paper.title,
            authors=[a.name for a in paper.authors],
            year=paper.year,
            excerpt=(paper.abstract or "")[:300],
            relevance=relevance,
            edge_id=edge_id,
            paper_id=paper.paper_id,
            citation_count=paper.citation_count,
            search_query=query,
        )
