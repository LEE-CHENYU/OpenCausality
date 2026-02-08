"""
PaperScout Agent: Search literature for edge justification.

PaperScout finds citations that support or challenge proposed
DAG edges using multiple academic APIs: Semantic Scholar, OpenAlex,
arXiv, and CORE.  Results are deduplicated by DOI and enriched
with Open Access PDF URLs via Unpaywall.
"""

from __future__ import annotations

import json
import logging
import re
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
    source: str = "semantic_scholar"  # which API found this paper
    pdf_url: str = ""           # OA PDF URL

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
            "source": self.source,
            "pdf_url": self.pdf_url,
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
    using Semantic Scholar, OpenAlex, arXiv, and CORE APIs.
    Results are deduplicated by DOI and enriched via Unpaywall.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        s2_client: Any | None = None,
        s2_api_key: str | None = None,
        openalex_mailto: str | None = None,
        unpaywall_email: str | None = None,
        core_api_key: str | None = None,
    ):
        self.output_dir = output_dir or Path("outputs/agentic/citations")
        self.bundles: list[CitationBundle] = []
        self.search_status: dict[str, str] = {}  # edge_id -> status

        # Lazy-init clients via properties
        self._s2_client = s2_client
        self._s2_api_key = s2_api_key
        self._openalex_mailto = openalex_mailto
        self._unpaywall_email = unpaywall_email
        self._core_api_key = core_api_key

        # Cached client instances
        self._openalex_client: Any | None = None
        self._unpaywall_client: Any | None = None
        self._core_client: Any | None = None
        self._arxiv_client: Any | None = None

    # ------------------------------------------------------------------
    # Lazy-initialized client properties
    # ------------------------------------------------------------------

    @property
    def s2_client(self) -> Any:
        if self._s2_client is None:
            from shared.data.semantic_scholar import SemanticScholarClient
            self._s2_client = SemanticScholarClient(api_key=self._s2_api_key)
        return self._s2_client

    @property
    def openalex_client(self) -> Any | None:
        if self._openalex_client is None and self._openalex_mailto is not None:
            from shared.data.openalex import OpenAlexClient
            self._openalex_client = OpenAlexClient(mailto=self._openalex_mailto)
        # Also initialize without mailto (works, just slower)
        if self._openalex_client is None and self._openalex_mailto is None:
            from shared.data.openalex import OpenAlexClient
            self._openalex_client = OpenAlexClient()
        return self._openalex_client

    @property
    def unpaywall_client(self) -> Any | None:
        if self._unpaywall_client is None and self._unpaywall_email:
            from shared.data.unpaywall import UnpaywallClient
            self._unpaywall_client = UnpaywallClient(email=self._unpaywall_email)
        return self._unpaywall_client

    @property
    def core_client(self) -> Any | None:
        if self._core_client is None and self._core_api_key:
            from shared.data.core_api import COREClient
            self._core_client = COREClient(api_key=self._core_api_key)
        return self._core_client

    @property
    def arxiv_client(self) -> Any:
        if self._arxiv_client is None:
            from shared.data.arxiv import ArXivClient
            self._arxiv_client = ArXivClient()
        return self._arxiv_client

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
        per_source_limit: int = 3,
    ) -> CitationBundle:
        """
        Search multiple sources for papers relevant to a single edge.

        Args:
            edge_id: Edge identifier
            from_node_name: Treatment node human name
            from_node_desc: Treatment node description
            to_node_name: Outcome node human name
            to_node_desc: Outcome node description
            limit: Max total papers to return after dedup
            per_source_limit: Max papers per source before dedup

        Returns:
            CitationBundle for this edge
        """
        query = self._build_search_query(
            from_node_name, from_node_desc, to_node_name, to_node_desc,
        )
        logger.info(f"PaperScout: searching for '{query}' (edge={edge_id})")

        # Search all sources, collecting citations
        all_citations: list[Citation] = []
        all_citations += self._search_source_safe("semantic_scholar", query, edge_id, per_source_limit)
        all_citations += self._search_source_safe("openalex", query, edge_id, per_source_limit)
        all_citations += self._search_source_safe("core", query, edge_id, per_source_limit)
        all_citations += self._search_source_safe("arxiv", query, edge_id, per_source_limit)

        # Deduplicate by DOI, then by normalized title
        deduped = self._deduplicate(all_citations)

        # Enrich with Unpaywall PDF URLs
        enriched = self._enrich_with_unpaywall(deduped)

        # Sort by citation count (descending) and take top `limit`
        enriched.sort(key=lambda c: c.citation_count, reverse=True)
        final = enriched[:limit]

        bundle = CitationBundle(edge_id=edge_id, citations=final)
        self.bundles.append(bundle)
        self.search_status[edge_id] = "SEARCHED"

        sources_used = {c.source for c in final}
        logger.info(
            f"PaperScout: {len(final)} papers for {edge_id} "
            f"(sources={sources_used}, "
            f"supporting={sum(1 for c in final if c.relevance == 'supporting')}, "
            f"challenging={sum(1 for c in final if c.relevance == 'challenging')}, "
            f"methodological={sum(1 for c in final if c.relevance == 'methodological')})"
        )
        return bundle

    def search_and_extract(
        self,
        dag: Any,
        llm: Any,
        edge_id: str | None = None,
    ) -> list:
        """
        Search literature and extract causal claims as proposed DAG edges.

        Args:
            dag: DAGSpec instance
            llm: LLMClient instance
            edge_id: If specified, search for papers related to this edge only.
                     If None, search broadly for the DAG's target node.

        Returns:
            List of ProposedEdge from paper_dag_extractor
        """
        from shared.agentic.agents.paper_dag_extractor import PaperDAGExtractor

        # Gather citations
        if edge_id:
            node_map = {n.id: n for n in dag.nodes}
            edge = dag.get_edge(edge_id)
            if edge is None:
                logger.warning(f"Edge {edge_id} not found in DAG")
                return []
            from_node = node_map.get(edge.from_node)
            to_node = node_map.get(edge.to_node)
            if from_node is None or to_node is None:
                return []
            bundle = self.search_for_edge(
                edge_id=edge.id,
                from_node_name=from_node.name,
                from_node_desc=getattr(from_node, "description", ""),
                to_node_name=to_node.name,
                to_node_desc=getattr(to_node, "description", ""),
                limit=10,
            )
            papers = bundle.citations
        else:
            # Search broadly using DAG target node
            target_name = dag.metadata.target_node or ""
            target_node = dag.get_node(target_name)
            desc = getattr(target_node, "description", "") if target_node else ""
            target_label = getattr(target_node, "name", target_name) if target_node else target_name
            bundle = self.search_for_edge(
                edge_id="dag_broad_search",
                from_node_name="causal factors",
                from_node_desc="various economic variables",
                to_node_name=target_label,
                to_node_desc=desc,
                limit=10,
            )
            papers = bundle.citations

        # Try PDF extraction for papers with downloaded PDFs
        pdf_claims: list[dict] = []
        for paper in papers:
            if paper.pdf_url:
                # Check if we have a local PDF file matching this paper
                pdf_dir = self.output_dir / "pdfs"
                if pdf_dir.exists():
                    for pdf_file in pdf_dir.glob("*.pdf"):
                        if paper.doi and paper.doi.replace("/", "_") in pdf_file.stem:
                            pdf_claims.extend(self.extract_from_pdf(pdf_file))
                            break

        # Extract and propose
        extractor = PaperDAGExtractor(llm=llm, dag=dag)
        return extractor.propose_edges(papers)

    def extract_from_pdf(self, pdf_path: Path) -> list[dict]:
        """Extract causal claims from a downloaded PDF.

        Uses LLM-based extraction for richer results than abstract-only.
        Falls back gracefully if the LLM client or PDF reader is unavailable.

        Args:
            pdf_path: Path to a PDF file.

        Returns:
            List of causal claim dicts, or empty list on failure.
        """
        try:
            from shared.llm.pdf_extractor import extract_claims_from_pdf
            from shared.llm.client import get_llm_client

            client = get_llm_client()
            claims = extract_claims_from_pdf(pdf_path, client)
            logger.info(f"PaperScout: extracted {len(claims)} claims from PDF {pdf_path.name}")
            return claims
        except Exception as e:
            logger.warning(f"PaperScout: PDF extraction failed for {pdf_path}: {e}")
            return []

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
    # Multi-source dispatch
    # ------------------------------------------------------------------

    def _search_source_safe(
        self,
        source: str,
        query: str,
        edge_id: str,
        limit: int,
    ) -> list[Citation]:
        """Dispatch to source-specific search; catch all errors, return []."""
        try:
            if source == "semantic_scholar":
                return self._search_semantic_scholar(query, edge_id, limit)
            elif source == "openalex":
                return self._search_openalex(query, edge_id, limit)
            elif source == "core":
                return self._search_core(query, edge_id, limit)
            elif source == "arxiv":
                return self._search_arxiv(query, edge_id, limit)
            else:
                logger.warning(f"PaperScout: unknown source '{source}'")
                return []
        except Exception as e:
            logger.warning(f"PaperScout: {source} search failed for {edge_id}: {e}")
            return []

    def _search_semantic_scholar(
        self, query: str, edge_id: str, limit: int,
    ) -> list[Citation]:
        """Search Semantic Scholar and convert to Citations."""
        papers = self.s2_client.search_papers(query, limit=limit)
        citations = []
        for paper in papers:
            from_name = query.split()[0] if query else ""
            to_name = query.split()[-2] if len(query.split()) > 2 else ""
            relevance = self._categorize_citation(paper, from_name, to_name)
            citations.append(self._s2_paper_to_citation(paper, edge_id, relevance, query))
        return citations

    def _search_openalex(
        self, query: str, edge_id: str, limit: int,
    ) -> list[Citation]:
        """Search OpenAlex and convert to Citations."""
        client = self.openalex_client
        if client is None:
            return []
        works = client.search_works(query, limit=limit)
        citations = []
        for work in works:
            # Create a lightweight object with .abstract for _categorize_citation
            proxy = _AbstractProxy(work.abstract)
            from_name = query.split()[0] if query else ""
            to_name = query.split()[-2] if len(query.split()) > 2 else ""
            relevance = self._categorize_citation(proxy, from_name, to_name)
            citations.append(Citation(
                doi=work.doi or "",
                title=work.title,
                authors=work.authors,
                year=work.year,
                excerpt=(work.abstract or "")[:300],
                relevance=relevance,
                edge_id=edge_id,
                citation_count=work.cited_by_count,
                search_query=query,
                source="openalex",
                pdf_url=work.oa_url or "",
            ))
        return citations

    def _search_core(
        self, query: str, edge_id: str, limit: int,
    ) -> list[Citation]:
        """Search CORE and convert to Citations."""
        client = self.core_client
        if client is None:
            return []
        works = client.search_works(query, limit=limit)
        citations = []
        for work in works:
            proxy = _AbstractProxy(work.abstract)
            from_name = query.split()[0] if query else ""
            to_name = query.split()[-2] if len(query.split()) > 2 else ""
            relevance = self._categorize_citation(proxy, from_name, to_name)
            citations.append(Citation(
                doi=work.doi or "",
                title=work.title,
                authors=work.authors,
                year=work.year,
                excerpt=(work.abstract or "")[:300],
                relevance=relevance,
                edge_id=edge_id,
                search_query=query,
                source="core",
                pdf_url=work.download_url or "",
            ))
        return citations

    def _search_arxiv(
        self, query: str, edge_id: str, limit: int,
    ) -> list[Citation]:
        """Search arXiv and convert to Citations."""
        papers = self.arxiv_client.search_papers(query, limit=limit)
        citations = []
        for paper in papers:
            proxy = _AbstractProxy(paper.abstract)
            from_name = query.split()[0] if query else ""
            to_name = query.split()[-2] if len(query.split()) > 2 else ""
            relevance = self._categorize_citation(proxy, from_name, to_name)
            citations.append(Citation(
                doi=paper.doi or "",
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                excerpt=(paper.abstract or "")[:300],
                relevance=relevance,
                edge_id=edge_id,
                paper_id=paper.arxiv_id,
                search_query=query,
                source="arxiv",
                pdf_url=paper.pdf_url or "",
            ))
        return citations

    # ------------------------------------------------------------------
    # Deduplication and enrichment
    # ------------------------------------------------------------------

    def _deduplicate(self, citations: list[Citation]) -> list[Citation]:
        """
        Merge duplicates by DOI (keep highest citation_count, preserve pdf_url).
        For DOI-less papers, deduplicate by normalized title.
        """
        doi_map: dict[str, Citation] = {}
        title_map: dict[str, Citation] = {}

        for c in citations:
            if c.doi:
                existing = doi_map.get(c.doi)
                if existing is None:
                    doi_map[c.doi] = c
                else:
                    # Keep the one with higher citation count
                    if c.citation_count > existing.citation_count:
                        # Preserve pdf_url if the existing had one
                        if existing.pdf_url and not c.pdf_url:
                            c.pdf_url = existing.pdf_url
                        doi_map[c.doi] = c
                    elif c.pdf_url and not existing.pdf_url:
                        existing.pdf_url = c.pdf_url
            else:
                norm_title = self._normalize_title(c.title)
                if not norm_title:
                    continue
                existing = title_map.get(norm_title)
                if existing is None:
                    title_map[norm_title] = c
                else:
                    if c.citation_count > existing.citation_count:
                        if existing.pdf_url and not c.pdf_url:
                            c.pdf_url = existing.pdf_url
                        title_map[norm_title] = c
                    elif c.pdf_url and not existing.pdf_url:
                        existing.pdf_url = c.pdf_url

        # Combine: DOI-matched papers + title-only papers
        # But skip title-only if their title matches a DOI-matched paper
        doi_titles = {self._normalize_title(c.title) for c in doi_map.values()}
        result = list(doi_map.values())
        for norm_title, c in title_map.items():
            if norm_title not in doi_titles:
                result.append(c)

        return result

    def _enrich_with_unpaywall(self, citations: list[Citation]) -> list[Citation]:
        """
        For citations with DOI but no pdf_url, resolve via Unpaywall.
        Skip if no unpaywall_email configured.
        """
        client = self.unpaywall_client
        if client is None:
            return citations

        dois_to_resolve = [
            c.doi for c in citations
            if c.doi and not c.pdf_url
        ]
        if not dois_to_resolve:
            return citations

        logger.info(f"PaperScout: enriching {len(dois_to_resolve)} DOIs via Unpaywall")
        resolved = client.resolve_batch(dois_to_resolve)

        for c in citations:
            if c.doi and not c.pdf_url and c.doi in resolved:
                result = resolved[c.doi]
                if result.pdf_url:
                    c.pdf_url = result.pdf_url

        return citations

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize title for deduplication: lowercase, strip punctuation/spaces."""
        return re.sub(r"[^a-z0-9]", "", title.lower())

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
        """Build a search query from node names."""
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
            source="semantic_scholar",
        )


class _AbstractProxy:
    """Lightweight proxy so _categorize_citation can operate on any source."""

    def __init__(self, abstract: str | None):
        self.abstract = abstract
