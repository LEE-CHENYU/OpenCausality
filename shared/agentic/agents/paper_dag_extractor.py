"""
Paper-to-DAG Extraction Pipeline.

Uses LLM to extract causal claims from papers and propose DAG edges.

Pipeline:
    PaperScout.search_for_edge() -> papers with abstracts
        |
    PaperDAGExtractor.extract_causal_claims(papers) -> CausalClaim[]
        |
    PaperDAGExtractor.propose_edges(claims, existing_dag) -> ProposedEdge[]
        |
    User reviews in REPL -> accept/reject/modify
        |
    Accepted edges added to DAGSpec
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────


@dataclass
class CausalClaim:
    """A single causal claim extracted from a paper."""

    paper_title: str = ""
    paper_doi: str = ""
    treatment: str = ""
    outcome: str = ""
    mechanism: str = ""
    direction: str = ""  # positive / negative / ambiguous
    identification: str = ""  # IV, RCT, DiD, etc.
    confidence: str = ""  # high / medium / low
    quote: str = ""
    edge_type_suggestion: str = "causal"  # causal / immutable / mechanical


@dataclass
class ProposedEdge:
    """A proposed DAG edge derived from literature evidence."""

    from_node: str = ""
    to_node: str = ""
    edge_id: str = ""
    edge_type: str = "causal"
    evidence: list[CausalClaim] = field(default_factory=list)
    requires_new_nodes: list[str] = field(default_factory=list)
    match_confidence: float = 0.0
    is_existing: bool = False  # True if matched to existing DAG edge

    def to_dict(self) -> dict[str, Any]:
        provenance: dict[str, Any] = {
            "source": "paper_scout",
            "added_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        if self.evidence:
            provenance["paper_doi"] = self.evidence[0].paper_doi
            provenance["paper_title"] = self.evidence[0].paper_title
        provenance["match_confidence"] = self.match_confidence

        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "edge_id": self.edge_id,
            "edge_type": self.edge_type,
            "evidence_count": len(self.evidence),
            "requires_new_nodes": self.requires_new_nodes,
            "match_confidence": self.match_confidence,
            "is_existing": self.is_existing,
            "provenance": provenance,
        }


# ──────────────────────────────────────────────────────────────────────
# Extraction schemas
# ──────────────────────────────────────────────────────────────────────

CLAIM_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "treatment": {"type": "string"},
                    "outcome": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "direction": {"type": "string", "enum": ["positive", "negative", "ambiguous"]},
                    "identification": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "quote": {"type": "string"},
                    "edge_type_suggestion": {
                        "type": "string",
                        "enum": ["causal", "immutable", "mechanical"],
                    },
                },
                "required": ["treatment", "outcome", "direction"],
            },
        },
    },
    "required": ["claims"],
}

NODE_MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "matched_from_node": {"type": "string"},
        "matched_to_node": {"type": "string"},
        "requires_new_nodes": {"type": "array", "items": {"type": "string"}},
        "match_confidence": {"type": "number"},
        "edge_id_suggestion": {"type": "string"},
    },
    "required": ["matched_from_node", "matched_to_node", "match_confidence"],
}


# ──────────────────────────────────────────────────────────────────────
# PaperDAGExtractor
# ──────────────────────────────────────────────────────────────────────


class PaperDAGExtractor:
    """
    Extract causal claims from papers and propose DAG edges.

    Uses LLM for claim extraction and node matching.
    """

    def __init__(self, llm: Any, dag: Any):
        """
        Args:
            llm: LLMClient instance
            dag: DAGSpec instance
        """
        self.llm = llm
        self.dag = dag
        self._node_map = {n.id: n for n in dag.nodes}

    def extract_causal_claims(self, papers: list[Any]) -> list[CausalClaim]:
        """
        Extract causal claims from a list of papers (Citation objects).

        Args:
            papers: List of Citation dataclass instances with title, authors, excerpt, doi

        Returns:
            List of CausalClaim extracted from all papers
        """
        from shared.llm.prompts import (
            CAUSAL_CLAIM_EXTRACTION_SYSTEM,
            CAUSAL_CLAIM_EXTRACTION_USER,
        )

        all_claims: list[CausalClaim] = []

        for paper in papers:
            title = getattr(paper, "title", "")
            authors = getattr(paper, "authors", [])
            year = getattr(paper, "year", "")
            abstract = getattr(paper, "excerpt", "") or getattr(paper, "abstract", "")
            doi = getattr(paper, "doi", "")

            if not abstract:
                continue

            user_prompt = CAUSAL_CLAIM_EXTRACTION_USER.format(
                title=title,
                authors=", ".join(authors[:5]) if isinstance(authors, list) else str(authors),
                year=year,
                abstract=abstract,
            )

            try:
                result = self.llm.complete_structured(
                    CAUSAL_CLAIM_EXTRACTION_SYSTEM,
                    user_prompt,
                    CLAIM_EXTRACTION_SCHEMA,
                )
                for claim_data in result.get("claims", []):
                    claim = CausalClaim(
                        paper_title=title,
                        paper_doi=doi,
                        treatment=claim_data.get("treatment", ""),
                        outcome=claim_data.get("outcome", ""),
                        mechanism=claim_data.get("mechanism", ""),
                        direction=claim_data.get("direction", "ambiguous"),
                        identification=claim_data.get("identification", ""),
                        confidence=claim_data.get("confidence", "low"),
                        quote=claim_data.get("quote", ""),
                        edge_type_suggestion=claim_data.get("edge_type_suggestion", "causal"),
                    )
                    all_claims.append(claim)
                    logger.info(
                        f"Extracted claim: {claim.treatment} -> {claim.outcome} "
                        f"({claim.direction}, {claim.confidence})"
                    )
            except Exception as e:
                logger.warning(f"Failed to extract claims from '{title}': {e}")

        return all_claims

    def match_to_dag(self, claims: list[CausalClaim]) -> list[ProposedEdge]:
        """
        Match extracted claims to existing DAG nodes and propose edges.

        Args:
            claims: List of CausalClaim to match

        Returns:
            List of ProposedEdge with matched nodes
        """
        from shared.llm.prompts import NODE_MATCHING_SYSTEM, NODE_MATCHING_USER

        # Build node descriptions for the prompt
        node_descriptions = "\n".join(
            f"  - {n.id}: {n.name} ({n.description})"
            for n in self.dag.nodes
        )

        proposed: list[ProposedEdge] = []

        for claim in claims:
            user_prompt = NODE_MATCHING_USER.format(
                treatment=claim.treatment,
                outcome=claim.outcome,
                mechanism=claim.mechanism,
                direction=claim.direction,
            )

            try:
                result = self.llm.complete_structured(
                    NODE_MATCHING_SYSTEM.format(node_descriptions=node_descriptions),
                    user_prompt,
                    NODE_MATCH_SCHEMA,
                )
                from_node = result.get("matched_from_node", "")
                to_node = result.get("matched_to_node", "")
                confidence = result.get("match_confidence", 0.0)
                new_nodes = result.get("requires_new_nodes", [])
                edge_id = result.get("edge_id_suggestion", "")

                # Normalize node IDs from LLM
                from_node = from_node.strip().lower() if from_node else ""
                to_node = to_node.strip().lower() if to_node else ""

                # If LLM left a node empty but flagged new nodes, use canonical ID
                if not to_node and new_nodes:
                    import re as _re
                    to_node = _re.sub(r"[^a-z0-9_]", "_", new_nodes[0].lower()).strip("_")
                if not from_node and new_nodes:
                    import re as _re
                    from_node = _re.sub(r"[^a-z0-9_]", "_", new_nodes[-1].lower()).strip("_")

                if not edge_id and from_node and to_node:
                    edge_id = f"{from_node}_to_{to_node}"

                # Resolve edge using structural matching
                existing_edge, canonical_id = self.dag.resolve_edge(
                    edge_id, from_node, to_node,
                    edge_type=claim.edge_type_suggestion,
                )

                is_existing = existing_edge is not None
                if existing_edge:
                    edge_id = canonical_id  # Use canonical edge_id
                    logger.info(f"Matched to existing edge: {edge_id}")

                edge = ProposedEdge(
                    from_node=from_node,
                    to_node=to_node,
                    edge_id=edge_id,
                    edge_type=claim.edge_type_suggestion,
                    evidence=[claim],
                    requires_new_nodes=new_nodes,
                    match_confidence=confidence,
                    is_existing=is_existing,
                )
                proposed.append(edge)

            except Exception as e:
                logger.warning(f"Failed to match claim '{claim.treatment} -> {claim.outcome}': {e}")

        # Merge edges with same from/to (combine evidence)
        merged = self._merge_proposed(proposed)
        return merged

    def propose_edges(self, papers: list[Any]) -> list[ProposedEdge]:
        """
        Full pipeline: extract claims from papers, then match to DAG.

        Args:
            papers: List of Citation objects

        Returns:
            List of ProposedEdge ready for user review
        """
        claims = self.extract_causal_claims(papers)
        if not claims:
            logger.info("No causal claims extracted")
            return []
        return self.match_to_dag(claims)

    def _merge_proposed(self, edges: list[ProposedEdge]) -> list[ProposedEdge]:
        """Merge proposed edges with the same from/to nodes."""
        by_pair: dict[tuple[str, str], ProposedEdge] = {}
        for edge in edges:
            key = (edge.from_node, edge.to_node)
            if key in by_pair:
                existing = by_pair[key]
                existing.evidence.extend(edge.evidence)
                existing.match_confidence = max(
                    existing.match_confidence, edge.match_confidence,
                )
                existing.requires_new_nodes = list(set(
                    existing.requires_new_nodes + edge.requires_new_nodes
                ))
                existing.is_existing = existing.is_existing or edge.is_existing
            else:
                by_pair[key] = edge

        merged = list(by_pair.values())

        # Log recovery metrics
        existing_count = sum(1 for e in merged if e.is_existing)
        new_count = len(merged) - existing_count
        logger.info(
            f"Matched {existing_count}/{len(self.dag.edges)} existing edges, "
            f"{new_count} new proposed"
        )

        return merged
