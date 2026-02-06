"""
DAGScout Agent: Propose candidate nodes/edges from domain knowledge.

DAGScout operates in EXPLORATION mode only. It proposes candidate
edges with justification text, supporting sources, and preliminary
identifiability notes.

Output is written to specs/candidates/candidate_dag.yaml (quarantined
location requiring HITL promotion to canonical DAG).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CandidateEdge:
    """A proposed candidate edge from DAGScout."""

    edge_id: str
    source_node: str
    target_node: str
    justification_text: str
    supporting_sources: list[dict[str, str]] = field(default_factory=list)
    suggested_designs: list[str] = field(default_factory=list)
    data_requirements: list[dict[str, str]] = field(default_factory=list)
    identifiability_note: str = ""
    preliminary_claim_level: str = "REDUCED_FORM"
    requires_human_approval: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "justification_text": self.justification_text,
            "supporting_sources": self.supporting_sources,
            "suggested_designs": self.suggested_designs,
            "data_requirements": self.data_requirements,
            "identifiability_note": self.identifiability_note,
            "preliminary_claim_level": self.preliminary_claim_level,
            "requires_human_approval": self.requires_human_approval,
        }


class DAGScout:
    """
    DAGScout agent: proposes candidate edges for DAG expansion.

    All proposals go to quarantined location (specs/candidates/).
    Promotion to canonical DAG requires HITL approval.
    """

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("specs/candidates")
        self.candidates: list[CandidateEdge] = []

    def propose(self, candidate: CandidateEdge) -> CandidateEdge:
        """Propose a candidate edge."""
        candidate.requires_human_approval = True
        self.candidates.append(candidate)
        logger.info(f"DAGScout proposed: {candidate.edge_id}")
        return candidate

    def save_candidates(self) -> Path:
        """Save all candidates to YAML."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "candidate_dag.yaml"

        data = {
            "candidate_edges": [c.to_dict() for c in self.candidates],
        }

        with open(output_path, "w") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)

        return output_path
