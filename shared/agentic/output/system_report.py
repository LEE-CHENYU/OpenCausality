"""
System Report.

The final output of a complete DAG run, aggregating all EdgeCards
and providing a summary of the causal chain analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import yaml

from shared.agentic.output.edge_card import EdgeCard


@dataclass
class EdgeSummary:
    """Summary of a single edge's results."""

    edge_id: str
    treatment: str
    outcome: str
    design: str
    estimate: float | None
    se: float | None
    credibility_rating: str
    credibility_score: float
    diagnostics_pass: bool
    is_precisely_null: bool
    status: str  # "DONE", "DONE_SUGGESTIVE", "BLOCKED_ID", "FAILED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "design": self.design,
            "estimate": self.estimate,
            "se": self.se,
            "credibility_rating": self.credibility_rating,
            "credibility_score": self.credibility_score,
            "diagnostics_pass": self.diagnostics_pass,
            "is_precisely_null": self.is_precisely_null,
            "status": self.status,
        }


@dataclass
class CriticalPathSummary:
    """Summary of the critical path to target."""

    target_node: str
    path_edges: list[str]
    path_complete: bool
    min_credibility: float
    blocking_edges: list[str]


@dataclass
class SystemReport:
    """
    Complete system report for a DAG run.

    Aggregates all EdgeCards and provides overall assessment.
    """

    # Metadata
    dag_name: str
    dag_version_hash: str
    created_at: datetime = field(default_factory=datetime.now)
    mode: str = "EXPLORATION"  # or "CONFIRMATION"
    iteration: int = 0

    # Edge cards
    edge_cards: list[EdgeCard] = field(default_factory=list)
    edge_summaries: list[EdgeSummary] = field(default_factory=list)

    # Blocked edges
    blocked_edges: dict[str, str] = field(default_factory=dict)  # edge_id -> reason

    # Critical path
    critical_path: CriticalPathSummary | None = None

    # Overall assessment
    all_edges_complete: bool = False
    all_critical_edges_pass: bool = False
    min_credibility_score: float = 0.0
    max_credibility_score: float = 0.0
    mean_credibility_score: float = 0.0

    # Audit
    total_iterations: int = 0
    refinements_applied: int = 0

    def add_edge_card(self, card: EdgeCard, status: str = "DONE") -> None:
        """Add an edge card to the report."""
        self.edge_cards.append(card)

        # Create summary
        summary = EdgeSummary(
            edge_id=card.edge_id,
            treatment=card.spec_details.controls[0] if card.spec_details.controls else "",
            outcome="",  # Would need DAG info
            design=card.spec_details.design,
            estimate=card.estimates.point if card.estimates else None,
            se=card.estimates.se if card.estimates else None,
            credibility_rating=card.credibility_rating,
            credibility_score=card.credibility_score,
            diagnostics_pass=card.all_diagnostics_pass(),
            is_precisely_null=card.is_precisely_null,
            status=status,
        )
        self.edge_summaries.append(summary)

        self._update_stats()

    def add_blocked_edge(self, edge_id: str, reason: str) -> None:
        """Add a blocked edge."""
        self.blocked_edges[edge_id] = reason

    def _update_stats(self) -> None:
        """Update aggregate statistics."""
        if not self.edge_summaries:
            return

        scores = [s.credibility_score for s in self.edge_summaries]
        self.min_credibility_score = min(scores)
        self.max_credibility_score = max(scores)
        self.mean_credibility_score = sum(scores) / len(scores)

        self.all_edges_complete = len(self.blocked_edges) == 0

    def get_edges_by_rating(self, rating: str) -> list[EdgeSummary]:
        """Get all edges with a specific rating."""
        return [s for s in self.edge_summaries if s.credibility_rating == rating]

    def get_weak_edges(self, threshold: float = 0.6) -> list[EdgeSummary]:
        """Get edges below credibility threshold."""
        return [s for s in self.edge_summaries if s.credibility_score < threshold]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dag_name": self.dag_name,
            "dag_version_hash": self.dag_version_hash,
            "created_at": self.created_at.isoformat(),
            "mode": self.mode,
            "iteration": self.iteration,
            "edge_summaries": [s.to_dict() for s in self.edge_summaries],
            "blocked_edges": self.blocked_edges,
            "critical_path": {
                "target_node": self.critical_path.target_node,
                "path_edges": self.critical_path.path_edges,
                "path_complete": self.critical_path.path_complete,
                "min_credibility": self.critical_path.min_credibility,
                "blocking_edges": self.critical_path.blocking_edges,
            } if self.critical_path else None,
            "all_edges_complete": self.all_edges_complete,
            "all_critical_edges_pass": self.all_critical_edges_pass,
            "credibility_stats": {
                "min": self.min_credibility_score,
                "max": self.max_credibility_score,
                "mean": self.mean_credibility_score,
            },
            "audit": {
                "total_iterations": self.total_iterations,
                "refinements_applied": self.refinements_applied,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# System Report: {self.dag_name}",
            "",
            f"**Generated:** {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Mode:** {self.mode}",
            f"**Iteration:** {self.iteration}",
            f"**DAG Version:** `{self.dag_version_hash[:12]}...`",
            "",
            "## Summary",
            "",
            f"- **Edges Completed:** {len(self.edge_summaries)}",
            f"- **Edges Blocked:** {len(self.blocked_edges)}",
            f"- **Mean Credibility:** {self.mean_credibility_score:.2f}",
            "",
        ]

        # Credibility distribution
        lines.append("### Credibility Distribution")
        lines.append("")
        for rating in ["A", "B", "C", "D"]:
            edges = self.get_edges_by_rating(rating)
            lines.append(f"- **{rating}:** {len(edges)} edges")
        lines.append("")

        # Edge results table
        lines.append("## Edge Results")
        lines.append("")
        lines.append("| Edge | Design | Estimate | SE | Rating | Score | Diagnostics |")
        lines.append("|------|--------|----------|-----|--------|-------|-------------|")

        for s in self.edge_summaries:
            est = f"{s.estimate:.4f}" if s.estimate is not None else "-"
            se = f"{s.se:.4f}" if s.se is not None else "-"
            diag = "✓" if s.diagnostics_pass else "✗"
            if s.is_precisely_null:
                est += " (null)"
            lines.append(
                f"| {s.edge_id} | {s.design} | {est} | {se} | "
                f"{s.credibility_rating} | {s.credibility_score:.2f} | {diag} |"
            )
        lines.append("")

        # Blocked edges
        if self.blocked_edges:
            lines.append("## Blocked Edges")
            lines.append("")
            for edge_id, reason in self.blocked_edges.items():
                lines.append(f"- **{edge_id}:** {reason}")
            lines.append("")

        # Critical path
        if self.critical_path:
            lines.append("## Critical Path")
            lines.append("")
            lines.append(f"**Target:** {self.critical_path.target_node}")
            lines.append(f"**Path:** {' → '.join(self.critical_path.path_edges)}")
            status = "✓ Complete" if self.critical_path.path_complete else "✗ Incomplete"
            lines.append(f"**Status:** {status}")
            lines.append(f"**Min Credibility:** {self.critical_path.min_credibility:.2f}")
            if self.critical_path.blocking_edges:
                lines.append(f"**Blocking:** {', '.join(self.critical_path.blocking_edges)}")
            lines.append("")

        # Weak edges warning
        weak_edges = self.get_weak_edges()
        if weak_edges:
            lines.append("## ⚠️ Weak Edges (Credibility < 0.60)")
            lines.append("")
            for s in weak_edges:
                lines.append(f"- **{s.edge_id}:** {s.credibility_score:.2f}")
            lines.append("")

        lines.append("---")
        lines.append(f"*Generated by Agentic DAG System*")

        return "\n".join(lines)

    def summary(self) -> str:
        """Generate brief text summary."""
        lines = [
            "=" * 60,
            f"SYSTEM REPORT: {self.dag_name}",
            "=" * 60,
            f"Mode: {self.mode} | Iteration: {self.iteration}",
            f"Edges: {len(self.edge_summaries)} complete, {len(self.blocked_edges)} blocked",
            f"Credibility: min={self.min_credibility_score:.2f}, "
            f"mean={self.mean_credibility_score:.2f}, "
            f"max={self.max_credibility_score:.2f}",
        ]

        # Rating counts
        for rating in ["A", "B", "C", "D"]:
            count = len(self.get_edges_by_rating(rating))
            if count > 0:
                lines.append(f"  {rating}: {count} edges")

        if self.blocked_edges:
            lines.append(f"\nBlocked edges: {list(self.blocked_edges.keys())}")

        lines.append("=" * 60)

        return "\n".join(lines)
