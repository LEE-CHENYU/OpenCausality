"""
Critic Schema: Types, Validation, and Principles Loader.

Defines the data structures for the DAG Critic Loop:
- RevisionType enum (closed set of allowed revision actions)
- CriticRevision (single proposed change)
- CriticFeedback (structured per-iteration diagnostics)
- CriticReport (full critic loop outcome)
- CriticRevisionValidator (code-validates each revision against DAG state)
- load_principles() (loads config/agentic/critic_principles.yaml)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Principles Loader
# ──────────────────────────────────────────────────────────────────

_PRINCIPLES_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "config"
    / "agentic"
    / "critic_principles.yaml"
)


def load_principles(path: Path | None = None) -> dict[str, Any]:
    """Load critic principles from YAML config."""
    p = path or _PRINCIPLES_PATH
    with open(p) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────
# Revision Types
# ──────────────────────────────────────────────────────────────────

class RevisionType(str, Enum):
    """Closed set of allowed critic revisions."""

    UPGRADE_EDGE_TYPE = "upgrade_edge_type"
    ADD_IDENTIFICATION_STRATEGY = "add_identification_strategy"
    FIX_FORMULA = "fix_formula"
    ADD_MISSING_NODE = "add_missing_node"
    ADD_MISSING_EDGE = "add_missing_edge"
    ADD_STRUCTURAL_METADATA = "add_structural_metadata"
    ADD_UNIT_SPECIFICATION = "add_unit_specification"
    INVERT_EDGE_DIRECTION = "invert_edge_direction"


# Map revision types to whether they require re-estimation
REVISION_REQUIRES_REESTIMATION: dict[RevisionType, bool] = {
    RevisionType.UPGRADE_EDGE_TYPE: True,
    RevisionType.ADD_IDENTIFICATION_STRATEGY: False,
    RevisionType.FIX_FORMULA: True,
    RevisionType.ADD_MISSING_NODE: True,
    RevisionType.ADD_MISSING_EDGE: True,
    RevisionType.ADD_STRUCTURAL_METADATA: False,
    RevisionType.ADD_UNIT_SPECIFICATION: False,
    RevisionType.INVERT_EDGE_DIRECTION: True,
}


# ──────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────

@dataclass
class CriticRevision:
    """A single proposed revision from the critic."""

    revision_type: RevisionType
    target_edge_id: str
    confidence: float
    reasoning: str
    details: dict[str, Any] = field(default_factory=dict)
    # Populated by validator
    validated: bool = False
    rejection_reason: str = ""

    @property
    def requires_reestimation(self) -> bool:
        return REVISION_REQUIRES_REESTIMATION.get(self.revision_type, True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_type": self.revision_type.value,
            "target_edge_id": self.target_edge_id,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "details": self.details,
            "validated": self.validated,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CriticRevision:
        return cls(
            revision_type=RevisionType(d["revision_type"]),
            target_edge_id=d["target_edge_id"],
            confidence=d.get("confidence", 0.0),
            reasoning=d.get("reasoning", ""),
            details=d.get("details", {}),
            validated=d.get("validated", False),
            rejection_reason=d.get("rejection_reason", ""),
        )


@dataclass
class QualityScore:
    """Composite quality score with component breakdown."""

    edge_type_diversity: float = 0.0
    identification_coverage: float = 0.0
    metadata_completeness: float = 0.0
    structural_soundness: float = 0.0
    unit_completeness: float = 0.0
    formula_correctness: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "edge_type_diversity": self.edge_type_diversity,
            "identification_coverage": self.identification_coverage,
            "metadata_completeness": self.metadata_completeness,
            "structural_soundness": self.structural_soundness,
            "unit_completeness": self.unit_completeness,
            "formula_correctness": self.formula_correctness,
            "total": self.total,
        }


@dataclass
class CriticFeedback:
    """Structured feedback collected per iteration for the critic prompt."""

    quality_score: QualityScore
    edge_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    open_issues: list[dict[str, Any]] = field(default_factory=list)
    placebo_results: list[dict[str, Any]] = field(default_factory=list)
    edges_missing_identification: list[str] = field(default_factory=list)
    edges_missing_units: list[str] = field(default_factory=list)
    formula_violations: list[dict[str, Any]] = field(default_factory=list)
    structural_gaps: list[dict[str, Any]] = field(default_factory=list)
    causal_logic_probes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_score": self.quality_score.to_dict(),
            "edge_diagnostics": self.edge_diagnostics,
            "open_issues": self.open_issues,
            "placebo_results": self.placebo_results,
            "edges_missing_identification": self.edges_missing_identification,
            "edges_missing_units": self.edges_missing_units,
            "formula_violations": self.formula_violations,
            "structural_gaps": self.structural_gaps,
            "causal_logic_probes": self.causal_logic_probes,
        }


@dataclass
class CriticReport:
    """Full critic loop outcome."""

    iterations: int
    initial_score: QualityScore
    final_score: QualityScore
    revisions_applied: list[CriticRevision] = field(default_factory=list)
    revisions_rejected: list[CriticRevision] = field(default_factory=list)
    convergence_reason: str = ""
    score_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "initial_score": self.initial_score.to_dict(),
            "final_score": self.final_score.to_dict(),
            "revisions_applied": [r.to_dict() for r in self.revisions_applied],
            "revisions_rejected": [r.to_dict() for r in self.revisions_rejected],
            "convergence_reason": self.convergence_reason,
            "score_history": self.score_history,
        }


# ──────────────────────────────────────────────────────────────────
# Revision Validator
# ──────────────────────────────────────────────────────────────────

# Valid edge types (INVARIANTS.md Section 2.3)
VALID_EDGE_TYPES = {"causal", "reaction_function", "mechanical", "immutable", "identity"}


class CriticRevisionValidator:
    """Code-validates each proposed revision against current DAG state.

    All revisions must pass validation before application.
    This is the Layer 1 code gate that prevents invalid LLM proposals.
    """

    def __init__(
        self,
        dag_config: dict[str, Any],
        confidence_threshold: float = 0.7,
    ):
        self.dag = dag_config
        self.nodes = {n["id"]: n for n in dag_config.get("nodes", [])}
        self.edges = {e["id"]: e for e in dag_config.get("edges", [])}
        self.confidence_threshold = confidence_threshold

    def validate(self, revision: CriticRevision) -> CriticRevision:
        """Validate a single revision. Sets validated/rejection_reason fields."""
        # Confidence gate
        if revision.confidence < self.confidence_threshold:
            revision.validated = False
            revision.rejection_reason = (
                f"Confidence {revision.confidence:.2f} below threshold "
                f"{self.confidence_threshold:.2f}"
            )
            return revision

        # Dispatch to type-specific validator
        validator_name = f"_validate_{revision.revision_type.value}"
        validator = getattr(self, validator_name, None)
        if validator is None:
            revision.validated = False
            revision.rejection_reason = f"No validator for {revision.revision_type.value}"
            return revision

        return validator(revision)

    def validate_batch(self, revisions: list[CriticRevision]) -> list[CriticRevision]:
        """Validate a batch of revisions.

        Sorts ADD_MISSING_NODE before ADD_MISSING_EDGE so that newly
        added nodes are registered before edge validation checks endpoints.
        """
        def _order(r: CriticRevision) -> int:
            if r.revision_type == RevisionType.ADD_MISSING_NODE:
                return 0
            if r.revision_type == RevisionType.ADD_MISSING_EDGE:
                return 2
            return 1

        sorted_revs = sorted(revisions, key=_order)
        results: list[CriticRevision] = []
        for r in sorted_revs:
            validated = self.validate(r)
            # Register validated nodes so subsequent edge validation sees them
            if (
                validated.validated
                and validated.revision_type == RevisionType.ADD_MISSING_NODE
            ):
                node_id = validated.details.get("node_definition", {}).get("id", "")
                if node_id:
                    self.nodes[node_id] = {"id": node_id}
            results.append(validated)
        return results

    # ── Type-specific validators ──────────────────────────────────

    def _validate_upgrade_edge_type(self, rev: CriticRevision) -> CriticRevision:
        """Validate edge type upgrade."""
        edge = self.edges.get(rev.target_edge_id)
        if edge is None:
            rev.validated = False
            rev.rejection_reason = f"Edge '{rev.target_edge_id}' not found in DAG"
            return rev

        new_type = rev.details.get("new_edge_type", "")
        if new_type not in VALID_EDGE_TYPES:
            rev.validated = False
            rev.rejection_reason = (
                f"Invalid edge_type '{new_type}'. "
                f"Must be one of: {VALID_EDGE_TYPES}"
            )
            return rev

        old_type = edge.get("edge_type", "")
        if new_type == old_type:
            rev.validated = False
            rev.rejection_reason = f"Edge already has type '{new_type}'"
            return rev

        # If upgrading to 'identity', verify the to_node is derived with a formula
        if new_type == "identity":
            to_node = self.nodes.get(edge.get("to", ""))
            if not to_node or not to_node.get("derived"):
                rev.validated = False
                rev.rejection_reason = (
                    f"Cannot set edge_type=identity: to_node "
                    f"'{edge.get('to', '')}' is not a derived node"
                )
                return rev

        rev.validated = True
        return rev

    def _validate_add_identification_strategy(self, rev: CriticRevision) -> CriticRevision:
        """Validate identification strategy addition."""
        edge = self.edges.get(rev.target_edge_id)
        if edge is None:
            rev.validated = False
            rev.rejection_reason = f"Edge '{rev.target_edge_id}' not found in DAG"
            return rev

        strategy = rev.details.get("identification_strategy", {})
        if not strategy.get("type"):
            rev.validated = False
            rev.rejection_reason = "identification_strategy.type is required"
            return rev

        rev.validated = True
        return rev

    def _validate_fix_formula(self, rev: CriticRevision) -> CriticRevision:
        """Validate formula fix."""
        node_id = rev.details.get("node_id", "")
        node = self.nodes.get(node_id)
        if node is None:
            rev.validated = False
            rev.rejection_reason = f"Node '{node_id}' not found in DAG"
            return rev

        new_formula = rev.details.get("new_formula", "")
        if not new_formula:
            rev.validated = False
            rev.rejection_reason = "new_formula is required"
            return rev

        # Verify no double-log in new formula
        identity = node.get("identity", {})
        deps = identity.get("depends_on") or node.get("depends_on", [])
        for dep in deps:
            dep_node = self.nodes.get(dep)
            if not dep_node:
                continue
            if "log" in dep_node.get("transforms", []) and f"log({dep})" in new_formula:
                rev.validated = False
                rev.rejection_reason = (
                    f"New formula still double-logs '{dep}' "
                    f"(has transforms: [log] and log() in formula)"
                )
                return rev

        rev.validated = True
        return rev

    def _validate_add_missing_node(self, rev: CriticRevision) -> CriticRevision:
        """Validate missing node addition."""
        node_def = rev.details.get("node_definition", {})
        node_id = node_def.get("id", "")
        if not node_id:
            rev.validated = False
            rev.rejection_reason = "node_definition.id is required"
            return rev

        if node_id in self.nodes:
            rev.validated = False
            rev.rejection_reason = f"Node '{node_id}' already exists in DAG"
            return rev

        # Validate snake_case ID
        if not re.match(r'^[a-z0-9][a-z0-9_]*[a-z0-9]$|^[a-z0-9]$', node_id):
            rev.validated = False
            rev.rejection_reason = f"Node ID '{node_id}' is not valid snake_case"
            return rev

        rev.validated = True
        return rev

    def _validate_add_missing_edge(self, rev: CriticRevision) -> CriticRevision:
        """Validate missing edge addition."""
        from_node = rev.details.get("from_node", "")
        to_node = rev.details.get("to_node", "")

        if not from_node or not to_node:
            rev.validated = False
            rev.rejection_reason = "from_node and to_node are required"
            return rev

        if from_node not in self.nodes:
            rev.validated = False
            rev.rejection_reason = f"from_node '{from_node}' not found in DAG"
            return rev

        if to_node not in self.nodes:
            rev.validated = False
            rev.rejection_reason = f"to_node '{to_node}' not found in DAG"
            return rev

        # Check for duplicate edge
        edge_id = f"{from_node}_to_{to_node}"
        if edge_id in self.edges:
            rev.validated = False
            rev.rejection_reason = f"Edge '{edge_id}' already exists"
            return rev

        # Check it wouldn't create a cycle (simple check: to_node shouldn't
        # reach from_node via existing edges)
        if self._would_create_cycle(from_node, to_node):
            rev.validated = False
            rev.rejection_reason = (
                f"Adding {from_node} -> {to_node} would create a cycle"
            )
            return rev

        rev.validated = True
        return rev

    def _validate_add_structural_metadata(self, rev: CriticRevision) -> CriticRevision:
        """Validate structural metadata addition."""
        edge = self.edges.get(rev.target_edge_id)
        if edge is None:
            rev.validated = False
            rev.rejection_reason = f"Edge '{rev.target_edge_id}' not found in DAG"
            return rev

        metadata = rev.details.get("metadata", {})
        if not metadata:
            rev.validated = False
            rev.rejection_reason = "metadata dict is required"
            return rev

        # Validate expected_sign values
        sign = metadata.get("expected_sign")
        if sign and sign not in ("positive", "negative", "any"):
            rev.validated = False
            rev.rejection_reason = f"Invalid expected_sign '{sign}'"
            return rev

        rev.validated = True
        return rev

    def _validate_add_unit_specification(self, rev: CriticRevision) -> CriticRevision:
        """Validate unit specification addition."""
        edge = self.edges.get(rev.target_edge_id)
        if edge is None:
            rev.validated = False
            rev.rejection_reason = f"Edge '{rev.target_edge_id}' not found in DAG"
            return rev

        units = rev.details.get("unit_specification", {})
        if not units.get("treatment_unit") and not units.get("outcome_unit"):
            rev.validated = False
            rev.rejection_reason = "At least one of treatment_unit/outcome_unit required"
            return rev

        rev.validated = True
        return rev

    def _validate_invert_edge_direction(self, rev: CriticRevision) -> CriticRevision:
        """Validate edge direction inversion."""
        edge = self.edges.get(rev.target_edge_id)
        if edge is None:
            rev.validated = False
            rev.rejection_reason = f"Edge '{rev.target_edge_id}' not found in DAG"
            return rev

        from_node = edge.get("from", "")
        to_node = edge.get("to", "")

        # Verify inversion wouldn't create a cycle
        if self._would_create_cycle(to_node, from_node):
            rev.validated = False
            rev.rejection_reason = (
                f"Inverting {from_node} -> {to_node} would create a cycle"
            )
            return rev

        rev.validated = True
        return rev

    # ── Helpers ────────────────────────────────────────────────────

    def _would_create_cycle(self, new_from: str, new_to: str) -> bool:
        """Check if adding new_from -> new_to would create a cycle."""
        # BFS from new_to following existing edges to see if we reach new_from
        adj: dict[str, list[str]] = {n: [] for n in self.nodes}
        for edge in self.edges.values():
            fn = edge.get("from", "")
            tn = edge.get("to", "")
            if fn in adj:
                adj[fn].append(tn)

        visited: set[str] = set()
        frontier = [new_to]
        while frontier:
            node = frontier.pop()
            if node == new_from:
                return True
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    frontier.append(neighbor)
        return False
