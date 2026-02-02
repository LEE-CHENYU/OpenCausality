"""
Credibility-Based Design Selector.

Selects the best identification design based on:
1. Feasibility given data
2. Credibility ranking (NOT significance chasing)
3. Bad control enforcement
4. Design priority from edge specification
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from shared.agentic.dag.parser import EdgeSpec
from shared.agentic.design.registry import DesignRegistry, DesignSpec
from shared.agentic.design.feasibility import (
    DataReport,
    FeasibilityChecker,
    FeasibilityResult,
)

logger = logging.getLogger(__name__)


@dataclass
class SelectedDesign:
    """A selected design with full specification."""

    design: DesignSpec
    edge_id: str

    # Specification details
    controls: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    fixed_effects: list[str] = field(default_factory=list)
    se_method: str = "cluster"

    # Credibility info
    credibility_weight: float = 0.0
    selection_reason: str = ""

    # Spec hash for audit
    spec_hash: str = ""

    def __post_init__(self):
        if not self.spec_hash:
            self.spec_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of the specification for audit."""
        spec_dict = {
            "design_id": self.design.id,
            "edge_id": self.edge_id,
            "controls": sorted(self.controls),
            "instruments": sorted(self.instruments),
            "fixed_effects": sorted(self.fixed_effects),
            "se_method": self.se_method,
        }
        content = json.dumps(spec_dict, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "design_id": self.design.id,
            "design_name": self.design.name,
            "edge_id": self.edge_id,
            "controls": self.controls,
            "instruments": self.instruments,
            "fixed_effects": self.fixed_effects,
            "se_method": self.se_method,
            "credibility_weight": self.credibility_weight,
            "selection_reason": self.selection_reason,
            "spec_hash": self.spec_hash,
        }


@dataclass
class NotIdentified:
    """
    Returned when no causal design is feasible.

    This is NOT an error - it's a valid outcome indicating
    that the edge cannot be identified with available data.
    """

    edge_id: str
    reason: str
    fallback: str = "ASSOCIATION_ONLY"  # or "BLOCKED"
    attempted_designs: list[str] = field(default_factory=list)
    feasibility_issues: dict[str, list[str]] = field(default_factory=dict)

    @property
    def message(self) -> str:
        return f"Edge {self.edge_id} not identifiable: {self.reason}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "edge_id": self.edge_id,
            "reason": self.reason,
            "fallback": self.fallback,
            "attempted_designs": self.attempted_designs,
            "feasibility_issues": self.feasibility_issues,
            "message": self.message,
        }


class DesignSelector:
    """
    Credibility-based design selector.

    Selects the best design based on:
    1. Filter to allowed designs from edge spec
    2. Check data requirements for each design
    3. Check feasibility (DiD needs treated/control, RDD needs density, etc.)
    4. Enforce bad-control rules (no descendants of treatment)
    5. Return highest-credibility feasible design
    6. If NONE feasible: return NotIdentified

    IMPORTANT: Does NOT chase significance (no t-stat thresholds).
    """

    def __init__(self, registry: DesignRegistry):
        """
        Initialize selector.

        Args:
            registry: The design registry
        """
        self.registry = registry

    def select(
        self,
        edge: EdgeSpec,
        data_report: DataReport,
        forbidden_controls: set[str] | None = None,
        proposed_controls: list[str] | None = None,
    ) -> SelectedDesign | NotIdentified:
        """
        Select the best design for an edge.

        Args:
            edge: The edge specification
            data_report: Report on available data
            forbidden_controls: Nodes that cannot be controls (descendants of treatment)
            proposed_controls: Proposed control variables

        Returns:
            SelectedDesign if feasible, NotIdentified otherwise
        """
        forbidden_controls = forbidden_controls or set()
        proposed_controls = proposed_controls or []

        # Get allowed designs from edge spec
        allowed_ids = edge.allowed_designs
        priority_ids = edge.design_priority or allowed_ids

        # Get design specs
        allowed_designs = [
            self.registry.get(d) for d in allowed_ids
            if self.registry.get(d) is not None
        ]

        if not allowed_designs:
            return NotIdentified(
                edge_id=edge.id,
                reason="No designs found in registry",
                fallback="BLOCKED",
                attempted_designs=allowed_ids,
            )

        # Check feasibility for each design
        checker = FeasibilityChecker(data_report)
        feasibility_results = checker.check_all(allowed_designs)

        # Filter to feasible designs
        feasible_designs = [
            d for d in allowed_designs
            if feasibility_results[d.id].is_feasible
        ]

        if not feasible_designs:
            # No feasible designs - return NotIdentified
            issues = {
                d.id: [str(i.message) for i in feasibility_results[d.id].issues]
                for d in allowed_designs
            }
            return NotIdentified(
                edge_id=edge.id,
                reason="No feasible designs given data constraints",
                fallback="ASSOCIATION_ONLY",
                attempted_designs=allowed_ids,
                feasibility_issues=issues,
            )

        # Sort by priority (if specified) then by credibility
        def sort_key(design: DesignSpec) -> tuple[int, float]:
            # Priority index (lower = higher priority)
            if design.id in priority_ids:
                priority = priority_ids.index(design.id)
            else:
                priority = len(priority_ids)

            # Credibility weight (higher = better)
            credibility = design.credibility_weight

            return (priority, -credibility)

        feasible_designs.sort(key=sort_key)

        # Select the best design
        best_design = feasible_designs[0]

        # Enforce bad control rules
        clean_controls = self.enforce_bad_controls(
            proposed_controls,
            forbidden_controls,
        )

        # Build selected design
        selected = SelectedDesign(
            design=best_design,
            edge_id=edge.id,
            controls=clean_controls,
            instruments=list(edge.instruments),
            fixed_effects=best_design.template.fixed_effects.copy(),
            se_method=best_design.template.standard_errors,
            credibility_weight=best_design.credibility_weight,
            selection_reason=self._build_selection_reason(
                best_design,
                feasibility_results[best_design.id],
                len(feasible_designs),
            ),
        )

        logger.info(
            f"Selected {best_design.id} for edge {edge.id} "
            f"(credibility: {best_design.credibility_weight:.2f})"
        )

        return selected

    def enforce_bad_controls(
        self,
        proposed_controls: list[str],
        forbidden: set[str],
    ) -> list[str]:
        """
        Remove forbidden controls from proposed set.

        Args:
            proposed_controls: Proposed control variables
            forbidden: Set of forbidden controls (descendants of treatment)

        Returns:
            Clean list of controls with forbidden removed
        """
        clean = []
        removed = []

        for ctrl in proposed_controls:
            if ctrl in forbidden:
                removed.append(ctrl)
            else:
                clean.append(ctrl)

        if removed:
            logger.warning(
                f"Removed {len(removed)} bad controls (descendants of treatment): {removed}"
            )

        return clean

    def _build_selection_reason(
        self,
        design: DesignSpec,
        feasibility: FeasibilityResult,
        n_feasible: int,
    ) -> str:
        """Build human-readable selection reason."""
        parts = [
            f"Selected {design.name}",
            f"({n_feasible} feasible designs considered)",
        ]

        if feasibility.warnings:
            parts.append(f"Warnings: {', '.join(feasibility.warnings)}")

        return "; ".join(parts)

    def check_feasibility(
        self,
        design_id: str,
        data_report: DataReport,
    ) -> FeasibilityResult | None:
        """
        Check feasibility for a specific design.

        Args:
            design_id: The design ID
            data_report: Report on available data

        Returns:
            FeasibilityResult or None if design not found
        """
        design = self.registry.get(design_id)
        if not design:
            return None

        checker = FeasibilityChecker(data_report)
        return checker.check(design)

    def get_feasible_designs(
        self,
        allowed_designs: list[str],
        data_report: DataReport,
    ) -> list[DesignSpec]:
        """
        Get all feasible designs from allowed list.

        Args:
            allowed_designs: List of allowed design IDs
            data_report: Report on available data

        Returns:
            List of feasible DesignSpec objects
        """
        designs = [
            self.registry.get(d) for d in allowed_designs
            if self.registry.get(d) is not None
        ]

        checker = FeasibilityChecker(data_report)
        return checker.get_feasible(designs)


def select_design(
    edge: EdgeSpec,
    data_report: DataReport,
    registry: DesignRegistry | None = None,
    forbidden_controls: set[str] | None = None,
) -> SelectedDesign | NotIdentified:
    """
    Select the best design for an edge.

    Convenience function that creates a selector and runs selection.

    Args:
        edge: The edge specification
        data_report: Report on available data
        registry: Design registry (uses default if None)
        forbidden_controls: Nodes that cannot be controls

    Returns:
        SelectedDesign if feasible, NotIdentified otherwise
    """
    if registry is None:
        registry = DesignRegistry()

    selector = DesignSelector(registry)
    return selector.select(edge, data_report, forbidden_controls)
