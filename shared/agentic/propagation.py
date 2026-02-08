"""
Propagation Engine: Compute causal effect paths through a DAG.

Traverses the DAG from source to target, applying:
- Mode gating (STRUCTURAL / REDUCED_FORM / DESCRIPTIVE)
- Unit compatibility checks
- Frequency alignment checks
- TSGuard gating (high-risk / counterfactual-blocked)
- IssueLedger gating (CRITICAL open issues)
- Counterfactual eligibility (shock vs policy)
- SE propagation via delta method (independence assumed)

Key design decisions:
- mode_propagation_allowed in stored EdgeCards is empty; roles are
  derived at runtime via derive_propagation_role(edge_type, claim_level).
- Multi-path results shown separately by default (no auto-summing).
- SE method is always labeled "delta_independence_naive".
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Unit Algebra
# ──────────────────────────────────────────────────────────────────────

UNIT_KINDS = Literal[
    "pp", "pct", "log_point", "bn_kzt", "ratio", "index", "sd", "bps", "unknown"
]

# Patterns for parsing free-text unit specifications
_UNIT_PATTERNS: list[tuple[str, str]] = [
    (r"pp\b|percentage\s*point", "pp"),
    (r"\bpct\b|percent|%", "pct"),
    (r"\blog[_ ]?point", "log_point"),
    (r"\bbn[_ ]?kzt\b", "bn_kzt"),
    (r"\bratio\b", "ratio"),
    (r"\bindex\b", "index"),
    (r"\bsd\b|standard\s*deviation", "sd"),
    (r"\bbps\b|basis\s*point", "bps"),
]


@dataclass
class UnitSpec:
    """Structured unit for dimensional analysis during chain propagation."""

    kind: str = "unknown"  # one of UNIT_KINDS
    scale: float = 1.0

    @staticmethod
    def from_structured(kind: str, scale: float = 1.0) -> UnitSpec:
        """Create from structured DAG spec fields (preferred)."""
        return UnitSpec(kind=kind, scale=scale)

    @staticmethod
    def parse(text: str) -> UnitSpec:
        """Parse free-text unit_specification into structured UnitSpec."""
        if not text:
            return UnitSpec(kind="unknown")
        text_lower = text.lower().strip()

        # Try to extract scale (e.g., "10% depreciation" -> scale=0.10)
        scale = 1.0
        scale_match = re.match(r"^(\d+(?:\.\d+)?)\s*(%|pp|sd|bps)", text_lower)
        if scale_match:
            num = float(scale_match.group(1))
            unit_hint = scale_match.group(2)
            if unit_hint == "%":
                scale = num / 100.0
            else:
                scale = num

        for pattern, kind in _UNIT_PATTERNS:
            if re.search(pattern, text_lower):
                return UnitSpec(kind=kind, scale=scale)

        return UnitSpec(kind="unknown")

    def compatible_with(self, other: UnitSpec) -> bool:
        """True if units are the same kind (scale differences handled by rescaling)."""
        return self.kind == other.kind


# ──────────────────────────────────────────────────────────────────────
# Frequency helpers
# ──────────────────────────────────────────────────────────────────────

_FREQ_MAP = {
    "day": "monthly",
    "month": "monthly",
    "quarter": "quarterly",
    "year": "annual",
}


def _lag_unit_to_frequency(lag_unit: str) -> str:
    return _FREQ_MAP.get(lag_unit, "monthly")


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────


@dataclass
class EdgeInPath:
    """A single edge within a propagation path."""

    edge_id: str
    from_node: str
    to_node: str
    coefficient: float
    se: float
    edge_type: str
    role: str
    claim_level: str
    treatment_unit: UnitSpec = field(default_factory=UnitSpec)
    outcome_unit: UnitSpec = field(default_factory=UnitSpec)
    frequency: str = "monthly"
    horizon: int = 0
    is_estimated: bool = True


@dataclass
class PropagationPath:
    """One source-to-target path with aggregated effect and SE."""

    edges: list[EdgeInPath] = field(default_factory=list)
    total_effect: float = 0.0
    total_se: float = 0.0
    ci_95: tuple[float, float] = (0.0, 0.0)
    se_method: str = "delta_independence_naive"
    warnings: list[str] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return len(self.blocked_reasons) > 0


@dataclass
class BlockedEdgeInfo:
    """Why a specific edge was blocked from propagation."""

    edge_id: str
    from_node: str
    to_node: str
    reason: str


@dataclass
class PropagationResult:
    """Complete result of propagation between source and target."""

    paths: list[PropagationPath] = field(default_factory=list)
    blocked_edges: list[BlockedEdgeInfo] = field(default_factory=list)
    aggregation_policy: str = "single_path"
    aggregated_effect: float | None = None
    aggregated_se: float | None = None
    warnings: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Unit resolution helpers
# ──────────────────────────────────────────────────────────────────────


def _resolve_treatment_unit(edge_spec: Any) -> UnitSpec:
    """Resolve treatment unit from DAG EdgeSpec."""
    # 1. Structured unit_kind field (preferred)
    uk = getattr(edge_spec, "unit_kind", None)
    if isinstance(uk, dict):
        kind = uk.get("treatment", "unknown")
        scale = uk.get("treatment_scale", 1.0)
        if kind != "unknown":
            return UnitSpec.from_structured(kind, scale)

    # 2. Free-text from estimates or edge card
    est = getattr(edge_spec, "estimates", None)
    if est and getattr(est, "treatment_unit", ""):
        return UnitSpec.parse(est.treatment_unit)

    return UnitSpec(kind="unknown")


def _resolve_outcome_unit(edge_spec: Any) -> UnitSpec:
    """Resolve outcome unit from DAG EdgeSpec."""
    uk = getattr(edge_spec, "unit_kind", None)
    if isinstance(uk, dict):
        kind = uk.get("outcome", "unknown")
        scale = uk.get("outcome_scale", 1.0)
        if kind != "unknown":
            return UnitSpec.from_structured(kind, scale)

    est = getattr(edge_spec, "estimates", None)
    if est and getattr(est, "outcome_unit", ""):
        return UnitSpec.parse(est.outcome_unit)

    return UnitSpec(kind="unknown")


# ──────────────────────────────────────────────────────────────────────
# SE propagation (delta method, independence assumed)
# ──────────────────────────────────────────────────────────────────────


def _compute_chain_effect_and_se(
    edges: list[EdgeInPath],
) -> tuple[float, float, list[str]]:
    """
    Compute product of coefficients and SE via delta method.

    Var(Z) = sum_i[(prod_{j!=i} X_j)^2 * se_i^2] for estimated edges.
    Bridge/identity edges have SE=0 and are skipped in SE computation.

    Returns (total_effect, total_se, warnings).
    """
    warns: list[str] = []
    if not edges:
        return 0.0, 0.0, warns

    coefficients = [e.coefficient for e in edges]
    total_effect = math.prod(coefficients) if coefficients else 0.0

    # SE propagation
    estimated_edges = [e for e in edges if e.is_estimated and e.se > 0]
    if not estimated_edges:
        return total_effect, 0.0, warns

    if len(estimated_edges) > 1:
        warns.append(
            "SE assumes independence between edge estimates "
            "(likely violated: shared data/shocks). CI may be too tight."
        )

    variance = 0.0
    for i, ei in enumerate(edges):
        if not ei.is_estimated or ei.se <= 0:
            continue
        # Product of all OTHER coefficients
        others_product = 1.0
        for j, ej in enumerate(edges):
            if j != i:
                others_product *= ej.coefficient
        variance += (others_product ** 2) * (ei.se ** 2)

    total_se = math.sqrt(variance) if variance > 0 else 0.0
    return total_effect, total_se, warns


# ──────────────────────────────────────────────────────────────────────
# Propagation Engine
# ──────────────────────────────────────────────────────────────────────


class PropagationEngine:
    """
    Traverse a DAG to compute effect propagation paths.

    Reuses existing modules:
    - dag/parser.py -> DAGSpec, EdgeSpec, EdgeTiming
    - query_mode.py -> derive_propagation_role, is_edge_allowed_for_propagation
    - output/edge_card.py -> EdgeCard, CounterfactualBlock, IdentificationBlock
    - ts_guard.py -> TSGuardResult
    - issues/issue_ledger.py -> IssueLedger
    """

    def __init__(
        self,
        dag: Any,  # DAGSpec
        edge_cards: dict[str, Any] | None = None,  # edge_id -> EdgeCard
        tsguard_results: dict[str, Any] | None = None,  # edge_id -> TSGuardResult
        issue_ledger: Any | None = None,  # IssueLedger
        mode_config: Any | None = None,  # QueryModeConfig
    ):
        self.dag = dag
        self.edge_cards = edge_cards or {}
        self.tsguard_results = tsguard_results or {}
        self.issue_ledger = issue_ledger
        self._mode_config = mode_config

        # Build adjacency list
        self._adj: dict[str, list[str]] = {}
        self._edge_by_pair: dict[tuple[str, str], list[Any]] = {}
        for edge in dag.edges:
            self._adj.setdefault(edge.from_node, []).append(edge.to_node)
            self._edge_by_pair.setdefault((edge.from_node, edge.to_node), []).append(edge)

    @property
    def mode_config(self) -> Any:
        if self._mode_config is None:
            from shared.agentic.query_mode import QueryModeConfig
            self._mode_config = QueryModeConfig.load()
        return self._mode_config

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def find_all_paths(
        self,
        source: str,
        target: str,
        mode: str = "REDUCED_FORM",
        scenario_type: str = "shock",
        aggregation: str = "single_path",
        max_depth: int = 10,
        allow_unknown_units: bool = False,
    ) -> PropagationResult:
        """Find all paths from source to target, applying guardrails."""
        result = PropagationResult(aggregation_policy=aggregation)

        # Validate nodes exist
        if self.dag.get_node(source) is None:
            result.warnings.append(f"Source node '{source}' not found in DAG")
            return result
        if self.dag.get_node(target) is None:
            result.warnings.append(f"Target node '{target}' not found in DAG")
            return result

        mode_spec = self.mode_config.get_spec(mode)

        # DFS to find all simple paths
        raw_paths: list[list[tuple[str, str, Any]]] = []  # list of [(from, to, EdgeSpec)]
        self._dfs(source, target, [], set(), raw_paths, max_depth)

        if not raw_paths:
            result.warnings.append(f"No paths found from {source} to {target}")
            return result

        # Process each path
        for raw_path in raw_paths:
            path = self._process_path(
                raw_path, mode, mode_spec, scenario_type, allow_unknown_units,
            )
            if path.blocked_reasons:
                # Collect blocked edges
                for edge in path.edges:
                    for reason in path.blocked_reasons:
                        if edge.edge_id in reason:
                            result.blocked_edges.append(BlockedEdgeInfo(
                                edge_id=edge.edge_id,
                                from_node=edge.from_node,
                                to_node=edge.to_node,
                                reason=reason,
                            ))
            result.paths.append(path)

        # Aggregation
        if aggregation == "sum_disjoint":
            self._try_aggregate_disjoint(result)

        return result

    def get_edges_summary(self, mode: str) -> list[dict]:
        """Summary of all edges with role/permissions for given mode."""
        from shared.agentic.query_mode import (
            derive_propagation_role,
            is_edge_allowed_for_propagation,
        )
        mode_spec = self.mode_config.get_spec(mode)
        summaries = []
        for edge in self.dag.edges:
            card = self.edge_cards.get(edge.id)
            claim_level = ""
            if card:
                claim_level = getattr(card.identification, "claim_level", "")
            edge_type = edge.get_edge_type()
            role = derive_propagation_role(edge_type, claim_level)
            allowed = is_edge_allowed_for_propagation(role, mode_spec)
            coeff = card.estimates.point if card and card.estimates else None
            se = card.estimates.se if card and card.estimates else None
            summaries.append({
                "edge_id": edge.id,
                "from": edge.from_node,
                "to": edge.to_node,
                "edge_type": edge_type,
                "role": role,
                "claim_level": claim_level,
                "allowed": allowed,
                "coefficient": coeff,
                "se": se,
            })
        return summaries

    def get_node_names(self) -> dict[str, str]:
        """Map of node_id -> human name."""
        return {n.id: n.name for n in self.dag.nodes}

    def get_all_node_ids(self) -> list[str]:
        return [n.id for n in self.dag.nodes]

    # ──────────────────────────────────────────────────────────────────
    # DFS path finding
    # ──────────────────────────────────────────────────────────────────

    def _dfs(
        self,
        current: str,
        target: str,
        path: list[tuple[str, str, Any]],
        visited: set[str],
        all_paths: list[list[tuple[str, str, Any]]],
        max_depth: int,
    ) -> None:
        if len(path) > max_depth:
            return
        if current == target and path:
            all_paths.append(list(path))
            return

        visited.add(current)
        for neighbor in self._adj.get(current, []):
            if neighbor not in visited:
                for edge_spec in self._edge_by_pair.get((current, neighbor), []):
                    path.append((current, neighbor, edge_spec))
                    self._dfs(neighbor, target, path, visited, all_paths, max_depth)
                    path.pop()
        visited.discard(current)

    # ──────────────────────────────────────────────────────────────────
    # Path processing with guardrails
    # ──────────────────────────────────────────────────────────────────

    def _process_path(
        self,
        raw_path: list[tuple[str, str, Any]],
        mode: str,
        mode_spec: Any,
        scenario_type: str,
        allow_unknown_units: bool,
    ) -> PropagationPath:
        from shared.agentic.query_mode import (
            derive_propagation_role,
            is_edge_allowed_for_propagation,
            is_shock_cf_allowed,
            is_policy_cf_allowed,
        )

        path = PropagationPath()
        edges_in_path: list[EdgeInPath] = []
        blocked_reasons: list[str] = []

        for from_node, to_node, edge_spec in raw_path:
            card = self.edge_cards.get(edge_spec.id)
            claim_level = ""
            if card:
                claim_level = getattr(card.identification, "claim_level", "")

            edge_type = edge_spec.get_edge_type()
            role = derive_propagation_role(edge_type, claim_level)

            # Get coefficient and SE from edge card
            coeff = 0.0
            se = 0.0
            if card and card.estimates:
                coeff = card.estimates.point
                se = card.estimates.se

            # Determine if estimated (bridge/identity have SE=0)
            is_estimated = role not in ("bridge", "identity")

            # Resolve units
            treatment_unit = _resolve_treatment_unit(edge_spec)
            outcome_unit = _resolve_outcome_unit(edge_spec)
            # Also try from edge card estimates
            if treatment_unit.kind == "unknown" and card and card.estimates:
                if card.estimates.treatment_unit:
                    treatment_unit = UnitSpec.parse(card.estimates.treatment_unit)
            if outcome_unit.kind == "unknown" and card and card.estimates:
                if card.estimates.outcome_unit:
                    outcome_unit = UnitSpec.parse(card.estimates.outcome_unit)

            # Frequency
            timing = getattr(edge_spec, "timing", None)
            freq = "monthly"
            horizon = 0
            if timing:
                freq = _lag_unit_to_frequency(timing.lag_unit)
                horizon = timing.lag

            edge_in_path = EdgeInPath(
                edge_id=edge_spec.id,
                from_node=from_node,
                to_node=to_node,
                coefficient=coeff,
                se=se,
                edge_type=edge_type,
                role=role,
                claim_level=claim_level,
                treatment_unit=treatment_unit,
                outcome_unit=outcome_unit,
                frequency=freq,
                horizon=horizon,
                is_estimated=is_estimated,
            )
            edges_in_path.append(edge_in_path)

            # ── Guardrail 1: Mode gating ──
            if not is_edge_allowed_for_propagation(role, mode_spec):
                blocked_reasons.append(
                    f"mode_restriction: {edge_spec.id} (role={role}) not allowed in {mode}"
                )

            # ── Guardrail 2: Counterfactual gating ──
            if scenario_type == "shock":
                cf_blocked = False
                if card and hasattr(card, "counterfactual_block"):
                    cf_block = card.counterfactual_block
                    if hasattr(cf_block, "shock_scenario_allowed"):
                        if not cf_block.shock_scenario_allowed:
                            cf_blocked = True
                            reason = getattr(cf_block, "reason_shock_blocked", "") or ""
                            blocked_reasons.append(
                                f"counterfactual_blocked: {edge_spec.id} shock CF blocked"
                                + (f" ({reason})" if reason else "")
                            )
                if not cf_blocked and claim_level:
                    allowed, reason = is_shock_cf_allowed(claim_level, mode_spec)
                    if not allowed and reason:
                        blocked_reasons.append(
                            f"counterfactual_blocked: {edge_spec.id} {reason}"
                        )

            elif scenario_type == "policy":
                cf_blocked = False
                if card and hasattr(card, "counterfactual_block"):
                    cf_block = card.counterfactual_block
                    if hasattr(cf_block, "policy_intervention_allowed"):
                        if not cf_block.policy_intervention_allowed:
                            cf_blocked = True
                            reason = getattr(cf_block, "reason_policy_blocked", "") or ""
                            blocked_reasons.append(
                                f"counterfactual_blocked: {edge_spec.id} policy CF blocked"
                                + (f" ({reason})" if reason else "")
                            )
                if not cf_blocked and claim_level:
                    allowed, reason = is_policy_cf_allowed(claim_level, mode_spec)
                    if not allowed and reason:
                        blocked_reasons.append(
                            f"counterfactual_blocked: {edge_spec.id} {reason}"
                        )

            # ── Guardrail 3: TSGuard gating ──
            ts_result = self.tsguard_results.get(edge_spec.id)
            if ts_result:
                if scenario_type == "shock" and getattr(ts_result, "counterfactual_blocked", False):
                    blocked_reasons.append(
                        f"tsguard_block: {edge_spec.id} counterfactual_blocked"
                    )
                if getattr(ts_result, "has_any_high_risk", False):
                    path.warnings.append(
                        f"TSGuard high-risk flag on {edge_spec.id}: "
                        f"dynamics={getattr(ts_result, 'dynamics_risk', {})}"
                    )

            # ── Guardrail 4: IssueLedger gating ──
            if self.issue_ledger:
                edge_issues = self.issue_ledger.get_issues_for_edge(edge_spec.id)
                for issue in edge_issues:
                    if issue.is_open and issue.is_critical:
                        blocked_reasons.append(
                            f"issue_ledger_block: {edge_spec.id} "
                            f"{issue.rule_id} - {issue.message}"
                        )

            # ── Guardrail 5: Reaction function always blocked ──
            if edge_type == "reaction_function":
                blocked_reasons.append(
                    f"mode_restriction: {edge_spec.id} reaction_function "
                    "edges are never allowed on propagation paths"
                )

        # ── Guardrail 6: Unit compatibility ──
        for i in range(len(edges_in_path) - 1):
            curr_outcome = edges_in_path[i].outcome_unit
            next_treatment = edges_in_path[i + 1].treatment_unit

            if curr_outcome.kind == "unknown" or next_treatment.kind == "unknown":
                if mode in ("STRUCTURAL", "REDUCED_FORM") and not allow_unknown_units:
                    blocked_reasons.append(
                        f"unit_mismatch: unknown unit between "
                        f"{edges_in_path[i].edge_id} -> {edges_in_path[i+1].edge_id}"
                    )
                else:
                    path.warnings.append(
                        f"Unknown unit between {edges_in_path[i].edge_id} "
                        f"-> {edges_in_path[i+1].edge_id}"
                    )
            elif not curr_outcome.compatible_with(next_treatment):
                blocked_reasons.append(
                    f"unit_mismatch: {edges_in_path[i].edge_id} outcome "
                    f"({curr_outcome.kind}) != {edges_in_path[i+1].edge_id} "
                    f"treatment ({next_treatment.kind})"
                )

        # ── Guardrail 7: Frequency alignment ──
        if len(edges_in_path) > 1:
            frequencies = {e.frequency for e in edges_in_path}
            if len(frequencies) > 1:
                # Check for frequency bridge edges
                has_bridge = any(
                    e.role in ("bridge", "identity")
                    and getattr(
                        self.dag.get_edge(e.edge_id), "frequency_bridge", False
                    )
                    for e in edges_in_path
                )
                if not has_bridge:
                    blocked_reasons.append(
                        f"frequency_mismatch: mixed frequencies {frequencies} "
                        "without explicit frequency bridge"
                    )

        path.edges = edges_in_path
        path.blocked_reasons = blocked_reasons

        if not blocked_reasons:
            total_effect, total_se, se_warns = _compute_chain_effect_and_se(edges_in_path)
            path.total_effect = total_effect
            path.total_se = total_se
            path.warnings.extend(se_warns)
            if total_se > 0:
                path.ci_95 = (
                    total_effect - 1.96 * total_se,
                    total_effect + 1.96 * total_se,
                )
            else:
                path.ci_95 = (total_effect, total_effect)

        return path

    # ──────────────────────────────────────────────────────────────────
    # Aggregation
    # ──────────────────────────────────────────────────────────────────

    def _try_aggregate_disjoint(self, result: PropagationResult) -> None:
        """Sum effects if paths are edge-disjoint."""
        valid_paths = [p for p in result.paths if not p.is_blocked]
        if len(valid_paths) < 2:
            return

        # Check edge-disjoint
        all_edge_ids: list[set[str]] = []
        for p in valid_paths:
            edge_set = {e.edge_id for e in p.edges}
            all_edge_ids.append(edge_set)

        for i in range(len(all_edge_ids)):
            for j in range(i + 1, len(all_edge_ids)):
                if all_edge_ids[i] & all_edge_ids[j]:
                    result.warnings.append(
                        "Paths share edges; cannot sum (would double-count)."
                    )
                    return

        # Sum effects
        total = sum(p.total_effect for p in valid_paths)
        total_var = sum(p.total_se ** 2 for p in valid_paths)
        result.aggregated_effect = total
        result.aggregated_se = math.sqrt(total_var) if total_var > 0 else 0.0
