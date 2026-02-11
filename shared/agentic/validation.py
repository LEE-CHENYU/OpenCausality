"""
DAG Validation Pipeline.

Automated validation checks for causal DAG specifications and EdgeCard outputs.
This module provides:
- Pre-estimation checks (DAG structure, unit presence, edge type)
- Post-estimation checks (N consistency, unit in cards, reaction function labels)
- Report consistency checks (report vs EdgeCard matching)

V3: Domain-agnostic validation that works for any causal inference DAG.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from shared.agentic.output.edge_card import EdgeCard, rating_from_score
from shared.agentic.identification.screen import IdentifiabilityScreen, IdentifiabilityResult


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _is_bad_float(x: float | None) -> bool:
    """True if x is NaN or Inf. Returns False for None (caller skips None)."""
    if x is None:
        return False
    return math.isnan(x) or math.isinf(x)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Must fix before proceeding
    WARNING = "warning"  # Should review, but can proceed
    INFO = "info"        # Informational only


@dataclass
class ValidationIssue:
    """A single validation issue found during checks."""
    check_id: str
    severity: ValidationSeverity
    message: str
    edge_id: str | None = None
    node_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_id": self.check_id,
            "severity": self.severity.value,
            "message": self.message,
            "edge_id": self.edge_id,
            "node_id": self.node_id,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Result of a validation run."""
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    def error_count(self) -> int:
        """Count of ERROR severity issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    def warning_count(self) -> int:
        """Count of WARNING severity issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the result."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.passed = False

    def to_markdown(self) -> str:
        """Generate markdown report of validation results."""
        lines = ["# Validation Report", ""]

        status = "PASSED" if self.passed else "FAILED"
        lines.append(f"**Status:** {status}")
        lines.append(f"**Errors:** {self.error_count()}")
        lines.append(f"**Warnings:** {self.warning_count()}")
        lines.append(f"**Checks Run:** {len(self.checks_run)}")
        lines.append("")

        if self.issues:
            lines.append("## Issues Found")
            lines.append("")

            # Group by severity
            for severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING, ValidationSeverity.INFO]:
                severity_issues = [i for i in self.issues if i.severity == severity]
                if severity_issues:
                    lines.append(f"### {severity.value.upper()}")
                    lines.append("")
                    for issue in severity_issues:
                        context = f" (edge: {issue.edge_id})" if issue.edge_id else ""
                        context += f" (node: {issue.node_id})" if issue.node_id else ""
                        lines.append(f"- **{issue.check_id}**{context}: {issue.message}")
                    lines.append("")
        else:
            lines.append("*No issues found.*")

        return "\n".join(lines)


class DAGValidator:
    """
    Validator for causal DAG specifications and outputs.

    Runs automated checks to ensure:
    1. DAG structure is valid (acyclic, all nodes/edges defined)
    2. Units are specified for all edges
    3. Edge types are properly labeled
    4. EdgeCards match DAG specifications
    5. Report output matches EdgeCard data
    """

    def __init__(self, dag_config: dict[str, Any]):
        """
        Initialize validator with DAG configuration.

        Args:
            dag_config: Parsed DAG YAML configuration
        """
        self.dag = dag_config
        self.nodes = {n["id"]: n for n in dag_config.get("nodes", [])}
        self.edges = {e["id"]: e for e in dag_config.get("edges", [])}

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "DAGValidator":
        """Load DAG from YAML file and create validator."""
        with open(yaml_path) as f:
            dag_config = yaml.safe_load(f)
        return cls(dag_config)

    # =========================================================================
    # PRE-ESTIMATION CHECKS
    # =========================================================================

    def validate_pre_estimation(self) -> ValidationResult:
        """
        Run all pre-estimation validation checks.

        These checks should pass before running any estimators.
        """
        result = ValidationResult(passed=True)

        # Run all pre-checks
        self._check_dag_acyclic(result)
        self._check_unit_presence(result)
        self._check_edge_type_presence(result)
        self._check_node_sources_defined(result)
        self._check_edge_nodes_exist(result)
        self._check_target_reachable(result)
        self._check_identity_deps_complete(result)
        self._check_sink_nodes(result)
        self._check_edge_id_syntax(result)
        self._check_identity_formula_double_transform(result)

        return result

    def _check_dag_acyclic(self, result: ValidationResult) -> None:
        """Check that DAG has no cycles."""
        result.checks_run.append("dag_acyclic")

        # Build adjacency list
        adj: dict[str, list[str]] = {n: [] for n in self.nodes}
        for edge in self.edges.values():
            from_node = edge["from"]
            to_node = edge["to"]
            if from_node in adj:
                adj[from_node].append(to_node)

        # DFS to detect cycles
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self.nodes}

        def has_cycle(node: str) -> bool:
            color[node] = GRAY
            for neighbor in adj.get(node, []):
                if color.get(neighbor) == GRAY:
                    return True
                if color.get(neighbor) == WHITE and has_cycle(neighbor):
                    return True
            color[node] = BLACK
            return False

        for node in self.nodes:
            if color[node] == WHITE:
                if has_cycle(node):
                    result.add_issue(ValidationIssue(
                        check_id="dag_acyclic",
                        severity=ValidationSeverity.ERROR,
                        message="DAG contains a cycle",
                        details={"starting_node": node},
                    ))
                    return

    def _check_unit_presence(self, result: ValidationResult) -> None:
        """Check that all edges have unit_specification."""
        result.checks_run.append("unit_presence")

        for edge_id, edge in self.edges.items():
            unit_spec = edge.get("unit_specification", {})
            treatment_unit = unit_spec.get("treatment_unit", "")
            outcome_unit = unit_spec.get("outcome_unit", "")

            if not treatment_unit:
                result.add_issue(ValidationIssue(
                    check_id="unit_presence",
                    severity=ValidationSeverity.ERROR,
                    message="Missing treatment_unit in unit_specification",
                    edge_id=edge_id,
                ))

            if not outcome_unit:
                result.add_issue(ValidationIssue(
                    check_id="unit_presence",
                    severity=ValidationSeverity.ERROR,
                    message="Missing outcome_unit in unit_specification",
                    edge_id=edge_id,
                ))

    def _check_edge_type_presence(self, result: ValidationResult) -> None:
        """Check that all edges have edge_type specified."""
        result.checks_run.append("edge_type_presence")

        valid_types = {"causal", "reaction_function", "mechanical", "immutable", "identity"}

        for edge_id, edge in self.edges.items():
            edge_type = edge.get("edge_type")

            if not edge_type:
                result.add_issue(ValidationIssue(
                    check_id="edge_type_presence",
                    severity=ValidationSeverity.ERROR,
                    message="Missing edge_type",
                    edge_id=edge_id,
                ))
            elif edge_type not in valid_types:
                result.add_issue(ValidationIssue(
                    check_id="edge_type_presence",
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid edge_type '{edge_type}'. Must be one of: {valid_types}",
                    edge_id=edge_id,
                ))

    def _check_node_sources_defined(self, result: ValidationResult) -> None:
        """Check that observed nodes have data sources defined."""
        result.checks_run.append("node_source_defined")

        for node_id, node in self.nodes.items():
            if node.get("observed", True) and not node.get("derived", False):
                source = node.get("source", {})
                if not source.get("preferred") and not source.get("fallback"):
                    result.add_issue(ValidationIssue(
                        check_id="node_source_defined",
                        severity=ValidationSeverity.WARNING,
                        message="Observed node has no data source defined",
                        node_id=node_id,
                    ))

    def _check_edge_nodes_exist(self, result: ValidationResult) -> None:
        """Check that all edge endpoints reference existing nodes."""
        result.checks_run.append("edge_nodes_exist")

        for edge_id, edge in self.edges.items():
            from_node = edge.get("from")
            to_node = edge.get("to")

            if from_node not in self.nodes:
                result.add_issue(ValidationIssue(
                    check_id="edge_nodes_exist",
                    severity=ValidationSeverity.ERROR,
                    message=f"Edge 'from' node '{from_node}' does not exist",
                    edge_id=edge_id,
                ))

            if to_node not in self.nodes:
                result.add_issue(ValidationIssue(
                    check_id="edge_nodes_exist",
                    severity=ValidationSeverity.ERROR,
                    message=f"Edge 'to' node '{to_node}' does not exist",
                    edge_id=edge_id,
                ))

    def _check_target_reachable(self, result: ValidationResult) -> None:
        """Check that the target node is reachable from at least one root."""
        result.checks_run.append("target_reachable")

        target = self.dag.get("metadata", {}).get("target_node", "")
        if not target or target not in self.nodes:
            return

        # Build reverse adjacency (to -> [from])
        rev_adj: dict[str, list[str]] = {n: [] for n in self.nodes}
        for edge in self.edges.values():
            tn = edge.get("to", "")
            fn = edge.get("from", "")
            if tn in rev_adj:
                rev_adj[tn].append(fn)

        # BFS backward from target
        visited: set[str] = set()
        frontier = [target]
        while frontier:
            node = frontier.pop()
            if node in visited:
                continue
            visited.add(node)
            for pred in rev_adj.get(node, []):
                if pred not in visited:
                    frontier.append(pred)

        # Roots are nodes with no incoming edges
        incoming = {edge.get("to") for edge in self.edges.values()}
        roots = [n for n in self.nodes if n not in incoming]

        if not roots:
            return  # degenerate

        # Target itself doesn't count as a valid root
        reachable_roots = [r for r in roots if r in visited and r != target]
        if not reachable_roots:
            result.add_issue(ValidationIssue(
                check_id="target_reachable",
                severity=ValidationSeverity.ERROR,
                message=(
                    f"Target node '{target}' is unreachable from any root node. "
                    f"Roots: {roots}; reachable from target (backward): "
                    f"{sorted(visited)}"
                ),
            ))

    def _check_identity_deps_complete(self, result: ValidationResult) -> None:
        """Check that all identity-formula dependencies have edges into derived nodes."""
        result.checks_run.append("identity_deps_complete")

        edge_pairs = {
            (e.get("from"), e.get("to")) for e in self.edges.values()
        }

        for node_id, node in self.nodes.items():
            if not node.get("derived"):
                continue
            identity = node.get("identity")
            if not identity:
                continue
            deps = identity.get("depends_on") or node.get("depends_on", [])
            for dep in deps:
                if dep not in self.nodes:
                    continue  # dependency not in DAG at all
                if (dep, node_id) not in edge_pairs:
                    result.add_issue(ValidationIssue(
                        check_id="identity_deps_complete",
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Derived node '{node_id}' has identity dependency "
                            f"'{dep}' but no edge {dep} -> {node_id} exists"
                        ),
                        node_id=node_id,
                        details={"missing_dep": dep},
                    ))

    def _check_sink_nodes(self, result: ValidationResult) -> None:
        """Check for non-target sink nodes (incoming edges but no outgoing)."""
        result.checks_run.append("sink_node_not_target")

        target = self.dag.get("metadata", {}).get("target_node", "")
        outgoing: dict[str, int] = {n: 0 for n in self.nodes}
        incoming: dict[str, int] = {n: 0 for n in self.nodes}

        for edge in self.edges.values():
            fn = edge.get("from", "")
            tn = edge.get("to", "")
            if fn in outgoing:
                outgoing[fn] += 1
            if tn in incoming:
                incoming[tn] += 1

        for node_id in self.nodes:
            if node_id == target:
                continue
            if incoming[node_id] > 0 and outgoing[node_id] == 0:
                result.add_issue(ValidationIssue(
                    check_id="sink_node_not_target",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Node '{node_id}' is a sink (has incoming edges but "
                        f"no outgoing edges) and is not the target node"
                    ),
                    node_id=node_id,
                ))

    def _check_edge_id_syntax(self, result: ValidationResult) -> None:
        """Check that edge IDs use valid snake_case syntax."""
        result.checks_run.append("edge_id_syntax")

        valid_pattern = re.compile(r'^[a-z0-9][a-z0-9_]*[a-z0-9]$|^[a-z0-9]$')
        for edge_id in self.edges:
            if not valid_pattern.match(edge_id):
                result.add_issue(ValidationIssue(
                    check_id="edge_id_syntax",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Edge ID '{edge_id}' contains invalid characters. "
                        f"Must be snake_case (lowercase, digits, underscores)"
                    ),
                    edge_id=edge_id,
                ))

    def _check_identity_formula_double_transform(self, result: ValidationResult) -> None:
        """Warn when an identity formula applies log() to a node with transforms: [log]."""
        result.checks_run.append("identity_formula_double_transform")

        for node_id, node in self.nodes.items():
            identity = node.get("identity")
            if not identity:
                continue
            formula = identity.get("formula", "")
            if not formula:
                continue
            deps = identity.get("depends_on") or node.get("depends_on", [])
            for dep in deps:
                dep_node = self.nodes.get(dep)
                if not dep_node:
                    continue
                dep_transforms = dep_node.get("transforms", [])
                if "log" in dep_transforms and f"log({dep})" in formula:
                    result.add_issue(ValidationIssue(
                        check_id="identity_formula_double_transform",
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Formula for '{node_id}' applies log() to '{dep}' "
                            f"which already has transforms: [log]. "
                            f"Risk of double-logging."
                        ),
                        node_id=node_id,
                        details={"formula": formula, "dep": dep},
                    ))

    # =========================================================================
    # POST-ESTIMATION CHECKS
    # =========================================================================

    def validate_post_estimation(
        self,
        edge_cards: dict[str, EdgeCard],
    ) -> ValidationResult:
        """
        Run post-estimation validation checks.

        Args:
            edge_cards: Dictionary of edge_id -> EdgeCard
        """
        result = ValidationResult(passed=True)

        self._check_n_consistency(result, edge_cards)
        self._check_unit_in_card(result, edge_cards)
        self._check_reaction_function_labeled(result, edge_cards)
        self._check_interpolation_fraction(result, edge_cards)
        self._check_sign_consistency(result, edge_cards)

        return result

    def _check_n_consistency(
        self,
        result: ValidationResult,
        edge_cards: dict[str, EdgeCard],
    ) -> None:
        """Check that N values are consistently reported."""
        result.checks_run.append("n_consistency")

        for edge_id, card in edge_cards.items():
            if card.estimates:
                n_cal = card.estimates.n_calendar_periods
                n_eff = card.estimates.n_effective_obs_h0

                if n_cal is None and n_eff is None:
                    result.add_issue(ValidationIssue(
                        check_id="n_consistency",
                        severity=ValidationSeverity.WARNING,
                        message="No sample size (N) reported in EdgeCard",
                        edge_id=edge_id,
                    ))
                elif n_eff is not None and n_cal is not None and n_eff > n_cal:
                    result.add_issue(ValidationIssue(
                        check_id="n_consistency",
                        severity=ValidationSeverity.ERROR,
                        message=f"n_effective ({n_eff}) > n_calendar ({n_cal}), which is impossible",
                        edge_id=edge_id,
                    ))

    def _check_unit_in_card(
        self,
        result: ValidationResult,
        edge_cards: dict[str, EdgeCard],
    ) -> None:
        """Check that EdgeCards have units populated."""
        result.checks_run.append("unit_in_card")

        for edge_id, card in edge_cards.items():
            if card.estimates:
                if not card.estimates.treatment_unit:
                    result.add_issue(ValidationIssue(
                        check_id="unit_in_card",
                        severity=ValidationSeverity.ERROR,
                        message="EdgeCard missing treatment_unit",
                        edge_id=edge_id,
                    ))
                if not card.estimates.outcome_unit:
                    result.add_issue(ValidationIssue(
                        check_id="unit_in_card",
                        severity=ValidationSeverity.ERROR,
                        message="EdgeCard missing outcome_unit",
                        edge_id=edge_id,
                    ))

    def _check_reaction_function_labeled(
        self,
        result: ValidationResult,
        edge_cards: dict[str, EdgeCard],
    ) -> None:
        """Check that reaction function edges are properly labeled."""
        result.checks_run.append("reaction_function_labeled")

        for edge_id, edge in self.edges.items():
            if edge.get("edge_type") == "reaction_function":
                card = edge_cards.get(edge_id)
                if card:
                    # Check that forbidden_uses includes policy_counterfactual
                    forbidden = card.interpretation.forbidden_uses
                    if "policy_counterfactual" not in forbidden:
                        result.add_issue(ValidationIssue(
                            check_id="reaction_function_labeled",
                            severity=ValidationSeverity.WARNING,
                            message="Reaction function edge should forbid 'policy_counterfactual' use",
                            edge_id=edge_id,
                        ))

                    # Check that allowed_uses is restrictive
                    allowed = card.interpretation.allowed_uses
                    if "shock_counterfactual" in allowed:
                        result.add_issue(ValidationIssue(
                            check_id="reaction_function_labeled",
                            severity=ValidationSeverity.ERROR,
                            message="Reaction function edge should NOT allow 'shock_counterfactual' use",
                            edge_id=edge_id,
                        ))

    def _check_interpolation_fraction(
        self,
        result: ValidationResult,
        edge_cards: dict[str, EdgeCard],
        threshold: float = 0.30,
    ) -> None:
        """Check that interpolated observations are below threshold."""
        result.checks_run.append("interpolation_fraction")

        for edge_id, card in edge_cards.items():
            prov = card.data_provenance
            if prov.missing_rate is not None and prov.missing_rate > threshold:
                result.add_issue(ValidationIssue(
                    check_id="interpolation_fraction",
                    severity=ValidationSeverity.WARNING,
                    message=f"Interpolated fraction ({prov.missing_rate:.1%}) exceeds {threshold:.0%}",
                    edge_id=edge_id,
                ))

    def _check_sign_consistency(
        self,
        result: ValidationResult,
        edge_cards: dict[str, EdgeCard],
    ) -> None:
        """Check that estimated signs match expected signs from DAG."""
        result.checks_run.append("sign_consistency")

        for edge_id, edge in self.edges.items():
            card = edge_cards.get(edge_id)
            if not card or not card.estimates:
                continue

            plausibility = edge.get("acceptance_criteria", {}).get("plausibility", {})
            expected_sign = plausibility.get("expected_sign")

            if expected_sign and expected_sign != "any":
                actual = card.estimates.point
                if expected_sign == "positive" and actual < 0:
                    result.add_issue(ValidationIssue(
                        check_id="sign_consistency",
                        severity=ValidationSeverity.WARNING,
                        message=f"Expected positive sign, got {actual:.4f}",
                        edge_id=edge_id,
                    ))
                elif expected_sign == "negative" and actual > 0:
                    result.add_issue(ValidationIssue(
                        check_id="sign_consistency",
                        severity=ValidationSeverity.WARNING,
                        message=f"Expected negative sign, got {actual:.4f}",
                        edge_id=edge_id,
                    ))

    # =========================================================================
    # IDENTIFIABILITY SCREEN INTEGRATION
    # =========================================================================

    def validate_identifiability(
        self,
        edge_cards: dict[str, EdgeCard],
    ) -> tuple[ValidationResult, dict[str, IdentifiabilityResult]]:
        """
        Run identifiability screen on all edge cards.

        Returns:
            Tuple of (ValidationResult, dict of edge_id -> IdentifiabilityResult)
        """
        result = ValidationResult(passed=True)
        result.checks_run.append("identifiability_screen")
        screen = IdentifiabilityScreen()
        id_results: dict[str, IdentifiabilityResult] = {}

        for edge_id, card in edge_cards.items():
            design = card.spec_details.design if card.spec_details else ""
            id_result = screen.screen_post_estimation(
                edge_id=edge_id,
                design=design,
                diagnostics=card.diagnostics,
            )
            id_results[edge_id] = id_result

            # Flag significant-but-not-identified
            if (card.estimates and card.estimates.pvalue is not None
                    and card.estimates.pvalue < 0.05
                    and id_result.claim_level != "IDENTIFIED_CAUSAL"):
                result.add_issue(ValidationIssue(
                    check_id="significant_but_not_identified",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"p={card.estimates.pvalue:.4f} but claim_level={id_result.claim_level}. "
                        "Statistical significance does not establish causation."
                    ),
                    edge_id=edge_id,
                ))

            # Flag blocked identification
            if id_result.claim_level == "BLOCKED_ID":
                result.add_issue(ValidationIssue(
                    check_id="identification_blocked",
                    severity=ValidationSeverity.ERROR,
                    message=f"Identification blocked: {', '.join(id_result.required_structure_missing)}",
                    edge_id=edge_id,
                ))

        return result, id_results

    # =========================================================================
    # REPORT CONSISTENCY CHECKS
    # =========================================================================

    def validate_report_consistency(
        self,
        report_content: str,
        edge_cards: dict[str, EdgeCard],
    ) -> ValidationResult:
        """
        Check that report content matches EdgeCard data.

        Args:
            report_content: Markdown content of the report
            edge_cards: Dictionary of edge_id -> EdgeCard
        """
        result = ValidationResult(passed=True)

        self._check_report_vs_card_match(result, report_content, edge_cards)
        self._check_unit_table_present(result, report_content)
        self._check_reaction_function_warning(result, report_content)

        return result

    def _check_report_vs_card_match(
        self,
        result: ValidationResult,
        report_content: str,
        edge_cards: dict[str, EdgeCard],
    ) -> None:
        """Check that report values match EdgeCard values."""
        result.checks_run.append("report_vs_card_match")

        # Parse edge IDs from report (look for backtick-quoted edge names)
        edge_pattern = re.compile(r"`([a-z_]+_to_[a-z_]+)`")
        found_edges = set(edge_pattern.findall(report_content))

        for edge_id in found_edges:
            card = edge_cards.get(edge_id)
            if not card or not card.estimates:
                continue

            # Check if the point estimate appears in the report
            point_str = f"{card.estimates.point:.2f}"
            if point_str not in report_content and f"{card.estimates.point:.4f}" not in report_content:
                # Allow some tolerance for formatting differences
                point_pattern = re.compile(rf"{re.escape(edge_id)}.*?(-?[\d.]+)")
                matches = point_pattern.findall(report_content)
                if not any(abs(float(m) - card.estimates.point) < 0.01 for m in matches if m):
                    result.add_issue(ValidationIssue(
                        check_id="report_vs_card_match",
                        severity=ValidationSeverity.WARNING,
                        message=f"Point estimate {card.estimates.point:.4f} not found in report",
                        edge_id=edge_id,
                    ))

    def _check_unit_table_present(
        self,
        result: ValidationResult,
        report_content: str,
    ) -> None:
        """Check that Unit Normalization Reference table is present."""
        result.checks_run.append("unit_table_present")

        if "Unit Normalization" not in report_content:
            result.add_issue(ValidationIssue(
                check_id="unit_table_present",
                severity=ValidationSeverity.WARNING,
                message="Report missing 'Unit Normalization Reference' section",
            ))

    def _check_reaction_function_warning(
        self,
        result: ValidationResult,
        report_content: str,
    ) -> None:
        """Check that reaction function edges have warnings in report."""
        result.checks_run.append("reaction_function_warning")

        reaction_edges = [
            eid for eid, e in self.edges.items()
            if e.get("edge_type") == "reaction_function"
        ]

        for edge_id in reaction_edges:
            if edge_id in report_content:
                # Check for warning near this edge
                # Look for "reaction function" or "NOT" or "endogenous" in context
                edge_pattern = re.compile(
                    rf"{re.escape(edge_id)}.*?(reaction|NOT|endogenous|warning)",
                    re.IGNORECASE | re.DOTALL
                )
                if not edge_pattern.search(report_content):
                    result.add_issue(ValidationIssue(
                        check_id="reaction_function_warning",
                        severity=ValidationSeverity.WARNING,
                        message="Reaction function edge appears in report without warning",
                        edge_id=edge_id,
                    ))


def validate_mode_consistency(
    edge_cards: dict[str, EdgeCard],
    query_mode: str,
    critical_path_edges: list[str] | None = None,
) -> ValidationResult:
    """Validate that critical-path edges have roles allowed in the query mode."""
    from shared.agentic.query_mode import QueryModeConfig

    result = ValidationResult(passed=True)
    result.checks_run.append("mode_consistency")

    config = QueryModeConfig.load()
    mode_spec = config.get_spec(query_mode)

    check_edges = critical_path_edges or list(edge_cards.keys())
    for edge_id in check_edges:
        card = edge_cards.get(edge_id)
        if not card:
            continue
        role = card.propagation_role.role
        if role not in mode_spec.propagation_requires:
            result.add_issue(ValidationIssue(
                check_id="mode_consistency",
                severity=ValidationSeverity.ERROR,
                message=(
                    f"Edge role '{role}' not allowed for propagation in "
                    f"{query_mode} mode (requires {mode_spec.propagation_requires})"
                ),
                edge_id=edge_id,
            ))
    return result


def validate_double_counting(
    edge_cards: dict[str, EdgeCard],
    dag_config: dict[str, Any],
    query_mode: str,
) -> ValidationResult:
    """Check for direct + indirect path overlap (double counting risk)."""
    from shared.agentic.query_mode import QueryModeConfig

    result = ValidationResult(passed=True)
    result.checks_run.append("double_counting")

    config = QueryModeConfig.load()
    mode_spec = config.get_spec(query_mode)

    # Build adjacency from edges allowed for propagation in this mode
    edges_raw = dag_config.get("edges", [])
    adj: dict[str, list[tuple[str, str]]] = {}  # from_node -> [(to_node, edge_id)]
    for e in edges_raw:
        eid = e["id"]
        card = edge_cards.get(eid)
        if card and card.propagation_role.role in mode_spec.propagation_requires:
            fn = e["from"]
            tn = e["to"]
            adj.setdefault(fn, []).append((tn, eid))

    # For each pair of nodes (A, C), check if there is both:
    #   - a direct edge A->C
    #   - an indirect path A->B->...->C (length >= 2)
    direct_edges: dict[tuple[str, str], str] = {}
    for e in edges_raw:
        eid = e["id"]
        card = edge_cards.get(eid)
        if card and card.propagation_role.role in mode_spec.propagation_requires:
            direct_edges[(e["from"], e["to"])] = eid

    # BFS from each node to find reachable nodes via paths of length >= 2
    for (from_n, to_n), direct_eid in direct_edges.items():
        # Check if to_n is reachable from from_n via a path of length >= 2
        visited = set()
        frontier = []
        for next_n, _ in adj.get(from_n, []):
            if next_n != to_n:
                frontier.append(next_n)
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            for next_n, _ in adj.get(current, []):
                if next_n == to_n:
                    result.add_issue(ValidationIssue(
                        check_id="double_counting",
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Direct edge {direct_eid} ({from_n}->{to_n}) "
                            f"and indirect path both active; double counting risk"
                        ),
                        edge_id=direct_eid,
                        details={"indirect_via": current},
                    ))
                    break
                if next_n not in visited:
                    frontier.append(next_n)

    return result


def validate_edge_type_presence_for_mode(
    edge_cards: dict[str, EdgeCard],
    dag_config: dict[str, Any],
    dag_mode: str = "EXPLORATION",
) -> ValidationResult:
    """Check that edges have explicit edge_type (required in CONFIRMATION)."""
    result = ValidationResult(passed=True)
    result.checks_run.append("edge_type_explicit")

    edges_raw = dag_config.get("edges", [])
    for e in edges_raw:
        eid = e["id"]
        explicit_type = e.get("edge_type", "")
        if not explicit_type:
            severity = (
                ValidationSeverity.ERROR
                if dag_mode == "CONFIRMATION"
                else ValidationSeverity.INFO
            )
            result.add_issue(ValidationIssue(
                check_id="edge_type_explicit",
                severity=severity,
                message=(
                    f"edge_type not explicit (inferred from edge_status). "
                    f"{'Required' if dag_mode == 'CONFIRMATION' else 'Info'} in {dag_mode} mode."
                ),
                edge_id=eid,
            ))
    return result


def run_full_validation(
    dag_path: str | Path,
    edge_cards: dict[str, EdgeCard] | None = None,
    report_path: str | Path | None = None,
    query_mode: str | None = None,
    critical_path_edges: list[str] | None = None,
) -> ValidationResult:
    """
    Run full validation pipeline.

    Args:
        dag_path: Path to DAG YAML file
        edge_cards: Optional dict of EdgeCards for post-estimation checks
        report_path: Optional path to report markdown for consistency checks

    Returns:
        Combined ValidationResult from all checks
    """
    validator = DAGValidator.from_yaml(dag_path)

    # Combine all results
    combined = ValidationResult(passed=True)

    # Pre-estimation checks
    pre_result = validator.validate_pre_estimation()
    combined.issues.extend(pre_result.issues)
    combined.checks_run.extend(pre_result.checks_run)
    if not pre_result.passed:
        combined.passed = False

    # Post-estimation checks (if edge cards provided)
    if edge_cards:
        post_result = validator.validate_post_estimation(edge_cards)
        combined.issues.extend(post_result.issues)
        combined.checks_run.extend(post_result.checks_run)
        if not post_result.passed:
            combined.passed = False

        # Identifiability screen
        id_result, _ = validator.validate_identifiability(edge_cards)
        combined.issues.extend(id_result.issues)
        combined.checks_run.extend(id_result.checks_run)
        if not id_result.passed:
            combined.passed = False

    # Query mode checks (if mode and edge cards provided)
    if edge_cards and query_mode:
        mode_result = validate_mode_consistency(
            edge_cards, query_mode, critical_path_edges,
        )
        combined.issues.extend(mode_result.issues)
        combined.checks_run.extend(mode_result.checks_run)
        if not mode_result.passed:
            combined.passed = False

        # Double counting check
        with open(dag_path) as f:
            dag_raw = yaml.safe_load(f)
        dc_result = validate_double_counting(edge_cards, dag_raw, query_mode)
        combined.issues.extend(dc_result.issues)
        combined.checks_run.extend(dc_result.checks_run)
        if not dc_result.passed:
            combined.passed = False

    # Report consistency checks (if report provided)
    if report_path and edge_cards:
        with open(report_path) as f:
            report_content = f.read()
        report_result = validator.validate_report_consistency(report_content, edge_cards)
        combined.issues.extend(report_result.issues)
        combined.checks_run.extend(report_result.checks_run)
        if not report_result.passed:
            combined.passed = False

    return combined


# =========================================================================
# EdgeCard-level validation
# =========================================================================

def validate_edge_card(card: EdgeCard) -> ValidationResult:
    """
    Validate a single EdgeCard for numeric and structural integrity.

    Checks NaN/Inf estimates, negative SEs, CI ordering, p-value range,
    IRF length mismatches, score bounds, and rating consistency.

    Returns ValidationResult (passed=True means no ERRORs found).
    """
    result = ValidationResult(passed=True)
    result.checks_run.append("validate_edge_card")
    eid = card.edge_id

    est = card.estimates

    # Skip all estimate checks if estimates is None
    if est is not None:
        # 1. NaN/Inf point estimate
        if est.point is None or _is_bad_float(est.point):
            result.add_issue(ValidationIssue(
                check_id="nan_inf_point", severity=ValidationSeverity.ERROR,
                message="Point estimate is NaN, Inf, or None", edge_id=eid,
            ))

        # 2. NaN/Inf CI bounds
        if est.ci_95 is not None:
            if _is_bad_float(est.ci_95[0]) or _is_bad_float(est.ci_95[1]):
                result.add_issue(ValidationIssue(
                    check_id="nan_inf_ci", severity=ValidationSeverity.ERROR,
                    message="CI bound is NaN or Inf", edge_id=eid,
                ))

        # 3. NaN/Inf SE
        if _is_bad_float(est.se):
            result.add_issue(ValidationIssue(
                check_id="nan_inf_se", severity=ValidationSeverity.ERROR,
                message="Standard error is NaN or Inf", edge_id=eid,
            ))

        # 4. Negative SE (only if SE is finite)
        if est.se is not None and not _is_bad_float(est.se) and est.se < 0:
            result.add_issue(ValidationIssue(
                check_id="negative_se", severity=ValidationSeverity.ERROR,
                message=f"Negative standard error: {est.se}", edge_id=eid,
            ))

        # 5. CI order (skip if either bound is bad)
        if (est.ci_95 is not None
                and not _is_bad_float(est.ci_95[0])
                and not _is_bad_float(est.ci_95[1])
                and est.ci_95[0] > est.ci_95[1]):
            result.add_issue(ValidationIssue(
                check_id="ci_order", severity=ValidationSeverity.ERROR,
                message=f"CI lower ({est.ci_95[0]}) > upper ({est.ci_95[1]})",
                edge_id=eid,
            ))

        # 6. Point outside CI (WARNING, skip if any value is bad or degenerate)
        if (est.ci_95 is not None
                and est.point is not None
                and not _is_bad_float(est.point)
                and not _is_bad_float(est.ci_95[0])
                and not _is_bad_float(est.ci_95[1])
                and not (est.ci_95[0] == est.ci_95[1] == est.point)):
            if est.point < est.ci_95[0] or est.point > est.ci_95[1]:
                result.add_issue(ValidationIssue(
                    check_id="point_outside_ci", severity=ValidationSeverity.WARNING,
                    message=f"Point {est.point} outside CI [{est.ci_95[0]}, {est.ci_95[1]}]",
                    edge_id=eid,
                ))

        # 7. NaN/Inf p-value
        if _is_bad_float(est.pvalue):
            result.add_issue(ValidationIssue(
                check_id="nan_inf_pvalue", severity=ValidationSeverity.ERROR,
                message="p-value is NaN or Inf", edge_id=eid,
            ))

        # 8. p-value range (skip if NaN)
        if (est.pvalue is not None
                and not _is_bad_float(est.pvalue)
                and not (0 <= est.pvalue <= 1)):
            result.add_issue(ValidationIssue(
                check_id="pvalue_range", severity=ValidationSeverity.ERROR,
                message=f"p-value out of [0,1]: {est.pvalue}", edge_id=eid,
            ))

        # 9. IRF length mismatch
        if est.horizons is not None:
            n_h = len(est.horizons)
            for vec_name in ("irf", "irf_ci_lower", "irf_ci_upper"):
                vec = getattr(est, vec_name, None)
                if vec is not None and len(vec) != n_h:
                    result.add_issue(ValidationIssue(
                        check_id="irf_length_mismatch", severity=ValidationSeverity.ERROR,
                        message=f"{vec_name} length ({len(vec)}) != horizons length ({n_h})",
                        edge_id=eid,
                    ))

        # 13. Missing units (WARNING)
        if not est.treatment_unit:
            result.add_issue(ValidationIssue(
                check_id="missing_treatment_unit", severity=ValidationSeverity.WARNING,
                message="Missing treatment_unit", edge_id=eid,
            ))
        if not est.outcome_unit:
            result.add_issue(ValidationIssue(
                check_id="missing_outcome_unit", severity=ValidationSeverity.WARNING,
                message="Missing outcome_unit", edge_id=eid,
            ))

    # 10. NaN/Inf credibility score
    if _is_bad_float(card.credibility_score):
        result.add_issue(ValidationIssue(
            check_id="nan_inf_score", severity=ValidationSeverity.ERROR,
            message="Credibility score is NaN or Inf", edge_id=eid,
        ))

    # 11. Score out of [0,1] (skip if NaN)
    if (card.credibility_score is not None
            and not _is_bad_float(card.credibility_score)
            and not (0 <= card.credibility_score <= 1)):
        result.add_issue(ValidationIssue(
            check_id="score_range", severity=ValidationSeverity.ERROR,
            message=f"Credibility score out of [0,1]: {card.credibility_score}",
            edge_id=eid,
        ))

    # 12. Rating too generous (WARNING) — uses centralized thresholds
    if (card.credibility_score is not None
            and not _is_bad_float(card.credibility_score)
            and card.credibility_rating is not None):
        expected_rating = rating_from_score(card.credibility_score)
        # "too generous" = card rating is strictly better (earlier in A>B>C>D)
        rating_order = {"A": 0, "B": 1, "C": 2, "D": 3}
        card_rank = rating_order.get(card.credibility_rating, 4)
        expected_rank = rating_order.get(expected_rating, 4)
        if card_rank < expected_rank:
            result.add_issue(ValidationIssue(
                check_id="rating_too_generous", severity=ValidationSeverity.WARNING,
                message=(
                    f"Rating '{card.credibility_rating}' is more generous than "
                    f"expected '{expected_rating}' for score {card.credibility_score:.3f}"
                ),
                edge_id=eid,
            ))

    return result


# =========================================================================
# Chain unit compatibility
# =========================================================================

_UNIT_STOPWORDS = frozenset({
    "in", "of", "per", "to", "and", "or", "the", "a", "an",
    "change", "changes", "increase", "decrease", "percent",
    "1", "1%", "sd",
})


def _extract_unit_tokens(unit_str: str) -> set[str]:
    """Normalize unit string to meaningful keyword tokens."""
    # Replace punctuation (/, (, ), %, comma) with spaces
    cleaned = re.sub(r'[/(),%.]+', ' ', unit_str)
    tokens = cleaned.lower().split()
    # Drop short tokens (len < 2) and stopwords
    return {t for t in tokens if len(t) >= 2 and t not in _UNIT_STOPWORDS}


def validate_chain_units(
    edge_cards: dict[str, EdgeCard],
    dag_edges: dict[str, tuple[str, str]],
) -> ValidationResult:
    """
    Validate unit compatibility for consecutive edges sharing intermediate nodes.

    For each pair of consecutive edges (A->B, B->C), checks that the
    upstream outcome_unit and downstream treatment_unit share meaningful
    keyword tokens.

    Args:
        edge_cards: dict of edge_id -> EdgeCard
        dag_edges: dict of edge_id -> (treatment_node, outcome_node)

    Returns:
        ValidationResult with WARNING issues for incompatible unit pairs
    """
    result = ValidationResult(passed=True)
    result.checks_run.append("validate_chain_units")

    # Build node->edge mapping
    outcome_of: dict[str, list[str]] = {}  # node -> edge_ids where node is outcome
    treatment_of: dict[str, list[str]] = {}  # node -> edge_ids where node is treatment

    for edge_id, (treat_node, out_node) in dag_edges.items():
        if edge_id not in edge_cards:
            continue
        outcome_of.setdefault(out_node, []).append(edge_id)
        treatment_of.setdefault(treat_node, []).append(edge_id)

    # For each intermediate node, check unit compatibility
    intermediate_nodes = set(outcome_of.keys()) & set(treatment_of.keys())

    for node in intermediate_nodes:
        for upstream_eid in outcome_of[node]:
            upstream_card = edge_cards.get(upstream_eid)
            if not upstream_card or not upstream_card.estimates:
                continue
            upstream_unit = upstream_card.estimates.outcome_unit or ""
            if not upstream_unit.strip():
                continue

            for downstream_eid in treatment_of[node]:
                downstream_card = edge_cards.get(downstream_eid)
                if not downstream_card or not downstream_card.estimates:
                    continue
                downstream_unit = downstream_card.estimates.treatment_unit or ""
                if not downstream_unit.strip():
                    continue

                upstream_tokens = _extract_unit_tokens(upstream_unit)
                downstream_tokens = _extract_unit_tokens(downstream_unit)

                if not upstream_tokens or not downstream_tokens:
                    continue

                overlap = upstream_tokens & downstream_tokens
                if not overlap:
                    result.add_issue(ValidationIssue(
                        check_id="chain_unit_mismatch",
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Unit mismatch at node '{node}': "
                            f"upstream '{upstream_eid}' outcome_unit='{upstream_unit}' "
                            f"vs downstream '{downstream_eid}' treatment_unit='{downstream_unit}'"
                        ),
                        edge_id=f"{upstream_eid}->{downstream_eid}",
                    ))

    return result
