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

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from shared.agentic.output.edge_card import EdgeCard
from shared.agentic.identification.screen import IdentifiabilityScreen, IdentifiabilityResult


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

        valid_types = {"causal", "reaction_function", "mechanical", "immutable"}

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


def run_full_validation(
    dag_path: str | Path,
    edge_cards: dict[str, EdgeCard] | None = None,
    report_path: str | Path | None = None,
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
