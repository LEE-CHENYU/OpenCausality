"""
DAG Semantic Validator.

Validates DAG specifications for:
1. Referential integrity (edge endpoints exist)
2. Cycle detection via temporal expansion
3. Identity dependency consistency
4. Forbidden controls (auto-mark descendants of treatment)
5. Backdoor adjustment set candidates
6. Scope consistency (BNS + KSPI scope matching)
7. Bidirectional policy rate modeling
8. RWA mechanism validation (no direct CPI -> RWA)
9. Immutable evidence artifact protection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from shared.agentic.dag.parser import DAGSpec, EdgeSpec, NodeSpec
from shared.agentic.dag.temporal import check_temporal_cycle, expand_temporal_dag

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    ERROR = "error"      # Must be fixed
    WARNING = "warning"  # Should be reviewed
    INFO = "info"        # For information only


@dataclass
class ValidationIssue:
    """A single validation issue."""

    code: str
    message: str
    severity: ValidationSeverity
    location: str = ""  # e.g., "node:income" or "edge:fx_to_cpi"
    suggestion: str = ""

    def __str__(self):
        prefix = f"[{self.severity.value.upper()}]"
        loc = f" ({self.location})" if self.location else ""
        suggestion = f"\n  Suggestion: {self.suggestion}" if self.suggestion else ""
        return f"{prefix}{loc} {self.message}{suggestion}"


class ValidationError(Exception):
    """Exception raised when validation fails."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        error_msgs = [str(i) for i in issues if i.severity == ValidationSeverity.ERROR]
        super().__init__(f"DAG validation failed with {len(error_msgs)} errors:\n" +
                        "\n".join(error_msgs))


@dataclass
class ForbiddenControlsResult:
    """Result of forbidden controls computation for an edge."""

    edge_id: str
    treatment_node: str
    descendants: set[str]
    explicitly_forbidden: set[str]
    total_forbidden: set[str]


@dataclass
class ValidationReport:
    """Complete validation report for a DAG."""

    dag_name: str
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    # Computed constraints
    forbidden_controls: dict[str, ForbiddenControlsResult] = field(default_factory=dict)
    backdoor_candidates: dict[str, set[str]] = field(default_factory=dict)
    identity_dependencies: dict[str, set[str]] = field(default_factory=dict)

    def errors(self) -> list[ValidationIssue]:
        """Get all errors."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    def warnings(self) -> list[ValidationIssue]:
        """Get all warnings."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            f"DAG VALIDATION REPORT: {self.dag_name}",
            "=" * 60,
            "",
            f"Status: {'VALID' if self.is_valid else 'INVALID'}",
            f"Errors: {len(self.errors())}",
            f"Warnings: {len(self.warnings())}",
            "",
        ]

        if self.errors():
            lines.append("ERRORS:")
            for issue in self.errors():
                lines.append(f"  {issue}")
            lines.append("")

        if self.warnings():
            lines.append("WARNINGS:")
            for issue in self.warnings():
                lines.append(f"  {issue}")
            lines.append("")

        if self.forbidden_controls:
            lines.append("FORBIDDEN CONTROLS (per edge):")
            for edge_id, result in self.forbidden_controls.items():
                lines.append(f"  {edge_id}: {sorted(result.total_forbidden)}")
            lines.append("")

        if self.backdoor_candidates:
            lines.append("BACKDOOR CANDIDATES (per edge):")
            for edge_id, candidates in self.backdoor_candidates.items():
                lines.append(f"  {edge_id}: {sorted(candidates)}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


class DAGValidator:
    """
    Validates DAG specifications for semantic correctness.

    Performs:
    1. Referential integrity checks
    2. Temporal cycle detection
    3. Identity dependency validation
    4. Forbidden controls computation
    5. Backdoor candidate identification
    """

    def __init__(self, dag: DAGSpec):
        self.dag = dag
        self.issues: list[ValidationIssue] = []

    def validate(self) -> ValidationReport:
        """
        Run all validation checks and return a report.

        Returns:
            ValidationReport with issues and computed constraints
        """
        self.issues = []

        # Run all checks
        self._check_referential_integrity()
        self._check_temporal_cycles()
        self._check_identity_dependencies()
        self._check_forbidden_controls_consistency()
        self._check_instrument_validity()
        self._check_design_feasibility()
        self._check_acceptance_criteria()

        # New checks for KSPI K2 DAG requirements
        self._check_scope_consistency()
        self._check_bidirectional_policy_rate()
        self._check_rwa_mechanism()
        self._check_immutable_evidence()
        self._check_edge_timing_defaults()

        # Compute constraints
        forbidden_controls = self._compute_all_forbidden_controls()
        backdoor_candidates = self._compute_backdoor_candidates()
        identity_deps = self._compute_identity_dependencies()

        # Determine validity
        is_valid = len(self.errors()) == 0

        return ValidationReport(
            dag_name=self.dag.metadata.name,
            is_valid=is_valid,
            issues=self.issues,
            forbidden_controls=forbidden_controls,
            backdoor_candidates=backdoor_candidates,
            identity_dependencies=identity_deps,
        )

    def errors(self) -> list[ValidationIssue]:
        """Get all errors."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    def _add_error(self, code: str, message: str, location: str = "",
                   suggestion: str = "") -> None:
        """Add an error issue."""
        self.issues.append(ValidationIssue(
            code=code,
            message=message,
            severity=ValidationSeverity.ERROR,
            location=location,
            suggestion=suggestion,
        ))

    def _add_warning(self, code: str, message: str, location: str = "",
                     suggestion: str = "") -> None:
        """Add a warning issue."""
        self.issues.append(ValidationIssue(
            code=code,
            message=message,
            severity=ValidationSeverity.WARNING,
            location=location,
            suggestion=suggestion,
        ))

    def _add_info(self, code: str, message: str, location: str = "",
                  suggestion: str = "") -> None:
        """Add an info issue."""
        self.issues.append(ValidationIssue(
            code=code,
            message=message,
            severity=ValidationSeverity.INFO,
            location=location,
            suggestion=suggestion,
        ))

    def _check_referential_integrity(self) -> None:
        """Check that all edge endpoints exist as nodes."""
        node_ids = {n.id for n in self.dag.nodes}

        for edge in self.dag.edges:
            if edge.from_node not in node_ids:
                self._add_error(
                    code="REF_FROM_NODE",
                    message=f"Edge references non-existent from_node: {edge.from_node}",
                    location=f"edge:{edge.id}",
                    suggestion=f"Add node '{edge.from_node}' to the nodes list",
                )

            if edge.to_node not in node_ids:
                self._add_error(
                    code="REF_TO_NODE",
                    message=f"Edge references non-existent to_node: {edge.to_node}",
                    location=f"edge:{edge.id}",
                    suggestion=f"Add node '{edge.to_node}' to the nodes list",
                )

            # Check required adjustments exist
            for adj in edge.required_adjustments:
                if adj not in node_ids:
                    self._add_error(
                        code="REF_ADJUSTMENT",
                        message=f"Required adjustment references non-existent node: {adj}",
                        location=f"edge:{edge.id}",
                        suggestion=f"Add node '{adj}' or remove from required_adjustments",
                    )

            # Check instruments exist
            for inst in edge.instruments:
                if inst not in node_ids:
                    self._add_error(
                        code="REF_INSTRUMENT",
                        message=f"Instrument references non-existent node: {inst}",
                        location=f"edge:{edge.id}",
                        suggestion=f"Add node '{inst}' or remove from instruments",
                    )

            # Check forbidden controls exist
            for ctrl in edge.forbidden_controls:
                if ctrl not in node_ids:
                    self._add_warning(
                        code="REF_FORBIDDEN",
                        message=f"Forbidden control references non-existent node: {ctrl}",
                        location=f"edge:{edge.id}",
                        suggestion=f"Add node '{ctrl}' or remove from forbidden_controls",
                    )

    def _check_temporal_cycles(self) -> None:
        """Check for cycles in the temporal expansion of the DAG."""
        has_cycle, cycle = check_temporal_cycle(self.dag)

        if has_cycle:
            cycle_str = " → ".join(cycle) if cycle else "unknown"
            self._add_error(
                code="TEMPORAL_CYCLE",
                message=f"Temporal cycle detected: {cycle_str}",
                location="dag",
                suggestion="Add appropriate lags to break the cycle, or verify the causal structure",
            )

        # Check for edges without timing that might cause issues
        for edge in self.dag.edges:
            if edge.timing.lag == 0 and not edge.timing.contemporaneous:
                self._add_warning(
                    code="DEFAULT_LAG",
                    message=f"Edge has default lag=0 without contemporaneous=True",
                    location=f"edge:{edge.id}",
                    suggestion="Set timing.lag > 0 or timing.contemporaneous=True explicitly",
                )

    def _check_identity_dependencies(self) -> None:
        """Check that identity formula dependencies are valid."""
        node_ids = {n.id for n in self.dag.nodes}

        for node in self.dag.nodes:
            if node.identity:
                for dep in node.depends_on:
                    if dep not in node_ids:
                        self._add_error(
                            code="IDENTITY_DEP",
                            message=f"Identity depends on non-existent node: {dep}",
                            location=f"node:{node.id}",
                            suggestion=f"Add node '{dep}' or fix the identity formula",
                        )

            # Check legacy identities field too
            for identity in node.identities:
                for dep in identity.depends_on:
                    if dep not in node_ids:
                        self._add_error(
                            code="IDENTITY_DEP",
                            message=f"Identity '{identity.name}' depends on non-existent node: {dep}",
                            location=f"node:{node.id}",
                            suggestion=f"Add node '{dep}' or fix the identity formula",
                        )

    def _check_forbidden_controls_consistency(self) -> None:
        """Check that forbidden controls don't include required adjustments."""
        for edge in self.dag.edges:
            forbidden = set(edge.forbidden_controls)
            required = set(edge.required_adjustments)

            overlap = forbidden & required
            if overlap:
                self._add_error(
                    code="FORBIDDEN_REQUIRED_CONFLICT",
                    message=f"Nodes are both forbidden and required: {overlap}",
                    location=f"edge:{edge.id}",
                    suggestion="Remove from either forbidden_controls or required_adjustments",
                )

    def _check_instrument_validity(self) -> None:
        """Check instrument requirements for IV designs."""
        for edge in self.dag.edges:
            if "IV_2SLS" in edge.allowed_designs:
                if not edge.instruments:
                    self._add_warning(
                        code="IV_NO_INSTRUMENT",
                        message="IV_2SLS design allowed but no instruments specified",
                        location=f"edge:{edge.id}",
                        suggestion="Add instruments to the edge specification",
                    )

            # Check instruments aren't descendants of treatment
            if edge.instruments:
                descendants = self.dag.get_descendants(edge.from_node)
                for inst in edge.instruments:
                    if inst in descendants:
                        self._add_error(
                            code="INSTRUMENT_DESCENDANT",
                            message=f"Instrument '{inst}' is a descendant of treatment",
                            location=f"edge:{edge.id}",
                            suggestion="Instruments must not be affected by treatment",
                        )

    def _check_design_feasibility(self) -> None:
        """Check that design requirements can potentially be met."""
        for edge in self.dag.edges:
            treatment_node = self.dag.get_node(edge.from_node)

            if treatment_node:
                # DiD requires binary treatment
                if "DID_EVENT_STUDY" in edge.allowed_designs:
                    if treatment_node.type != "binary":
                        self._add_warning(
                            code="DID_NON_BINARY",
                            message=f"DiD design with non-binary treatment: {treatment_node.type}",
                            location=f"edge:{edge.id}",
                            suggestion="DiD typically requires binary treatment indicator",
                        )

                # RDD requires running variable
                if "RDD" in edge.allowed_designs:
                    if treatment_node.type not in ["continuous", "index"]:
                        self._add_warning(
                            code="RDD_RUNNING_VAR",
                            message=f"RDD design but treatment type is {treatment_node.type}",
                            location=f"edge:{edge.id}",
                            suggestion="RDD requires a continuous running variable",
                        )

    def _check_acceptance_criteria(self) -> None:
        """Check acceptance criteria for p-hacking risks."""
        for edge in self.dag.edges:
            criteria = edge.acceptance_criteria

            # Warn about legacy min_tstat
            if criteria.min_tstat is not None:
                self._add_warning(
                    code="DEPRECATED_MIN_TSTAT",
                    message="min_tstat is deprecated and encourages p-hacking",
                    location=f"edge:{edge.id}",
                    suggestion="Use credibility scoring instead of significance thresholds",
                )

            # Check null acceptance is enabled
            if not criteria.null_acceptance.enabled:
                self._add_warning(
                    code="NULL_ACCEPTANCE_DISABLED",
                    message="Null acceptance is disabled, which may encourage p-hacking",
                    location=f"edge:{edge.id}",
                    suggestion="Enable null_acceptance to allow 'precisely null' results",
                )

    def _compute_all_forbidden_controls(self) -> dict[str, ForbiddenControlsResult]:
        """Compute forbidden controls for all edges."""
        results = {}

        for edge in self.dag.edges:
            descendants = self.dag.get_descendants(edge.from_node)
            explicit = set(edge.forbidden_controls)

            results[edge.id] = ForbiddenControlsResult(
                edge_id=edge.id,
                treatment_node=edge.from_node,
                descendants=descendants,
                explicitly_forbidden=explicit,
                total_forbidden=descendants | explicit,
            )

        return results

    def _compute_backdoor_candidates(self) -> dict[str, set[str]]:
        """
        Compute potential backdoor adjustment sets for each edge.

        A variable is a backdoor candidate if it:
        1. Is not a descendant of treatment
        2. Is an ancestor of either treatment or outcome
        3. Is not the treatment or outcome itself
        """
        candidates = {}

        for edge in self.dag.edges:
            treatment = edge.from_node
            outcome = edge.to_node

            # Get descendants of treatment (forbidden)
            descendants = self.dag.get_descendants(treatment)

            # Get ancestors of treatment and outcome
            treatment_ancestors = self.dag.get_ancestors(treatment)
            outcome_ancestors = self.dag.get_ancestors(outcome)

            # Candidates are ancestors that aren't descendants
            all_ancestors = treatment_ancestors | outcome_ancestors
            valid_candidates = all_ancestors - descendants - {treatment, outcome}

            candidates[edge.id] = valid_candidates

        return candidates

    def _compute_identity_dependencies(self) -> dict[str, set[str]]:
        """Compute identity dependencies for all derived nodes."""
        dependencies = {}

        for node in self.dag.nodes:
            if node.derived or node.identity:
                deps = set(node.depends_on)

                # Also include legacy identities
                for identity in node.identities:
                    deps.update(identity.depends_on)

                if deps:
                    dependencies[node.id] = deps

        return dependencies

    def compute_forbidden_controls(self, edge_id: str) -> set[str]:
        """
        Compute forbidden controls for a specific edge.

        Args:
            edge_id: The edge ID

        Returns:
            Set of node IDs that cannot be used as controls
        """
        edge = self.dag.get_edge(edge_id)
        if not edge:
            return set()

        descendants = self.dag.get_descendants(edge.from_node)
        explicit = set(edge.forbidden_controls)

        return descendants | explicit

    # =========================================================================
    # New validation methods for KSPI K2 DAG requirements
    # =========================================================================

    def _check_scope_consistency(self) -> None:
        """
        Check that node scopes are consistent across edges.

        Warns when BNS household data (kazakhstan_only) is combined
        with consolidated KSPI data, as this creates external validity risk.
        """
        # Build scope lookup from node attributes
        node_scopes: dict[str, str] = {}
        for node in self.dag.nodes:
            # Check for scope in node spec (stored as custom attribute)
            scope = getattr(node, 'scope', None)
            if scope is None:
                # Try to infer from tags or source
                if node.source and node.source.preferred:
                    connector = node.source.preferred[0].connector
                    if connector == "bns":
                        scope = "kazakhstan_only"
                    elif connector == "kspi_quarterly":
                        scope = "kazakhstan_only"
                    elif connector in ["fred", "baumeister"]:
                        scope = "global"
            node_scopes[node.id] = scope or "unknown"

        # Check edges for scope mismatches
        for edge in self.dag.edges:
            from_scope = node_scopes.get(edge.from_node, "unknown")
            to_scope = node_scopes.get(edge.to_node, "unknown")

            # Warn about BNS + consolidated KSPI combinations
            if from_scope == "kazakhstan_only" and to_scope == "consolidated":
                self._add_warning(
                    code="SCOPE_MISMATCH",
                    message=f"Edge combines kazakhstan_only source with consolidated target",
                    location=f"edge:{edge.id}",
                    suggestion="Document as external validity caveat in interpretation",
                )
            elif from_scope == "consolidated" and to_scope == "kazakhstan_only":
                self._add_warning(
                    code="SCOPE_MISMATCH",
                    message=f"Edge combines consolidated source with kazakhstan_only target",
                    location=f"edge:{edge.id}",
                    suggestion="Use kazakhstan_only data for both nodes",
                )

    def _check_bidirectional_policy_rate(self) -> None:
        """
        Check that policy rate is modeled bidirectionally if present.

        The policy rate is endogenous to inflation and FX conditions,
        so both reaction (conditions -> rate) and transmission (rate -> bank)
        directions must be modeled.
        """
        # Find policy rate node
        policy_rate_nodes = [
            n.id for n in self.dag.nodes
            if "policy_rate" in n.id.lower() or "nbk_rate" in n.id.lower()
        ]

        if not policy_rate_nodes:
            return  # No policy rate in this DAG

        for policy_node in policy_rate_nodes:
            # Check for edges TO policy rate (reaction function)
            edges_to_rate = self.dag.get_edges_to(policy_node)
            # Check for edges FROM policy rate (transmission)
            edges_from_rate = self.dag.get_edges_from(policy_node)

            has_reaction = len(edges_to_rate) > 0
            has_transmission = len(edges_from_rate) > 0

            if has_transmission and not has_reaction:
                self._add_error(
                    code="POLICY_RATE_NOT_BIDIRECTIONAL",
                    message=f"Policy rate '{policy_node}' has transmission but no reaction edges",
                    location=f"node:{policy_node}",
                    suggestion="Add edges for NBK reaction function (cpi -> rate, fx -> rate)",
                )

            if has_reaction and not has_transmission:
                self._add_warning(
                    code="POLICY_RATE_REACTION_ONLY",
                    message=f"Policy rate '{policy_node}' has reaction but no transmission edges",
                    location=f"node:{policy_node}",
                    suggestion="Add transmission edges if policy effects on bank are relevant",
                )

    def _check_rwa_mechanism(self) -> None:
        """
        Check that RWA is not directly linked from CPI.

        For unsecured consumer lending, the RWA mechanism is:
        - loan_portfolio -> rwa (volume effect)
        - portfolio_mix -> rwa (composition effect)
        - regulatory_changes -> rwa (rule changes)

        CPI -> RWA is incorrect for unsecured books (no collateral channel).
        """
        # Find RWA nodes
        rwa_nodes = [
            n.id for n in self.dag.nodes
            if "rwa" in n.id.lower()
        ]

        # Find CPI nodes
        cpi_nodes = [
            n.id for n in self.dag.nodes
            if "cpi" in n.id.lower()
        ]

        for rwa_node in rwa_nodes:
            edges_to_rwa = self.dag.get_edges_to(rwa_node)

            for edge in edges_to_rwa:
                if edge.from_node in cpi_nodes:
                    self._add_warning(
                        code="RWA_CPI_DIRECT_LINK",
                        message=f"CPI '{edge.from_node}' directly linked to RWA '{rwa_node}'",
                        location=f"edge:{edge.id}",
                        suggestion="Remove direct CPI->RWA edge. "
                                   "Use portfolio volume/mix for unsecured consumer books.",
                    )

    def _check_immutable_evidence(self) -> None:
        """
        Check that validated evidence artifacts are not re-estimable.

        Edges with validated_evidence.immutable=True should not be
        re-estimated by the agentic loop.
        """
        for edge in self.dag.edges:
            # Check for validated_evidence in edge spec (custom field)
            validated = getattr(edge, 'validated_evidence', None)
            if validated is None:
                # Check in notes field for indication
                if "immutable" in edge.notes.lower() or "block_" in edge.notes.lower():
                    self._add_info(
                        code="EVIDENCE_IN_NOTES",
                        message=f"Edge mentions validated evidence in notes but lacks validated_evidence field",
                        location=f"edge:{edge.id}",
                        suggestion="Add validated_evidence block to formally mark as immutable",
                    )

    def _check_edge_timing_defaults(self) -> None:
        """
        Check that all edges have explicit timing.

        If timing.lag is at default (0) without contemporaneous=True,
        emit warning and suggest default of lag=1.
        """
        for edge in self.dag.edges:
            timing = edge.timing

            # Already checked in _check_temporal_cycles, but add more specific message
            if timing.lag == 0 and not timing.contemporaneous:
                # Check if this looks like an identity edge (no designs)
                if not edge.allowed_designs:
                    # Identity edges can have lag=0
                    continue

                self._add_warning(
                    code="MISSING_TIMING_LAG",
                    message=f"Edge has no explicit timing lag (defaulting to 0)",
                    location=f"edge:{edge.id}",
                    suggestion="Set timing.lag=1 or timing.contemporaneous=True explicitly",
                )


def validate_dag(dag: DAGSpec, raise_on_error: bool = True) -> ValidationReport:
    """
    Validate a DAG specification.

    Args:
        dag: The DAG to validate
        raise_on_error: If True, raise ValidationError on errors

    Returns:
        ValidationReport with all issues and computed constraints

    Raises:
        ValidationError: If raise_on_error=True and validation fails
    """
    validator = DAGValidator(dag)
    report = validator.validate()

    if raise_on_error and not report.is_valid:
        raise ValidationError(report.errors())

    return report
