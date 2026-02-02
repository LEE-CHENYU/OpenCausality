"""
EdgeCard Output Format.

The EdgeCard is the complete output artifact for a causal edge estimation,
including:
- Data provenance
- Specification hash
- Estimates with uncertainty
- Diagnostics with pass/fail
- Interpretation boundary
- Failure flags
- Counterfactual applicability
- Credibility rating
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import yaml

from shared.agentic.output.provenance import (
    DataProvenance,
    SpecDetails,
    AuditRecord,
)


@dataclass
class Estimates:
    """Point estimates with uncertainty quantification."""

    point: float
    se: float
    ci_95: tuple[float, float]
    pvalue: float | None = None

    # For IRF/Local Projections
    horizons: list[int] | None = None
    irf: list[float] | None = None
    irf_ci_lower: list[float] | None = None
    irf_ci_upper: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "point": self.point,
            "se": self.se,
            "ci_95": list(self.ci_95),
            "pvalue": self.pvalue,
        }
        if self.horizons:
            d["horizons"] = self.horizons
            d["irf"] = self.irf
            d["irf_ci_lower"] = self.irf_ci_lower
            d["irf_ci_upper"] = self.irf_ci_upper
        return d


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic test."""

    name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    pvalue: float | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "pvalue": self.pvalue,
            "message": self.message,
        }


@dataclass
class Interpretation:
    """
    Interpretation boundary for the estimate.

    CRITICAL: This prevents overclaiming of causal effects.
    """

    estimand: str  # What we're actually estimating
    is_not: str  # What this is NOT
    channels: list[str] = field(default_factory=list)  # Possible channels
    population: str = ""  # Population the estimate applies to
    conditions: str = ""  # Conditions under which estimate holds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "estimand": self.estimand,
            "is_not": self.is_not,
            "channels": self.channels,
            "population": self.population,
            "conditions": self.conditions,
        }


@dataclass
class FailureFlags:
    """
    Flags indicating potential issues with the estimate.

    These don't prevent estimation but should be considered
    when interpreting results.
    """

    weak_identification: bool = False
    potential_bad_control: bool = False
    mechanical_identity_risk: bool = False
    regime_break_detected: bool = False
    small_sample: bool = False
    high_missing_rate: bool = False

    def any_flagged(self) -> bool:
        """Check if any flags are raised."""
        return any([
            self.weak_identification,
            self.potential_bad_control,
            self.mechanical_identity_risk,
            self.regime_break_detected,
            self.small_sample,
            self.high_missing_rate,
        ])

    def to_dict(self) -> dict[str, bool]:
        """Convert to dictionary."""
        return {
            "weak_identification": self.weak_identification,
            "potential_bad_control": self.potential_bad_control,
            "mechanical_identity_risk": self.mechanical_identity_risk,
            "regime_break_detected": self.regime_break_detected,
            "small_sample": self.small_sample,
            "high_missing_rate": self.high_missing_rate,
        }


@dataclass
class CounterfactualApplicability:
    """
    Defines what counterfactual questions the estimate can answer.

    IMPORTANT: Prevents using reduced-form estimates for
    structural counterfactuals.
    """

    supports_shock_path: bool = True
    supports_policy_intervention: bool = False
    intervention_note: str = ""
    external_validity: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "supports_shock_path": self.supports_shock_path,
            "supports_policy_intervention": self.supports_policy_intervention,
            "intervention_note": self.intervention_note,
            "external_validity": self.external_validity,
        }


@dataclass
class EdgeCard:
    """
    Complete output artifact for a causal edge estimation.

    This is the primary output format for the agentic system,
    containing all information needed for audit, interpretation,
    and downstream use.
    """

    # Identity
    edge_id: str
    dag_version_hash: str
    created_at: datetime = field(default_factory=datetime.now)

    # Data provenance
    data_provenance: DataProvenance = field(default_factory=DataProvenance)

    # Spec hash (for audit)
    spec_hash: str = ""
    spec_details: SpecDetails = field(default_factory=SpecDetails)

    # Estimates
    estimates: Estimates | None = None

    # Diagnostics (each with pass/fail)
    diagnostics: dict[str, DiagnosticResult] = field(default_factory=dict)

    # Interpretation boundary
    interpretation: Interpretation = field(default_factory=Interpretation)

    # Failure flags
    failure_flags: FailureFlags = field(default_factory=FailureFlags)

    # Counterfactual applicability
    counterfactual: CounterfactualApplicability = field(
        default_factory=CounterfactualApplicability
    )

    # Credibility
    credibility_rating: Literal["A", "B", "C", "D"] = "D"
    credibility_score: float = 0.0

    # Null acceptance
    is_precisely_null: bool = False
    null_equivalence_bound: float | None = None

    def all_diagnostics_pass(self) -> bool:
        """Check if all diagnostics pass."""
        return all(d.passed for d in self.diagnostics.values())

    def compute_result_hash(self) -> str:
        """Compute hash of results for audit."""
        result_dict = {
            "edge_id": self.edge_id,
            "estimates": self.estimates.to_dict() if self.estimates else None,
            "diagnostics": {k: v.to_dict() for k, v in self.diagnostics.items()},
            "credibility_score": self.credibility_score,
        }
        content = json.dumps(result_dict, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def get_audit_record(self) -> AuditRecord:
        """Create audit record for this card."""
        return AuditRecord(
            edge_id=self.edge_id,
            dag_version_hash=self.dag_version_hash,
            data_hash=self.data_provenance.compute_hash(),
            spec_hash=self.spec_hash,
            result_hash=self.compute_result_hash(),
            timestamp=self.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "edge_id": self.edge_id,
            "dag_version_hash": self.dag_version_hash,
            "created_at": self.created_at.isoformat(),
            "data_provenance": self.data_provenance.to_dict(),
            "spec_hash": self.spec_hash,
            "spec_details": self.spec_details.to_dict(),
            "estimates": self.estimates.to_dict() if self.estimates else None,
            "diagnostics": {k: v.to_dict() for k, v in self.diagnostics.items()},
            "all_diagnostics_pass": self.all_diagnostics_pass(),
            "interpretation": self.interpretation.to_dict(),
            "failure_flags": self.failure_flags.to_dict(),
            "counterfactual": self.counterfactual.to_dict(),
            "credibility_rating": self.credibility_rating,
            "credibility_score": self.credibility_score,
            "is_precisely_null": self.is_precisely_null,
            "null_equivalence_bound": self.null_equivalence_bound,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def to_markdown(self) -> str:
        """Generate markdown report for human consumption."""
        lines = [
            f"# EdgeCard: {self.edge_id}",
            "",
            f"**Created:** {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**DAG Version:** `{self.dag_version_hash[:12]}...`",
            f"**Spec Hash:** `{self.spec_hash[:12]}...`",
            "",
            f"## Credibility: {self.credibility_rating} ({self.credibility_score:.2f})",
            "",
        ]

        # Estimates
        lines.append("## Estimates")
        if self.estimates:
            lines.append("")
            lines.append(f"- **Point estimate:** {self.estimates.point:.4f}")
            lines.append(f"- **Standard error:** {self.estimates.se:.4f}")
            lines.append(f"- **95% CI:** [{self.estimates.ci_95[0]:.4f}, {self.estimates.ci_95[1]:.4f}]")
            if self.estimates.pvalue is not None:
                lines.append(f"- **p-value:** {self.estimates.pvalue:.4f}")

            if self.is_precisely_null:
                lines.append("")
                lines.append(f"**Note:** Effect is precisely null (|β| < {self.null_equivalence_bound})")
        else:
            lines.append("*No estimates available*")
        lines.append("")

        # Diagnostics
        lines.append("## Diagnostics")
        lines.append("")
        if self.diagnostics:
            lines.append("| Test | Status | Value | Threshold |")
            lines.append("|------|--------|-------|-----------|")
            for name, diag in self.diagnostics.items():
                status = "✓ PASS" if diag.passed else "✗ FAIL"
                value = f"{diag.value:.4f}" if diag.value is not None else "-"
                threshold = f"{diag.threshold:.4f}" if diag.threshold is not None else "-"
                lines.append(f"| {name} | {status} | {value} | {threshold} |")
            lines.append("")
            overall = "✓ All Pass" if self.all_diagnostics_pass() else "✗ Some Fail"
            lines.append(f"**Overall:** {overall}")
        else:
            lines.append("*No diagnostics run*")
        lines.append("")

        # Interpretation
        lines.append("## Interpretation Boundary")
        lines.append("")
        lines.append(f"**Estimand:** {self.interpretation.estimand}")
        lines.append("")
        lines.append(f"**This is NOT:** {self.interpretation.is_not}")
        if self.interpretation.channels:
            lines.append("")
            lines.append(f"**Possible channels:** {', '.join(self.interpretation.channels)}")
        lines.append("")

        # Failure Flags
        if self.failure_flags.any_flagged():
            lines.append("## ⚠️ Failure Flags")
            lines.append("")
            flags = self.failure_flags.to_dict()
            for flag, value in flags.items():
                if value:
                    lines.append(f"- **{flag.replace('_', ' ').title()}**")
            lines.append("")

        # Counterfactual
        lines.append("## Counterfactual Applicability")
        lines.append("")
        lines.append(f"- Supports shock path analysis: {'Yes' if self.counterfactual.supports_shock_path else 'No'}")
        lines.append(f"- Supports policy intervention: {'Yes' if self.counterfactual.supports_policy_intervention else 'No'}")
        if self.counterfactual.intervention_note:
            lines.append(f"- Note: {self.counterfactual.intervention_note}")
        lines.append("")

        # Specification
        lines.append("## Specification Details")
        lines.append("")
        lines.append(f"- **Design:** {self.spec_details.design}")
        lines.append(f"- **Controls:** {', '.join(self.spec_details.controls) or 'None'}")
        lines.append(f"- **Instruments:** {', '.join(self.spec_details.instruments) or 'None'}")
        lines.append(f"- **Fixed Effects:** {', '.join(self.spec_details.fixed_effects) or 'None'}")
        lines.append(f"- **SE Method:** {self.spec_details.se_method}")
        lines.append("")

        lines.append("---")
        lines.append(f"*Generated by Agentic DAG System*")

        return "\n".join(lines)


def compute_credibility_score(
    diagnostics: dict[str, DiagnosticResult],
    failure_flags: FailureFlags,
    design_weight: float = 0.6,
    data_coverage: float = 1.0,
) -> tuple[float, str]:
    """
    Compute credibility score for an EdgeCard.

    IMPORTANT: Does NOT use statistical significance.

    Weights:
    - Diagnostics pass rate: 40%
    - Design strength: 10%
    - Stability (no failure flags): 30%
    - Data coverage: 20%

    Args:
        diagnostics: Diagnostic results
        failure_flags: Failure flags
        design_weight: Credibility weight of the design (0-1)
        data_coverage: Data coverage score (0-1)

    Returns:
        Tuple of (score, rating)
    """
    # Diagnostics pass rate (40%)
    if diagnostics:
        pass_rate = sum(1 for d in diagnostics.values() if d.passed) / len(diagnostics)
    else:
        pass_rate = 0.5  # Default if no diagnostics

    # Design strength (10%)
    design_score = design_weight

    # Stability / no failure flags (30%)
    flag_count = sum([
        failure_flags.weak_identification,
        failure_flags.potential_bad_control,
        failure_flags.mechanical_identity_risk,
        failure_flags.regime_break_detected,
        failure_flags.small_sample,
        failure_flags.high_missing_rate,
    ])
    stability_score = max(0, 1 - flag_count * 0.2)

    # Data coverage (20%)
    coverage_score = data_coverage

    # Weighted total
    score = (
        0.40 * pass_rate +
        0.10 * design_score +
        0.30 * stability_score +
        0.20 * coverage_score
    )

    # Rating
    if score >= 0.80:
        rating = "A"
    elif score >= 0.60:
        rating = "B"
    elif score >= 0.40:
        rating = "C"
    else:
        rating = "D"

    return score, rating
