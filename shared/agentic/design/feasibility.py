"""
Feasibility Checking for Designs.

Checks whether a design can be applied given:
- Data availability and structure
- Variable types
- Sample size requirements
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from shared.agentic.design.registry import DesignSpec

logger = logging.getLogger(__name__)


@dataclass
class DataReport:
    """
    Report on available data for estimation.

    This is passed to the feasibility checker to determine
    which designs are viable.
    """

    # Data structure
    is_panel: bool = False
    is_time_series: bool = False
    has_running_variable: bool = False

    # Sample size
    n_obs: int = 0
    n_periods: int = 0
    n_units: int = 0

    # Variable information
    treatment_type: str = "continuous"
    outcome_type: str = "continuous"
    n_instruments: int = 0

    # Data quality
    missing_rate: float = 0.0
    has_pre_period: bool = False
    n_pre_periods: int = 0
    has_treated_control: bool = False

    # RDD-specific
    density_near_cutoff: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_panel": self.is_panel,
            "is_time_series": self.is_time_series,
            "has_running_variable": self.has_running_variable,
            "n_obs": self.n_obs,
            "n_periods": self.n_periods,
            "n_units": self.n_units,
            "treatment_type": self.treatment_type,
            "outcome_type": self.outcome_type,
            "n_instruments": self.n_instruments,
            "missing_rate": self.missing_rate,
            "has_pre_period": self.has_pre_period,
            "n_pre_periods": self.n_pre_periods,
            "has_treated_control": self.has_treated_control,
            "density_near_cutoff": self.density_near_cutoff,
        }


@dataclass
class FeasibilityIssue:
    """A single feasibility issue."""

    check: str
    message: str
    required: Any
    actual: Any


@dataclass
class FeasibilityResult:
    """Result of feasibility check for a design."""

    design_id: str
    is_feasible: bool
    issues: list[FeasibilityIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary string."""
        status = "FEASIBLE" if self.is_feasible else "NOT FEASIBLE"
        lines = [f"{self.design_id}: {status}"]

        for issue in self.issues:
            lines.append(f"  - {issue.check}: {issue.message}")
            lines.append(f"    Required: {issue.required}, Actual: {issue.actual}")

        for warning in self.warnings:
            lines.append(f"  [WARN] {warning}")

        return "\n".join(lines)


class FeasibilityChecker:
    """
    Checks feasibility of designs given data characteristics.

    Performs:
    - Data structure checks (panel, time series, running variable)
    - Sample size checks
    - Variable type checks
    - Design-specific requirements
    """

    def __init__(self, data_report: DataReport):
        """
        Initialize checker with data report.

        Args:
            data_report: Report on available data
        """
        self.data = data_report

    def check(self, design: DesignSpec) -> FeasibilityResult:
        """
        Check if a design is feasible.

        Args:
            design: The design to check

        Returns:
            FeasibilityResult with issues and warnings
        """
        issues = []
        warnings = []

        # Data structure checks
        self._check_data_structure(design, issues)

        # Sample size checks
        self._check_sample_size(design, issues, warnings)

        # Variable type checks
        self._check_variable_types(design, issues)

        # Design-specific checks
        self._check_design_specific(design, issues, warnings)

        return FeasibilityResult(
            design_id=design.id,
            is_feasible=len(issues) == 0,
            issues=issues,
            warnings=warnings,
        )

    def _check_data_structure(
        self,
        design: DesignSpec,
        issues: list[FeasibilityIssue],
    ) -> None:
        """Check data structure requirements."""
        req = design.data_requirements

        if req.panel and not self.data.is_panel:
            issues.append(FeasibilityIssue(
                check="panel_required",
                message="Design requires panel data",
                required=True,
                actual=self.data.is_panel,
            ))

        if req.time_series and not self.data.is_time_series:
            issues.append(FeasibilityIssue(
                check="time_series_required",
                message="Design requires time series data",
                required=True,
                actual=self.data.is_time_series,
            ))

        if req.running_variable and not self.data.has_running_variable:
            issues.append(FeasibilityIssue(
                check="running_variable_required",
                message="Design requires a running variable",
                required=True,
                actual=self.data.has_running_variable,
            ))

    def _check_sample_size(
        self,
        design: DesignSpec,
        issues: list[FeasibilityIssue],
        warnings: list[str],
    ) -> None:
        """Check sample size requirements."""
        req = design.data_requirements

        if self.data.n_obs < req.min_obs:
            issues.append(FeasibilityIssue(
                check="min_obs",
                message=f"Insufficient observations",
                required=req.min_obs,
                actual=self.data.n_obs,
            ))

        if req.panel:
            if self.data.n_periods < req.min_periods:
                issues.append(FeasibilityIssue(
                    check="min_periods",
                    message="Insufficient time periods",
                    required=req.min_periods,
                    actual=self.data.n_periods,
                ))

            if self.data.n_units < req.min_units:
                issues.append(FeasibilityIssue(
                    check="min_units",
                    message="Insufficient units",
                    required=req.min_units,
                    actual=self.data.n_units,
                ))

        # Warn if close to minimum
        if self.data.n_obs < req.min_obs * 1.5:
            warnings.append(f"Sample size ({self.data.n_obs}) is close to minimum ({req.min_obs})")

    def _check_variable_types(
        self,
        design: DesignSpec,
        issues: list[FeasibilityIssue],
    ) -> None:
        """Check variable type requirements."""
        req = design.variable_requirements

        if self.data.treatment_type not in req.treatment_type:
            issues.append(FeasibilityIssue(
                check="treatment_type",
                message=f"Treatment type '{self.data.treatment_type}' not allowed",
                required=req.treatment_type,
                actual=self.data.treatment_type,
            ))

        if self.data.outcome_type not in req.outcome_type:
            issues.append(FeasibilityIssue(
                check="outcome_type",
                message=f"Outcome type '{self.data.outcome_type}' not allowed",
                required=req.outcome_type,
                actual=self.data.outcome_type,
            ))

        if req.instrument_count_min > 0:
            if self.data.n_instruments < req.instrument_count_min:
                issues.append(FeasibilityIssue(
                    check="instrument_count",
                    message="Insufficient instruments",
                    required=req.instrument_count_min,
                    actual=self.data.n_instruments,
                ))

    def _check_design_specific(
        self,
        design: DesignSpec,
        issues: list[FeasibilityIssue],
        warnings: list[str],
    ) -> None:
        """Check design-specific requirements."""

        if design.id == "DID_EVENT_STUDY":
            self._check_did_requirements(issues, warnings)

        elif design.id == "RDD":
            self._check_rdd_requirements(issues, warnings)

        elif design.id == "IV_2SLS":
            self._check_iv_requirements(issues, warnings)

    def _check_did_requirements(
        self,
        issues: list[FeasibilityIssue],
        warnings: list[str],
    ) -> None:
        """Check DiD-specific requirements."""
        if not self.data.has_treated_control:
            issues.append(FeasibilityIssue(
                check="did_treated_control",
                message="DiD requires treated and control groups",
                required=True,
                actual=False,
            ))

        if not self.data.has_pre_period:
            issues.append(FeasibilityIssue(
                check="did_pre_period",
                message="DiD requires pre-treatment period",
                required=True,
                actual=False,
            ))

        if self.data.n_pre_periods < 4:
            warnings.append(
                f"Only {self.data.n_pre_periods} pre-periods available; "
                "pre-trend tests may have low power"
            )

    def _check_rdd_requirements(
        self,
        issues: list[FeasibilityIssue],
        warnings: list[str],
    ) -> None:
        """Check RDD-specific requirements."""
        if self.data.density_near_cutoff < 0.1:
            issues.append(FeasibilityIssue(
                check="rdd_density",
                message="Insufficient density near cutoff",
                required=0.1,
                actual=self.data.density_near_cutoff,
            ))

        # RDD needs substantial observations on both sides
        # This is approximated by density check above

    def _check_iv_requirements(
        self,
        issues: list[FeasibilityIssue],
        warnings: list[str],
    ) -> None:
        """Check IV-specific requirements."""
        if self.data.n_instruments == 0:
            issues.append(FeasibilityIssue(
                check="iv_instruments",
                message="IV requires at least one instrument",
                required=1,
                actual=0,
            ))

        # Warn about potential weak IV
        warnings.append("IV designs require first-stage F > 10 for validity")

    def check_all(self, designs: list[DesignSpec]) -> dict[str, FeasibilityResult]:
        """
        Check feasibility for multiple designs.

        Args:
            designs: List of designs to check

        Returns:
            Dictionary mapping design ID to FeasibilityResult
        """
        return {d.id: self.check(d) for d in designs}

    def get_feasible(self, designs: list[DesignSpec]) -> list[DesignSpec]:
        """
        Get all feasible designs.

        Args:
            designs: List of designs to check

        Returns:
            List of feasible designs
        """
        results = self.check_all(designs)
        return [d for d in designs if results[d.id].is_feasible]


def check_feasibility(
    design: DesignSpec,
    data_report: DataReport,
) -> FeasibilityResult:
    """
    Check if a design is feasible given data characteristics.

    Args:
        design: The design to check
        data_report: Report on available data

    Returns:
        FeasibilityResult with feasibility status and issues
    """
    checker = FeasibilityChecker(data_report)
    return checker.check(design)
