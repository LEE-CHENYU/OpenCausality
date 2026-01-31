"""
Sample construction and treatment assignment for credit default study.

Handles:
- Eligibility determination
- Treatment group assignment
- Sample restriction validation
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SampleCriteria:
    """Criteria for sample construction."""

    # Origination restrictions
    origination_before: date

    # Outcome window
    outcome_start: str  # YYYY-MM
    outcome_end: str  # YYYY-MM

    # Optional: origination after date
    origination_after: date | None = None

    # Loan status requirements
    require_active_at_treatment: bool = True
    treatment_date: str = "2024-01"  # YYYY-MM

    # Minimum wage design requirements
    require_payroll: bool = True
    min_payroll_months: int = 3
    payroll_window_start: str = "2023-10"
    payroll_window_end: str = "2023-12"

    # Pension RDD requirements
    pension_age_window_men: tuple[int, int] = (60, 66)
    pension_age_window_women: tuple[int, int] = (58, 64)


class SampleConstructor:
    """
    Constructs analysis samples with proper treatment assignment.

    Ensures:
    - Clear eligibility criteria
    - Correct treatment assignment
    - No leakage from post-treatment data
    """

    def __init__(self, criteria: SampleCriteria | None = None):
        """
        Initialize sample constructor.

        Args:
            criteria: Sample construction criteria
        """
        self.criteria = criteria or SampleCriteria(
            origination_before=date(2023, 12, 1),
            outcome_start="2024-01",
            outcome_end="2024-05",
        )

    def apply_eligibility_filters(
        self,
        panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply eligibility filters to panel.

        Args:
            panel: Raw panel data

        Returns:
            Filtered panel with eligibility flag
        """
        df = panel.copy()

        # Initialize eligibility
        df["eligible"] = True
        df["ineligibility_reason"] = None

        # Check origination date
        if "origination_date" in df.columns:
            df["origination_date"] = pd.to_datetime(df["origination_date"])

            # Before cutoff
            too_late = df["origination_date"] >= pd.Timestamp(
                self.criteria.origination_before
            )
            df.loc[too_late, "eligible"] = False
            df.loc[too_late, "ineligibility_reason"] = "originated_after_cutoff"

            # After minimum (if specified)
            if self.criteria.origination_after:
                too_early = df["origination_date"] < pd.Timestamp(
                    self.criteria.origination_after
                )
                df.loc[too_early & df["eligible"], "eligible"] = False
                df.loc[too_early & df["eligible"], "ineligibility_reason"] = "originated_before_minimum"

        # Check payroll requirement
        if self.criteria.require_payroll:
            if "has_payroll_data" in df.columns:
                no_payroll = ~df["has_payroll_data"]
                df.loc[no_payroll & df["eligible"], "eligible"] = False
                df.loc[no_payroll & df["eligible"], "ineligibility_reason"] = "no_payroll_data"

        # Log eligibility summary
        n_total = df["loan_id"].nunique() if "loan_id" in df.columns else len(df)
        n_eligible = df[df["eligible"]]["loan_id"].nunique() if "loan_id" in df.columns else df["eligible"].sum()

        logger.info(
            f"Eligibility filter: {n_eligible}/{n_total} loans eligible "
            f"({n_eligible/n_total:.1%})"
        )

        if "ineligibility_reason" in df.columns:
            inelig_reasons = df[~df["eligible"]]["ineligibility_reason"].value_counts()
            for reason, count in inelig_reasons.items():
                logger.info(f"  Ineligible ({reason}): {count}")

        return df

    def assign_mw_treatment(
        self,
        panel: pd.DataFrame,
        old_minimum_wage: int = 70000,
        bandwidth: int | None = None,
    ) -> pd.DataFrame:
        """
        Assign treatment groups for MW diff-in-discs design.

        Treatment: Workers earning below old MW at baseline
        Control: Workers earning above old MW at baseline

        Args:
            panel: Panel data with pre_policy_payroll
            old_minimum_wage: Old minimum wage threshold
            bandwidth: Optional bandwidth restriction around cutoff

        Returns:
            Panel with treatment assignment
        """
        df = panel.copy()

        if "pre_policy_payroll" not in df.columns:
            raise ValueError("Panel missing pre_policy_payroll - cannot assign treatment")

        # Treatment assignment
        df["mw_treated"] = (df["pre_policy_payroll"] < old_minimum_wage).astype(int)

        # Bandwidth restriction (if specified)
        if bandwidth:
            df["mw_in_bandwidth"] = (
                (df["pre_policy_payroll"] >= old_minimum_wage - bandwidth) &
                (df["pre_policy_payroll"] <= old_minimum_wage + bandwidth)
            ).astype(int)
        else:
            df["mw_in_bandwidth"] = 1

        # Log treatment balance
        if "mw_in_bandwidth" in df.columns:
            in_bw = df[df["mw_in_bandwidth"] == 1]
            n_treated = in_bw["mw_treated"].sum()
            n_control = len(in_bw) - n_treated

            logger.info(
                f"MW design treatment assignment (in bandwidth): "
                f"treated={n_treated}, control={n_control}"
            )

        return df

    def assign_pension_treatment(
        self,
        panel: pd.DataFrame,
        reference_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Assign treatment for pension fuzzy RDD.

        Running variable: Age relative to pension threshold
        Treatment: 1(Age >= threshold)

        Args:
            panel: Panel data with age, gender
            reference_date: Date for age calculation

        Returns:
            Panel with pension RDD variables
        """
        df = panel.copy()

        if "age" not in df.columns or "gender" not in df.columns:
            logger.warning("Panel missing age/gender - cannot assign pension treatment")
            return df

        # Get thresholds
        threshold_men = self.criteria.pension_age_window_men[0] + 3  # 63
        threshold_women = self.criteria.pension_age_window_women[0] + 3  # 61

        # Assign cutoff based on gender
        df["pension_cutoff"] = df["gender"].map(
            lambda g: threshold_men if str(g).upper() == "M" else threshold_women
        )

        # Running variable: age - cutoff
        df["pension_running_var"] = df["age"] - df["pension_cutoff"]

        # Treatment: above cutoff
        df["pension_above_cutoff"] = (df["pension_running_var"] >= 0).astype(int)

        # In bandwidth (for RDD)
        bw_men = self.criteria.pension_age_window_men
        bw_women = self.criteria.pension_age_window_women

        df["pension_in_bandwidth"] = (
            ((df["gender"].str.upper() == "M") &
             (df["age"] >= bw_men[0]) & (df["age"] <= bw_men[1])) |
            ((df["gender"].str.upper() == "F") &
             (df["age"] >= bw_women[0]) & (df["age"] <= bw_women[1]))
        ).astype(int)

        # Log assignment
        in_bw = df[df["pension_in_bandwidth"] == 1]
        n_above = in_bw["pension_above_cutoff"].sum()
        n_below = len(in_bw) - n_above

        logger.info(
            f"Pension RDD assignment (in bandwidth): "
            f"above_cutoff={n_above}, below_cutoff={n_below}"
        )

        return df

    def validate_sample(
        self,
        panel: pd.DataFrame,
        design: str = "mw",
    ) -> dict[str, Any]:
        """
        Validate sample for analysis.

        Args:
            panel: Constructed sample
            design: "mw" or "pension"

        Returns:
            Dictionary with validation results
        """
        results = {
            "design": design,
            "pass": True,
            "warnings": [],
            "errors": [],
        }

        # Basic counts
        results["n_loan_months"] = len(panel)
        results["n_loans"] = panel["loan_id"].nunique() if "loan_id" in panel.columns else None
        results["n_borrowers"] = panel["borrower_id"].nunique() if "borrower_id" in panel.columns else None

        # Design-specific validation
        if design == "mw":
            # Check treatment balance
            if "mw_treated" in panel.columns and "mw_in_bandwidth" in panel.columns:
                in_bw = panel[panel["mw_in_bandwidth"] == 1]
                n_treated = in_bw["mw_treated"].sum()
                n_control = len(in_bw) - n_treated

                results["n_treated"] = n_treated
                results["n_control"] = n_control

                if n_treated < 100:
                    results["warnings"].append(f"Low treatment group size: {n_treated}")
                if n_control < 100:
                    results["warnings"].append(f"Low control group size: {n_control}")

            # Check payroll coverage
            if "has_payroll_data" in panel.columns:
                payroll_rate = panel["has_payroll_data"].mean()
                results["payroll_coverage"] = payroll_rate

                if payroll_rate < 0.5:
                    results["errors"].append(
                        f"Insufficient payroll coverage: {payroll_rate:.1%}"
                    )
                    results["pass"] = False

        elif design == "pension":
            # Check age distribution
            if "pension_in_bandwidth" in panel.columns:
                in_bw = panel[panel["pension_in_bandwidth"] == 1]
                results["n_in_bandwidth"] = len(in_bw)

                if len(in_bw) < 200:
                    results["warnings"].append(
                        f"Small RDD sample: {len(in_bw)}"
                    )

            # Check pension inflow data
            if "pension_inflow" in panel.columns:
                pension_rate = (panel["pension_inflow"] > 0).mean()
                results["pension_inflow_rate"] = pension_rate

        # Outcome rates
        if "dpd30" in panel.columns:
            results["dpd30_rate"] = panel["dpd30"].mean()

        return results


def construct_mw_sample(
    panel: pd.DataFrame,
    old_minimum_wage: int = 70000,
    bandwidth: int = 15000,
) -> pd.DataFrame:
    """
    Convenience function to construct MW design sample.

    Args:
        panel: Raw panel data
        old_minimum_wage: Cutoff for treatment assignment
        bandwidth: Bandwidth around cutoff

    Returns:
        Filtered and assigned panel for MW analysis
    """
    constructor = SampleConstructor()

    # Apply filters
    panel = constructor.apply_eligibility_filters(panel)
    panel = panel[panel["eligible"]].copy()

    # Assign treatment
    panel = constructor.assign_mw_treatment(
        panel,
        old_minimum_wage=old_minimum_wage,
        bandwidth=bandwidth,
    )

    # Restrict to bandwidth
    panel = panel[panel["mw_in_bandwidth"] == 1].copy()

    # Validate
    validation = constructor.validate_sample(panel, design="mw")
    if not validation["pass"]:
        logger.error(f"Sample validation failed: {validation['errors']}")

    return panel
