"""
Loan-month panel construction for credit default study.

Builds analysis panel from loan, cashflow, and default data.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from studies.credit_default.src.internal_loans import InternalLoansLoader

logger = logging.getLogger(__name__)


@dataclass
class PanelConfig:
    """Configuration for panel construction."""

    # Sample restrictions
    origination_before: date
    outcome_start: str  # YYYY-MM
    outcome_end: str  # YYYY-MM

    # Pre-policy payroll window
    payroll_window_start: str  # YYYY-MM
    payroll_window_end: str  # YYYY-MM

    # Minimum wage thresholds
    old_minimum_wage: int = 70000
    new_minimum_wage: int = 85000

    # Treatment date
    treatment_date: str = "2024-01"  # YYYY-MM

    # Pension thresholds
    pension_age_men: int = 63
    pension_age_women: int = 61


class LoanPanelBuilder:
    """
    Builds loan-month panel for credit default analysis.

    Panel structure: loan_id x month
    """

    def __init__(self, config: PanelConfig | None = None):
        """
        Initialize panel builder.

        Args:
            config: Panel configuration
        """
        self.config = config or PanelConfig(
            origination_before=date(2023, 12, 1),
            outcome_start="2024-01",
            outcome_end="2024-05",
            payroll_window_start="2023-10",
            payroll_window_end="2023-12",
        )
        self.loader = InternalLoansLoader()

    def build(self) -> pd.DataFrame:
        """
        Build the analysis panel.

        Returns:
            Panel DataFrame with loan-month observations
        """
        # Load raw data
        loans = self.loader.load_loans(
            origination_before=self.config.origination_before
        )

        if loans.empty:
            raise ValueError(
                "No loan data available. "
                "This study requires internal fintech/lender data."
            )

        borrower_ids = loans["borrower_id"].unique().tolist()

        # Load cashflows
        cashflows = self.loader.load_cashflows(
            borrower_ids=borrower_ids,
            start_month=self.config.payroll_window_start,
            end_month=self.config.outcome_end,
        )

        # Load defaults
        loan_ids = loans["loan_id"].unique().tolist()
        defaults = self.loader.load_defaults(
            loan_ids=loan_ids,
            start_month=self.config.outcome_start,
            end_month=self.config.outcome_end,
        )

        # Create panel skeleton
        panel = self._create_skeleton(loans)

        # Add pre-policy payroll (running variable for MW design)
        panel = self._add_payroll(panel, cashflows)

        # Add treatment indicators
        panel = self._add_treatment_indicators(panel)

        # Add default outcomes
        panel = self._add_outcomes(panel, defaults)

        # Add covariates
        panel = self._add_covariates(panel, loans)

        return panel

    def _create_skeleton(self, loans: pd.DataFrame) -> pd.DataFrame:
        """Create panel skeleton with all loan-month combinations."""
        # Generate months in outcome window
        months = pd.date_range(
            start=self.config.outcome_start,
            end=self.config.outcome_end,
            freq="MS",
        ).strftime("%Y-%m").tolist()

        # Create all loan x month combinations
        rows = []
        for _, loan in loans.iterrows():
            for month in months:
                rows.append({
                    "loan_id": loan["loan_id"],
                    "borrower_id": loan["borrower_id"],
                    "month": month,
                })

        panel = pd.DataFrame(rows)
        logger.info(f"Created panel skeleton: {len(panel)} loan-months")

        return panel

    def _add_payroll(
        self,
        panel: pd.DataFrame,
        cashflows: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add pre-policy payroll to panel."""
        if cashflows.empty:
            logger.warning("No cashflow data - payroll variables will be missing")
            panel["pre_policy_payroll"] = np.nan
            panel["has_payroll_data"] = False
            return panel

        # Compute pre-policy payroll per borrower
        pre_policy = self.loader.compute_pre_policy_payroll(
            cashflows,
            self.config.payroll_window_start,
            self.config.payroll_window_end,
        )

        # Merge to panel
        panel = panel.merge(
            pre_policy[["borrower_id", "pre_policy_monthly_payroll", "has_payroll_data"]],
            on="borrower_id",
            how="left",
        )

        panel["pre_policy_payroll"] = panel["pre_policy_monthly_payroll"]
        panel["has_payroll_data"] = panel["has_payroll_data"].fillna(False)

        logger.info(
            f"Added payroll data. "
            f"{panel['has_payroll_data'].sum()} observations have payroll data."
        )

        return panel

    def _add_treatment_indicators(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add treatment indicators for MW diff-in-discs."""
        # Post indicator
        panel["post"] = (panel["month"] >= self.config.treatment_date).astype(int)

        # Below cutoff indicator (for MW design)
        panel["below_old_mw"] = (
            panel["pre_policy_payroll"] < self.config.old_minimum_wage
        ).astype(int)

        # Near cutoff indicator (for bandwidth restriction)
        # Default bandwidth: 15,000 tenge around cutoff
        bandwidth = 15000
        panel["near_cutoff"] = (
            (panel["pre_policy_payroll"] >= self.config.old_minimum_wage - bandwidth) &
            (panel["pre_policy_payroll"] <= self.config.old_minimum_wage + bandwidth)
        ).astype(int)

        # Running variable centered at cutoff
        panel["payroll_centered"] = (
            panel["pre_policy_payroll"] - self.config.old_minimum_wage
        )

        # Predicted statutory raise (intensity instrument)
        panel["predicted_raise"] = np.maximum(
            0,
            self.config.new_minimum_wage - panel["pre_policy_payroll"]
        )

        # Treatment interaction (main coefficient of interest)
        panel["post_x_below"] = panel["post"] * panel["below_old_mw"]
        panel["post_x_raise"] = panel["post"] * panel["predicted_raise"]

        return panel

    def _add_outcomes(
        self,
        panel: pd.DataFrame,
        defaults: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add default outcomes to panel."""
        if defaults.empty:
            logger.warning("No default data - outcomes will be missing")
            for col in ["dpd15", "dpd30", "dpd60", "dpd90", "missed_payment"]:
                panel[col] = np.nan
            return panel

        # Merge defaults
        panel = panel.merge(
            defaults,
            on=["loan_id", "month"],
            how="left",
        )

        # Fill missing outcomes with 0 (loan was current)
        outcome_cols = ["dpd15", "dpd30", "dpd60", "dpd90", "missed_payment"]
        for col in outcome_cols:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(int)
            else:
                panel[col] = 0

        logger.info(
            f"Added outcomes. DPD30 rate: {panel['dpd30'].mean():.2%}"
        )

        return panel

    def _add_covariates(
        self,
        panel: pd.DataFrame,
        loans: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add loan-level covariates to panel."""
        # Select covariates from loans
        covariate_cols = [
            "loan_id",
            "loan_amount",
            "loan_term_months",
            "interest_rate",
            "region",
            "employer_type",
            "origination_date",
        ]
        available_cols = [c for c in covariate_cols if c in loans.columns]

        if len(available_cols) > 1:  # More than just loan_id
            panel = panel.merge(
                loans[available_cols],
                on="loan_id",
                how="left",
            )

        return panel

    def build_mw_sample(self) -> pd.DataFrame:
        """
        Build sample specifically for MW diff-in-discs design.

        Applies additional restrictions:
        - Requires payroll inflow data
        - Restricts to near-cutoff bandwidth
        """
        panel = self.build()

        # Restrict to borrowers with payroll data
        panel = panel[panel["has_payroll_data"] == True].copy()

        if len(panel) == 0:
            raise ValueError(
                "No observations with payroll data. "
                "Cannot run MW design without cashflow-based income measurement."
            )

        # Restrict to near cutoff
        panel = panel[panel["near_cutoff"] == 1].copy()

        logger.info(
            f"MW design sample: {len(panel)} loan-months, "
            f"{panel['loan_id'].nunique()} loans"
        )

        return panel


def build_analysis_panel(
    config: PanelConfig | None = None,
) -> pd.DataFrame:
    """
    Convenience function to build analysis panel.

    Args:
        config: Panel configuration

    Returns:
        Panel DataFrame
    """
    builder = LoanPanelBuilder(config)
    return builder.build()
