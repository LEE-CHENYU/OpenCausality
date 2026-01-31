"""
Internal loan data loader for credit default study.

Handles:
- Loan-level data from fintech/lender systems
- Cashflow extraction (CRITICAL: payroll, pension, benefit identification)
- Default outcome construction (DPD15, DPD30, DPD60, DPD90)
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoanRecord:
    """A single loan record."""

    loan_id: str
    borrower_id: str
    origination_date: date
    loan_amount: float
    loan_term_months: int
    interest_rate: float
    region: str | None = None
    employer_type: str | None = None  # public/private/self-employed


@dataclass
class CashflowRecord:
    """Monthly cashflow record for a borrower."""

    borrower_id: str
    month: str  # YYYY-MM
    payroll_inflow: float  # Identified payroll deposits
    pension_inflow: float  # Identified pension deposits
    benefit_inflow: float  # Government benefit deposits
    total_inflow: float
    total_outflow: float
    income_source_flags: list[str]  # payroll/pension/benefits/other


@dataclass
class DefaultRecord:
    """Default outcome record."""

    loan_id: str
    month: str  # YYYY-MM
    dpd15: int  # 1 if 15+ days past due
    dpd30: int  # 1 if 30+ days past due
    dpd60: int  # 1 if 60+ days past due
    dpd90: int  # 1 if 90+ days past due
    missed_payment: int  # 1 if missed scheduled payment


class InternalLoansLoader:
    """
    Loader for internal loan and cashflow data.

    CRITICAL: Income must be measured from cashflows (not stated income)
    to avoid attenuation bias.
    """

    def __init__(self, data_path: Path | None = None):
        """
        Initialize loader.

        Args:
            data_path: Path to internal data directory
        """
        self.data_path = data_path or Path("data/internal")

    def load_loans(
        self,
        origination_before: date | None = None,
        origination_after: date | None = None,
    ) -> pd.DataFrame:
        """
        Load loan-level data.

        Args:
            origination_before: Filter loans originated before this date
            origination_after: Filter loans originated after this date

        Returns:
            DataFrame with loan records
        """
        # Check if data exists
        loans_file = self.data_path / "loans.parquet"

        if not loans_file.exists():
            logger.warning(
                f"Loan data not found at {loans_file}. "
                "This module requires internal fintech/lender data. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        df = pd.read_parquet(loans_file)

        # Apply filters
        if "origination_date" in df.columns:
            df["origination_date"] = pd.to_datetime(df["origination_date"])

            if origination_before:
                df = df[df["origination_date"] < pd.Timestamp(origination_before)]

            if origination_after:
                df = df[df["origination_date"] > pd.Timestamp(origination_after)]

        logger.info(f"Loaded {len(df)} loans")
        return df

    def load_cashflows(
        self,
        borrower_ids: list[str] | None = None,
        start_month: str | None = None,
        end_month: str | None = None,
    ) -> pd.DataFrame:
        """
        Load borrower cashflow data.

        CRITICAL: This is the primary income measurement.
        Stated income is too noisy and will attenuate results.

        Args:
            borrower_ids: Filter to specific borrowers
            start_month: Start month (YYYY-MM)
            end_month: End month (YYYY-MM)

        Returns:
            DataFrame with cashflow records
        """
        cashflows_file = self.data_path / "cashflows.parquet"

        if not cashflows_file.exists():
            logger.warning(
                f"Cashflow data not found at {cashflows_file}. "
                "CRITICAL: Income must be measured from cashflows, not stated income. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        df = pd.read_parquet(cashflows_file)

        # Apply filters
        if borrower_ids:
            df = df[df["borrower_id"].isin(borrower_ids)]

        if start_month:
            df = df[df["month"] >= start_month]

        if end_month:
            df = df[df["month"] <= end_month]

        # Validate required columns
        required_cols = ["borrower_id", "month", "payroll_inflow"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"Cashflow data missing required columns: {missing}")
            return pd.DataFrame()

        logger.info(f"Loaded {len(df)} cashflow records for {df['borrower_id'].nunique()} borrowers")
        return df

    def load_defaults(
        self,
        loan_ids: list[str] | None = None,
        start_month: str | None = None,
        end_month: str | None = None,
    ) -> pd.DataFrame:
        """
        Load default outcome data.

        Args:
            loan_ids: Filter to specific loans
            start_month: Start month (YYYY-MM)
            end_month: End month (YYYY-MM)

        Returns:
            DataFrame with default records
        """
        defaults_file = self.data_path / "defaults.parquet"

        if not defaults_file.exists():
            logger.warning(
                f"Default data not found at {defaults_file}. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        df = pd.read_parquet(defaults_file)

        # Apply filters
        if loan_ids:
            df = df[df["loan_id"].isin(loan_ids)]

        if start_month:
            df = df[df["month"] >= start_month]

        if end_month:
            df = df[df["month"] <= end_month]

        logger.info(f"Loaded {len(df)} default records for {df['loan_id'].nunique()} loans")
        return df

    def compute_pre_policy_payroll(
        self,
        cashflows: pd.DataFrame,
        window_start: str,
        window_end: str,
    ) -> pd.DataFrame:
        """
        Compute pre-policy monthly payroll for each borrower.

        This is the running variable for the MW design.

        Args:
            cashflows: Cashflow DataFrame
            window_start: Start of window (YYYY-MM)
            window_end: End of window (YYYY-MM)

        Returns:
            DataFrame with borrower_id, pre_policy_monthly_payroll
        """
        # Filter to window
        window = cashflows[
            (cashflows["month"] >= window_start) &
            (cashflows["month"] <= window_end)
        ].copy()

        # Compute average monthly payroll
        payroll = (
            window
            .groupby("borrower_id")["payroll_inflow"]
            .mean()
            .reset_index()
            .rename(columns={"payroll_inflow": "pre_policy_monthly_payroll"})
        )

        # Flag borrowers with payroll data
        payroll["has_payroll_data"] = payroll["pre_policy_monthly_payroll"] > 0

        logger.info(
            f"Computed pre-policy payroll for {len(payroll)} borrowers. "
            f"{payroll['has_payroll_data'].sum()} have identifiable payroll."
        )

        return payroll

    def identify_formal_workers(
        self,
        cashflows: pd.DataFrame,
        min_payroll_months: int = 3,
    ) -> pd.DataFrame:
        """
        Identify formal workers with consistent payroll inflows.

        Args:
            cashflows: Cashflow DataFrame
            min_payroll_months: Minimum months with payroll for eligibility

        Returns:
            DataFrame with borrower_id, is_formal_worker, payroll_months
        """
        # Count months with payroll inflow
        payroll_counts = (
            cashflows[cashflows["payroll_inflow"] > 0]
            .groupby("borrower_id")
            .size()
            .reset_index(name="payroll_months")
        )

        # Flag formal workers
        payroll_counts["is_formal_worker"] = (
            payroll_counts["payroll_months"] >= min_payroll_months
        )

        logger.info(
            f"Identified {payroll_counts['is_formal_worker'].sum()} formal workers "
            f"out of {len(payroll_counts)} borrowers with any payroll."
        )

        return payroll_counts

    def identify_near_retirees(
        self,
        borrowers: pd.DataFrame,
        reference_date: date,
        age_window_men: tuple[int, int] = (60, 66),
        age_window_women: tuple[int, int] = (58, 64),
    ) -> pd.DataFrame:
        """
        Identify borrowers near pension eligibility threshold.

        Args:
            borrowers: Borrower DataFrame with birth_date, gender
            reference_date: Date to compute age
            age_window_men: (min_age, max_age) for men
            age_window_women: (min_age, max_age) for women

        Returns:
            DataFrame with borrower_id, age, gender, near_pension_threshold
        """
        df = borrowers.copy()

        if "birth_date" not in df.columns or "gender" not in df.columns:
            logger.error("Borrower data missing birth_date or gender columns")
            return pd.DataFrame()

        # Compute age
        df["birth_date"] = pd.to_datetime(df["birth_date"])
        df["age"] = (
            (pd.Timestamp(reference_date) - df["birth_date"]).dt.days / 365.25
        ).astype(int)

        # Apply gender-specific windows
        df["near_pension_threshold"] = False

        # Men
        men_mask = df["gender"].str.upper() == "M"
        df.loc[
            men_mask &
            (df["age"] >= age_window_men[0]) &
            (df["age"] <= age_window_men[1]),
            "near_pension_threshold"
        ] = True

        # Women
        women_mask = df["gender"].str.upper() == "F"
        df.loc[
            women_mask &
            (df["age"] >= age_window_women[0]) &
            (df["age"] <= age_window_women[1]),
            "near_pension_threshold"
        ] = True

        logger.info(
            f"Identified {df['near_pension_threshold'].sum()} borrowers "
            "near pension threshold."
        )

        return df[["borrower_id", "age", "gender", "near_pension_threshold"]]


def validate_data_quality(
    loans: pd.DataFrame,
    cashflows: pd.DataFrame,
    defaults: pd.DataFrame,
) -> dict[str, Any]:
    """
    Validate data quality for credit default study.

    Args:
        loans: Loan data
        cashflows: Cashflow data
        defaults: Default data

    Returns:
        Dictionary with validation results
    """
    results = {
        "pass": True,
        "issues": [],
        "warnings": [],
    }

    # Check loans
    if loans.empty:
        results["pass"] = False
        results["issues"].append("No loan data available")
    else:
        results["n_loans"] = len(loans)

    # Check cashflows
    if cashflows.empty:
        results["pass"] = False
        results["issues"].append("No cashflow data available - CRITICAL")
    else:
        results["n_borrowers_with_cashflows"] = cashflows["borrower_id"].nunique()

        # Check payroll coverage
        if "payroll_inflow" in cashflows.columns:
            payroll_coverage = (
                cashflows[cashflows["payroll_inflow"] > 0]["borrower_id"].nunique() /
                cashflows["borrower_id"].nunique()
            )
            results["payroll_coverage"] = payroll_coverage

            if payroll_coverage < 0.3:
                results["warnings"].append(
                    f"Low payroll coverage ({payroll_coverage:.1%}) - MW design may be underpowered"
                )

    # Check defaults
    if defaults.empty:
        results["pass"] = False
        results["issues"].append("No default data available")
    else:
        results["n_loan_months"] = len(defaults)

        # Check outcome rates
        if "dpd30" in defaults.columns:
            dpd30_rate = defaults["dpd30"].mean()
            results["dpd30_rate"] = dpd30_rate

            if dpd30_rate < 0.01:
                results["warnings"].append(
                    f"Very low DPD30 rate ({dpd30_rate:.2%}) - may lack power"
                )
            if dpd30_rate > 0.30:
                results["warnings"].append(
                    f"Very high DPD30 rate ({dpd30_rate:.2%}) - check sample selection"
                )

    return results
