"""
Credit bureau data loader for credit default study.

PARTNERSHIP REQUIRED: Access to credit bureau data requires formal
partnership agreement with State Credit Bureau or FCB.

This module provides the interface for credit bureau data if available.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CreditBureauConfig:
    """Configuration for credit bureau access."""

    provider: str  # "state_credit_bureau" or "fcb"
    api_endpoint: str | None = None
    api_key: str | None = None
    batch_size: int = 1000

    # Data availability
    has_total_debt: bool = True
    has_num_active_loans: bool = True
    has_delinquency_history: bool = True
    has_credit_score: bool = False  # Often not available


@dataclass
class BorrowerCreditProfile:
    """Credit profile for a borrower."""

    borrower_id: str

    # Debt burden
    total_debt: float | None  # Total outstanding debt
    num_active_loans: int | None  # Number of active loans
    monthly_debt_service: float | None  # Monthly debt payments

    # Delinquency history
    has_current_delinquency: bool | None
    max_dpd_last_12m: int | None  # Max days past due in last 12 months
    num_delinquencies_last_12m: int | None

    # Credit score (if available)
    credit_score: float | None = None

    # Metadata
    as_of_date: date | None = None
    data_quality_flag: str | None = None  # "good", "partial", "stale"


class CreditBureauDataSource(ABC):
    """Abstract base class for credit bureau data sources."""

    @abstractmethod
    def get_borrower_profile(
        self,
        borrower_id: str,
        as_of_date: date | None = None,
    ) -> BorrowerCreditProfile | None:
        """
        Get credit profile for a single borrower.

        Args:
            borrower_id: Borrower identifier
            as_of_date: Date for profile (default: latest)

        Returns:
            BorrowerCreditProfile or None if not found
        """
        pass

    @abstractmethod
    def get_batch_profiles(
        self,
        borrower_ids: list[str],
        as_of_date: date | None = None,
    ) -> dict[str, BorrowerCreditProfile]:
        """
        Get credit profiles for multiple borrowers.

        Args:
            borrower_ids: List of borrower identifiers
            as_of_date: Date for profiles

        Returns:
            Dictionary of borrower_id to profile
        """
        pass

    @abstractmethod
    def get_debt_service_ratio(
        self,
        borrower_id: str,
        income: float,
        as_of_date: date | None = None,
    ) -> float | None:
        """
        Calculate debt service to income ratio.

        Args:
            borrower_id: Borrower identifier
            income: Monthly income
            as_of_date: Date for calculation

        Returns:
            DSR ratio or None if unavailable
        """
        pass


class MockCreditBureauSource(CreditBureauDataSource):
    """
    Mock credit bureau source for development/testing.

    Use this when actual credit bureau access is not available.
    """

    def __init__(self):
        """Initialize mock source."""
        logger.warning(
            "Using MOCK credit bureau data. "
            "Results are synthetic and should not be used for production."
        )
        self._mock_data: dict[str, BorrowerCreditProfile] = {}

    def get_borrower_profile(
        self,
        borrower_id: str,
        as_of_date: date | None = None,
    ) -> BorrowerCreditProfile | None:
        """Get mock profile."""
        if borrower_id in self._mock_data:
            return self._mock_data[borrower_id]

        # Generate synthetic profile
        import hashlib

        hash_val = int(hashlib.md5(borrower_id.encode()).hexdigest(), 16)

        return BorrowerCreditProfile(
            borrower_id=borrower_id,
            total_debt=100000 + (hash_val % 500000),
            num_active_loans=1 + (hash_val % 5),
            monthly_debt_service=10000 + (hash_val % 30000),
            has_current_delinquency=(hash_val % 10) == 0,
            max_dpd_last_12m=0 if (hash_val % 5) > 0 else (hash_val % 90),
            num_delinquencies_last_12m=0 if (hash_val % 5) > 0 else (hash_val % 3),
            as_of_date=as_of_date or date.today(),
            data_quality_flag="mock",
        )

    def get_batch_profiles(
        self,
        borrower_ids: list[str],
        as_of_date: date | None = None,
    ) -> dict[str, BorrowerCreditProfile]:
        """Get batch mock profiles."""
        return {
            bid: self.get_borrower_profile(bid, as_of_date)
            for bid in borrower_ids
            if self.get_borrower_profile(bid, as_of_date) is not None
        }

    def get_debt_service_ratio(
        self,
        borrower_id: str,
        income: float,
        as_of_date: date | None = None,
    ) -> float | None:
        """Calculate mock DSR."""
        profile = self.get_borrower_profile(borrower_id, as_of_date)
        if profile is None or profile.monthly_debt_service is None:
            return None
        if income <= 0:
            return None
        return profile.monthly_debt_service / income


class CreditBureauLoader:
    """
    Loader for credit bureau data.

    PARTNERSHIP REQUIRED for production use.
    Falls back to mock data if no partnership configured.
    """

    def __init__(
        self,
        config: CreditBureauConfig | None = None,
        use_mock: bool = True,
    ):
        """
        Initialize loader.

        Args:
            config: Credit bureau configuration
            use_mock: Whether to use mock data (default True for safety)
        """
        self.config = config
        self.use_mock = use_mock

        if use_mock or config is None:
            self._source: CreditBureauDataSource = MockCreditBureauSource()
        else:
            # Would initialize real source here
            raise NotImplementedError(
                "Real credit bureau integration requires partnership. "
                "Contact State Credit Bureau or FCB for access."
            )

    def enrich_loan_panel(
        self,
        panel: pd.DataFrame,
        borrower_id_col: str = "borrower_id",
        as_of_date_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Enrich loan panel with credit bureau data.

        Args:
            panel: Loan panel DataFrame
            borrower_id_col: Column with borrower IDs
            as_of_date_col: Column with dates for profile lookup

        Returns:
            Panel with credit bureau columns added
        """
        df = panel.copy()

        # Get unique borrowers
        borrower_ids = df[borrower_id_col].unique().tolist()

        logger.info(f"Fetching credit bureau data for {len(borrower_ids)} borrowers")

        # Get profiles
        profiles = self._source.get_batch_profiles(borrower_ids)

        # Create lookup DataFrame
        profile_data = []
        for bid, profile in profiles.items():
            profile_data.append({
                borrower_id_col: bid,
                "cb_total_debt": profile.total_debt,
                "cb_num_active_loans": profile.num_active_loans,
                "cb_monthly_debt_service": profile.monthly_debt_service,
                "cb_has_delinquency": profile.has_current_delinquency,
                "cb_max_dpd_12m": profile.max_dpd_last_12m,
                "cb_num_delinq_12m": profile.num_delinquencies_last_12m,
                "cb_data_quality": profile.data_quality_flag,
            })

        if profile_data:
            profile_df = pd.DataFrame(profile_data)
            df = df.merge(profile_df, on=borrower_id_col, how="left")
        else:
            # Add empty columns
            for col in [
                "cb_total_debt",
                "cb_num_active_loans",
                "cb_monthly_debt_service",
                "cb_has_delinquency",
                "cb_max_dpd_12m",
                "cb_num_delinq_12m",
                "cb_data_quality",
            ]:
                df[col] = None

        logger.info(f"Added credit bureau data for {len(profiles)} borrowers")

        return df

    def compute_dti(
        self,
        panel: pd.DataFrame,
        income_col: str = "monthly_income",
        borrower_id_col: str = "borrower_id",
    ) -> pd.DataFrame:
        """
        Compute debt-to-income ratio.

        Args:
            panel: Panel with income data
            income_col: Column with monthly income
            borrower_id_col: Column with borrower IDs

        Returns:
            Panel with DTI column added
        """
        df = panel.copy()

        # Ensure credit bureau data is present
        if "cb_monthly_debt_service" not in df.columns:
            df = self.enrich_loan_panel(df, borrower_id_col)

        # Compute DTI
        df["dti"] = df["cb_monthly_debt_service"] / df[income_col]
        df["dti"] = df["dti"].replace([float("inf"), float("-inf")], None)

        return df

    def get_partnership_requirements(self) -> str:
        """Get information about partnership requirements."""
        return """
CREDIT BUREAU DATA PARTNERSHIP REQUIREMENTS

To access credit bureau data in Kazakhstan, you need:

1. STATE CREDIT BUREAU (First Credit Bureau - FCB)
   - Website: https://1cb.kz
   - Required: Formal partnership agreement
   - Data available:
     * Total outstanding debt
     * Number of active loans
     * Payment history
     * Delinquency records

2. ACCESS METHODS
   - API access: Requires technical integration
   - Batch files: Available for large queries
   - Real-time queries: Per-borrower lookups

3. CONSENT REQUIREMENTS
   - Borrower consent required for credit checks
   - Consent typically obtained at loan application
   - Must comply with data protection regulations

4. COSTS
   - Per-query fees apply
   - Volume discounts available
   - Partnership setup fees

5. INTEGRATION STEPS
   a) Contact FCB business development
   b) Sign partnership agreement
   c) Complete technical integration
   d) Test with sandbox environment
   e) Go live with production access

For more information:
- State Credit Bureau: https://1cb.kz
- National Bank of Kazakhstan: https://nationalbank.kz
"""


def get_credit_bureau_loader(use_mock: bool = True) -> CreditBureauLoader:
    """
    Get credit bureau loader instance.

    Args:
        use_mock: Whether to use mock data (default True)

    Returns:
        CreditBureauLoader instance
    """
    return CreditBureauLoader(use_mock=use_mock)
