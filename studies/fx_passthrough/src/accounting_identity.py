"""
Block C: Real Income Decomposition (Accounting Identity)

Identity:
    Δlog(real income) = Δlog(nominal income) - Δlog(CPI)

Procedure:
1. Estimate nominal income responses separately (from Block B)
2. Estimate CPI responses separately (from Block A aggregate)
3. Construct real income effects as derived object

CAUTION:
    Do NOT run IV directly on "real income" as outcome when CPI is instrumented.
    This creates mechanical IV issues.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RealIncomeDecomposition:
    """
    Decomposition of real income response.

    Δlog(real) = Δlog(nominal) - Δlog(CPI)
    """

    horizons: list[int]

    # Nominal income response (from Block B)
    nominal_response: np.ndarray
    nominal_se: np.ndarray

    # CPI response (from Block A aggregate)
    cpi_response: np.ndarray
    cpi_se: np.ndarray

    # Derived real income response
    real_response: np.ndarray
    real_se: np.ndarray  # Computed via delta method

    # Decomposition
    price_channel: np.ndarray  # -CPI response
    nominal_channel: np.ndarray  # Nominal response

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "=" * 70,
            "Block C: Real Income Decomposition",
            "=" * 70,
            "",
            "Identity: Δlog(real) = Δlog(nominal) - Δlog(CPI)",
            "",
            f"{'Horizon':>8} {'Real':>10} {'Nominal':>10} {'CPI':>10} "
            f"{'Price Ch.':>10} {'Nom. Ch.':>10}",
            "-" * 70,
        ]

        for i, h in enumerate(self.horizons):
            lines.append(
                f"{h:>8} {self.real_response[i]:>10.4f} "
                f"{self.nominal_response[i]:>10.4f} {self.cpi_response[i]:>10.4f} "
                f"{self.price_channel[i]:>10.4f} {self.nominal_channel[i]:>10.4f}"
            )

        lines.extend([
            "",
            "Interpretation:",
            f"  Total real income effect (cumulative): {self.real_response[-1]:.4f}",
            f"  Due to price level (CPI): {self.price_channel[-1]:.4f}",
            f"  Due to nominal income: {self.nominal_channel[-1]:.4f}",
        ])

        return "\n".join(lines)


class AccountingIdentity:
    """
    Accounting identity decomposition for real income.

    Combines Block A (CPI response) and Block B (nominal income response)
    to construct real income effects without running a problematic IV regression.
    """

    def __init__(
        self,
        block_a_result: Any = None,
        block_b_result: Any = None,
    ):
        """
        Initialize with Block A and B results.

        Args:
            block_a_result: CPIPassThroughResult from Block A
            block_b_result: IncomeLPIVResult from Block B
        """
        self.block_a = block_a_result
        self.block_b = block_b_result
        self.decomposition: RealIncomeDecomposition | None = None

    def set_block_a(self, result: Any) -> None:
        """Set Block A results."""
        self.block_a = result

    def set_block_b(self, result: Any) -> None:
        """Set Block B results."""
        self.block_b = result

    def compute_decomposition(
        self,
        horizons: list[int] | None = None,
    ) -> RealIncomeDecomposition:
        """
        Compute real income decomposition.

        Δlog(real) = Δlog(nominal) - Δlog(CPI)

        Args:
            horizons: List of horizons to compute

        Returns:
            RealIncomeDecomposition with channel attribution
        """
        if self.block_a is None or self.block_b is None:
            raise ValueError("Both Block A and Block B results required")

        # Get horizons from results
        if horizons is None:
            horizons = self.block_b.horizons

        n_horizons = len(horizons)

        # Extract nominal income responses (from Block B)
        nominal_response = self.block_b.coefficients
        nominal_se = self.block_b.std_errors

        # Extract CPI responses (from Block A)
        # Need to align horizons and potentially interpolate
        if hasattr(self.block_a, "irf_coefficients") and self.block_a.irf_coefficients is not None:
            cpi_response = self.block_a.irf_coefficients
            cpi_se = self.block_a.irf_std_errors
        else:
            # Use static coefficient, assume same response at all horizons
            cpi_response = np.full(n_horizons, self.block_a.beta)
            cpi_se = np.full(n_horizons, self.block_a.beta_se)

        # Ensure same length
        min_len = min(len(nominal_response), len(cpi_response), n_horizons)
        horizons = horizons[:min_len]
        nominal_response = nominal_response[:min_len]
        nominal_se = nominal_se[:min_len]
        cpi_response = cpi_response[:min_len]
        cpi_se = cpi_se[:min_len]

        # Compute real income response
        # Δlog(real) = Δlog(nominal) - Δlog(CPI)
        # But Block A estimates CPI pass-through, not CPI level change
        # So we need to think about units carefully

        # Interpretation:
        # - Block B: β_h = d(income_growth)/d(inflation) instrumented
        # - Block A: β_A = d(category_inflation)/d(import_share × FX)

        # For accounting identity, we want:
        # real_income_response = nominal_income_response - cpi_response
        real_response = nominal_response - cpi_response

        # Delta method for SE: Var(X-Y) = Var(X) + Var(Y) - 2Cov(X,Y)
        # Assuming independence (conservative): Var(X-Y) = Var(X) + Var(Y)
        real_se = np.sqrt(nominal_se**2 + cpi_se**2)

        # Decomposition into channels
        price_channel = -cpi_response  # Negative: higher CPI reduces real income
        nominal_channel = nominal_response

        self.decomposition = RealIncomeDecomposition(
            horizons=horizons,
            nominal_response=nominal_response,
            nominal_se=nominal_se,
            cpi_response=cpi_response,
            cpi_se=cpi_se,
            real_response=real_response,
            real_se=real_se,
            price_channel=price_channel,
            nominal_channel=nominal_channel,
        )

        return self.decomposition

    def compute_from_data(
        self,
        data: pd.DataFrame,
        nominal_col: str = "nominal_income",
        cpi_col: str = "cpi_index",
        inflation_col: str = "headline_inflation",
        instrument_col: str = "imported_inflation",
        max_horizon: int = 12,
    ) -> RealIncomeDecomposition:
        """
        Compute decomposition directly from data.

        This is a convenience method that estimates both components.

        Args:
            data: Time series DataFrame
            nominal_col: Nominal income column
            cpi_col: CPI index column
            inflation_col: Inflation column (endogenous)
            instrument_col: Imported inflation instrument
            max_horizon: Maximum horizon

        Returns:
            RealIncomeDecomposition
        """
        import statsmodels.api as sm

        data = data.copy()

        # Compute growth rates
        data["nominal_growth"] = np.log(data[nominal_col]).diff()
        data["cpi_growth"] = np.log(data[cpi_col]).diff()
        data["real_growth"] = data["nominal_growth"] - data["cpi_growth"]

        horizons = list(range(max_horizon + 1))

        nominal_responses = []
        nominal_ses = []
        cpi_responses = []
        cpi_ses = []

        for h in horizons:
            # Create h-ahead outcomes
            data[f"nominal_h{h}"] = data["nominal_growth"].shift(-h)
            data[f"cpi_h{h}"] = data["cpi_growth"].shift(-h)

            # LP-IV for nominal income
            try:
                y = data[f"nominal_h{h}"].dropna()
                X = sm.add_constant(data.loc[y.index, inflation_col])
                Z = sm.add_constant(data.loc[y.index, instrument_col])

                # First stage
                first_stage = sm.OLS(X[inflation_col], Z).fit()
                fitted = first_stage.fittedvalues

                # Second stage
                X_fitted = sm.add_constant(fitted)
                second_stage = sm.OLS(y, X_fitted).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

                nominal_responses.append(second_stage.params.iloc[1])
                nominal_ses.append(second_stage.bse.iloc[1])

            except Exception:
                nominal_responses.append(np.nan)
                nominal_ses.append(np.nan)

            # LP-IV for CPI
            try:
                y = data[f"cpi_h{h}"].dropna()
                X = sm.add_constant(data.loc[y.index, inflation_col])
                Z = sm.add_constant(data.loc[y.index, instrument_col])

                first_stage = sm.OLS(X[inflation_col], Z).fit()
                fitted = first_stage.fittedvalues

                X_fitted = sm.add_constant(fitted)
                second_stage = sm.OLS(y, X_fitted).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

                cpi_responses.append(second_stage.params.iloc[1])
                cpi_ses.append(second_stage.bse.iloc[1])

            except Exception:
                cpi_responses.append(np.nan)
                cpi_ses.append(np.nan)

        nominal_responses = np.array(nominal_responses)
        nominal_ses = np.array(nominal_ses)
        cpi_responses = np.array(cpi_responses)
        cpi_ses = np.array(cpi_ses)

        # Real income = nominal - CPI
        real_responses = nominal_responses - cpi_responses
        real_ses = np.sqrt(nominal_ses**2 + cpi_ses**2)

        self.decomposition = RealIncomeDecomposition(
            horizons=horizons,
            nominal_response=nominal_responses,
            nominal_se=nominal_ses,
            cpi_response=cpi_responses,
            cpi_se=cpi_ses,
            real_response=real_responses,
            real_se=real_ses,
            price_channel=-cpi_responses,
            nominal_channel=nominal_responses,
        )

        return self.decomposition

    def welfare_interpretation(self) -> dict[str, Any]:
        """
        Interpret decomposition in welfare terms.

        Returns dictionary with economic interpretation.
        """
        if self.decomposition is None:
            raise ValueError("Run compute_decomposition first")

        d = self.decomposition

        # Cumulative effects at maximum horizon
        final_real = d.real_response[-1]
        final_price = d.price_channel[-1]
        final_nominal = d.nominal_channel[-1]

        interpretation = {
            "total_real_effect": final_real,
            "price_level_effect": final_price,
            "nominal_income_effect": final_nominal,
            "price_share_of_total": abs(final_price) / (abs(final_price) + abs(final_nominal))
            if (abs(final_price) + abs(final_nominal)) > 0 else np.nan,
            "nominal_share_of_total": abs(final_nominal) / (abs(final_price) + abs(final_nominal))
            if (abs(final_price) + abs(final_nominal)) > 0 else np.nan,
        }

        # Narrative interpretation
        if final_real < 0:
            interpretation["narrative"] = (
                f"Externally-driven inflation reduces real income by {abs(final_real):.2%}. "
                f"Of this, {abs(final_price):.2%} comes from higher prices "
                f"and {abs(final_nominal):.2%} from nominal income changes."
            )
        else:
            interpretation["narrative"] = (
                f"Externally-driven inflation increases real income by {final_real:.2%}. "
                f"Nominal income adjustment ({final_nominal:.2%}) exceeds "
                f"price level effect ({abs(final_price):.2%})."
            )

        return interpretation
