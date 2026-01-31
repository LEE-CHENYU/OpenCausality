"""
Local Projections estimator for credit quality responses to external shocks.

Estimates: ΔCreditQuality_{t+h} = α_h + β_h S_t + Γ_h'X_{t-1} + u_{t,h}

Where S_t are exogenous external shocks (oil, risk, demand).

This is REDUCED-FORM analysis: we estimate shock → credit quality,
NOT causal income → default elasticity.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class LPResult:
    """Results from a single horizon local projection."""

    horizon: int
    coefficient: float
    std_error: float
    t_stat: float
    pvalue: float
    conf_int: tuple[float, float]
    n_obs: int


@dataclass
class CreditLPResults:
    """Full local projection results for credit quality analysis."""

    # IRF estimates by horizon
    irf: dict[str, list[LPResult]]  # shock_name -> list of LPResult by horizon

    # Metadata
    outcome: str
    shocks: list[str]
    controls: list[str]
    max_horizon: int
    n_obs: int
    sample_start: str
    sample_end: str

    # Falsification
    leads_tests: dict[str, list[LPResult]] | None = None

    def get_irf_dataframe(self, shock: str) -> pd.DataFrame:
        """Get IRF as DataFrame for plotting."""
        if shock not in self.irf:
            raise ValueError(f"Shock {shock} not in results")

        records = []
        for result in self.irf[shock]:
            records.append({
                "horizon": result.horizon,
                "coefficient": result.coefficient,
                "std_error": result.std_error,
                "ci_lower": result.conf_int[0],
                "ci_upper": result.conf_int[1],
                "pvalue": result.pvalue,
            })

        return pd.DataFrame(records)

    def summary(self) -> str:
        """Generate text summary of results."""
        lines = []
        lines.append("=" * 70)
        lines.append("CREDIT QUALITY LOCAL PROJECTIONS RESULTS")
        lines.append("=" * 70)
        lines.append(f"\nOutcome: {self.outcome}")
        lines.append(f"Sample: {self.sample_start} to {self.sample_end}")
        lines.append(f"N observations: {self.n_obs}")
        lines.append(f"Max horizon: {self.max_horizon}")

        lines.append(f"\nShocks analyzed: {', '.join(self.shocks)}")
        lines.append(f"Controls: {', '.join(self.controls) if self.controls else 'None'}")

        for shock in self.shocks:
            lines.append(f"\n--- {shock} ---")
            lines.append(f"{'Horizon':>8} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8} {'90% CI':>20}")
            lines.append("-" * 70)

            for result in self.irf[shock]:
                ci_str = f"[{result.conf_int[0]:.4f}, {result.conf_int[1]:.4f}]"
                sig = "*" if result.pvalue < 0.10 else ""
                lines.append(
                    f"{result.horizon:>8} {result.coefficient:>10.4f} "
                    f"{result.std_error:>10.4f} {result.t_stat:>8.2f} "
                    f"{result.pvalue:>8.4f} {ci_str:>20}{sig}"
                )

        # Falsification tests
        if self.leads_tests:
            lines.append("\n--- FALSIFICATION: LEADS TESTS ---")
            lines.append("(Future shocks should NOT predict current credit quality)")
            for shock, results in self.leads_tests.items():
                lines.append(f"\n{shock}:")
                for result in results:
                    sig = "FAIL" if result.pvalue < 0.10 else "pass"
                    lines.append(
                        f"  h={result.horizon}: coef={result.coefficient:.4f}, "
                        f"p={result.pvalue:.4f} [{sig}]"
                    )

        lines.append("\n" + "=" * 70)
        lines.append("IMPORTANT CAVEATS:")
        lines.append("- This is REDUCED-FORM evidence (shock → credit quality)")
        lines.append("- NOT causal income → default elasticity")
        lines.append("- Aggregate NPL may differ from specific loan product defaults")
        lines.append("=" * 70)

        return "\n".join(lines)


class CreditLocalProjections:
    """
    Local projections estimator for credit quality responses.

    Estimates impulse response functions of credit quality measures
    to external shocks using Jordà (2005) local projections.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        date_col: str = "date",
    ):
        """
        Initialize LP estimator.

        Args:
            data: Time series data with shocks and credit quality
            date_col: Name of date column
        """
        self.data = data.copy()
        self.date_col = date_col

        # Ensure date is datetime and sorted
        if date_col in self.data.columns:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            self.data = self.data.sort_values(date_col).reset_index(drop=True)

    def estimate(
        self,
        outcome: str,
        shocks: list[str],
        controls: list[str] | None = None,
        max_horizon: int = 12,
        horizons: list[int] | None = None,
        newey_west_lags: int = 4,
        confidence_level: float = 0.90,
    ) -> CreditLPResults:
        """
        Estimate local projections.

        Args:
            outcome: Credit quality outcome variable
            shocks: List of shock variables
            controls: List of control variables (lagged)
            max_horizon: Maximum forecast horizon
            horizons: Specific horizons to estimate (default: 0 to max_horizon)
            newey_west_lags: Lags for HAC standard errors
            confidence_level: Confidence level for intervals

        Returns:
            CreditLPResults with IRF estimates
        """
        controls = controls or []
        horizons = horizons or list(range(max_horizon + 1))

        # Validate columns
        required = [outcome] + shocks + controls
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Store results
        irf = {shock: [] for shock in shocks}

        for h in horizons:
            logger.info(f"Estimating horizon h={h}")

            # Create forward outcome
            y_col = f"{outcome}_h{h}"
            self.data[y_col] = self.data[outcome].shift(-h)

            # Build regression data
            df_reg = self.data.dropna(subset=[y_col] + shocks + controls)

            if len(df_reg) < 20:
                logger.warning(f"Insufficient observations at h={h}: {len(df_reg)}")
                continue

            # Design matrix: shocks + controls
            X_cols = shocks + controls
            X = sm.add_constant(df_reg[X_cols])
            y = df_reg[y_col]

            # Estimate with HAC standard errors
            try:
                model = sm.OLS(y, X).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": newey_west_lags},
                )

                # Extract shock coefficients
                alpha = 1 - confidence_level
                z = stats.norm.ppf(1 - alpha / 2)

                for shock in shocks:
                    coef = model.params[shock]
                    se = model.bse[shock]
                    t = model.tvalues[shock]
                    p = model.pvalues[shock]
                    ci = (coef - z * se, coef + z * se)

                    irf[shock].append(LPResult(
                        horizon=h,
                        coefficient=coef,
                        std_error=se,
                        t_stat=t,
                        pvalue=p,
                        conf_int=ci,
                        n_obs=len(y),
                    ))

            except Exception as e:
                logger.warning(f"Estimation failed at h={h}: {e}")

        # Sample info
        sample_start = self.data[self.date_col].min().strftime("%Y-%m-%d")
        sample_end = self.data[self.date_col].max().strftime("%Y-%m-%d")

        return CreditLPResults(
            irf=irf,
            outcome=outcome,
            shocks=shocks,
            controls=controls,
            max_horizon=max_horizon,
            n_obs=len(self.data),
            sample_start=sample_start,
            sample_end=sample_end,
        )

    def run_leads_test(
        self,
        outcome: str,
        shocks: list[str],
        controls: list[str] | None = None,
        leads: list[int] | None = None,
        newey_west_lags: int = 4,
    ) -> dict[str, list[LPResult]]:
        """
        Run falsification test: future shocks should not predict current outcome.

        This tests whether S_{t+k} predicts Y_t for k > 0.
        Under correct specification, these should be zero.

        Args:
            outcome: Outcome variable
            shocks: Shock variables
            controls: Control variables
            leads: Lead periods to test (default: [1, 2, 3])
            newey_west_lags: Lags for HAC SE

        Returns:
            Dictionary of shock -> list of LPResult for each lead
        """
        controls = controls or []
        leads = leads or [1, 2, 3]

        results = {shock: [] for shock in shocks}

        for k in leads:
            # Create lead shock variables
            df_test = self.data.copy()

            for shock in shocks:
                lead_col = f"{shock}_lead{k}"
                df_test[lead_col] = df_test[shock].shift(-k)

            # Regression: Y_t on S_{t+k}
            lead_shocks = [f"{s}_lead{k}" for s in shocks]
            df_reg = df_test.dropna(subset=[outcome] + lead_shocks + controls)

            if len(df_reg) < 20:
                logger.warning(f"Insufficient obs for leads test k={k}")
                continue

            X = sm.add_constant(df_reg[lead_shocks + controls])
            y = df_reg[outcome]

            try:
                model = sm.OLS(y, X).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": newey_west_lags},
                )

                for shock in shocks:
                    lead_col = f"{shock}_lead{k}"
                    coef = model.params[lead_col]
                    se = model.bse[lead_col]
                    t = model.tvalues[lead_col]
                    p = model.pvalues[lead_col]

                    results[shock].append(LPResult(
                        horizon=-k,  # Negative to indicate lead
                        coefficient=coef,
                        std_error=se,
                        t_stat=t,
                        pvalue=p,
                        conf_int=(coef - 1.645 * se, coef + 1.645 * se),
                        n_obs=len(y),
                    ))

            except Exception as e:
                logger.warning(f"Leads test failed for k={k}: {e}")

        return results

    def estimate_with_falsification(
        self,
        outcome: str,
        shocks: list[str],
        controls: list[str] | None = None,
        max_horizon: int = 12,
        leads: list[int] | None = None,
    ) -> CreditLPResults:
        """
        Estimate LP with falsification tests included.

        Args:
            outcome: Outcome variable
            shocks: Shock variables
            controls: Control variables
            max_horizon: Max horizon
            leads: Lead periods for falsification

        Returns:
            CreditLPResults with leads_tests populated
        """
        # Main estimation
        results = self.estimate(
            outcome=outcome,
            shocks=shocks,
            controls=controls,
            max_horizon=max_horizon,
        )

        # Falsification
        leads_results = self.run_leads_test(
            outcome=outcome,
            shocks=shocks,
            controls=controls,
            leads=leads,
        )

        results.leads_tests = leads_results

        return results


def estimate_credit_lp(
    data: pd.DataFrame,
    outcome: str = "npl_ratio",
    shocks: list[str] | None = None,
    max_horizon: int = 12,
) -> CreditLPResults:
    """
    Convenience function to estimate credit quality local projections.

    Args:
        data: Time series data
        outcome: Credit quality outcome
        shocks: External shocks (default: oil_supply, vix)
        max_horizon: Maximum horizon

    Returns:
        CreditLPResults
    """
    shocks = shocks or ["oil_supply_shock", "vix_innovation"]

    lp = CreditLocalProjections(data)
    return lp.estimate_with_falsification(
        outcome=outcome,
        shocks=shocks,
        max_horizon=max_horizon,
    )
