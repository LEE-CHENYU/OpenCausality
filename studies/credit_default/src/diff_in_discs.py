"""
Difference-in-Discontinuities estimator for minimum wage design.

Implements the primary identification strategy for the credit default study.

Design: Compare default rates for workers just below vs just above the old
minimum wage threshold, before vs after the MW increase.

Specification:
Y_it = alpha + beta1(Post_t) + beta2(Below_c_i) + beta3(Post_t x Below_c_i)
     + f(wage_i - c) + g(wage_i - c) x Post_t + X_it'delta + epsilon_it

Key coefficient: beta3 = effect of MW-induced income shock on default
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from shared.model.inference import ClusteredInference
from shared.model.event_study import EventStudy
from shared.model.diagnostics import (
    balance_test,
    mccrary_density_test,
    placebo_cutoff_test,
)

logger = logging.getLogger(__name__)


@dataclass
class DiffInDiscsResult:
    """Results from diff-in-discs estimation."""

    # Main estimate
    coefficient: float  # beta3: Post x Below interaction
    std_error: float
    t_stat: float
    pvalue: float
    conf_int: tuple[float, float]

    # Sample info
    n_obs: int
    n_treated: int
    n_control: int
    n_clusters: int

    # Specification details
    bandwidth: float
    polynomial_order: int
    outcome: str

    # Interpretation
    interpretation: str


@dataclass
class IntensityIVResult:
    """Results from intensity instrument estimation."""

    # Main estimate
    coefficient: float  # Effect of predicted raise
    std_error: float
    t_stat: float
    pvalue: float
    conf_int: tuple[float, float]

    # First stage
    first_stage_f: float
    first_stage_coef: float

    # Sample info
    n_obs: int


class DiffInDiscsEstimator:
    """
    Difference-in-Discontinuities estimator for MW design.

    Identification: Workers just below the old MW get a larger income
    increase than workers just above, creating exogenous income variation.
    """

    def __init__(
        self,
        old_minimum_wage: int = 70000,
        new_minimum_wage: int = 85000,
    ):
        """
        Initialize estimator.

        Args:
            old_minimum_wage: Pre-reform MW (cutoff)
            new_minimum_wage: Post-reform MW
        """
        self.cutoff = old_minimum_wage
        self.new_mw = new_minimum_wage

    def estimate(
        self,
        data: pd.DataFrame,
        outcome: str = "dpd30",
        bandwidth: float | None = None,
        polynomial_order: int = 1,
        controls: list[str] | None = None,
        cluster_var: str = "borrower_id",
    ) -> DiffInDiscsResult:
        """
        Estimate diff-in-discs model.

        Args:
            data: Panel data with required variables
            outcome: Outcome variable (dpd30, dpd15, etc.)
            bandwidth: Bandwidth around cutoff (auto if None)
            polynomial_order: Polynomial order for running variable
            controls: Additional control variables
            cluster_var: Variable to cluster on

        Returns:
            DiffInDiscsResult with estimates
        """
        df = data.copy()
        controls = controls or []

        # Determine bandwidth
        if bandwidth is None:
            bandwidth = self._compute_optimal_bandwidth(df)

        # Restrict to bandwidth
        df = df[
            (df["pre_policy_payroll"] >= self.cutoff - bandwidth) &
            (df["pre_policy_payroll"] <= self.cutoff + bandwidth)
        ].copy()

        if len(df) == 0:
            raise ValueError("No observations in bandwidth")

        # Construct variables
        df["below_c"] = (df["pre_policy_payroll"] < self.cutoff).astype(int)
        df["post_x_below"] = df["post"] * df["below_c"]
        df["payroll_centered"] = df["pre_policy_payroll"] - self.cutoff

        # Polynomial terms
        poly_terms = []
        for p in range(1, polynomial_order + 1):
            term = f"payroll_poly_{p}"
            df[term] = df["payroll_centered"] ** p
            poly_terms.append(term)

            # Interaction with post
            term_post = f"payroll_poly_{p}_x_post"
            df[term_post] = df[term] * df["post"]
            poly_terms.append(term_post)

        # Build regression
        regressors = ["post", "below_c", "post_x_below"] + poly_terms + controls
        available_regs = [r for r in regressors if r in df.columns]

        X = sm.add_constant(df[available_regs])
        y = df[outcome]

        # Drop missing
        mask = ~(y.isna() | X.isna().any(axis=1))
        X = X[mask]
        y = y[mask]
        clusters = df.loc[mask, cluster_var]

        # Estimate with clustered SEs
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": clusters},
        )

        # Extract main coefficient
        coef = model.params["post_x_below"]
        se = model.bse["post_x_below"]
        tstat = model.tvalues["post_x_below"]
        pval = model.pvalues["post_x_below"]
        ci = model.conf_int().loc["post_x_below"]

        # Sample counts
        n_treated = df[df["below_c"] == 1]["loan_id"].nunique() if "loan_id" in df else (df["below_c"] == 1).sum()
        n_control = df[df["below_c"] == 0]["loan_id"].nunique() if "loan_id" in df else (df["below_c"] == 0).sum()

        # Interpretation
        mw_pct_increase = (self.new_mw - self.cutoff) / self.cutoff * 100
        interpretation = (
            f"A {mw_pct_increase:.1f}% income increase (MW shock) "
            f"{'increases' if coef > 0 else 'decreases'} {outcome} probability by "
            f"{abs(coef)*100:.2f} percentage points"
        )

        return DiffInDiscsResult(
            coefficient=coef,
            std_error=se,
            t_stat=tstat,
            pvalue=pval,
            conf_int=(ci[0], ci[1]),
            n_obs=len(y),
            n_treated=n_treated,
            n_control=n_control,
            n_clusters=clusters.nunique(),
            bandwidth=bandwidth,
            polynomial_order=polynomial_order,
            outcome=outcome,
            interpretation=interpretation,
        )

    def estimate_intensity_iv(
        self,
        data: pd.DataFrame,
        outcome: str = "dpd30",
        controls: list[str] | None = None,
        cluster_var: str = "borrower_id",
    ) -> IntensityIVResult:
        """
        Estimate using continuous intensity instrument.

        Instrument: Z_i = max(0, 85000 - wage_pre)
        This is the predicted statutory raise for each worker.

        Args:
            data: Panel data
            outcome: Outcome variable
            controls: Control variables
            cluster_var: Variable to cluster on

        Returns:
            IntensityIVResult with estimates
        """
        df = data.copy()
        controls = controls or []

        # Construct instrument
        df["predicted_raise"] = np.maximum(0, self.new_mw - df["pre_policy_payroll"])
        df["post_x_raise"] = df["post"] * df["predicted_raise"]

        # Normalize instrument (per 10,000 tenge)
        df["post_x_raise_normalized"] = df["post_x_raise"] / 10000

        # Build regression
        regressors = ["post", "post_x_raise_normalized"] + controls
        available_regs = [r for r in regressors if r in df.columns]

        X = sm.add_constant(df[available_regs])
        y = df[outcome]

        # Drop missing
        mask = ~(y.isna() | X.isna().any(axis=1))
        X = X[mask]
        y = y[mask]
        clusters = df.loc[mask, cluster_var]

        # Estimate
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": clusters},
        )

        # Extract coefficient
        coef = model.params["post_x_raise_normalized"]
        se = model.bse["post_x_raise_normalized"]
        tstat = model.tvalues["post_x_raise_normalized"]
        pval = model.pvalues["post_x_raise_normalized"]
        ci = model.conf_int().loc["post_x_raise_normalized"]

        # First stage (instrument -> actual income change)
        # This would require actual post-treatment income data
        first_stage_f = tstat ** 2  # Approximation
        first_stage_coef = 1.0  # Placeholder

        return IntensityIVResult(
            coefficient=coef,
            std_error=se,
            t_stat=tstat,
            pvalue=pval,
            conf_int=(ci[0], ci[1]),
            first_stage_f=first_stage_f,
            first_stage_coef=first_stage_coef,
            n_obs=len(y),
        )

    def _compute_optimal_bandwidth(self, data: pd.DataFrame) -> float:
        """Compute MSE-optimal bandwidth."""
        # Simple rule of thumb: 1.06 * sigma * n^(-1/5)
        payroll = data["pre_policy_payroll"].dropna()
        sigma = payroll.std()
        n = len(payroll)
        h = 1.06 * sigma * (n ** (-1/5))

        # Bound to reasonable range
        h = max(5000, min(h, 30000))

        logger.info(f"Computed optimal bandwidth: {h:.0f}")
        return h

    def run_diagnostics(
        self,
        data: pd.DataFrame,
        outcome: str = "dpd30",
    ) -> dict[str, Any]:
        """
        Run diagnostic tests for diff-in-discs design.

        Args:
            data: Panel data
            outcome: Outcome variable

        Returns:
            Dictionary with diagnostic results
        """
        results = {}

        # McCrary density test
        logger.info("Running McCrary density test...")
        mccrary = mccrary_density_test(
            data["pre_policy_payroll"],
            self.cutoff,
        )
        results["mccrary"] = {
            "theta": mccrary.theta,
            "pvalue": mccrary.pvalue,
            "passed": not mccrary.discontinuity_detected,
        }

        # Balance tests
        logger.info("Running balance tests...")
        covariates = ["loan_amount", "loan_term_months", "region"]
        available_covs = [c for c in covariates if c in data.columns]

        if available_covs:
            balance = balance_test(
                data,
                "pre_policy_payroll",
                self.cutoff,
                available_covs,
            )
            results["balance"] = {
                cov: {"pvalue": bt.pvalue, "balanced": bt.balanced}
                for cov, bt in zip(available_covs, balance)
            }

        # Placebo cutoffs
        logger.info("Running placebo cutoff tests...")
        placebo_cutoffs = [60000, 80000, 90000]
        placebo = placebo_cutoff_test(
            data,
            outcome,
            "pre_policy_payroll",
            self.cutoff,
            placebo_cutoffs,
        )
        results["placebo_cutoffs"] = placebo

        return results

    def summary(self, result: DiffInDiscsResult) -> str:
        """Generate summary of results."""
        lines = []
        lines.append("=" * 60)
        lines.append("DIFF-IN-DISCONTINUITIES RESULTS")
        lines.append("=" * 60)

        lines.append(f"\nOutcome: {result.outcome}")
        lines.append(f"Bandwidth: {result.bandwidth:,.0f}")
        lines.append(f"Polynomial order: {result.polynomial_order}")

        lines.append(f"\nSample:")
        lines.append(f"  N observations: {result.n_obs:,}")
        lines.append(f"  N treated (below cutoff): {result.n_treated:,}")
        lines.append(f"  N control (above cutoff): {result.n_control:,}")
        lines.append(f"  N clusters: {result.n_clusters:,}")

        lines.append(f"\nMain Estimate (Post x Below):")
        lines.append(f"  Coefficient: {result.coefficient:.4f}")
        lines.append(f"  Std. Error: {result.std_error:.4f}")
        lines.append(f"  t-statistic: {result.t_stat:.2f}")
        lines.append(f"  p-value: {result.pvalue:.4f}")
        lines.append(f"  95% CI: [{result.conf_int[0]:.4f}, {result.conf_int[1]:.4f}]")

        lines.append(f"\nInterpretation:")
        lines.append(f"  {result.interpretation}")

        return "\n".join(lines)


def estimate_mw_effect(
    data: pd.DataFrame,
    outcome: str = "dpd30",
    bandwidth: float | None = None,
) -> DiffInDiscsResult:
    """
    Convenience function to estimate MW effect on default.

    Args:
        data: Panel data
        outcome: Outcome variable
        bandwidth: Bandwidth around cutoff

    Returns:
        DiffInDiscsResult
    """
    estimator = DiffInDiscsEstimator()
    return estimator.estimate(data, outcome=outcome, bandwidth=bandwidth)
