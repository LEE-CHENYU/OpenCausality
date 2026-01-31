"""
Fuzzy RDD estimator for pension eligibility design.

Implements the secondary identification strategy for the credit default study.

Design: Pension eligibility at age threshold creates discontinuous jump in
income from pension receipt.

First stage:  PensionInflow_i = gamma0 + gamma1(Age_i >= c) + f(Age_i - c) + X_i'delta + nu_i
Second stage: Default_i = alpha + beta(PensionInflow_hat_i) + f(Age_i - c) + X_i'delta + epsilon_i
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from shared.model.diagnostics import (
    mccrary_density_test,
    balance_test,
    first_stage_f_test,
)

logger = logging.getLogger(__name__)


@dataclass
class FuzzyRDDResult:
    """Results from fuzzy RDD estimation."""

    # Second stage (main estimate)
    coefficient: float  # Effect of pension income on default
    std_error: float
    t_stat: float
    pvalue: float
    conf_int: tuple[float, float]

    # First stage
    first_stage_f: float
    first_stage_coef: float
    first_stage_se: float
    first_stage_pvalue: float

    # Sample info
    n_obs: int
    n_above_cutoff: int
    n_below_cutoff: int
    bandwidth: float

    # Design details
    cutoff_men: int
    cutoff_women: int
    kernel: str

    # Interpretation
    interpretation: str


class FuzzyRDDEstimator:
    """
    Fuzzy RDD estimator for pension eligibility design.

    Running variable: Age relative to pension threshold
    Instrument: 1(Age >= threshold)
    Endogenous: Pension inflow observed in cashflows
    """

    def __init__(
        self,
        cutoff_men: int = 63,
        cutoff_women: int = 61,
    ):
        """
        Initialize estimator.

        Args:
            cutoff_men: Pension eligibility age for men
            cutoff_women: Pension eligibility age for women
        """
        self.cutoff_men = cutoff_men
        self.cutoff_women = cutoff_women

    def estimate(
        self,
        data: pd.DataFrame,
        outcome: str = "dpd30",
        bandwidth: float | None = None,
        polynomial_order: int = 1,
        kernel: str = "triangular",
        controls: list[str] | None = None,
        cluster_var: str = "borrower_id",
    ) -> FuzzyRDDResult:
        """
        Estimate fuzzy RDD model.

        Args:
            data: Panel data with age, gender, pension_inflow
            outcome: Outcome variable
            bandwidth: Bandwidth in years (auto if None)
            polynomial_order: Polynomial order for running variable
            kernel: Kernel for weighting (triangular/uniform)
            controls: Additional control variables
            cluster_var: Variable to cluster on

        Returns:
            FuzzyRDDResult with estimates
        """
        df = data.copy()
        controls = controls or []

        # Validate required columns
        required = ["age", "gender", "pension_inflow"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Assign cutoff based on gender
        df["cutoff"] = df["gender"].map(
            lambda g: self.cutoff_men if str(g).upper() == "M" else self.cutoff_women
        )

        # Running variable
        df["age_centered"] = df["age"] - df["cutoff"]

        # Determine bandwidth
        if bandwidth is None:
            bandwidth = self._compute_optimal_bandwidth(df)

        # Restrict to bandwidth
        df = df[df["age_centered"].abs() <= bandwidth].copy()

        if len(df) == 0:
            raise ValueError("No observations in bandwidth")

        # Instrument
        df["above_cutoff"] = (df["age_centered"] >= 0).astype(int)

        # Kernel weights
        df["weight"] = self._compute_weights(df["age_centered"], bandwidth, kernel)

        # Polynomial terms
        poly_terms = []
        for p in range(1, polynomial_order + 1):
            term = f"age_poly_{p}"
            df[term] = df["age_centered"] ** p
            poly_terms.append(term)

        # === FIRST STAGE ===
        # Regress pension inflow on instrument
        first_stage_regs = ["above_cutoff"] + poly_terms + controls
        available_regs = [r for r in first_stage_regs if r in df.columns]

        X_fs = sm.add_constant(df[available_regs])
        y_fs = df["pension_inflow"]

        # Drop missing
        mask = ~(y_fs.isna() | X_fs.isna().any(axis=1))
        X_fs_clean = X_fs[mask]
        y_fs_clean = y_fs[mask]
        weights_clean = df.loc[mask, "weight"]
        clusters = df.loc[mask, cluster_var]

        # First stage estimation (WLS with kernel weights)
        fs_model = sm.WLS(y_fs_clean, X_fs_clean, weights=weights_clean).fit(
            cov_type="cluster",
            cov_kwds={"groups": clusters},
        )

        fs_coef = fs_model.params["above_cutoff"]
        fs_se = fs_model.bse["above_cutoff"]
        fs_tstat = fs_model.tvalues["above_cutoff"]
        fs_pval = fs_model.pvalues["above_cutoff"]
        fs_f = fs_tstat ** 2

        # Check first stage strength
        if fs_f < 10:
            logger.warning(f"Weak first stage: F = {fs_f:.2f} < 10")

        # === REDUCED FORM ===
        # Regress outcome on instrument
        rf_regs = ["above_cutoff"] + poly_terms + controls
        available_rf = [r for r in rf_regs if r in df.columns]

        X_rf = sm.add_constant(df.loc[mask, available_rf])
        y_rf = df.loc[mask, outcome]

        rf_model = sm.WLS(y_rf, X_rf, weights=weights_clean).fit(
            cov_type="cluster",
            cov_kwds={"groups": clusters},
        )

        rf_coef = rf_model.params["above_cutoff"]

        # === FUZZY RDD (2SLS) ===
        # beta = reduced form / first stage
        fuzzy_coef = rf_coef / fs_coef if fs_coef != 0 else np.nan

        # Standard error via delta method
        # SE(beta) ≈ SE(rf) / fs_coef (simplified)
        fuzzy_se = rf_model.bse["above_cutoff"] / abs(fs_coef) if fs_coef != 0 else np.nan

        fuzzy_tstat = fuzzy_coef / fuzzy_se if fuzzy_se > 0 else 0
        fuzzy_pval = 2 * (1 - stats.t.cdf(abs(fuzzy_tstat), len(y_rf) - len(available_rf)))

        fuzzy_ci = (
            fuzzy_coef - 1.96 * fuzzy_se,
            fuzzy_coef + 1.96 * fuzzy_se,
        )

        # Sample counts
        n_above = (df["above_cutoff"] == 1).sum()
        n_below = (df["above_cutoff"] == 0).sum()

        # Interpretation
        interpretation = (
            f"Receiving pension income "
            f"{'increases' if fuzzy_coef > 0 else 'decreases'} {outcome} probability by "
            f"{abs(fuzzy_coef)*100:.2f} percentage points "
            f"(LATE at pension threshold)"
        )

        return FuzzyRDDResult(
            coefficient=fuzzy_coef,
            std_error=fuzzy_se,
            t_stat=fuzzy_tstat,
            pvalue=fuzzy_pval,
            conf_int=fuzzy_ci,
            first_stage_f=fs_f,
            first_stage_coef=fs_coef,
            first_stage_se=fs_se,
            first_stage_pvalue=fs_pval,
            n_obs=len(y_rf),
            n_above_cutoff=n_above,
            n_below_cutoff=n_below,
            bandwidth=bandwidth,
            cutoff_men=self.cutoff_men,
            cutoff_women=self.cutoff_women,
            kernel=kernel,
            interpretation=interpretation,
        )

    def _compute_optimal_bandwidth(self, data: pd.DataFrame) -> float:
        """Compute optimal bandwidth using IK method."""
        # Simple rule of thumb for age data
        age_centered = data["age_centered"].dropna()
        n = len(age_centered)

        # IK-like bandwidth
        h = 2.702 * age_centered.std() * (n ** (-1/5))

        # Bound to reasonable range for age (1-5 years)
        h = max(1.0, min(h, 5.0))

        logger.info(f"Computed optimal bandwidth: {h:.1f} years")
        return h

    def _compute_weights(
        self,
        running_var: pd.Series,
        bandwidth: float,
        kernel: str,
    ) -> pd.Series:
        """Compute kernel weights."""
        u = running_var / bandwidth

        if kernel == "triangular":
            weights = np.maximum(0, 1 - np.abs(u))
        elif kernel == "uniform":
            weights = (np.abs(u) <= 1).astype(float)
        elif kernel == "epanechnikov":
            weights = np.maximum(0, 0.75 * (1 - u**2))
        else:
            weights = pd.Series(1.0, index=running_var.index)

        return weights

    def run_diagnostics(
        self,
        data: pd.DataFrame,
        outcome: str = "dpd30",
        bandwidth: float | None = None,
    ) -> dict[str, Any]:
        """
        Run diagnostic tests for fuzzy RDD.

        Args:
            data: Panel data
            outcome: Outcome variable
            bandwidth: Bandwidth for tests

        Returns:
            Dictionary with diagnostic results
        """
        results = {}

        df = data.copy()

        # Assign cutoff
        df["cutoff"] = df["gender"].map(
            lambda g: self.cutoff_men if str(g).upper() == "M" else self.cutoff_women
        )
        df["age_centered"] = df["age"] - df["cutoff"]

        if bandwidth is None:
            bandwidth = self._compute_optimal_bandwidth(df)

        # McCrary density test
        logger.info("Running McCrary density test...")
        mccrary = mccrary_density_test(df["age"], df["cutoff"].mean(), bandwidth)
        results["mccrary"] = {
            "theta": mccrary.theta,
            "pvalue": mccrary.pvalue,
            "passed": not mccrary.discontinuity_detected,
        }

        # Balance tests
        logger.info("Running balance tests...")
        covariates = ["loan_amount", "pre_policy_payroll", "region"]
        available_covs = [c for c in covariates if c in df.columns]

        if available_covs:
            balance = balance_test(
                df,
                "age_centered",
                0,  # Cutoff is at 0 for centered variable
                available_covs,
                bandwidth,
            )
            results["balance"] = {
                cov: {"pvalue": bt.pvalue, "balanced": bt.balanced}
                for cov, bt in zip(available_covs, balance)
            }

        # First stage F-test
        logger.info("Running first-stage F-test...")
        if "pension_inflow" in df.columns:
            df["above_cutoff"] = (df["age_centered"] >= 0).astype(int)
            df_bw = df[df["age_centered"].abs() <= bandwidth]

            fs_result = first_stage_f_test(
                df_bw,
                endogenous="pension_inflow",
                instrument="above_cutoff",
            )
            results["first_stage"] = {
                "f_stat": fs_result.f_stat,
                "pvalue": fs_result.f_pvalue,
                "passed": fs_result.passes_threshold,
            }

        # Placebo cutoffs
        logger.info("Running placebo cutoff tests...")
        placebo_ages_men = [60, 62, 64]
        placebo_ages_women = [58, 60, 62]

        # For simplicity, test men's cutoffs
        df_men = df[df["gender"].str.upper() == "M"].copy()
        if len(df_men) > 100:
            placebo_results = {}
            for age in placebo_ages_men:
                if age != self.cutoff_men:
                    df_men["placebo_centered"] = df_men["age"] - age
                    df_men["placebo_above"] = (df_men["placebo_centered"] >= 0).astype(int)

                    try:
                        X = sm.add_constant(df_men[["placebo_above"]])
                        y = df_men[outcome]
                        mask = ~(y.isna() | X.isna().any(axis=1))
                        model = sm.OLS(y[mask], X[mask]).fit()
                        placebo_results[age] = model.pvalues["placebo_above"]
                    except Exception:
                        placebo_results[age] = np.nan

            results["placebo_cutoffs_men"] = placebo_results

        return results

    def summary(self, result: FuzzyRDDResult) -> str:
        """Generate summary of results."""
        lines = []
        lines.append("=" * 60)
        lines.append("FUZZY RDD RESULTS (PENSION ELIGIBILITY)")
        lines.append("=" * 60)

        lines.append(f"\nCutoffs:")
        lines.append(f"  Men: {result.cutoff_men} years")
        lines.append(f"  Women: {result.cutoff_women} years")
        lines.append(f"Bandwidth: {result.bandwidth:.1f} years")
        lines.append(f"Kernel: {result.kernel}")

        lines.append(f"\nSample:")
        lines.append(f"  N observations: {result.n_obs:,}")
        lines.append(f"  N above cutoff: {result.n_above_cutoff:,}")
        lines.append(f"  N below cutoff: {result.n_below_cutoff:,}")

        lines.append(f"\nFirst Stage (Instrument -> Pension Inflow):")
        lines.append(f"  F-statistic: {result.first_stage_f:.2f}")
        lines.append(f"  Coefficient: {result.first_stage_coef:.4f}")
        lines.append(f"  p-value: {result.first_stage_pvalue:.4f}")
        status = "PASS (F > 10)" if result.first_stage_f > 10 else "FAIL (F < 10)"
        lines.append(f"  Status: {status}")

        lines.append(f"\nSecond Stage (Fuzzy RDD):")
        lines.append(f"  Coefficient: {result.coefficient:.4f}")
        lines.append(f"  Std. Error: {result.std_error:.4f}")
        lines.append(f"  t-statistic: {result.t_stat:.2f}")
        lines.append(f"  p-value: {result.pvalue:.4f}")
        lines.append(f"  95% CI: [{result.conf_int[0]:.4f}, {result.conf_int[1]:.4f}]")

        lines.append(f"\nInterpretation:")
        lines.append(f"  {result.interpretation}")

        return "\n".join(lines)


def estimate_pension_effect(
    data: pd.DataFrame,
    outcome: str = "dpd30",
    bandwidth: float | None = None,
) -> FuzzyRDDResult:
    """
    Convenience function to estimate pension effect on default.

    Args:
        data: Panel data
        outcome: Outcome variable
        bandwidth: Bandwidth in years

    Returns:
        FuzzyRDDResult
    """
    estimator = FuzzyRDDEstimator()
    return estimator.estimate(data, outcome=outcome, bandwidth=bandwidth)
