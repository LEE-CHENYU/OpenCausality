"""
Diagnostic tests for econometric models.

Implements:
- First-stage F-test for IV strength
- McCrary density test for manipulation
- Balance tests at RDD cutoff
- Placebo cutoff tests
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass
class FirstStageResult:
    """Results from first-stage regression."""

    f_stat: float
    f_pvalue: float
    coefficient: float
    std_error: float
    t_stat: float
    n_obs: int
    passes_threshold: bool  # F > 10
    instrument: str


@dataclass
class McCraryResult:
    """Results from McCrary density test."""

    theta: float  # Log difference in density
    se: float
    z_stat: float
    pvalue: float
    discontinuity_detected: bool
    cutoff: float
    bandwidth: float | None
    n_left: int
    n_right: int


@dataclass
class BalanceTestResult:
    """Results from balance test at RDD cutoff."""

    variable: str
    diff_at_cutoff: float
    se: float
    t_stat: float
    pvalue: float
    balanced: bool  # p > 0.05
    n_obs: int


@dataclass
class DiagnosticSuite:
    """Complete suite of diagnostic test results."""

    first_stage: FirstStageResult | None = None
    mccrary: McCraryResult | None = None
    balance_tests: list[BalanceTestResult] = field(default_factory=list)
    placebo_tests: dict[float, float] = field(default_factory=dict)  # cutoff -> pvalue

    def all_pass(self) -> bool:
        """Check if all diagnostics pass."""
        if self.first_stage and not self.first_stage.passes_threshold:
            return False
        if self.mccrary and self.mccrary.discontinuity_detected:
            return False
        if any(not bt.balanced for bt in self.balance_tests):
            return False
        return True

    def summary(self) -> str:
        """Generate summary of all diagnostics."""
        lines = []
        lines.append("=" * 60)
        lines.append("DIAGNOSTIC TEST SUMMARY")
        lines.append("=" * 60)

        # First stage
        if self.first_stage:
            status = "PASS" if self.first_stage.passes_threshold else "FAIL"
            lines.append(f"\nFirst-Stage F-Test: {status}")
            lines.append(f"  F-statistic: {self.first_stage.f_stat:.2f}")
            lines.append(f"  p-value: {self.first_stage.f_pvalue:.4f}")
            lines.append(f"  Threshold: F > 10")

        # McCrary
        if self.mccrary:
            status = "FAIL" if self.mccrary.discontinuity_detected else "PASS"
            lines.append(f"\nMcCrary Density Test: {status}")
            lines.append(f"  Log density difference: {self.mccrary.theta:.4f}")
            lines.append(f"  z-statistic: {self.mccrary.z_stat:.2f}")
            lines.append(f"  p-value: {self.mccrary.pvalue:.4f}")

        # Balance tests
        if self.balance_tests:
            n_pass = sum(1 for bt in self.balance_tests if bt.balanced)
            n_total = len(self.balance_tests)
            lines.append(f"\nBalance Tests: {n_pass}/{n_total} pass")
            for bt in self.balance_tests:
                status = "PASS" if bt.balanced else "FAIL"
                lines.append(f"  {bt.variable}: diff={bt.diff_at_cutoff:.4f}, "
                           f"p={bt.pvalue:.3f} [{status}]")

        # Placebo tests
        if self.placebo_tests:
            lines.append(f"\nPlacebo Cutoff Tests:")
            for cutoff, pval in sorted(self.placebo_tests.items()):
                sig = "*" if pval < 0.05 else ""
                lines.append(f"  Cutoff {cutoff}: p={pval:.3f}{sig}")

        lines.append("\n" + "=" * 60)
        overall = "ALL DIAGNOSTICS PASS" if self.all_pass() else "SOME DIAGNOSTICS FAIL"
        lines.append(f"Overall: {overall}")

        return "\n".join(lines)


def first_stage_f_test(
    data: pd.DataFrame,
    endogenous: str,
    instrument: str,
    controls: list[str] | None = None,
    cluster_var: str | None = None,
) -> FirstStageResult:
    """
    Test first-stage strength for IV estimation.

    Args:
        data: DataFrame with variables
        endogenous: Endogenous variable name
        instrument: Instrument variable name
        controls: Control variables
        cluster_var: Variable to cluster on

    Returns:
        FirstStageResult with F-statistic and assessment
    """
    data = data.copy()
    controls = controls or []

    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()

    # Build regression
    X_vars = [instrument] + controls
    X = sm.add_constant(data[X_vars])
    y = data[endogenous]

    # Drop missing
    mask = ~(y.isna() | X.isna().any(axis=1))
    X = X[mask]
    y = y[mask]

    # Estimate
    if cluster_var:
        clusters = data.loc[mask, cluster_var]
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": clusters},
        )
    else:
        model = sm.OLS(y, X).fit()

    # Extract instrument coefficient
    coef = model.params[instrument]
    se = model.bse[instrument]
    t_stat = model.tvalues[instrument]

    # F-test for instrument
    # For single instrument, F = t^2
    f_stat = t_stat ** 2
    f_pvalue = model.pvalues[instrument]

    return FirstStageResult(
        f_stat=f_stat,
        f_pvalue=f_pvalue,
        coefficient=coef,
        std_error=se,
        t_stat=t_stat,
        n_obs=len(y),
        passes_threshold=f_stat > 10,
        instrument=instrument,
    )


def mccrary_density_test(
    running_var: pd.Series,
    cutoff: float,
    bandwidth: float | None = None,
    n_bins: int = 50,
) -> McCraryResult:
    """
    McCrary (2008) density discontinuity test.

    Tests for manipulation of the running variable at the cutoff.

    Args:
        running_var: Running variable values
        cutoff: RDD cutoff value
        bandwidth: Bandwidth for local linear regression (auto if None)
        n_bins: Number of bins for histogram

    Returns:
        McCraryResult with test statistics
    """
    running_var = running_var.dropna()

    # Center at cutoff
    centered = running_var - cutoff

    # Compute bandwidth if not provided (rule of thumb)
    if bandwidth is None:
        bandwidth = 1.06 * centered.std() * (len(centered) ** (-1/5))

    # Restrict to bandwidth
    left = centered[(centered >= -bandwidth) & (centered < 0)]
    right = centered[(centered >= 0) & (centered <= bandwidth)]

    n_left = len(left)
    n_right = len(right)

    if n_left < 10 or n_right < 10:
        logger.warning("Too few observations near cutoff for McCrary test")
        return McCraryResult(
            theta=np.nan,
            se=np.nan,
            z_stat=np.nan,
            pvalue=np.nan,
            discontinuity_detected=False,
            cutoff=cutoff,
            bandwidth=bandwidth,
            n_left=n_left,
            n_right=n_right,
        )

    # Estimate densities using histogram approach
    # This is a simplified version - full McCrary uses local linear regression

    # Create bins
    bin_width = 2 * bandwidth / n_bins
    bins_left = np.arange(-bandwidth, 0, bin_width)
    bins_right = np.arange(0, bandwidth + bin_width, bin_width)

    # Count in bins
    counts_left, _ = np.histogram(left, bins=bins_left)
    counts_right, _ = np.histogram(right, bins=bins_right)

    # Estimate density at cutoff (from each side)
    # Use counts in bins closest to cutoff
    if len(counts_left) > 0 and len(counts_right) > 0:
        density_left = counts_left[-1] / bin_width if len(counts_left) > 0 else 0
        density_right = counts_right[0] / bin_width if len(counts_right) > 0 else 0

        # Log difference in density
        if density_left > 0 and density_right > 0:
            theta = np.log(density_right) - np.log(density_left)
        else:
            theta = 0

        # Approximate SE using delta method
        # SE(log(n)) ≈ 1/sqrt(n)
        se = np.sqrt(1/max(density_left, 1) + 1/max(density_right, 1))

        z_stat = theta / se if se > 0 else 0
        pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        theta = 0
        se = np.nan
        z_stat = 0
        pvalue = 1.0

    return McCraryResult(
        theta=theta,
        se=se,
        z_stat=z_stat,
        pvalue=pvalue,
        discontinuity_detected=pvalue < 0.05,
        cutoff=cutoff,
        bandwidth=bandwidth,
        n_left=n_left,
        n_right=n_right,
    )


def balance_test(
    data: pd.DataFrame,
    running_var: str,
    cutoff: float,
    covariates: list[str],
    bandwidth: float | None = None,
) -> list[BalanceTestResult]:
    """
    Test for covariate balance at RDD cutoff.

    Args:
        data: DataFrame with variables
        running_var: Running variable name
        cutoff: RDD cutoff
        covariates: List of covariates to test
        bandwidth: Bandwidth around cutoff

    Returns:
        List of BalanceTestResult for each covariate
    """
    data = data.copy()

    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()

    # Center running variable
    data["centered"] = data[running_var] - cutoff

    # Determine bandwidth
    if bandwidth is None:
        bandwidth = data["centered"].std()

    # Restrict to bandwidth
    data = data[(data["centered"].abs() <= bandwidth)]

    # Treatment indicator
    data["treated"] = (data["centered"] >= 0).astype(int)

    results = []
    for cov in covariates:
        if cov not in data.columns:
            continue

        # Simple t-test for difference at cutoff
        treated = data[data["treated"] == 1][cov].dropna()
        control = data[data["treated"] == 0][cov].dropna()

        if len(treated) < 5 or len(control) < 5:
            continue

        # Welch's t-test
        t_stat, pvalue = stats.ttest_ind(treated, control, equal_var=False)
        diff = treated.mean() - control.mean()
        se = np.sqrt(treated.var()/len(treated) + control.var()/len(control))

        results.append(BalanceTestResult(
            variable=cov,
            diff_at_cutoff=diff,
            se=se,
            t_stat=t_stat,
            pvalue=pvalue,
            balanced=pvalue > 0.05,
            n_obs=len(treated) + len(control),
        ))

    return results


def placebo_cutoff_test(
    data: pd.DataFrame,
    outcome: str,
    running_var: str,
    true_cutoff: float,
    placebo_cutoffs: list[float],
    bandwidth: float | None = None,
) -> dict[float, float]:
    """
    Test for effects at placebo cutoffs.

    Args:
        data: DataFrame with variables
        outcome: Outcome variable
        running_var: Running variable
        true_cutoff: True RDD cutoff
        placebo_cutoffs: List of placebo cutoffs to test
        bandwidth: Bandwidth around cutoff

    Returns:
        Dictionary mapping placebo cutoff to p-value
    """
    data = data.copy()

    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()

    results = {}

    for cutoff in placebo_cutoffs:
        # Center at placebo cutoff
        data["centered"] = data[running_var] - cutoff

        # Determine bandwidth
        bw = bandwidth or data["centered"].std()

        # Restrict to bandwidth (excluding true cutoff region)
        subset = data[
            (data["centered"].abs() <= bw) &
            (data[running_var] != true_cutoff)
        ]

        if len(subset) < 20:
            continue

        # Treatment indicator
        subset["treated"] = (subset["centered"] >= 0).astype(int)

        # Simple regression
        X = sm.add_constant(subset["treated"])
        y = subset[outcome]

        mask = ~(y.isna() | X.isna().any(axis=1))
        X = X[mask]
        y = y[mask]

        if len(y) < 10:
            continue

        model = sm.OLS(y, X).fit()
        pvalue = model.pvalues["treated"]

        results[cutoff] = pvalue

    return results


def run_rdd_diagnostics(
    data: pd.DataFrame,
    outcome: str,
    running_var: str,
    cutoff: float,
    covariates: list[str] | None = None,
    placebo_cutoffs: list[float] | None = None,
    bandwidth: float | None = None,
) -> DiagnosticSuite:
    """
    Run complete RDD diagnostic suite.

    Args:
        data: DataFrame with variables
        outcome: Outcome variable
        running_var: Running variable
        cutoff: RDD cutoff
        covariates: Covariates for balance test
        placebo_cutoffs: Placebo cutoffs to test
        bandwidth: Bandwidth around cutoff

    Returns:
        DiagnosticSuite with all test results
    """
    suite = DiagnosticSuite()

    # McCrary test
    logger.info("Running McCrary density test...")
    suite.mccrary = mccrary_density_test(
        data[running_var], cutoff, bandwidth
    )

    # Balance tests
    if covariates:
        logger.info("Running balance tests...")
        suite.balance_tests = balance_test(
            data, running_var, cutoff, covariates, bandwidth
        )

    # Placebo cutoff tests
    if placebo_cutoffs:
        logger.info("Running placebo cutoff tests...")
        suite.placebo_tests = placebo_cutoff_test(
            data, outcome, running_var, cutoff, placebo_cutoffs, bandwidth
        )

    return suite
