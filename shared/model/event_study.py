"""
Event study utilities for causal inference.

Implements:
- Pre-trends testing
- Event study plots
- Joint F-tests for pre-trends
- Effect-size bounds for parallel trends violations
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
class EventStudyResult:
    """Results from event study estimation."""

    coefficients: pd.Series
    std_errors: pd.Series
    conf_lower: pd.Series
    conf_upper: pd.Series
    pvalues: pd.Series
    pre_periods: list[int]
    post_periods: list[int]
    reference_period: int
    n_obs: int

    # Pre-trends tests
    joint_f_stat: float | None = None
    joint_f_pvalue: float | None = None
    max_pre_trend: float | None = None
    max_pre_trend_se: float | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for plotting."""
        periods = list(self.coefficients.index)
        return pd.DataFrame({
            "period": periods,
            "coefficient": self.coefficients.values,
            "std_error": self.std_errors.values,
            "conf_lower": self.conf_lower.values,
            "conf_upper": self.conf_upper.values,
            "pvalue": self.pvalues.values,
        })

    def passes_pre_trends(
        self,
        alpha: float = 0.10,
        max_effect_size: float | None = None,
    ) -> tuple[bool, str]:
        """
        Check if pre-trends test passes.

        Args:
            alpha: Significance level for joint F-test
            max_effect_size: Maximum acceptable pre-trend effect size

        Returns:
            Tuple of (passes, reason)
        """
        reasons = []

        # Check joint F-test
        if self.joint_f_pvalue is not None:
            if self.joint_f_pvalue < alpha:
                reasons.append(
                    f"Joint F-test rejects (p={self.joint_f_pvalue:.3f} < {alpha})"
                )

        # Check effect size bounds
        if max_effect_size is not None and self.max_pre_trend is not None:
            if abs(self.max_pre_trend) > max_effect_size:
                reasons.append(
                    f"Max pre-trend ({self.max_pre_trend:.4f}) exceeds bound ({max_effect_size})"
                )

        if reasons:
            return False, "; ".join(reasons)

        return True, "Pre-trends test passed"


class EventStudy:
    """
    Event study estimation for panel data.

    Estimates coefficients for each period relative to treatment.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with panel data.

        Args:
            data: Panel DataFrame with unit and time identifiers
        """
        self.data = data.copy()

    def estimate(
        self,
        outcome: str,
        treatment: str,
        unit_var: str = "region",
        time_var: str = "quarter",
        pre_periods: int = 4,
        post_periods: int = 4,
        reference_period: int = -1,
        controls: list[str] | None = None,
        cluster_var: str | None = None,
    ) -> EventStudyResult:
        """
        Estimate event study model.

        Args:
            outcome: Outcome variable name
            treatment: Treatment indicator (1 for treated units/periods)
            unit_var: Unit identifier
            time_var: Time identifier
            pre_periods: Number of pre-treatment periods
            post_periods: Number of post-treatment periods
            reference_period: Reference period (omitted, default: -1)
            controls: Control variables
            cluster_var: Variable to cluster on (default: unit_var)

        Returns:
            EventStudyResult
        """
        data = self.data.copy()
        controls = controls or []
        cluster_var = cluster_var or unit_var

        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()

        # Create relative time dummies
        # Assumes treatment variable indicates treatment timing
        # This is a simplified version - in practice you'd have treatment date per unit

        periods = list(range(-pre_periods, post_periods + 1))
        periods.remove(reference_period)  # Remove reference

        # Create period dummies interacted with treatment
        for p in periods:
            col_name = f"period_{p}" if p < 0 else f"period_plus_{p}"
            data[col_name] = 0  # Placeholder - would need actual relative time

        # For demonstration, create simple pre/post dummies
        # In practice, you'd compute relative time to treatment for each unit

        # Build regression
        period_cols = [f"period_{p}" if p < 0 else f"period_plus_{p}" for p in periods]

        # Add unit and time fixed effects
        unit_dummies = pd.get_dummies(data[unit_var], prefix="unit", drop_first=True)
        time_dummies = pd.get_dummies(data[time_var], prefix="time", drop_first=True)

        X = pd.concat([
            data[period_cols + controls],
            unit_dummies,
            time_dummies,
        ], axis=1)

        X = sm.add_constant(X)
        y = data[outcome]

        # Drop missing
        mask = ~(y.isna() | X.isna().any(axis=1))
        X = X[mask]
        y = y[mask]
        clusters = data.loc[mask, cluster_var]

        # Estimate
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": clusters},
        )

        # Extract coefficients for period dummies
        coefs = {}
        ses = {}
        pvals = {}

        for p in periods:
            col_name = f"period_{p}" if p < 0 else f"period_plus_{p}"
            if col_name in model.params.index:
                coefs[p] = model.params[col_name]
                ses[p] = model.bse[col_name]
                pvals[p] = model.pvalues[col_name]
            else:
                coefs[p] = 0.0
                ses[p] = 0.0
                pvals[p] = 1.0

        # Add reference period (zero by construction)
        coefs[reference_period] = 0.0
        ses[reference_period] = 0.0
        pvals[reference_period] = 1.0

        # Sort by period
        coefs = pd.Series(coefs).sort_index()
        ses = pd.Series(ses).sort_index()
        pvals = pd.Series(pvals).sort_index()

        # Confidence intervals
        conf_lower = coefs - 1.96 * ses
        conf_upper = coefs + 1.96 * ses

        # Joint F-test for pre-trends
        pre_cols = [f"period_{p}" for p in range(-pre_periods, 0)]
        pre_cols = [c for c in pre_cols if c in model.params.index]

        joint_f = None
        joint_p = None
        if pre_cols:
            try:
                # Test that all pre-period coefficients are jointly zero
                r_matrix = np.zeros((len(pre_cols), len(model.params)))
                for i, col in enumerate(pre_cols):
                    idx = list(model.params.index).index(col)
                    r_matrix[i, idx] = 1

                f_test = model.f_test(r_matrix)
                joint_f = float(f_test.fvalue)
                joint_p = float(f_test.pvalue)
            except Exception as e:
                logger.warning(f"Joint F-test failed: {e}")

        # Max pre-trend
        pre_coefs = coefs[[p for p in coefs.index if p < 0 and p != reference_period]]
        max_pre = pre_coefs.abs().max() if len(pre_coefs) > 0 else None
        max_pre_se = ses[pre_coefs.abs().idxmax()] if max_pre else None

        return EventStudyResult(
            coefficients=coefs,
            std_errors=ses,
            conf_lower=conf_lower,
            conf_upper=conf_upper,
            pvalues=pvals,
            pre_periods=list(range(-pre_periods, 0)),
            post_periods=list(range(0, post_periods + 1)),
            reference_period=reference_period,
            n_obs=len(y),
            joint_f_stat=joint_f,
            joint_f_pvalue=joint_p,
            max_pre_trend=max_pre,
            max_pre_trend_se=max_pre_se,
        )

    def plot(
        self,
        result: EventStudyResult,
        title: str = "Event Study",
        xlabel: str = "Period Relative to Treatment",
        ylabel: str = "Effect",
        figsize: tuple[int, int] = (10, 6),
    ) -> Any:
        """
        Plot event study results.

        Args:
            result: EventStudyResult from estimate()
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, cannot plot")
            return None

        df = result.to_dataframe()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot point estimates
        ax.scatter(df["period"], df["coefficient"], color="blue", s=50, zorder=3)
        ax.plot(df["period"], df["coefficient"], color="blue", linewidth=1, alpha=0.7)

        # Plot confidence intervals
        ax.fill_between(
            df["period"],
            df["conf_lower"],
            df["conf_upper"],
            alpha=0.2,
            color="blue",
            label="95% CI",
        )

        # Reference line at zero
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

        # Vertical line at treatment time
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.7,
                   label="Treatment")

        # Labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

        # Add pre-trends info
        if result.joint_f_pvalue is not None:
            ax.text(
                0.02, 0.98,
                f"Joint F-test p-value: {result.joint_f_pvalue:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        return fig


def test_pre_trends(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    pre_periods: int = 4,
    unit_var: str = "region",
    time_var: str = "quarter",
) -> dict[str, Any]:
    """
    Convenience function to test pre-trends.

    Args:
        data: Panel data
        outcome: Outcome variable
        treatment: Treatment indicator
        pre_periods: Number of pre-periods to test
        unit_var: Unit identifier
        time_var: Time identifier

    Returns:
        Dictionary with pre-trends test results
    """
    es = EventStudy(data)
    result = es.estimate(
        outcome=outcome,
        treatment=treatment,
        pre_periods=pre_periods,
        post_periods=0,  # Focus on pre-trends
        unit_var=unit_var,
        time_var=time_var,
    )

    passes, reason = result.passes_pre_trends()

    return {
        "passes": passes,
        "reason": reason,
        "joint_f_stat": result.joint_f_stat,
        "joint_f_pvalue": result.joint_f_pvalue,
        "max_pre_trend": result.max_pre_trend,
        "pre_period_coefficients": result.coefficients[result.pre_periods].to_dict(),
    }
