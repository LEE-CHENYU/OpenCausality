"""
Inference methods: Driscoll-Kraay, BHJ shock-level regression, and clustered SE.

V1: Driscoll-Kraay HAC (via linearmodels)
V2: BHJ shock-level regression for shift-share designs
V3: Clustered standard errors with wild bootstrap
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass
class BHJResult:
    """Results from BHJ shock-level regression."""

    coefficient: float
    std_error: float
    t_stat: float
    pvalue: float
    conf_int: tuple[float, float]
    n_periods: int
    bandwidth: int
    method: str = "BHJ"


@dataclass
class ClusteredSEResult:
    """Results from clustered standard error estimation."""

    coefficient: float
    std_error: float
    t_stat: float
    pvalue: float
    conf_int: tuple[float, float]
    n_obs: int
    n_clusters: int
    cluster_var: str
    method: str = "clustered"


class BHJInference:
    """
    Borusyak-Hull-Jaravel shock-level regression.

    For shift-share designs where identification comes from shocks:
    1. Residualize outcomes on region FE + controls
    2. Compute cross-sectional projection each quarter onto exposure(s)
    3. Regress that time series on shock series using HAC/Newey-West

    This directly matches "identification comes from shocks" logic.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with panel data.

        Args:
            data: Panel DataFrame with region-quarter observations
        """
        self.data = data.copy()

    def estimate(
        self,
        outcome: str,
        exposure: str,
        shock: str,
        controls: list[str] | None = None,
        bandwidth: int | None = None,
    ) -> BHJResult:
        """
        Estimate coefficient using BHJ shock-level regression.

        Args:
            outcome: Outcome variable name
            exposure: Exposure variable name
            shock: Shock variable name
            controls: Optional control variables
            bandwidth: HAC bandwidth (default: Newey-West automatic)

        Returns:
            BHJResult with estimate and inference
        """
        data = self.data.copy()
        controls = controls or []

        # Reset index if needed
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()

        # Step 1: Residualize outcome on region FE + controls
        y_resid = self._residualize_on_fe(data, outcome, controls)

        # Step 2: Compute cross-sectional projection each quarter
        beta_t = self._cross_sectional_projections(data, y_resid, exposure)

        # Step 3: Match with shock series
        shock_data = self._get_shock_series(data, shock)

        # Merge beta_t with shocks
        merged = pd.merge(beta_t, shock_data, on="quarter", how="inner")

        if len(merged) < 10:
            raise ValueError(f"Too few periods for BHJ: {len(merged)}")

        # Step 4: Regress beta_t on shock using HAC
        result = self._hac_regression(
            merged["beta"],
            merged[shock],
            bandwidth=bandwidth,
        )

        return result

    def _residualize_on_fe(
        self,
        data: pd.DataFrame,
        outcome: str,
        controls: list[str],
    ) -> pd.Series:
        """Residualize outcome on region fixed effects + controls."""
        # Create region dummies
        region_dummies = pd.get_dummies(data["region"], prefix="region", drop_first=True)

        # Combine regressors
        X = region_dummies
        if controls:
            for ctrl in controls:
                if ctrl in data.columns:
                    X[ctrl] = data[ctrl]

        X = sm.add_constant(X)
        y = data[outcome]

        # Drop missing
        mask = ~(y.isna() | X.isna().any(axis=1))
        X = X[mask]
        y = y[mask]

        # Regress and get residuals
        model = sm.OLS(y, X).fit()
        residuals = model.resid

        # Align with original data
        resid_series = pd.Series(index=data.index, dtype=float)
        resid_series.loc[mask] = residuals.values

        return resid_series

    def _cross_sectional_projections(
        self,
        data: pd.DataFrame,
        y_resid: pd.Series,
        exposure: str,
    ) -> pd.DataFrame:
        """Compute cross-sectional projection of residualized y on exposure each quarter."""
        data = data.copy()
        data["y_resid"] = y_resid

        # For each quarter, regress y_resid on exposure
        results = []
        for quarter in data["quarter"].unique():
            qdata = data[data["quarter"] == quarter].dropna(subset=["y_resid", exposure])

            if len(qdata) < 3:
                continue

            # Simple OLS: y_resid = a + beta * exposure
            X = sm.add_constant(qdata[exposure])
            y = qdata["y_resid"]

            try:
                model = sm.OLS(y, X).fit()
                beta = model.params[exposure]
            except Exception:
                beta = np.nan

            results.append({"quarter": quarter, "beta": beta})

        return pd.DataFrame(results)

    def _get_shock_series(
        self,
        data: pd.DataFrame,
        shock: str,
    ) -> pd.DataFrame:
        """Extract shock series by quarter."""
        # Get unique shock value per quarter
        if shock not in data.columns:
            raise ValueError(f"Shock variable {shock} not in data")

        shock_data = (
            data.groupby("quarter")[shock]
            .first()
            .reset_index()
        )

        return shock_data

    def _hac_regression(
        self,
        y: pd.Series,
        X: pd.Series,
        bandwidth: int | None = None,
    ) -> BHJResult:
        """Regress y on X using HAC standard errors."""
        # Prepare data
        y = y.dropna()
        X = X.loc[y.index].dropna()
        y = y.loc[X.index]

        n = len(y)
        if bandwidth is None:
            # Newey-West automatic bandwidth
            bandwidth = int(np.floor(4 * (n / 100) ** (2 / 9)))

        # Add constant
        X_const = sm.add_constant(X)

        # OLS with HAC
        model = sm.OLS(y, X_const).fit(cov_type="HAC", cov_kwds={"maxlags": bandwidth})

        # Extract coefficient for shock variable
        shock_name = X.name
        coef = model.params[shock_name]
        se = model.bse[shock_name]
        tstat = model.tvalues[shock_name]
        pval = model.pvalues[shock_name]
        ci = model.conf_int().loc[shock_name]

        return BHJResult(
            coefficient=coef,
            std_error=se,
            t_stat=tstat,
            pvalue=pval,
            conf_int=(ci[0], ci[1]),
            n_periods=n,
            bandwidth=bandwidth,
        )


class ClusteredInference:
    """
    Clustered standard error estimation with wild bootstrap.

    For panel data with potential serial correlation within clusters.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with panel data.

        Args:
            data: Panel DataFrame
        """
        self.data = data.copy()

    def estimate_clustered(
        self,
        y: str,
        X: list[str],
        cluster_var: str = "region",
        two_way: bool = False,
        time_var: str | None = None,
    ) -> ClusteredSEResult:
        """
        Estimate with clustered standard errors.

        Args:
            y: Outcome variable
            X: List of regressors
            cluster_var: Variable to cluster on
            two_way: If True, use two-way clustering
            time_var: Time variable for two-way clustering

        Returns:
            ClusteredSEResult
        """
        data = self.data.copy()

        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()

        # Prepare data
        y_data = data[y]
        X_data = sm.add_constant(data[X])

        # Drop missing
        mask = ~(y_data.isna() | X_data.isna().any(axis=1))
        y_data = y_data[mask]
        X_data = X_data[mask]
        clusters = data.loc[mask, cluster_var]

        # Fit with clustered SEs
        if two_way and time_var:
            time_clusters = data.loc[mask, time_var]
            # Two-way clustering requires more complex implementation
            # For now, fall back to one-way
            logger.warning("Two-way clustering not yet implemented, using one-way")

        model = sm.OLS(y_data, X_data).fit(
            cov_type="cluster",
            cov_kwds={"groups": clusters},
        )

        # Extract first non-constant coefficient
        coef_name = X[0]
        coef = model.params[coef_name]
        se = model.bse[coef_name]
        tstat = model.tvalues[coef_name]
        pval = model.pvalues[coef_name]
        ci = model.conf_int().loc[coef_name]

        return ClusteredSEResult(
            coefficient=coef,
            std_error=se,
            t_stat=tstat,
            pvalue=pval,
            conf_int=(ci[0], ci[1]),
            n_obs=len(y_data),
            n_clusters=clusters.nunique(),
            cluster_var=cluster_var,
        )

    def wild_bootstrap(
        self,
        y: str,
        X: list[str],
        cluster_var: str = "region",
        n_bootstrap: int = 999,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Wild cluster bootstrap for inference.

        Args:
            y: Outcome variable
            X: List of regressors
            cluster_var: Variable to cluster on
            n_bootstrap: Number of bootstrap iterations
            seed: Random seed

        Returns:
            Dictionary with bootstrap results
        """
        if seed is not None:
            np.random.seed(seed)

        data = self.data.copy()
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()

        # Get original estimate
        y_data = data[y]
        X_data = sm.add_constant(data[X])
        mask = ~(y_data.isna() | X_data.isna().any(axis=1))
        y_data = y_data[mask]
        X_data = X_data[mask]
        clusters = data.loc[mask, cluster_var]

        original_model = sm.OLS(y_data, X_data).fit()
        original_coef = original_model.params[X[0]]
        residuals = original_model.resid

        # Bootstrap
        bootstrap_coefs = []
        cluster_ids = clusters.unique()

        for _ in range(n_bootstrap):
            # Rademacher weights (by cluster)
            weights = np.random.choice([-1, 1], size=len(cluster_ids))
            weight_map = dict(zip(cluster_ids, weights))
            cluster_weights = clusters.map(weight_map)

            # Create bootstrap residuals
            boot_resid = residuals * cluster_weights
            boot_y = original_model.fittedvalues + boot_resid

            # Estimate
            boot_model = sm.OLS(boot_y, X_data).fit()
            bootstrap_coefs.append(boot_model.params[X[0]])

        bootstrap_coefs = np.array(bootstrap_coefs)

        # Compute p-value (two-sided)
        centered_boot = bootstrap_coefs - original_coef
        p_value = np.mean(np.abs(centered_boot) >= np.abs(original_coef))

        # Confidence interval (percentile method)
        ci_lower = np.percentile(bootstrap_coefs, 2.5)
        ci_upper = np.percentile(bootstrap_coefs, 97.5)

        return {
            "coefficient": original_coef,
            "bootstrap_se": np.std(bootstrap_coefs),
            "p_value_wild": p_value,
            "conf_int_wild": (ci_lower, ci_upper),
            "n_bootstrap": n_bootstrap,
            "bootstrap_coefs": bootstrap_coefs,
        }


def compare_inference_methods(
    panel: pd.DataFrame,
    outcome: str = "log_income_pc",
    exposure: str = "E_oil_r",
    shock: str = "oil_supply_shock",
) -> pd.DataFrame:
    """
    Compare Driscoll-Kraay vs BHJ inference.

    Args:
        panel: Panel data
        outcome: Outcome variable
        exposure: Exposure variable
        shock: Shock variable

    Returns:
        DataFrame comparing the two methods
    """
    results = []

    # BHJ
    bhj = BHJInference(panel)
    try:
        bhj_result = bhj.estimate(outcome, exposure, shock)
        results.append({
            "method": "BHJ",
            "coefficient": bhj_result.coefficient,
            "std_error": bhj_result.std_error,
            "pvalue": bhj_result.pvalue,
            "n_obs": bhj_result.n_periods,
        })
    except Exception as e:
        logger.warning(f"BHJ estimation failed: {e}")

    # Clustered
    clustered = ClusteredInference(panel)
    try:
        # Need to create interaction first
        panel_copy = panel.copy()
        if isinstance(panel_copy.index, pd.MultiIndex):
            panel_copy = panel_copy.reset_index()

        interaction_name = f"{exposure}_x_{shock.replace('_shock', '')}"
        if exposure in panel_copy.columns and shock in panel_copy.columns:
            panel_copy[interaction_name] = panel_copy[exposure] * panel_copy[shock]

            clustered_data = ClusteredInference(panel_copy)
            clustered_result = clustered_data.estimate_clustered(
                outcome, [interaction_name], "region"
            )
            results.append({
                "method": "Clustered (region)",
                "coefficient": clustered_result.coefficient,
                "std_error": clustered_result.std_error,
                "pvalue": clustered_result.pvalue,
                "n_obs": clustered_result.n_obs,
            })
    except Exception as e:
        logger.warning(f"Clustered estimation failed: {e}")

    return pd.DataFrame(results)
