"""
Inference methods: Driscoll-Kraay and BHJ shock-level regression.

V1: Driscoll-Kraay HAC (via linearmodels)
V2: BHJ shock-level regression for shift-share designs
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
    from src.model.shift_share import ShiftShareModel, ShiftShareSpec

    results = []

    # Driscoll-Kraay
    spec = ShiftShareSpec(
        name="dk",
        outcome=outcome,
        interactions=[(exposure, shock)],
        cov_type="kernel",
    )

    model = ShiftShareModel(panel)
    dk_result = model.fit(spec)

    interaction_name = f"{exposure}_x_{shock.replace('_shock', '')}"
    if interaction_name in dk_result.params.index:
        results.append({
            "method": "Driscoll-Kraay",
            "coefficient": dk_result.params[interaction_name],
            "std_error": dk_result.std_errors[interaction_name],
            "pvalue": dk_result.pvalues[interaction_name],
            "n_obs": dk_result.nobs,
        })

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

    return pd.DataFrame(results)
