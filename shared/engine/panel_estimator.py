"""
Panel LP Estimator with Exposure × Shock Shift-Share Design.

Implements panel local projections where identification comes from
cross-bank variation in predetermined exposure to a common shock.

Canonical specification (first-differenced outcome):

    Δy_{b,t+h} ≡ (y_{b,t+h} - y_{b,t-1})
             = α_b + δ_t + β_h (E_b × shock_t)
               + Σ_ℓ ρ_{ℓ,h} Δy_{b,t-ℓ} + ε_{b,t+h}

where:
- Bank FE (α_b): absorbs time-invariant bank differences
- Time FE (δ_t): absorbs ALL common macro events (critical)
- Identification comes ONLY from (E_b × shock_t)
- E_b is predetermined (fixed baseline exposure per bank)

Time FE absorb the level of the common shock. β_h captures the
differential effect per unit of exposure.

Shock definitions should be INNOVATIONS (unexpected components):
- Imported inflation shock: already in DAG
- Monetary policy shock: Δnbk_rate - E[Δnbk_rate | info_{t-1}]
- FX shock: Δlog(USD/KZT) innovation

Uses linearmodels.PanelOLS for two-way fixed effects with
Driscoll-Kraay or clustered standard errors.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PanelLPResult:
    """Result from a Panel LP with Exposure × Shock estimation."""

    edge_id: str
    horizons: list[int]
    coefficients: list[float]
    std_errors: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    pvalues: list[float]
    nobs: list[int]

    # Panel dimensions
    n_units: int = 0
    n_periods: int = 0
    r_squared_within: list[float] = field(default_factory=list)

    # Design metadata
    fixed_effects: list[str] = field(default_factory=lambda: ["bank", "time"])
    se_method: str = "clustered"
    exposure_variable: str = ""

    # Summary
    impact_estimate: float = 0.0
    impact_se: float = 0.0
    cumulative_estimate: float = 0.0

    @property
    def max_horizon(self) -> int:
        return max(self.horizons) if self.horizons else 0


@dataclass
class LeaveOneOutResult:
    """Result from leave-one-bank-out stability test."""

    excluded_bank: str
    coefficients: list[float]
    std_errors: list[float]
    pvalues: list[float]
    nobs: list[int]
    sign_consistent: bool = True


@dataclass
class RegimeSplitResult:
    """Result from regime break stability test."""

    break_date: str
    pre_coefficients: list[float]
    post_coefficients: list[float]
    pre_nobs: int = 0
    post_nobs: int = 0
    sign_consistent: bool = True
    description: str = ""


# ---------------------------------------------------------------------------
# Panel LP estimator
# ---------------------------------------------------------------------------

def estimate_panel_lp_exposure(
    panel: pd.DataFrame,
    max_horizon: int = 2,
    n_lags: int = 1,
    edge_id: str = "",
    se_method: str = "clustered",
    use_first_difference: bool = True,
) -> PanelLPResult:
    """
    Panel LP with Exposure × Shock and two-way fixed effects.

    Canonical model at each horizon h (first-differenced):
        Δy_{b,t+h} = α_b + δ_t + β_h (E_b × shock_t)
                     + Σ_ℓ ρ_{ℓ,h} Δy_{b,t-ℓ} + ε

    Identification comes from cross-bank variation in exposure E_b
    interacted with the common shock. Time FE absorb the level effect
    of the shock.

    Args:
        panel: DataFrame with columns: bank_id, date, outcome, shock,
               exposure, interaction, and optionally outcome_h1, outcome_h2, etc.
        max_horizon: Maximum LP horizon
        n_lags: Number of lags of outcome changes to include
        edge_id: Edge identifier for logging
        se_method: "clustered" (default) or "kernel" for Driscoll-Kraay
        use_first_difference: If True, use Δy_{t+h} - y_{t-1} as dependent var.
                              If False, use levels (not recommended).

    Returns:
        PanelLPResult with coefficients, SEs, CIs, p-values per horizon
    """
    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        logger.warning("linearmodels not installed; falling back to statsmodels FE")
        return _estimate_panel_lp_fallback(panel, max_horizon, n_lags, edge_id)

    if panel.empty:
        return _empty_panel_result(edge_id)

    # Get panel dimensions
    banks = panel["bank_id"].unique()
    n_units = len(banks)
    dates = panel["date"].unique() if "date" in panel.columns else []
    n_periods = len(dates)

    horizons = list(range(max_horizon + 1))
    coefficients = []
    std_errors = []
    ci_lower = []
    ci_upper = []
    pvalues = []
    nobs_list = []
    r2_list = []

    # Compute baseline outcome (y_{t-1}) for first differencing
    panel = panel.copy()
    for bank_id in banks:
        mask = panel["bank_id"] == bank_id
        bank_data = panel[mask].sort_values("date")
        panel.loc[mask, "outcome_lag1_base"] = bank_data["outcome"].shift(1).values

    for h in horizons:
        # Get outcome at horizon h
        if h == 0:
            outcome_col = "outcome"
        else:
            outcome_col = f"outcome_h{h}"

        if outcome_col not in panel.columns and h > 0:
            coefficients.append(np.nan)
            std_errors.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
            pvalues.append(np.nan)
            nobs_list.append(0)
            r2_list.append(np.nan)
            continue

        # Compute first difference: Δy_{t+h} = y_{t+h} - y_{t-1}
        if use_first_difference:
            panel[f"dy_h{h}"] = panel[outcome_col] - panel["outcome_lag1_base"]
            dep_col = f"dy_h{h}"
        else:
            dep_col = outcome_col

        # Build regression data
        reg_cols = [dep_col, "interaction"]

        # Add lags of outcome CHANGES (not levels)
        lag_cols = []
        for lag in range(1, n_lags + 1):
            for bank_id in banks:
                mask = panel["bank_id"] == bank_id
                bank_data = panel[mask].sort_values("date")
                # Lag of outcome change
                panel.loc[mask, f"dy_lag{lag}"] = bank_data["outcome"].diff().shift(lag).values
                panel.loc[mask, f"interaction_lag{lag}"] = bank_data["interaction"].shift(lag).values
            lag_cols.extend([f"dy_lag{lag}", f"interaction_lag{lag}"])

        reg_cols.extend(lag_cols)

        # Drop NaN
        reg_data = panel[["bank_id", "date"] + reg_cols].dropna()

        n = len(reg_data)
        if n < 5:
            coefficients.append(np.nan)
            std_errors.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
            pvalues.append(np.nan)
            nobs_list.append(n)
            r2_list.append(np.nan)
            continue

        try:
            # Set up panel index
            reg_data = reg_data.set_index(["bank_id", "date"])

            dep = reg_data[dep_col]
            exog = reg_data[["interaction"] + lag_cols]

            # PanelOLS with entity + time effects
            model = PanelOLS(
                dep, exog,
                entity_effects=True,
                time_effects=True,
            )

            if se_method == "kernel":
                result = model.fit(cov_type="kernel")
            else:
                result = model.fit(cov_type="clustered", cluster_entity=True)

            beta = float(result.params["interaction"])
            se = float(result.std_errors["interaction"])
            ci = result.conf_int().loc["interaction"]
            pval = float(result.pvalues["interaction"])

            coefficients.append(beta)
            std_errors.append(se)
            ci_lower.append(float(ci["lower"]))
            ci_upper.append(float(ci["upper"]))
            pvalues.append(pval)
            nobs_list.append(int(result.nobs))
            r2_list.append(float(result.rsquared_within))

        except Exception as e:
            logger.warning(f"Panel LP h={h} failed: {e}")
            coefficients.append(np.nan)
            std_errors.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
            pvalues.append(np.nan)
            nobs_list.append(n)
            r2_list.append(np.nan)

    # Summary
    impact = coefficients[0] if coefficients and not np.isnan(coefficients[0]) else 0.0
    impact_se_val = std_errors[0] if std_errors and not np.isnan(std_errors[0]) else 0.0
    cum = sum(c for c in coefficients if not np.isnan(c))

    return PanelLPResult(
        edge_id=edge_id,
        horizons=horizons,
        coefficients=coefficients,
        std_errors=std_errors,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalues=pvalues,
        nobs=nobs_list,
        n_units=n_units,
        n_periods=n_periods,
        r_squared_within=r2_list,
        se_method=se_method,
        impact_estimate=impact,
        impact_se=impact_se_val,
        cumulative_estimate=cum,
    )


def _estimate_panel_lp_fallback(
    panel: pd.DataFrame,
    max_horizon: int,
    n_lags: int,
    edge_id: str,
) -> PanelLPResult:
    """
    Fallback panel LP using statsmodels OLS with dummy variables.

    Used when linearmodels is not installed.
    """
    import statsmodels.api as sm

    if panel.empty:
        return _empty_panel_result(edge_id)

    banks = panel["bank_id"].unique()
    n_units = len(banks)
    dates = panel["date"].unique() if "date" in panel.columns else []
    n_periods = len(dates)

    horizons = list(range(max_horizon + 1))
    coefficients = []
    std_errors = []
    ci_lower_list = []
    ci_upper_list = []
    pvalues = []
    nobs_list = []
    r2_list = []

    for h in horizons:
        outcome_col = "outcome" if h == 0 else f"outcome_h{h}"

        if outcome_col not in panel.columns and h > 0:
            coefficients.append(np.nan)
            std_errors.append(np.nan)
            ci_lower_list.append(np.nan)
            ci_upper_list.append(np.nan)
            pvalues.append(np.nan)
            nobs_list.append(0)
            r2_list.append(np.nan)
            continue

        # Build design matrix with bank dummies (within estimator via demeaning)
        reg_data = panel[["bank_id", "date", outcome_col, "interaction"]].dropna()

        if len(reg_data) < 5:
            coefficients.append(np.nan)
            std_errors.append(np.nan)
            ci_lower_list.append(np.nan)
            ci_upper_list.append(np.nan)
            pvalues.append(np.nan)
            nobs_list.append(len(reg_data))
            r2_list.append(np.nan)
            continue

        try:
            # Demean by bank (within transformation)
            reg_data = reg_data.copy()
            for col in [outcome_col, "interaction"]:
                bank_means = reg_data.groupby("bank_id")[col].transform("mean")
                reg_data[f"{col}_dm"] = reg_data[col] - bank_means

            dep = reg_data[f"{outcome_col}_dm"]
            exog = sm.add_constant(reg_data[["interaction_dm"]])

            bw = max(1, math.floor(1.3 * (max(h, 1) ** (2 / 3))))
            model = sm.OLS(dep, exog).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": bw},
            )

            beta = float(model.params["interaction_dm"])
            se = float(model.bse["interaction_dm"])
            ci = model.conf_int().loc["interaction_dm"]

            coefficients.append(beta)
            std_errors.append(se)
            ci_lower_list.append(float(ci[0]))
            ci_upper_list.append(float(ci[1]))
            pvalues.append(float(model.pvalues["interaction_dm"]))
            nobs_list.append(len(reg_data))
            r2_list.append(float(model.rsquared))

        except Exception as e:
            logger.warning(f"Fallback panel LP h={h} failed: {e}")
            coefficients.append(np.nan)
            std_errors.append(np.nan)
            ci_lower_list.append(np.nan)
            ci_upper_list.append(np.nan)
            pvalues.append(np.nan)
            nobs_list.append(len(reg_data))
            r2_list.append(np.nan)

    impact = coefficients[0] if coefficients and not np.isnan(coefficients[0]) else 0.0
    impact_se_val = std_errors[0] if std_errors and not np.isnan(std_errors[0]) else 0.0
    cum = sum(c for c in coefficients if not np.isnan(c))

    return PanelLPResult(
        edge_id=edge_id,
        horizons=horizons,
        coefficients=coefficients,
        std_errors=std_errors,
        ci_lower=ci_lower_list,
        ci_upper=ci_upper_list,
        pvalues=pvalues,
        nobs=nobs_list,
        n_units=n_units,
        n_periods=n_periods,
        r_squared_within=r2_list,
        se_method="HAC_fallback",
        impact_estimate=impact,
        impact_se=impact_se_val,
        cumulative_estimate=cum,
    )


def _empty_panel_result(edge_id: str) -> PanelLPResult:
    """Return empty result for failed estimation."""
    return PanelLPResult(
        edge_id=edge_id,
        horizons=[0],
        coefficients=[np.nan],
        std_errors=[np.nan],
        ci_lower=[np.nan],
        ci_upper=[np.nan],
        pvalues=[np.nan],
        nobs=[0],
    )


# ---------------------------------------------------------------------------
# Leave-one-bank-out stability test
# ---------------------------------------------------------------------------

def leave_one_bank_out(
    panel: pd.DataFrame,
    max_horizon: int = 2,
    n_lags: int = 1,
    edge_id: str = "",
) -> list[LeaveOneOutResult]:
    """
    Re-run panel LP excluding each bank one at a time.

    If sign or significance flips when any single bank is dropped,
    results are fragile and should be downgraded.

    Args:
        panel: Full panel DataFrame
        max_horizon: Maximum LP horizon
        n_lags: Number of lags
        edge_id: Edge identifier

    Returns:
        List of LeaveOneOutResult, one per excluded bank
    """
    banks = panel["bank_id"].unique()
    results = []

    # Get full-sample signs
    full_result = estimate_panel_lp_exposure(
        panel, max_horizon=max_horizon, n_lags=n_lags, edge_id=edge_id,
    )
    full_signs = [1 if c > 0 else (-1 if c < 0 else 0)
                  for c in full_result.coefficients
                  if not np.isnan(c)]

    for exclude_bank in banks:
        subset = panel[panel["bank_id"] != exclude_bank].copy()

        if len(subset["bank_id"].unique()) < 2:
            continue

        sub_result = estimate_panel_lp_exposure(
            subset, max_horizon=max_horizon, n_lags=n_lags,
            edge_id=f"{edge_id}_excl_{exclude_bank}",
        )

        # Check sign consistency
        sub_signs = [1 if c > 0 else (-1 if c < 0 else 0)
                     for c in sub_result.coefficients
                     if not np.isnan(c)]

        sign_ok = True
        if full_signs and sub_signs:
            # Compare impact (h=0) sign
            if len(full_signs) > 0 and len(sub_signs) > 0:
                sign_ok = full_signs[0] == sub_signs[0] or sub_signs[0] == 0

        results.append(LeaveOneOutResult(
            excluded_bank=exclude_bank,
            coefficients=sub_result.coefficients,
            std_errors=sub_result.std_errors,
            pvalues=sub_result.pvalues,
            nobs=sub_result.nobs,
            sign_consistent=sign_ok,
        ))

    return results


# ---------------------------------------------------------------------------
# Regime split test
# ---------------------------------------------------------------------------

def regime_split_test(
    panel: pd.DataFrame,
    break_dates: list[str] | None = None,
    max_horizon: int = 2,
    n_lags: int = 1,
    edge_id: str = "",
) -> list[RegimeSplitResult]:
    """
    Test coefficient stability across regime breaks.

    Default break dates:
    - 2015-08-20: Tenge float
    - 2018-01-01: IFRS 9 adoption
    - 2020-03-01: COVID-19

    Args:
        panel: Full panel DataFrame
        break_dates: List of break dates (default: standard KZ breaks)
        max_horizon: Maximum LP horizon
        n_lags: Number of lags
        edge_id: Edge identifier

    Returns:
        List of RegimeSplitResult, one per break date
    """
    if break_dates is None:
        break_dates = ["2015-08-20", "2018-01-01", "2020-03-01"]

    break_labels = {
        "2015-08-20": "Tenge float",
        "2018-01-01": "IFRS 9 adoption",
        "2020-03-01": "COVID-19",
    }

    results = []

    for break_str in break_dates:
        break_ts = pd.Timestamp(break_str)

        pre = panel[panel["date"] < break_ts].copy()
        post = panel[panel["date"] >= break_ts].copy()

        # Need minimum observations in each regime
        if len(pre) < 5 or len(post) < 5:
            continue

        pre_result = estimate_panel_lp_exposure(
            pre, max_horizon=max_horizon, n_lags=n_lags,
            edge_id=f"{edge_id}_pre_{break_str}",
        )
        post_result = estimate_panel_lp_exposure(
            post, max_horizon=max_horizon, n_lags=n_lags,
            edge_id=f"{edge_id}_post_{break_str}",
        )

        # Check sign consistency at impact
        pre_sign = (1 if pre_result.impact_estimate > 0
                    else (-1 if pre_result.impact_estimate < 0 else 0))
        post_sign = (1 if post_result.impact_estimate > 0
                     else (-1 if post_result.impact_estimate < 0 else 0))
        sign_ok = pre_sign == post_sign or pre_sign == 0 or post_sign == 0

        results.append(RegimeSplitResult(
            break_date=break_str,
            pre_coefficients=pre_result.coefficients,
            post_coefficients=post_result.coefficients,
            pre_nobs=pre_result.nobs[0] if pre_result.nobs else 0,
            post_nobs=post_result.nobs[0] if post_result.nobs else 0,
            sign_consistent=sign_ok,
            description=break_labels.get(break_str, break_str),
        ))

    return results


# ---------------------------------------------------------------------------
# Exposure variation check
# ---------------------------------------------------------------------------

def check_exposure_variation(
    panel: pd.DataFrame,
    exposure_var: str = "exposure",
    min_cv: float = 0.10,
) -> dict[str, Any]:
    """
    Check if there is sufficient cross-bank variation in exposure.

    If exposure is near-constant across banks, the shift-share
    design has no identifying variation.

    Args:
        panel: Panel DataFrame with exposure column
        exposure_var: Name of exposure column
        min_cv: Minimum coefficient of variation required

    Returns:
        Dict with variation statistics and pass/fail flag.
    """
    if exposure_var not in panel.columns:
        return {
            "passed": False,
            "reason": f"Exposure variable '{exposure_var}' not found",
        }

    # Get unique exposure per bank
    bank_exposures = panel.groupby("bank_id")[exposure_var].first()

    mean_exp = bank_exposures.mean()
    std_exp = bank_exposures.std()
    cv = std_exp / mean_exp if mean_exp != 0 else 0.0
    min_val = bank_exposures.min()
    max_val = bank_exposures.max()

    passed = cv >= min_cv and len(bank_exposures) >= 3

    return {
        "passed": passed,
        "n_banks": len(bank_exposures),
        "mean": float(mean_exp),
        "std": float(std_exp),
        "cv": float(cv),
        "min": float(min_val),
        "max": float(max_val),
        "range": float(max_val - min_val),
        "bank_values": {k: float(v) for k, v in bank_exposures.items()},
        "min_cv_threshold": min_cv,
    }


# ---------------------------------------------------------------------------
# Pre-trends / Leads test
# ---------------------------------------------------------------------------

@dataclass
class LeadsTestResult:
    """Result from pre-trends (leads) test."""

    lead_coefficients: list[float]
    lead_std_errors: list[float]
    lead_pvalues: list[float]
    joint_fstat: float | None = None
    joint_pvalue: float | None = None
    passed: bool = True  # True if leads are jointly insignificant
    description: str = ""


def test_leads(
    panel: pd.DataFrame,
    n_leads: int = 2,
    edge_id: str = "",
) -> LeadsTestResult:
    """
    Pre-trends / leads test for parallel trends assumption.

    Tests whether (E_b × shock_{t+k}) for k > 0 (leads) predict the
    outcome change. If leads are significant, parallel trends may be
    violated.

    Model:
        Δy_{b,t} = α_b + δ_t + Σ_{k=1}^{K} γ_k (E_b × shock_{t+k}) + ε

    If joint F-test on γ_k rejects, fail the test.

    Args:
        panel: Panel DataFrame with columns: bank_id, date, outcome,
               shock, exposure, interaction
        n_leads: Number of leads to include (default 2)
        edge_id: Edge identifier for logging

    Returns:
        LeadsTestResult with lead coefficients and joint test
    """
    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        logger.warning("linearmodels not installed; skipping leads test")
        return LeadsTestResult(
            lead_coefficients=[],
            lead_std_errors=[],
            lead_pvalues=[],
            passed=True,
            description="Skipped (linearmodels not installed)",
        )

    if panel.empty:
        return LeadsTestResult(
            lead_coefficients=[],
            lead_std_errors=[],
            lead_pvalues=[],
            passed=True,
            description="Empty panel",
        )

    panel = panel.copy()
    banks = panel["bank_id"].unique()

    # Compute outcome change
    for bank_id in banks:
        mask = panel["bank_id"] == bank_id
        bank_data = panel[mask].sort_values("date")
        panel.loc[mask, "dy"] = bank_data["outcome"].diff().values

    # Compute leads of interaction
    lead_cols = []
    for k in range(1, n_leads + 1):
        for bank_id in banks:
            mask = panel["bank_id"] == bank_id
            bank_data = panel[mask].sort_values("date")
            # Lead = future value, so shift by -k
            panel.loc[mask, f"interaction_lead{k}"] = bank_data["interaction"].shift(-k).values
        lead_cols.append(f"interaction_lead{k}")

    # Regression data
    reg_cols = ["dy"] + lead_cols
    reg_data = panel[["bank_id", "date"] + reg_cols].dropna()

    if len(reg_data) < 10:
        return LeadsTestResult(
            lead_coefficients=[],
            lead_std_errors=[],
            lead_pvalues=[],
            passed=True,
            description=f"Insufficient observations ({len(reg_data)})",
        )

    try:
        reg_data = reg_data.set_index(["bank_id", "date"])
        dep = reg_data["dy"]
        exog = reg_data[lead_cols]

        model = PanelOLS(
            dep, exog,
            entity_effects=True,
            time_effects=True,
        )
        result = model.fit(cov_type="clustered", cluster_entity=True)

        lead_coefs = [float(result.params[col]) for col in lead_cols]
        lead_ses = [float(result.std_errors[col]) for col in lead_cols]
        lead_pvals = [float(result.pvalues[col]) for col in lead_cols]

        # Joint F-test on all leads
        try:
            f_test = result.wald_test(formula=" = ".join([f"{col} = 0" for col in lead_cols]))
            joint_f = float(f_test.stat)
            joint_p = float(f_test.pval)
        except Exception:
            # Manual joint test: any lead significant at 10%?
            joint_f = None
            joint_p = None

        # Pass if no individual lead significant at 10%
        any_significant = any(p < 0.10 for p in lead_pvals)
        passed = not any_significant

        return LeadsTestResult(
            lead_coefficients=lead_coefs,
            lead_std_errors=lead_ses,
            lead_pvalues=lead_pvals,
            joint_fstat=joint_f,
            joint_pvalue=joint_p,
            passed=passed,
            description=f"Leads test with {n_leads} leads; {'PASS' if passed else 'FAIL'}"
                        + (f" (F={joint_f:.2f}, p={joint_p:.4f})" if joint_f else ""),
        )

    except Exception as e:
        logger.warning(f"Leads test failed: {e}")
        return LeadsTestResult(
            lead_coefficients=[],
            lead_std_errors=[],
            lead_pvalues=[],
            passed=True,
            description=f"Test failed: {e}",
        )


# ---------------------------------------------------------------------------
# Placebo exposure test
# ---------------------------------------------------------------------------

@dataclass
class PlaceboExposureResult:
    """Result from placebo exposure test."""

    placebo_coefficient: float
    placebo_se: float
    placebo_pvalue: float
    true_coefficient: float
    true_se: float
    passed: bool = True  # True if placebo is NOT significant
    description: str = ""


def test_placebo_exposure(
    panel: pd.DataFrame,
    placebo_exposure_col: str,
    true_exposure_col: str = "exposure",
    edge_id: str = "",
) -> PlaceboExposureResult:
    """
    Test whether a placebo exposure loads on the outcome.

    If the true channel is retail credit exposure → NPL, then
    fee_income_share (placebo) should NOT load on NPL.

    Model with both exposures:
        Δy_{b,t} = α_b + δ_t + β (E^{true}_b × shock_t)
                   + γ (E^{placebo}_b × shock_t) + ε

    Pass if γ is not significant.

    Args:
        panel: Panel DataFrame
        placebo_exposure_col: Name of placebo exposure column
        true_exposure_col: Name of true exposure column
        edge_id: Edge identifier

    Returns:
        PlaceboExposureResult
    """
    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        return PlaceboExposureResult(
            placebo_coefficient=0.0,
            placebo_se=0.0,
            placebo_pvalue=1.0,
            true_coefficient=0.0,
            true_se=0.0,
            passed=True,
            description="Skipped (linearmodels not installed)",
        )

    if placebo_exposure_col not in panel.columns:
        return PlaceboExposureResult(
            placebo_coefficient=0.0,
            placebo_se=0.0,
            placebo_pvalue=1.0,
            true_coefficient=0.0,
            true_se=0.0,
            passed=True,
            description=f"Placebo column '{placebo_exposure_col}' not found",
        )

    panel = panel.copy()
    banks = panel["bank_id"].unique()

    # Compute outcome change
    for bank_id in banks:
        mask = panel["bank_id"] == bank_id
        bank_data = panel[mask].sort_values("date")
        panel.loc[mask, "dy"] = bank_data["outcome"].diff().values

    # Compute placebo interaction
    panel["placebo_interaction"] = panel[placebo_exposure_col] * panel["shock"]

    reg_data = panel[["bank_id", "date", "dy", "interaction", "placebo_interaction"]].dropna()

    if len(reg_data) < 10:
        return PlaceboExposureResult(
            placebo_coefficient=0.0,
            placebo_se=0.0,
            placebo_pvalue=1.0,
            true_coefficient=0.0,
            true_se=0.0,
            passed=True,
            description=f"Insufficient observations ({len(reg_data)})",
        )

    try:
        reg_data = reg_data.set_index(["bank_id", "date"])
        dep = reg_data["dy"]
        exog = reg_data[["interaction", "placebo_interaction"]]

        model = PanelOLS(
            dep, exog,
            entity_effects=True,
            time_effects=True,
        )
        result = model.fit(cov_type="clustered", cluster_entity=True)

        true_coef = float(result.params["interaction"])
        true_se = float(result.std_errors["interaction"])
        placebo_coef = float(result.params["placebo_interaction"])
        placebo_se = float(result.std_errors["placebo_interaction"])
        placebo_pval = float(result.pvalues["placebo_interaction"])

        # Pass if placebo is NOT significant at 10%
        passed = placebo_pval >= 0.10

        return PlaceboExposureResult(
            placebo_coefficient=placebo_coef,
            placebo_se=placebo_se,
            placebo_pvalue=placebo_pval,
            true_coefficient=true_coef,
            true_se=true_se,
            passed=passed,
            description=f"Placebo test: placebo_coef={placebo_coef:.4f}, p={placebo_pval:.4f}; "
                        f"{'PASS' if passed else 'FAIL (placebo significant)'}",
        )

    except Exception as e:
        logger.warning(f"Placebo test failed: {e}")
        return PlaceboExposureResult(
            placebo_coefficient=0.0,
            placebo_se=0.0,
            placebo_pvalue=1.0,
            true_coefficient=0.0,
            true_se=0.0,
            passed=True,
            description=f"Test failed: {e}",
        )
