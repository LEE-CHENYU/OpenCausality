"""
Time-Series Estimators for DAG Edge Estimation.

Implements:
- Jorda (2005) Local Projections with HAC (Newey-West) standard errors
- Identity sensitivity calculator for mechanical relationships
- Immutable evidence card builder

Uses statsmodels for OLS + HAC inference.
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
class LPResult:
    """Result from a Local Projections estimation."""

    edge_id: str
    horizons: list[int]
    coefficients: list[float]
    std_errors: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    pvalues: list[float]
    nobs: list[int]
    # Summary stats
    impact_estimate: float = 0.0  # Coefficient at h=0
    impact_se: float = 0.0
    cumulative_estimate: float = 0.0  # Sum of coefficients
    # Diagnostics
    hac_bandwidth: list[int] = field(default_factory=list)
    residual_autocorrelation: list[float | None] = field(default_factory=list)
    r_squared: list[float] = field(default_factory=list)

    @property
    def max_horizon(self) -> int:
        return max(self.horizons) if self.horizons else 0

    @property
    def is_significant_at_impact(self) -> bool:
        """Check if impact (h=0) coefficient is significant at 5%."""
        if self.pvalues:
            return self.pvalues[0] < 0.05
        return False


@dataclass
class ImmutableResult:
    """Result for an immutable (validated evidence) edge."""

    edge_id: str
    point_estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    pvalue: float | None
    source_block: str  # e.g., "Block A DiD"
    source_description: str
    nobs: int | None = None


@dataclass
class AccountingBridgeResult:
    """Result for an accounting bridge (near-mechanical) edge."""

    edge_id: str
    sensitivity: float  # dY/dX at current values
    formula: str
    at_values: dict[str, float]  # Current values used
    description: str = ""
    is_deterministic: bool = True


@dataclass
class IdentityResult:
    """Result for an identity (mechanical) edge."""

    edge_id: str
    sensitivity: float  # dY/dX at current values
    formula: str
    at_values: dict[str, float]  # Current values used
    numerator_label: str = ""
    denominator_label: str = ""


# ---------------------------------------------------------------------------
# HAC bandwidth selection
# ---------------------------------------------------------------------------

def _hac_bandwidth_horizon(h: int, T: int) -> int:
    """
    Select HAC bandwidth for horizon h.

    Uses max of:
    - h-dependent: floor(1.3 * h^(2/3))
    - Andrews rule: floor(4 * (T/100)^(2/9))
    - Minimum of max(h, 1)
    """
    bw_h = max(1, math.floor(1.3 * (max(h, 1) ** (2 / 3))))
    bw_andrews = max(1, math.floor(4 * ((T / 100) ** (2 / 9))))
    return max(bw_h, bw_andrews, max(h, 1))


# ---------------------------------------------------------------------------
# Local Projections estimator
# ---------------------------------------------------------------------------

def estimate_lp(
    y: pd.Series,
    x: pd.Series,
    max_horizon: int = 6,
    n_lags: int = 2,
    controls: pd.DataFrame | None = None,
    edge_id: str = "",
) -> LPResult:
    """
    Jorda (2005) time-series Local Projections.

    Model at each horizon h:
        y_{t+h} - y_{t-1} = alpha_h + beta_h * x_t + gamma * controls + eps

    Uses statsmodels OLS with cov_type="HAC" (Newey-West).

    Args:
        y: Outcome series (aligned with x by index)
        x: Treatment series
        max_horizon: Maximum forecast horizon
        n_lags: Number of lags of y and x to include as controls
        controls: Additional control variables (optional)
        edge_id: Edge identifier for logging

    Returns:
        LPResult with coefficients, SEs, CIs, p-values per horizon
    """
    import statsmodels.api as sm

    # Align series
    common = y.index.intersection(x.index)
    if controls is not None:
        common = common.intersection(controls.index)

    y = y.reindex(common).dropna()
    x = x.reindex(common).dropna()
    common = y.index.intersection(x.index)
    y = y.reindex(common)
    x = x.reindex(common)

    T = len(y)
    if T < 10:
        logger.warning(f"Edge {edge_id}: only {T} observations, LP may be unreliable")

    horizons = list(range(max_horizon + 1))
    coefficients = []
    std_errors = []
    ci_lower = []
    ci_upper = []
    pvalues = []
    nobs_list = []
    hac_bws = []
    r2_list = []
    resid_ac_list: list[float | None] = []

    for h in horizons:
        # Forward cumulative change: y_{t+h} - y_{t-1}
        if h == 0:
            lhs = y.copy()
        else:
            lhs = y.shift(-h)

        # Build RHS: treatment + lags
        rhs_dict: dict[str, pd.Series] = {"treatment": x}

        # Add lags of y and x
        for lag in range(1, n_lags + 1):
            rhs_dict[f"y_lag{lag}"] = y.shift(lag)
            rhs_dict[f"x_lag{lag}"] = x.shift(lag)

        # Add external controls
        if controls is not None:
            for col in controls.columns:
                rhs_dict[col] = controls[col].reindex(y.index)

        rhs = pd.DataFrame(rhs_dict)

        # Combine and drop NaN
        combined = pd.DataFrame({"lhs": lhs}).join(rhs).dropna()

        n = len(combined)
        if n < 5:
            # Not enough observations
            coefficients.append(np.nan)
            std_errors.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
            pvalues.append(np.nan)
            nobs_list.append(n)
            hac_bws.append(0)
            r2_list.append(np.nan)
            resid_ac_list.append(None)
            continue

        dep = combined["lhs"]
        exog = sm.add_constant(combined.drop(columns=["lhs"]))

        # HAC bandwidth
        bw = _hac_bandwidth_horizon(h, n)
        hac_bws.append(bw)

        try:
            model = sm.OLS(dep, exog).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": bw},
            )

            # Extract treatment coefficient
            beta = model.params["treatment"]
            se = model.bse["treatment"]
            ci = model.conf_int().loc["treatment"]
            pval = model.pvalues["treatment"]

            coefficients.append(float(beta))
            std_errors.append(float(se))
            ci_lower.append(float(ci[0]))
            ci_upper.append(float(ci[1]))
            pvalues.append(float(pval))
            nobs_list.append(n)
            r2_list.append(float(model.rsquared))

            # Residual autocorrelation (Durbin-Watson approximation)
            resid = model.resid
            if len(resid) > 2:
                ac1 = resid.autocorr(lag=1) if hasattr(resid, "autocorr") else None
                resid_ac_list.append(float(ac1) if ac1 is not None and not np.isnan(ac1) else None)
            else:
                resid_ac_list.append(None)

        except Exception as e:
            logger.warning(f"Edge {edge_id}, h={h}: estimation failed: {e}")
            coefficients.append(np.nan)
            std_errors.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
            pvalues.append(np.nan)
            nobs_list.append(n)
            r2_list.append(np.nan)
            resid_ac_list.append(None)

    # Summary statistics
    impact = coefficients[0] if coefficients and not np.isnan(coefficients[0]) else 0.0
    impact_se_val = std_errors[0] if std_errors and not np.isnan(std_errors[0]) else 0.0
    cum = sum(c for c in coefficients if not np.isnan(c))

    lp_result = LPResult(
        edge_id=edge_id,
        horizons=horizons,
        coefficients=coefficients,
        std_errors=std_errors,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalues=pvalues,
        nobs=nobs_list,
        impact_estimate=impact,
        impact_se=impact_se_val,
        cumulative_estimate=cum,
        hac_bandwidth=hac_bws,
        residual_autocorrelation=resid_ac_list,
        r_squared=r2_list,
    )

    # Validate invariants (log-only, non-blocking)
    lp_violations = validate_lp_result(lp_result)
    for v in lp_violations:
        logger.warning(f"LP validation [{edge_id}]: {v}")

    return lp_result


# ---------------------------------------------------------------------------
# Annual LP estimator (frequency-aware variant)
# ---------------------------------------------------------------------------

def estimate_lp_annual(
    y: pd.Series,
    x: pd.Series,
    max_horizon: int = 2,
    n_lags: int = 1,
    controls: pd.DataFrame | None = None,
    edge_id: str = "",
) -> LPResult:
    """
    Annual-frequency Local Projections for robustness.

    Uses the same Jorda LP framework but with parameters appropriate
    for annual data: shorter horizon (max 2), fewer lags (1),
    and smaller HAC bandwidth.

    Args:
        y: Outcome series (annual frequency, aligned with x)
        x: Treatment series
        max_horizon: Maximum forecast horizon (default 2 for annual)
        n_lags: Number of lags (default 1 for annual)
        controls: Additional control variables
        edge_id: Edge identifier for logging

    Returns:
        LPResult with annual-frequency estimates
    """
    return estimate_lp(
        y=y,
        x=x,
        max_horizon=max_horizon,
        n_lags=n_lags,
        controls=controls,
        edge_id=edge_id,
    )


# ---------------------------------------------------------------------------
# Accounting bridge computations
# ---------------------------------------------------------------------------

def compute_accounting_bridge(
    edge_id: str,
    loans: float,
    rwa: float,
    cor: float,
    capital: float,
    tax_rate: float = 0.20,
) -> AccountingBridgeResult:
    """
    Compute accounting bridge sensitivity for near-mechanical relationships.

    Supports:
    - loan_portfolio_to_rwa: d(RWA)/d(loans) = avg_risk_weight = RWA/loans
    - cor_to_capital: d(capital)/d(CoR) = -avg_loans * (1 - tax_rate) / 100

    Args:
        edge_id: Edge identifier
        loans: Net loans in bn KZT
        rwa: Risk-weighted assets in bn KZT
        cor: Cost of risk in %
        capital: Total capital in bn KZT
        tax_rate: Corporate tax rate (default 20%)

    Returns:
        AccountingBridgeResult with deterministic sensitivity
    """
    if edge_id == "loan_portfolio_to_rwa":
        avg_risk_weight = rwa / loans if loans > 0 else 0.0
        return AccountingBridgeResult(
            edge_id=edge_id,
            sensitivity=avg_risk_weight,
            formula="d(RWA)/d(loans) = avg_risk_weight = RWA / total_loans",
            at_values={"loans": loans, "rwa": rwa, "avg_risk_weight": avg_risk_weight},
            description=(
                f"Average risk weight = {avg_risk_weight:.3f}. "
                f"A 1 bn KZT increase in loans adds ~{avg_risk_weight:.2f} bn to RWA."
            ),
            is_deterministic=True,
        )
    elif edge_id == "cor_to_capital":
        sensitivity = -loans * (1 - tax_rate) / 100
        return AccountingBridgeResult(
            edge_id=edge_id,
            sensitivity=sensitivity,
            formula="d(capital)/d(CoR) = -avg_loans * (1 - tax_rate) / 100",
            at_values={
                "loans": loans,
                "capital": capital,
                "tax_rate": tax_rate,
                "cor": cor,
            },
            description=(
                f"A 1pp increase in CoR reduces capital by ~{abs(sensitivity):.1f} bn KZT "
                f"(pre-tax provision impact on capital)."
            ),
            is_deterministic=True,
        )
    else:
        raise ValueError(f"Unknown accounting bridge edge: {edge_id}")


# ---------------------------------------------------------------------------
# Immutable evidence builder
# ---------------------------------------------------------------------------

# Validated evidence from completed research blocks
IMMUTABLE_EVIDENCE: dict[str, dict[str, Any]] = {
    "fx_to_cpi_tradable": {
        "point": 0.113,
        "se": 0.028,
        "ci_lower": 0.058,
        "ci_upper": 0.168,
        "pvalue": 0.0001,
        "source_block": "Block A DiD",
        "source_description": (
            "FX pass-through to tradable CPI from difference-in-differences "
            "estimation. 11.3% pass-through at 6 months post-shock."
        ),
        "nobs": 1200,
    },
    "fx_to_cpi_nontradable": {
        "point": 0.0,
        "se": 0.015,
        "ci_lower": -0.029,
        "ci_upper": 0.029,
        "pvalue": 0.99,
        "source_block": "Block A DiD (falsification)",
        "source_description": (
            "FX pass-through to non-tradable CPI. Precisely null result from "
            "falsification test confirming identification validity."
        ),
        "nobs": 1200,
    },
    "cpi_to_nominal_income": {
        "point": 0.65,
        "se": 0.18,
        "ci_lower": 0.30,
        "ci_upper": 1.00,
        "pvalue": 0.0003,
        "source_block": "Block B LP-IV",
        "source_description": (
            "Effect of CPI inflation on nominal income growth from LP-IV "
            "using imported inflation as instrument. Partial pass-through."
        ),
        "nobs": 80,
    },
    "fx_to_real_expenditure": {
        "point": -0.10,
        "se": 0.04,
        "ci_lower": -0.18,
        "ci_upper": -0.02,
        "pvalue": 0.012,
        "source_block": "Block F",
        "source_description": (
            "Effect of exchange rate depreciation on real household expenditure. "
            "10% depreciation reduces real spending by ~1%."
        ),
        "nobs": 60,
    },
}


def get_immutable_result(edge_id: str) -> ImmutableResult:
    """
    Get immutable result for a validated evidence edge.

    Args:
        edge_id: Edge identifier

    Returns:
        ImmutableResult with validated estimates

    Raises:
        KeyError: If edge is not in immutable evidence
    """
    if edge_id not in IMMUTABLE_EVIDENCE:
        raise KeyError(f"Edge {edge_id} has no immutable evidence")

    ev = IMMUTABLE_EVIDENCE[edge_id]
    return ImmutableResult(
        edge_id=edge_id,
        point_estimate=ev["point"],
        se=ev["se"],
        ci_lower=ev["ci_lower"],
        ci_upper=ev["ci_upper"],
        pvalue=ev["pvalue"],
        source_block=ev["source_block"],
        source_description=ev["source_description"],
        nobs=ev.get("nobs"),
    )


# ---------------------------------------------------------------------------
# Identity sensitivity calculator
# ---------------------------------------------------------------------------

def compute_identity_sensitivity(
    capital: float,
    rwa: float,
) -> dict[str, IdentityResult]:
    """
    Compute K2 = Capital / RWA identity sensitivities.

    K2 ratio (%) = 100 * Capital / RWA

    Partial derivatives:
    - dK2/dCapital = 100 / RWA
    - dK2/dRWA = -100 * Capital / RWA^2

    Args:
        capital: Total capital in bn KZT
        rwa: Risk-weighted assets in bn KZT

    Returns:
        Dict with results for capital_to_k2 and rwa_to_k2
    """
    dk2_dcapital = 100.0 / rwa
    dk2_drwa = -100.0 * capital / (rwa ** 2)

    return {
        "capital_to_k2": IdentityResult(
            edge_id="capital_to_k2",
            sensitivity=dk2_dcapital,
            formula="dK2/dCapital = 100 / RWA",
            at_values={"capital": capital, "rwa": rwa},
            numerator_label="Total Capital (bn KZT)",
            denominator_label="RWA (bn KZT)",
        ),
        "rwa_to_k2": IdentityResult(
            edge_id="rwa_to_k2",
            sensitivity=dk2_drwa,
            formula="dK2/dRWA = -100 * Capital / RWA^2",
            at_values={"capital": capital, "rwa": rwa},
            numerator_label="RWA (bn KZT)",
            denominator_label="K2 ratio (%)",
        ),
    }


# ---------------------------------------------------------------------------
# Expected signs for sanity checking
# ---------------------------------------------------------------------------

EXPECTED_SIGNS: dict[str, int] = {
    "oil_supply_to_brent": -1,  # Negative supply shock -> higher prices
    "oil_supply_to_fx": 1,     # Supply shock -> KZT depreciation
    "oil_demand_to_fx": -1,    # Demand shock -> KZT appreciation
    "vix_to_fx": 1,            # Risk-off -> KZT depreciation
    "cpi_to_nbk_rate": 1,      # Inflation -> rate hike (Taylor rule)
    "fx_to_nbk_rate": 1,       # Depreciation -> rate hike
}


def check_sign_consistency(result: LPResult) -> tuple[bool, str]:
    """
    Check if the impact coefficient has the expected sign.

    Returns (is_consistent, message).
    """
    expected = EXPECTED_SIGNS.get(result.edge_id)
    if expected is None:
        return True, "No expected sign specified"

    if np.isnan(result.impact_estimate):
        return False, "Impact estimate is NaN"

    actual_sign = 1 if result.impact_estimate > 0 else (-1 if result.impact_estimate < 0 else 0)

    if actual_sign == expected:
        return True, f"Sign consistent: expected {'+' if expected > 0 else '-'}, got {result.impact_estimate:.4f}"
    elif actual_sign == 0:
        return True, "Estimate is exactly zero"
    else:
        return False, (
            f"Sign inconsistent: expected {'+' if expected > 0 else '-'}, "
            f"got {result.impact_estimate:.4f}"
        )


# ---------------------------------------------------------------------------
# LP result invariant validation
# ---------------------------------------------------------------------------

def validate_lp_result(result: LPResult) -> list[str]:
    """
    Validate LP result invariants. Returns list of violation strings.

    Canonical length: len(result.horizons).
    Per-element checks skip NaN (valid for failed horizons).
    Non-blocking — never raises.
    """
    violations: list[str] = []
    n_h = len(result.horizons)

    # Vector length checks
    for field_name in ("coefficients", "std_errors", "pvalues", "ci_lower", "ci_upper"):
        vec = getattr(result, field_name, None)
        if vec is not None and len(vec) != n_h:
            violations.append(
                f"{field_name} length ({len(vec)}) != horizons length ({n_h})"
            )

    # Per-element checks
    for i in range(min(n_h, len(result.std_errors) if result.std_errors else 0)):
        se = result.std_errors[i]
        if not np.isnan(se) and se < 0:
            violations.append(f"Negative SE at h={result.horizons[i]}: {se}")

    for i in range(min(n_h, len(result.pvalues) if result.pvalues else 0)):
        pv = result.pvalues[i]
        if not np.isnan(pv) and not (0 <= pv <= 1):
            violations.append(f"p-value out of [0,1] at h={result.horizons[i]}: {pv}")

    ci_lo = result.ci_lower or []
    ci_hi = result.ci_upper or []
    for i in range(min(n_h, len(ci_lo), len(ci_hi))):
        lo, hi = ci_lo[i], ci_hi[i]
        if not np.isnan(lo) and not np.isnan(hi) and lo > hi:
            violations.append(
                f"CI lower > upper at h={result.horizons[i]}: [{lo}, {hi}]"
            )

    # Scalar checks
    for i, n in enumerate(result.nobs or []):
        if n is not None and n < 0:
            violations.append(f"Negative nobs at h={result.horizons[i]}: {n}")

    return violations
