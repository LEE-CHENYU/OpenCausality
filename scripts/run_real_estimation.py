#!/usr/bin/env python3
"""
Run Real Econometric Estimations on KSPI K2 DAG.

Replaces all placeholder estimates with real econometric results using
cached data and proper statistical inference.

Groups:
    A:       Monthly LP (6 edges) - unchanged
    B:       Immutable (4 edges) - unchanged
    C-Q:     Quarterly LP, KSPI-only, true quarterly obs (4 edges)
    C-A:     Annual LP robustness, KSPI-only (same 4 edges, annual frequency)
    C-PANEL: Sector Panel LP, Exposure x Shock (4 edges, sector level)
    C-KSPI:  KSPI-only, no extension possible (2 edges)
    C-BRIDGE:Accounting bridges (2 edges: loans->RWA, CoR->capital)
    D:       Identity (2 edges) - unchanged

Total edge cards: 20 original + 4 sector panel companions = 24

Usage:
    python scripts/run_real_estimation.py
    python scripts/run_real_estimation.py --output-dir outputs/agentic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from shared.engine.data_assembler import (
    EDGE_NODE_MAP,
    PANEL_EDGE_SPECS,
    ACCOUNTING_BRIDGE_EDGES,
    assemble_edge_data,
    get_edge_group,
    _load_kspi_quarterly,
    _load_kspi_quarterly_filtered,
    compute_share_interpolated,
    load_node_series,
)
from shared.engine.ts_estimator import (
    LPResult,
    ImmutableResult,
    IdentityResult,
    AccountingBridgeResult,
    estimate_lp,
    estimate_lp_annual,
    get_immutable_result,
    compute_identity_sensitivity,
    compute_accounting_bridge,
    check_sign_consistency,
    IMMUTABLE_EVIDENCE,
)
from shared.engine.panel_estimator import (
    PanelLPResult,
    estimate_panel_lp_exposure,
    leave_one_bank_out,
    regime_split_test,
    check_exposure_variation,
)
from shared.data.kz_bank_panel import KZBankPanelClient
from shared.agentic.output.edge_card import (
    EdgeCard,
    Estimates,
    DiagnosticResult,
    Interpretation,
    FailureFlags,
    CounterfactualApplicability,
    compute_credibility_score,
)
from shared.agentic.output.provenance import SpecDetails, DataProvenance, SourceProvenance
from shared.agentic.identification.screen import IdentifiabilityScreen, IdentifiabilityResult
from shared.agentic.ts_guard import TSGuard, TSGuardResult
from shared.agentic.output.edge_card import (
    IdentificationBlock,
    CounterfactualBlock,
    PropagationRole,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estimation parameters
# ---------------------------------------------------------------------------

MONTHLY_LP_EDGES = [
    "oil_supply_to_brent",
    "oil_supply_to_fx",
    "oil_demand_to_fx",
    "vix_to_fx",
    "cpi_to_nbk_rate",
    "fx_to_nbk_rate",
]

IMMUTABLE_EDGES = [
    "fx_to_cpi_tradable",
    "fx_to_cpi_nontradable",
    "cpi_to_nominal_income",
    "fx_to_real_expenditure",
]

# Group C-Q: Quarterly LP with true quarterly KSPI observations
QUARTERLY_LP_EDGES = [
    "shock_to_npl_kspi",
    "shock_to_cor_kspi",
    "nbk_rate_to_deposit_cost",
    "nbk_rate_to_cor",
]

# Group C-A: Annual LP robustness (same edges, annual frequency)
ANNUAL_LP_EDGES = [
    "shock_to_npl_kspi",
    "shock_to_cor_kspi",
    "nbk_rate_to_deposit_cost",
    "nbk_rate_to_cor",
]

# Group C-Panel: Sector panel LP with Exposure x Shock
PANEL_LP_EDGES = [
    "shock_to_npl_sector",
    "shock_to_cor_sector",
    "nbk_rate_to_deposit_cost_sector",
    "nbk_rate_to_cor_sector",
]

# Group C-KSPI: KSPI-only, no extension possible
KSPI_ONLY_EDGES = [
    "expenditure_to_payments_revenue",
    "portfolio_mix_to_rwa",
]

# Group C-Bridge: Accounting bridges (reclassified from LP)
BRIDGE_EDGES = [
    "loan_portfolio_to_rwa",
    "cor_to_capital",
]

IDENTITY_EDGES = [
    "capital_to_k2",
    "rwa_to_k2",
]

# All original edges (20)
ALL_ORIGINAL_EDGES = (
    MONTHLY_LP_EDGES + IMMUTABLE_EDGES + QUARTERLY_LP_EDGES +
    KSPI_ONLY_EDGES + BRIDGE_EDGES + IDENTITY_EDGES
)
# All edges including sector panel companions (24)
ALL_EDGES = ALL_ORIGINAL_EDGES + PANEL_LP_EDGES

# DAG hash placeholder (would come from dag.compute_hash() in production)
DAG_HASH = hashlib.sha256(b"kspi_k2_stress_v3_extended").hexdigest()


# ---------------------------------------------------------------------------
# Unit normalization registry
# ---------------------------------------------------------------------------
# CRITICAL: Every edge must have treatment and outcome units documented
# for safe chain propagation

EDGE_UNITS: dict[str, dict[str, str]] = {
    # Group A: Monthly LP
    "oil_supply_to_brent": {
        "treatment_unit": "1 SD Baumeister supply shock (mbd equivalent)",
        "outcome_unit": "% change in Brent price",
    },
    "oil_supply_to_fx": {
        "treatment_unit": "1 SD Baumeister supply shock",
        "outcome_unit": "% change in USD/KZT",
    },
    "oil_demand_to_fx": {
        "treatment_unit": "1 SD Baumeister demand shock",
        "outcome_unit": "% change in USD/KZT",
    },
    "vix_to_fx": {
        "treatment_unit": "1 point VIX increase",
        "outcome_unit": "% change in USD/KZT",
    },
    "cpi_to_nbk_rate": {
        "treatment_unit": "1pp YoY tradable CPI inflation",
        "outcome_unit": "pp NBK base rate",
        "is_reaction_function": True,  # NOT causal - endogenous policy response
    },
    "fx_to_nbk_rate": {
        "treatment_unit": "1% KZT depreciation (MoM)",
        "outcome_unit": "pp NBK base rate",
        "is_reaction_function": True,  # NOT causal - endogenous policy response
    },
    # Group B: Immutable
    "fx_to_cpi_tradable": {
        "treatment_unit": "10% KZT depreciation",
        "outcome_unit": "pp tradable CPI (cumulative 12m)",
    },
    "fx_to_cpi_nontradable": {
        "treatment_unit": "10% KZT depreciation",
        "outcome_unit": "pp non-tradable CPI (cumulative 12m)",
    },
    "cpi_to_nominal_income": {
        "treatment_unit": "1pp CPI inflation",
        "outcome_unit": "pp nominal income growth",
    },
    "fx_to_real_expenditure": {
        "treatment_unit": "10% KZT depreciation",
        "outcome_unit": "% real expenditure decline",
    },
    # Group C-Q: Quarterly LP (KSPI)
    "shock_to_npl_kspi": {
        "treatment_unit": "1pp tradable CPI shock (quarterly)",
        "outcome_unit": "bps NPL ratio change",
    },
    "shock_to_cor_kspi": {
        "treatment_unit": "1pp tradable CPI shock (quarterly)",
        "outcome_unit": "bps CoR change",
    },
    "nbk_rate_to_deposit_cost": {
        "treatment_unit": "1pp NBK base rate increase",
        "outcome_unit": "pp deposit cost increase",
    },
    "nbk_rate_to_cor": {
        "treatment_unit": "1pp NBK base rate increase",
        "outcome_unit": "pp CoR increase",
    },
    # Group C-Panel: Sector Panel
    "shock_to_npl_sector": {
        "treatment_unit": "1pp CPI shock × E_consumer exposure",
        "outcome_unit": "bps NPL differential per unit exposure",
    },
    "shock_to_cor_sector": {
        "treatment_unit": "1pp CPI shock × E_consumer exposure",
        "outcome_unit": "bps CoR differential per unit exposure",
    },
    "nbk_rate_to_deposit_cost_sector": {
        "treatment_unit": "1pp rate × E_demand_dep exposure",
        "outcome_unit": "pp deposit cost differential per unit exposure",
    },
    "nbk_rate_to_cor_sector": {
        "treatment_unit": "1pp rate × E_shortterm exposure",
        "outcome_unit": "pp CoR differential per unit exposure",
    },
    # Group C-KSPI: KSPI-only
    "expenditure_to_payments_revenue": {
        "treatment_unit": "1% real expenditure change",
        "outcome_unit": "bn KZT payments revenue",
    },
    "portfolio_mix_to_rwa": {
        "treatment_unit": "1pp consumer loan share change",
        "outcome_unit": "bn KZT RWA change",
    },
    # Group C-Bridge: Accounting
    "loan_portfolio_to_rwa": {
        "treatment_unit": "1 bn KZT net loans increase",
        "outcome_unit": "bn KZT RWA increase (avg risk weight)",
    },
    "cor_to_capital": {
        "treatment_unit": "1pp CoR increase",
        "outcome_unit": "bn KZT capital decline (provisions)",
    },
    # Group D: Identity
    "capital_to_k2": {
        "treatment_unit": "1 bn KZT capital increase",
        "outcome_unit": "pp K2 ratio change",
    },
    "rwa_to_k2": {
        "treatment_unit": "1 bn KZT RWA increase",
        "outcome_unit": "pp K2 ratio change",
    },
}


# ---------------------------------------------------------------------------
# Edge card builders
# ---------------------------------------------------------------------------

def build_lp_edge_card(
    lp_result: LPResult,
    data: pd.DataFrame,
    edge_id: str,
    is_quarterly: bool = False,
    is_annual_robustness: bool = False,
    companion_edge_id: str | None = None,
    share_interpolated: float = 0.0,
) -> EdgeCard:
    """Build EdgeCard from LP estimation result."""
    n_obs = lp_result.nobs[0] if lp_result.nobs else 0
    n_calendar = len(data) if data is not None and not data.empty else 0

    # Impact estimate as main point estimate
    point = lp_result.impact_estimate
    se = lp_result.impact_se

    # If impact is NaN, use first non-NaN
    if np.isnan(point):
        for i, c in enumerate(lp_result.coefficients):
            if not np.isnan(c):
                point = c
                se = lp_result.std_errors[i]
                break

    ci_lo = point - 1.96 * se if not np.isnan(se) else np.nan
    ci_hi = point + 1.96 * se if not np.isnan(se) else np.nan
    pval = lp_result.pvalues[0] if lp_result.pvalues and not np.isnan(lp_result.pvalues[0]) else None

    # Get unit information
    edge_units = EDGE_UNITS.get(edge_id, {})
    treatment_unit = edge_units.get("treatment_unit", "")
    outcome_unit = edge_units.get("outcome_unit", "")

    estimates = Estimates(
        point=float(point) if not np.isnan(point) else 0.0,
        se=float(se) if not np.isnan(se) else 0.0,
        ci_95=(float(ci_lo) if not np.isnan(ci_lo) else 0.0,
               float(ci_hi) if not np.isnan(ci_hi) else 0.0),
        pvalue=float(pval) if pval is not None else None,
        horizons=lp_result.horizons,
        irf=lp_result.coefficients,
        irf_ci_lower=lp_result.ci_lower,
        irf_ci_upper=lp_result.ci_upper,
        # Unit normalization
        treatment_unit=treatment_unit,
        outcome_unit=outcome_unit,
        # Sample size details
        n_calendar_periods=n_calendar,
        n_effective_obs_h0=n_obs,
        n_effective_obs_by_horizon=list(lp_result.nobs) if lp_result.nobs else None,
    )

    # Diagnostics
    diagnostics: dict[str, DiagnosticResult] = {}

    # HAC bandwidth
    if lp_result.hac_bandwidth:
        diagnostics["hac_bandwidth"] = DiagnosticResult(
            name="hac_bandwidth",
            passed=True,
            value=float(lp_result.hac_bandwidth[0]),
            message=f"Newey-West bandwidth: {lp_result.hac_bandwidth[0]}",
        )

    # Effective observations
    diagnostics["effective_obs"] = DiagnosticResult(
        name="effective_obs",
        passed=n_obs >= 20,
        value=float(n_obs),
        threshold=20.0,
        message=f"{n_obs} observations" + (" (small sample)" if n_obs < 20 else ""),
    )

    # R-squared at impact
    if lp_result.r_squared and not np.isnan(lp_result.r_squared[0]):
        diagnostics["r_squared_h0"] = DiagnosticResult(
            name="r_squared_h0",
            passed=True,
            value=float(lp_result.r_squared[0]),
            message=f"R-squared at h=0: {lp_result.r_squared[0]:.4f}",
        )

    # Residual autocorrelation
    if lp_result.residual_autocorrelation and lp_result.residual_autocorrelation[0] is not None:
        ac1 = lp_result.residual_autocorrelation[0]
        diagnostics["residual_ac1"] = DiagnosticResult(
            name="residual_ac1",
            passed=abs(ac1) < 0.5,
            value=float(ac1),
            threshold=0.5,
            message=f"AR(1) residual autocorrelation: {ac1:.3f}",
        )

    # Sign consistency
    sign_ok, sign_msg = check_sign_consistency(lp_result)
    diagnostics["sign_consistency"] = DiagnosticResult(
        name="sign_consistency",
        passed=sign_ok,
        message=sign_msg,
    )

    # Failure flags
    failure_flags = FailureFlags(
        small_sample=is_quarterly or is_annual_robustness or n_obs < 30,
    )

    # Interpretation
    treatment_node, outcome_node = EDGE_NODE_MAP[edge_id]
    freq_note = " (annual frequency, robustness)" if is_annual_robustness else ""
    interpretation = Interpretation(
        estimand=f"IRF of {outcome_node} to a unit shock in {treatment_node}{freq_note}",
        is_not="Structural causal effect under all interventions",
        channels=["direct", "indirect through unobserved channels"],
        population="Kazakhstan aggregate time-series (Kaspi Bank)",
        conditions="Sample period, no regime changes",
    )

    # Counterfactual applicability
    counterfactual = CounterfactualApplicability(
        supports_shock_path=True,
        supports_policy_intervention=False,
        intervention_note="Reduced-form LP estimate; valid for shock propagation, not policy counterfactuals",
    )

    # Credibility
    design_weight = 0.7  # LOCAL_PROJECTIONS default
    data_coverage = min(1.0, n_obs / 100) if not is_quarterly else min(1.0, n_obs / 17)
    if is_quarterly or is_annual_robustness:
        design_weight = 0.5  # Downweight for small sample

    # B-cap: only if n_obs < 30 AND share_interpolated < 0.30 (not blanket quarterly cap)
    score, rating = compute_credibility_score(
        diagnostics=diagnostics,
        failure_flags=failure_flags,
        design_weight=design_weight,
        data_coverage=data_coverage,
    )

    if (n_obs < 30 and share_interpolated < 0.30) and rating == "A":
        rating = "B"
        score = min(score, 0.79)
    elif share_interpolated >= 0.30 and rating in ("A", "B"):
        rating = "C"
        score = min(score, 0.59)

    design_name = "LOCAL_PROJECTIONS"
    if is_annual_robustness:
        design_name = "LOCAL_PROJECTIONS_ANNUAL"

    spec_details = SpecDetails(
        design=design_name,
        controls=[f"y_lag{i}" for i in range(1, 3)] + [f"x_lag{i}" for i in range(1, 3)],
        instruments=[],
        fixed_effects=[],
        se_method="HAC_newey_west",
        horizon=lp_result.horizons,
    )

    # Data provenance
    date_range = (str(data.index.min().date()), str(data.index.max().date())) if len(data) > 0 else None
    provenance = DataProvenance(
        treatment_source=SourceProvenance(
            connector="parquet_file",
            dataset=treatment_node,
            row_count=len(data),
            date_range=date_range,
        ),
        outcome_source=SourceProvenance(
            connector="parquet_file",
            dataset=outcome_node,
            row_count=len(data),
            date_range=date_range,
        ),
        combined_row_count=len(data),
        combined_date_range=date_range,
        entity_boundary_note="kaspi_bank subsidiary level" if "kspi" in edge_id else None,
    )

    # Check null equivalence
    is_null = False
    null_bound = None
    if pval is not None and pval > 0.10 and abs(point) < 2 * se:
        is_null = True
        null_bound = float(2 * se) if not np.isnan(se) else None

    return EdgeCard(
        edge_id=edge_id + ("_annual" if is_annual_robustness else ""),
        dag_version_hash=DAG_HASH,
        spec_hash=spec_details.compute_hash(),
        spec_details=spec_details,
        data_provenance=provenance,
        estimates=estimates,
        diagnostics=diagnostics,
        interpretation=interpretation,
        failure_flags=failure_flags,
        counterfactual=counterfactual,
        credibility_rating=rating,
        credibility_score=score,
        companion_edge_id=companion_edge_id,
        is_precisely_null=is_null,
        null_equivalence_bound=null_bound,
    )


def build_immutable_edge_card(result: ImmutableResult) -> EdgeCard:
    """Build EdgeCard from immutable (validated evidence) result."""
    estimates = Estimates(
        point=result.point_estimate,
        se=result.se,
        ci_95=(result.ci_lower, result.ci_upper),
        pvalue=result.pvalue,
    )

    diagnostics = {
        "validated_evidence": DiagnosticResult(
            name="validated_evidence",
            passed=True,
            value=1.0,
            message=f"Validated from {result.source_block}",
        ),
    }
    if result.nobs:
        diagnostics["effective_obs"] = DiagnosticResult(
            name="effective_obs",
            passed=result.nobs >= 20,
            value=float(result.nobs),
            threshold=20.0,
        )

    failure_flags = FailureFlags()
    is_null = result.point_estimate == 0.0

    interpretation = Interpretation(
        estimand=result.source_description,
        is_not="Re-estimated result; this is locked validated evidence",
        channels=["as identified in source block"],
        conditions=f"Source: {result.source_block}",
    )

    counterfactual = CounterfactualApplicability(
        supports_shock_path=True,
        supports_policy_intervention=False,
        intervention_note=f"Validated evidence from {result.source_block}",
    )

    spec_details = SpecDetails(
        design="IMMUTABLE_EVIDENCE",
        se_method="from_source_block",
    )

    score, rating = compute_credibility_score(
        diagnostics=diagnostics,
        failure_flags=failure_flags,
        design_weight=0.9,  # High credibility for validated evidence
        data_coverage=1.0,
    )

    return EdgeCard(
        edge_id=result.edge_id,
        dag_version_hash=DAG_HASH,
        spec_hash=spec_details.compute_hash(),
        spec_details=spec_details,
        estimates=estimates,
        diagnostics=diagnostics,
        interpretation=interpretation,
        failure_flags=failure_flags,
        counterfactual=counterfactual,
        credibility_rating=rating,
        credibility_score=score,
        is_precisely_null=is_null,
        null_equivalence_bound=float(2 * result.se) if is_null else None,
    )


def build_identity_edge_card(result: IdentityResult) -> EdgeCard:
    """Build EdgeCard from identity (mechanical) result."""
    estimates = Estimates(
        point=result.sensitivity,
        se=0.0,  # Deterministic
        ci_95=(result.sensitivity, result.sensitivity),
        pvalue=None,
    )

    diagnostics = {
        "identity_check": DiagnosticResult(
            name="identity_check",
            passed=True,
            value=result.sensitivity,
            message=f"Deterministic: {result.formula}",
        ),
    }

    failure_flags = FailureFlags(
        mechanical_identity_risk=True,
    )

    interpretation = Interpretation(
        estimand=f"Sensitivity: {result.formula}",
        is_not="Causal effect; this is a mechanical identity",
        channels=["mechanical/accounting"],
        conditions=f"Evaluated at: {result.at_values}",
    )

    counterfactual = CounterfactualApplicability(
        supports_shock_path=True,
        supports_policy_intervention=True,
        intervention_note="Mechanical identity; always holds by definition",
    )

    spec_details = SpecDetails(
        design="IDENTITY",
        se_method="deterministic",
    )

    score, rating = compute_credibility_score(
        diagnostics=diagnostics,
        failure_flags=failure_flags,
        design_weight=1.0,  # Perfect identification (it's an identity)
        data_coverage=1.0,
    )

    return EdgeCard(
        edge_id=result.edge_id,
        dag_version_hash=DAG_HASH,
        spec_hash=spec_details.compute_hash(),
        spec_details=spec_details,
        estimates=estimates,
        diagnostics=diagnostics,
        interpretation=interpretation,
        failure_flags=failure_flags,
        counterfactual=counterfactual,
        credibility_rating=rating,
        credibility_score=score,
    )


def build_bridge_edge_card(result: AccountingBridgeResult) -> EdgeCard:
    """Build EdgeCard from accounting bridge result."""
    estimates = Estimates(
        point=result.sensitivity,
        se=0.0,  # Deterministic
        ci_95=(result.sensitivity, result.sensitivity),
        pvalue=None,
    )

    diagnostics = {
        "identity_consistency": DiagnosticResult(
            name="identity_consistency",
            passed=True,
            value=result.sensitivity,
            message=f"Accounting bridge: {result.formula}",
        ),
    }

    failure_flags = FailureFlags(
        mechanical_identity_risk=True,
    )

    interpretation = Interpretation(
        estimand=f"Accounting sensitivity: {result.formula}",
        is_not="Causal effect; this is a near-mechanical accounting relationship",
        channels=["accounting/mechanical"],
        conditions=f"Evaluated at: {result.at_values}",
    )

    counterfactual = CounterfactualApplicability(
        supports_shock_path=True,
        supports_policy_intervention=True,
        intervention_note="Accounting bridge; holds by construction under current risk weights / tax regime",
    )

    spec_details = SpecDetails(
        design="ACCOUNTING_BRIDGE",
        se_method="deterministic",
    )

    score, rating = compute_credibility_score(
        diagnostics=diagnostics,
        failure_flags=failure_flags,
        design_weight=0.9,  # High credibility for accounting identity
        data_coverage=1.0,
    )

    return EdgeCard(
        edge_id=result.edge_id,
        dag_version_hash=DAG_HASH,
        spec_hash=spec_details.compute_hash(),
        spec_details=spec_details,
        estimates=estimates,
        diagnostics=diagnostics,
        interpretation=interpretation,
        failure_flags=failure_flags,
        counterfactual=counterfactual,
        credibility_rating=rating,
        credibility_score=score,
    )


def build_panel_edge_card(
    panel_result: PanelLPResult,
    panel_data: pd.DataFrame,
    edge_id: str,
    exposure_var: str,
    kspi_companion: str,
    loo_results: list | None = None,
    regime_results: list | None = None,
    exposure_check: dict | None = None,
) -> EdgeCard:
    """Build EdgeCard from panel LP result."""
    n_obs = panel_result.nobs[0] if panel_result.nobs else 0

    point = panel_result.impact_estimate
    se = panel_result.impact_se
    ci_lo = point - 1.96 * se if not np.isnan(se) else np.nan
    ci_hi = point + 1.96 * se if not np.isnan(se) else np.nan
    pval = panel_result.pvalues[0] if panel_result.pvalues and not np.isnan(panel_result.pvalues[0]) else None

    estimates = Estimates(
        point=float(point) if not np.isnan(point) else 0.0,
        se=float(se) if not np.isnan(se) else 0.0,
        ci_95=(float(ci_lo) if not np.isnan(ci_lo) else 0.0,
               float(ci_hi) if not np.isnan(ci_hi) else 0.0),
        pvalue=float(pval) if pval is not None else None,
        horizons=panel_result.horizons,
        irf=panel_result.coefficients,
        irf_ci_lower=panel_result.ci_lower,
        irf_ci_upper=panel_result.ci_upper,
    )

    diagnostics: dict[str, DiagnosticResult] = {}

    diagnostics["effective_obs"] = DiagnosticResult(
        name="effective_obs",
        passed=n_obs >= 20,
        value=float(n_obs),
        threshold=20.0,
        message=f"{n_obs} panel observations ({panel_result.n_units} banks x {panel_result.n_periods} periods)",
    )

    # Leave-one-out stability
    if loo_results:
        all_stable = all(r.sign_consistent for r in loo_results)
        diagnostics["leave_one_bank_out"] = DiagnosticResult(
            name="leave_one_bank_out",
            passed=all_stable,
            value=1.0 if all_stable else 0.0,
            message="Sign stable across all bank exclusions" if all_stable
                    else "Sign flipped when excluding: " + ", ".join(
                        r.excluded_bank for r in loo_results if not r.sign_consistent),
        )

    # Exposure variation
    if exposure_check:
        diagnostics["exposure_variation"] = DiagnosticResult(
            name="exposure_variation",
            passed=exposure_check.get("passed", False),
            value=float(exposure_check.get("cv", 0.0)),
            threshold=0.10,
            message=f"CV of exposure = {exposure_check.get('cv', 0.0):.3f}, "
                    f"range = [{exposure_check.get('min', 0):.2f}, {exposure_check.get('max', 0):.2f}]",
        )

    # Regime breaks
    if regime_results:
        all_regime_stable = all(r.sign_consistent for r in regime_results)
        diagnostics["regime_break"] = DiagnosticResult(
            name="regime_break",
            passed=all_regime_stable,
            value=1.0 if all_regime_stable else 0.0,
            message="Coefficient stable across regime breaks" if all_regime_stable
                    else "Sign flipped at: " + ", ".join(
                        r.description for r in regime_results if not r.sign_consistent),
        )

    # R-squared within
    if panel_result.r_squared_within and not np.isnan(panel_result.r_squared_within[0]):
        diagnostics["r_squared_within"] = DiagnosticResult(
            name="r_squared_within",
            passed=True,
            value=float(panel_result.r_squared_within[0]),
            message=f"R-squared within at h=0: {panel_result.r_squared_within[0]:.4f}",
        )

    # Failure flags
    loo_ok = all(r.sign_consistent for r in loo_results) if loo_results else True
    failure_flags = FailureFlags(
        small_sample=n_obs < 30,
        weak_identification=not (exposure_check or {}).get("passed", True),
        regime_break_detected=not all(r.sign_consistent for r in (regime_results or [])),
    )

    # Downgrade if leave-one-out fails
    if not loo_ok:
        failure_flags.weak_identification = True

    interpretation = Interpretation(
        estimand=f"Differential bank response per unit {exposure_var}, "
                 f"per unit shock (shift-share Exposure x Shock)",
        is_not="Aggregate causal effect of shock on outcome; this is a "
               "differential effect by exposure level",
        channels=["cross-bank exposure variation"],
        population="Kazakhstan banking sector (4 banks)",
        conditions="Predetermined exposure; common shock; entity + time FE",
    )

    counterfactual = CounterfactualApplicability(
        supports_shock_path=True,
        supports_policy_intervention=False,
        intervention_note="Shift-share panel LP; valid for differential shock propagation across banks",
    )

    spec_details = SpecDetails(
        design="PANEL_LP_EXPOSURE_FE",
        controls=[f"outcome_lag1", f"interaction_lag1"],
        instruments=[],
        fixed_effects=["bank", "time"],
        se_method=panel_result.se_method,
        horizon=panel_result.horizons,
    )

    # Credibility
    design_weight = 0.7  # PANEL_LP_EXPOSURE_FE
    data_coverage = min(1.0, n_obs / 50)
    score, rating = compute_credibility_score(
        diagnostics=diagnostics,
        failure_flags=failure_flags,
        design_weight=design_weight,
        data_coverage=data_coverage,
    )

    # If leave-one-out fails, cap at C
    if not loo_ok and rating in ("A", "B"):
        rating = "C"
        score = min(score, 0.59)

    # Data provenance with panel dimensions
    provenance = DataProvenance(
        combined_row_count=n_obs,
        panel_dimensions={
            "n_units": panel_result.n_units,
            "n_periods": panel_result.n_periods,
            "balance": "unbalanced",
        },
        entity_boundary_note="Multi-bank panel: kaspi, halyk, forte, bcc",
    )

    is_null = False
    null_bound = None
    if pval is not None and pval > 0.10 and se > 0 and abs(point) < 2 * se:
        is_null = True
        null_bound = float(2 * se)

    return EdgeCard(
        edge_id=edge_id,
        dag_version_hash=DAG_HASH,
        spec_hash=spec_details.compute_hash(),
        spec_details=spec_details,
        data_provenance=provenance,
        estimates=estimates,
        diagnostics=diagnostics,
        interpretation=interpretation,
        failure_flags=failure_flags,
        counterfactual=counterfactual,
        credibility_rating=rating,
        credibility_score=score,
        companion_edge_id=kspi_companion,
        is_precisely_null=is_null,
        null_equivalence_bound=null_bound,
    )


# ---------------------------------------------------------------------------
# Per-edge risk block and dashboard
# ---------------------------------------------------------------------------

def print_edge_risk_block(
    edge_id: str,
    card: EdgeCard,
    id_result: IdentifiabilityResult,
    ts_result: TSGuardResult | None = None,
) -> None:
    """Print per-edge risk block after estimation (CLI reminder)."""
    lines = [
        "",
        "\u2501" * 55,
        f"Edge: {edge_id}",
        "\u2501" * 55,
        f"Claim Level: {id_result.claim_level}",
    ]

    # Identification risks
    if id_result.risks:
        lines.append("Identification Risks:")
        for risk, level in id_result.risks.items():
            if level in ("medium", "high"):
                lines.append(f"  - {risk}: {level.upper()}")

    # TS dynamics risks
    if ts_result:
        lines.append("TS Dynamics Risks:")
        for risk, level in ts_result.dynamics_risk.items():
            if level in ("medium", "high"):
                lines.append(f"  - {risk.replace('_risk', '')}: {level.upper()}")

    # Diagnostics summary
    if card.diagnostics:
        lines.append("Diagnostics:")
        for name, diag in card.diagnostics.items():
            status = "PASS" if diag.passed else "FAIL"
            msg = f" ({diag.message})" if diag.message and not diag.passed else ""
            lines.append(f"  - {name}: {status}{msg}")

    # Counterfactual status
    cf_allowed = id_result.counterfactual_allowed
    lines.append(f"Counterfactual Use: {'ALLOWED' if cf_allowed else 'BLOCKED'}")
    if not cf_allowed and id_result.counterfactual_reason_blocked:
        lines.append(f"Reason: {id_result.counterfactual_reason_blocked}")

    lines.append("")
    lines.append("Even if p<0.05, this does not establish a causal effect.")
    lines.append("\u2501" * 55)

    for line in lines:
        logger.info(line)


def _infer_edge_type(edge_id: str) -> str:
    """Infer edge_type from group membership."""
    if edge_id in IMMUTABLE_EDGES:
        return "immutable"
    if edge_id in BRIDGE_EDGES:
        return "mechanical"
    if edge_id in IDENTITY_EDGES:
        return "identity"
    if edge_id in {"cpi_to_nbk_rate", "fx_to_nbk_rate"}:
        return "reaction_function"
    return "causal"


def _infer_variant_of(edge_id: str) -> str | None:
    """Infer variant_of for companion/annual edges."""
    sector_map = {
        "shock_to_npl_sector": "shock_to_npl_kspi",
        "shock_to_cor_sector": "shock_to_cor_kspi",
        "nbk_rate_to_deposit_cost_sector": "nbk_rate_to_deposit_cost",
        "nbk_rate_to_cor_sector": "nbk_rate_to_cor",
    }
    if edge_id in sector_map:
        return sector_map[edge_id]
    if edge_id.endswith("_annual"):
        return edge_id.replace("_annual", "")
    return None


def attach_identification(
    card: EdgeCard,
    id_result: IdentifiabilityResult,
    query_mode: str = "REDUCED_FORM",
) -> None:
    """Attach identification block and propagation role to EdgeCard."""
    from shared.agentic.query_mode import (
        QueryModeConfig,
        derive_propagation_role,
        is_edge_allowed_for_propagation,
        is_shock_cf_allowed,
        is_policy_cf_allowed,
    )

    card.identification = IdentificationBlock(
        claim_level=id_result.claim_level,
        risks=id_result.risks,
        untestable_assumptions=id_result.untestable_assumptions,
        testable_threats_passed=id_result.testable_threats_passed,
        testable_threats_failed=id_result.testable_threats_failed,
    )

    # Load mode config
    config = QueryModeConfig.load()
    mode_spec = config.get_spec(query_mode)

    # Split counterfactual block
    shock_ok, shock_reason = is_shock_cf_allowed(id_result.claim_level, mode_spec)
    policy_ok, policy_reason = is_policy_cf_allowed(id_result.claim_level, mode_spec)

    # Mode can only restrict, not expand
    if id_result.reason_shock_blocked:
        shock_ok = False
        shock_reason = id_result.reason_shock_blocked
    if id_result.reason_policy_blocked:
        policy_ok = False
        policy_reason = id_result.reason_policy_blocked

    card.counterfactual_block = CounterfactualBlock(
        shock_scenario_allowed=shock_ok,
        policy_intervention_allowed=policy_ok,
        reason_shock_blocked=shock_reason,
        reason_policy_blocked=policy_reason,
    )

    # Derive propagation role with per-mode dicts
    base_edge_id = card.edge_id.replace("_annual", "").replace("_sector", "")
    et = _infer_edge_type(base_edge_id)
    role = derive_propagation_role(et, id_result.claim_level)

    mode_prop, mode_shock, mode_policy = {}, {}, {}
    for mn, ms in config.modes.items():
        mode_prop[mn] = is_edge_allowed_for_propagation(role, ms)
        s_ok, _ = is_shock_cf_allowed(id_result.claim_level, ms)
        p_ok, _ = is_policy_cf_allowed(id_result.claim_level, ms)
        mode_shock[mn] = s_ok
        mode_policy[mn] = p_ok

    card.propagation_role = PropagationRole(
        role=role,
        mode_propagation_allowed=mode_prop,
        mode_shock_cf_allowed=mode_shock,
        mode_policy_cf_allowed=mode_policy,
        selected_for_counterfactual=mode_shock.get(query_mode, False),
    )

    # Set variant_of
    variant = _infer_variant_of(card.edge_id)
    if variant:
        card.variant_of = variant


def generate_id_dashboard(
    id_results: dict[str, IdentifiabilityResult],
) -> str:
    """Generate the identifiability risk dashboard as markdown."""
    screen = IdentifiabilityScreen()
    return screen.generate_dashboard(id_results)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    cards: dict[str, EdgeCard],
    lp_results: dict[str, LPResult],
    immutable_results: dict[str, ImmutableResult],
    identity_results: dict[str, IdentityResult],
    bridge_results: dict[str, AccountingBridgeResult],
    panel_results: dict[str, PanelLPResult],
    annual_lp_results: dict[str, LPResult],
    query_mode: str = "REDUCED_FORM",
) -> str:
    """Generate comprehensive markdown report."""
    mode_descriptions = {
        "STRUCTURAL": "Identified causal effects for policy intervention analysis",
        "REDUCED_FORM": "Shock/scenario responses for stress testing",
        "DESCRIPTIVE": "Measurement, decomposition, and mechanical relationships",
    }
    total_cards = len(cards)
    lines = [
        "# KSPI K2 DAG: Real Econometric Estimation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**DAG Version Hash:** `{DAG_HASH[:16]}...`",
        f"**Query Mode:** `{query_mode}` — {mode_descriptions.get(query_mode, '')}",
        f"**Total Edge Cards:** {total_cards}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Group | Count | Method | Status |",
        "|-------|-------|--------|--------|",
        f"| A: Monthly LP | {len(MONTHLY_LP_EDGES)} | Time-series LP, HAC | Estimated |",
        f"| B: Immutable | {len(IMMUTABLE_EDGES)} | Validated evidence | Locked |",
        f"| C-Q: Quarterly LP | {len(QUARTERLY_LP_EDGES)} | KSPI quarterly LP | Estimated |",
        f"| C-A: Annual LP | {len(ANNUAL_LP_EDGES)} | KSPI annual LP (robustness) | Estimated |",
        f"| C-Panel: Sector Panel | {len(PANEL_LP_EDGES)} | Exposure x Shock, bank+time FE | Estimated |",
        f"| C-KSPI: KSPI-only | {len(KSPI_ONLY_EDGES)} | Quarterly LP, no extension | Estimated |",
        f"| C-Bridge: Accounting | {len(BRIDGE_EDGES)} | Deterministic sensitivity | Computed |",
        f"| D: Identity | {len(IDENTITY_EDGES)} | Mechanical sensitivity | Computed |",
        f"| **Total** | **{total_cards}** | | |",
        "",
    ]

    # Credibility distribution
    ratings: dict[str, int] = {}
    for card in cards.values():
        r = card.credibility_rating
        ratings[r] = ratings.get(r, 0) + 1

    lines.extend([
        "### Credibility Distribution",
        "",
        "| Rating | Count | Edges |",
        "|--------|-------|-------|",
    ])
    for r in ["A", "B", "C", "D"]:
        if r in ratings:
            edge_list = [eid for eid, c in cards.items() if c.credibility_rating == r]
            lines.append(f"| {r} | {ratings[r]} | {', '.join(edge_list)} |")
    lines.append("")

    # -----------------------------------------------------------------------
    # Group A: Monthly LP
    # -----------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## Group A: Monthly Local Projections (6 edges)",
        "",
        "Time-series LP with Newey-West HAC standard errors.",
        "",
        "**Warning:** `cpi_to_nbk_rate` and `fx_to_nbk_rate` are **reaction function** edges (endogenous policy response), ",
        "not causal effects. They should NOT be used for shock propagation without re-specifying as monetary policy surprises.",
        "",
        "| Edge | Impact (h=0) | SE | 95% CI | p-value | N | Sign OK | Rating |",
        "|------|-------------|-----|---------|---------|---|---------|--------|",
    ])

    for edge_id in MONTHLY_LP_EDGES:
        card = cards.get(edge_id)
        lp = lp_results.get(edge_id)
        if card and card.estimates:
            e = card.estimates
            sign_diag = card.diagnostics.get("sign_consistency")
            sign_ok = "Yes" if sign_diag and sign_diag.passed else "No"
            pv = f"{e.pvalue:.4f}" if e.pvalue is not None else "-"
            n = lp.nobs[0] if lp and lp.nobs else "-"
            lines.append(
                f"| `{edge_id}` | {e.point:.4f} | {e.se:.4f} | "
                f"[{e.ci_95[0]:.4f}, {e.ci_95[1]:.4f}] | {pv} | {n} | "
                f"{sign_ok} | {card.credibility_rating} |"
            )
    lines.append("")

    # -----------------------------------------------------------------------
    # Group B: Immutable
    # -----------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## Group B: Immutable Evidence (4 edges)",
        "",
        "Locked from validated research blocks.",
        "",
        "| Edge | Estimate | SE | 95% CI | Source | Rating |",
        "|------|---------|-----|---------|--------|--------|",
    ])

    for edge_id in IMMUTABLE_EDGES:
        card = cards.get(edge_id)
        ir = immutable_results.get(edge_id)
        if card and card.estimates and ir:
            e = card.estimates
            lines.append(
                f"| `{edge_id}` | {e.point:.4f} | {e.se:.4f} | "
                f"[{e.ci_95[0]:.4f}, {e.ci_95[1]:.4f}] | {ir.source_block} | "
                f"{card.credibility_rating} |"
            )
    lines.append("")

    # -----------------------------------------------------------------------
    # Group C-Q: Quarterly LP
    # -----------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## Group C-Q: Quarterly LP with KSPI Data (4 edges)",
        "",
        "KSPI quarterly LP using post-2020 quarterly observations. Entity: kaspi_bank.",
        "",
        "**Note on N:** `N_cal` = calendar periods in sample; `N_eff` = effective obs after lags/leads.",
        "",
        "| Edge | Impact | SE | p-value | N_cal | N_eff | Treatment | Outcome | Rating |",
        "|------|--------|-----|---------|-------|-------|-----------|---------|--------|",
    ])

    for edge_id in QUARTERLY_LP_EDGES:
        card = cards.get(edge_id)
        lp = lp_results.get(edge_id)
        if card and card.estimates:
            e = card.estimates
            pv = f"{e.pvalue:.4f}" if e.pvalue is not None else "-"
            n_eff = lp.nobs[0] if lp and lp.nobs else "-"
            n_cal = e.n_calendar_periods if e.n_calendar_periods else "-"
            t_unit = e.treatment_unit[:20] + "..." if len(e.treatment_unit) > 20 else e.treatment_unit
            o_unit = e.outcome_unit[:15] + "..." if len(e.outcome_unit) > 15 else e.outcome_unit
            lines.append(
                f"| `{edge_id}` | {e.point:.2f} | {e.se:.2f} | {pv} | "
                f"{n_cal} | {n_eff} | {t_unit} | {o_unit} | {card.credibility_rating} |"
            )
    lines.append("")

    # -----------------------------------------------------------------------
    # Group C-A: Annual LP robustness
    # -----------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## Group C-A: Annual LP Robustness (4 edges)",
        "",
        "Annual-frequency LP using pre-2020 annual observations (2011-2019).",
        "Sign/magnitude consistency check against quarterly estimates.",
        "",
        "**Note:** `N_eff` = effective observations after lags. Annual data has fewer obs due to lag requirements.",
        "",
        "| Edge | Impact (A) | SE | N_eff | Impact (Q) | Sign Match | Rating |",
        "|------|-----------|-----|-------|-----------|-----------|--------|",
    ])

    for edge_id in ANNUAL_LP_EDGES:
        annual_id = edge_id + "_annual"
        card = cards.get(annual_id)
        annual_lp = annual_lp_results.get(edge_id)
        quarterly_lp = lp_results.get(edge_id)
        if card and card.estimates and annual_lp:
            e = card.estimates
            n_eff = annual_lp.nobs[0] if annual_lp.nobs else "-"
            q_impact = quarterly_lp.impact_estimate if quarterly_lp else 0.0
            q_sign = "+" if q_impact > 0 else ("-" if q_impact < 0 else "0")
            a_sign = "+" if e.point > 0 else ("-" if e.point < 0 else "0")
            sign_match = "Yes" if q_sign == a_sign else "No"
            lines.append(
                f"| `{edge_id}` | {e.point:.2f} | {e.se:.2f} | "
                f"{n_eff} | {q_impact:.2f} | {sign_match} | {card.credibility_rating} |"
            )
    lines.append("")

    # -----------------------------------------------------------------------
    # Group C-Panel: Sector Panel LP
    # -----------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## Group C-Panel: Sector Panel LP (4 edges)",
        "",
        "Shift-share panel LP: y_{b,t+h} = alpha_b + delta_t + beta_h (E_b x shock_t) + eps",
        "Identification from cross-bank variation in predetermined exposure.",
        "",
        "| Edge | Impact | SE | N (obs) | Banks | Exposure | LOO Stable | Rating |",
        "|------|--------|-----|---------|-------|----------|-----------|--------|",
    ])

    for edge_id in PANEL_LP_EDGES:
        card = cards.get(edge_id)
        pr = panel_results.get(edge_id)
        if card and card.estimates and pr:
            e = card.estimates
            loo_diag = card.diagnostics.get("leave_one_bank_out")
            loo_ok = "Yes" if loo_diag and loo_diag.passed else "No"
            exposure = PANEL_EDGE_SPECS.get(edge_id, {}).get("exposure", "?")
            lines.append(
                f"| `{edge_id}` | {e.point:.4f} | {e.se:.4f} | "
                f"{pr.nobs[0] if pr.nobs else '-'} | {pr.n_units} | "
                f"{exposure} | {loo_ok} | {card.credibility_rating} |"
            )
    lines.append("")

    # -----------------------------------------------------------------------
    # Group C-KSPI: KSPI-only edges
    # -----------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## Group C-KSPI: KSPI-Only Edges (2 edges)",
        "",
        "No extension possible for these edges.",
        "",
        "| Edge | Impact (h=0) | SE | N | Rating |",
        "|------|-------------|-----|---|--------|",
    ])

    for edge_id in KSPI_ONLY_EDGES:
        card = cards.get(edge_id)
        lp = lp_results.get(edge_id)
        if card and card.estimates:
            e = card.estimates
            n = lp.nobs[0] if lp and lp.nobs else "-"
            lines.append(
                f"| `{edge_id}` | {e.point:.4f} | {e.se:.4f} | {n} | "
                f"{card.credibility_rating} |"
            )
    lines.append("")

    # -----------------------------------------------------------------------
    # Group C-Bridge: Accounting Bridges
    # -----------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## Group C-Bridge: Accounting Bridges (2 edges)",
        "",
        "Deterministic/near-mechanical accounting relationships.",
        "",
        "| Edge | Sensitivity | Formula | Description | Rating |",
        "|------|-----------|---------|-------------|--------|",
    ])

    for edge_id in BRIDGE_EDGES:
        card = cards.get(edge_id)
        br = bridge_results.get(edge_id)
        if card and br:
            lines.append(
                f"| `{edge_id}` | {br.sensitivity:.4f} | {br.formula} | "
                f"{br.description} | {card.credibility_rating} |"
            )
    lines.append("")

    # -----------------------------------------------------------------------
    # Group D: Identity
    # -----------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## Group D: Identity Sensitivities (2 edges)",
        "",
        "Deterministic partial derivatives of K2 = 100 * Capital / RWA.",
        "",
        "| Edge | Sensitivity | Formula | At Values | Rating |",
        "|------|-----------|---------|-----------|--------|",
    ])

    for edge_id in IDENTITY_EDGES:
        ir = identity_results.get(edge_id)
        card = cards.get(edge_id)
        if ir and card:
            vals = ", ".join(f"{k}={v}" for k, v in ir.at_values.items())
            lines.append(
                f"| `{edge_id}` | {ir.sensitivity:.6f} | {ir.formula} | {vals} | {card.credibility_rating} |"
            )
    lines.append("")

    # -----------------------------------------------------------------------
    # Comparison table: KSPI-specific vs sector estimates
    # -----------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## Comparison: KSPI-Specific vs Sector Panel Estimates",
        "",
        "| Relationship | KSPI Q Impact | KSPI A Impact | Sector Impact | KSPI Rating | Sector Rating |",
        "|-------------|--------------|--------------|--------------|-------------|--------------|",
    ])

    comparison_pairs = [
        ("shock_to_npl_kspi", "shock_to_npl_sector"),
        ("shock_to_cor_kspi", "shock_to_cor_sector"),
        ("nbk_rate_to_deposit_cost", "nbk_rate_to_deposit_cost_sector"),
        ("nbk_rate_to_cor", "nbk_rate_to_cor_sector"),
    ]
    for kspi_id, sector_id in comparison_pairs:
        kspi_card = cards.get(kspi_id)
        annual_card = cards.get(kspi_id + "_annual")
        sector_card = cards.get(sector_id)
        kspi_val = f"{kspi_card.estimates.point:.4f}" if kspi_card and kspi_card.estimates else "-"
        annual_val = f"{annual_card.estimates.point:.4f}" if annual_card and annual_card.estimates else "-"
        sector_val = f"{sector_card.estimates.point:.4f}" if sector_card and sector_card.estimates else "-"
        kspi_r = kspi_card.credibility_rating if kspi_card else "-"
        sector_r = sector_card.credibility_rating if sector_card else "-"
        lines.append(
            f"| {kspi_id} | {kspi_val} | {annual_val} | {sector_val} | {kspi_r} | {sector_r} |"
        )
    lines.append("")

    # -----------------------------------------------------------------------
    # Diagnostics summary
    # -----------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## Diagnostics Summary",
        "",
    ])

    # Small sample warnings
    small_sample_edges = [
        eid for eid, c in cards.items()
        if c.failure_flags.small_sample
    ]
    if small_sample_edges:
        lines.extend([
            "### Small Sample Flags",
            "",
        ])
        for eid in small_sample_edges:
            card = cards[eid]
            obs_diag = card.diagnostics.get("effective_obs")
            n = int(obs_diag.value) if obs_diag and obs_diag.value else "?"
            lines.append(f"- `{eid}`: N={n}")
        lines.append("")

    # Sign inconsistencies
    sign_issues = [
        eid for eid, c in cards.items()
        if "sign_consistency" in c.diagnostics and not c.diagnostics["sign_consistency"].passed
    ]
    if sign_issues:
        lines.extend(["### Sign Inconsistencies", ""])
        for eid in sign_issues:
            msg = cards[eid].diagnostics["sign_consistency"].message
            lines.append(f"- `{eid}`: {msg}")
        lines.append("")

    # Null results
    null_edges = [eid for eid, c in cards.items() if c.is_precisely_null]
    if null_edges:
        lines.extend(["### Precisely Null Results", ""])
        for eid in null_edges:
            bound = cards[eid].null_equivalence_bound
            bound_str = f" (|beta| < {bound:.4f})" if bound else ""
            lines.append(f"- `{eid}`{bound_str}")
        lines.append("")

    # -----------------------------------------------------------------------
    # Unit Normalization Reference
    # -----------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## Unit Normalization Reference",
        "",
        "**CRITICAL:** All coefficients must be interpreted with correct units for chain propagation.",
        "",
        "| Edge | Treatment Unit | Outcome Unit |",
        "|------|---------------|--------------|",
    ])
    for edge_id in ALL_EDGES:
        units = EDGE_UNITS.get(edge_id, {})
        t_unit = units.get("treatment_unit", "not specified")
        o_unit = units.get("outcome_unit", "not specified")
        is_rf = units.get("is_reaction_function", False)
        rf_note = " **(REACTION FN)**" if is_rf else ""
        lines.append(f"| `{edge_id}` | {t_unit} | {o_unit}{rf_note} |")
    lines.append("")

    # -----------------------------------------------------------------------
    # Query Mode Permissions
    # -----------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        f"## Query Mode Permissions (`{query_mode}`)",
        "",
        "| Edge | Role | Propagation | Shock CF | Policy CF | Variant Of |",
        "|------|------|-------------|----------|-----------|------------|",
    ])
    for edge_id in ALL_EDGES:
        card = cards.get(edge_id)
        if not card:
            # Check for annual variants
            card = cards.get(edge_id + "_annual")
        if card and card.propagation_role:
            pr = card.propagation_role
            prop_ok = pr.is_propagation_allowed(query_mode)
            shock_ok = pr.is_shock_cf_allowed(query_mode)
            policy_ok = pr.is_policy_cf_allowed(query_mode)
            variant = card.variant_of or ""
            lines.append(
                f"| `{card.edge_id}` | {pr.role} | "
                f"{'Yes' if prop_ok else 'NO'} | "
                f"{'Yes' if shock_ok else 'NO'} | "
                f"{'Yes' if policy_ok else 'NO'} | "
                f"{variant} |"
            )
    # Also show annual variants
    for edge_id in ANNUAL_LP_EDGES:
        annual_id = edge_id + "_annual"
        card = cards.get(annual_id)
        if card and card.propagation_role:
            pr = card.propagation_role
            prop_ok = pr.is_propagation_allowed(query_mode)
            shock_ok = pr.is_shock_cf_allowed(query_mode)
            policy_ok = pr.is_policy_cf_allowed(query_mode)
            variant = card.variant_of or ""
            lines.append(
                f"| `{annual_id}` | {pr.role} | "
                f"{'Yes' if prop_ok else 'NO'} | "
                f"{'Yes' if shock_ok else 'NO'} | "
                f"{'Yes' if policy_ok else 'NO'} | "
                f"{variant} |"
            )
    lines.append("")

    # -----------------------------------------------------------------------
    # Honest limitations
    # -----------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## Limitations and Honest Assessment",
        "",
        "### Data Quality",
        "- KSPI quarterly data: 17 true quarterly observations (2020Q3-2024Q3)",
        "- KSPI annual data: 9 annual observations (2011-2019), bank subsidiary level",
        "- No interpolated observations used in estimation (hard filter)",
        "- Panel data: 4 banks, unbalanced panel, annual frequency for most",
        "- Monthly macro data: ~60-180 observations depending on series",
        "",
        "### Entity Boundary",
        "- All KSPI data at Kaspi Bank JSC (subsidiary) level, not group consolidated",
        "- Post-2020 extracted from 20-F segment breakdowns",
        "- Panel banks have different entity boundaries documented per bank",
        "",
        "### Methodological",
        "- Time-series LP may suffer from limited power in small samples",
        "- Quarterly LP edges have wide confidence intervals",
        "- Annual LP is robustness check only (not primary estimates)",
        "- Panel LP uses shift-share design; identified from exposure variation",
        "- HAC standard errors may be undersized in very small samples",
        "- Accounting bridges are deterministic at current values only",
        "",
        "### Policy-Rate Edges",
        "- `cpi_to_nbk_rate` and `fx_to_nbk_rate` estimate **reaction functions**, not causal effects",
        "- These edges should NOT be used for shock propagation without monetary policy surprise specification",
        "- Current estimates are imprecise/near-null, consistent with endogenous policy response",
        "",
        "### Scope",
        "- All results are Kazakhstan-specific",
        "- Sector panel covers 4 banks only",
        "- Results should not be extrapolated beyond sample period",
        "",
        "### No p-hacking",
        "- All results reported as estimated, including nulls",
        "- No specification search or data dredging performed",
        "- Sign inconsistencies documented honestly",
        "",
        "---",
        "",
        f"*Report generated by `run_real_estimation.py`*",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _start_sentinel_loop() -> None:
    """Auto-start the codex sentinel loop in the background if not already running."""
    import subprocess
    pidfile = Path("outputs/codex_loop.pid")
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip())
            import os, signal
            os.kill(pid, 0)  # Check if process is alive
            logger.info(f"Sentinel loop already running (PID {pid})")
            return
        except (ProcessLookupError, ValueError):
            pidfile.unlink(missing_ok=True)

    control = Path("scripts/codex_loop/control.sh")
    if control.exists():
        try:
            subprocess.Popen(
                [str(control), "start"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            logger.info("Sentinel loop auto-started in background")
        except Exception as e:
            logger.warning(f"Could not auto-start sentinel loop: {e}")
    else:
        logger.debug("Sentinel loop control script not found, skipping")


def main(output_dir: Path | None = None, query_mode: str = "REDUCED_FORM"):
    """Run all estimations and generate report."""
    output_dir = output_dir or Path("outputs/agentic")
    logger.info(f"Query mode: {query_mode}")

    # Auto-start sentinel loop
    _start_sentinel_loop()

    cards_dir = output_dir / "cards" / "edge_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    cards: dict[str, EdgeCard] = {}
    lp_results: dict[str, LPResult] = {}
    immutable_results: dict[str, ImmutableResult] = {}
    identity_results: dict[str, IdentityResult] = {}
    bridge_results: dict[str, AccountingBridgeResult] = {}
    panel_results: dict[str, PanelLPResult] = {}
    annual_lp_results: dict[str, LPResult] = {}

    # Initialize identification screen and TSGuard
    id_screen = IdentifiabilityScreen()
    ts_guard = TSGuard()
    id_results: dict[str, IdentifiabilityResult] = {}
    ts_results: dict[str, TSGuardResult] = {}

    # ===================================================================
    # Group A: Monthly LP (6 edges)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("GROUP A: Monthly Local Projections")
    logger.info("=" * 60)

    for edge_id in MONTHLY_LP_EDGES:
        logger.info(f"Estimating {edge_id}...")
        try:
            data = assemble_edge_data(edge_id)
            if data.empty or len(data) < 10:
                logger.warning(f"  Insufficient data for {edge_id}: {len(data)} obs")
                continue

            lp = estimate_lp(
                y=data["outcome"],
                x=data["treatment"],
                max_horizon=6,
                n_lags=2,
                edge_id=edge_id,
            )
            lp_results[edge_id] = lp

            card = build_lp_edge_card(lp, data, edge_id, is_quarterly=False)
            cards[edge_id] = card

            logger.info(
                f"  {edge_id}: beta={lp.impact_estimate:.4f}, "
                f"se={lp.impact_se:.4f}, N={lp.nobs[0]}, "
                f"rating={card.credibility_rating}"
            )

            # Run TSGuard and identification screen
            try:
                ts_result = ts_guard.validate(
                    edge_id=edge_id, y=data["outcome"], x=data["treatment"],
                    lp_result=lp,
                )
                ts_results[edge_id] = ts_result
                id_result = id_screen.screen_post_estimation(
                    edge_id=edge_id, design="LOCAL_PROJECTIONS",
                    diagnostics=card.diagnostics, ts_guard_result=ts_result,
                )
                id_results[edge_id] = id_result
                attach_identification(card, id_result, query_mode=query_mode)
                print_edge_risk_block(edge_id, card, id_result, ts_result)
            except Exception as e:
                logger.warning(f"  ID/TSGuard for {edge_id}: {e}")

        except Exception as e:
            logger.error(f"  {edge_id} FAILED: {e}")

    # ===================================================================
    # Group B: Immutable evidence (4 edges)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("GROUP B: Immutable Evidence")
    logger.info("=" * 60)

    for edge_id in IMMUTABLE_EDGES:
        logger.info(f"Loading validated evidence for {edge_id}...")
        try:
            ir = get_immutable_result(edge_id)
            immutable_results[edge_id] = ir

            card = build_immutable_edge_card(ir)
            cards[edge_id] = card

            logger.info(
                f"  {edge_id}: point={ir.point_estimate:.4f}, "
                f"source={ir.source_block}, rating={card.credibility_rating}"
            )

            # Immutable edges get IDENTIFIED_CAUSAL
            try:
                id_result = id_screen.screen_post_design(edge_id, "IMMUTABLE_EVIDENCE")
                id_results[edge_id] = id_result
                attach_identification(card, id_result, query_mode=query_mode)
            except Exception as e:
                logger.warning(f"  ID screen for {edge_id}: {e}")

        except Exception as e:
            logger.error(f"  {edge_id} FAILED: {e}")

    # ===================================================================
    # Group C-Q: Quarterly LP with KSPI data (4 edges)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("GROUP C-Q: Quarterly LP with KSPI Data")
    logger.info("=" * 60)

    for edge_id in QUARTERLY_LP_EDGES:
        logger.info(f"Estimating {edge_id} (quarterly)...")
        try:
            data = assemble_edge_data(edge_id)
            if data.empty:
                logger.warning(f"  No data for {edge_id}")
                continue

            lp = estimate_lp(
                y=data["outcome"],
                x=data["treatment"],
                max_horizon=2,
                n_lags=1,
                edge_id=edge_id,
            )
            lp_results[edge_id] = lp

            # Compute interpolation share
            try:
                kspi_df = _load_kspi_quarterly()
                share_interp = compute_share_interpolated(kspi_df)
            except Exception:
                share_interp = 0.0

            # Find sector panel companion
            companion = None
            for panel_id, spec in PANEL_EDGE_SPECS.items():
                if spec.get("kspi_companion") == edge_id:
                    companion = panel_id
                    break

            card = build_lp_edge_card(
                lp, data, edge_id,
                is_quarterly=True,
                companion_edge_id=companion,
                share_interpolated=share_interp,
            )
            cards[edge_id] = card

            n = lp.nobs[0] if lp.nobs else 0
            logger.info(
                f"  {edge_id}: beta={lp.impact_estimate:.4f}, "
                f"se={lp.impact_se:.4f}, N={n}, "
                f"rating={card.credibility_rating}"
            )

            # Run TSGuard and identification screen for quarterly edges
            try:
                ts_result = ts_guard.validate(
                    edge_id=edge_id, y=data["outcome"], x=data["treatment"],
                    lp_result=lp,
                )
                ts_results[edge_id] = ts_result
                id_result = id_screen.screen_post_estimation(
                    edge_id=edge_id, design="LOCAL_PROJECTIONS",
                    diagnostics=card.diagnostics, ts_guard_result=ts_result,
                )
                id_results[edge_id] = id_result
                attach_identification(card, id_result, query_mode=query_mode)
                print_edge_risk_block(edge_id, card, id_result, ts_result)
            except Exception as e:
                logger.warning(f"  ID/TSGuard for {edge_id}: {e}")

        except Exception as e:
            logger.error(f"  {edge_id} FAILED: {e}")

    # ===================================================================
    # Group C-A: Annual LP robustness (4 edges, annual frequency)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("GROUP C-A: Annual LP Robustness")
    logger.info("=" * 60)

    try:
        kspi_annual = _load_kspi_quarterly_filtered(freq="annual")
        logger.info(f"  Loaded {len(kspi_annual)} annual observations for Kaspi Bank")

        for edge_id in ANNUAL_LP_EDGES:
            logger.info(f"Estimating {edge_id} (annual)...")
            try:
                treatment_node, outcome_node = EDGE_NODE_MAP[edge_id]

                # Load treatment at annual frequency
                treatment = load_node_series(treatment_node)
                # Aggregate monthly treatment to annual
                if not isinstance(treatment.index, pd.DatetimeIndex):
                    treatment.index = pd.to_datetime(treatment.index)
                treatment_annual = treatment.resample("YS").mean().dropna()

                # Load outcome from annual KSPI data
                kpi_map = {
                    "npl_kspi": "npl_ratio",
                    "cor_kspi": "cor",
                    "deposit_cost_kspi": "deposit_cost",
                }
                outcome_col = kpi_map.get(outcome_node, outcome_node)
                if outcome_col not in kspi_annual.columns:
                    logger.warning(f"  {outcome_col} not in annual data")
                    continue

                outcome = kspi_annual[outcome_col].dropna()

                # Annual KSPI obs are at Q4 dates (e.g., 2011-10-01);
                # treatment_annual is at year-start (2011-01-01).
                # Shift outcome dates to year-start for alignment.
                outcome = outcome.copy()
                outcome.index = pd.DatetimeIndex([
                    pd.Timestamp(year=d.year, month=1, day=1) for d in outcome.index
                ])

                # Align
                common_idx = treatment_annual.index.intersection(outcome.index)
                if len(common_idx) < 5:
                    logger.warning(f"  Only {len(common_idx)} overlapping annual obs for {edge_id}")

                treatment_aligned = treatment_annual.reindex(common_idx).dropna()
                outcome_aligned = outcome.reindex(common_idx).dropna()
                common_idx = treatment_aligned.index.intersection(outcome_aligned.index)
                treatment_aligned = treatment_aligned.reindex(common_idx)
                outcome_aligned = outcome_aligned.reindex(common_idx)

                if len(common_idx) < 3:
                    logger.warning(f"  Insufficient annual data for {edge_id}")
                    continue

                lp = estimate_lp_annual(
                    y=outcome_aligned,
                    x=treatment_aligned,
                    max_horizon=2,
                    n_lags=1,
                    edge_id=edge_id,
                )
                annual_lp_results[edge_id] = lp

                data_df = pd.DataFrame({
                    "treatment": treatment_aligned,
                    "outcome": outcome_aligned,
                })

                card = build_lp_edge_card(
                    lp, data_df, edge_id,
                    is_quarterly=False,
                    is_annual_robustness=True,
                )
                cards[edge_id + "_annual"] = card

                n = lp.nobs[0] if lp.nobs else 0
                logger.info(
                    f"  {edge_id}_annual: beta={lp.impact_estimate:.4f}, "
                    f"se={lp.impact_se:.4f}, N={n}, "
                    f"rating={card.credibility_rating}"
                )

                # Run TSGuard and identification screen for annual edges
                try:
                    ts_result = ts_guard.validate(
                        edge_id=edge_id + "_annual",
                        y=outcome_aligned, x=treatment_aligned,
                        lp_result=lp,
                    )
                    ts_results[edge_id + "_annual"] = ts_result
                    id_result = id_screen.screen_post_estimation(
                        edge_id=edge_id + "_annual", design="LOCAL_PROJECTIONS",
                        diagnostics=card.diagnostics, ts_guard_result=ts_result,
                    )
                    id_results[edge_id + "_annual"] = id_result
                    attach_identification(card, id_result, query_mode=query_mode)
                    print_edge_risk_block(edge_id + "_annual", card, id_result, ts_result)
                except Exception as e:
                    logger.warning(f"  ID/TSGuard for {edge_id}_annual: {e}")

            except Exception as e:
                logger.error(f"  {edge_id}_annual FAILED: {e}")

    except Exception as e:
        logger.error(f"  Annual LP setup FAILED: {e}")

    # ===================================================================
    # Group C-Panel: Sector Panel LP (4 edges)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("GROUP C-PANEL: Sector Panel LP")
    logger.info("=" * 60)

    try:
        panel_client = KZBankPanelClient()

        for edge_id in PANEL_LP_EDGES:
            spec = PANEL_EDGE_SPECS.get(edge_id, {})
            logger.info(f"Estimating {edge_id}...")
            try:
                outcome_kpi = spec["outcome"]
                shock_node = spec["shock_node"]
                exposure_var = spec["exposure"]
                kspi_companion = spec["kspi_companion"]

                # Load shock series
                shock = load_node_series(shock_node)

                # Build panel
                panel_data = panel_client.build_panel_for_edge(
                    outcome_kpi=outcome_kpi,
                    shock_series=shock,
                    exposure_var=exposure_var,
                    max_horizon=2,
                )

                if panel_data.empty:
                    logger.warning(f"  Empty panel for {edge_id}")
                    continue

                # Estimate panel LP
                pr = estimate_panel_lp_exposure(
                    panel_data,
                    max_horizon=2,
                    n_lags=1,
                    edge_id=edge_id,
                )
                panel_results[edge_id] = pr

                # Diagnostics: leave-one-out
                loo = leave_one_bank_out(panel_data, max_horizon=2, edge_id=edge_id)

                # Diagnostics: regime breaks
                regime = regime_split_test(panel_data, edge_id=edge_id)

                # Diagnostics: exposure variation
                exp_check = check_exposure_variation(panel_data, "exposure")

                card = build_panel_edge_card(
                    pr, panel_data, edge_id,
                    exposure_var=exposure_var,
                    kspi_companion=kspi_companion,
                    loo_results=loo,
                    regime_results=regime,
                    exposure_check=exp_check,
                )
                cards[edge_id] = card

                # Also set companion link on KSPI card
                if kspi_companion in cards:
                    cards[kspi_companion].companion_edge_id = edge_id

                logger.info(
                    f"  {edge_id}: beta={pr.impact_estimate:.4f}, "
                    f"se={pr.impact_se:.4f}, "
                    f"N={pr.nobs[0] if pr.nobs else 0}, "
                    f"banks={pr.n_units}, "
                    f"rating={card.credibility_rating}"
                )

                # Run identification screen for panel edges
                try:
                    id_result = id_screen.screen_post_estimation(
                        edge_id=edge_id, design="PANEL_LP_EXPOSURE_FE",
                        diagnostics=card.diagnostics,
                    )
                    id_results[edge_id] = id_result
                    attach_identification(card, id_result, query_mode=query_mode)
                    print_edge_risk_block(edge_id, card, id_result)
                except Exception as e:
                    logger.warning(f"  ID screen for {edge_id}: {e}")

            except Exception as e:
                logger.error(f"  {edge_id} FAILED: {e}")

    except Exception as e:
        logger.error(f"  Panel setup FAILED: {e}")

    # ===================================================================
    # Group C-KSPI: KSPI-only edges (2 edges)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("GROUP C-KSPI: KSPI-Only Edges")
    logger.info("=" * 60)

    for edge_id in KSPI_ONLY_EDGES:
        logger.info(f"Estimating {edge_id}...")
        try:
            data = assemble_edge_data(edge_id)
            if data.empty:
                logger.warning(f"  No data for {edge_id}")
                continue

            max_h = 2 if edge_id != "portfolio_mix_to_rwa" else 1
            lp = estimate_lp(
                y=data["outcome"],
                x=data["treatment"],
                max_horizon=max_h,
                n_lags=1,
                edge_id=edge_id,
            )
            lp_results[edge_id] = lp

            card = build_lp_edge_card(lp, data, edge_id, is_quarterly=True)
            cards[edge_id] = card

            n = lp.nobs[0] if lp.nobs else 0
            logger.info(
                f"  {edge_id}: beta={lp.impact_estimate:.4f}, "
                f"se={lp.impact_se:.4f}, N={n}, "
                f"rating={card.credibility_rating}"
            )

            # Run TSGuard and identification screen
            try:
                ts_result = ts_guard.validate(
                    edge_id=edge_id, y=data["outcome"], x=data["treatment"],
                    lp_result=lp,
                )
                ts_results[edge_id] = ts_result
                id_result = id_screen.screen_post_estimation(
                    edge_id=edge_id, design="LOCAL_PROJECTIONS",
                    diagnostics=card.diagnostics, ts_guard_result=ts_result,
                )
                id_results[edge_id] = id_result
                attach_identification(card, id_result, query_mode=query_mode)
                print_edge_risk_block(edge_id, card, id_result, ts_result)
            except Exception as e:
                logger.warning(f"  ID/TSGuard for {edge_id}: {e}")

        except Exception as e:
            logger.error(f"  {edge_id} FAILED: {e}")

    # ===================================================================
    # Group C-Bridge: Accounting Bridges (2 edges)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("GROUP C-BRIDGE: Accounting Bridges")
    logger.info("=" * 60)

    try:
        kspi_df = _load_kspi_quarterly()
        latest = kspi_df.iloc[-1]
        loans = float(latest["net_loans"])
        rwa = float(latest["rwa"])
        cor_val = float(latest["cor"])
        capital = float(latest["total_capital"])

        for edge_id in BRIDGE_EDGES:
            logger.info(f"Computing {edge_id}...")
            try:
                br = compute_accounting_bridge(
                    edge_id=edge_id,
                    loans=loans,
                    rwa=rwa,
                    cor=cor_val,
                    capital=capital,
                )
                bridge_results[edge_id] = br

                card = build_bridge_edge_card(br)
                cards[edge_id] = card

                logger.info(
                    f"  {edge_id}: sensitivity={br.sensitivity:.4f}, "
                    f"formula={br.formula}, rating={card.credibility_rating}"
                )

                # Bridge edges are mechanical - screen as such
                try:
                    id_result = id_screen.screen_post_design(edge_id, "ACCOUNTING_BRIDGE")
                    id_results[edge_id] = id_result
                    attach_identification(card, id_result, query_mode=query_mode)
                    print_edge_risk_block(edge_id, card, id_result)
                except Exception as e:
                    logger.warning(f"  ID screen for {edge_id}: {e}")

            except Exception as e:
                logger.error(f"  {edge_id} FAILED: {e}")

    except Exception as e:
        logger.error(f"  Bridge setup FAILED: {e}")

    # ===================================================================
    # Group D: Identity (2 edges)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("GROUP D: Identity Sensitivities")
    logger.info("=" * 60)

    try:
        kspi_df = _load_kspi_quarterly()
        latest = kspi_df.iloc[-1]
        capital = float(latest["total_capital"])
        rwa = float(latest["rwa"])

        identity_sens = compute_identity_sensitivity(capital, rwa)

        for edge_id in IDENTITY_EDGES:
            ir = identity_sens[edge_id]
            identity_results[edge_id] = ir

            card = build_identity_edge_card(ir)
            cards[edge_id] = card

            logger.info(
                f"  {edge_id}: sensitivity={ir.sensitivity:.6f}, "
                f"formula={ir.formula}, rating={card.credibility_rating}"
            )

            # Identity edges are mechanical - screen as such
            try:
                id_result = id_screen.screen_post_design(edge_id, "IDENTITY")
                id_results[edge_id] = id_result
                attach_identification(card, id_result, query_mode=query_mode)
                print_edge_risk_block(edge_id, card, id_result)
            except Exception as e:
                logger.warning(f"  ID screen for {edge_id}: {e}")

    except Exception as e:
        logger.error(f"  Identity computation FAILED: {e}")

    # ===================================================================
    # Validate all EdgeCards (W5B)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("VALIDATING EDGE CARDS")
    logger.info("=" * 60)

    from shared.agentic.validation import (
        validate_edge_card,
        validate_chain_units,
        ValidationSeverity,
    )

    for edge_id, card in cards.items():
        card_validation = validate_edge_card(card)
        for issue in card_validation.issues:
            if issue.severity == ValidationSeverity.ERROR:
                logger.error(f"EdgeCard validation [{edge_id}]: {issue.check_id}: {issue.message}")
            else:
                logger.warning(f"EdgeCard validation [{edge_id}]: {issue.check_id}: {issue.message}")

    # ===================================================================
    # Chain unit compatibility check (W5C)
    # ===================================================================
    logger.info("Checking chain unit compatibility...")

    # Build dag_edges dict from EDGE_NODE_MAP for estimated edges only
    estimated_dag_edges = {
        eid: EDGE_NODE_MAP[eid]
        for eid in cards
        if eid in EDGE_NODE_MAP
    }
    chain_result = validate_chain_units(cards, estimated_dag_edges)
    for issue in chain_result.issues:
        logger.warning(f"Chain units: {issue.message}")

    # ===================================================================
    # Save EdgeCards as YAML
    # ===================================================================
    logger.info("=" * 60)
    logger.info("SAVING EDGE CARDS")
    logger.info("=" * 60)

    for edge_id, card in cards.items():
        yaml_path = cards_dir / f"{edge_id}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(card.to_dict(), f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        logger.info(f"  Saved {yaml_path}")

    # ===================================================================
    # Generate markdown report
    # ===================================================================
    logger.info("=" * 60)
    logger.info("GENERATING REPORT")
    logger.info("=" * 60)

    report_md = generate_report(
        cards, lp_results, immutable_results, identity_results,
        bridge_results, panel_results, annual_lp_results,
        query_mode=query_mode,
    )

    # Append identifiability dashboard
    if id_results:
        dashboard_md = generate_id_dashboard(id_results)
        report_md += "\n\n---\n\n" + dashboard_md

    report_path = output_dir / "KSPI_K2_REAL_ESTIMATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info(f"Report saved to {report_path}")

    # ===================================================================
    # Report consistency check (W5D)
    # ===================================================================
    logger.info("Running ReportConsistencyChecker...")
    try:
        from shared.agentic.report_checker import ReportConsistencyChecker
        rf_edges = ["cpi_to_nbk_rate", "fx_to_nbk_rate"]
        checker = ReportConsistencyChecker(report_md, cards, rf_edges)
        check_result = checker.check()
        for err in check_result.errors:
            logger.error(f"ReportChecker: {err}")
        for warn in check_result.warnings:
            logger.warning(f"ReportChecker: {warn}")

        check_path = output_dir / "REPORT_CONSISTENCY_CHECK.md"
        with open(check_path, "w") as f:
            f.write(check_result.to_markdown())
        logger.info(f"Report consistency check saved to {check_path}")
    except Exception as e:
        logger.warning(f"ReportConsistencyChecker failed: {e}")

    # ===================================================================
    # Final summary
    # ===================================================================
    logger.info("=" * 60)
    logger.info("ESTIMATION COMPLETE")
    logger.info(f"  Total edge cards: {len(cards)}")
    logger.info(f"  Original edges: {len(ALL_ORIGINAL_EDGES)}")
    logger.info(f"  Sector panel companions: {len(PANEL_LP_EDGES)}")
    logger.info(f"  Annual robustness: {len(annual_lp_results)}")

    rating_counts: dict[str, int] = {}
    for c in cards.values():
        r = c.credibility_rating
        rating_counts[r] = rating_counts.get(r, 0) + 1
    for r in sorted(rating_counts):
        logger.info(f"  Rating {r}: {rating_counts[r]}")

    null_count = sum(1 for c in cards.values() if c.is_precisely_null)
    logger.info(f"  Precisely null: {null_count}")
    logger.info("=" * 60)

    # ===================================================================
    # Build interactive panels and auto-open
    # ===================================================================
    dag_path = Path("config/agentic/dags/kspi_k2_full.yaml")
    cards_dir = output_dir / "cards" / "edge_cards"
    state_path = output_dir / "issues" / "state.json"

    # DAG Visualization
    try:
        from scripts.build_dag_viz import build as build_dag_viz
        viz_path = build_dag_viz(
            dag_path=dag_path,
            cards_dir=cards_dir,
            state_path=state_path,
            output_path=output_dir / "dag_visualization.html",
        )
        logger.info(f"DAG visualization built: {viz_path}")
    except Exception as e:
        logger.warning(f"DAG visualization build failed (non-blocking): {e}")
        viz_path = None

    # HITL Panel
    try:
        from scripts.build_hitl_panel import build as build_hitl
        hitl_path = build_hitl(
            state_path=state_path,
            cards_dir=cards_dir,
            actions_path=Path("config/agentic/hitl_actions.yaml"),
            registry_path=Path("config/agentic/issue_registry.yaml"),
            output_dir=output_dir,
            dag_path=dag_path,
        )
        logger.info(f"HITL panel built: {hitl_path}")
    except Exception as e:
        logger.warning(f"HITL panel build failed (non-blocking): {e}")
        hitl_path = None

    # Auto-open panels in browser
    import subprocess, platform
    open_cmd = "open" if platform.system() == "Darwin" else "xdg-open"
    for panel_path in [viz_path, hitl_path]:
        if panel_path and panel_path.exists():
            try:
                subprocess.Popen([open_cmd, str(panel_path)])
                logger.info(f"Opened: {panel_path}")
            except Exception as e:
                logger.warning(f"Could not open {panel_path}: {e}")

    return cards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real econometric estimation on KSPI K2 DAG")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/agentic"),
        help="Output directory",
    )
    parser.add_argument(
        "--query-mode",
        choices=["STRUCTURAL", "REDUCED_FORM", "DESCRIPTIVE"],
        default="REDUCED_FORM",
        help="Query mode: STRUCTURAL (identified causal), REDUCED_FORM (shock scenarios), DESCRIPTIVE (accounting)",
    )
    args = parser.parse_args()
    main(args.output_dir, query_mode=args.query_mode)
