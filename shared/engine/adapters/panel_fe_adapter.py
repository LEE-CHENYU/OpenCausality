"""
Panel Fixed-Effects Backdoor Adapter.

Uses linearmodels.PanelOLS with entity and time fixed effects,
clustered standard errors by entity. Designed for panel data
where identification relies on within-unit variation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class PanelFEBackdoorAdapter(EstimatorAdapter):
    """Adapter for Panel Fixed-Effects Backdoor estimation.

    Requires panel data with ``req.unit`` and ``req.time`` columns.
    Uses entity + time fixed effects and clusters standard errors
    by entity.
    """

    def supported_designs(self) -> list[str]:
        return ["PANEL_FE_BACKDOOR"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        if not req.unit:
            errors.append("PanelFEBackdoorAdapter requires req.unit (entity column)")
        elif req.unit not in req.df.columns:
            errors.append(f"Unit column '{req.unit}' not in DataFrame")
        if not req.time:
            errors.append("PanelFEBackdoorAdapter requires req.time (time column)")
        elif req.time not in req.df.columns:
            errors.append(f"Time column '{req.time}' not in DataFrame")
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        """Run panel FE estimation with entity + time effects."""
        from linearmodels.panel import PanelOLS

        df = req.df.copy()

        # Set multi-index for panel structure
        df = df.set_index([req.unit, req.time])

        # Build formula
        exog_cols = [req.treatment] + req.controls
        y = df[req.outcome]
        X = df[exog_cols]

        # Estimate with entity + time effects, clustered SEs
        model = PanelOLS(y, X, entity_effects=True, time_effects=True)
        result = model.fit(cov_type="clustered", cluster_entity=True)

        point = float(result.params[req.treatment])
        se = float(result.std_errors[req.treatment])
        pval = float(result.pvalues[req.treatment])
        ci = result.conf_int().loc[req.treatment]
        ci_lo = float(ci.iloc[0])
        ci_hi = float(ci.iloc[1])

        # Diagnostics
        diagnostics: dict = {
            "r2_within": float(result.rsquared_within),
            "r2_between": float(result.rsquared_between),
            "r2_overall": float(result.rsquared_overall),
            "n_entities": int(result.entity_info["total"]),
        }

        return EstimationResult(
            point=point,
            se=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            pvalue=pval,
            n_obs=int(result.nobs),
            method_name="PANEL_FE_BACKDOOR",
            library="linearmodels",
            library_version=self._get_linearmodels_version(),
            diagnostics=diagnostics,
            metadata={
                "entity_effects": True,
                "time_effects": True,
                "cov_type": "clustered",
            },
        )

    @staticmethod
    def _get_linearmodels_version() -> str:
        try:
            import linearmodels
            return getattr(linearmodels, "__version__", "unknown")
        except Exception:
            return "unknown"
