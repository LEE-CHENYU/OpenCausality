"""
DID Event Study Adapter.

Wraps linearmodels.panel.PanelOLS for two-way fixed effects
difference-in-differences into the EstimatorAdapter interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class DIDEventStudyAdapter(EstimatorAdapter):
    """Adapter for Difference-in-Differences with two-way fixed effects."""

    def supported_designs(self) -> list[str]:
        return ["DID_EVENT_STUDY"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        if req.unit is None:
            errors.append("DIDEventStudyAdapter requires 'unit' for entity FE")
        if req.time is None:
            errors.append("DIDEventStudyAdapter requires 'time' for time FE")
        if "treatment" not in req.extra and req.treatment not in req.df.columns:
            errors.append(
                "DIDEventStudyAdapter requires a treatment interaction column "
                "(treated × post) as the treatment variable"
            )
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        """Run TWFE DID estimation.

        The treatment variable should be the interaction (treated_i × post_t).
        Entity and time fixed effects are included via PanelOLS.
        Standard errors are clustered at the entity level.
        """
        import linearmodels
        from linearmodels.panel import PanelOLS

        df = req.df.copy()

        # Set up multi-index for panel
        df = df.set_index([req.unit, req.time]) if not isinstance(df.index, pd.MultiIndex) else df

        dependent = df[req.outcome]
        exog_cols = [req.treatment] + req.controls
        exog = df[exog_cols]

        model = PanelOLS(dependent, exog, entity_effects=True, time_effects=True)
        se_type = req.extra.get("cluster_entity", True)
        if se_type:
            result = model.fit(cov_type="clustered", cluster_entity=True)
        else:
            result = model.fit(cov_type="robust")

        point = float(result.params[req.treatment])
        se = float(result.std_errors[req.treatment])
        pvalue = float(result.pvalues[req.treatment])
        ci = result.conf_int().loc[req.treatment]
        ci_lower = float(ci.iloc[0])
        ci_upper = float(ci.iloc[1])
        n_obs = int(result.nobs)

        # Diagnostics
        diagnostics: dict = {}

        # Pre-trend test: if pre-treatment lead columns are provided
        pre_trend_cols = req.extra.get("pre_trend_cols", [])
        if pre_trend_cols:
            pre_pvalues = []
            for col in pre_trend_cols:
                if col in result.params.index:
                    pre_pvalues.append(float(result.pvalues[col]))
            if pre_pvalues:
                # Joint test: all pre-treatment leads should be insignificant
                diagnostics["pre_trend_pvalues"] = pre_pvalues
                diagnostics["pre_trend_joint_pass"] = all(p > 0.05 for p in pre_pvalues)

        diagnostics["entity_effects"] = True
        diagnostics["time_effects"] = True
        diagnostics["r_squared_within"] = float(result.rsquared_within) if hasattr(result, "rsquared_within") else None

        return EstimationResult(
            point=point,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            pvalue=pvalue,
            n_obs=n_obs,
            method_name="DID_EVENT_STUDY",
            library="linearmodels",
            library_version=linearmodels.__version__,
            diagnostics=diagnostics,
            metadata={
                "entity_var": req.unit,
                "time_var": req.time,
                "controls": req.controls,
            },
        )
