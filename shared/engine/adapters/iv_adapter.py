"""
IV 2SLS Adapter.

Wraps linearmodels.iv.IV2SLS into the EstimatorAdapter interface.
"""

from __future__ import annotations

import numpy as np

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class IV2SLSAdapter(EstimatorAdapter):
    """Adapter for Two-Stage Least Squares instrumental variables estimation."""

    def supported_designs(self) -> list[str]:
        return ["IV_2SLS"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        if not req.instruments:
            errors.append("IV2SLSAdapter requires at least one instrument in req.instruments")
        else:
            for inst in req.instruments:
                if inst not in req.df.columns:
                    errors.append(f"Instrument column '{inst}' not in DataFrame")
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        """Run IV 2SLS estimation.

        Uses linearmodels.iv.IV2SLS. The treatment variable is endogenous,
        instrumented by req.instruments.
        """
        import linearmodels
        from linearmodels.iv import IV2SLS

        df = req.df.dropna(subset=[req.outcome, req.treatment] + (req.instruments or []) + req.controls)

        dependent = df[req.outcome]
        endog = df[[req.treatment]]
        instruments = df[req.instruments]

        # Exogenous regressors: constant + controls
        if req.controls:
            import statsmodels.api as sm
            exog = sm.add_constant(df[req.controls])
        else:
            import statsmodels.api as sm
            exog = sm.add_constant(df[[req.treatment]].iloc[:, :0])  # constant only

        model = IV2SLS(dependent, exog, endog, instruments)
        result = model.fit(cov_type="robust")

        # Extract treatment coefficient
        point = float(result.params[req.treatment])
        se = float(result.std_errors[req.treatment])
        pvalue = float(result.pvalues[req.treatment])
        ci = result.conf_int().loc[req.treatment]
        ci_lower = float(ci.iloc[0])
        ci_upper = float(ci.iloc[1])
        n_obs = int(result.nobs)

        # First-stage diagnostics
        diagnostics: dict = {}
        try:
            first_stage = result.first_stage
            if hasattr(first_stage, "diagnostics"):
                fs_diag = first_stage.diagnostics
                # Extract F-stat for endogenous variable
                if hasattr(fs_diag, "iloc"):
                    f_stat = float(fs_diag.iloc[0]["f.stat"])
                    diagnostics["first_stage_f"] = f_stat
                    diagnostics["weak_instrument"] = f_stat < 10
        except Exception:
            pass

        # Try Sargan test for overidentification
        if req.instruments and len(req.instruments) > 1:
            try:
                sargan = result.sargan
                diagnostics["sargan_stat"] = float(sargan.stat)
                diagnostics["sargan_pvalue"] = float(sargan.pval)
            except Exception:
                pass

        return EstimationResult(
            point=point,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            pvalue=pvalue,
            n_obs=n_obs,
            method_name="IV_2SLS",
            library="linearmodels",
            library_version=linearmodels.__version__,
            diagnostics=diagnostics,
            metadata={
                "instruments": req.instruments,
                "r_squared": float(result.rsquared) if hasattr(result, "rsquared") else None,
            },
        )
