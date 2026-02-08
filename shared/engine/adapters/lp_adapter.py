"""
Local Projections Adapter.

Wraps shared.engine.ts_estimator.estimate_lp() into the
EstimatorAdapter interface.
"""

from __future__ import annotations

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class LPAdapter(EstimatorAdapter):
    """Adapter for Jorda (2005) Local Projections (time-series)."""

    def supported_designs(self) -> list[str]:
        return ["LOCAL_PROJECTIONS"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        if req.unit is not None:
            errors.append(
                "LPAdapter is for time-series data. "
                "For panel data, use PanelLPAdapter."
            )
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        """Run LP estimation, returning impact (h=0) result.

        If req.horizon is set, estimates at that horizon.
        Full IRF coefficients are stored in metadata.
        """
        import statsmodels

        from shared.engine.ts_estimator import estimate_lp

        max_horizon = 6
        if isinstance(req.horizon, int) and req.horizon > 0:
            max_horizon = req.horizon
        elif isinstance(req.horizon, list) and req.horizon:
            max_horizon = max(req.horizon)

        n_lags = req.extra.get("n_lags", 2)

        controls_df = None
        if req.controls:
            controls_df = req.df[req.controls]

        lp_result = estimate_lp(
            y=req.df[req.outcome],
            x=req.df[req.treatment],
            max_horizon=max_horizon,
            n_lags=n_lags,
            controls=controls_df,
            edge_id=req.edge_id,
        )

        # Build diagnostics dict
        diagnostics = {}
        if lp_result.residual_autocorrelation:
            diagnostics["residual_autocorrelation"] = lp_result.residual_autocorrelation
        if lp_result.hac_bandwidth:
            diagnostics["hac_bandwidth"] = lp_result.hac_bandwidth

        return EstimationResult(
            point=lp_result.impact_estimate,
            se=lp_result.impact_se,
            ci_lower=lp_result.ci_lower[0] if lp_result.ci_lower else lp_result.impact_estimate - 1.96 * lp_result.impact_se,
            ci_upper=lp_result.ci_upper[0] if lp_result.ci_upper else lp_result.impact_estimate + 1.96 * lp_result.impact_se,
            pvalue=lp_result.pvalues[0] if lp_result.pvalues else None,
            n_obs=lp_result.nobs[0] if lp_result.nobs else 0,
            method_name="LOCAL_PROJECTIONS",
            library="statsmodels",
            library_version=statsmodels.__version__,
            diagnostics=diagnostics,
            metadata={
                "horizons": lp_result.horizons,
                "coefficients": lp_result.coefficients,
                "std_errors": lp_result.std_errors,
                "pvalues": lp_result.pvalues,
                "cumulative_estimate": lp_result.cumulative_estimate,
            },
        )
