"""
Panel LP with Exposure x Shock Adapter.

Wraps shared.engine.panel_estimator.estimate_panel_lp_exposure() into the
EstimatorAdapter interface.
"""

from __future__ import annotations

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class PanelLPAdapter(EstimatorAdapter):
    """Adapter for Panel LP with Exposure x Shock (shift-share design)."""

    def supported_designs(self) -> list[str]:
        return ["PANEL_LP_EXPOSURE_FE"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        if req.unit is None:
            errors.append("PanelLPAdapter requires 'unit' column for panel identification")
        if req.time is None:
            errors.append("PanelLPAdapter requires 'time' column for panel identification")
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        """Run panel LP estimation, returning impact (h=0) result."""
        import linearmodels

        from shared.engine.panel_estimator import estimate_panel_lp_exposure

        max_horizon = 2
        if isinstance(req.horizon, int) and req.horizon > 0:
            max_horizon = req.horizon
        elif isinstance(req.horizon, list) and req.horizon:
            max_horizon = max(req.horizon)

        n_lags = req.extra.get("n_lags", 1)
        se_method = req.se_type if req.se_type in ("clustered", "driscoll_kraay") else "clustered"
        use_fd = req.extra.get("use_first_difference", True)

        panel_result = estimate_panel_lp_exposure(
            panel=req.df,
            max_horizon=max_horizon,
            n_lags=n_lags,
            edge_id=req.edge_id,
            se_method=se_method,
            use_first_difference=use_fd,
        )

        diagnostics = {}
        if hasattr(panel_result, "leads_test_passed"):
            diagnostics["leads_test_passed"] = panel_result.leads_test_passed
        if hasattr(panel_result, "loo_results"):
            diagnostics["loo_results"] = panel_result.loo_results

        return EstimationResult(
            point=panel_result.impact_estimate,
            se=panel_result.impact_se,
            ci_lower=panel_result.ci_lower[0] if panel_result.ci_lower else panel_result.impact_estimate - 1.96 * panel_result.impact_se,
            ci_upper=panel_result.ci_upper[0] if panel_result.ci_upper else panel_result.impact_estimate + 1.96 * panel_result.impact_se,
            pvalue=panel_result.pvalues[0] if panel_result.pvalues else None,
            n_obs=panel_result.nobs[0] if panel_result.nobs else 0,
            method_name="PANEL_LP_EXPOSURE_FE",
            library="linearmodels",
            library_version=linearmodels.__version__,
            diagnostics=diagnostics,
            metadata={
                "horizons": panel_result.horizons,
                "coefficients": panel_result.coefficients,
                "std_errors": panel_result.std_errors,
                "pvalues": panel_result.pvalues,
            },
        )
