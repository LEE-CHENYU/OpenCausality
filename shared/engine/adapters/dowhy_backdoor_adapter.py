"""
DoWhy Backdoor Adapter.

Uses DoWhy's CausalModel with backdoor identification and estimation.
Supports linear regression, propensity score weighting, and matching.
"""

from __future__ import annotations

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class DoWhyBackdoorAdapter(EstimatorAdapter):
    """Adapter for DoWhy backdoor adjustment estimation.

    Requires ``req.extra["graph_gml"]``: a GML string describing the causal graph.
    Use ``dagspec_to_dowhy_graph()`` from ``shared.agentic.dag.graph_convert``.

    Optional ``req.extra["dowhy_method"]``:
        - ``"backdoor.linear_regression"`` (default)
        - ``"backdoor.propensity_score_weighting"``
        - ``"backdoor.propensity_score_matching"``
    """

    def supported_designs(self) -> list[str]:
        return ["DOWHY_BACKDOOR"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        if "graph_gml" not in req.extra:
            errors.append(
                "DoWhyBackdoorAdapter requires req.extra['graph_gml']. "
                "Use dagspec_to_dowhy_graph() to generate it."
            )
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        import dowhy
        from dowhy import CausalModel

        graph_gml = req.extra["graph_gml"]
        method = req.extra.get("dowhy_method", "backdoor.linear_regression")

        df = req.df.dropna(subset=[req.outcome, req.treatment] + req.controls)

        model = CausalModel(
            data=df,
            treatment=req.treatment,
            outcome=req.outcome,
            graph=graph_gml,
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        estimate = model.estimate_effect(
            identified_estimand,
            method_name=method,
        )

        point = float(estimate.value)

        # DoWhy doesn't always provide SE/CI directly; bootstrap for confidence intervals
        ci_lower = float("nan")
        ci_upper = float("nan")
        se = float("nan")
        pvalue = None

        if hasattr(estimate, "get_confidence_intervals"):
            try:
                ci = estimate.get_confidence_intervals()
                if ci is not None and len(ci) == 2:
                    ci_lower = float(ci[0])
                    ci_upper = float(ci[1])
                    se = (ci_upper - ci_lower) / 3.92  # approximate from 95% CI
            except Exception:
                pass

        if hasattr(estimate, "test_stat_significance"):
            try:
                sig = estimate.test_stat_significance()
                if sig and "p_value" in sig:
                    pvalue = float(sig["p_value"])
            except Exception:
                pass

        return EstimationResult(
            point=point,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            pvalue=pvalue,
            n_obs=len(df),
            method_name=f"DoWhy_{method}",
            library="dowhy",
            library_version=dowhy.__version__,
            diagnostics={
                "estimand_type": str(identified_estimand.estimand_type) if identified_estimand else "",
                "backdoor_variables": (
                    list(identified_estimand.get_backdoor_variables())
                    if identified_estimand and hasattr(identified_estimand, "get_backdoor_variables")
                    else []
                ),
            },
            metadata={
                "dowhy_method": method,
                "controls": req.controls,
            },
        )
