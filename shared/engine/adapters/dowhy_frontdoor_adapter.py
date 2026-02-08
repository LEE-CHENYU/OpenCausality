"""
DoWhy Frontdoor Adapter.

Frontdoor criterion estimation via DoWhy. Requires a known mediator
through which the treatment fully affects the outcome.
"""

from __future__ import annotations

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class DoWhyFrontdoorAdapter(EstimatorAdapter):
    """Adapter for DoWhy frontdoor criterion estimation.

    Requires:
        - ``req.extra["graph_gml"]``: GML string of the causal graph.
        - ``req.extra["mediators"]``: list of mediator column names.

    The causal graph must encode the frontdoor path:
        Treatment -> Mediator -> Outcome
    with no direct Treatment -> Outcome edge and unobserved confounding
    only between Treatment and Outcome.
    """

    def supported_designs(self) -> list[str]:
        return ["DOWHY_FRONTDOOR"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        if "graph_gml" not in req.extra:
            errors.append("DoWhyFrontdoorAdapter requires req.extra['graph_gml'].")
        mediators = req.extra.get("mediators", [])
        if not mediators:
            errors.append("DoWhyFrontdoorAdapter requires req.extra['mediators'].")
        else:
            for m in mediators:
                if m not in req.df.columns:
                    errors.append(f"Mediator column '{m}' not in DataFrame")
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        import dowhy
        from dowhy import CausalModel

        graph_gml = req.extra["graph_gml"]
        mediators = req.extra["mediators"]

        cols = [req.outcome, req.treatment] + mediators + req.controls
        df = req.df.dropna(subset=cols)

        model = CausalModel(
            data=df,
            treatment=req.treatment,
            outcome=req.outcome,
            graph=graph_gml,
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Use two-stage linear regression for frontdoor estimation
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="frontdoor.two_stage_regression",
        )

        point = float(estimate.value)

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
                    se = (ci_upper - ci_lower) / 3.92
            except Exception:
                pass

        return EstimationResult(
            point=point,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            pvalue=pvalue,
            n_obs=len(df),
            method_name="DoWhy_Frontdoor",
            library="dowhy",
            library_version=dowhy.__version__,
            diagnostics={
                "estimand_type": str(identified_estimand.estimand_type) if identified_estimand else "",
                "mediators": mediators,
            },
            metadata={
                "mediators": mediators,
                "controls": req.controls,
            },
        )
