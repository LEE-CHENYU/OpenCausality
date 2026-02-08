"""
DoWhy IV Adapter.

Graph-aware instrumental variables via DoWhy's identification + estimation.
Complements the existing IV2SLSAdapter by adding DoWhy's identification
verification against the causal graph.
"""

from __future__ import annotations

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class DoWhyIVAdapter(EstimatorAdapter):
    """Adapter for DoWhy graph-aware IV estimation.

    Requires:
        - ``req.extra["graph_gml"]``: GML string of the causal graph.
        - ``req.instruments``: list of instrument column names.
    """

    def supported_designs(self) -> list[str]:
        return ["DOWHY_IV"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        if "graph_gml" not in req.extra:
            errors.append("DoWhyIVAdapter requires req.extra['graph_gml'].")
        if not req.instruments:
            errors.append("DoWhyIVAdapter requires at least one instrument in req.instruments.")
        else:
            for inst in req.instruments:
                if inst not in req.df.columns:
                    errors.append(f"Instrument column '{inst}' not in DataFrame")
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        import dowhy
        from dowhy import CausalModel

        graph_gml = req.extra["graph_gml"]
        cols = [req.outcome, req.treatment] + (req.instruments or []) + req.controls
        df = req.df.dropna(subset=cols)

        model = CausalModel(
            data=df,
            treatment=req.treatment,
            outcome=req.outcome,
            graph=graph_gml,
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        estimate = model.estimate_effect(
            identified_estimand,
            method_name="iv.instrumental_variable",
            method_params={
                "iv_instrument_name": req.instruments[0],
            },
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
            method_name="DoWhy_IV",
            library="dowhy",
            library_version=dowhy.__version__,
            diagnostics={
                "estimand_type": str(identified_estimand.estimand_type) if identified_estimand else "",
                "iv_identified": identified_estimand is not None,
            },
            metadata={
                "instruments": req.instruments,
                "controls": req.controls,
            },
        )
