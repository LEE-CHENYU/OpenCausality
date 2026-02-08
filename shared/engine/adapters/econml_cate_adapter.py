"""
EconML CATE Adapter.

Estimates Conditional Average Treatment Effects (CATE) using EconML.
Returns ATE as the point estimate plus CATE heterogeneity diagnostics.
"""

from __future__ import annotations

import numpy as np

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class EconMLCATEAdapter(EstimatorAdapter):
    """Adapter for EconML heterogeneous treatment effect estimation.

    Supports ``req.extra["cate_model"]``:
        - ``"linear_dml"`` (default): LinearDML for linear CATE.
        - ``"causal_forest"``: CausalForestDML for nonlinear CATE.

    Returns ATE as the point estimate and CATE heterogeneity diagnostics:
        - ``diagnostics["cate_mean"]``: mean CATE across observations.
        - ``diagnostics["cate_std"]``: std of CATE.
        - ``diagnostics["cate_p10"]``: 10th percentile.
        - ``diagnostics["cate_p90"]``: 90th percentile.
        - ``diagnostics["heterogeneity_detected"]``: True if p90/p10 spread is large.
    """

    def supported_designs(self) -> list[str]:
        return ["ECONML_CATE"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        if not req.controls:
            errors.append(
                "EconMLCATEAdapter requires at least one control variable "
                "to estimate heterogeneous effects over."
            )
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        import econml

        cate_model_name = req.extra.get("cate_model", "linear_dml")

        cols = [req.outcome, req.treatment] + req.controls
        df = req.df[cols].dropna()

        Y = df[req.outcome].values
        T = df[req.treatment].values
        X = df[req.controls].values

        # Build the CATE model
        if cate_model_name == "causal_forest":
            from econml.dml import CausalForestDML
            model = CausalForestDML(
                n_estimators=200,
                max_depth=5,
                random_state=42,
            )
        else:
            from econml.dml import LinearDML
            model = LinearDML(
                random_state=42,
            )

        model.fit(Y=Y, T=T, X=X)

        # ATE as main point estimate
        ate = float(model.ate(X=X))

        # CATE for each observation
        cate = model.effect(X=X)
        cate_flat = np.asarray(cate).flatten()

        cate_mean = float(np.mean(cate_flat))
        cate_std = float(np.std(cate_flat))
        cate_p10 = float(np.percentile(cate_flat, 10))
        cate_p90 = float(np.percentile(cate_flat, 90))

        # Heterogeneity detection: if the spread is > 50% of the mean effect
        heterogeneity_detected = (
            (cate_p90 - cate_p10) > 0.5 * max(abs(cate_mean), 1e-10)
        )

        # Inference
        ci_lower = float("nan")
        ci_upper = float("nan")
        se = float("nan")

        try:
            ate_inference = model.ate_inference(X=X)
            ci = ate_inference.conf_int()
            ci_lower = float(ci[0][0])
            ci_upper = float(ci[1][0])
            se = float(ate_inference.std_err)
        except Exception:
            se = cate_std / np.sqrt(len(df))
            ci_lower = ate - 1.96 * se
            ci_upper = ate + 1.96 * se

        pvalue = None
        try:
            ate_inf = model.ate_inference(X=X)
            pvalue = float(ate_inf.pvalue())
        except Exception:
            pass

        return EstimationResult(
            point=ate,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            pvalue=pvalue,
            n_obs=len(df),
            method_name=f"EconML_{cate_model_name}",
            library="econml",
            library_version=econml.__version__,
            diagnostics={
                "cate_mean": cate_mean,
                "cate_std": cate_std,
                "cate_p10": cate_p10,
                "cate_p90": cate_p90,
                "heterogeneity_detected": heterogeneity_detected,
            },
            metadata={
                "cate_model": cate_model_name,
                "controls": req.controls,
                "n_controls": len(req.controls),
            },
        )
