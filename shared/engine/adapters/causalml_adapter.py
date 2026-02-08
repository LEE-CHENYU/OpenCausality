"""
CausalML Uplift Adapter.

S/T/X-learner meta-learners for uplift estimation via CausalML.
Binary treatment only. Returns conditional uplift (CATE) as ATE estimate.
"""

from __future__ import annotations

import numpy as np

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class CausalMLUpliftAdapter(EstimatorAdapter):
    """Adapter for CausalML uplift/meta-learner estimation.

    Supports ``req.extra["learner"]``:
        - ``"s_learner"`` (default): Single-model approach.
        - ``"t_learner"``: Two-model approach (separate models for treated/control).
        - ``"x_learner"``: Cross-learner (propensity-weighted combination).

    Requires binary treatment (0/1).
    """

    def supported_designs(self) -> list[str]:
        return ["CAUSALML_UPLIFT"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        unique_vals = req.df[req.treatment].dropna().unique()
        if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            errors.append(
                f"CausalMLUpliftAdapter requires binary treatment (0/1), "
                f"got unique values: {sorted(unique_vals)}"
            )
        if not req.controls:
            errors.append("CausalMLUpliftAdapter requires at least one control variable.")
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        import causalml

        learner_name = req.extra.get("learner", "s_learner")

        cols = [req.outcome, req.treatment] + req.controls
        df = req.df[cols].dropna()

        Y = df[req.outcome].values
        T = df[req.treatment].values.astype(int)
        X = df[req.controls].values

        # Select meta-learner
        if learner_name == "t_learner":
            from causalml.inference.meta import BaseTRegressor
            from sklearn.linear_model import LinearRegression
            model = BaseTRegressor(learner=LinearRegression())
        elif learner_name == "x_learner":
            from causalml.inference.meta import BaseXRegressor
            from sklearn.linear_model import LinearRegression
            model = BaseXRegressor(learner=LinearRegression())
        else:
            from causalml.inference.meta import BaseSRegressor
            from sklearn.linear_model import LinearRegression
            model = BaseSRegressor(learner=LinearRegression())

        # Estimate individual treatment effects
        cate = model.fit_predict(X=X, treatment=T, y=Y)
        cate_flat = np.asarray(cate).flatten()

        # ATE as mean of individual effects
        ate = float(np.mean(cate_flat))
        se = float(np.std(cate_flat, ddof=1) / np.sqrt(len(cate_flat)))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        return EstimationResult(
            point=ate,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            pvalue=None,
            n_obs=len(df),
            method_name=f"CausalML_{learner_name}",
            library="causalml",
            library_version=causalml.__version__,
            diagnostics={
                "cate_mean": float(np.mean(cate_flat)),
                "cate_std": float(np.std(cate_flat)),
                "cate_p10": float(np.percentile(cate_flat, 10)),
                "cate_p90": float(np.percentile(cate_flat, 90)),
                "n_treated": int(T.sum()),
                "n_control": int(len(T) - T.sum()),
            },
            metadata={
                "learner": learner_name,
                "controls": req.controls,
            },
        )
