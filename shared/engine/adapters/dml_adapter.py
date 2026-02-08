"""
DoubleML Adapter.

Wraps DoubleML's PLR, IRM, and PLIV models into the EstimatorAdapter interface.
Supports cross-fitting with configurable ML learners.
"""

from __future__ import annotations

import numpy as np

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class DMLAdapter(EstimatorAdapter):
    """Adapter for DoubleML estimation.

    Supports three DML models via ``req.extra["dml_model"]``:
        - ``"PLR"`` (default): Partially Linear Regression — continuous treatment.
        - ``"IRM"``: Interactive Regression Model — binary treatment.
        - ``"PLIV"``: Partially Linear IV Model — requires instruments.

    Optional ``req.extra["ml_learner"]``:
        - ``"lasso"`` (default): Lasso regularization.
        - ``"random_forest"``: Random forest learners.

    Optional ``req.extra["n_folds"]``: Number of cross-fitting folds (default 5).
    Optional ``req.extra["n_rep"]``: Number of repeated cross-fitting (default 1).
    """

    def supported_designs(self) -> list[str]:
        return ["DML_PLR", "DML_IRM", "DML_PLIV"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        dml_model = req.extra.get("dml_model", "PLR")
        if dml_model == "PLIV" and not req.instruments:
            errors.append("DML_PLIV requires at least one instrument in req.instruments.")
        if dml_model == "IRM":
            unique_vals = req.df[req.treatment].dropna().unique()
            if len(unique_vals) > 2:
                errors.append(
                    f"DML_IRM requires binary treatment, got {len(unique_vals)} unique values."
                )
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        import doubleml as dml
        from sklearn.linear_model import LassoCV
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        dml_model = req.extra.get("dml_model", "PLR")
        ml_learner = req.extra.get("ml_learner", "lasso")
        n_folds = req.extra.get("n_folds", 5)
        n_rep = req.extra.get("n_rep", 1)

        # Build DoubleML data object
        cols = [req.outcome, req.treatment] + req.controls
        if dml_model == "PLIV" and req.instruments:
            cols += req.instruments
        df = req.df[cols].dropna()

        x_cols = req.controls if req.controls else None

        dml_data = dml.DoubleMLData(
            df,
            y_col=req.outcome,
            d_cols=req.treatment,
            x_cols=x_cols,
            z_cols=req.instruments[0] if dml_model == "PLIV" and req.instruments else None,
        )

        # Select ML learners
        if ml_learner == "random_forest":
            ml_l = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            ml_m = (
                RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                if dml_model == "IRM"
                else RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            )
        else:
            ml_l = LassoCV(cv=5)
            ml_m = (
                # For IRM, the propensity score model must be a classifier
                # Use logistic regression via sklearn
                self._get_logistic_learner() if dml_model == "IRM" else LassoCV(cv=5)
            )

        # Build DML model
        if dml_model == "PLR":
            dml_obj = dml.DoubleMLPLR(
                dml_data, ml_l=ml_l, ml_m=ml_m,
                n_folds=n_folds, n_rep=n_rep,
            )
        elif dml_model == "IRM":
            dml_obj = dml.DoubleMLIRM(
                dml_data, ml_g=ml_l, ml_m=ml_m,
                n_folds=n_folds, n_rep=n_rep,
            )
        elif dml_model == "PLIV":
            ml_r = LassoCV(cv=5)
            dml_obj = dml.DoubleMLPLIV(
                dml_data, ml_l=ml_l, ml_m=ml_m, ml_r=ml_r,
                n_folds=n_folds, n_rep=n_rep,
            )
        else:
            raise ValueError(f"Unknown DML model: {dml_model}")

        # Fit
        dml_obj.fit()

        # Extract results
        point = float(dml_obj.coef[0])
        se = float(dml_obj.se[0])
        ci = dml_obj.confint(level=0.95)
        ci_lower = float(ci.iloc[0, 0])
        ci_upper = float(ci.iloc[0, 1])
        pvalue = float(dml_obj.pval[0])

        return EstimationResult(
            point=point,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            pvalue=pvalue,
            n_obs=len(df),
            method_name=f"DML_{dml_model}",
            library="doubleml",
            library_version=dml.__version__,
            diagnostics={
                "n_folds": n_folds,
                "n_rep": n_rep,
                "ml_learner": ml_learner,
                "t_stat": float(dml_obj.t_stat[0]),
            },
            metadata={
                "dml_model": dml_model,
                "controls": req.controls,
            },
        )

    @staticmethod
    def _get_logistic_learner():
        """Return a logistic regression learner for propensity score estimation."""
        from sklearn.linear_model import LogisticRegressionCV
        return LogisticRegressionCV(cv=5, max_iter=1000)
