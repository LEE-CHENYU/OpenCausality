"""
Regression Kink Design Adapter.

Estimates change in slope at a known kink point using local-linear
WLS with triangular kernel weights. Analogous to RDD but identifies
off a slope change rather than a level jump.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class RegressionKinkAdapter(EstimatorAdapter):
    """Adapter for Regression Kink Design estimation.

    The request must supply ``extra["running_variable"]`` (column name)
    and ``extra["kink_point"]`` (numeric threshold where the slope changes).

    Model: y = a + b1*(x - c) + b2*(x - c)*above + e
    The kink estimate is b2 (change in slope at the kink).
    """

    def supported_designs(self) -> list[str]:
        return ["REGRESSION_KINK"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        if "running_variable" not in req.extra:
            errors.append("RegressionKinkAdapter requires extra['running_variable']")
        elif req.extra["running_variable"] not in req.df.columns:
            errors.append(
                f"Running variable '{req.extra['running_variable']}' not in DataFrame"
            )
        if "kink_point" not in req.extra:
            errors.append("RegressionKinkAdapter requires extra['kink_point']")
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        """Run regression kink estimation."""
        running_var = req.extra["running_variable"]
        kink = float(req.extra["kink_point"])
        bandwidth = req.extra.get("bandwidth")

        df = req.df.dropna(subset=[req.outcome, running_var]).copy()
        df["_rv_centered"] = df[running_var] - kink

        # Auto bandwidth (Silverman rule-of-thumb) if not specified
        if bandwidth is None:
            std = df["_rv_centered"].std()
            n = len(df)
            bandwidth = 2.702 * std * (n ** (-1 / 5)) if n > 0 else 1.0
            bandwidth = max(0.5, bandwidth)

        # Restrict to bandwidth
        df = df[df["_rv_centered"].abs() <= bandwidth].copy()
        if len(df) < 10:
            raise ValueError(
                f"Too few observations ({len(df)}) within bandwidth {bandwidth}"
            )

        # Triangular kernel weights
        df["_weight"] = np.maximum(0, 1 - np.abs(df["_rv_centered"] / bandwidth))
        df["_above"] = (df["_rv_centered"] >= 0).astype(float)
        df["_interaction"] = df["_rv_centered"] * df["_above"]

        # Model: y = a + b1*(x-c) + b2*(x-c)*above + e
        X = sm.add_constant(df[["_rv_centered", "_interaction"]])
        model = sm.WLS(df[req.outcome], X, weights=df["_weight"]).fit(
            cov_type="HC1",
        )

        point = float(model.params["_interaction"])
        se = float(model.bse["_interaction"])
        pval = float(model.pvalues["_interaction"])
        ci = model.conf_int().loc["_interaction"]
        ci_lo = float(ci.iloc[0])
        ci_hi = float(ci.iloc[1])

        # Density test at kink
        n_above = int((df["_above"] == 1).sum())
        n_below = int((df["_above"] == 0).sum())
        ratio = n_above / max(n_below, 1)
        density_pass = 0.5 <= ratio <= 2.0

        diagnostics: dict = {
            "density_ratio": ratio,
            "density_test_pass": density_pass,
            "bandwidth": bandwidth,
        }

        return EstimationResult(
            point=point,
            se=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            pvalue=pval,
            n_obs=len(df),
            method_name="REGRESSION_KINK",
            library="statsmodels",
            library_version=sm.__version__ if hasattr(sm, "__version__") else "unknown",
            diagnostics=diagnostics,
            metadata={
                "kink_point": kink,
                "bandwidth": bandwidth,
                "n_above": n_above,
                "n_below": n_below,
            },
        )
