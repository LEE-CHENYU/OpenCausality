"""
RDD Adapter.

Wraps FuzzyRDDEstimator and a simple sharp RD implementation
into the EstimatorAdapter interface.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class RDDAdapter(EstimatorAdapter):
    """Adapter for Regression Discontinuity Design estimation.

    Supports both sharp and fuzzy RD.  The request must supply
    ``extra["running_variable"]`` (column name), and
    ``extra["cutoff"]`` (numeric threshold).  For fuzzy RD, also
    provide the treatment column in ``req.treatment``.

    Sharp RD is implemented inline via local-linear regression.
    Fuzzy RD delegates to ``FuzzyRDDEstimator`` when panel data
    with the required columns is supplied.
    """

    def supported_designs(self) -> list[str]:
        return ["RDD"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        if "running_variable" not in req.extra:
            errors.append("RDDAdapter requires extra['running_variable']")
        elif req.extra["running_variable"] not in req.df.columns:
            errors.append(
                f"Running variable '{req.extra['running_variable']}' not in DataFrame"
            )
        if "cutoff" not in req.extra:
            errors.append("RDDAdapter requires extra['cutoff']")
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        """Run RDD estimation (sharp or fuzzy).

        For sharp RD: local-linear regression around the cutoff.
        For fuzzy RD: Wald estimator (reduced-form / first-stage).
        """
        running_var = req.extra["running_variable"]
        cutoff = float(req.extra["cutoff"])
        bandwidth = req.extra.get("bandwidth")
        kernel = req.extra.get("kernel", "triangular")

        df = req.df.dropna(subset=[req.outcome, running_var]).copy()
        df["_rv_centered"] = df[running_var] - cutoff

        # Auto bandwidth if not specified (Silverman rule-of-thumb)
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

        # Kernel weights
        df["_weight"] = self._kernel_weights(df["_rv_centered"], bandwidth, kernel)
        df["_above"] = (df["_rv_centered"] >= 0).astype(float)

        # Determine sharp vs fuzzy.
        # Sharp if treatment is perfectly determined by the running variable
        # (i.e. treatment == above_cutoff indicator), OR if treatment is not
        # supplied / not in columns.
        is_fuzzy = False
        if (req.treatment in df.columns
                and req.treatment != "_above"
                and df[req.treatment].nunique() > 1):
            # Check if treatment is perfectly collinear with _above
            if not (df[req.treatment] == df["_above"]).all():
                is_fuzzy = True

        diagnostics: dict = {}

        if is_fuzzy:
            point, se, pvalue, ci_lo, ci_hi, diag = self._fuzzy_rd(
                df, req.outcome, req.treatment, bandwidth, kernel,
            )
            diagnostics.update(diag)
            method = "FUZZY_RDD"
        else:
            point, se, pvalue, ci_lo, ci_hi = self._sharp_rd(
                df, req.outcome,
            )
            method = "SHARP_RDD"

        # McCrary density test (lightweight version)
        n_above = int((df["_above"] == 1).sum())
        n_below = int((df["_above"] == 0).sum())
        ratio = n_above / max(n_below, 1)
        mccrary_pass = 0.5 <= ratio <= 2.0
        diagnostics["density_ratio"] = ratio
        diagnostics["density_test_pass"] = mccrary_pass

        # Covariate balance (if controls provided)
        if req.controls:
            balance_pass = self._balance_check(df, req.controls)
            diagnostics["covariate_balance_pass"] = balance_pass

        return EstimationResult(
            point=point,
            se=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            pvalue=pvalue,
            n_obs=len(df),
            method_name=method,
            library="statsmodels",
            library_version=sm.__version__ if hasattr(sm, "__version__") else "unknown",
            diagnostics=diagnostics,
            metadata={
                "bandwidth": bandwidth,
                "cutoff": cutoff,
                "kernel": kernel,
                "n_above": n_above,
                "n_below": n_below,
            },
        )

    # ------------------------------------------------------------------
    # Internal estimation helpers
    # ------------------------------------------------------------------

    def _sharp_rd(self, df, outcome: str):
        """Sharp RD via local-linear WLS."""
        X = sm.add_constant(df[["_above", "_rv_centered"]])
        model = sm.WLS(df[outcome], X, weights=df["_weight"]).fit(
            cov_type="HC1",
        )
        point = float(model.params["_above"])
        se = float(model.bse["_above"])
        pval = float(model.pvalues["_above"])
        ci = model.conf_int().loc["_above"]
        return point, se, pval, float(ci.iloc[0]), float(ci.iloc[1])

    def _fuzzy_rd(self, df, outcome: str, treatment: str, bandwidth: float, kernel: str):
        """Fuzzy RD via Wald estimator (reduced-form / first-stage)."""
        X = sm.add_constant(df[["_above", "_rv_centered"]])
        w = df["_weight"]

        # First stage: treatment ~ above + running_var
        fs = sm.WLS(df[treatment], X, weights=w).fit(cov_type="HC1")
        fs_coef = float(fs.params["_above"])
        fs_f = float(fs.tvalues["_above"] ** 2)

        # Reduced form: outcome ~ above + running_var
        rf = sm.WLS(df[outcome], X, weights=w).fit(cov_type="HC1")
        rf_coef = float(rf.params["_above"])
        rf_se = float(rf.bse["_above"])

        # Wald: beta = rf / fs
        if abs(fs_coef) < 1e-12:
            raise ValueError("First stage is zero — fuzzy RD not identified")
        point = rf_coef / fs_coef
        se = rf_se / abs(fs_coef)
        from scipy import stats as sp_stats
        t_stat = point / se if se > 0 else 0.0
        dof = max(len(df) - 3, 1)
        pval = float(2 * (1 - sp_stats.t.cdf(abs(t_stat), dof)))
        ci_lo = point - 1.96 * se
        ci_hi = point + 1.96 * se

        diagnostics = {
            "first_stage_f": fs_f,
            "first_stage_coef": fs_coef,
            "weak_instrument": fs_f < 10,
        }
        return point, se, pval, ci_lo, ci_hi, diagnostics

    @staticmethod
    def _kernel_weights(rv: "pd.Series", bw: float, kernel: str) -> "pd.Series":
        u = rv / bw
        if kernel == "triangular":
            return np.maximum(0, 1 - np.abs(u))
        elif kernel == "epanechnikov":
            return np.maximum(0, 0.75 * (1 - u ** 2))
        else:  # uniform
            return (np.abs(u) <= 1).astype(float)

    @staticmethod
    def _balance_check(df, controls: list[str]) -> bool:
        """Simple balance check: t-test for each control above vs below cutoff."""
        from scipy.stats import ttest_ind
        all_pass = True
        above = df[df["_above"] == 1]
        below = df[df["_above"] == 0]
        for c in controls:
            if c not in df.columns:
                continue
            _, p = ttest_ind(
                above[c].dropna(), below[c].dropna(), equal_var=False,
            )
            if p < 0.05:
                all_pass = False
        return all_pass
