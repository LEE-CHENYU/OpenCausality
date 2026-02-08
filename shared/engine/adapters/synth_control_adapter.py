"""
Synthetic Control Adapter.

Implements the Abadie, Diamond & Hainmueller (2010) synthetic control method.
Uses scipy.optimize for donor weight optimization with constraints (w >= 0, sum = 1).
Includes Fisher permutation test (placebo-in-space) for inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class SynthControlAdapter(EstimatorAdapter):
    """Synthetic Control Method adapter.

    Expected DataFrame format:
        - Panel data with ``req.unit`` and ``req.time`` columns.
        - ``req.outcome``: outcome variable.
        - ``req.treatment``: binary indicator (1 for treated unit-periods).

    Required ``req.extra``:
        - ``"treated_unit"``: ID of the treated unit.
        - ``"treatment_period"``: First period of treatment.

    Optional ``req.extra``:
        - ``"n_placebo"``: Number of placebo permutations (default 20).
    """

    def supported_designs(self) -> list[str]:
        return ["SYNTHETIC_CONTROL"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = super().validate_request(req)
        if not req.unit:
            errors.append("SynthControlAdapter requires req.unit (unit identifier column).")
        if not req.time:
            errors.append("SynthControlAdapter requires req.time (time column).")
        if "treated_unit" not in req.extra:
            errors.append("SynthControlAdapter requires req.extra['treated_unit'].")
        if "treatment_period" not in req.extra:
            errors.append("SynthControlAdapter requires req.extra['treatment_period'].")
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        import scipy

        treated_unit = req.extra["treated_unit"]
        treatment_period = req.extra["treatment_period"]
        n_placebo = req.extra.get("n_placebo", 20)

        df = req.df.dropna(subset=[req.outcome, req.unit, req.time])

        # Pivot to wide format: rows=time, columns=units
        wide = df.pivot(index=req.time, columns=req.unit, values=req.outcome)

        if treated_unit not in wide.columns:
            raise ValueError(f"Treated unit '{treated_unit}' not found in data.")

        # Split donor pool
        donor_units = [c for c in wide.columns if c != treated_unit]
        if len(donor_units) < 2:
            raise ValueError(f"Need at least 2 donor units, got {len(donor_units)}.")

        # Pre/post split
        pre_mask = wide.index < treatment_period
        post_mask = wide.index >= treatment_period

        y_treated_pre = wide.loc[pre_mask, treated_unit].values
        y_donors_pre = wide.loc[pre_mask, donor_units].values
        y_treated_post = wide.loc[post_mask, treated_unit].values
        y_donors_post = wide.loc[post_mask, donor_units].values

        # Optimize donor weights
        weights = self._optimize_weights(y_treated_pre, y_donors_pre)

        # Synthetic control outcomes
        synth_pre = y_donors_pre @ weights
        synth_post = y_donors_post @ weights

        # Treatment effect (ATT): average gap in post-period
        gaps_post = y_treated_post - synth_post
        att = float(np.mean(gaps_post))

        # Pre-treatment RMSPE
        gaps_pre = y_treated_pre - synth_pre
        pre_rmspe = float(np.sqrt(np.mean(gaps_pre ** 2)))

        # Post-pre RMSPE ratio (measure of effect size)
        post_rmspe = float(np.sqrt(np.mean(gaps_post ** 2)))
        post_pre_ratio = post_rmspe / max(pre_rmspe, 1e-10)

        # Fisher permutation test (placebo-in-space)
        fisher_pvalue = self._fisher_permutation_test(
            wide, donor_units, treatment_period, att, pre_rmspe,
            n_placebo=min(n_placebo, len(donor_units)),
        )

        # Approximate SE from gap variation
        se = float(np.std(gaps_post, ddof=1)) if len(gaps_post) > 1 else float("nan")
        ci_lower = att - 1.96 * se
        ci_upper = att + 1.96 * se

        # Build donor weights dict
        donor_weights = {
            str(unit): float(w)
            for unit, w in zip(donor_units, weights)
            if w > 0.001
        }

        return EstimationResult(
            point=att,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            pvalue=fisher_pvalue,
            n_obs=len(df),
            method_name="Synthetic_Control",
            library="scipy",
            library_version=scipy.__version__,
            diagnostics={
                "pre_rmspe": pre_rmspe,
                "post_rmspe": post_rmspe,
                "post_pre_ratio": post_pre_ratio,
                "fisher_pvalue": fisher_pvalue,
                "n_donors": len(donor_units),
                "n_pre_periods": int(pre_mask.sum()),
                "n_post_periods": int(post_mask.sum()),
            },
            metadata={
                "treated_unit": treated_unit,
                "treatment_period": treatment_period,
                "donor_weights": donor_weights,
            },
        )

    @staticmethod
    def _optimize_weights(
        y_treated: np.ndarray,
        y_donors: np.ndarray,
    ) -> np.ndarray:
        """Find optimal donor weights minimizing pre-treatment RMSPE.

        Constraints: w >= 0, sum(w) = 1.
        """
        n_donors = y_donors.shape[1]
        w0 = np.ones(n_donors) / n_donors

        def objective(w):
            synth = y_donors @ w
            return np.sum((y_treated - synth) ** 2)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n_donors

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        return result.x

    @staticmethod
    def _fisher_permutation_test(
        wide: pd.DataFrame,
        donor_units: list,
        treatment_period: int | float,
        observed_att: float,
        observed_pre_rmspe: float,
        n_placebo: int = 20,
    ) -> float | None:
        """Fisher permutation test: assign treatment to each donor and compare effects.

        Returns the fraction of placebo effects as extreme as the observed effect,
        weighted by pre-treatment fit quality (post/pre RMSPE ratio).
        """
        pre_mask = wide.index < treatment_period
        post_mask = wide.index >= treatment_period

        ratios = []

        for placebo_unit in donor_units[:n_placebo]:
            other_donors = [u for u in donor_units if u != placebo_unit]
            if len(other_donors) < 2:
                continue

            y_placebo_pre = wide.loc[pre_mask, placebo_unit].values
            y_others_pre = wide.loc[pre_mask, other_donors].values

            try:
                w = SynthControlAdapter._optimize_weights(y_placebo_pre, y_others_pre)
            except Exception:
                continue

            synth_pre = y_others_pre @ w
            synth_post = wide.loc[post_mask, other_donors].values @ w

            gaps_pre = y_placebo_pre - synth_pre
            gaps_post = wide.loc[post_mask, placebo_unit].values - synth_post

            pre_rmspe_placebo = float(np.sqrt(np.mean(gaps_pre ** 2)))
            post_rmspe_placebo = float(np.sqrt(np.mean(gaps_post ** 2)))
            ratio = post_rmspe_placebo / max(pre_rmspe_placebo, 1e-10)
            ratios.append(ratio)

        if not ratios:
            return None

        # Observed ratio
        observed_post_rmspe = float(np.sqrt(observed_att ** 2))
        observed_ratio = observed_post_rmspe / max(observed_pre_rmspe, 1e-10)

        # p-value: fraction of placebo ratios >= observed
        n_extreme = sum(1 for r in ratios if r >= observed_ratio)
        pvalue = (n_extreme + 1) / (len(ratios) + 1)  # +1 for the treated unit itself

        return float(pvalue)
