"""
Identity Sensitivity Adapter.

Wraps shared.engine.ts_estimator.compute_identity_sensitivity() into the
EstimatorAdapter interface.
"""

from __future__ import annotations

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class IdentityAdapter(EstimatorAdapter):
    """Adapter for mechanical identity edges (K2 = Capital / RWA)."""

    def supported_designs(self) -> list[str]:
        return ["IDENTITY"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = []
        if "capital" not in req.extra and "capital" not in req.df.columns:
            errors.append("IdentityAdapter requires 'capital' in extra or DataFrame")
        if "rwa" not in req.extra and "rwa" not in req.df.columns:
            errors.append("IdentityAdapter requires 'rwa' in extra or DataFrame")
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        """Compute K2 identity sensitivity and return as EstimationResult."""
        from shared.engine.ts_estimator import compute_identity_sensitivity

        capital = req.extra.get("capital", 0.0)
        rwa = req.extra.get("rwa", 0.0)

        results = compute_identity_sensitivity(capital, rwa)
        ir = results[req.edge_id]

        return EstimationResult(
            point=ir.sensitivity,
            se=0.0,
            ci_lower=ir.sensitivity,
            ci_upper=ir.sensitivity,
            pvalue=None,
            n_obs=1,
            method_name="IDENTITY",
            library="deterministic",
            library_version="1.0",
            diagnostics={"formula": ir.formula},
            metadata={
                "at_values": ir.at_values,
                "numerator_label": ir.numerator_label,
                "denominator_label": ir.denominator_label,
            },
        )
