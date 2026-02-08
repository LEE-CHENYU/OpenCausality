"""
Accounting Bridge Adapter.

Wraps shared.engine.ts_estimator.compute_accounting_bridge() into the
EstimatorAdapter interface.
"""

from __future__ import annotations

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class AccountingBridgeAdapter(EstimatorAdapter):
    """Adapter for accounting bridge (near-mechanical) edges."""

    def supported_designs(self) -> list[str]:
        return ["ACCOUNTING_BRIDGE"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        errors = []
        required_keys = ["loans", "rwa", "cor", "capital"]
        for key in required_keys:
            if key not in req.extra and key not in req.df.columns:
                errors.append(f"AccountingBridgeAdapter requires '{key}' in extra or DataFrame")
        return errors

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        """Compute accounting bridge sensitivity and return as EstimationResult."""
        from shared.engine.ts_estimator import compute_accounting_bridge

        loans = req.extra.get("loans", 0.0)
        rwa = req.extra.get("rwa", 0.0)
        cor = req.extra.get("cor", 0.0)
        capital = req.extra.get("capital", 0.0)
        tax_rate = req.extra.get("tax_rate", 0.20)

        result = compute_accounting_bridge(
            edge_id=req.edge_id,
            loans=loans,
            rwa=rwa,
            cor=cor,
            capital=capital,
            tax_rate=tax_rate,
        )

        return EstimationResult(
            point=result.sensitivity,
            se=0.0,
            ci_lower=result.sensitivity,
            ci_upper=result.sensitivity,
            pvalue=None,
            n_obs=1,
            method_name="ACCOUNTING_BRIDGE",
            library="deterministic",
            library_version="1.0",
            diagnostics={
                "formula": result.formula,
                "is_deterministic": result.is_deterministic,
            },
            metadata={
                "at_values": result.at_values,
                "description": result.description,
            },
        )
