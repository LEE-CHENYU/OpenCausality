"""
Immutable Evidence Adapter.

Wraps shared.engine.ts_estimator.get_immutable_result() into the
EstimatorAdapter interface.
"""

from __future__ import annotations

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class ImmutableAdapter(EstimatorAdapter):
    """Adapter for validated evidence (immutable) edges."""

    def supported_designs(self) -> list[str]:
        return ["IMMUTABLE_EVIDENCE"]

    def validate_request(self, req: EstimationRequest) -> list[str]:
        # Immutable edges need only the edge_id, not data columns
        return []

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        """Return validated evidence as EstimationResult."""
        from shared.engine.ts_estimator import get_immutable_result

        ir = get_immutable_result(req.edge_id)

        return EstimationResult(
            point=ir.point_estimate,
            se=ir.se,
            ci_lower=ir.ci_lower,
            ci_upper=ir.ci_upper,
            pvalue=ir.pvalue,
            n_obs=ir.nobs or 0,
            method_name="IMMUTABLE_EVIDENCE",
            library="validated_evidence",
            library_version="1.0",
            diagnostics={"source_block": ir.source_block},
            metadata={
                "source_description": ir.source_description,
                "is_precisely_null": ir.point_estimate == 0.0,
            },
        )
