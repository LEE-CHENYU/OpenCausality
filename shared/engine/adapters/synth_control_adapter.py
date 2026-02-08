"""
Synthetic Control Adapter (Stub).

Placeholder that raises NotImplementedError until a full
synthetic control implementation is added.
"""

from __future__ import annotations

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class SynthControlAdapter(EstimatorAdapter):
    """Stub adapter for Synthetic Control Method.

    Raises NotImplementedError on estimation.  The design is registered
    so that get_adapter("SYNTHETIC_CONTROL") resolves and DAG edges
    referencing it receive a clear error rather than a missing-adapter crash.
    """

    def supported_designs(self) -> list[str]:
        return ["SYNTHETIC_CONTROL"]

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        raise NotImplementedError(
            "Synthetic Control requires a panel donor pool and is not yet "
            "implemented in the adapter layer.  Use the design_templates.synthetic_control "
            "specification in design_registry.yaml for documentation."
        )
