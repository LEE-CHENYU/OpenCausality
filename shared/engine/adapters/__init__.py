"""Estimator Adapter Framework.

Provides a unified interface for different estimation backends.
"""

from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter
from shared.engine.adapters.registry import get_adapter

__all__ = [
    "EstimationRequest",
    "EstimationResult",
    "EstimatorAdapter",
    "get_adapter",
]
