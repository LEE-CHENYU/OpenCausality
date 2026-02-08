"""
Estimator Adapter Base Classes.

Defines the EstimationRequest/EstimationResult dataclasses and the
EstimatorAdapter ABC that all adapters must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class EstimationRequest:
    """Standardized estimation request.

    Encapsulates all inputs needed to run an estimation, regardless
    of the underlying library or design.
    """

    df: pd.DataFrame
    outcome: str
    treatment: str
    controls: list[str] = field(default_factory=list)
    instruments: list[str] | None = None

    # Panel dimensions
    unit: str | None = None
    time: str | None = None

    # IRF / Local Projections
    horizon: int | list[int] | None = None

    # Standard errors
    se_type: str = "robust"

    # Edge metadata
    edge_id: str = ""

    # Extra adapter-specific parameters
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EstimationResult:
    """Standardized estimation result.

    One result per estimation call. For multi-horizon IRFs,
    the adapter calls estimate() per horizon or returns multiple
    results via the diagnostics/metadata fields.
    """

    point: float
    se: float
    ci_lower: float
    ci_upper: float
    pvalue: float | None
    n_obs: int
    method_name: str
    library: str
    library_version: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class EstimatorAdapter(ABC):
    """Abstract base class for estimation adapters.

    Each adapter wraps a specific estimation library or design
    and translates between EstimationRequest/EstimationResult and
    the library's native API.
    """

    @abstractmethod
    def estimate(self, req: EstimationRequest) -> EstimationResult:
        """Run estimation and return standardized result.

        Args:
            req: Standardized estimation request.

        Returns:
            Standardized estimation result.
        """
        ...

    @abstractmethod
    def supported_designs(self) -> list[str]:
        """Return list of design IDs this adapter supports.

        These should match IDs in design_registry.yaml.
        """
        ...

    def validate_request(self, req: EstimationRequest) -> list[str]:
        """Validate request before estimation. Returns list of error messages.

        Override in subclasses for design-specific validation.
        An empty list means the request is valid.
        """
        errors = []
        if req.outcome not in req.df.columns:
            errors.append(f"Outcome column '{req.outcome}' not in DataFrame")
        if req.treatment not in req.df.columns:
            errors.append(f"Treatment column '{req.treatment}' not in DataFrame")
        for c in req.controls:
            if c not in req.df.columns:
                errors.append(f"Control column '{c}' not in DataFrame")
        return errors
