"""
Base scenario simulator class for econometric research.

Provides common infrastructure for scenario simulation across studies.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Results from a scenario simulation."""

    scenario_name: str
    segments: list[str]
    effects: dict[str, np.ndarray]
    aggregate_effect: np.ndarray
    time_periods: list[str]
    parameters_used: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with segment-period effects."""
        records = []
        for segment, effect_array in self.effects.items():
            for i, period in enumerate(self.time_periods):
                records.append({
                    "segment": segment,
                    "period": period,
                    "effect": effect_array[i] if i < len(effect_array) else np.nan,
                })
        return pd.DataFrame(records)

    def summary(self) -> str:
        """Generate text summary of results."""
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"Scenario Results: {self.scenario_name}")
        lines.append(f"{'='*70}")
        lines.append(f"\nPeriods: {self.time_periods[0]} to {self.time_periods[-1]}")
        lines.append(f"Segments: {len(self.segments)}")
        lines.append(f"\nParameters used:")
        for param, value in self.parameters_used.items():
            lines.append(f"  {param}: {value:.4f}")

        # Aggregate effects
        lines.append(f"\nAggregate Effects:")
        lines.append(f"  Peak effect: {np.min(self.aggregate_effect):.4f}")
        lines.append(f"  Cumulative: {np.sum(self.aggregate_effect):.4f}")

        return "\n".join(lines)


class ScenarioBase(ABC):
    """
    Abstract base class for scenario simulators.

    Subclasses should implement:
    - _get_elasticities(): Return estimated elasticities for simulation
    - _apply_shock(): Apply a shock to a segment
    """

    def __init__(self, elasticities: dict[str, float] | None = None):
        """
        Initialize scenario simulator.

        Args:
            elasticities: Dictionary of estimated elasticities by segment
        """
        self.elasticities = elasticities or {}
        self._validation_results: list[dict] = []

    @abstractmethod
    def _get_elasticities(self) -> dict[str, float]:
        """
        Get elasticities for simulation.

        Returns:
            Dictionary mapping segment/group to elasticity
        """
        pass

    @abstractmethod
    def _apply_shock(
        self,
        segment: str,
        shock_size: float,
        elasticity: float,
    ) -> float:
        """
        Apply a shock to a segment.

        Args:
            segment: Segment identifier
            shock_size: Size of shock (e.g., -0.10 for 10% decrease)
            elasticity: Estimated elasticity for this segment

        Returns:
            Predicted effect on outcome
        """
        pass

    def simulate(
        self,
        scenario_name: str,
        shock_path: dict[str, float],
        segments: list[str] | None = None,
    ) -> ScenarioResult:
        """
        Simulate a scenario.

        Args:
            scenario_name: Name for the scenario
            shock_path: Dictionary mapping time period to shock size
            segments: List of segments to simulate (default: all)

        Returns:
            ScenarioResult with simulated effects
        """
        elasticities = self._get_elasticities()
        segments = segments or list(elasticities.keys())
        time_periods = sorted(shock_path.keys())

        effects = {}
        for segment in segments:
            elasticity = elasticities.get(segment, 0.0)
            segment_effects = []

            for period in time_periods:
                shock = shock_path[period]
                effect = self._apply_shock(segment, shock, elasticity)
                segment_effects.append(effect)

            effects[segment] = np.array(segment_effects)

        # Compute aggregate (simple average across segments)
        aggregate_effect = np.mean(
            [effects[s] for s in segments],
            axis=0,
        )

        return ScenarioResult(
            scenario_name=scenario_name,
            segments=segments,
            effects=effects,
            aggregate_effect=aggregate_effect,
            time_periods=time_periods,
            parameters_used=elasticities,
        )

    def validate_extrapolation(
        self,
        segment: str,
        shock_size: float,
    ) -> dict[str, Any]:
        """
        Check if simulation involves extrapolation beyond identification sample.

        Args:
            segment: Segment to check
            shock_size: Proposed shock size

        Returns:
            Dictionary with validation results and warnings
        """
        result = {
            "segment": segment,
            "shock_size": shock_size,
            "warnings": [],
            "valid": True,
        }

        # Check if segment is in identification sample
        elasticities = self._get_elasticities()
        if segment not in elasticities:
            result["warnings"].append(
                f"Segment '{segment}' not in identification sample - extrapolating"
            )
            result["valid"] = False

        # Check shock size reasonableness
        if abs(shock_size) > 0.30:  # 30% shock
            result["warnings"].append(
                f"Large shock ({shock_size:.1%}) may be outside linear response range"
            )

        self._validation_results.append(result)
        return result

    def get_external_validity_caveats(self) -> list[str]:
        """
        Get caveats about external validity of simulation.

        Returns:
            List of caveat strings
        """
        caveats = [
            "Estimated elasticities are local average treatment effects (LATE)",
            "Apply primarily to borrowers/units similar to identification sample",
            "Extrapolation to dissimilar segments requires additional assumptions",
        ]

        # Add segment-specific caveats
        for result in self._validation_results:
            if result["warnings"]:
                caveats.extend(result["warnings"])

        return list(set(caveats))  # Deduplicate
