"""
Scenario simulator for credit default study.

Applies estimated elasticities to income shocks to predict default rate changes.

IMPORTANT: External validity caveats apply - see get_external_validity_caveats()
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from shared.engine.scenario_base import ScenarioBase, ScenarioResult
from studies.credit_default.src.elasticity_store import ElasticityStore, Elasticity

logger = logging.getLogger(__name__)


@dataclass
class CreditScenarioResult:
    """Results from credit default scenario simulation."""

    scenario_name: str

    # Predicted changes
    baseline_default_rate: float
    predicted_default_rate: float
    default_rate_change: float  # Absolute change
    default_rate_pct_change: float  # Relative change

    # By segment
    segment_effects: dict[str, dict[str, float]]

    # Confidence
    conf_int_change: tuple[float, float]

    # Metadata
    income_shock: float  # e.g., -0.10 for 10% decrease
    elasticities_used: dict[str, float]
    external_validity_caveats: list[str]

    def summary(self) -> str:
        """Generate summary."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"CREDIT DEFAULT SCENARIO: {self.scenario_name}")
        lines.append("=" * 60)

        lines.append(f"\nIncome Shock: {self.income_shock*100:+.1f}%")

        lines.append(f"\nDefault Rate Impact:")
        lines.append(f"  Baseline rate: {self.baseline_default_rate*100:.2f}%")
        lines.append(f"  Predicted rate: {self.predicted_default_rate*100:.2f}%")
        lines.append(f"  Change: {self.default_rate_change*100:+.3f} pp")
        lines.append(f"  95% CI: [{self.conf_int_change[0]*100:+.3f}, {self.conf_int_change[1]*100:+.3f}] pp")

        lines.append(f"\nBy Segment:")
        for segment, effects in self.segment_effects.items():
            lines.append(f"  {segment}:")
            lines.append(f"    Elasticity: {effects['elasticity']:.4f}")
            lines.append(f"    Effect: {effects['effect']*100:+.3f} pp")

        lines.append(f"\nExternal Validity Caveats:")
        for caveat in self.external_validity_caveats:
            lines.append(f"  - {caveat}")

        return "\n".join(lines)


class CreditScenarioSimulator(ScenarioBase):
    """
    Simulates credit default scenarios using estimated elasticities.

    Computes:
        Delta_default_rate = elasticity x Delta_income_pct

    With proper external validity caveats.
    """

    def __init__(
        self,
        elasticity_store: ElasticityStore | None = None,
        baseline_default_rate: float | None = None,
    ):
        """
        Initialize simulator.

        Args:
            elasticity_store: Store of estimated elasticities
            baseline_default_rate: Portfolio baseline DPD30 rate
        """
        super().__init__()
        self.store = elasticity_store or ElasticityStore()
        self.baseline_default_rate = baseline_default_rate or 0.05  # 5% default

    def _get_elasticities(self) -> dict[str, float]:
        """Get elasticities for simulation."""
        eset = self.store.get("credit_default")
        if eset is None:
            return {}
        return {e.segment: e.coefficient for e in eset.elasticities}

    def _apply_shock(
        self,
        segment: str,
        shock_size: float,
        elasticity: float,
    ) -> float:
        """Apply income shock to get default rate change."""
        # elasticity = delta_default / delta_income_pct
        # so delta_default = elasticity * delta_income_pct
        return elasticity * shock_size

    def simulate_income_shock(
        self,
        income_change_pct: float,
        segments: list[str] | None = None,
        segment_weights: dict[str, float] | None = None,
    ) -> CreditScenarioResult:
        """
        Simulate effect of income shock on default rates.

        Args:
            income_change_pct: Income change as decimal (e.g., -0.10 for -10%)
            segments: Segments to include (default: all available)
            segment_weights: Weights for aggregating across segments

        Returns:
            CreditScenarioResult with predicted effects
        """
        elasticities = self._get_elasticities()

        if not elasticities:
            raise ValueError(
                "No elasticities available. "
                "Run estimation first and store results."
            )

        segments = segments or list(elasticities.keys())
        segment_weights = segment_weights or {s: 1.0/len(segments) for s in segments}

        # Compute effects by segment
        segment_effects = {}
        total_effect = 0.0
        total_se = 0.0

        for segment in segments:
            if segment not in elasticities:
                logger.warning(f"Segment {segment} not in elasticities, skipping")
                continue

            elasticity = elasticities[segment]
            effect = self._apply_shock(segment, income_change_pct, elasticity)
            weight = segment_weights.get(segment, 0)

            segment_effects[segment] = {
                "elasticity": elasticity,
                "effect": effect,
                "weighted_effect": effect * weight,
            }

            total_effect += effect * weight

            # Get SE for confidence interval
            elast = self.store.get_elasticity("credit_default", segment)
            if elast:
                se = elast.std_error * income_change_pct * weight
                total_se += se ** 2

        total_se = np.sqrt(total_se)

        # Predicted default rate
        predicted_rate = self.baseline_default_rate + total_effect
        predicted_rate = max(0, min(1, predicted_rate))  # Bound to [0,1]

        # Confidence interval
        conf_int = (
            total_effect - 1.96 * total_se,
            total_effect + 1.96 * total_se,
        )

        # External validity caveats
        caveats = self.get_external_validity_caveats()

        return CreditScenarioResult(
            scenario_name=f"Income shock {income_change_pct*100:+.0f}%",
            baseline_default_rate=self.baseline_default_rate,
            predicted_default_rate=predicted_rate,
            default_rate_change=total_effect,
            default_rate_pct_change=total_effect / self.baseline_default_rate if self.baseline_default_rate > 0 else 0,
            segment_effects=segment_effects,
            conf_int_change=conf_int,
            income_shock=income_change_pct,
            elasticities_used=elasticities,
            external_validity_caveats=caveats,
        )

    def get_external_validity_caveats(self) -> list[str]:
        """Get external validity caveats for simulation."""
        base_caveats = [
            "Estimated elasticities are local average treatment effects (LATE)",
            "Apply primarily to borrowers similar to identification sample",
        ]

        # Add segment-specific caveats
        eset = self.store.get("credit_default")
        if eset:
            for e in eset.elasticities:
                base_caveats.extend(e.external_validity_caveats)

        # Add general caveats
        base_caveats.extend([
            "Linear extrapolation may not hold for large shocks",
            "Portfolio composition may differ from study sample",
            "General equilibrium effects not captured",
        ])

        return list(set(base_caveats))  # Deduplicate


def simulate_income_shock(
    income_change_pct: float,
    baseline_rate: float = 0.05,
) -> CreditScenarioResult:
    """
    Convenience function to simulate income shock effect.

    Args:
        income_change_pct: Income change as decimal
        baseline_rate: Baseline default rate

    Returns:
        CreditScenarioResult
    """
    simulator = CreditScenarioSimulator(baseline_default_rate=baseline_rate)
    return simulator.simulate_income_shock(income_change_pct)
