"""
Scenario simulator for household welfare study.

Applies estimated multipliers to shock paths to generate welfare predictions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from shared.engine.scenario_base import ScenarioBase, ScenarioResult
from studies.household_welfare.src.panel_data import CANONICAL_REGIONS

logger = logging.getLogger(__name__)


@dataclass
class Multiplier:
    """A single multiplier estimate."""

    name: str
    coefficient: float
    std_error: float
    exposure: str
    shock: str
    horizon: int = 0


@dataclass
class MultiplierSet:
    """Collection of multipliers from a model estimation."""

    name: str
    multipliers: list[Multiplier]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShockPath:
    """Time series of shock values."""

    name: str
    quarters: list[str]
    values: np.ndarray


@dataclass
class Scenario:
    """A scenario with multiple shock paths."""

    name: str
    scenario_type: str
    shock_paths: dict[str, ShockPath]
    description: str = ""


@dataclass
class RegionEffect:
    """Effect on a single region."""

    region: str
    quarters: list[str]
    effects: np.ndarray
    cumulative: float
    peak: float
    exposures: dict[str, float]


@dataclass
class SimulationResult:
    """Results from a scenario simulation."""

    scenario_name: str
    region_effects: dict[str, RegionEffect]
    aggregate_effect: np.ndarray
    quarters: list[str]
    multipliers_used: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with region-quarter effects."""
        records = []
        for region, effect in self.region_effects.items():
            for i, quarter in enumerate(effect.quarters):
                records.append({
                    "region": region,
                    "quarter": quarter,
                    "effect": effect.effects[i],
                    "cumulative": effect.cumulative,
                    "peak": effect.peak,
                })
        return pd.DataFrame(records)

    def summary(self) -> str:
        """Generate text summary of results."""
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"Simulation Results: {self.scenario_name}")
        lines.append(f"{'='*70}")
        lines.append(f"\nQuarters: {self.quarters[0]} to {self.quarters[-1]}")
        lines.append(f"Multipliers used: {', '.join(self.multipliers_used)}")

        # Aggregate effects
        lines.append(f"\nAggregate Effects:")
        lines.append(f"  Peak effect: {np.min(self.aggregate_effect):.4f}")
        lines.append(f"  Cumulative: {np.sum(self.aggregate_effect):.4f}")

        # Top affected regions
        lines.append(f"\nTop 5 Most Affected Regions:")
        sorted_regions = sorted(
            self.region_effects.items(),
            key=lambda x: abs(x[1].cumulative),
            reverse=True,
        )[:5]

        for region, effect in sorted_regions:
            lines.append(f"  {region}: cumulative={effect.cumulative:.4f}, peak={effect.peak:.4f}")

        return "\n".join(lines)


class WelfareSimulator(ScenarioBase):
    """
    Simulates welfare scenarios using estimated multipliers.

    Computes:
        Delta_y_{r,t} = Sum_k beta_k x E_k,r x S_k,t

    where beta_k are estimated multipliers, E_k,r are region exposures,
    and S_k,t are shock paths.
    """

    def __init__(
        self,
        multipliers: dict[str, float] | None = None,
        exposures: pd.DataFrame | None = None,
    ):
        """
        Initialize simulator.

        Args:
            multipliers: Dictionary of estimated multipliers
            exposures: DataFrame with region exposures (region, E_oil_r, E_cyc_r, etc.)
        """
        super().__init__(multipliers)
        self._exposures = exposures
        self._default_exposures = self._create_default_exposures()

    def _get_elasticities(self) -> dict[str, float]:
        """Get elasticities (multipliers) for simulation."""
        return self.elasticities

    def _apply_shock(
        self,
        segment: str,
        shock_size: float,
        elasticity: float,
    ) -> float:
        """Apply a shock to a segment (region)."""
        # Get exposure for this region
        exposure = self._get_region_exposure(segment)
        return elasticity * exposure * shock_size

    def _get_region_exposure(self, region: str) -> float:
        """Get oil exposure for a region."""
        exposures = self.exposures
        region_row = exposures[exposures["region"] == region]
        if len(region_row) > 0:
            return region_row["E_oil_r"].iloc[0]
        return 0.1  # Default

    def _create_default_exposures(self) -> pd.DataFrame:
        """Create default exposures based on stylized facts."""
        # Stylized oil exposures for Kazakhstan regions
        oil_exposures = {
            "Atyrau": 0.80,
            "Mangystau": 0.70,
            "West Kazakhstan": 0.50,
            "Kyzylorda": 0.30,
            "Aktobe": 0.25,
            "Karaganda": 0.15,
            "Pavlodar": 0.12,
            "East Kazakhstan": 0.08,
            "Kostanay": 0.05,
            "Akmola": 0.05,
            "North Kazakhstan": 0.03,
            "Almaty Region": 0.03,
            "Jambyl": 0.02,
            "South Kazakhstan": 0.02,
            "Almaty City": 0.05,
            "Astana": 0.03,
        }

        # Cyclical exposures (rough approximation)
        cyclical_exposures = {region: 0.3 for region in CANONICAL_REGIONS}
        cyclical_exposures["Almaty City"] = 0.5
        cyclical_exposures["Astana"] = 0.45

        records = []
        for region in CANONICAL_REGIONS:
            records.append({
                "region": region,
                "E_oil_r": oil_exposures.get(region, 0.05),
                "E_cyc_r": cyclical_exposures.get(region, 0.3),
                "E_debt_r": 0.1,  # Placeholder
            })

        return pd.DataFrame(records)

    @property
    def exposures(self) -> pd.DataFrame:
        """Get exposure data."""
        if self._exposures is not None:
            return self._exposures
        return self._default_exposures

    def simulate_scenario(
        self,
        scenario: Scenario,
        multiplier_set: MultiplierSet | None = None,
    ) -> SimulationResult:
        """
        Simulate a scenario.

        Args:
            scenario: Scenario to simulate
            multiplier_set: Set of multipliers to use

        Returns:
            SimulationResult with region-level effects
        """
        # Get multipliers
        if multiplier_set is None:
            multiplier_set = self._create_default_multipliers()

        # Get quarters from scenario
        all_quarters = set()
        for path in scenario.shock_paths.values():
            all_quarters.update(path.quarters)
        quarters = sorted(all_quarters)

        # Compute effects for each region
        region_effects = {}
        multipliers_used = []

        for _, row in self.exposures.iterrows():
            region = row["region"]
            effects = np.zeros(len(quarters))

            for shock_type, shock_path in scenario.shock_paths.items():
                # Find matching multiplier
                multiplier = self._find_multiplier(multiplier_set, shock_type)
                if multiplier is None:
                    continue

                multipliers_used.append(multiplier.name)

                # Get exposure
                exposure_col = multiplier.exposure
                if exposure_col in row:
                    exposure = row[exposure_col]
                else:
                    exposure = row.get("E_oil_r", 0.1)  # Default

                # Compute effect: beta x E x S
                for i, quarter in enumerate(quarters):
                    if quarter in shock_path.quarters:
                        shock_idx = shock_path.quarters.index(quarter)
                        shock_value = shock_path.values[shock_idx]
                        effects[i] += multiplier.coefficient * exposure * shock_value

            region_effects[region] = RegionEffect(
                region=region,
                quarters=quarters,
                effects=effects,
                cumulative=np.sum(effects),
                peak=effects[np.argmax(np.abs(effects))] if len(effects) > 0 else 0,
                exposures={col: row[col] for col in row.index if col.startswith("E_")},
            )

        # Compute aggregate effect (weighted by exposures or simple average)
        aggregate_effect = np.mean(
            [re.effects for re in region_effects.values()],
            axis=0,
        )

        return SimulationResult(
            scenario_name=scenario.name,
            region_effects=region_effects,
            aggregate_effect=aggregate_effect,
            quarters=quarters,
            multipliers_used=list(set(multipliers_used)),
            metadata={
                "scenario_type": scenario.scenario_type,
            },
        )

    def _find_multiplier(
        self,
        mset: MultiplierSet,
        shock_type: str,
    ) -> Multiplier | None:
        """Find matching multiplier for a shock type."""
        # Try exact match
        for m in mset.multipliers:
            if m.shock == shock_type:
                return m

        # Try partial match
        for m in mset.multipliers:
            if shock_type.replace("_shock", "") in m.shock:
                return m

        return None

    def _create_default_multipliers(self) -> MultiplierSet:
        """Create default multipliers for when none are estimated."""
        # Based on literature/priors
        multipliers = [
            Multiplier(
                name="E_oil_r_x_oil_supply",
                coefficient=-0.05,  # 1 SD supply shock -> -5% income in high-oil regions
                std_error=0.02,
                exposure="E_oil_r",
                shock="oil_supply_shock",
            ),
            Multiplier(
                name="E_oil_r_x_aggregate_demand",
                coefficient=-0.08,  # Demand shocks have larger effect
                std_error=0.03,
                exposure="E_oil_r",
                shock="aggregate_demand_shock",
            ),
            Multiplier(
                name="E_cyc_r_x_global_activity",
                coefficient=-0.04,
                std_error=0.015,
                exposure="E_cyc_r",
                shock="global_activity_shock",
            ),
            Multiplier(
                name="E_oil_r_x_vix",
                coefficient=-0.02,
                std_error=0.01,
                exposure="E_oil_r",
                shock="vix_shock",
            ),
        ]

        return MultiplierSet(
            name="default",
            multipliers=multipliers,
            metadata={"source": "default/literature priors"},
        )


def create_oil_supply_disruption(
    start_quarter: str = "2024Q1",
    duration: int = 4,
    shock_size: float = -2.0,
) -> Scenario:
    """Create an oil supply disruption scenario."""
    # Generate quarters
    year, q = int(start_quarter[:4]), int(start_quarter[-1])
    quarters = []
    for _ in range(duration):
        quarters.append(f"{year}Q{q}")
        q += 1
        if q > 4:
            q = 1
            year += 1

    # Create shock path (decaying)
    values = np.array([shock_size * (0.7 ** i) for i in range(duration)])

    return Scenario(
        name="oil_supply_disruption",
        scenario_type="shock_space",
        shock_paths={
            "oil_supply_shock": ShockPath(
                name="oil_supply_shock",
                quarters=quarters,
                values=values,
            ),
        },
        description=f"Oil supply disruption starting {start_quarter}",
    )


def run_scenario(
    scenario_name: str,
    multipliers: dict[str, float] | None = None,
    **kwargs: Any,
) -> SimulationResult:
    """
    Convenience function to run a predefined scenario.

    Args:
        scenario_name: Name of predefined scenario
        multipliers: Optional multipliers to use
        **kwargs: Override scenario parameters

    Returns:
        SimulationResult
    """
    simulator = WelfareSimulator(multipliers=multipliers)

    if scenario_name == "oil_supply_disruption":
        scenario = create_oil_supply_disruption(**kwargs)
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    return simulator.simulate_scenario(scenario)
