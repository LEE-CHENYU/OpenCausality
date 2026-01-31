"""
Scenario simulator.

Applies estimated multipliers to shock paths to generate welfare predictions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.engine.multipliers import MultiplierStore, MultiplierSet, Multiplier
from src.engine.shock_paths import Scenario, ShockPath
from src.model.panel_data import CANONICAL_REGIONS

logger = logging.getLogger(__name__)


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


class ScenarioSimulator:
    """
    Simulates scenarios using estimated multipliers.

    Computes:
        Δy_{r,t} = Σ_k β_k × E_k,r × S_k,t

    where β_k are estimated multipliers, E_k,r are region exposures,
    and S_k,t are shock paths.
    """

    def __init__(
        self,
        multiplier_store: MultiplierStore | None = None,
        exposures: pd.DataFrame | None = None,
    ):
        """
        Initialize simulator.

        Args:
            multiplier_store: Store of estimated multipliers
            exposures: DataFrame with region exposures (region, E_oil_r, E_cyc_r, etc.)
        """
        self.multiplier_store = multiplier_store or MultiplierStore()
        self._exposures = exposures
        self._default_exposures = self._create_default_exposures()

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

    def simulate(
        self,
        scenario: Scenario,
        multiplier_set_name: str = "shift_share",
        use_local_projections: bool = False,
    ) -> SimulationResult:
        """
        Simulate a scenario.

        Args:
            scenario: Scenario to simulate
            multiplier_set_name: Name of multiplier set to use
            use_local_projections: If True, use LP multipliers for dynamics

        Returns:
            SimulationResult with region-level effects
        """
        # Get multipliers
        mset = self.multiplier_store.get(multiplier_set_name)
        if mset is None:
            logger.warning(f"Multiplier set {multiplier_set_name} not found, using defaults")
            mset = self._create_default_multipliers()

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
                multiplier = self._find_multiplier(mset, shock_type)
                if multiplier is None:
                    continue

                multipliers_used.append(multiplier.name)

                # Get exposure
                exposure_col = multiplier.exposure
                if exposure_col in row:
                    exposure = row[exposure_col]
                else:
                    exposure = row.get("E_oil_r", 0.1)  # Default

                # Compute effect: β × E × S
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
                "multiplier_set": multiplier_set_name,
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
        from src.engine.multipliers import MultiplierSet, Multiplier

        # Based on literature/priors
        multipliers = [
            Multiplier(
                name="E_oil_r_x_oil_supply",
                coefficient=-0.05,  # 1 SD supply shock → -5% income in high-oil regions
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

    def backtest(
        self,
        scenario: Scenario,
        actual_data: pd.DataFrame,
        outcome: str = "log_income_pc",
    ) -> dict[str, Any]:
        """
        Backtest a historical scenario against actual data.

        Args:
            scenario: Historical scenario to backtest
            actual_data: Panel data with actual outcomes
            outcome: Outcome variable name

        Returns:
            Dictionary with backtest metrics
        """
        # Simulate scenario
        sim_result = self.simulate(scenario)

        # Get actual changes
        actual = actual_data.copy()
        if isinstance(actual.index, pd.MultiIndex):
            actual = actual.reset_index()

        # Filter to scenario quarters
        actual = actual[actual["quarter"].isin(sim_result.quarters)]

        # Compute actual changes by region
        metrics = {
            "scenario": scenario.name,
            "quarters": sim_result.quarters,
            "by_region": {},
        }

        for region in CANONICAL_REGIONS:
            if region not in sim_result.region_effects:
                continue

            predicted = sim_result.region_effects[region].effects

            region_actual = actual[actual["region"] == region].sort_values("quarter")
            if outcome in region_actual.columns:
                actual_values = region_actual[outcome].values
                if len(actual_values) >= 2:
                    actual_changes = np.diff(actual_values)
                    # Align with predicted
                    min_len = min(len(predicted), len(actual_changes))
                    if min_len > 0:
                        pred = predicted[:min_len]
                        act = actual_changes[:min_len]

                        # Compute metrics
                        rmse = np.sqrt(np.mean((pred - act) ** 2))
                        correlation = np.corrcoef(pred, act)[0, 1] if len(pred) > 1 else np.nan
                        bias = np.mean(pred - act)

                        metrics["by_region"][region] = {
                            "rmse": rmse,
                            "correlation": correlation,
                            "bias": bias,
                            "n_quarters": min_len,
                        }

        # Aggregate metrics
        if metrics["by_region"]:
            metrics["aggregate"] = {
                "mean_rmse": np.mean([m["rmse"] for m in metrics["by_region"].values()]),
                "mean_correlation": np.nanmean([m["correlation"] for m in metrics["by_region"].values()]),
                "mean_bias": np.mean([m["bias"] for m in metrics["by_region"].values()]),
            }

        return metrics


def run_scenario(
    scenario_name: str,
    multiplier_set: str = "shift_share",
    **scenario_kwargs: Any,
) -> SimulationResult:
    """
    Convenience function to run a predefined scenario.

    Args:
        scenario_name: Name of predefined scenario
        multiplier_set: Multiplier set to use
        **scenario_kwargs: Override scenario parameters

    Returns:
        SimulationResult
    """
    from src.engine.shock_paths import (
        ShockSpaceScenarioBuilder,
        get_historical_scenario,
    )

    # Check if historical scenario
    if scenario_name in ["oil_collapse_2014", "pandemic_2020", "energy_crisis_2022"]:
        scenario = get_historical_scenario(scenario_name)
    else:
        # Build custom scenario
        builder = ShockSpaceScenarioBuilder()
        if scenario_name == "oil_supply_disruption":
            scenario = builder.oil_supply_disruption(**scenario_kwargs)
        elif scenario_name == "global_demand_collapse":
            scenario = builder.global_demand_collapse(**scenario_kwargs)
        elif scenario_name == "combined_oil_shock":
            scenario = builder.combined_oil_shock(**scenario_kwargs)
        elif scenario_name == "vix_spike":
            scenario = builder.vix_spike(**scenario_kwargs)
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")

    simulator = ScenarioSimulator()
    return simulator.simulate(scenario, multiplier_set)
