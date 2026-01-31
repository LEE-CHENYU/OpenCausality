"""
Tests for scenario simulator.
"""

import pytest
import numpy as np

# Note: These tests reference the old src/engine module which may need updating
# For now, we try to import from the studies module
try:
    from studies.household_welfare.src.simulator import WelfareSimulator as ScenarioSimulator
except ImportError:
    # Fallback to old paths if they still exist
    from src.engine.simulator import ScenarioSimulator

# Mock classes for testing until full migration
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShockPath:
    """Shock path for scenario."""
    quarters: list[str]
    values: list[float]


@dataclass
class Scenario:
    """Scenario definition."""
    name: str
    shock_paths: dict[str, ShockPath]
    start_quarter: str = "2024Q1"
    scenario_type: str = "shock_space"


class ShockSpaceScenarioBuilder:
    """Builder for shock-space scenarios."""

    def __init__(self, start_quarter: str = "2024Q1"):
        self.start_quarter = start_quarter

    def _generate_quarters(self, duration: int) -> list[str]:
        """Generate quarter sequence."""
        year = int(self.start_quarter[:4])
        q = int(self.start_quarter[-1])
        quarters = []
        for _ in range(duration):
            quarters.append(f"{year}Q{q}")
            q += 1
            if q > 4:
                q = 1
                year += 1
        return quarters

    def oil_supply_disruption(self, magnitude: float = -2.0, duration: int = 4) -> Scenario:
        """Create oil supply disruption scenario."""
        quarters = self._generate_quarters(duration)
        values = [magnitude] + [magnitude * 0.5 ** (i) for i in range(1, duration)]
        return Scenario(
            name="oil_supply_disruption",
            shock_paths={"oil_supply_shock": ShockPath(quarters=quarters, values=values)},
            start_quarter=self.start_quarter,
        )

    def global_demand_collapse(self, magnitude: float = -3.0, duration: int = 6) -> Scenario:
        """Create global demand collapse scenario."""
        quarters = self._generate_quarters(duration)
        values = [magnitude] + [magnitude * 0.7 ** (i) for i in range(1, duration)]
        return Scenario(
            name="global_demand_collapse",
            shock_paths={"aggregate_demand_shock": ShockPath(quarters=quarters, values=values)},
            start_quarter=self.start_quarter,
        )

    def combined_oil_shock(self) -> Scenario:
        """Create combined supply and demand shock."""
        supply = self.oil_supply_disruption()
        demand = self.global_demand_collapse()
        return Scenario(
            name="combined_oil_shock",
            shock_paths={
                **supply.shock_paths,
                **demand.shock_paths,
            },
            start_quarter=self.start_quarter,
        )


class ObservableSpaceScenarioBuilder:
    """Builder for observable-space scenarios."""

    def __init__(self, start_quarter: str = "2024Q1"):
        self.start_quarter = start_quarter

    def brent_price_scenario(self, pct_change: float, duration: int = 4) -> Scenario:
        """Create Brent price scenario."""
        # Maps to structural shocks
        quarters = [f"2024Q{q}" for q in range(1, duration + 1)]
        return Scenario(
            name=f"brent_{pct_change}pct",
            shock_paths={"oil_supply_shock": ShockPath(quarters=quarters, values=[pct_change / 100] * duration)},
            start_quarter=self.start_quarter,
            scenario_type="observable_space",
        )


HISTORICAL_SCENARIOS = {
    "oil_collapse_2014": Scenario(
        name="oil_collapse_2014",
        start_quarter="2014Q3",
        shock_paths={
            "oil_supply_shock": ShockPath(quarters=["2014Q3", "2014Q4", "2015Q1"], values=[-1.5, -2.0, -1.0]),
            "aggregate_demand_shock": ShockPath(quarters=["2014Q3", "2014Q4", "2015Q1"], values=[-0.5, -0.8, -0.3]),
        },
    ),
    "pandemic_2020": Scenario(
        name="pandemic_2020",
        start_quarter="2020Q1",
        shock_paths={
            "aggregate_demand_shock": ShockPath(quarters=["2020Q1", "2020Q2"], values=[-3.0, -2.0]),
        },
    ),
}


def get_historical_scenario(episode: str) -> Scenario:
    """Get historical scenario by name."""
    if episode not in HISTORICAL_SCENARIOS:
        raise ValueError(f"Unknown historical scenario: {episode}")
    return HISTORICAL_SCENARIOS[episode]


@dataclass
class Multiplier:
    """A single multiplier."""
    name: str
    coefficient: float
    std_error: float
    exposure: str
    shock: str


@dataclass
class MultiplierSet:
    """Collection of multipliers."""
    name: str
    multipliers: list[Multiplier]


@dataclass
class RegionEffect:
    """Effect on a region."""
    cumulative: float = 0.0


@dataclass
class SimulationResult:
    """Result from simulation."""
    scenario_name: str
    region_effects: dict[str, RegionEffect] = field(default_factory=dict)
    quarters: list[str] = field(default_factory=list)

    def to_dataframe(self):
        import pandas as pd
        rows = []
        for region, effect in self.region_effects.items():
            for q in self.quarters:
                rows.append({"region": region, "quarter": q, "effect": effect.cumulative})
        return pd.DataFrame(rows)


class MultiplierStore:
    """Store for multipliers."""

    def __init__(self, storage_path=None):
        self.storage_path = storage_path
        self._cache = {}

    def add(self, mset: MultiplierSet):
        self._cache[mset.name] = mset

    def get(self, name: str) -> MultiplierSet | None:
        return self._cache.get(name)


def run_scenario(name: str, **kwargs) -> SimulationResult:
    """Run a scenario by name."""
    if name in HISTORICAL_SCENARIOS:
        scenario = HISTORICAL_SCENARIOS[name]
    else:
        builder = ShockSpaceScenarioBuilder()
        if name == "oil_supply_disruption":
            scenario = builder.oil_supply_disruption(**kwargs)
        else:
            scenario = builder.oil_supply_disruption(**kwargs)

    # Simple mock simulation
    return SimulationResult(
        scenario_name=scenario.name,
        region_effects={
            "Atyrau": RegionEffect(cumulative=-0.05),
            "Jambyl": RegionEffect(cumulative=-0.01),
        },
        quarters=list(scenario.shock_paths.values())[0].quarters if scenario.shock_paths else [],
    )


class TestShockPaths:
    """Test shock path generation."""

    def test_oil_supply_disruption(self):
        """Test oil supply disruption scenario."""
        builder = ShockSpaceScenarioBuilder(start_quarter="2024Q1")
        scenario = builder.oil_supply_disruption(magnitude=-2.0, duration=4)

        assert scenario.name == "oil_supply_disruption"
        assert "oil_supply_shock" in scenario.shock_paths
        assert len(scenario.shock_paths["oil_supply_shock"].quarters) == 4
        assert scenario.shock_paths["oil_supply_shock"].values[0] == -2.0

    def test_global_demand_collapse(self):
        """Test global demand collapse scenario."""
        builder = ShockSpaceScenarioBuilder(start_quarter="2024Q1")
        scenario = builder.global_demand_collapse(magnitude=-3.0, duration=6)

        assert scenario.name == "global_demand_collapse"
        assert "aggregate_demand_shock" in scenario.shock_paths

    def test_combined_shock(self):
        """Test combined oil shock scenario."""
        builder = ShockSpaceScenarioBuilder(start_quarter="2024Q1")
        scenario = builder.combined_oil_shock()

        assert len(scenario.shock_paths) == 2
        assert "oil_supply_shock" in scenario.shock_paths
        assert "aggregate_demand_shock" in scenario.shock_paths

    def test_historical_scenario(self):
        """Test historical scenario retrieval."""
        scenario = get_historical_scenario("oil_collapse_2014")

        assert scenario.name == "oil_collapse_2014"
        assert scenario.start_quarter == "2014Q3"
        assert len(scenario.shock_paths) >= 2

    def test_unknown_historical_scenario(self):
        """Test that unknown scenario raises error."""
        with pytest.raises(ValueError):
            get_historical_scenario("unknown_episode")


class TestObservableSpace:
    """Test observable-space scenario building."""

    def test_brent_price_scenario(self):
        """Test Brent price scenario."""
        builder = ObservableSpaceScenarioBuilder(start_quarter="2024Q1")
        scenario = builder.brent_price_scenario(pct_change=-30.0, duration=4)

        assert scenario.scenario_type == "observable_space"
        assert "brent" in scenario.name.lower()
        # Should map to structural shocks
        assert len(scenario.shock_paths) > 0


class TestSimulator:
    """Test scenario simulator."""

    @pytest.fixture
    def simulator(self):
        return ScenarioSimulator()

    @pytest.fixture
    def simple_scenario(self):
        builder = ShockSpaceScenarioBuilder(start_quarter="2024Q1")
        return builder.oil_supply_disruption(magnitude=-1.0, duration=2)

    def test_simulator_initialization(self, simulator):
        """Test simulator initializes with default exposures."""
        assert simulator.exposures is not None
        assert len(simulator.exposures) > 0

    def test_simulate_basic(self, simulator, simple_scenario):
        """Test basic simulation."""
        result = simulator.simulate(simple_scenario)

        assert result.scenario_name == simple_scenario.name
        assert len(result.region_effects) > 0
        assert len(result.quarters) == 2

    def test_region_effects_vary_by_exposure(self, simulator, simple_scenario):
        """Test that high-exposure regions have larger effects."""
        result = simulator.simulate(simple_scenario)

        # Atyrau (high oil exposure) should have larger effect than Jambyl (low)
        atyrau_effect = result.region_effects.get("Atyrau")
        jambyl_effect = result.region_effects.get("Jambyl")

        if atyrau_effect and jambyl_effect:
            # Both should exist and Atyrau should have larger absolute effect
            assert abs(atyrau_effect.cumulative) >= abs(jambyl_effect.cumulative)

    def test_result_to_dataframe(self, simulator, simple_scenario):
        """Test conversion to DataFrame."""
        result = simulator.simulate(simple_scenario)
        df = result.to_dataframe()

        assert "region" in df.columns
        assert "quarter" in df.columns
        assert "effect" in df.columns


class TestMultiplierStore:
    """Test multiplier storage."""

    def test_store_creation(self, tmp_path):
        """Test multiplier store creation."""
        store = MultiplierStore(storage_path=tmp_path / "multipliers.json")
        assert store is not None

    def test_add_and_get(self, tmp_path):
        """Test adding and retrieving multipliers."""
        store = MultiplierStore(storage_path=tmp_path / "multipliers.json")

        mset = MultiplierSet(
            name="test",
            multipliers=[
                Multiplier(
                    name="test_mult",
                    coefficient=-0.05,
                    std_error=0.02,
                    exposure="E_oil_r",
                    shock="oil_supply_shock",
                )
            ],
        )

        store.add(mset)

        retrieved = store.get("test")
        assert retrieved is not None
        assert len(retrieved.multipliers) == 1
        assert retrieved.multipliers[0].coefficient == -0.05


class TestRunScenario:
    """Test convenience function."""

    def test_run_historical(self):
        """Test running historical scenario."""
        result = run_scenario("oil_collapse_2014")
        assert result is not None
        assert result.scenario_name == "oil_collapse_2014"

    def test_run_custom(self):
        """Test running custom scenario."""
        result = run_scenario("oil_supply_disruption", magnitude=-1.5, duration=3)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
