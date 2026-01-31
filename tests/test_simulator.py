"""
Tests for scenario simulator.
"""

import pytest
import numpy as np

from src.engine.shock_paths import (
    ShockSpaceScenarioBuilder,
    ObservableSpaceScenarioBuilder,
    get_historical_scenario,
    ShockPath,
    Scenario,
)
from src.engine.simulator import ScenarioSimulator, run_scenario
from src.engine.multipliers import MultiplierStore, MultiplierSet, Multiplier


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
