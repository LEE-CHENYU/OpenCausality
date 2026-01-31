"""
Shock path generation for scenario analysis.

Two types:
1. Shock-space scenarios: Define in structural shock units
2. Observable-space scenarios: Define in observable terms (e.g., oil price change)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ShockPath:
    """A path of shock values over time."""

    name: str
    shock_type: str
    values: np.ndarray
    quarters: list[str]
    units: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            "quarter": self.quarters,
            "shock": self.values,
            "shock_type": self.shock_type,
        })

    @property
    def duration(self) -> int:
        """Duration in quarters."""
        return len(self.quarters)

    @property
    def cumulative(self) -> float:
        """Cumulative shock."""
        return np.sum(self.values)

    @property
    def peak(self) -> float:
        """Peak shock value."""
        return np.max(np.abs(self.values)) * np.sign(self.values[np.argmax(np.abs(self.values))])


@dataclass
class Scenario:
    """A complete scenario with multiple shock paths."""

    name: str
    description: str
    shock_paths: dict[str, ShockPath]
    start_quarter: str
    scenario_type: Literal["shock_space", "observable_space"]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_shock_path(self, shock_type: str) -> ShockPath | None:
        """Get a specific shock path."""
        return self.shock_paths.get(shock_type)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all shock paths to a single DataFrame."""
        dfs = []
        for name, path in self.shock_paths.items():
            df = path.to_dataframe()
            df["shock_name"] = name
            dfs.append(df)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()


class ShockSpaceScenarioBuilder:
    """
    Build scenarios in shock-space (structural shock units).

    Clean causal interpretation: scenarios are defined in the same units
    as the estimated shocks.
    """

    def __init__(self, start_quarter: str = "2024Q1"):
        self.start_quarter = start_quarter

    def _generate_quarters(self, n_quarters: int) -> list[str]:
        """Generate quarter labels starting from start_quarter."""
        start_year = int(self.start_quarter[:4])
        start_q = int(self.start_quarter[-1])

        quarters = []
        for i in range(n_quarters):
            q = (start_q - 1 + i) % 4 + 1
            year = start_year + (start_q - 1 + i) // 4
            quarters.append(f"{year}Q{q}")

        return quarters

    def oil_supply_disruption(
        self,
        magnitude: float = -2.0,
        duration: int = 4,
        decay: float = 0.5,
    ) -> Scenario:
        """
        Create oil supply disruption scenario.

        Args:
            magnitude: Initial shock size in standard deviations
            duration: Duration in quarters
            decay: Exponential decay rate per quarter

        Returns:
            Scenario with oil supply shock path
        """
        quarters = self._generate_quarters(duration)
        values = np.array([magnitude * (decay ** i) for i in range(duration)])

        shock_path = ShockPath(
            name="oil_supply_disruption",
            shock_type="oil_supply_shock",
            values=values,
            quarters=quarters,
            units="standard deviations",
            description=f"Oil supply disruption: {magnitude} SD shock with {decay} decay",
        )

        return Scenario(
            name="oil_supply_disruption",
            description=f"Oil supply shock of {magnitude} SD, decaying over {duration} quarters",
            shock_paths={"oil_supply_shock": shock_path},
            start_quarter=self.start_quarter,
            scenario_type="shock_space",
            metadata={"magnitude": magnitude, "duration": duration, "decay": decay},
        )

    def global_demand_collapse(
        self,
        magnitude: float = -3.0,
        duration: int = 6,
        recovery_rate: float = 0.3,
    ) -> Scenario:
        """
        Create global demand collapse scenario (e.g., recession).

        Args:
            magnitude: Initial shock size in standard deviations
            duration: Duration in quarters
            recovery_rate: Rate of recovery per quarter

        Returns:
            Scenario with aggregate demand shock path
        """
        quarters = self._generate_quarters(duration)

        # V-shaped recovery pattern
        trough_quarter = duration // 3
        values = np.zeros(duration)

        for i in range(duration):
            if i <= trough_quarter:
                # Decline phase
                values[i] = magnitude * (1 - 0.5 * i / trough_quarter)
            else:
                # Recovery phase
                recovery_progress = (i - trough_quarter) / (duration - trough_quarter)
                values[i] = magnitude * 0.5 * (1 - recovery_progress)

        shock_path = ShockPath(
            name="global_demand_collapse",
            shock_type="aggregate_demand_shock",
            values=values,
            quarters=quarters,
            units="standard deviations",
            description=f"Global demand collapse: {magnitude} SD shock with recovery",
        )

        return Scenario(
            name="global_demand_collapse",
            description=f"Aggregate demand shock of {magnitude} SD with V-shaped recovery",
            shock_paths={"aggregate_demand_shock": shock_path},
            start_quarter=self.start_quarter,
            scenario_type="shock_space",
            metadata={"magnitude": magnitude, "duration": duration},
        )

    def combined_oil_shock(
        self,
        supply_magnitude: float = -1.5,
        demand_magnitude: float = -2.0,
        duration: int = 4,
    ) -> Scenario:
        """
        Create combined oil supply and demand shock scenario.

        Args:
            supply_magnitude: Supply shock size
            demand_magnitude: Demand shock size
            duration: Duration in quarters

        Returns:
            Scenario with both shock paths
        """
        quarters = self._generate_quarters(duration)

        # Supply shock decays faster
        supply_values = np.array([supply_magnitude * (0.6 ** i) for i in range(duration)])

        # Demand shock more persistent
        demand_values = np.array([demand_magnitude * (0.8 ** i) for i in range(duration)])

        supply_path = ShockPath(
            name="oil_supply",
            shock_type="oil_supply_shock",
            values=supply_values,
            quarters=quarters,
            units="standard deviations",
        )

        demand_path = ShockPath(
            name="oil_demand",
            shock_type="aggregate_demand_shock",
            values=demand_values,
            quarters=quarters,
            units="standard deviations",
        )

        return Scenario(
            name="combined_oil_shock",
            description=f"Combined supply ({supply_magnitude} SD) and demand ({demand_magnitude} SD) shocks",
            shock_paths={
                "oil_supply_shock": supply_path,
                "aggregate_demand_shock": demand_path,
            },
            start_quarter=self.start_quarter,
            scenario_type="shock_space",
        )

    def vix_spike(
        self,
        magnitude: float = 3.0,
        duration: int = 2,
    ) -> Scenario:
        """
        Create VIX spike (financial stress) scenario.

        Args:
            magnitude: VIX innovation size
            duration: Duration in quarters

        Returns:
            Scenario with VIX shock path
        """
        quarters = self._generate_quarters(duration)
        values = np.array([magnitude * (0.4 ** i) for i in range(duration)])

        shock_path = ShockPath(
            name="vix_spike",
            shock_type="vix_shock",
            values=values,
            quarters=quarters,
            units="VIX innovation units",
        )

        return Scenario(
            name="vix_spike",
            description=f"VIX spike of {magnitude} units",
            shock_paths={"vix_shock": shock_path},
            start_quarter=self.start_quarter,
            scenario_type="shock_space",
        )


class ObservableSpaceScenarioBuilder:
    """
    Build scenarios in observable-space.

    Define in observable terms (e.g., "Brent falls 30%"), then map to shocks.
    Requires a filter/mapping model to convert observables to structural shocks.
    """

    def __init__(
        self,
        start_quarter: str = "2024Q1",
        shock_mapping: dict[str, dict[str, float]] | None = None,
    ):
        self.start_quarter = start_quarter

        # Default mapping from observable changes to shock units
        # These should be estimated from data
        self.shock_mapping = shock_mapping or {
            "brent_pct_change": {
                "oil_supply_shock": -0.05,  # 1% Brent drop ≈ -0.05 SD supply shock
                "aggregate_demand_shock": -0.03,  # 1% Brent drop ≈ -0.03 SD demand shock
            },
            "vix_change": {
                "vix_shock": 0.1,  # 1 VIX point ≈ 0.1 SD VIX innovation
            },
        }

    def _generate_quarters(self, n_quarters: int) -> list[str]:
        """Generate quarter labels."""
        start_year = int(self.start_quarter[:4])
        start_q = int(self.start_quarter[-1])

        quarters = []
        for i in range(n_quarters):
            q = (start_q - 1 + i) % 4 + 1
            year = start_year + (start_q - 1 + i) // 4
            quarters.append(f"{year}Q{q}")

        return quarters

    def _map_observable_to_shocks(
        self,
        observable: str,
        value: float,
    ) -> dict[str, float]:
        """Map observable change to structural shocks."""
        if observable not in self.shock_mapping:
            logger.warning(f"No mapping for observable: {observable}")
            return {}

        mapping = self.shock_mapping[observable]
        return {shock: coef * value for shock, coef in mapping.items()}

    def brent_price_scenario(
        self,
        pct_change: float = -30.0,
        duration: int = 4,
        profile: Literal["step", "gradual", "vshape"] = "gradual",
    ) -> Scenario:
        """
        Create scenario from Brent price change.

        Args:
            pct_change: Percentage change in Brent price
            duration: Duration in quarters
            profile: Shape of price path

        Returns:
            Scenario with implied structural shocks
        """
        quarters = self._generate_quarters(duration)

        # Generate price path profile
        if profile == "step":
            # Immediate change, persists
            pct_values = np.full(duration, pct_change)
        elif profile == "gradual":
            # Gradual decline to trough, then partial recovery
            trough = duration // 2
            pct_values = np.zeros(duration)
            for i in range(duration):
                if i <= trough:
                    pct_values[i] = pct_change * (i + 1) / (trough + 1)
                else:
                    recovery = 0.3 * pct_change * (i - trough) / (duration - trough)
                    pct_values[i] = pct_change - recovery
        else:  # vshape
            trough = duration // 2
            pct_values = np.zeros(duration)
            for i in range(duration):
                if i <= trough:
                    pct_values[i] = pct_change * (i + 1) / (trough + 1)
                else:
                    pct_values[i] = pct_change * (duration - i) / (duration - trough)

        # Map to structural shocks
        shock_mapping = self._map_observable_to_shocks("brent_pct_change", 1.0)

        shock_paths = {}
        for shock_type, coef in shock_mapping.items():
            shock_values = pct_values * coef
            shock_paths[shock_type] = ShockPath(
                name=f"brent_implied_{shock_type}",
                shock_type=shock_type,
                values=shock_values,
                quarters=quarters,
                units="standard deviations (inferred)",
                description=f"Implied from {pct_change}% Brent change",
            )

        return Scenario(
            name=f"brent_{int(abs(pct_change))}pct_{'drop' if pct_change < 0 else 'rise'}",
            description=f"Brent price {'falls' if pct_change < 0 else 'rises'} {abs(pct_change):.0f}%",
            shock_paths=shock_paths,
            start_quarter=self.start_quarter,
            scenario_type="observable_space",
            metadata={
                "observable": "brent_price",
                "pct_change": pct_change,
                "profile": profile,
                "mapping_assumptions": "Requires filter model for accurate decomposition",
            },
        )


# Predefined historical scenarios for backtesting
def get_historical_scenario(episode: str, start_quarter: str | None = None) -> Scenario:
    """
    Get a predefined historical scenario for backtesting.

    Args:
        episode: Episode name (e.g., "oil_collapse_2014", "pandemic_2020")
        start_quarter: Override start quarter

    Returns:
        Scenario matching the historical episode
    """
    episodes = {
        "oil_collapse_2014": {
            "start": "2014Q3",
            "description": "2014-16 oil price collapse",
            "shocks": {
                "oil_supply_shock": [0.5, 1.0, 0.8, 0.3, 0.2, 0.1],  # Positive supply shock
                "aggregate_demand_shock": [-1.5, -2.0, -1.5, -1.0, -0.5, -0.3],  # Negative demand
            },
        },
        "pandemic_2020": {
            "start": "2020Q1",
            "description": "COVID-19 pandemic shock",
            "shocks": {
                "oil_supply_shock": [-0.5, 1.5, 0.5, 0.2],  # Initial disruption then glut
                "aggregate_demand_shock": [-3.0, -2.5, -1.0, 0.5],  # Sharp demand collapse
                "vix_shock": [4.0, 2.0, 0.5, 0.0],  # VIX spike
            },
        },
        "energy_crisis_2022": {
            "start": "2022Q1",
            "description": "2022 energy market disruption",
            "shocks": {
                "oil_supply_shock": [-2.0, -1.5, -1.0, -0.5],  # Supply disruption
                "aggregate_demand_shock": [-0.5, -0.3, -0.2, 0.0],  # Mild demand impact
            },
        },
    }

    if episode not in episodes:
        raise ValueError(f"Unknown episode: {episode}. Available: {list(episodes.keys())}")

    ep = episodes[episode]
    start = start_quarter or ep["start"]

    # Build shock paths
    builder = ShockSpaceScenarioBuilder(start_quarter=start)
    quarters = builder._generate_quarters(len(list(ep["shocks"].values())[0]))

    shock_paths = {}
    for shock_type, values in ep["shocks"].items():
        shock_paths[shock_type] = ShockPath(
            name=f"{episode}_{shock_type}",
            shock_type=shock_type,
            values=np.array(values),
            quarters=quarters[:len(values)],
            units="standard deviations (historical)",
            description=f"Historical {shock_type} for {episode}",
        )

    return Scenario(
        name=episode,
        description=ep["description"],
        shock_paths=shock_paths,
        start_quarter=start,
        scenario_type="shock_space",
        metadata={"source": "historical", "episode": episode},
    )
