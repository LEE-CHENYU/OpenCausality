"""Source modules for Kaspi.kz Holding Company Capital Adequacy Study."""

from studies.kaspi_holdco.src.bank_state import BankState, BankScenario
from studies.kaspi_holdco.src.bank_stress import simulate_bank_stress, run_bank_stress_grid
from studies.kaspi_holdco.src.holdco_state import HoldCoState, HoldCoScenario
from studies.kaspi_holdco.src.holdco_simulator import (
    HoldCoSimResult,
    simulate_holdco_12m,
    scenario_grid,
    HoldCoStressTester,
)
from studies.kaspi_holdco.src.bank_integration import (
    PassthroughConfig,
    compute_dividend_payout_fraction,
    compute_capital_call,
    build_holdco_scenario_from_bank_stress,
)

__all__ = [
    "BankState",
    "BankScenario",
    "simulate_bank_stress",
    "run_bank_stress_grid",
    "HoldCoState",
    "HoldCoScenario",
    "HoldCoSimResult",
    "simulate_holdco_12m",
    "scenario_grid",
    "HoldCoStressTester",
    "PassthroughConfig",
    "compute_dividend_payout_fraction",
    "compute_capital_call",
    "build_holdco_scenario_from_bank_stress",
]
