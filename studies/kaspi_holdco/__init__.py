"""
Kaspi.kz Holding Company Capital Adequacy Study
================================================

Research Question
-----------------
Can Kaspi.kz's holding company absorb capital calls from its banking subsidiary
during stress scenarios while maintaining solvency and liquidity?

This study analyzes holding company capital adequacy from four perspectives:
1. Stand-alone solvency - Parent liabilities vs assets
2. Liquidity runway - Survival under dividend stop
3. Capital call capacity - Can parent inject capital if ARDFM requires?
4. Bank stress passthrough - K2 shortfall → capital call translation

Key Framework
-------------
- Bank-level stress (NBK framework): K1-2 and K2 ratio stress testing
- HoldCo cash flow simulation: 12-month liquidity runway analysis
- Passthrough integration: Bank K2 shortfall → HoldCo capital call

Data Sources
------------
- Kaspi.kz FY2024 20-F filing (parent-only and consolidated)
- NBK regulatory framework (K1-2, K2 minimums)
- Risk weight assumptions (NBK vs Basel III differences)

Key Files
---------
- src/bank_state.py: BankState, BankScenario dataclasses
- src/bank_stress.py: Bank K2/K1-2 stress simulation
- src/holdco_state.py: HoldCoState, HoldCoScenario dataclasses
- src/holdco_simulator.py: 12-month cash flow simulation
- src/bank_integration.py: K2 shortfall → capital call translation
- src/stress_scenarios.py: Pre-defined scenarios and FY2024 baseline
- src/cli.py: CLI commands (kzkaspi)
"""

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
from studies.kaspi_holdco.src.stress_scenarios import (
    FY2024_BANK_STATE,
    FY2024_HOLDCO_STATE,
    FY2024_BASELINE_SCENARIO,
    BANK_STRESS_SCENARIOS,
    HOLDCO_SCENARIOS,
)

__all__ = [
    # Bank-level
    "BankState",
    "BankScenario",
    "simulate_bank_stress",
    "run_bank_stress_grid",
    # HoldCo-level
    "HoldCoState",
    "HoldCoScenario",
    "HoldCoSimResult",
    "simulate_holdco_12m",
    "scenario_grid",
    "HoldCoStressTester",
    # Integration
    "PassthroughConfig",
    "compute_dividend_payout_fraction",
    "compute_capital_call",
    "build_holdco_scenario_from_bank_stress",
    # Pre-defined data
    "FY2024_BANK_STATE",
    "FY2024_HOLDCO_STATE",
    "FY2024_BASELINE_SCENARIO",
    "BANK_STRESS_SCENARIOS",
    "HOLDCO_SCENARIOS",
]
