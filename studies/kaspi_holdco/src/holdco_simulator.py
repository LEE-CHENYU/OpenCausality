"""
Holding company cash flow simulation engine.

Performs 12-month cash flow projections to assess holdco liquidity and
capital call capacity.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from studies.kaspi_holdco.src.holdco_state import HoldCoState, HoldCoScenario


@dataclass
class HoldCoSimResult:
    """
    Results from 12-month holdco cash flow simulation.

    Attributes:
        scenario_name: Name of the scenario
        starting_cash: Beginning cash balance
        ending_cash: Ending cash balance after 12 months
        total_inflows: Total inflows over 12 months
        total_fixed_costs: Total fixed costs over 12 months
        total_discretionary_outflows: Total discretionary outflows
        total_required_outflows: Total required outflows (capital calls)
        cash_coverage_ratio: Ending cash / annual fixed costs
        months_of_runway: Months of runway at ending cash level
        went_negative: Whether cash went negative during simulation
        min_cash_balance: Minimum cash balance during simulation
        needs_external_funding: Whether external funding is required
    """

    scenario_name: str
    starting_cash: float
    ending_cash: float
    total_inflows: float
    total_fixed_costs: float
    total_discretionary_outflows: float
    total_required_outflows: float
    cash_coverage_ratio: float
    months_of_runway: float
    went_negative: bool
    min_cash_balance: float
    needs_external_funding: bool

    def summary(self) -> str:
        """Generate formatted summary of simulation results."""
        status = "NEEDS FUNDING" if self.needs_external_funding else "Viable"
        neg_flag = "YES" if self.went_negative else "No"

        lines = [
            "=" * 60,
            f"HoldCo Simulation: {self.scenario_name}",
            "=" * 60,
            "",
            f"Status: {status}",
            "",
            "Cash Flow (KZT bn):",
            f"  Starting Cash:     {self.starting_cash:.1f}",
            f"  + Inflows:         {self.total_inflows:.1f}",
            f"  - Fixed Costs:     {self.total_fixed_costs:.1f}",
            f"  - Discretionary:   {self.total_discretionary_outflows:.1f}",
            f"  - Capital Calls:   {self.total_required_outflows:.1f}",
            f"  = Ending Cash:     {self.ending_cash:.1f}",
            "",
            "Metrics:",
            f"  Cash Coverage:     {self.cash_coverage_ratio:.1f}x annual fixed",
            f"  Runway:            {self.months_of_runway:.0f} months",
            f"  Min Balance:       {self.min_cash_balance:.1f}",
            f"  Went Negative:     {neg_flag}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "scenario": self.scenario_name,
            "starting_cash": self.starting_cash,
            "ending_cash": self.ending_cash,
            "total_inflows": self.total_inflows,
            "total_fixed_costs": self.total_fixed_costs,
            "total_discretionary": self.total_discretionary_outflows,
            "total_capital_calls": self.total_required_outflows,
            "cash_coverage_ratio": self.cash_coverage_ratio,
            "months_of_runway": self.months_of_runway,
            "went_negative": self.went_negative,
            "min_cash_balance": self.min_cash_balance,
            "needs_external_funding": self.needs_external_funding,
        }


def simulate_holdco_12m(
    state: HoldCoState,
    scenario: HoldCoScenario,
    monthly_detail: bool = False,
) -> HoldCoSimResult | tuple[HoldCoSimResult, pd.DataFrame]:
    """
    12-month holding company cash flow simulation.

    Simulates monthly cash flows assuming:
    - Dividends received evenly over the year
    - Fixed costs paid monthly
    - Discretionary outflows (shareholder dividends) paid evenly
    - Capital injections made as lump sum at start of year

    Formula:
        Ending Cash = Starting Cash
                    + Dividends from bank
                    + Dividends from other subs
                    + New equity/debt raised
                    - Annual fixed costs (G&A + interest)
                    - Dividends to shareholders
                    - Buybacks
                    - Capital injection to bank
                    - Acquisitions

    Args:
        state: Holdco balance sheet
        scenario: Cash flow scenario
        monthly_detail: Whether to return monthly cash flow DataFrame

    Returns:
        HoldCoSimResult, or tuple of (result, monthly_df) if monthly_detail=True
    """
    # Initialize
    cash_balance = state.cash
    monthly_fixed = state.monthly_fixed_costs
    monthly_inflows = scenario.total_inflows / 12
    monthly_discretionary = scenario.total_discretionary_outflows / 12

    # Capital injection assumed at start of year (worst case for liquidity)
    cash_balance -= scenario.capital_injection_to_bank

    # Track metrics
    min_balance = cash_balance
    went_negative = cash_balance < 0
    monthly_data = []

    # Month 0 (after capital call)
    monthly_data.append(
        {
            "month": 0,
            "cash_balance": cash_balance,
            "inflows": 0,
            "fixed_costs": 0,
            "discretionary": 0,
            "capital_call": scenario.capital_injection_to_bank,
        }
    )

    # Simulate 12 months
    for month in range(1, 13):
        inflows = monthly_inflows
        outflows = monthly_fixed + monthly_discretionary

        cash_balance = cash_balance + inflows - outflows

        min_balance = min(min_balance, cash_balance)
        if cash_balance < 0:
            went_negative = True

        monthly_data.append(
            {
                "month": month,
                "cash_balance": cash_balance,
                "inflows": inflows,
                "fixed_costs": monthly_fixed,
                "discretionary": monthly_discretionary,
                "capital_call": 0,
            }
        )

    # Calculate final metrics
    ending_cash = cash_balance
    annual_fixed = state.annual_fixed_costs

    if annual_fixed > 0:
        cash_coverage_ratio = ending_cash / annual_fixed
        months_of_runway = ending_cash / monthly_fixed if monthly_fixed > 0 else float("inf")
    else:
        cash_coverage_ratio = float("inf")
        months_of_runway = float("inf")

    needs_external_funding = went_negative or ending_cash < 0

    result = HoldCoSimResult(
        scenario_name=scenario.name,
        starting_cash=state.cash,
        ending_cash=ending_cash,
        total_inflows=scenario.total_inflows,
        total_fixed_costs=annual_fixed,
        total_discretionary_outflows=scenario.total_discretionary_outflows,
        total_required_outflows=scenario.capital_injection_to_bank,
        cash_coverage_ratio=cash_coverage_ratio,
        months_of_runway=months_of_runway,
        went_negative=went_negative,
        min_cash_balance=min_balance,
        needs_external_funding=needs_external_funding,
    )

    if monthly_detail:
        return result, pd.DataFrame(monthly_data)
    return result


def scenario_grid(
    state: HoldCoState,
    bank_dividends: list[float] | None = None,
    capital_calls: list[float] | None = None,
    shareholder_dividends: list[float] | None = None,
    other_sub_dividends: float | None = None,
) -> pd.DataFrame:
    """
    Run holdco simulation across a grid of scenarios.

    Args:
        state: Holdco balance sheet
        bank_dividends: List of bank dividend amounts (default: 0 to 300 by 50)
        capital_calls: List of capital call amounts (default: 0 to 400 by 50)
        shareholder_dividends: List of shareholder payout amounts
            (default: [0, 200, 400, 646])
        other_sub_dividends: Dividend from other subs (default: 467.5 from FY2024)

    Returns:
        DataFrame with simulation results for each combination
    """
    if bank_dividends is None:
        bank_dividends = [0, 50, 100, 150, 200, 250, 285]
    if capital_calls is None:
        capital_calls = [0, 50, 100, 150, 200, 300, 400]
    if shareholder_dividends is None:
        shareholder_dividends = [0, 200, 400, 646]
    if other_sub_dividends is None:
        other_sub_dividends = 467.5

    results = []

    for bank_div in bank_dividends:
        for capital_call in capital_calls:
            for sh_div in shareholder_dividends:
                scenario = HoldCoScenario(
                    name=f"BD{bank_div:.0f}_CC{capital_call:.0f}_SD{sh_div:.0f}",
                    dividend_from_bank=bank_div,
                    dividend_from_other_subs=other_sub_dividends,
                    dividends_to_shareholders=sh_div,
                    capital_injection_to_bank=capital_call,
                )

                result = simulate_holdco_12m(state, scenario)
                row = result.to_dict()
                row["bank_dividend"] = bank_div
                row["capital_call"] = capital_call
                row["shareholder_dividend"] = sh_div
                results.append(row)

    return pd.DataFrame(results)


class HoldCoStressTester:
    """
    Main stress testing class with pre-built scenarios.

    Provides convenient methods for running individual scenarios
    and comprehensive stress tests.
    """

    def __init__(self, state: HoldCoState | None = None):
        """
        Initialize stress tester.

        Args:
            state: Holdco state (defaults to FY2024)
        """
        self.state = state or HoldCoState.from_fy2024()

    def run_scenario(
        self,
        scenario: HoldCoScenario,
        monthly_detail: bool = False,
    ) -> HoldCoSimResult | tuple[HoldCoSimResult, pd.DataFrame]:
        """Run a single scenario."""
        return simulate_holdco_12m(self.state, scenario, monthly_detail=monthly_detail)

    def run_named_scenario(
        self,
        name: str,
        monthly_detail: bool = False,
    ) -> HoldCoSimResult | tuple[HoldCoSimResult, pd.DataFrame]:
        """
        Run a pre-defined named scenario.

        Args:
            name: Scenario name (baseline, no_bank_dividend, no_dividends,
                  dividend_stop_no_payout, moderate_capital_call, severe_capital_call)
            monthly_detail: Whether to return monthly DataFrame

        Returns:
            Simulation result
        """
        # Import here to avoid circular import
        from studies.kaspi_holdco.src.stress_scenarios import HOLDCO_SCENARIOS

        if name not in HOLDCO_SCENARIOS:
            raise ValueError(f"Unknown scenario: {name}. Available: {list(HOLDCO_SCENARIOS.keys())}")

        return simulate_holdco_12m(self.state, HOLDCO_SCENARIOS[name], monthly_detail=monthly_detail)

    def run_all_named_scenarios(self) -> dict[str, HoldCoSimResult]:
        """Run all pre-defined scenarios."""
        from studies.kaspi_holdco.src.stress_scenarios import HOLDCO_SCENARIOS

        return {name: simulate_holdco_12m(self.state, sc) for name, sc in HOLDCO_SCENARIOS.items()}

    def run_sensitivity_grid(
        self,
        bank_dividends: list[float] | None = None,
        capital_calls: list[float] | None = None,
        shareholder_dividends: list[float] | None = None,
    ) -> pd.DataFrame:
        """Run full sensitivity grid."""
        return scenario_grid(
            self.state,
            bank_dividends=bank_dividends,
            capital_calls=capital_calls,
            shareholder_dividends=shareholder_dividends,
        )

    def find_max_capital_call(
        self,
        bank_dividend: float,
        shareholder_dividend: float = 0.0,
        tolerance: float = 1.0,
    ) -> float:
        """
        Find maximum capital call holdco can absorb.

        Uses binary search to find the largest capital call that doesn't
        result in negative cash.

        Args:
            bank_dividend: Bank dividend received
            shareholder_dividend: Shareholder dividend paid
            tolerance: Precision of search (KZT bn)

        Returns:
            Maximum capital call amount (KZT bn)
        """
        low, high = 0.0, 1000.0

        while high - low > tolerance:
            mid = (low + high) / 2
            scenario = HoldCoScenario(
                name="search",
                dividend_from_bank=bank_dividend,
                dividend_from_other_subs=467.5,
                dividends_to_shareholders=shareholder_dividend,
                capital_injection_to_bank=mid,
            )
            result = simulate_holdco_12m(self.state, scenario)
            if result.needs_external_funding:
                high = mid
            else:
                low = mid

        return low

    def capacity_analysis(self) -> dict[str, float]:
        """
        Analyze capital call capacity under various dividend assumptions.

        Returns:
            Dictionary with max capital call under each scenario
        """
        return {
            "full_dividends": self.find_max_capital_call(
                bank_dividend=285.2,
                shareholder_dividend=646.0,
            ),
            "no_shareholder_dividend": self.find_max_capital_call(
                bank_dividend=285.2,
                shareholder_dividend=0.0,
            ),
            "no_bank_dividend": self.find_max_capital_call(
                bank_dividend=0.0,
                shareholder_dividend=0.0,
            ),
            "half_bank_dividend": self.find_max_capital_call(
                bank_dividend=142.6,
                shareholder_dividend=0.0,
            ),
        }
