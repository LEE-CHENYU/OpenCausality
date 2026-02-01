"""
Bank-level stress simulation engine.

Performs joint liquidity + solvency stress testing under NBK framework.
Key outputs feed into holdco capital call calculations.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from studies.kaspi_holdco.src.bank_state import BankState, BankScenario


# NBK regulatory minimums
K1_2_MINIMUM = 0.105  # 10.5%
K2_MINIMUM = 0.12  # 12.0%


@dataclass
class BankStressResult:
    """
    Results from bank-level stress simulation.

    Attributes:
        scenario_name: Name of the scenario
        k1_2_before: K1-2 ratio before stress
        k1_2_after: K1-2 ratio after stress
        k2_before: K2 ratio before stress
        k2_after: K2 ratio after stress
        capital_shortfall: Amount below minimum K2 (KZT bn, 0 if above)
        liquidity_gap: Unfilled cash need after all liquidity tools (KZT bn)
        support_needed: Whether external support is required
        retained_earnings: Earnings retained during stress
        credit_losses: Credit losses (KZT bn)
        fire_sale_losses: Losses from security fire sales (KZT bn)
        mtm_losses: Mark-to-market losses on securities (KZT bn)
        total_outflows: Total liquidity outflows (KZT bn)
        repo_used: Amount of repo capacity used (KZT bn)
        securities_sold: Securities liquidated (KZT bn)
        rwa_after: RWA after stress
        total_capital_after: Total capital after stress
    """

    scenario_name: str
    k1_2_before: float
    k1_2_after: float
    k2_before: float
    k2_after: float
    capital_shortfall: float
    liquidity_gap: float
    support_needed: bool
    retained_earnings: float
    credit_losses: float
    fire_sale_losses: float
    mtm_losses: float
    total_outflows: float
    repo_used: float
    securities_sold: float
    rwa_after: float
    total_capital_after: float

    def summary(self) -> str:
        """Generate formatted summary of stress results."""
        status = "SUPPORT NEEDED" if self.support_needed else "Viable"
        lines = [
            "=" * 60,
            f"Bank Stress Result: {self.scenario_name}",
            "=" * 60,
            "",
            f"Status: {status}",
            "",
            "Capital Ratios:",
            f"  K1-2: {self.k1_2_before:.1%} -> {self.k1_2_after:.1%} (min 10.5%)",
            f"  K2:   {self.k2_before:.1%} -> {self.k2_after:.1%} (min 12.0%)",
            "",
            "Capital Impact (KZT bn):",
            f"  Retained Earnings:  +{self.retained_earnings:.1f}",
            f"  Credit Losses:      -{self.credit_losses:.1f}",
            f"  Fire-Sale Losses:   -{self.fire_sale_losses:.1f}",
            f"  MTM Losses:         -{self.mtm_losses:.1f}",
            f"  Total Capital:      {self.total_capital_after:.1f}",
            f"  RWA After:          {self.rwa_after:.1f}",
            "",
            f"  Capital Shortfall:  {self.capital_shortfall:.1f}",
            "",
            "Liquidity (KZT bn):",
            f"  Total Outflows:   {self.total_outflows:.1f}",
            f"  Repo Used:        {self.repo_used:.1f}",
            f"  Securities Sold:  {self.securities_sold:.1f}",
            f"  Liquidity Gap:    {self.liquidity_gap:.1f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "scenario": self.scenario_name,
            "k1_2_before": self.k1_2_before,
            "k1_2_after": self.k1_2_after,
            "k2_before": self.k2_before,
            "k2_after": self.k2_after,
            "capital_shortfall": self.capital_shortfall,
            "liquidity_gap": self.liquidity_gap,
            "support_needed": self.support_needed,
            "retained_earnings": self.retained_earnings,
            "credit_losses": self.credit_losses,
            "fire_sale_losses": self.fire_sale_losses,
            "mtm_losses": self.mtm_losses,
            "total_capital_after": self.total_capital_after,
            "rwa_after": self.rwa_after,
        }


def simulate_bank_stress(state: BankState, scenario: BankScenario) -> BankStressResult:
    """
    Joint liquidity + solvency stress simulation.

    Liquidity waterfall:
    1. Calculate outflows (deposit runs, wholesale outflows)
    2. Use available cash
    3. Use repo capacity (with haircut)
    4. Sell securities (fire-sale discount)
    5. Any remaining need = liquidity gap

    Solvency:
    1. Start with total capital
    2. Add retained earnings (after profit compression)
    3. Subtract credit losses
    4. Subtract fire-sale losses (if securities sold)
    5. Subtract MTM losses (if hits capital)
    6. Calculate K2 against stressed RWA
    7. Shortfall = amount below K2 minimum

    Args:
        state: Bank balance sheet and regulatory position
        scenario: Stress scenario parameters

    Returns:
        BankStressResult with all outputs
    """
    # Record initial ratios
    k1_2_before = state.k1_2_ratio
    k2_before = state.k2_ratio

    # --- LIQUIDITY STRESS ---

    # Calculate outflows (30-day stress)
    retail_outflow = state.retail_deposits * scenario.retail_run_rate
    corporate_outflow = state.corporate_deposits * scenario.corporate_run_rate
    wholesale_outflow = min(scenario.wholesale_outflow, state.other_short_term_funding)
    total_outflows = retail_outflow + corporate_outflow + wholesale_outflow

    # Available liquidity (cash + due from banks, excluding mandatory)
    available_cash = state.cash - state.mandatory_cash + state.due_from_banks

    # Liquidity waterfall
    remaining_need = max(0, total_outflows - available_cash)

    # Use repo capacity
    repo_capacity = scenario.repo_limit * (1 - scenario.repo_haircut)
    repo_used = min(remaining_need, repo_capacity)
    remaining_need -= repo_used

    # Sell securities (fire-sale)
    securities_available = state.securities * (1 - scenario.sale_haircut)
    securities_sold_value = min(remaining_need, securities_available)
    # Calculate actual securities sold (before haircut) and fire-sale loss
    if securities_available > 0:
        securities_sold = securities_sold_value / (1 - scenario.sale_haircut)
        fire_sale_losses = securities_sold * scenario.sale_haircut
    else:
        securities_sold = 0.0
        fire_sale_losses = 0.0
    remaining_need -= securities_sold_value

    liquidity_gap = max(0, remaining_need)

    # --- SOLVENCY STRESS ---

    # Credit losses
    credit_losses = state.gross_loans * scenario.credit_loss_rate

    # MTM losses (if applied to capital)
    mtm_losses = 0.0
    if scenario.mtm_hits_capital:
        mtm_losses = state.securities * scenario.securities_mtm_shock

    # Retained earnings
    retained_earnings = scenario.retained_earnings

    # RWA after stress
    rwa_after = state.rwa * scenario.rwa_multiplier

    # Total capital after stress
    total_capital_after = (
        state.total_capital
        + retained_earnings
        - credit_losses
        - fire_sale_losses
        - mtm_losses
    )

    # Tier 1 capital (assume proportional impact)
    tier1_ratio = state.tier1_capital / state.total_capital if state.total_capital > 0 else 1.0
    tier1_after = total_capital_after * tier1_ratio

    # Calculate stressed ratios
    k1_2_after = tier1_after / rwa_after if rwa_after > 0 else 0.0
    k2_after = total_capital_after / rwa_after if rwa_after > 0 else 0.0

    # Capital shortfall (below K2 minimum)
    required_capital = K2_MINIMUM * rwa_after
    capital_shortfall = max(0, required_capital - total_capital_after)

    # Support needed if capital shortfall or liquidity gap
    support_needed = capital_shortfall > 0 or liquidity_gap > 0

    return BankStressResult(
        scenario_name=scenario.name,
        k1_2_before=k1_2_before,
        k1_2_after=k1_2_after,
        k2_before=k2_before,
        k2_after=k2_after,
        capital_shortfall=capital_shortfall,
        liquidity_gap=liquidity_gap,
        support_needed=support_needed,
        retained_earnings=retained_earnings,
        credit_losses=credit_losses,
        fire_sale_losses=fire_sale_losses,
        mtm_losses=mtm_losses,
        total_outflows=total_outflows,
        repo_used=repo_used,
        securities_sold=securities_sold,
        rwa_after=rwa_after,
        total_capital_after=total_capital_after,
    )


def run_bank_stress_grid(
    state: BankState,
    base_annual_income: float,
    profit_multipliers: list[float] | None = None,
    credit_loss_rates: list[float] | None = None,
    rwa_multipliers: list[float] | None = None,
    retail_run_rates: list[float] | None = None,
) -> pd.DataFrame:
    """
    Run bank stress across a grid of parameters.

    Args:
        state: Bank balance sheet
        base_annual_income: Baseline annual net income
        profit_multipliers: List of profit multipliers (default: [1.0, 0.8, 0.5, 0.0])
        credit_loss_rates: List of credit loss rates (default: [0.01, 0.02, 0.03, 0.05, 0.07])
        rwa_multipliers: List of RWA multipliers (default: [1.0, 1.05, 1.10, 1.15, 1.25])
        retail_run_rates: List of retail run rates (default: [0.05, 0.10, 0.15, 0.25, 0.30])

    Returns:
        DataFrame with stress results for each combination
    """
    if profit_multipliers is None:
        profit_multipliers = [1.0, 0.8, 0.5, 0.0]
    if credit_loss_rates is None:
        credit_loss_rates = [0.01, 0.02, 0.03, 0.05, 0.07]
    if rwa_multipliers is None:
        rwa_multipliers = [1.0, 1.05, 1.10, 1.15, 1.25]
    if retail_run_rates is None:
        retail_run_rates = [0.05, 0.10, 0.15, 0.25, 0.30]

    results = []

    for profit_mult in profit_multipliers:
        for credit_loss in credit_loss_rates:
            for rwa_mult in rwa_multipliers:
                for run_rate in retail_run_rates:
                    scenario = BankScenario(
                        name=f"P{profit_mult:.0%}_CL{credit_loss:.0%}_RWA{rwa_mult:.0%}_R{run_rate:.0%}",
                        annual_net_income=base_annual_income,
                        profit_multiplier=profit_mult,
                        credit_loss_rate=credit_loss,
                        rwa_multiplier=rwa_mult,
                        retail_run_rate=run_rate,
                        corporate_run_rate=run_rate * 0.5,  # Corporate runs less
                    )

                    result = simulate_bank_stress(state, scenario)
                    row = result.to_dict()
                    row["profit_multiplier"] = profit_mult
                    row["credit_loss_rate"] = credit_loss
                    row["rwa_multiplier"] = rwa_mult
                    row["retail_run_rate"] = run_rate
                    results.append(row)

    return pd.DataFrame(results)


def make_standard_scenarios(base_annual_income: float) -> dict[str, BankScenario]:
    """
    Create standard bank stress scenarios.

    Args:
        base_annual_income: Baseline annual net income (KZT bn)

    Returns:
        Dictionary of named scenarios
    """
    return {
        "baseline": BankScenario(
            name="Baseline",
            annual_net_income=base_annual_income,
            profit_multiplier=1.0,
            credit_loss_rate=0.01,
            rwa_multiplier=1.0,
            retail_run_rate=0.05,
            corporate_run_rate=0.025,
        ),
        "mild": BankScenario(
            name="Mild Stress",
            annual_net_income=base_annual_income,
            profit_multiplier=0.8,
            credit_loss_rate=0.02,
            rwa_multiplier=1.05,
            retail_run_rate=0.10,
            corporate_run_rate=0.05,
        ),
        "moderate": BankScenario(
            name="Moderate Stress",
            annual_net_income=base_annual_income,
            profit_multiplier=0.5,
            credit_loss_rate=0.03,
            rwa_multiplier=1.10,
            retail_run_rate=0.15,
            corporate_run_rate=0.075,
        ),
        "severe": BankScenario(
            name="Severe Stress",
            annual_net_income=base_annual_income,
            profit_multiplier=0.0,
            credit_loss_rate=0.05,
            rwa_multiplier=1.15,
            retail_run_rate=0.25,
            corporate_run_rate=0.125,
        ),
        "oil_crisis": BankScenario(
            name="Oil Crisis",
            annual_net_income=base_annual_income,
            profit_multiplier=0.0,
            credit_loss_rate=0.07,
            rwa_multiplier=1.25,
            retail_run_rate=0.30,
            corporate_run_rate=0.15,
            securities_mtm_shock=0.15,
            mtm_hits_capital=True,
        ),
    }
