"""
Pre-defined scenarios and FY2024 baseline data.

This module contains:
- FY2024 baseline data from 20-F filing
- Standard bank stress scenarios
- Standard holdco scenarios
"""

from studies.kaspi_holdco.src.bank_state import BankState, BankScenario
from studies.kaspi_holdco.src.holdco_state import HoldCoState, HoldCoScenario


# =============================================================================
# FY2024 BASELINE DATA (from 20-F filing)
# =============================================================================

# Annual net income (approximate, for stress scenarios)
FY2024_ANNUAL_NET_INCOME = 450.0  # KZT bn (approximate)

FY2024_BANK_STATE = BankState(
    # Regulatory (NBK-style)
    rwa=8059.0,  # Risk-weighted assets (NBK methodology)
    tier1_capital=1016.0,  # Tier 1 capital
    total_capital=1027.0,  # Total capital
    # Funding / liquidity
    retail_deposits=6250.0,
    corporate_deposits=312.0,
    other_short_term_funding=0.0,
    # Liquid assets
    cash=619.0,
    mandatory_cash=57.0,
    due_from_banks=38.0,
    securities=1507.0,
    # Credit base
    gross_loans=5747.0,
)

FY2024_HOLDCO_STATE = HoldCoState(
    cash=324.993,
    annual_gna=20.810,
    annual_interest=0.0,
    other_fixed=0.0,
    parent_liabilities=0.155,
)

FY2024_BASELINE_SCENARIO = HoldCoScenario(
    name="FY2024 Baseline",
    dividend_from_bank=285.206,
    dividend_from_other_subs=467.500,
    dividends_to_shareholders=646.056,
    buybacks=0.0,
    capital_injection_to_bank=0.0,
    acquisition_cash_outflow=0.0,
    new_equity_raised=0.0,
    new_debt_raised=0.0,
)


# =============================================================================
# BANK STRESS SCENARIOS
# =============================================================================
# K2 minimum = 12.0%, baseline = 12.7%, headroom = ~60 KZT bn

BANK_STRESS_SCENARIOS: dict[str, BankScenario] = {
    "baseline": BankScenario(
        name="Baseline",
        annual_net_income=FY2024_ANNUAL_NET_INCOME,
        profit_multiplier=1.0,
        payout_ratio=0.0,  # Retain all for stress analysis
        credit_loss_rate=0.01,  # 1% credit losses
        rwa_multiplier=1.0,
        retail_run_rate=0.05,  # 5% retail run
        corporate_run_rate=0.025,
    ),
    "mild": BankScenario(
        name="Mild Stress",
        annual_net_income=FY2024_ANNUAL_NET_INCOME,
        profit_multiplier=0.8,  # 20% earnings compression
        payout_ratio=0.0,
        credit_loss_rate=0.02,  # 2% credit losses
        rwa_multiplier=1.05,  # +5% RWA inflation
        retail_run_rate=0.10,
        corporate_run_rate=0.05,
    ),
    "moderate": BankScenario(
        name="Moderate Stress",
        annual_net_income=FY2024_ANNUAL_NET_INCOME,
        profit_multiplier=0.5,  # 50% earnings compression
        payout_ratio=0.0,
        credit_loss_rate=0.03,  # 3% credit losses
        rwa_multiplier=1.10,  # +10% RWA inflation
        retail_run_rate=0.15,
        corporate_run_rate=0.075,
    ),
    "severe": BankScenario(
        name="Severe Stress",
        annual_net_income=FY2024_ANNUAL_NET_INCOME,
        profit_multiplier=0.0,  # Zero profit
        payout_ratio=0.0,
        credit_loss_rate=0.05,  # 5% credit losses
        rwa_multiplier=1.15,  # +15% RWA inflation
        retail_run_rate=0.25,
        corporate_run_rate=0.125,
    ),
    "oil_crisis": BankScenario(
        name="Oil Crisis",
        annual_net_income=FY2024_ANNUAL_NET_INCOME,
        profit_multiplier=0.0,  # Zero profit
        payout_ratio=0.0,
        credit_loss_rate=0.07,  # 7% credit losses
        rwa_multiplier=1.25,  # +25% RWA inflation
        retail_run_rate=0.30,
        corporate_run_rate=0.15,
        securities_mtm_shock=0.15,  # 15% MTM loss
        mtm_hits_capital=True,
    ),
}


# =============================================================================
# HOLDCO SCENARIOS
# =============================================================================

HOLDCO_SCENARIOS: dict[str, HoldCoScenario] = {
    "baseline": HoldCoScenario(
        name="Baseline (FY2024 Actuals)",
        dividend_from_bank=285.206,
        dividend_from_other_subs=467.500,
        dividends_to_shareholders=646.056,
        buybacks=0.0,
        capital_injection_to_bank=0.0,
    ),
    "no_bank_dividend": HoldCoScenario(
        name="No Bank Dividend",
        dividend_from_bank=0.0,
        dividend_from_other_subs=467.500,
        dividends_to_shareholders=646.056,  # Keep shareholder dividend
        buybacks=0.0,
        capital_injection_to_bank=0.0,
    ),
    "no_dividends": HoldCoScenario(
        name="No Subsidiary Dividends",
        dividend_from_bank=0.0,
        dividend_from_other_subs=0.0,
        dividends_to_shareholders=0.0,  # No payout either
        buybacks=0.0,
        capital_injection_to_bank=0.0,
    ),
    "dividend_stop_no_payout": HoldCoScenario(
        name="Full Dividend Freeze",
        dividend_from_bank=0.0,
        dividend_from_other_subs=467.500,  # Other subs still pay
        dividends_to_shareholders=0.0,
        buybacks=0.0,
        capital_injection_to_bank=0.0,
    ),
    "moderate_capital_call": HoldCoScenario(
        name="Moderate Capital Call",
        dividend_from_bank=142.603,  # 50% of baseline
        dividend_from_other_subs=467.500,
        dividends_to_shareholders=0.0,  # Cut shareholder dividend
        buybacks=0.0,
        capital_injection_to_bank=100.0,  # 100 bn injection
    ),
    "severe_capital_call": HoldCoScenario(
        name="Severe Capital Call",
        dividend_from_bank=0.0,  # No bank dividend
        dividend_from_other_subs=467.500,
        dividends_to_shareholders=0.0,
        buybacks=0.0,
        capital_injection_to_bank=200.0,  # 200 bn injection
    ),
    "extreme_capital_call": HoldCoScenario(
        name="Extreme Capital Call",
        dividend_from_bank=0.0,
        dividend_from_other_subs=467.500,
        dividends_to_shareholders=0.0,
        buybacks=0.0,
        capital_injection_to_bank=400.0,  # 400 bn injection
    ),
}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_bank_state() -> BankState:
    """Get FY2024 bank state."""
    return FY2024_BANK_STATE


def get_holdco_state() -> HoldCoState:
    """Get FY2024 holdco state."""
    return FY2024_HOLDCO_STATE


def get_baseline_scenario() -> HoldCoScenario:
    """Get FY2024 baseline holdco scenario."""
    return FY2024_BASELINE_SCENARIO


def list_bank_scenarios() -> list[str]:
    """List available bank stress scenarios."""
    return list(BANK_STRESS_SCENARIOS.keys())


def list_holdco_scenarios() -> list[str]:
    """List available holdco scenarios."""
    return list(HOLDCO_SCENARIOS.keys())


def get_bank_scenario(name: str) -> BankScenario:
    """Get bank scenario by name."""
    if name not in BANK_STRESS_SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list_bank_scenarios()}")
    return BANK_STRESS_SCENARIOS[name]


def get_holdco_scenario(name: str) -> HoldCoScenario:
    """Get holdco scenario by name."""
    if name not in HOLDCO_SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list_holdco_scenarios()}")
    return HOLDCO_SCENARIOS[name]
