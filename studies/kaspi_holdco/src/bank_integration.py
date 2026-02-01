"""
Bank-to-HoldCo integration module.

Translates bank-level stress results (K2 shortfall) into holdco scenarios
(dividend cuts, capital calls).
"""

from dataclasses import dataclass

from studies.kaspi_holdco.src.bank_stress import BankStressResult
from studies.kaspi_holdco.src.holdco_state import HoldCoScenario


@dataclass
class PassthroughConfig:
    """
    Parameterized mapping from bank stress to holdco impact.

    The dividend payout formula uses buffer-based interpolation:
        payout = clip((k2 - k2_min) / (k2_baseline - k2_min), 0, 1) ^ gamma

    This avoids inventing arbitrary thresholds and ties directly to
    the NBK regulatory framework.

    Attributes:
        k2_baseline: FY2024 actual K2 ratio (12.7%)
        k2_minimum: NBK regulatory minimum (12.0%)
        gamma: Dividend payout curvature (1=linear, >1=convex/conservative)
        capital_call_multiplier: Buffer/add-on for capital calls (1.0 = exact shortfall)
    """

    k2_baseline: float = 0.127  # 12.7%
    k2_minimum: float = 0.12  # 12.0%
    gamma: float = 1.0  # Linear by default
    capital_call_multiplier: float = 1.0  # No buffer by default

    @property
    def k2_buffer(self) -> float:
        """K2 buffer above minimum (70 bps in baseline)."""
        return self.k2_baseline - self.k2_minimum


def compute_dividend_payout_fraction(
    k2_stressed: float,
    config: PassthroughConfig | None = None,
) -> float:
    """
    Compute dividend payout fraction based on stressed K2 ratio.

    Uses buffer-based interpolation:
        buffer_used = (k2_baseline - k2_stressed) / (k2_baseline - k2_min)
        buffer_remaining = clip(1 - buffer_used, 0, 1)
        payout = buffer_remaining ^ gamma

    Interpretation:
        - At baseline K2 (12.7%): payout = 100%
        - At minimum K2 (12.0%): payout = 0%
        - Linear interpolation between (or convex with gamma > 1)

    Args:
        k2_stressed: K2 ratio after stress
        config: Passthrough configuration (uses defaults if None)

    Returns:
        Dividend payout fraction [0, 1]
    """
    if config is None:
        config = PassthroughConfig()

    if config.k2_buffer <= 0:
        # Edge case: baseline = minimum, so any stress means no dividend
        return 0.0 if k2_stressed < config.k2_baseline else 1.0

    # Calculate buffer usage
    buffer_used = (config.k2_baseline - k2_stressed) / config.k2_buffer
    buffer_remaining = max(0.0, min(1.0, 1.0 - buffer_used))

    # Apply curvature
    return buffer_remaining**config.gamma


def compute_capital_call(
    shortfall: float,
    config: PassthroughConfig | None = None,
) -> float:
    """
    Compute capital call amount from shortfall.

    Capital call = shortfall × multiplier

    The multiplier accounts for:
        - Supervisory buffers beyond minimum
        - Remediation costs
        - Buffer rebuilding cushion

    Args:
        shortfall: Capital shortfall below K2 minimum (KZT bn)
        config: Passthrough configuration (uses defaults if None)

    Returns:
        Capital call amount (KZT bn)
    """
    if config is None:
        config = PassthroughConfig()

    if shortfall <= 0:
        return 0.0

    return shortfall * config.capital_call_multiplier


def build_holdco_scenario_from_bank_stress(
    bank_result: BankStressResult,
    baseline_scenario: HoldCoScenario,
    config: PassthroughConfig | None = None,
    cut_shareholder_dividends: bool = True,
) -> HoldCoScenario:
    """
    Translate bank stress result into holdco scenario.

    Logic:
        1. Compute dividend payout fraction from stressed K2
        2. Apply fraction to baseline bank dividend
        3. Compute capital call from shortfall
        4. Optionally cut shareholder dividends in stress

    Args:
        bank_result: Result from bank stress simulation
        baseline_scenario: Baseline holdco scenario (FY2024 actuals)
        config: Passthrough configuration (uses defaults if None)
        cut_shareholder_dividends: Whether to cut shareholder dividends
            when bank dividend is reduced (default: True)

    Returns:
        HoldCoScenario reflecting bank stress impact
    """
    if config is None:
        config = PassthroughConfig()

    # Dividend payout fraction
    payout_frac = compute_dividend_payout_fraction(bank_result.k2_after, config)

    # Bank dividend after haircut
    bank_dividend = baseline_scenario.dividend_from_bank * payout_frac

    # Capital call
    capital_call = compute_capital_call(bank_result.capital_shortfall, config)

    # Shareholder dividend treatment
    if cut_shareholder_dividends and (payout_frac < 1.0 or capital_call > 0):
        # Cut shareholder dividends proportionally or fully in severe stress
        if capital_call > 0:
            # Full cut if capital call needed
            shareholder_div = 0.0
        else:
            # Proportional cut if dividend reduced
            shareholder_div = baseline_scenario.dividends_to_shareholders * payout_frac
    else:
        shareholder_div = baseline_scenario.dividends_to_shareholders

    return HoldCoScenario(
        name=f"Bank Stress: {bank_result.scenario_name}",
        dividend_from_bank=bank_dividend,
        dividend_from_other_subs=baseline_scenario.dividend_from_other_subs,
        dividends_to_shareholders=shareholder_div,
        buybacks=0.0,  # No buybacks in stress
        capital_injection_to_bank=capital_call,
        acquisition_cash_outflow=0.0,  # No acquisitions in stress
        new_equity_raised=baseline_scenario.new_equity_raised,
        new_debt_raised=baseline_scenario.new_debt_raised,
    )


def analyze_passthrough_sensitivity(
    bank_result: BankStressResult,
    baseline_scenario: HoldCoScenario,
    gammas: list[float] | None = None,
    multipliers: list[float] | None = None,
) -> list[dict]:
    """
    Analyze sensitivity of holdco scenario to passthrough parameters.

    Args:
        bank_result: Bank stress result
        baseline_scenario: Baseline holdco scenario
        gammas: List of gamma values to test
        multipliers: List of capital call multipliers to test

    Returns:
        List of dictionaries with scenario details
    """
    if gammas is None:
        gammas = [0.5, 1.0, 1.5, 2.0]
    if multipliers is None:
        multipliers = [1.0, 1.25, 1.5]

    results = []

    for gamma in gammas:
        for mult in multipliers:
            config = PassthroughConfig(gamma=gamma, capital_call_multiplier=mult)

            holdco_scenario = build_holdco_scenario_from_bank_stress(
                bank_result, baseline_scenario, config
            )

            results.append(
                {
                    "gamma": gamma,
                    "capital_call_multiplier": mult,
                    "payout_fraction": compute_dividend_payout_fraction(
                        bank_result.k2_after, config
                    ),
                    "bank_dividend": holdco_scenario.dividend_from_bank,
                    "capital_call": holdco_scenario.capital_injection_to_bank,
                    "shareholder_dividend": holdco_scenario.dividends_to_shareholders,
                }
            )

    return results
