"""
Bank-level state and scenario dataclasses for Kaspi.kz stress testing.

This module defines the data structures for representing the bank's balance sheet
and regulatory position, as well as stress scenarios that can be applied.
"""

from dataclasses import dataclass


@dataclass
class BankState:
    """
    Bank balance sheet + regulatory snapshot (KZT billions).

    Represents the key metrics for NBK-style capital adequacy analysis.
    All values are in KZT billions unless otherwise noted.

    Attributes:
        rwa: Risk-weighted assets (NBK methodology, ~45% higher than Basel III)
        tier1_capital: Tier 1 capital
        total_capital: Total regulatory capital
        retail_deposits: Retail customer deposits
        corporate_deposits: Corporate customer deposits
        other_short_term_funding: Other short-term funding sources
        cash: Cash and cash equivalents
        mandatory_cash: Mandatory reserves at NBK
        due_from_banks: Amounts due from other banks
        securities: Investment securities portfolio
        gross_loans: Gross loan portfolio
    """

    # Regulatory (NBK-style)
    rwa: float
    tier1_capital: float
    total_capital: float

    # Funding / liquidity
    retail_deposits: float
    corporate_deposits: float
    other_short_term_funding: float = 0.0

    # Liquid assets
    cash: float = 0.0
    mandatory_cash: float = 0.0
    due_from_banks: float = 0.0
    securities: float = 0.0

    # Credit base
    gross_loans: float = 0.0

    @property
    def k1_2_ratio(self) -> float:
        """K1-2 ratio (Tier 1 capital / RWA). Minimum: 10.5%."""
        if self.rwa <= 0:
            return 0.0
        return self.tier1_capital / self.rwa

    @property
    def k2_ratio(self) -> float:
        """K2 ratio (Total capital / RWA). Minimum: 12.0%."""
        if self.rwa <= 0:
            return 0.0
        return self.total_capital / self.rwa

    @property
    def total_deposits(self) -> float:
        """Total deposits (retail + corporate)."""
        return self.retail_deposits + self.corporate_deposits

    @property
    def total_funding(self) -> float:
        """Total funding (deposits + other short-term)."""
        return self.total_deposits + self.other_short_term_funding

    @property
    def available_liquidity(self) -> float:
        """Available liquid assets (cash + due from banks + securities - mandatory)."""
        return self.cash + self.due_from_banks + self.securities - self.mandatory_cash

    @property
    def k1_2_headroom(self) -> float:
        """K1-2 headroom above 10.5% minimum (KZT bn)."""
        return max(0, self.tier1_capital - 0.105 * self.rwa)

    @property
    def k2_headroom(self) -> float:
        """K2 headroom above 12.0% minimum (KZT bn)."""
        return max(0, self.total_capital - 0.12 * self.rwa)

    def summary(self) -> str:
        """Generate formatted summary of bank state."""
        lines = [
            "=" * 60,
            "Bank State Summary (KZT bn)",
            "=" * 60,
            "",
            "Regulatory Ratios:",
            f"  K1-2 Ratio: {self.k1_2_ratio:.1%} (min 10.5%)",
            f"  K2 Ratio:   {self.k2_ratio:.1%} (min 12.0%)",
            f"  K1-2 Headroom: {self.k1_2_headroom:.1f} bn",
            f"  K2 Headroom:   {self.k2_headroom:.1f} bn",
            "",
            "Capital:",
            f"  Tier 1 Capital: {self.tier1_capital:.1f}",
            f"  Total Capital:  {self.total_capital:.1f}",
            f"  RWA:            {self.rwa:.1f}",
            "",
            "Funding:",
            f"  Retail Deposits:    {self.retail_deposits:.1f}",
            f"  Corporate Deposits: {self.corporate_deposits:.1f}",
            f"  Other Short-Term:   {self.other_short_term_funding:.1f}",
            f"  Total Funding:      {self.total_funding:.1f}",
            "",
            "Liquidity:",
            f"  Cash:            {self.cash:.1f}",
            f"  Mandatory Cash:  {self.mandatory_cash:.1f}",
            f"  Due from Banks:  {self.due_from_banks:.1f}",
            f"  Securities:      {self.securities:.1f}",
            f"  Available:       {self.available_liquidity:.1f}",
            "",
            "Credit:",
            f"  Gross Loans: {self.gross_loans:.1f}",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class BankScenario:
    """
    Stress scenario for bank-level simulation.

    Combines earnings stress, credit loss stress, RWA inflation,
    and liquidity stress into a single scenario.

    Attributes:
        name: Scenario identifier
        annual_net_income: Baseline annual net income (KZT bn)
        profit_multiplier: Multiplier for earnings (0.5 = 50% compression)
        payout_ratio: Dividend payout ratio (0 in stress = retain all)
        credit_loss_rate: Credit losses as % of gross loans (0.05 = 5%)
        rwa_multiplier: RWA inflation factor (1.10 = +10%)
        securities_mtm_shock: Mark-to-market shock on securities (0.05 = -5%)
        mtm_hits_capital: Whether MTM losses impact regulatory capital
        retail_run_rate: 30-day retail deposit outflow rate
        corporate_run_rate: 30-day corporate deposit outflow rate
        wholesale_outflow: Other short-term funding outflow (absolute, KZT bn)
        repo_limit: NBK/ELA repo capacity (KZT bn)
        repo_haircut: Haircut on repo collateral
        sale_haircut: Fire-sale discount on securities
    """

    name: str
    annual_net_income: float
    profit_multiplier: float = 1.0
    payout_ratio: float = 0.0
    credit_loss_rate: float = 0.0
    rwa_multiplier: float = 1.0

    # Securities / fire-sale
    securities_mtm_shock: float = 0.0
    mtm_hits_capital: bool = False

    # 30-day liquidity stress
    retail_run_rate: float = 0.0
    corporate_run_rate: float = 0.0
    wholesale_outflow: float = 0.0

    # Liquidity tools
    repo_limit: float = 0.0
    repo_haircut: float = 0.20
    sale_haircut: float = 0.10

    @property
    def stressed_income(self) -> float:
        """Net income after profit compression."""
        return self.annual_net_income * self.profit_multiplier

    @property
    def retained_earnings(self) -> float:
        """Retained earnings after payout."""
        return self.stressed_income * (1 - self.payout_ratio)

    def summary(self) -> str:
        """Generate formatted summary of scenario."""
        lines = [
            f"Scenario: {self.name}",
            "-" * 40,
            f"  Profit Multiplier: {self.profit_multiplier:.0%}",
            f"  Credit Loss Rate:  {self.credit_loss_rate:.1%}",
            f"  RWA Multiplier:    {self.rwa_multiplier:.2f}x",
            f"  Retail Run Rate:   {self.retail_run_rate:.0%}",
            f"  Corporate Run:     {self.corporate_run_rate:.0%}",
        ]
        return "\n".join(lines)
