"""
Holding company state and scenario dataclasses.

Represents the parent company (holdco) balance sheet and cash flow scenarios
for Kaspi.kz capital adequacy analysis.
"""

from dataclasses import dataclass


@dataclass
class HoldCoState:
    """
    Holding company balance sheet snapshot (KZT billions).

    Represents the parent company's stand-alone financial position,
    separate from consolidated group financials.

    Attributes:
        cash: Parent cash and equivalents
        annual_gna: Annual general & administrative expenses
        annual_interest: Annual interest expense on parent debt
        other_fixed: Other fixed obligations (annual)
        parent_liabilities: Total parent liabilities
    """

    cash: float
    annual_gna: float
    annual_interest: float = 0.0
    other_fixed: float = 0.0
    parent_liabilities: float = 0.0

    @property
    def annual_fixed_costs(self) -> float:
        """Total annual fixed costs (G&A + interest + other)."""
        return self.annual_gna + self.annual_interest + self.other_fixed

    @property
    def monthly_fixed_costs(self) -> float:
        """Monthly fixed costs."""
        return self.annual_fixed_costs / 12

    @property
    def months_of_runway_standalone(self) -> float:
        """
        Months of runway with no inflows.

        How long can holdco survive on cash alone with no dividends?
        """
        if self.monthly_fixed_costs <= 0:
            return float("inf")
        return self.cash / self.monthly_fixed_costs

    @classmethod
    def from_fy2024(cls) -> "HoldCoState":
        """Create HoldCoState from FY2024 20-F data."""
        return cls(
            cash=324.993,
            annual_gna=20.810,
            annual_interest=0.0,  # Minimal parent debt
            other_fixed=0.0,
            parent_liabilities=0.155,
        )

    def summary(self) -> str:
        """Generate formatted summary of holdco state."""
        lines = [
            "=" * 60,
            "HoldCo State Summary (KZT bn)",
            "=" * 60,
            "",
            "Balance Sheet:",
            f"  Cash:        {self.cash:.1f}",
            f"  Liabilities: {self.parent_liabilities:.3f}",
            "",
            "Annual Fixed Costs:",
            f"  G&A:         {self.annual_gna:.1f}",
            f"  Interest:    {self.annual_interest:.1f}",
            f"  Other:       {self.other_fixed:.1f}",
            f"  Total:       {self.annual_fixed_costs:.1f}",
            "",
            f"Monthly Fixed: {self.monthly_fixed_costs:.2f}",
            f"Standalone Runway: {self.months_of_runway_standalone:.0f} months",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class HoldCoScenario:
    """
    Cash flow scenario for holding company simulation.

    Represents a 12-month cash flow projection with inflows (dividends,
    capital raises) and outflows (shareholder dividends, buybacks,
    capital injections, acquisitions).

    Attributes:
        name: Scenario identifier
        dividend_from_bank: Dividend received from bank subsidiary
        dividend_from_other_subs: Dividends from non-bank subsidiaries
        dividends_to_shareholders: Dividends paid to public shareholders
        buybacks: Share buyback program spending
        capital_injection_to_bank: Capital injection to bank (if K2 shortfall)
        acquisition_cash_outflow: Cash for acquisitions
        new_equity_raised: New equity capital raised
        new_debt_raised: New debt capital raised
    """

    name: str
    dividend_from_bank: float
    dividend_from_other_subs: float
    dividends_to_shareholders: float = 0.0
    buybacks: float = 0.0
    capital_injection_to_bank: float = 0.0
    acquisition_cash_outflow: float = 0.0
    new_equity_raised: float = 0.0
    new_debt_raised: float = 0.0

    @property
    def total_inflows(self) -> float:
        """Total cash inflows from subsidiaries and capital raises."""
        return (
            self.dividend_from_bank
            + self.dividend_from_other_subs
            + self.new_equity_raised
            + self.new_debt_raised
        )

    @property
    def total_discretionary_outflows(self) -> float:
        """Total discretionary outflows (can be reduced in stress)."""
        return self.dividends_to_shareholders + self.buybacks + self.acquisition_cash_outflow

    @property
    def total_required_outflows(self) -> float:
        """Total required outflows (capital injections)."""
        return self.capital_injection_to_bank

    @property
    def net_cash_flow_before_fixed(self) -> float:
        """Net cash flow before fixed costs."""
        return self.total_inflows - self.total_discretionary_outflows - self.total_required_outflows

    @classmethod
    def from_fy2024_baseline(cls) -> "HoldCoScenario":
        """Create baseline scenario from FY2024 actuals."""
        return cls(
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

    def summary(self) -> str:
        """Generate formatted summary of scenario."""
        lines = [
            f"Scenario: {self.name}",
            "-" * 40,
            "Inflows (KZT bn):",
            f"  Bank Dividend:     {self.dividend_from_bank:.1f}",
            f"  Other Sub Divs:    {self.dividend_from_other_subs:.1f}",
            f"  New Equity:        {self.new_equity_raised:.1f}",
            f"  New Debt:          {self.new_debt_raised:.1f}",
            f"  Total Inflows:     {self.total_inflows:.1f}",
            "",
            "Outflows (KZT bn):",
            f"  Shareholder Divs:  {self.dividends_to_shareholders:.1f}",
            f"  Buybacks:          {self.buybacks:.1f}",
            f"  Acquisitions:      {self.acquisition_cash_outflow:.1f}",
            f"  Capital Injection: {self.capital_injection_to_bank:.1f}",
            "",
            f"Net (before fixed): {self.net_cash_flow_before_fixed:.1f}",
        ]
        return "\n".join(lines)
