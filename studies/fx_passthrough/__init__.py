"""
FX-to-Expenditure Causal Chain Study (fx_passthrough)

Research Question: How do exchange rate shocks affect household welfare
in Kazakhstan through inflation, income, and expenditure channels?

Core Causal Chain:
    Exchange Rate -> Inflation -> (Nominal Income & Transfers) -> Real Income -> Expenditure

Identification Strategy:
    CPI category DiD constructs an exogenous "imported inflation" instrument.
    This powers all downstream causal claims.

Blocks:
    A: CPI Category Pass-Through (Quasi-Experimental DiD)
    B: Income Response to Externally-Driven Inflation (LP-IV)
    C: Real Income Decomposition (Accounting Identity)
    D: Transfer Mechanism Tests
    E: Expenditure Response (LP-IV)
"""

from studies.fx_passthrough.src.cpi_pass_through import (
    CPIPassThroughSpec,
    CPIPassThroughResult,
    CPIPassThroughModel,
)
from studies.fx_passthrough.src.income_lp_iv import (
    IncomeLPIVSpec,
    IncomeLPIVResult,
    IncomeLPIVModel,
)
from studies.fx_passthrough.src.causal_chain import CausalChainAnalysis

__all__ = [
    "CPIPassThroughSpec",
    "CPIPassThroughResult",
    "CPIPassThroughModel",
    "IncomeLPIVSpec",
    "IncomeLPIVResult",
    "IncomeLPIVModel",
    "CausalChainAnalysis",
]
