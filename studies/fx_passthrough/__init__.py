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
    F: Spending Response to FX-Driven Purchasing Power Shocks (LP-IRF + MPC Ratio)

Block F Note:
    MPC-like ratio = IRF_C(h) / IRF_Y(0)
    CAVEAT: This is NOT a universal MPC. It captures spending response to
    externally-driven purchasing power shocks (imported inflation via FX).
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
from studies.fx_passthrough.src.spending_response import (
    SpendingResponseSpec,
    SpendingResponseResult,
    SpendingResponseModel,
    ShockType,
)
from studies.fx_passthrough.src.depreciation_backtest import (
    DepreciationBacktest,
    DepreciationBacktestSpec,
    DepreciationEvent,
    BacktestResult,
)
from studies.fx_passthrough.src.block_f_falsification import (
    BlockFFalsification,
    BlockFFalsificationResults,
    FalsificationTestResult,
)
from studies.fx_passthrough.src.causal_chain import CausalChainAnalysis

__all__ = [
    # Block A
    "CPIPassThroughSpec",
    "CPIPassThroughResult",
    "CPIPassThroughModel",
    # Block B
    "IncomeLPIVSpec",
    "IncomeLPIVResult",
    "IncomeLPIVModel",
    # Block F
    "SpendingResponseSpec",
    "SpendingResponseResult",
    "SpendingResponseModel",
    "ShockType",
    # Block F Backtest
    "DepreciationBacktest",
    "DepreciationBacktestSpec",
    "DepreciationEvent",
    "BacktestResult",
    # Block F Falsification
    "BlockFFalsification",
    "BlockFFalsificationResults",
    "FalsificationTestResult",
    # Orchestrator
    "CausalChainAnalysis",
]
