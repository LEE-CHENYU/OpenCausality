"""
Source modules for FX Passthrough study.

Blocks:
    A: cpi_pass_through - CPI category DiD
    B: income_lp_iv - Income LP-IV
    C: accounting_identity - Real income decomposition
    D: transfer_mechanism - Transfer tests
    E: expenditure_response - Expenditure LP-IV
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
from studies.fx_passthrough.src.accounting_identity import (
    AccountingIdentity,
    RealIncomeDecomposition,
)
from studies.fx_passthrough.src.transfer_mechanism import (
    TransferMechanismSpec,
    TransferMechanismResult,
    TransferMechanismModel,
)
from studies.fx_passthrough.src.expenditure_response import (
    ExpenditureLPIVSpec,
    ExpenditureLPIVResult,
    ExpenditureLPIVModel,
)
from studies.fx_passthrough.src.causal_chain import CausalChainAnalysis

__all__ = [
    "CPIPassThroughSpec",
    "CPIPassThroughResult",
    "CPIPassThroughModel",
    "IncomeLPIVSpec",
    "IncomeLPIVResult",
    "IncomeLPIVModel",
    "AccountingIdentity",
    "RealIncomeDecomposition",
    "TransferMechanismSpec",
    "TransferMechanismResult",
    "TransferMechanismModel",
    "ExpenditureLPIVSpec",
    "ExpenditureLPIVResult",
    "ExpenditureLPIVModel",
    "CausalChainAnalysis",
]
