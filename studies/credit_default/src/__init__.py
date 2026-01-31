"""
Credit default study source modules.

Estimates the causal effect of income changes on credit default risk
using minimum wage and pension eligibility natural experiments.
"""

from studies.credit_default.src.panel_data import LoanPanelBuilder
from studies.credit_default.src.sample_construction import SampleConstructor
from studies.credit_default.src.confound_checks import ConfoundChecker, check_confounds
from studies.credit_default.src.diff_in_discs import (
    DiffInDiscsEstimator,
    DiffInDiscsResult,
    estimate_mw_effect,
)
from studies.credit_default.src.fuzzy_rdd import (
    FuzzyRDDEstimator,
    FuzzyRDDResult,
    estimate_pension_effect,
)
from studies.credit_default.src.elasticity_store import (
    ElasticityStore,
    Elasticity,
    ElasticitySet,
)
from studies.credit_default.src.scenario_simulator import (
    CreditScenarioSimulator,
    CreditScenarioResult,
    simulate_income_shock,
)
from studies.credit_default.src.portfolio_stress import (
    PortfolioStressTester,
    PortfolioStressResult,
    PortfolioSegment,
    StressScenario,
)
from studies.credit_default.src.credit_bureau import (
    CreditBureauLoader,
    BorrowerCreditProfile,
)

__all__ = [
    # Panel construction
    "LoanPanelBuilder",
    "SampleConstructor",
    # Validation
    "ConfoundChecker",
    "check_confounds",
    # Estimation
    "DiffInDiscsEstimator",
    "DiffInDiscsResult",
    "estimate_mw_effect",
    "FuzzyRDDEstimator",
    "FuzzyRDDResult",
    "estimate_pension_effect",
    # Elasticity storage
    "ElasticityStore",
    "Elasticity",
    "ElasticitySet",
    # Simulation
    "CreditScenarioSimulator",
    "CreditScenarioResult",
    "simulate_income_shock",
    # Stress testing
    "PortfolioStressTester",
    "PortfolioStressResult",
    "PortfolioSegment",
    "StressScenario",
    # Credit bureau
    "CreditBureauLoader",
    "BorrowerCreditProfile",
]
