"""
Credit quality study source modules.

REVISED DESIGN (v2.0): Uses aggregate NBK credit data with external shocks.
This is REDUCED-FORM analysis: external shocks → credit quality.

Original micro design (loan-level diff-in-discs, fuzzy RDD) requires
internal fintech data that is not available. Those modules are retained
for future use if micro data becomes available.

Current analysis:
- credit_data_pipeline.py: Fetches NBK, IMF, FRED data
- credit_lp.py: Local projections for shock → credit quality IRFs
"""

# =============================================================================
# AGGREGATE / REDUCED-FORM MODULES (Active)
# =============================================================================

from studies.credit_default.src.credit_data_pipeline import (
    CreditDataPipeline,
    build_credit_panel,
)
from studies.credit_default.src.credit_lp import (
    CreditLocalProjections,
    CreditLPResults,
    LPResult,
    estimate_credit_lp,
)

# =============================================================================
# MICRO DESIGN MODULES (Retained for future use with loan-level data)
# =============================================================================

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
    # ==========================================================================
    # AGGREGATE ANALYSIS (v2.0 - Active)
    # ==========================================================================
    # Data pipeline
    "CreditDataPipeline",
    "build_credit_panel",
    # Local projections
    "CreditLocalProjections",
    "CreditLPResults",
    "LPResult",
    "estimate_credit_lp",

    # ==========================================================================
    # MICRO DESIGN (Retained for future use)
    # ==========================================================================
    # Panel construction
    "LoanPanelBuilder",
    "SampleConstructor",
    # Validation
    "ConfoundChecker",
    "check_confounds",
    # Estimation (micro)
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
