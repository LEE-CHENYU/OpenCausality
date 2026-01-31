"""
Shared model infrastructure for Kazakhstan econometric research.

Contains:
- inference.py: Driscoll-Kraay, clustered SE, wild bootstrap
- event_study.py: Event study utilities
- diagnostics.py: First-stage F, McCrary, balance tests
"""

from shared.model.inference import BHJInference, BHJResult, compare_inference_methods
from shared.model.event_study import EventStudy, EventStudyResult
from shared.model.diagnostics import (
    DiagnosticSuite,
    first_stage_f_test,
    mccrary_density_test,
    balance_test,
    placebo_cutoff_test,
)

__all__ = [
    "BHJInference",
    "BHJResult",
    "compare_inference_methods",
    "EventStudy",
    "EventStudyResult",
    "DiagnosticSuite",
    "first_stage_f_test",
    "mccrary_density_test",
    "balance_test",
    "placebo_cutoff_test",
]
