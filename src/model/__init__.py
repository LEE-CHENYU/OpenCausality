"""
Econometric model modules.
"""

from src.model.panel_data import PanelBuilder, REGION_CROSSWALK, CANONICAL_REGIONS
from src.model.shift_share import ShiftShareModel, ShiftShareSpec, estimate_shift_share
from src.model.local_projections import LocalProjections, LocalProjectionSpec, estimate_local_projections
from src.model.inference import BHJInference, compare_inference_methods
from src.model.falsification import FalsificationTests, run_falsification_suite

__all__ = [
    "PanelBuilder",
    "REGION_CROSSWALK",
    "CANONICAL_REGIONS",
    "ShiftShareModel",
    "ShiftShareSpec",
    "estimate_shift_share",
    "LocalProjections",
    "LocalProjectionSpec",
    "estimate_local_projections",
    "BHJInference",
    "compare_inference_methods",
    "FalsificationTests",
    "run_falsification_suite",
]
