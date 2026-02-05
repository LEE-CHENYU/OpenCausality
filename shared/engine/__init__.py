"""
Shared engine infrastructure for Kazakhstan econometric research.

Contains:
- scenario_base.py: Base scenario simulator class
- data_assembler.py: DAG node-to-data mapping and edge data assembly
- ts_estimator.py: Time-series LP and identity estimators
"""

from shared.engine.scenario_base import ScenarioBase, ScenarioResult

__all__ = [
    "ScenarioBase",
    "ScenarioResult",
]
