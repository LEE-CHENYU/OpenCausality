"""
Agentic DAG orchestration for causal econometric analysis.

This module provides infrastructure for:
- DAG specification parsing and validation
- Dependency-aware task scheduling
- Credibility-based design selection
- Governance and audit logging
- Agent loop orchestration
"""

from shared.agentic.dag.parser import (
    DAGSpec,
    NodeSpec,
    EdgeSpec,
    EdgeTiming,
    ReleaseLag,
    IdentityDef,
    EstimandSpec,
    AcceptanceCriteria,
    parse_dag,
)
from shared.agentic.dag.validator import DAGValidator, ValidationReport

__all__ = [
    "DAGSpec",
    "NodeSpec",
    "EdgeSpec",
    "EdgeTiming",
    "ReleaseLag",
    "IdentityDef",
    "EstimandSpec",
    "AcceptanceCriteria",
    "parse_dag",
    "DAGValidator",
    "ValidationReport",
]
