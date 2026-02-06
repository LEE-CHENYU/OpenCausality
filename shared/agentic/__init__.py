"""
Agentic DAG orchestration for causal econometric analysis.

This module provides infrastructure for:
- DAG specification parsing and validation
- Dependency-aware task scheduling
- Credibility-based design selection
- Governance and audit logging
- Agent loop orchestration
- Post-estimation validation (v3)
- Report consistency checking (v3)
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

# V3: Post-estimation validation and report consistency
from shared.agentic.validation import (
    DAGValidator as PostEstimationValidator,
    ValidationResult,
    ValidationIssue,
    run_full_validation,
)
from shared.agentic.report_checker import (
    ReportConsistencyChecker,
    ReportCheckResult,
    check_report_consistency,
)

__all__ = [
    # DAG parsing
    "DAGSpec",
    "NodeSpec",
    "EdgeSpec",
    "EdgeTiming",
    "ReleaseLag",
    "IdentityDef",
    "EstimandSpec",
    "AcceptanceCriteria",
    "parse_dag",
    # Pre-estimation validation
    "DAGValidator",
    "ValidationReport",
    # Post-estimation validation (v3)
    "PostEstimationValidator",
    "ValidationResult",
    "ValidationIssue",
    "run_full_validation",
    # Report consistency (v3)
    "ReportConsistencyChecker",
    "ReportCheckResult",
    "check_report_consistency",
]
