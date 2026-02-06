"""
Issue Gates: Conditions that block mode transitions or require human input.

Evaluates the current issue state against gate conditions defined
in config/agentic/issue_gates.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from shared.agentic.issues.issue_ledger import Issue, IssueLedger

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of evaluating a single gate."""

    gate_name: str
    triggered: bool
    action: str
    blocking_issues: list[Issue]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "triggered": self.triggered,
            "action": self.action,
            "blocking_issue_count": len(self.blocking_issues),
            "blocking_issues": [i.issue_key for i in self.blocking_issues],
            "description": self.description,
        }


@dataclass
class GateEvaluation:
    """Result of evaluating all gates."""

    results: list[GateResult]
    can_proceed: bool
    can_promote_to_confirmation: bool
    requires_hitl: bool
    auto_fixable_count: int

    def summary(self) -> str:
        lines = [
            f"Gate Evaluation: {'PASS' if self.can_proceed else 'BLOCKED'}",
            f"  Can promote to CONFIRMATION: {self.can_promote_to_confirmation}",
            f"  Requires HITL: {self.requires_hitl}",
            f"  Auto-fixable issues: {self.auto_fixable_count}",
        ]
        for result in self.results:
            if result.triggered:
                lines.append(f"  TRIGGERED: {result.gate_name} -> {result.action}")
                for issue in result.blocking_issues[:3]:
                    lines.append(f"    - [{issue.severity}] {issue.rule_id}: {issue.message}")
        return "\n".join(lines)


class IssueGates:
    """
    Evaluate gate conditions against current issue state.

    Gates control:
    - Whether CONFIRMATION mode can be entered
    - Whether the loop must pause for human input
    - Whether auto-fixes can be applied
    """

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or Path("config/agentic/issue_gates.yaml")

    def evaluate(
        self,
        ledger: IssueLedger,
        current_mode: str = "EXPLORATION",
    ) -> GateEvaluation:
        """Evaluate all gates against current issue state."""
        open_issues = ledger.get_open_issues()
        results = []

        # Gate 1: block_confirmation
        critical_open = [i for i in open_issues if i.severity == "CRITICAL"]
        results.append(GateResult(
            gate_name="block_confirmation",
            triggered=len(critical_open) > 0,
            action="prevent mode promotion to CONFIRMATION",
            blocking_issues=critical_open,
            description="Any open CRITICAL issue blocks CONFIRMATION",
        ))

        # Gate 2: require_hitl
        hitl_issues = [i for i in open_issues if i.requires_human]
        results.append(GateResult(
            gate_name="require_hitl",
            triggered=len(hitl_issues) > 0,
            action="pause loop, produce HITL checklist",
            blocking_issues=hitl_issues,
            description="Open issues requiring human decision",
        ))

        # Gate 3: auto_fix_allowed
        auto_fixable = [
            i for i in open_issues
            if i.auto_fixable and current_mode != "CONFIRMATION"
        ]
        results.append(GateResult(
            gate_name="auto_fix_allowed",
            triggered=len(auto_fixable) > 0,
            action="apply suggested_fix from PatchPolicy whitelist",
            blocking_issues=auto_fixable,
            description="Auto-fixable issues available",
        ))

        # Gate 4: block_counterfactual (SIGNIFICANT_BUT_NOT_IDENTIFIED)
        sig_not_id = [
            i for i in open_issues
            if i.rule_id == "SIGNIFICANT_BUT_NOT_IDENTIFIED"
        ]
        results.append(GateResult(
            gate_name="block_counterfactual",
            triggered=len(sig_not_id) > 0,
            action="block counterfactual propagation for affected edges",
            blocking_issues=sig_not_id,
            description="Significant-but-unidentified edges blocked from counterfactuals",
        ))

        # Gate 5: block_propagation_timing (LEADS_SIGNIFICANT_TIMING_FAIL)
        timing_fail = [
            i for i in open_issues
            if i.rule_id == "LEADS_SIGNIFICANT_TIMING_FAIL"
        ]
        results.append(GateResult(
            gate_name="block_propagation_timing",
            triggered=len(timing_fail) > 0,
            action="block shock propagation for affected edges",
            blocking_issues=timing_fail,
            description="Timing failures block shock propagation",
        ))

        # Compute aggregate state
        can_promote = not any(
            r.triggered for r in results if r.gate_name == "block_confirmation"
        )
        requires_hitl = any(
            r.triggered for r in results if r.gate_name == "require_hitl"
        )
        # Can proceed if no HITL required or no critical blocking
        can_proceed = not requires_hitl

        return GateEvaluation(
            results=results,
            can_proceed=can_proceed,
            can_promote_to_confirmation=can_promote,
            requires_hitl=requires_hitl,
            auto_fixable_count=len(auto_fixable),
        )

    def check_confirmation_ready(self, ledger: IssueLedger) -> tuple[bool, str]:
        """Check if the system is ready for CONFIRMATION mode."""
        evaluation = self.evaluate(ledger, current_mode="EXPLORATION")
        if evaluation.can_promote_to_confirmation:
            return True, "No blocking issues for CONFIRMATION mode."
        else:
            blocking = []
            for result in evaluation.results:
                if result.gate_name == "block_confirmation" and result.triggered:
                    for issue in result.blocking_issues:
                        blocking.append(f"  [{issue.severity}] {issue.rule_id}: {issue.message}")
            return False, "BLOCKED:\n" + "\n".join(blocking)
