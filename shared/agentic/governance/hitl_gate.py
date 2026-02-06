"""
HITL Gate: Human-in-the-Loop Trigger Detection and Checklist Generation.

Detects conditions requiring human decisions and generates structured
checklists for human review. The agent loop pauses when HITL triggers
are active.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from shared.agentic.issues.issue_ledger import Issue, IssueLedger

logger = logging.getLogger(__name__)


@dataclass
class HITLTrigger:
    """A single HITL trigger definition."""

    trigger_id: str
    condition: str
    checklist_item: str
    priority: str = "MEDIUM"


@dataclass
class HITLChecklistItem:
    """A single item in the HITL checklist."""

    trigger_id: str
    description: str
    context: dict[str, Any] = field(default_factory=dict)
    edge_id: str | None = None
    suggested_resolution: str | None = None
    resolved: bool = False
    resolution: str | None = None


@dataclass
class HITLChecklist:
    """Complete HITL checklist for a run."""

    run_id: str
    items: list[HITLChecklistItem] = field(default_factory=list)

    @property
    def pending_count(self) -> int:
        return sum(1 for item in self.items if not item.resolved)

    @property
    def is_complete(self) -> bool:
        return all(item.resolved for item in self.items)

    def to_markdown(self) -> str:
        """Generate markdown checklist for human consumption."""
        lines = [
            "# Human-in-the-Loop Checklist",
            f"Run ID: {self.run_id}",
            "",
            "## Pending Decisions",
            "",
        ]

        section_num = 0
        for item in self.items:
            if item.resolved:
                continue
            section_num += 1
            lines.append(f"### {section_num}. {item.description}")
            if item.edge_id:
                lines.append(f"- Edge: `{item.edge_id}`")
            if item.suggested_resolution:
                lines.append(f"- Suggested: {item.suggested_resolution}")
            if item.context:
                for k, v in item.context.items():
                    lines.append(f"- {k}: {v}")
            lines.append(f"- [ ] Decision: _________________")
            lines.append("")

        if section_num == 0:
            lines.append("*No pending decisions.*")

        return "\n".join(lines)


class HITLGate:
    """
    Detects HITL triggers and generates checklists.

    Usage:
        gate = HITLGate()
        checklist = gate.evaluate(
            dag_config=dag,
            edge_cards=cards,
            ledger=ledger,
            run_id="abc123",
        )
        if checklist.pending_count > 0:
            print(checklist.to_markdown())
    """

    def __init__(self, config_path: Path | None = None):
        self.triggers = self._load_triggers(
            config_path or Path("config/agentic/hitl_triggers.yaml")
        )

    def _load_triggers(self, config_path: Path) -> list[HITLTrigger]:
        """Load HITL triggers from YAML config."""
        if not config_path.exists():
            return self._default_triggers()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        triggers = []
        for t in data.get("triggers", []):
            triggers.append(HITLTrigger(
                trigger_id=t["trigger_id"],
                condition=t["condition"],
                checklist_item=t["checklist_item"],
                priority=t.get("priority", "MEDIUM"),
            ))
        return triggers

    def _default_triggers(self) -> list[HITLTrigger]:
        """Default triggers if config file not found."""
        return [
            HITLTrigger("edge_type_unspecified", "", "Classify edge type", "HIGH"),
            HITLTrigger("expected_sign_undefined", "", "Define expected sign", "MEDIUM"),
        ]

    def evaluate(
        self,
        dag_config: dict[str, Any] | None = None,
        edge_cards: dict[str, Any] | None = None,
        ledger: IssueLedger | None = None,
        ts_guard_results: dict[str, Any] | None = None,
        run_id: str = "",
    ) -> HITLChecklist:
        """Evaluate all HITL triggers and build checklist."""
        checklist = HITLChecklist(run_id=run_id)
        edges = {}
        if dag_config:
            edges = {e["id"]: e for e in dag_config.get("edges", [])}

        # Check edge type classification
        for edge_id, edge in edges.items():
            edge_type = edge.get("edge_type")
            if not edge_type or edge_type not in ("causal", "reaction_function", "bridge", "identity"):
                checklist.items.append(HITLChecklistItem(
                    trigger_id="edge_type_unspecified",
                    description="Edge Type Classification",
                    edge_id=edge_id,
                    context={"current_type": edge_type or "none"},
                    suggested_resolution="Classify as causal, reaction_function, bridge, or identity",
                ))

        # Check expected sign
        for edge_id, edge in edges.items():
            acceptance = edge.get("acceptance_criteria", {})
            plausibility = acceptance.get("plausibility", {})
            if not plausibility.get("expected_sign"):
                checklist.items.append(HITLChecklistItem(
                    trigger_id="expected_sign_undefined",
                    description="Expected Sign Definition",
                    edge_id=edge_id,
                    suggested_resolution="Define expected sign based on economic theory",
                ))

        # Check regime instability from TSGuard
        if ts_guard_results:
            for edge_id, ts_result in ts_guard_results.items():
                if hasattr(ts_result, "diagnostics_results"):
                    if ts_result.diagnostics_results.get("regime_stability") == "fail":
                        checklist.items.append(HITLChecklistItem(
                            trigger_id="regime_instability_detected",
                            description="Regime Instability Decision",
                            edge_id=edge_id,
                            suggested_resolution="Split estimand by regime or restrict counterfactual scope",
                        ))

        # Check issues requiring human from ledger
        if ledger:
            for issue in ledger.get_issues_requiring_human():
                checklist.items.append(HITLChecklistItem(
                    trigger_id=f"issue:{issue.rule_id}",
                    description=issue.message,
                    edge_id=issue.edge_id,
                    context=issue.evidence,
                ))

        return checklist
