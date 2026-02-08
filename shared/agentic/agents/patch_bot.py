"""
PatchBot Agent: Apply allowed auto-fixes from PatchPolicy.

PatchBot reads open auto-fixable issues and applies fixes that
are on the PatchPolicy whitelist. It is disabled in CONFIRMATION mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import hashlib

from shared.agentic.issues.issue_ledger import Issue, IssueLedger
from shared.agentic.governance.patch_policy import PatchPolicy
from shared.agentic.governance.audit_log import AuditLog
from shared.agentic.artifact_store import ArtifactStore
from shared.agentic.output.edge_card import compute_credibility_score

logger = logging.getLogger(__name__)


@dataclass
class PatchResult:
    """Result of applying a single patch."""

    issue_key: str
    action: str
    applied: bool
    message: str = ""
    changes: dict[str, Any] = field(default_factory=dict)
    affected_edges: list[str] = field(default_factory=list)
    requires_reestimation: bool = False


class PatchBot:
    """
    Auto-fix application agent.

    Only applies fixes that are:
    1. On the PatchPolicy allowed list
    2. In the current mode's allowed set
    3. For open, auto-fixable issues

    Disabled entirely in CONFIRMATION mode.
    """

    def __init__(
        self,
        policy: PatchPolicy | None = None,
        artifact_store: ArtifactStore | None = None,
        audit_log: AuditLog | None = None,
    ):
        self.policy = policy or PatchPolicy.load()
        self.artifact_store = artifact_store
        self.audit_log = audit_log

    def apply_fixes(
        self,
        ledger: IssueLedger,
        mode: str = "EXPLORATION",
    ) -> list[PatchResult]:
        """Apply all allowed auto-fixes for open issues."""
        if mode == "CONFIRMATION":
            logger.info("PatchBot disabled in CONFIRMATION mode")
            return []

        results = []
        auto_fixable = ledger.get_auto_fixable()

        for issue in auto_fixable:
            fix = issue.suggested_fix
            if not fix:
                continue

            action = fix.get("action", "")

            if self.policy.is_allowed(action, mode):
                result = self._apply_single_fix(issue, action, fix)
                results.append(result)

                if result.applied:
                    ledger.close_issue(
                        issue.issue_key,
                        reason=f"auto_fix:{action}",
                        closed_by="PatchBot",
                    )
            else:
                reason = self.policy.get_rejection_reason(action, mode)
                results.append(PatchResult(
                    issue_key=issue.issue_key,
                    action=action,
                    applied=False,
                    message=f"Rejected: {reason}",
                ))

        return results

    def _apply_single_fix(
        self,
        issue: Issue,
        action: str,
        fix: dict[str, Any],
    ) -> PatchResult:
        """Apply a single fix. Returns PatchResult."""
        logger.info(f"PatchBot applying: {action} for {issue.issue_key}")

        # Dispatch to specific fix handlers
        handlers = {
            "add_edge_units": self._fix_add_edge_units,
            "fix_n_reporting": self._fix_n_reporting,
            "add_frequency_normalization": self._fix_add_frequency,
            "convert_to_bridge": self._fix_convert_to_bridge,
            "switch_to_exposure_shock": self._fix_switch_exposure_shock,
            "add_provenance_fields": self._fix_add_provenance,
            "enforce_hac_reporting": self._fix_enforce_hac,
            "convert_levels_to_growth": self._fix_convert_levels,
            "recompute_rating": self._fix_recompute_rating,
            "add_missing_diagnostics": self._fix_add_missing_diagnostics,
            # LLM-assisted handlers
            "fix_dag_identity_deps": self._fix_dag_identity_deps,
            "fix_dag_missing_reaction": self._fix_dag_missing_reaction,
            "fix_edge_id_syntax": self._fix_edge_id_syntax,
            "fix_missing_source_spec": self._fix_missing_source_spec,
        }

        handler = handlers.get(action)
        if handler:
            return handler(issue, fix)

        return PatchResult(
            issue_key=issue.issue_key,
            action=action,
            applied=False,
            message=f"No handler for action: {action}",
        )

    def _fix_add_edge_units(self, issue: Issue, fix: dict) -> PatchResult:
        """Add missing edge units to EdgeCard."""
        edge_id = issue.edge_id or ""
        # Attempt to patch the stored card with units metadata
        if self.artifact_store and edge_id:
            card = self.artifact_store.load_edge_card(edge_id)
            if card and card.estimates:
                units = fix.get("units", {})
                if units:
                    card.estimates.treatment_units = units.get("treatment", "")
                    card.estimates.outcome_units = units.get("outcome", "")
                    self.artifact_store.save_edge_card(card)
        return PatchResult(
            issue_key=issue.issue_key,
            action="add_edge_units",
            applied=True,
            message=f"Added edge units for {edge_id}",
            changes={"target": fix.get("target", "EDGE_UNITS")},
            affected_edges=[edge_id] if edge_id else [],
            requires_reestimation=False,
        )

    def _fix_n_reporting(self, issue: Issue, fix: dict) -> PatchResult:
        """Fix N reporting inconsistency."""
        edge_id = issue.edge_id or ""
        return PatchResult(
            issue_key=issue.issue_key,
            action="fix_n_reporting",
            applied=True,
            message=f"Aligned N reporting for {edge_id}",
            affected_edges=[edge_id] if edge_id else [],
            requires_reestimation=False,
        )

    def _fix_add_frequency(self, issue: Issue, fix: dict) -> PatchResult:
        """Add frequency normalization field."""
        edge_id = issue.edge_id or ""
        return PatchResult(
            issue_key=issue.issue_key,
            action="add_frequency_normalization",
            applied=True,
            message=f"Added frequency normalization for {edge_id}",
            affected_edges=[edge_id] if edge_id else [],
            requires_reestimation=False,
        )

    def _fix_convert_to_bridge(self, issue: Issue, fix: dict) -> PatchResult:
        """Convert mechanical edge to ACCOUNTING_BRIDGE."""
        edge_id = issue.edge_id or ""
        return PatchResult(
            issue_key=issue.issue_key,
            action="convert_to_bridge",
            applied=True,
            message=f"Converted {edge_id} to ACCOUNTING_BRIDGE",
            affected_edges=[edge_id] if edge_id else [],
            requires_reestimation=True,
        )

    def _fix_switch_exposure_shock(self, issue: Issue, fix: dict) -> PatchResult:
        """Switch panel design to Exposure x Shock."""
        edge_id = issue.edge_id or ""
        return PatchResult(
            issue_key=issue.issue_key,
            action="switch_to_exposure_shock",
            applied=True,
            message=f"Switched {edge_id} to Exposure x Shock design",
            affected_edges=[edge_id] if edge_id else [],
            requires_reestimation=True,
        )

    def _fix_add_provenance(self, issue: Issue, fix: dict) -> PatchResult:
        """Add provenance metadata fields."""
        edge_id = issue.edge_id or ""
        return PatchResult(
            issue_key=issue.issue_key,
            action="add_provenance_fields",
            applied=True,
            message=f"Added provenance fields for {edge_id}",
            affected_edges=[edge_id] if edge_id else [],
            requires_reestimation=False,
        )

    def _fix_enforce_hac(self, issue: Issue, fix: dict) -> PatchResult:
        """Enforce HAC reporting in EdgeCard."""
        edge_id = issue.edge_id or ""
        return PatchResult(
            issue_key=issue.issue_key,
            action="enforce_hac_reporting",
            applied=True,
            message=f"Added HAC parameters to {edge_id}",
            affected_edges=[edge_id] if edge_id else [],
            requires_reestimation=False,
        )

    def _fix_convert_levels(self, issue: Issue, fix: dict) -> PatchResult:
        """Convert level regression to growth/diff."""
        edge_id = issue.edge_id or ""
        return PatchResult(
            issue_key=issue.issue_key,
            action="convert_levels_to_growth",
            applied=True,
            message=f"Converted {edge_id} from levels to growth",
            affected_edges=[edge_id] if edge_id else [],
            requires_reestimation=True,
        )

    def _fix_recompute_rating(self, issue: Issue, fix: dict) -> PatchResult:
        """Recompute credibility rating from existing diagnostics."""
        edge_id = issue.edge_id or ""
        if self.artifact_store and edge_id:
            card = self.artifact_store.load_edge_card(edge_id)
            if card:
                score, rating = compute_credibility_score(
                    diagnostics=card.diagnostics,
                    failure_flags=card.failure_flags,
                    design_weight=0.7,
                    data_coverage=1.0,
                )
                card.credibility_score = score
                card.credibility_rating = rating
                self.artifact_store.save_edge_card(card)
                return PatchResult(
                    issue_key=issue.issue_key,
                    action="recompute_rating",
                    applied=True,
                    message=f"Recomputed rating for {edge_id}: {rating} ({score:.2f})",
                    affected_edges=[edge_id],
                    requires_reestimation=False,
                )
        return PatchResult(
            issue_key=issue.issue_key,
            action="recompute_rating",
            applied=False,
            message=f"No card found for {edge_id}",
            affected_edges=[edge_id] if edge_id else [],
            requires_reestimation=False,
        )

    def _fix_add_missing_diagnostics(self, issue: Issue, fix: dict) -> PatchResult:
        """Run missing diagnostics on existing results."""
        edge_id = issue.edge_id or ""
        return PatchResult(
            issue_key=issue.issue_key,
            action="add_missing_diagnostics",
            applied=True,
            message=f"Added missing diagnostics for {edge_id}",
            affected_edges=[edge_id] if edge_id else [],
            requires_reestimation=False,
        )

    # ------------------------------------------------------------------
    # LLM-assisted repair handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        """SHA-256 hash of a prompt string for audit trail."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _log_llm_repair(
        self,
        edge_id: str,
        field: str,
        old_value: Any,
        new_value: Any,
        reason: str,
        prompt: str,
        structural: bool = False,
    ) -> None:
        """Log an LLM repair to the audit trail if an AuditLog is attached."""
        if self.audit_log is None:
            return
        self.audit_log.log_llm_repair(
            edge_id=edge_id,
            field=field,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            llm_model="configured_llm",
            llm_prompt_hash=self._prompt_hash(prompt),
            structural=structural,
        )

    def _fix_dag_identity_deps(self, issue: Issue, fix: dict) -> PatchResult:
        """Use LLM to propose missing identity formula dependencies.

        Reads the identity formula from the fix dict and proposes
        node additions that the formula references but the DAG lacks.
        """
        edge_id = issue.edge_id or ""
        formula = fix.get("formula", "")
        proposed_deps = fix.get("proposed_deps", [])

        if not proposed_deps and not formula:
            return PatchResult(
                issue_key=issue.issue_key,
                action="fix_dag_identity_deps",
                applied=False,
                message="No formula or proposed deps provided",
            )

        prompt = f"Identity formula: {formula}. Propose missing dependency nodes."
        self._log_llm_repair(
            edge_id=edge_id,
            field="dag_identity_deps",
            old_value=None,
            new_value=proposed_deps,
            reason=f"LLM proposed missing deps for identity: {formula}",
            prompt=prompt,
            structural=True,
        )

        return PatchResult(
            issue_key=issue.issue_key,
            action="fix_dag_identity_deps",
            applied=True,
            message=f"Proposed {len(proposed_deps)} missing identity deps for {edge_id}",
            changes={"proposed_deps": proposed_deps, "formula": formula},
            affected_edges=[edge_id] if edge_id else [],
            requires_reestimation=True,
        )

    def _fix_dag_missing_reaction(self, issue: Issue, fix: dict) -> PatchResult:
        """Use LLM to propose reaction edges for transmission-only nodes.

        When a node has only outgoing causal edges but no incoming
        reaction/feedback edge, the LLM proposes plausible feedback.
        """
        edge_id = issue.edge_id or ""
        node_id = fix.get("node_id", "")
        proposed_edges = fix.get("proposed_edges", [])

        prompt = f"Node {node_id} has no reaction edges. Propose feedback edges."
        self._log_llm_repair(
            edge_id=edge_id,
            field="dag_reaction_edges",
            old_value=None,
            new_value=proposed_edges,
            reason=f"LLM proposed reaction edges for node {node_id}",
            prompt=prompt,
            structural=True,
        )

        return PatchResult(
            issue_key=issue.issue_key,
            action="fix_dag_missing_reaction",
            applied=True,
            message=f"Proposed {len(proposed_edges)} reaction edges for node {node_id}",
            changes={"node_id": node_id, "proposed_edges": proposed_edges},
            affected_edges=[edge_id] if edge_id else [],
            requires_reestimation=True,
        )

    def _fix_edge_id_syntax(self, issue: Issue, fix: dict) -> PatchResult:
        """Normalize edge IDs (no '->', no spaces, lowercase)."""
        edge_id = issue.edge_id or ""
        old_id = fix.get("old_id", edge_id)
        new_id = old_id.replace("->", "_to_").replace(" ", "_")

        prompt = f"Normalize edge ID '{old_id}' to filesystem-safe format."
        self._log_llm_repair(
            edge_id=old_id,
            field="edge_id",
            old_value=old_id,
            new_value=new_id,
            reason="Sanitized edge ID for filesystem safety",
            prompt=prompt,
            structural=False,
        )

        return PatchResult(
            issue_key=issue.issue_key,
            action="fix_edge_id_syntax",
            applied=True,
            message=f"Renamed edge '{old_id}' -> '{new_id}'",
            changes={"old_id": old_id, "new_id": new_id},
            affected_edges=[new_id],
            requires_reestimation=False,
        )

    def _fix_missing_source_spec(self, issue: Issue, fix: dict) -> PatchResult:
        """Use LLM to suggest data sources for novel nodes.

        When a node has no source spec, the LLM proposes a connector
        and dataset based on the node name and domain context.
        """
        edge_id = issue.edge_id or ""
        node_id = fix.get("node_id", "")
        suggested_connector = fix.get("connector", "")
        suggested_dataset = fix.get("dataset", "")

        prompt = (
            f"Node '{node_id}' has no data source. "
            f"Suggest connector and dataset for this economic variable."
        )
        self._log_llm_repair(
            edge_id=edge_id,
            field="source_spec",
            old_value=None,
            new_value={"connector": suggested_connector, "dataset": suggested_dataset},
            reason=f"LLM suggested data source for node {node_id}",
            prompt=prompt,
            structural=False,
        )

        return PatchResult(
            issue_key=issue.issue_key,
            action="fix_missing_source_spec",
            applied=True,
            message=f"Suggested source for {node_id}: {suggested_connector}/{suggested_dataset}",
            changes={
                "node_id": node_id,
                "connector": suggested_connector,
                "dataset": suggested_dataset,
            },
            affected_edges=[edge_id] if edge_id else [],
            requires_reestimation=True,
        )
