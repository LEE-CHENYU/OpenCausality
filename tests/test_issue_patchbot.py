"""Tests for PatchBot + PatchPolicy + IssueLedger interactions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from shared.agentic.agents.patch_bot import PatchBot, PatchResult
from shared.agentic.governance.patch_policy import PatchPolicy, PatchAction
from shared.agentic.issues.issue_ledger import Issue, IssueLedger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_issue(
    rule_id: str = "UNIT_MISSING",
    edge_id: str = "a_to_b",
    auto_fixable: bool = True,
    action: str = "add_edge_units",
    severity: str = "MEDIUM",
) -> Issue:
    return Issue(
        run_id="test",
        timestamp="2025-01-01T00:00:00Z",
        severity=severity,
        rule_id=rule_id,
        scope="edge",
        message=f"Test issue: {rule_id}",
        edge_id=edge_id,
        auto_fixable=auto_fixable,
        suggested_fix={"action": action, "units": {"treatment": "pp", "outcome": "bps"}},
    )


def _make_policy(allowed_actions: list[str], disallowed_actions: list[str] | None = None):
    allowed = [PatchAction(action=a, modes=["EXPLORATION", "CONFIRMATION"]) for a in allowed_actions]
    disallowed = [PatchAction(action=a, reason="forbidden") for a in (disallowed_actions or [])]
    return PatchPolicy(allowed=allowed, disallowed=disallowed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPatchBotDisabledInConfirmation:
    def test_returns_empty_in_confirmation(self):
        policy = _make_policy(["add_edge_units"])
        bot = PatchBot(policy=policy)
        ledger = IssueLedger(run_id="test")
        ledger.add(_make_issue())

        results = bot.apply_fixes(ledger, mode="CONFIRMATION")
        assert results == []


class TestPatchBotAppliesAllowedFix:
    def test_applies_and_closes(self):
        policy = _make_policy(["add_edge_units"])
        bot = PatchBot(policy=policy, artifact_store=MagicMock())
        ledger = IssueLedger(run_id="test")
        issue = _make_issue()
        ledger.add(issue)

        results = bot.apply_fixes(ledger, mode="EXPLORATION")

        assert len(results) == 1
        assert results[0].applied is True
        assert results[0].action == "add_edge_units"
        # Issue should be closed
        assert issue.status == "CLOSED"


class TestPatchBotRejectsDisallowedFix:
    def test_rejects_with_reason(self):
        policy = _make_policy([], disallowed_actions=["add_edge_units"])
        bot = PatchBot(policy=policy)
        ledger = IssueLedger(run_id="test")
        ledger.add(_make_issue())

        results = bot.apply_fixes(ledger, mode="EXPLORATION")

        assert len(results) == 1
        assert results[0].applied is False
        assert "Rejected" in results[0].message


class TestLLMRepairPolicyLevels:
    def test_metadata_repair_allowed(self):
        policy = PatchPolicy()
        assert policy.is_llm_repair_allowed("fix_edge_id_syntax", mode="EXPLORATION") is True
        assert policy.is_llm_repair_allowed("fix_missing_source_spec", mode="CONFIRMATION") is True

    def test_structural_repair_exploration_allowed(self):
        policy = PatchPolicy()
        assert policy.is_llm_repair_allowed("fix_dag_identity_deps", mode="EXPLORATION") is True
        assert policy.is_llm_repair_allowed("fix_dag_missing_reaction", mode="EXPLORATION") is True

    def test_structural_repair_confirmation_blocked(self):
        policy = PatchPolicy()
        assert policy.is_llm_repair_allowed("fix_dag_identity_deps", mode="CONFIRMATION") is False
        assert policy.is_llm_repair_allowed("fix_dag_missing_reaction", mode="CONFIRMATION") is False

    def test_structural_repair_confirmation_hitl_approved(self):
        policy = PatchPolicy()
        assert policy.is_llm_repair_allowed(
            "fix_dag_identity_deps", mode="CONFIRMATION", hitl_approved=True,
        ) is True

    def test_get_llm_repair_level(self):
        policy = PatchPolicy()
        assert policy.get_llm_repair_level("fix_edge_id_syntax") == "LLM_METADATA_REPAIR"
        assert policy.get_llm_repair_level("fix_dag_identity_deps") == "LLM_DAG_REPAIR"
        assert policy.get_llm_repair_level("add_edge_units") == "STANDARD"


class TestPatchBotEdgeIdSyntaxFix:
    def test_normalizes_arrow_syntax(self):
        policy = _make_policy(["fix_edge_id_syntax"])
        bot = PatchBot(policy=policy, audit_log=MagicMock())
        issue = _make_issue(
            rule_id="invalid_edge_id",
            edge_id="oil->cpi",
            action="fix_edge_id_syntax",
        )
        issue.suggested_fix = {"action": "fix_edge_id_syntax", "old_id": "oil->cpi"}
        ledger = IssueLedger(run_id="test")
        ledger.add(issue)

        results = bot.apply_fixes(ledger, mode="EXPLORATION")

        assert len(results) == 1
        assert results[0].applied is True
        assert results[0].changes["new_id"] == "oil_to_cpi"
