"""
PatchPolicy: Guardrail-Compliant Auto-Fix Enforcement.

Controls which automated fixes PatchBot is allowed to apply.
Prevents p-hacking by explicitly forbidding data-driven modifications
like sample trimming, control shopping, and lag searching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PatchAction:
    """A single allowed or disallowed patch action."""

    action: str
    description: str = ""
    modes: list[str] | None = None  # Modes where allowed (None = disallowed)
    requires_logging: bool = False
    reason: str = ""  # Why disallowed


class PatchPolicy:
    """
    Enforces patch policy: what auto-fixes are allowed.

    Usage:
        policy = PatchPolicy.load()
        if policy.is_allowed("add_edge_units", mode="EXPLORATION"):
            # Apply the fix
        else:
            # Log rejection
    """

    def __init__(
        self,
        allowed: list[PatchAction] | None = None,
        disallowed: list[PatchAction] | None = None,
    ):
        self.allowed = {a.action: a for a in (allowed or [])}
        self.disallowed = {a.action: a for a in (disallowed or [])}

    @classmethod
    def load(cls, config_path: Path | None = None) -> PatchPolicy:
        """Load patch policy from YAML config."""
        if config_path is None:
            config_path = Path("config/agentic/patch_policy.yaml")

        if not config_path.exists():
            logger.warning(f"Patch policy not found: {config_path}")
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        allowed = []
        for item in data.get("allowed_auto_fixes", []):
            allowed.append(PatchAction(
                action=item["action"],
                description=item.get("description", ""),
                modes=item.get("modes"),
                requires_logging=item.get("requires_logging", False),
            ))

        disallowed = []
        for item in data.get("disallowed_auto_fixes", []):
            disallowed.append(PatchAction(
                action=item["action"],
                reason=item.get("reason", ""),
            ))

        return cls(allowed=allowed, disallowed=disallowed)

    def is_allowed(self, action: str, mode: str = "EXPLORATION") -> bool:
        """Check if a patch action is allowed in the given mode."""
        # Explicitly disallowed always wins
        if action in self.disallowed:
            return False

        # Check allowed list
        if action in self.allowed:
            patch = self.allowed[action]
            if patch.modes is None:
                return True
            return mode in patch.modes

        # Unknown actions are disallowed by default
        return False

    def get_rejection_reason(self, action: str, mode: str = "EXPLORATION") -> str:
        """Get the reason why an action is rejected."""
        if action in self.disallowed:
            return self.disallowed[action].reason

        if action in self.allowed:
            patch = self.allowed[action]
            if patch.modes and mode not in patch.modes:
                return f"Action '{action}' not allowed in {mode} mode (allowed: {patch.modes})"

        return f"Unknown action '{action}' is disallowed by default"

    def get_allowed_actions(self, mode: str = "EXPLORATION") -> list[str]:
        """Get list of all allowed actions for a mode."""
        actions = []
        for action_name, patch in self.allowed.items():
            if action_name not in self.disallowed:
                if patch.modes is None or mode in patch.modes:
                    actions.append(action_name)
        return actions

    def requires_logging(self, action: str) -> bool:
        """Check if an action requires explicit audit logging."""
        if action in self.allowed:
            return self.allowed[action].requires_logging
        return True  # Unknown actions always require logging

    # ------------------------------------------------------------------
    # LLM repair policy levels
    # ------------------------------------------------------------------

    # Safe LLM repair actions that are allowed without HITL confirmation.
    _LLM_METADATA_REPAIRS: set[str] = {
        "fix_edge_id_syntax",
        "fix_missing_source_spec",
    }

    # Structural LLM repair actions that require HITL unless within safe boundaries.
    _LLM_DAG_REPAIRS: set[str] = {
        "fix_dag_identity_deps",
        "fix_dag_missing_reaction",
    }

    def is_llm_repair_allowed(
        self,
        action: str,
        mode: str = "EXPLORATION",
        hitl_approved: bool = False,
    ) -> bool:
        """Check whether an LLM-assisted repair action is allowed.

        LLM_METADATA_REPAIR actions are always allowed (audit-logged).
        LLM_DAG_REPAIR actions require HITL confirmation OR auto-approve
        when in EXPLORATION mode and the change is within safe boundaries.

        Args:
            action: The repair action name.
            mode: Current pipeline mode.
            hitl_approved: Whether HITL has pre-approved this repair.

        Returns:
            True if the repair may proceed.
        """
        if action in self._LLM_METADATA_REPAIRS:
            return True
        if action in self._LLM_DAG_REPAIRS:
            if hitl_approved:
                return True
            # Auto-approve safe structural repairs in EXPLORATION mode
            if mode == "EXPLORATION":
                return True
            return False
        # Explicitly disallowed actions still take precedence
        if action in self.disallowed:
            return False
        return self.is_allowed(action, mode)

    def get_llm_repair_level(self, action: str) -> str:
        """Return the LLM repair policy level for an action.

        Returns:
            'LLM_METADATA_REPAIR', 'LLM_DAG_REPAIR', or 'STANDARD'.
        """
        if action in self._LLM_METADATA_REPAIRS:
            return "LLM_METADATA_REPAIR"
        if action in self._LLM_DAG_REPAIRS:
            return "LLM_DAG_REPAIR"
        return "STANDARD"
