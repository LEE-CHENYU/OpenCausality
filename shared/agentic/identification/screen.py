"""
Identifiability Screen: First-Class Module for Causal Claim Assessment.

Determines what causal claims can be made for each edge, given the
design, data, and diagnostics. This is the core module that enforces
the principle: "Significance is never a promotion criterion; stability
and identification are."

Execution points:
1. Pre-design (ModelSmith input): Can this edge ever be identified?
2. Post-design (before estimation): Does chosen design achieve identification?
3. Post-estimation (Judge rating): Given diagnostics, what's the final claim?
4. Report generation: Global caveat dashboard
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Claim level hierarchy (ordered from strongest to weakest)
CLAIM_LEVELS = [
    "IDENTIFIED_CAUSAL",   # Causal identification achieved
    "REDUCED_FORM",        # Statistical relationship, not fully identified
    "DESCRIPTIVE",         # Descriptive association only
    "BLOCKED_ID",          # Cannot make any causal claim
]


@dataclass
class IdentifiabilityResult:
    """Result of an identifiability screen for a single edge."""

    claim_level: Literal["IDENTIFIED_CAUSAL", "REDUCED_FORM", "DESCRIPTIVE", "BLOCKED_ID"]

    risks: dict[str, Literal["low", "medium", "high"]] = field(default_factory=dict)
    # Keys: unmeasured_confounding, simultaneity, weak_variation,
    #        measurement_error, selection

    required_structure_missing: list[str] = field(default_factory=list)
    # e.g., ["needs exposure heterogeneity", "needs instrument"]

    allowed_actions: list[str] = field(default_factory=list)
    # What can be done with this edge given its claim level

    untestable_assumptions: list[str] = field(default_factory=list)
    # Assumptions that cannot be tested with available data

    counterfactual_allowed: bool = False
    counterfactual_reason_blocked: str | None = None

    # Split counterfactual fields (mode-aware)
    shock_scenario_allowed: bool = False
    policy_intervention_allowed: bool = False
    reason_shock_blocked: str | None = None
    reason_policy_blocked: str | None = None

    # Diagnostics that were used to determine claim level
    testable_threats_passed: list[str] = field(default_factory=list)
    testable_threats_failed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_level": self.claim_level,
            "risks": self.risks,
            "required_structure_missing": self.required_structure_missing,
            "allowed_actions": self.allowed_actions,
            "untestable_assumptions": self.untestable_assumptions,
            "counterfactual_allowed": self.counterfactual_allowed,
            "counterfactual_reason_blocked": self.counterfactual_reason_blocked,
            "shock_scenario_allowed": self.shock_scenario_allowed,
            "policy_intervention_allowed": self.policy_intervention_allowed,
            "reason_shock_blocked": self.reason_shock_blocked,
            "reason_policy_blocked": self.reason_policy_blocked,
            "testable_threats_passed": self.testable_threats_passed,
            "testable_threats_failed": self.testable_threats_failed,
        }

    @property
    def is_causal(self) -> bool:
        return self.claim_level == "IDENTIFIED_CAUSAL"

    @property
    def is_blocked(self) -> bool:
        return self.claim_level == "BLOCKED_ID"

    def cap_to(self, max_level: str) -> None:
        """Cap claim level to at most the given level."""
        current_idx = CLAIM_LEVELS.index(self.claim_level)
        max_idx = CLAIM_LEVELS.index(max_level)
        if current_idx < max_idx:
            self.claim_level = max_level
            if max_level in ("DESCRIPTIVE", "BLOCKED_ID"):
                self.counterfactual_allowed = False
                self.counterfactual_reason_blocked = (
                    f"Claim level capped to {max_level}"
                )
                self.shock_scenario_allowed = False
                self.policy_intervention_allowed = False
                self.reason_shock_blocked = f"Claim level capped to {max_level}"
                self.reason_policy_blocked = f"Claim level capped to {max_level}"


class IdentifiabilityScreen:
    """
    Screen edges for identifiability at multiple execution points.

    The screen is conservative: it defaults to REDUCED_FORM unless
    strong design features establish identification.
    """

    # Design -> base claim level mapping
    DESIGN_CLAIM_MAP: dict[str, str] = {
        "IV": "IDENTIFIED_CAUSAL",
        "DID": "IDENTIFIED_CAUSAL",
        "EVENT_STUDY": "IDENTIFIED_CAUSAL",
        "RDD": "IDENTIFIED_CAUSAL",
        "PANEL_LP_EXPOSURE_FE": "IDENTIFIED_CAUSAL",
        "TS_LOCAL_PROJECTION": "REDUCED_FORM",
        "LOCAL_PROJECTIONS": "REDUCED_FORM",
        "LOCAL_PROJECTIONS_ANNUAL": "REDUCED_FORM",
        "VAR": "REDUCED_FORM",
        "OLS": "DESCRIPTIVE",
        "IDENTITY": "IDENTIFIED_CAUSAL",  # Mechanical identity
        "ACCOUNTING_BRIDGE": "IDENTIFIED_CAUSAL",  # Deterministic
        "IMMUTABLE_EVIDENCE": "IDENTIFIED_CAUSAL",  # Validated
    }

    def screen_pre_design(
        self,
        edge_id: str,
        edge_type: str,
        has_instrument: bool = False,
        has_panel: bool = False,
        has_exposure_variation: bool = False,
    ) -> IdentifiabilityResult:
        """
        Screen 1: Pre-design. Can this edge ever be identified?

        Called before ModelSmith selects a design.
        """
        risks = {
            "unmeasured_confounding": "medium",
            "simultaneity": "low",
            "weak_variation": "low",
            "measurement_error": "low",
            "selection": "low",
        }

        if edge_type == "reaction_function":
            return IdentifiabilityResult(
                claim_level="BLOCKED_ID",
                risks={"simultaneity": "high", **{k: v for k, v in risks.items() if k != "simultaneity"}},
                required_structure_missing=["needs exogenous variation in policy"],
                counterfactual_allowed=False,
                counterfactual_reason_blocked="Reaction function edge; endogenous policy response",
                untestable_assumptions=["policy surprise identification"],
            )

        if edge_type == "identity":
            return IdentifiabilityResult(
                claim_level="IDENTIFIED_CAUSAL",
                risks={k: "low" for k in risks},
                counterfactual_allowed=True,
                allowed_actions=["shock_propagation", "policy_counterfactual"],
            )

        if edge_type == "bridge":
            return IdentifiabilityResult(
                claim_level="IDENTIFIED_CAUSAL",
                risks={k: "low" for k in risks},
                counterfactual_allowed=True,
                allowed_actions=["shock_propagation"],
                untestable_assumptions=["accounting regime unchanged"],
            )

        # Default for causal edges
        required_missing = []
        if not has_instrument and not has_panel:
            risks["unmeasured_confounding"] = "high"
            required_missing.append("needs instrument or panel structure")

        if not has_exposure_variation and has_panel:
            risks["weak_variation"] = "high"
            required_missing.append("needs exposure heterogeneity across units")

        claim = "REDUCED_FORM"
        if has_instrument:
            claim = "IDENTIFIED_CAUSAL"
        elif has_panel and has_exposure_variation:
            claim = "IDENTIFIED_CAUSAL"

        return IdentifiabilityResult(
            claim_level=claim,
            risks=risks,
            required_structure_missing=required_missing,
            counterfactual_allowed=(claim == "IDENTIFIED_CAUSAL"),
            counterfactual_reason_blocked=(
                "Reduced-form only; confounding not ruled out"
                if claim != "IDENTIFIED_CAUSAL" else None
            ),
            allowed_actions=(
                ["shock_propagation", "policy_counterfactual"]
                if claim == "IDENTIFIED_CAUSAL"
                else ["shock_propagation_with_caveat"]
            ),
        )

    def screen_post_design(
        self,
        edge_id: str,
        design: str,
        diagnostics_available: list[str] | None = None,
    ) -> IdentifiabilityResult:
        """
        Screen 2: Post-design, before estimation.

        Checks whether the chosen design achieves identification.
        """
        base_claim = self.DESIGN_CLAIM_MAP.get(design, "DESCRIPTIVE")

        risks = {
            "unmeasured_confounding": "low" if base_claim == "IDENTIFIED_CAUSAL" else "medium",
            "simultaneity": "low",
            "weak_variation": "low",
            "measurement_error": "low",
            "selection": "low",
        }

        untestable = []
        if design == "IV":
            untestable.append("Exclusion restriction")
        elif design == "PANEL_LP_EXPOSURE_FE":
            untestable.append("Exposure predetermined and exogenous to outcome trends")
        elif design in ("TS_LOCAL_PROJECTION", "LOCAL_PROJECTIONS"):
            untestable.append("No omitted common cause of treatment and outcome")
            risks["unmeasured_confounding"] = "medium"

        return IdentifiabilityResult(
            claim_level=base_claim,
            risks=risks,
            untestable_assumptions=untestable,
            counterfactual_allowed=(base_claim == "IDENTIFIED_CAUSAL"),
            counterfactual_reason_blocked=(
                f"Design {design} provides {base_claim} level only"
                if base_claim != "IDENTIFIED_CAUSAL" else None
            ),
        )

    def screen_post_estimation(
        self,
        edge_id: str,
        design: str,
        diagnostics: dict[str, Any],
        ts_guard_result: Any | None = None,
        query_mode: str | None = None,
    ) -> IdentifiabilityResult:
        """
        Screen 3: Post-estimation. Given diagnostics, what's the final claim?

        This is where diagnostic failures can downgrade the claim level.
        """
        result = self.screen_post_design(edge_id, design)

        passed = []
        failed = []

        for diag_name, diag in diagnostics.items():
            diag_passed = getattr(diag, "passed", True) if hasattr(diag, "passed") else True
            if diag_passed:
                passed.append(diag_name)
            else:
                failed.append(diag_name)

        result.testable_threats_passed = passed
        result.testable_threats_failed = failed

        # Apply diagnostic-based downgrades
        if "leads_test" in failed:
            result.cap_to("BLOCKED_ID")
            result.counterfactual_reason_blocked = "Leads test failed: timing violation"

        if "leave_one_bank_out" in failed or "leave_one_unit_out" in failed:
            result.cap_to("REDUCED_FORM")
            result.risks["weak_variation"] = "high"

        if "exposure_variation" in failed:
            result.cap_to("BLOCKED_ID")
            result.counterfactual_reason_blocked = "Insufficient exposure variation"

        if "first_stage_f" in failed:
            result.cap_to("REDUCED_FORM")
            result.risks["weak_variation"] = "high"

        # Apply TSGuard results if available
        if ts_guard_result is not None:
            if getattr(ts_guard_result, "counterfactual_blocked", False):
                result.counterfactual_allowed = False
                result.counterfactual_reason_blocked = (
                    getattr(ts_guard_result, "reminder_text", "TSGuard blocked")
                )
            cap = getattr(ts_guard_result, "claim_level_cap", None)
            if cap:
                result.cap_to(cap)

        # Initialize split CF fields from legacy field
        result.shock_scenario_allowed = result.counterfactual_allowed
        result.policy_intervention_allowed = (
            result.counterfactual_allowed
            and result.claim_level == "IDENTIFIED_CAUSAL"
        )
        if not result.shock_scenario_allowed:
            result.reason_shock_blocked = result.counterfactual_reason_blocked
        if not result.policy_intervention_allowed:
            result.reason_policy_blocked = (
                result.counterfactual_reason_blocked
                or f"Policy CF requires IDENTIFIED_CAUSAL, edge has {result.claim_level}"
            )

        # Apply mode-aware gating (mode can only RESTRICT, never EXPAND)
        if query_mode is not None:
            from shared.agentic.query_mode import (
                QueryModeConfig, QueryMode,
                is_shock_cf_allowed, is_policy_cf_allowed,
            )
            config = QueryModeConfig.load()
            mode_spec = config.get_spec(query_mode)
            shock_ok, shock_reason = is_shock_cf_allowed(result.claim_level, mode_spec)
            policy_ok, policy_reason = is_policy_cf_allowed(result.claim_level, mode_spec)
            if not shock_ok:
                result.shock_scenario_allowed = False
                result.reason_shock_blocked = shock_reason
            if not policy_ok:
                result.policy_intervention_allowed = False
                result.reason_policy_blocked = policy_reason

        return result

    def generate_dashboard_row(
        self,
        edge_id: str,
        result: IdentifiabilityResult,
    ) -> dict[str, str]:
        """Generate a single row for the identifiability risk dashboard."""
        main_risk = "none"
        for risk_name, risk_level in result.risks.items():
            if risk_level == "high":
                main_risk = risk_name
                break
        if main_risk == "none":
            for risk_name, risk_level in result.risks.items():
                if risk_level == "medium":
                    main_risk = risk_name
                    break

        n_passed = len(result.testable_threats_passed)
        n_total = n_passed + len(result.testable_threats_failed)
        diag_summary = f"{n_passed}/{n_total} pass" if n_total > 0 else "n/a"

        return {
            "edge": edge_id,
            "claim_level": result.claim_level,
            "main_risk": main_risk,
            "diagnostics": diag_summary,
            "counterfactual": "ALLOWED" if result.counterfactual_allowed else "BLOCKED",
        }

    def generate_dashboard(
        self,
        results: dict[str, IdentifiabilityResult],
    ) -> str:
        """Generate the full identifiability risk dashboard as markdown."""
        lines = [
            "## Identifiability Risk Dashboard",
            "",
            "| Edge | Claim Level | Main Risk | Diagnostics | Counterfactual |",
            "|------|-------------|-----------|-------------|----------------|",
        ]

        for edge_id, result in results.items():
            row = self.generate_dashboard_row(edge_id, result)
            lines.append(
                f"| {row['edge']} | {row['claim_level']} | {row['main_risk']} | "
                f"{row['diagnostics']} | {row['counterfactual']} |"
            )

        return "\n".join(lines)
