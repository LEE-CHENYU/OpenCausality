"""
ModelSmithCritic Agent: Validate design identification.

Reviews design proposals from ModelSmith and checks whether
the chosen identification strategy is valid. Can raise issues
that block estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from shared.agentic.issues.issue_ledger import Issue, IssueLedger
from shared.agentic.identification.screen import IdentifiabilityScreen, IdentifiabilityResult

logger = logging.getLogger(__name__)


@dataclass
class CriticVerdict:
    """Result of ModelSmithCritic review."""

    edge_id: str
    design: str
    approved: bool
    identifiability: IdentifiabilityResult | None = None
    issues_raised: list[str] | None = None
    recommendation: str = ""


class ModelSmithCritic:
    """
    Critic agent that validates design choices.

    Checks:
    1. Does the design match the edge type?
    2. Are identification assumptions plausible?
    3. Are required diagnostics available for this design?
    4. Is the exposure variable predetermined (for shift-share)?
    """

    def __init__(self):
        self.screen = IdentifiabilityScreen()

    def review(
        self,
        edge_id: str,
        design: str,
        edge_type: str,
        has_instrument: bool = False,
        has_panel: bool = False,
        has_exposure_variation: bool = False,
        ledger: IssueLedger | None = None,
    ) -> CriticVerdict:
        """Review a design proposal."""
        issues_raised = []

        # Run identifiability screen
        id_result = self.screen.screen_post_design(edge_id, design)

        # Check: reaction function edges shouldn't use causal designs
        if edge_type == "reaction_function" and design not in ("DESCRIPTIVE", "OLS"):
            issues_raised.append("REACTION_FUNCTION_EDGE")
            if ledger:
                ledger.add_from_rule(
                    rule_id="REACTION_FUNCTION_EDGE",
                    severity="CRITICAL",
                    message=f"Design {design} applied to reaction function edge {edge_id}",
                    edge_id=edge_id,
                    requires_human=True,
                )

        # Check: panel with time FE needs exposure x shock
        if has_panel and design in ("PANEL_LP", "PANEL_FE") and not has_exposure_variation:
            issues_raised.append("TIME_FE_ABSORBS_SHOCK")
            if ledger:
                ledger.add_from_rule(
                    rule_id="TIME_FE_ABSORBS_SHOCK",
                    severity="CRITICAL",
                    message=f"Panel design needs Exposure x Shock interaction for {edge_id}",
                    edge_id=edge_id,
                    auto_fixable=True,
                    suggested_fix={"action": "switch_to_exposure_shock"},
                )

        # Check: mechanical edge estimated as regression
        if edge_type in ("bridge", "identity") and design in ("LOCAL_PROJECTIONS", "OLS"):
            issues_raised.append("MECHANICAL_EDGE_ESTIMATED")
            if ledger:
                ledger.add_from_rule(
                    rule_id="MECHANICAL_EDGE_ESTIMATED",
                    severity="HIGH",
                    message=f"Mechanical edge {edge_id} should use bridge/identity design",
                    edge_id=edge_id,
                    auto_fixable=True,
                    suggested_fix={"action": "convert_to_bridge"},
                )

        approved = len(issues_raised) == 0

        return CriticVerdict(
            edge_id=edge_id,
            design=design,
            approved=approved,
            identifiability=id_result,
            issues_raised=issues_raised,
            recommendation=(
                "Design approved" if approved
                else f"Design has issues: {', '.join(issues_raised)}"
            ),
        )
