"""
Issue Registry: Load and apply issue detection rules.

Loads rules from config/agentic/issue_registry.yaml and provides
methods to detect issues in EdgeCards, DAG specs, and estimation results.
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
class IssueRule:
    """A single issue detection rule."""

    rule_id: str
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    trigger: Literal["pre_run", "post_run", "cross_run"]
    auto_fixable: bool
    requires_human: bool
    description: str
    action: str | None = None
    resolution_policy: str | None = None


class IssueRegistry:
    """
    Registry of issue detection rules loaded from YAML.

    Usage:
        registry = IssueRegistry.load()
        rule = registry.get_rule("UNIT_MISSING_IN_EDGECARD")
        pre_rules = registry.get_rules_by_trigger("pre_run")
    """

    def __init__(self, rules: list[IssueRule] | None = None):
        self.rules: dict[str, IssueRule] = {}
        if rules:
            for rule in rules:
                self.rules[rule.rule_id] = rule

    @classmethod
    def load(cls, config_path: Path | None = None) -> IssueRegistry:
        """Load rules from YAML config."""
        if config_path is None:
            config_path = Path("config/agentic/issue_registry.yaml")

        if not config_path.exists():
            logger.warning(f"Issue registry not found: {config_path}")
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        rules = []
        for rule_data in data.get("rules", []):
            rules.append(IssueRule(
                rule_id=rule_data["rule_id"],
                severity=rule_data["severity"],
                trigger=rule_data["trigger"],
                auto_fixable=rule_data.get("auto_fixable", False),
                requires_human=rule_data.get("requires_human", False),
                description=rule_data.get("description", ""),
                action=rule_data.get("action"),
                resolution_policy=rule_data.get("resolution_policy"),
            ))

        return cls(rules)

    def get_rule(self, rule_id: str) -> IssueRule | None:
        """Get a rule by ID."""
        return self.rules.get(rule_id)

    def get_rules_by_trigger(self, trigger: str) -> list[IssueRule]:
        """Get all rules for a given trigger phase."""
        return [r for r in self.rules.values() if r.trigger == trigger]

    def create_issue(
        self,
        rule_id: str,
        message: str,
        scope: str = "edge",
        edge_id: str | None = None,
        node_id: str | None = None,
        evidence: dict | None = None,
        suggested_fix: dict | None = None,
    ) -> Issue | None:
        """Create an Issue from a rule ID with pre-filled metadata."""
        rule = self.get_rule(rule_id)
        if rule is None:
            logger.warning(f"Unknown rule_id: {rule_id}")
            return None

        return Issue(
            run_id="",  # Set by ledger
            timestamp="",  # Set by ledger
            severity=rule.severity,
            rule_id=rule.rule_id,
            scope=scope,
            message=message,
            edge_id=edge_id,
            node_id=node_id,
            evidence=evidence or {},
            auto_fixable=rule.auto_fixable,
            suggested_fix=suggested_fix,
            requires_human=rule.requires_human,
        )

    def detect_pre_run_issues(
        self,
        dag_config: dict[str, Any],
        ledger: IssueLedger,
    ) -> list[Issue]:
        """Run pre-run issue detection on a DAG config."""
        detected = []
        edges = {e["id"]: e for e in dag_config.get("edges", [])}

        for edge_id, edge in edges.items():
            # UNIT_MISSING_IN_EDGECARD
            unit_spec = edge.get("unit_specification", {})
            if not unit_spec.get("treatment_unit") or not unit_spec.get("outcome_unit"):
                issue = self.create_issue(
                    "UNIT_MISSING_IN_EDGECARD",
                    "EdgeCard missing treatment_unit/outcome_unit; chain propagation unsafe.",
                    edge_id=edge_id,
                    evidence={"card_path": f"edge:{edge_id}"},
                    suggested_fix={"action": "add_edge_units_registry_entry", "target": "EDGE_UNITS.yaml"},
                )
                if issue:
                    detected.append(ledger.add(issue))

            # REACTION_FUNCTION_EDGE
            if edge.get("edge_type") == "reaction_function":
                issue = self.create_issue(
                    "REACTION_FUNCTION_EDGE",
                    "Policy reaction edge cannot be used for shock propagation.",
                    edge_id=edge_id,
                )
                if issue:
                    detected.append(ledger.add(issue))

            # FREQUENCY_ALIGNMENT_ERROR
            from_freq = edge.get("from_frequency")
            to_freq = edge.get("to_frequency")
            if from_freq and to_freq and from_freq != to_freq:
                agg_rule = edge.get("aggregation_rule")
                if not agg_rule:
                    issue = self.create_issue(
                        "FREQUENCY_ALIGNMENT_ERROR",
                        f"Frequency mismatch ({from_freq} -> {to_freq}) without aggregation rule.",
                        edge_id=edge_id,
                        suggested_fix={"action": "add_frequency_normalization"},
                    )
                    if issue:
                        detected.append(ledger.add(issue))

        return detected

    def detect_post_run_issues(
        self,
        edge_cards: dict[str, Any],
        ledger: IssueLedger,
    ) -> list[Issue]:
        """Run post-run issue detection on estimated EdgeCards."""
        detected = []

        for edge_id, card in edge_cards.items():
            estimates = card.estimates if hasattr(card, "estimates") else None
            if not estimates:
                continue

            # SMALL_SAMPLE_INFERENCE
            n_eff = getattr(estimates, "n_effective_obs_h0", None)
            if n_eff is not None and n_eff < 30:
                issue = self.create_issue(
                    "SMALL_SAMPLE_INFERENCE",
                    f"N={n_eff} < 30 with HAC SEs; inference fragile.",
                    edge_id=edge_id,
                    evidence={"n_effective": n_eff},
                )
                if issue:
                    detected.append(ledger.add(issue))

            # UNIT_MISSING_IN_EDGECARD (post-estimation check)
            if not getattr(estimates, "treatment_unit", "") or not getattr(estimates, "outcome_unit", ""):
                issue = self.create_issue(
                    "UNIT_MISSING_IN_EDGECARD",
                    "EdgeCard missing treatment_unit/outcome_unit.",
                    edge_id=edge_id,
                    suggested_fix={"action": "add_edge_units"},
                )
                if issue:
                    detected.append(ledger.add(issue))

            # SIGNIFICANT_BUT_NOT_IDENTIFIED
            identification = getattr(card, "identification", None)
            if identification:
                claim = getattr(identification, "claim_level", None)
                pvalue = getattr(estimates, "pvalue", None)
                if pvalue is not None and pvalue < 0.05 and claim != "IDENTIFIED_CAUSAL":
                    issue = self.create_issue(
                        "SIGNIFICANT_BUT_NOT_IDENTIFIED",
                        f"p={pvalue:.4f} but claim_level={claim}. "
                        "Significance does not establish causation.",
                        edge_id=edge_id,
                        evidence={"pvalue": pvalue, "claim_level": claim},
                    )
                    if issue:
                        detected.append(ledger.add(issue))

            # LOO_INSTABILITY
            diagnostics = getattr(card, "diagnostics", {})
            loo_diag = diagnostics.get("leave_one_bank_out") if isinstance(diagnostics, dict) else None
            if loo_diag and not getattr(loo_diag, "passed", True):
                issue = self.create_issue(
                    "LOO_INSTABILITY",
                    "Leave-one-out shows sign flip or >50% magnitude change.",
                    edge_id=edge_id,
                    evidence={"loo_message": getattr(loo_diag, "message", "")},
                )
                if issue:
                    detected.append(ledger.add(issue))

            # RATING_DIAGNOSTICS_CONFLICT
            rating = getattr(card, "credibility_rating", "D")
            all_pass = all(
                getattr(d, "passed", True)
                for d in diagnostics.values()
            ) if isinstance(diagnostics, dict) else True
            if rating == "A" and not all_pass:
                issue = self.create_issue(
                    "RATING_DIAGNOSTICS_CONFLICT",
                    "A-rating despite failed diagnostics.",
                    edge_id=edge_id,
                    suggested_fix={"action": "recompute_rating"},
                )
                if issue:
                    detected.append(ledger.add(issue))

        return detected
