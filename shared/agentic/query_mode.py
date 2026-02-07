"""
Query Mode: STRUCTURAL / REDUCED_FORM / DESCRIPTIVE.

Single-file module (avoids module-vs-package conflict per MEMORY.md).

Determines per-edge permissions based on:
  edge_type x claim_level x mode -> propagation, shock CF, policy CF

Core principle: Single DAG, per-edge role tags derived from
edge_type + claim_level. Mode = which roles are permitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Claim level hierarchy (strongest to weakest)
CLAIM_HIERARCHY = [
    "IDENTIFIED_CAUSAL",
    "REDUCED_FORM",
    "DESCRIPTIVE",
    "BLOCKED_ID",
]


class QueryMode(Enum):
    STRUCTURAL = "STRUCTURAL"
    REDUCED_FORM = "REDUCED_FORM"
    DESCRIPTIVE = "DESCRIPTIVE"


@dataclass
class QueryModeSpec:
    """Specification for a single query mode, loaded from YAML."""

    name: str
    description: str = ""
    propagation_requires: list[str] = field(default_factory=list)
    policy_requires_claim: str | None = None
    shock_requires_claim: str | None = None
    allows_policy_intervention: bool = False
    allows_shock_scenario: bool = False
    strict_diagnostics: bool = False


@dataclass
class QueryModeConfig:
    """All query mode definitions loaded from YAML."""

    modes: dict[str, QueryModeSpec] = field(default_factory=dict)
    default_mode: str = "REDUCED_FORM"

    @classmethod
    def load(cls, path: Path | str | None = None) -> QueryModeConfig:
        """Load query mode config from YAML."""
        path = Path(path) if path else Path("config/agentic/query_modes.yaml")
        if not path.exists():
            logger.warning(f"Query modes config not found at {path}, using defaults")
            return cls._default()
        with open(path) as f:
            raw = yaml.safe_load(f)
        config = cls(default_mode=raw.get("default_mode", "REDUCED_FORM"))
        for mode_name, mode_data in raw.get("modes", {}).items():
            cf = mode_data.get("counterfactual", {})
            config.modes[mode_name] = QueryModeSpec(
                name=mode_name,
                description=mode_data.get("description", ""),
                propagation_requires=mode_data.get("propagation_requires", []),
                policy_requires_claim=cf.get("policy_requires_claim"),
                shock_requires_claim=cf.get("shock_requires_claim"),
                allows_policy_intervention=mode_data.get("allows_policy_intervention", False),
                allows_shock_scenario=mode_data.get("allows_shock_scenario", False),
                strict_diagnostics=mode_data.get("strict_diagnostics", False),
            )
        return config

    @classmethod
    def _default(cls) -> QueryModeConfig:
        """Return hardcoded default config (matches YAML)."""
        return cls(
            default_mode="REDUCED_FORM",
            modes={
                "STRUCTURAL": QueryModeSpec(
                    name="STRUCTURAL",
                    propagation_requires=["structural", "bridge", "identity"],
                    policy_requires_claim="IDENTIFIED_CAUSAL",
                    shock_requires_claim="IDENTIFIED_CAUSAL",
                    allows_policy_intervention=True,
                    allows_shock_scenario=True,
                    strict_diagnostics=True,
                ),
                "REDUCED_FORM": QueryModeSpec(
                    name="REDUCED_FORM",
                    propagation_requires=["structural", "reduced_form", "bridge", "identity"],
                    policy_requires_claim=None,
                    shock_requires_claim="REDUCED_FORM",
                    allows_policy_intervention=False,
                    allows_shock_scenario=True,
                ),
                "DESCRIPTIVE": QueryModeSpec(
                    name="DESCRIPTIVE",
                    propagation_requires=["structural", "reduced_form", "bridge", "identity", "diagnostic_only"],
                    policy_requires_claim=None,
                    shock_requires_claim=None,
                    allows_policy_intervention=False,
                    allows_shock_scenario=False,
                ),
            },
        )

    def get_spec(self, mode: QueryMode | str) -> QueryModeSpec:
        """Get spec for a mode."""
        name = mode.value if isinstance(mode, QueryMode) else mode
        if name not in self.modes:
            raise ValueError(f"Unknown query mode: {name}")
        return self.modes[name]


# ---------------------------------------------------------------------------
# Derivation functions (single source of truth)
# ---------------------------------------------------------------------------

def derive_propagation_role(edge_type: str, claim_level: str) -> str:
    """
    Derive propagation role from edge_type and claim_level.

    Returns one of: "identity", "bridge", "structural", "reduced_form", "diagnostic_only"
    """
    if edge_type == "identity":
        return "identity"
    if edge_type in ("mechanical", "bridge"):
        return "bridge"
    if edge_type == "reaction_function":
        return "diagnostic_only"
    if edge_type == "immutable":
        return "structural"
    # causal edges: depends on claim_level
    if claim_level == "IDENTIFIED_CAUSAL":
        return "structural"
    if claim_level == "REDUCED_FORM":
        return "reduced_form"
    # DESCRIPTIVE or BLOCKED_ID
    return "diagnostic_only"


def is_edge_allowed_for_propagation(role: str, mode_spec: QueryModeSpec) -> bool:
    """Check if an edge with given role is allowed for propagation in this mode."""
    return role in mode_spec.propagation_requires


def claim_meets_threshold(claim_level: str, required_claim: str) -> bool:
    """Check if claim_level meets or exceeds the required threshold."""
    try:
        claim_idx = CLAIM_HIERARCHY.index(claim_level)
        req_idx = CLAIM_HIERARCHY.index(required_claim)
    except ValueError:
        return False
    return claim_idx <= req_idx  # lower index = stronger claim


def is_shock_cf_allowed(
    claim_level: str,
    mode_spec: QueryModeSpec,
) -> tuple[bool, str | None]:
    """Check if shock counterfactual is allowed for this claim_level in this mode."""
    if not mode_spec.allows_shock_scenario:
        return False, f"{mode_spec.name} mode does not allow shock scenarios"
    if mode_spec.shock_requires_claim is None:
        return False, f"{mode_spec.name} mode blocks all shock counterfactuals"
    if claim_meets_threshold(claim_level, mode_spec.shock_requires_claim):
        return True, None
    return (
        False,
        f"Shock CF requires {mode_spec.shock_requires_claim}+, "
        f"edge has {claim_level}",
    )


def is_policy_cf_allowed(
    claim_level: str,
    mode_spec: QueryModeSpec,
) -> tuple[bool, str | None]:
    """Check if policy counterfactual is allowed for this claim_level in this mode."""
    if not mode_spec.allows_policy_intervention:
        return False, f"{mode_spec.name} mode does not allow policy interventions"
    if mode_spec.policy_requires_claim is None:
        return False, f"{mode_spec.name} mode blocks all policy counterfactuals"
    if claim_meets_threshold(claim_level, mode_spec.policy_requires_claim):
        return True, None
    return (
        False,
        f"Policy CF requires {mode_spec.policy_requires_claim}+, "
        f"edge has {claim_level}",
    )
