"""
Refinement Whitelist.

Defines the constrained menu of allowed spec changes to prevent
p-hacking through unbounded specification search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


class GovernanceViolation(Exception):
    """Exception raised when a refinement violates governance rules."""

    def __init__(self, refinement: 'Refinement', rule_id: str | None = None):
        self.refinement = refinement
        self.rule_id = rule_id
        msg = f"Refinement not in whitelist: {refinement.field} = {refinement.new_value}"
        if rule_id:
            msg += f" (checked against rule: {rule_id})"
        super().__init__(msg)


@dataclass
class Refinement:
    """A single specification refinement."""

    edge_id: str
    field: str  # e.g., "controls", "lag_length", "bandwidth"
    old_value: Any
    new_value: Any
    reason: str
    iteration: int = 0

    # After approval
    new_spec_hash: str | None = None
    approved_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "iteration": self.iteration,
            "new_spec_hash": self.new_spec_hash,
            "approved_by": self.approved_by,
        }


@dataclass
class WhitelistRule:
    """A single whitelist rule."""

    id: str
    field: str
    allowed_values: list[Any] | None = None  # For discrete values
    value_range: tuple[float, float] | None = None  # For continuous values
    max_additions: int | None = None  # For list fields
    design: str | None = None  # Design-specific rule

    def allows(self, refinement: Refinement) -> bool:
        """Check if this rule allows the refinement."""
        if self.field != refinement.field:
            return False

        new_val = refinement.new_value

        # Check allowed values (discrete)
        if self.allowed_values is not None:
            if isinstance(new_val, list):
                # All values must be in allowed list
                return all(v in self.allowed_values for v in new_val)
            return new_val in self.allowed_values

        # Check value range (continuous)
        if self.value_range is not None:
            if not isinstance(new_val, (int, float)):
                return False
            return self.value_range[0] <= new_val <= self.value_range[1]

        # Check max additions (for list fields)
        if self.max_additions is not None:
            if not isinstance(new_val, list):
                return False
            old_val = refinement.old_value or []
            additions = len(new_val) - len(old_val)
            return additions <= self.max_additions

        return False


@dataclass
class RefinementWhitelist:
    """
    Constrained menu of allowed specification changes.

    This is the primary mechanism for preventing p-hacking through
    unbounded specification search.
    """

    # Control variable whitelist
    allowed_controls: list[str] = field(default_factory=list)

    # Lag length range
    lag_range: tuple[int, int] = (1, 6)

    # Instrument whitelist
    allowed_instruments: list[str] = field(default_factory=list)

    # Regime split dates
    allowed_regime_splits: list[str] = field(default_factory=list)

    # RDD bandwidth range
    bandwidth_range: tuple[float, float] = (0.5, 2.0)

    # Maximum controls to add per iteration
    max_controls_per_iteration: int = 2

    # Custom rules
    rules: list[WhitelistRule] = field(default_factory=list)

    def __post_init__(self):
        # Build default rules
        self._build_default_rules()

    def _build_default_rules(self) -> None:
        """Build default rules from configuration."""
        if self.allowed_controls:
            self.rules.append(WhitelistRule(
                id="add_control",
                field="controls",
                allowed_values=self.allowed_controls,
                max_additions=self.max_controls_per_iteration,
            ))

        self.rules.append(WhitelistRule(
            id="change_lag_length",
            field="lag_length",
            value_range=self.lag_range,
        ))

        if self.allowed_instruments:
            self.rules.append(WhitelistRule(
                id="swap_instrument",
                field="instruments",
                allowed_values=self.allowed_instruments,
            ))

        if self.allowed_regime_splits:
            self.rules.append(WhitelistRule(
                id="regime_split",
                field="regime_split_date",
                allowed_values=self.allowed_regime_splits,
            ))

        self.rules.append(WhitelistRule(
            id="bandwidth_change",
            field="bandwidth",
            value_range=self.bandwidth_range,
            design="RDD",
        ))

    def allows(self, refinement: Refinement) -> bool:
        """
        Check if a refinement is allowed.

        Args:
            refinement: The proposed refinement

        Returns:
            True if allowed, False otherwise
        """
        for rule in self.rules:
            if rule.allows(refinement):
                return True
        return False

    def get_rule_id(self, refinement: Refinement) -> str | None:
        """
        Get the rule ID that allows a refinement.

        Args:
            refinement: The refinement to check

        Returns:
            Rule ID if allowed, None otherwise
        """
        for rule in self.rules:
            if rule.allows(refinement):
                return rule.id
        return None

    def validate(self, refinement: Refinement) -> None:
        """
        Validate a refinement, raising GovernanceViolation if not allowed.

        Args:
            refinement: The refinement to validate

        Raises:
            GovernanceViolation: If refinement is not allowed
        """
        if not self.allows(refinement):
            raise GovernanceViolation(refinement)

    def validate_all(self, refinements: list[Refinement]) -> list[Refinement]:
        """
        Validate multiple refinements.

        Args:
            refinements: List of refinements to validate

        Returns:
            List of allowed refinements

        Raises:
            GovernanceViolation: If any refinement is not allowed
        """
        allowed = []
        for r in refinements:
            self.validate(r)  # Raises if not allowed
            r.approved_by = self.get_rule_id(r)
            allowed.append(r)
        return allowed

    def filter_allowed(self, refinements: list[Refinement]) -> list[Refinement]:
        """
        Filter refinements to only allowed ones (no exception).

        Args:
            refinements: List of refinements

        Returns:
            List of allowed refinements
        """
        allowed = []
        for r in refinements:
            if self.allows(r):
                r.approved_by = self.get_rule_id(r)
                allowed.append(r)
        return allowed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed_controls": self.allowed_controls,
            "lag_range": list(self.lag_range),
            "allowed_instruments": self.allowed_instruments,
            "allowed_regime_splits": self.allowed_regime_splits,
            "bandwidth_range": list(self.bandwidth_range),
            "max_controls_per_iteration": self.max_controls_per_iteration,
            "rules": [
                {
                    "id": r.id,
                    "field": r.field,
                    "allowed_values": r.allowed_values,
                    "value_range": list(r.value_range) if r.value_range else None,
                    "max_additions": r.max_additions,
                    "design": r.design,
                }
                for r in self.rules
            ],
        }


def create_whitelist_from_config(config: dict) -> RefinementWhitelist:
    """
    Create a whitelist from configuration dictionary.

    Args:
        config: Configuration with allowed refinements

    Returns:
        Configured RefinementWhitelist
    """
    return RefinementWhitelist(
        allowed_controls=config.get("allowed_controls", []),
        lag_range=tuple(config.get("lag_range", [1, 6])),
        allowed_instruments=config.get("allowed_instruments", []),
        allowed_regime_splits=config.get("allowed_regime_splits", []),
        bandwidth_range=tuple(config.get("bandwidth_range", [0.5, 2.0])),
        max_controls_per_iteration=config.get("max_controls_per_iteration", 2),
    )
