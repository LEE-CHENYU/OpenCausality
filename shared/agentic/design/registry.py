"""
Design Registry.

Loads and manages the design registry from YAML configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataRequirements:
    """Data requirements for a design."""

    panel: bool = False
    time_series: bool = False
    running_variable: bool = False
    min_periods: int = 8
    min_units: int = 10
    min_obs: int = 30


@dataclass
class VariableRequirements:
    """Variable type requirements for a design."""

    treatment_type: list[str] = field(default_factory=lambda: ["continuous", "binary"])
    outcome_type: list[str] = field(default_factory=lambda: ["continuous", "share", "index"])
    instrument_count_min: int = 0


@dataclass
class TemplateSpec:
    """Model template specification."""

    model: str
    fixed_effects: list[str] = field(default_factory=list)
    standard_errors: str = "cluster"
    horizon_max: int | None = None


@dataclass
class DiagnosticsSpec:
    """Diagnostics specification for a design."""

    required: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)


@dataclass
class DesignSpec:
    """
    Specification for an identification design.

    Loaded from design_registry.yaml.
    """

    id: str
    name: str
    description: str = ""
    data_requirements: DataRequirements = field(default_factory=DataRequirements)
    variable_requirements: VariableRequirements = field(default_factory=VariableRequirements)
    template: TemplateSpec = field(default_factory=TemplateSpec)
    diagnostics: DiagnosticsSpec = field(default_factory=DiagnosticsSpec)
    outputs: list[str] = field(default_factory=list)

    # Credibility weight (higher = more credible design)
    credibility_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "requirements": {
                "data": {
                    "panel": self.data_requirements.panel,
                    "time_series": self.data_requirements.time_series,
                    "running_variable": self.data_requirements.running_variable,
                    "min_periods": self.data_requirements.min_periods,
                    "min_units": self.data_requirements.min_units,
                    "min_obs": self.data_requirements.min_obs,
                },
                "variables": {
                    "treatment_type": self.variable_requirements.treatment_type,
                    "outcome_type": self.variable_requirements.outcome_type,
                    "instrument_count_min": self.variable_requirements.instrument_count_min,
                },
            },
            "template": {
                "model": self.template.model,
                "fixed_effects": self.template.fixed_effects,
                "standard_errors": self.template.standard_errors,
            },
            "diagnostics": {
                "required": self.diagnostics.required,
                "optional": self.diagnostics.optional,
            },
            "outputs": self.outputs,
            "credibility_weight": self.credibility_weight,
        }


class DesignRegistry:
    """
    Registry of identification designs.

    Loads design specifications from YAML and provides lookup.
    """

    # Design credibility weights (higher = more credible)
    DEFAULT_CREDIBILITY_WEIGHTS = {
        "RDD": 1.0,           # Gold standard
        "IV_2SLS": 0.9,       # Strong if valid
        "DID_EVENT_STUDY": 0.8,  # Good with pre-trends
        "LOCAL_PROJECTIONS": 0.7,  # Dynamic effects
        "PANEL_FE_BACKDOOR": 0.6,  # Basic but common
    }

    def __init__(self, registry_path: Path | str | None = None):
        """
        Initialize registry.

        Args:
            registry_path: Path to design_registry.yaml. If None, uses default.
        """
        self.registry_path = registry_path
        self._designs: dict[str, DesignSpec] = {}
        self._defaults: dict[str, Any] = {}

        if registry_path:
            self.load(Path(registry_path))
        else:
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default designs without a file."""
        # Default designs matching the design_registry.yaml structure
        self._designs = {
            "PANEL_FE_BACKDOOR": DesignSpec(
                id="PANEL_FE_BACKDOOR",
                name="Panel FE Backdoor",
                description="Fixed effects panel regression using a backdoor adjustment set.",
                data_requirements=DataRequirements(panel=True, min_periods=8, min_units=10),
                variable_requirements=VariableRequirements(
                    treatment_type=["continuous", "binary"],
                    outcome_type=["continuous", "share", "index"],
                ),
                template=TemplateSpec(
                    model="y_it = beta * x_it + gamma' * controls_it + alpha_i + tau_t + e_it",
                    fixed_effects=["unit", "time"],
                    standard_errors="cluster(unit)",
                ),
                diagnostics=DiagnosticsSpec(
                    required=["residual_checks", "influence"],
                    optional=["placebo_exposure", "pre_trends"],
                ),
                outputs=["beta", "se", "ci", "fixed_effects", "controls"],
                credibility_weight=0.6,
            ),
            "DID_EVENT_STUDY": DesignSpec(
                id="DID_EVENT_STUDY",
                name="Difference-in-Differences Event Study",
                description="TWFE event study with pre-trend checks.",
                data_requirements=DataRequirements(panel=True, min_periods=12),
                variable_requirements=VariableRequirements(
                    treatment_type=["binary"],
                    outcome_type=["continuous", "share", "index"],
                ),
                template=TemplateSpec(
                    model="y_it = sum_k beta_k * 1[t = k] * treated_i + alpha_i + tau_t + e_it",
                    fixed_effects=["unit", "time"],
                    standard_errors="cluster(unit)",
                ),
                diagnostics=DiagnosticsSpec(
                    required=["pre_trends", "placebo_event"],
                    optional=["balanced_panel"],
                ),
                outputs=["event_time_betas", "pretrend_p", "ci_bands"],
                credibility_weight=0.8,
            ),
            "IV_2SLS": DesignSpec(
                id="IV_2SLS",
                name="Instrumental Variables 2SLS",
                description="Two-stage least squares with explicit exclusion restriction notes.",
                data_requirements=DataRequirements(min_obs=50),
                variable_requirements=VariableRequirements(
                    treatment_type=["continuous"],
                    outcome_type=["continuous", "share", "index"],
                    instrument_count_min=1,
                ),
                template=TemplateSpec(
                    model="x_it = pi * z_it + controls_it + u_it; y_it = beta * xhat_it + controls_it + v_it",
                    standard_errors="robust",
                ),
                diagnostics=DiagnosticsSpec(
                    required=["weak_iv", "first_stage_f", "exclusion_narrative"],
                    optional=["overid"],
                ),
                outputs=["beta", "se", "first_stage_f", "overid_p"],
                credibility_weight=0.9,
            ),
            "LOCAL_PROJECTIONS": DesignSpec(
                id="LOCAL_PROJECTIONS",
                name="Local Projections IRF",
                description="Jorda local projections for dynamic impulse responses.",
                data_requirements=DataRequirements(time_series=True, min_periods=20),
                variable_requirements=VariableRequirements(
                    treatment_type=["continuous"],
                    outcome_type=["continuous", "share", "index"],
                ),
                template=TemplateSpec(
                    model="y_{t+h} = beta_h * shock_t + controls_t + e_{t+h}",
                    standard_errors="hac",
                    horizon_max=12,
                ),
                diagnostics=DiagnosticsSpec(
                    required=["stability", "lag_sensitivity"],
                    optional=["structural_breaks"],
                ),
                outputs=["irf", "ci_bands", "horizon"],
                credibility_weight=0.7,
            ),
            "RDD": DesignSpec(
                id="RDD",
                name="Regression Discontinuity",
                description="Local linear RD with density and bandwidth checks.",
                data_requirements=DataRequirements(running_variable=True, min_obs=200),
                variable_requirements=VariableRequirements(
                    treatment_type=["binary"],
                    outcome_type=["continuous", "share", "index"],
                ),
                template=TemplateSpec(
                    model="y = beta * 1[x >= c] + f(x) + e",
                    standard_errors="robust",
                ),
                diagnostics=DiagnosticsSpec(
                    required=["density", "bandwidth_sensitivity"],
                    optional=["covariate_balance"],
                ),
                outputs=["beta", "se", "bandwidth"],
                credibility_weight=1.0,
            ),
        }

        self._defaults = {
            "standard_errors": "cluster",
            "missing_rate_max": 0.10,
            "min_obs": 30,
        }

    def load(self, path: Path) -> None:
        """Load registry from YAML file."""
        if not path.exists():
            logger.warning(f"Registry file not found: {path}, using defaults")
            self._load_defaults()
            return

        with open(path) as f:
            data = yaml.safe_load(f)

        self._defaults = data.get("_defaults", {})

        for design_data in data.get("designs", []):
            design = self._parse_design(design_data)
            self._designs[design.id] = design

    def _parse_design(self, data: dict) -> DesignSpec:
        """Parse a design from YAML data."""
        requirements = data.get("requirements", {})
        data_req = requirements.get("data", {})
        var_req = requirements.get("variables", {})
        template = data.get("template", {})
        diagnostics = data.get("diagnostics", {})

        design_id = data["id"]
        credibility_weight = self.DEFAULT_CREDIBILITY_WEIGHTS.get(design_id, 0.5)

        return DesignSpec(
            id=design_id,
            name=data.get("name", design_id),
            description=data.get("description", ""),
            data_requirements=DataRequirements(
                panel=data_req.get("panel", False),
                time_series=data_req.get("time_series", False),
                running_variable=data_req.get("running_variable", False),
                min_periods=data_req.get("min_periods", 8),
                min_units=data_req.get("min_units", 10),
                min_obs=data_req.get("min_obs", self._defaults.get("min_obs", 30)),
            ),
            variable_requirements=VariableRequirements(
                treatment_type=var_req.get("treatment_type", ["continuous", "binary"]),
                outcome_type=var_req.get("outcome_type", ["continuous", "share", "index"]),
                instrument_count_min=var_req.get("instrument_count_min", 0),
            ),
            template=TemplateSpec(
                model=template.get("model", ""),
                fixed_effects=template.get("fixed_effects", []),
                standard_errors=template.get(
                    "standard_errors",
                    self._defaults.get("standard_errors", "cluster")
                ),
                horizon_max=template.get("horizon_max"),
            ),
            diagnostics=DiagnosticsSpec(
                required=diagnostics.get("required", []),
                optional=diagnostics.get("optional", []),
            ),
            outputs=data.get("outputs", {}).get("edge_card_fields", []),
            credibility_weight=credibility_weight,
        )

    def get(self, design_id: str) -> DesignSpec | None:
        """Get a design by ID."""
        return self._designs.get(design_id)

    def get_all(self) -> list[DesignSpec]:
        """Get all designs."""
        return list(self._designs.values())

    def get_by_credibility(self) -> list[DesignSpec]:
        """Get designs sorted by credibility (highest first)."""
        return sorted(
            self._designs.values(),
            key=lambda d: d.credibility_weight,
            reverse=True,
        )

    def __contains__(self, design_id: str) -> bool:
        """Check if design exists."""
        return design_id in self._designs

    def __iter__(self):
        """Iterate over designs."""
        return iter(self._designs.values())


def load_registry(path: Path | str | None = None) -> DesignRegistry:
    """
    Load a design registry.

    Args:
        path: Path to registry YAML. If None, uses defaults.

    Returns:
        Loaded DesignRegistry
    """
    return DesignRegistry(path)
