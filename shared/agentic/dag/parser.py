"""
DAG Parser with Temporal Semantics.

Parses DAG YAML specifications into Python dataclasses with full
temporal semantics (lags, release lags, contemporaneous effects).

Key features:
- NodeSpec with frequency, release_lag, and identity definitions
- EdgeSpec with timing constraints (lag, lag_unit, contemporaneous)
- EstimandSpec supporting IRF horizons
- Automatic dependency computation from identity formulas
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class ReleaseLag:
    """Publication timing for a data node."""

    periods: int = 1
    lag_unit: Literal["day", "month", "quarter", "year"] = "month"

    def to_dict(self) -> dict:
        return {"periods": self.periods, "lag_unit": self.lag_unit}


@dataclass
class IdentityDef:
    """Definition of a derived node via formula."""

    name: str
    formula: str
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "formula": self.formula,
            "depends_on": self.depends_on,
        }


@dataclass
class SourceEntry:
    """A single data source specification."""

    connector: str
    dataset: str
    series: str
    notes: str = ""

    def to_dict(self) -> dict:
        d = {
            "connector": self.connector,
            "dataset": self.dataset,
            "series": self.series,
        }
        if self.notes:
            d["notes"] = self.notes
        return d


@dataclass
class DataSourceConfig:
    """Data source configuration with preferred and fallback sources."""

    preferred: list[SourceEntry] = field(default_factory=list)
    fallback: list[SourceEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "preferred": [s.to_dict() for s in self.preferred],
            "fallback": [s.to_dict() for s in self.fallback],
        }


@dataclass
class AggregationConfig:
    """Configuration for frequency aggregation."""

    method: Literal["end_of_period", "average", "sum", "last"] = "average"
    seasonal_adjust: bool = False
    deflator: str | None = None
    deflator_timing: Literal["same", "lagged"] = "same"

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "seasonal_adjust": self.seasonal_adjust,
            "deflator": self.deflator,
            "deflator_timing": self.deflator_timing,
        }


@dataclass
class RoleConstraints:
    """Role constraints for a node in estimation."""

    never_control: bool = False
    never_instrument: bool = False

    def to_dict(self) -> dict:
        return {
            "never_control": self.never_control,
            "never_instrument": self.never_instrument,
        }


@dataclass
class ValidatedEvidence:
    """
    Validated evidence artifact for an edge or node.

    Used to mark Block A/B/F results as immutable to prevent
    re-estimation or optimization by the agentic loop.
    """

    block_id: str  # e.g., "block_a_fx_passthrough"
    description: str = ""
    estimate: float | None = None
    se: float | None = None
    result_hash: str | None = None  # SHA256 hash of original results
    immutable: bool = True  # Agent cannot re-estimate

    def to_dict(self) -> dict:
        d = {
            "block_id": self.block_id,
            "description": self.description,
            "immutable": self.immutable,
        }
        if self.estimate is not None:
            d["estimate"] = self.estimate
        if self.se is not None:
            d["se"] = self.se
        if self.result_hash:
            d["result_hash"] = self.result_hash
        return d


@dataclass
class EdgeInterpretation:
    """
    Interpretation boundary for a causal edge.

    Enforces what the edge estimate can and cannot be used for.
    """

    is_field: str = ""  # What the estimate IS (stored as 'is' in YAML)
    is_not: list[str] = field(default_factory=list)  # What this is NOT
    allowed_uses: list[str] = field(default_factory=list)  # Permitted use cases
    forbidden_uses: list[str] = field(default_factory=list)  # Forbidden use cases

    def to_dict(self) -> dict:
        d = {}
        if self.is_field:
            d["is"] = self.is_field
        if self.is_not:
            d["is_not"] = self.is_not
        if self.allowed_uses:
            d["allowed_uses"] = self.allowed_uses
        if self.forbidden_uses:
            d["forbidden_uses"] = self.forbidden_uses
        return d

    def is_use_allowed(self, use_case: str) -> bool:
        """Check if a use case is allowed."""
        if use_case in self.forbidden_uses:
            return False
        if not self.allowed_uses:
            return True
        return use_case in self.allowed_uses


@dataclass
class NodeSpec:
    """
    Specification for a DAG node (variable).

    Includes temporal semantics:
    - frequency: data frequency
    - release_lag: when data is published relative to reference period
    - derived: whether node is computed from identity
    - identity: formula definition for derived nodes
    - depends_on: auto-computed from formula
    """

    id: str
    name: str
    description: str = ""
    unit: str = "level"
    frequency: Literal["daily", "monthly", "quarterly", "annual", "static", "event"] = "monthly"
    type: Literal["continuous", "binary", "count", "share", "index", "exposure"] = "continuous"
    observed: bool = True
    latent: bool = False

    # Scope boundary (v2)
    scope: Literal["kazakhstan_only", "consolidated", "global", "unknown"] = "unknown"

    # Temporal semantics
    release_lag: ReleaseLag | None = None
    derived: bool = False
    identity: IdentityDef | None = None
    depends_on: list[str] = field(default_factory=list)

    # Data source
    source: DataSourceConfig | None = None

    # Transforms
    transforms: list[str] = field(default_factory=list)

    # Aggregation config (for mixed-frequency alignment)
    aggregation: AggregationConfig | None = None

    # Role constraints
    role_constraints: RoleConstraints | None = None

    # Validated evidence (for immutable Block A/B/F results)
    validated_evidence: ValidatedEvidence | None = None

    # Legacy identities field (for backward compatibility)
    identities: list[IdentityDef] = field(default_factory=list)

    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Auto-compute depends_on from identity formula
        if self.identity and not self.depends_on:
            self.depends_on = self._extract_dependencies(self.identity.formula)
            self.identity.depends_on = self.depends_on

        # Mark as derived if identity is present
        if self.identity:
            self.derived = True

    def _extract_dependencies(self, formula: str) -> list[str]:
        """Extract variable names from a formula string."""
        # Match word characters that could be variable names
        # Exclude common operators and functions
        tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', formula)
        # Filter out common math functions/operators
        excluded = {'log', 'exp', 'sqrt', 'abs', 'sin', 'cos', 'tan',
                   'min', 'max', 'sum', 'mean', 'std', 'diff', 'lag'}
        return [t for t in tokens if t not in excluded]

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        d = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "frequency": self.frequency,
            "type": self.type,
            "observed": self.observed,
            "latent": self.latent,
        }

        if self.scope != "unknown":
            d["scope"] = self.scope
        if self.release_lag:
            d["release_lag"] = self.release_lag.to_dict()
        if self.derived:
            d["derived"] = self.derived
        if self.identity:
            d["identity"] = self.identity.to_dict()
        if self.depends_on:
            d["depends_on"] = self.depends_on
        if self.source:
            d["source"] = self.source.to_dict()
        if self.transforms:
            d["transforms"] = self.transforms
        if self.aggregation:
            d["aggregation"] = self.aggregation.to_dict()
        if self.role_constraints:
            d["role_constraints"] = self.role_constraints.to_dict()
        if self.validated_evidence:
            d["validated_evidence"] = self.validated_evidence.to_dict()
        if self.identities:
            d["identities"] = [i.to_dict() for i in self.identities]
        if self.tags:
            d["tags"] = self.tags

        return d


@dataclass
class EdgeTiming:
    """
    Temporal semantics for a causal edge.

    Specifies the timing relationship between treatment and outcome:
    - lag: treatment[t] affects outcome[t+lag]
    - lag_unit: time unit for lag
    - contemporaneous: if True, treatment[t] affects outcome[t]
    - max_anticipation: leads allowed (usually 0)
    """

    lag: int = 1
    lag_unit: Literal["day", "month", "quarter", "year"] = "quarter"
    contemporaneous: bool = False
    max_anticipation: int = 0

    def to_dict(self) -> dict:
        return {
            "lag": self.lag,
            "lag_unit": self.lag_unit,
            "contemporaneous": self.contemporaneous,
            "max_anticipation": self.max_anticipation,
        }


@dataclass
class EstimandSpec:
    """
    Specification of the causal estimand.

    Supports:
    - ATE, ATT for average effects
    - elasticity for continuous treatments
    - IRF for impulse response functions (with horizon list)
    - ATE(h) for horizon-specific effects
    """

    type: Literal["ATE", "ATT", "elasticity", "IRF", "ATE(h)"] = "elasticity"
    horizon: int | list[int] = 0
    scale: Literal["level", "log", "pct"] = "level"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "horizon": self.horizon,
            "scale": self.scale,
        }


@dataclass
class DiagnosticsSpec:
    """Specification for diagnostic requirements."""

    pretrend_p_min: float = 0.10
    first_stage_f_min: float = 10.0
    placebo_p_min: float = 0.10
    density_p_min: float = 0.10

    def to_dict(self) -> dict:
        return {
            "pretrend_p_min": self.pretrend_p_min,
            "first_stage_f_min": self.first_stage_f_min,
            "placebo_p_min": self.placebo_p_min,
            "density_p_min": self.density_p_min,
        }


@dataclass
class DataQualitySpec:
    """Data quality requirements."""

    max_missing_rate: float = 0.05
    min_obs: int = 50
    min_pre_periods: int = 4

    def to_dict(self) -> dict:
        return {
            "max_missing_rate": self.max_missing_rate,
            "min_obs": self.min_obs,
            "min_pre_periods": self.min_pre_periods,
        }


@dataclass
class PlausibilitySpec:
    """Plausibility constraints for estimates."""

    expected_sign: Literal["positive", "negative", "any"] = "any"
    magnitude_range: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        d = {"expected_sign": self.expected_sign}
        if self.magnitude_range:
            d["magnitude_range"] = list(self.magnitude_range)
        return d


@dataclass
class StabilitySpec:
    """Stability requirements across subsamples."""

    regime_split_date: str | None = None
    max_coefficient_change: float = 0.5

    def to_dict(self) -> dict:
        return {
            "regime_split_date": self.regime_split_date,
            "max_coefficient_change": self.max_coefficient_change,
        }


@dataclass
class NullAcceptanceSpec:
    """Null acceptance configuration (prevents p-hacking)."""

    enabled: bool = True
    equivalence_bound: float = 0.1

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "equivalence_bound": self.equivalence_bound,
        }


@dataclass
class AcceptanceCriteria:
    """
    Credibility-based acceptance criteria.

    IMPORTANT: Does NOT include min_tstat to prevent p-hacking.
    Uses diagnostics pass/fail, plausibility, and stability instead.
    """

    diagnostics: DiagnosticsSpec = field(default_factory=DiagnosticsSpec)
    data: DataQualitySpec = field(default_factory=DataQualitySpec)
    plausibility: PlausibilitySpec = field(default_factory=PlausibilitySpec)
    stability: StabilitySpec = field(default_factory=StabilitySpec)
    null_acceptance: NullAcceptanceSpec = field(default_factory=NullAcceptanceSpec)

    # Legacy fields for backward compatibility
    min_tstat: float | None = None  # Deprecated: will emit warning
    max_pretrend_p: float | None = None
    max_missing_rate: float | None = None

    def to_dict(self) -> dict:
        return {
            "diagnostics": self.diagnostics.to_dict(),
            "data": self.data.to_dict(),
            "plausibility": self.plausibility.to_dict(),
            "stability": self.stability.to_dict(),
            "null_acceptance": self.null_acceptance.to_dict(),
        }


@dataclass
class EdgeSpec:
    """
    Specification for a DAG edge (causal link).

    Includes:
    - timing: temporal relationship (lag, contemporaneous)
    - estimand: what we're estimating
    - allowed_designs: feasible identification strategies
    - acceptance_criteria: credibility-based thresholds
    - interpretation: boundary enforcement (v2)
    - validated_evidence: immutable evidence artifacts (v2)
    - edge_status: classification for estimation (v2)
    """

    id: str
    from_node: str
    to_node: str

    # Temporal semantics
    timing: EdgeTiming = field(default_factory=EdgeTiming)

    # Estimand
    estimand: EstimandSpec = field(default_factory=EstimandSpec)

    # Design constraints
    allowed_designs: list[str] = field(default_factory=lambda: ["PANEL_FE_BACKDOOR"])
    design_priority: list[str] = field(default_factory=list)

    # Adjustment sets
    required_adjustments: list[str] = field(default_factory=list)
    forbidden_controls: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)

    # Diagnostics
    diagnostics_required: list[str] = field(default_factory=list)

    # Acceptance criteria
    acceptance_criteria: AcceptanceCriteria = field(default_factory=AcceptanceCriteria)

    # Interpretation boundary (v2)
    interpretation: EdgeInterpretation | None = None

    # Validated evidence artifact (v2)
    validated_evidence: ValidatedEvidence | None = None

    # Edge status classification (v2)
    # IMMUTABLE, ESTIMABLE_REDUCED_FORM, BLOCKED_DECOMPOSITION, NEEDS_CONNECTOR, IDENTITY
    edge_status: str = ""

    # Query mode support
    edge_type: str = ""        # causal|reaction_function|mechanical|immutable|identity
    variant_of: str = ""       # parent edge_id if this is a robustness variant

    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        d = {
            "id": self.id,
            "from": self.from_node,
            "to": self.to_node,
            "timing": self.timing.to_dict(),
            "estimand": self.estimand.to_dict(),
            "allowed_designs": self.allowed_designs,
        }

        if self.design_priority:
            d["design_priority"] = self.design_priority
        if self.required_adjustments:
            d["required_adjustments"] = self.required_adjustments
        if self.forbidden_controls:
            d["forbidden_controls"] = self.forbidden_controls
        if self.instruments:
            d["instruments"] = self.instruments
        if self.diagnostics_required:
            d["diagnostics_required"] = self.diagnostics_required

        d["acceptance_criteria"] = self.acceptance_criteria.to_dict()

        if self.interpretation:
            d["interpretation"] = self.interpretation.to_dict()
        if self.validated_evidence:
            d["validated_evidence"] = self.validated_evidence.to_dict()
        if self.edge_status:
            d["edge_status"] = self.edge_status
        if self.edge_type:
            d["edge_type"] = self.edge_type
        if self.variant_of:
            d["variant_of"] = self.variant_of
        if self.notes:
            d["notes"] = self.notes

        return d

    def get_edge_type(self) -> str:
        """Infer edge_type from edge_status if not explicitly set."""
        if self.edge_type:
            return self.edge_type
        status_map = {
            "IMMUTABLE": "immutable",
            "IDENTITY": "identity",
            "ESTIMABLE_REDUCED_FORM": "causal",
            "NEEDS_CONNECTOR": "causal",
        }
        return status_map.get(self.edge_status, "causal")

    def is_immutable(self) -> bool:
        """Check if this edge is marked as immutable (validated evidence)."""
        if self.validated_evidence:
            return self.validated_evidence.immutable
        if self.edge_status == "IMMUTABLE":
            return True
        return False

    def is_estimable(self) -> bool:
        """Check if this edge can be estimated (not blocked or identity)."""
        blocked_statuses = {"BLOCKED_DECOMPOSITION", "IDENTITY", "IMMUTABLE"}
        if self.edge_status in blocked_statuses:
            return False
        if self.is_immutable():
            return False
        return True


@dataclass
class DAGMetadata:
    """Metadata for a DAG specification."""

    name: str
    description: str = ""
    target_node: str = ""
    created: str = ""
    owner: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "target_node": self.target_node,
            "created": self.created,
            "owner": self.owner,
            "tags": self.tags,
        }


@dataclass
class AssumptionSpec:
    """An explicit causal assumption."""

    id: str
    statement: str
    applies_to_edges: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "applies_to_edges": self.applies_to_edges,
        }


@dataclass
class LatentSpec:
    """Specification for unobserved confounders."""

    id: str
    description: str = ""
    affects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "affects": self.affects,
        }


@dataclass
class DAGSpec:
    """
    Complete DAG specification.

    Contains all nodes, edges, assumptions, and latent confounders
    for a causal analysis.
    """

    metadata: DAGMetadata
    nodes: list[NodeSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)
    assumptions: list[AssumptionSpec] = field(default_factory=list)
    latents: list[LatentSpec] = field(default_factory=list)
    schema_version: int = 1

    # Computed properties
    _node_map: dict[str, NodeSpec] = field(default_factory=dict, repr=False)
    _edge_map: dict[str, EdgeSpec] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._build_indexes()

    def _build_indexes(self):
        """Build lookup indexes for nodes and edges."""
        self._node_map = {n.id: n for n in self.nodes}
        self._edge_map = {e.id: e for e in self.edges}

    def get_node(self, node_id: str) -> NodeSpec | None:
        """Get node by ID."""
        return self._node_map.get(node_id)

    def get_edge(self, edge_id: str) -> EdgeSpec | None:
        """Get edge by ID."""
        return self._edge_map.get(edge_id)

    def get_edges_from(self, node_id: str) -> list[EdgeSpec]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.from_node == node_id]

    def get_edges_to(self, node_id: str) -> list[EdgeSpec]:
        """Get all edges pointing to a node."""
        return [e for e in self.edges if e.to_node == node_id]

    def get_descendants(self, node_id: str) -> set[str]:
        """Get all descendants of a node (for forbidden controls)."""
        descendants = set()
        to_visit = [node_id]

        while to_visit:
            current = to_visit.pop()
            for edge in self.get_edges_from(current):
                if edge.to_node not in descendants:
                    descendants.add(edge.to_node)
                    to_visit.append(edge.to_node)

        return descendants

    def get_ancestors(self, node_id: str) -> set[str]:
        """Get all ancestors of a node."""
        ancestors = set()
        to_visit = [node_id]

        while to_visit:
            current = to_visit.pop()
            for edge in self.get_edges_to(current):
                if edge.from_node not in ancestors:
                    ancestors.add(edge.from_node)
                    to_visit.append(edge.from_node)

        return ancestors

    def compute_hash(self) -> str:
        """Compute SHA256 hash of the DAG spec for versioning."""
        content = yaml.dump(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        return {
            "schema_version": self.schema_version,
            "metadata": self.metadata.to_dict(),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "latents": [l.to_dict() for l in self.latents],
        }

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True)


def _parse_release_lag(data: dict | None) -> ReleaseLag | None:
    """Parse release lag from YAML data."""
    if not data:
        return None
    return ReleaseLag(
        periods=data.get("periods", 1),
        lag_unit=data.get("lag_unit", "month"),
    )


def _parse_identity(data: dict | None) -> IdentityDef | None:
    """Parse identity definition from YAML data."""
    if not data:
        return None
    return IdentityDef(
        name=data.get("name", ""),
        formula=data.get("formula", ""),
        depends_on=data.get("depends_on", []),
    )


def _parse_source_entry(data: dict) -> SourceEntry:
    """Parse a source entry from YAML data."""
    return SourceEntry(
        connector=data.get("connector", ""),
        dataset=data.get("dataset", ""),
        series=data.get("series", ""),
        notes=data.get("notes", ""),
    )


def _parse_source_config(data: dict | None) -> DataSourceConfig | None:
    """Parse data source configuration from YAML data."""
    if not data:
        return None

    preferred = [_parse_source_entry(s) for s in data.get("preferred", [])]
    fallback = [_parse_source_entry(s) for s in data.get("fallback", [])]

    return DataSourceConfig(preferred=preferred, fallback=fallback)


def _parse_aggregation(data: dict | None) -> AggregationConfig | None:
    """Parse aggregation config from YAML data."""
    if not data:
        return None
    return AggregationConfig(
        method=data.get("method", "average"),
        seasonal_adjust=data.get("seasonal_adjust", False),
        deflator=data.get("deflator"),
        deflator_timing=data.get("deflator_timing", "same"),
    )


def _parse_role_constraints(data: dict | None) -> RoleConstraints | None:
    """Parse role constraints from YAML data."""
    if not data:
        return None
    return RoleConstraints(
        never_control=data.get("never_control", False),
        never_instrument=data.get("never_instrument", False),
    )


def _parse_validated_evidence(data: dict | None) -> ValidatedEvidence | None:
    """Parse validated evidence from YAML data."""
    if not data:
        return None
    return ValidatedEvidence(
        block_id=data.get("block_id", ""),
        description=data.get("description", ""),
        estimate=data.get("estimate"),
        se=data.get("se"),
        result_hash=data.get("result_hash"),
        immutable=data.get("immutable", True),
    )


def _parse_edge_interpretation(data: dict | None) -> EdgeInterpretation | None:
    """Parse edge interpretation from YAML data."""
    if not data:
        return None

    # Handle is_not as either string or list
    is_not = data.get("is_not", [])
    if isinstance(is_not, str):
        is_not = [is_not] if is_not else []

    return EdgeInterpretation(
        is_field=data.get("is", ""),
        is_not=is_not,
        allowed_uses=data.get("allowed_uses", []),
        forbidden_uses=data.get("forbidden_uses", []),
    )


def _parse_node(data: dict) -> NodeSpec:
    """Parse a node specification from YAML data."""
    # Parse identities list (legacy format)
    identities = []
    for i in data.get("identities", []):
        if isinstance(i, dict):
            identities.append(IdentityDef(
                name=i.get("name", ""),
                formula=i.get("formula", ""),
                depends_on=i.get("depends_on", []),
            ))

    return NodeSpec(
        id=data["id"],
        name=data.get("name", data["id"]),
        description=data.get("description", ""),
        unit=data.get("unit", "level"),
        frequency=data.get("frequency", "monthly"),
        type=data.get("type", "continuous"),
        observed=data.get("observed", True),
        latent=data.get("latent", False),
        scope=data.get("scope", "unknown"),
        release_lag=_parse_release_lag(data.get("release_lag")),
        derived=data.get("derived", False),
        identity=_parse_identity(data.get("identity")),
        depends_on=data.get("depends_on", []),
        source=_parse_source_config(data.get("source")),
        transforms=data.get("transforms", []),
        aggregation=_parse_aggregation(data.get("aggregation")),
        role_constraints=_parse_role_constraints(data.get("role_constraints")),
        validated_evidence=_parse_validated_evidence(data.get("validated_evidence")),
        identities=identities,
        tags=data.get("tags", []),
    )


def _parse_edge_timing(data: dict | None) -> EdgeTiming:
    """Parse edge timing from YAML data."""
    if not data:
        return EdgeTiming()
    return EdgeTiming(
        lag=data.get("lag", 1),
        lag_unit=data.get("lag_unit", "quarter"),
        contemporaneous=data.get("contemporaneous", False),
        max_anticipation=data.get("max_anticipation", 0),
    )


def _parse_estimand(data: dict | None) -> EstimandSpec:
    """Parse estimand from YAML data."""
    if not data:
        return EstimandSpec()
    return EstimandSpec(
        type=data.get("type", "elasticity"),
        horizon=data.get("horizon", 0),
        scale=data.get("scale", "level"),
    )


def _parse_diagnostics_spec(data: dict | None) -> DiagnosticsSpec:
    """Parse diagnostics spec from YAML data."""
    if not data:
        return DiagnosticsSpec()
    return DiagnosticsSpec(
        pretrend_p_min=data.get("pretrend_p_min", 0.10),
        first_stage_f_min=data.get("first_stage_f_min", 10.0),
        placebo_p_min=data.get("placebo_p_min", 0.10),
        density_p_min=data.get("density_p_min", 0.10),
    )


def _parse_data_quality_spec(data: dict | None) -> DataQualitySpec:
    """Parse data quality spec from YAML data."""
    if not data:
        return DataQualitySpec()
    return DataQualitySpec(
        max_missing_rate=data.get("max_missing_rate", 0.05),
        min_obs=data.get("min_obs", 50),
        min_pre_periods=data.get("min_pre_periods", 4),
    )


def _parse_plausibility_spec(data: dict | None) -> PlausibilitySpec:
    """Parse plausibility spec from YAML data."""
    if not data:
        return PlausibilitySpec()

    magnitude_range = data.get("magnitude_range")
    if magnitude_range and isinstance(magnitude_range, list) and len(magnitude_range) == 2:
        magnitude_range = tuple(magnitude_range)
    else:
        magnitude_range = None

    return PlausibilitySpec(
        expected_sign=data.get("expected_sign", "any"),
        magnitude_range=magnitude_range,
    )


def _parse_stability_spec(data: dict | None) -> StabilitySpec:
    """Parse stability spec from YAML data."""
    if not data:
        return StabilitySpec()
    return StabilitySpec(
        regime_split_date=data.get("regime_split_date"),
        max_coefficient_change=data.get("max_coefficient_change", 0.5),
    )


def _parse_null_acceptance_spec(data: dict | None) -> NullAcceptanceSpec:
    """Parse null acceptance spec from YAML data."""
    if not data:
        return NullAcceptanceSpec()
    return NullAcceptanceSpec(
        enabled=data.get("enabled", True),
        equivalence_bound=data.get("equivalence_bound", 0.1),
    )


def _parse_acceptance_criteria(data: dict | None) -> AcceptanceCriteria:
    """Parse acceptance criteria from YAML data."""
    if not data:
        return AcceptanceCriteria()

    return AcceptanceCriteria(
        diagnostics=_parse_diagnostics_spec(data.get("diagnostics")),
        data=_parse_data_quality_spec(data.get("data")),
        plausibility=_parse_plausibility_spec(data.get("plausibility")),
        stability=_parse_stability_spec(data.get("stability")),
        null_acceptance=_parse_null_acceptance_spec(data.get("null_acceptance")),
        # Legacy fields
        min_tstat=data.get("min_tstat"),
        max_pretrend_p=data.get("max_pretrend_p"),
        max_missing_rate=data.get("max_missing_rate"),
    )


def _parse_edge(data: dict) -> EdgeSpec:
    """Parse an edge specification from YAML data."""
    return EdgeSpec(
        id=data["id"],
        from_node=data["from"],
        to_node=data["to"],
        timing=_parse_edge_timing(data.get("timing")),
        estimand=_parse_estimand(data.get("estimand")),
        allowed_designs=data.get("allowed_designs", ["PANEL_FE_BACKDOOR"]),
        design_priority=data.get("design_priority", []),
        required_adjustments=data.get("required_adjustments", []),
        forbidden_controls=data.get("forbidden_controls", []),
        instruments=data.get("instruments", []),
        diagnostics_required=data.get("diagnostics_required", []),
        acceptance_criteria=_parse_acceptance_criteria(data.get("acceptance_criteria")),
        interpretation=_parse_edge_interpretation(data.get("interpretation")),
        validated_evidence=_parse_validated_evidence(data.get("validated_evidence")),
        edge_status=data.get("edge_status", ""),
        edge_type=data.get("edge_type", ""),
        variant_of=data.get("variant_of", ""),
        notes=data.get("notes", ""),
    )


def _parse_assumption(data: dict) -> AssumptionSpec:
    """Parse an assumption from YAML data."""
    return AssumptionSpec(
        id=data["id"],
        statement=data.get("statement", ""),
        applies_to_edges=data.get("applies_to_edges", []),
    )


def _parse_latent(data: dict) -> LatentSpec:
    """Parse a latent confounder from YAML data."""
    return LatentSpec(
        id=data["id"],
        description=data.get("description", ""),
        affects=data.get("affects", []),
    )


def _parse_metadata(data: dict) -> DAGMetadata:
    """Parse metadata from YAML data."""
    return DAGMetadata(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
        target_node=data.get("target_node", ""),
        created=data.get("created", ""),
        owner=data.get("owner", ""),
        tags=data.get("tags", []),
    )


def parse_dag(path: Path | str) -> DAGSpec:
    """
    Parse a DAG YAML file into a DAGSpec.

    Args:
        path: Path to the DAG YAML file

    Returns:
        Parsed DAGSpec with all nodes, edges, and constraints

    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the YAML is invalid
        ValueError: If required fields are missing
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"DAG file not found: {path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty DAG file: {path}")

    # Parse components
    metadata = _parse_metadata(data.get("metadata", {}))
    nodes = [_parse_node(n) for n in data.get("nodes", [])]
    edges = [_parse_edge(e) for e in data.get("edges", [])]
    assumptions = [_parse_assumption(a) for a in data.get("assumptions", [])]
    latents = [_parse_latent(l) for l in data.get("latents", [])]

    return DAGSpec(
        metadata=metadata,
        nodes=nodes,
        edges=edges,
        assumptions=assumptions,
        latents=latents,
        schema_version=data.get("schema_version", 1),
    )
