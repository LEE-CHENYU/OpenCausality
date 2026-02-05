"""
Data Provenance Tracking.

Tracks the full lineage of data used in estimation for:
- Reproducibility
- Audit trails
- Version control
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SourceProvenance:
    """Provenance for a single data source."""

    connector: str
    dataset: str
    series: str | None = None
    retrieval_time: datetime | None = None
    file_path: str | None = None
    file_checksum: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    date_range: tuple[str, str] | None = None
    cache_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "connector": self.connector,
            "dataset": self.dataset,
            "series": self.series,
            "retrieval_time": self.retrieval_time.isoformat() if self.retrieval_time else None,
            "file_path": self.file_path,
            "file_checksum": self.file_checksum,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "date_range": list(self.date_range) if self.date_range else None,
            "cache_key": self.cache_key,
        }


@dataclass
class DataProvenance:
    """
    Complete data provenance for an estimation.

    Tracks sources for treatment, outcome, and all controls.
    """

    treatment_source: SourceProvenance | None = None
    outcome_source: SourceProvenance | None = None
    control_sources: dict[str, SourceProvenance] = field(default_factory=dict)
    instrument_sources: dict[str, SourceProvenance] = field(default_factory=dict)

    # Combined dataset info
    combined_row_count: int | None = None
    combined_date_range: tuple[str, str] | None = None
    missing_rate: float | None = None

    # Panel-specific provenance
    panel_dimensions: dict[str, Any] | None = None  # {"n_units": 4, "n_periods": 13, "balance": "unbalanced"}
    entity_boundary_note: str | None = None  # Documents entity definition consistency
    kpi_definitions: dict[str, str] | None = None  # {"npl": "90+ DPD per IFRS 9", ...}

    def compute_hash(self) -> str:
        """Compute hash of provenance for versioning."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "treatment_source": self.treatment_source.to_dict() if self.treatment_source else None,
            "outcome_source": self.outcome_source.to_dict() if self.outcome_source else None,
            "control_sources": {k: v.to_dict() for k, v in self.control_sources.items()},
            "instrument_sources": {k: v.to_dict() for k, v in self.instrument_sources.items()},
            "combined_row_count": self.combined_row_count,
            "combined_date_range": list(self.combined_date_range) if self.combined_date_range else None,
            "missing_rate": self.missing_rate,
            "panel_dimensions": self.panel_dimensions,
            "entity_boundary_note": self.entity_boundary_note,
            "kpi_definitions": self.kpi_definitions,
        }


@dataclass
class SpecDetails:
    """Details of the model specification."""

    design: str
    controls: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    fixed_effects: list[str] = field(default_factory=list)
    se_method: str = "cluster"
    sample_filter: str | None = None
    horizon: int | list[int] | None = None
    bandwidth: float | None = None  # For RDD

    def compute_hash(self) -> str:
        """Compute hash of spec for versioning."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "design": self.design,
            "controls": self.controls,
            "instruments": self.instruments,
            "fixed_effects": self.fixed_effects,
            "se_method": self.se_method,
            "sample_filter": self.sample_filter,
            "horizon": self.horizon,
            "bandwidth": self.bandwidth,
        }


@dataclass
class AuditRecord:
    """Complete audit record for reproducibility."""

    edge_id: str
    dag_version_hash: str
    data_hash: str
    spec_hash: str
    result_hash: str
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0
    mode: str = "EXPLORATION"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "edge_id": self.edge_id,
            "dag_version_hash": self.dag_version_hash,
            "data_hash": self.data_hash,
            "spec_hash": self.spec_hash,
            "result_hash": self.result_hash,
            "timestamp": self.timestamp.isoformat(),
            "iteration": self.iteration,
            "mode": self.mode,
        }
