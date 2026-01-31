"""
Data lineage tracking for Kazakhstan household welfare study.

Tracks which data sources returned real vs. placeholder data,
logs warnings when synthetic fallbacks are used, and includes
data lineage in model output.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import json

logger = logging.getLogger(__name__)


class DataSourceStatus(Enum):
    """Status of a data source fetch."""
    REAL = "real"  # Successfully fetched real data
    CACHED = "cached"  # Loaded from cache (originally real)
    SYNTHETIC = "synthetic"  # Generated placeholder data
    PARTIAL = "partial"  # Real data with significant gaps
    FAILED = "failed"  # Fetch failed, no fallback used
    UNKNOWN = "unknown"  # Status not determined


class DataQualityLevel(Enum):
    """Quality assessment of data."""
    EXCELLENT = "excellent"  # High-quality, verified
    GOOD = "good"  # Acceptable with minor issues
    FAIR = "fair"  # Usable but with caveats
    POOR = "poor"  # Significant issues
    UNUSABLE = "unusable"  # Should not be used for analysis


@dataclass
class DataSourceRecord:
    """Record of a single data source fetch."""

    source_name: str
    status: DataSourceStatus
    quality: DataQualityLevel
    timestamp: datetime
    rows: int = 0
    columns: int = 0
    missing_pct: float = 0.0
    date_range: tuple[str, str] | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source_name": self.source_name,
            "status": self.status.value,
            "quality": self.quality.value,
            "timestamp": self.timestamp.isoformat(),
            "rows": self.rows,
            "columns": self.columns,
            "missing_pct": self.missing_pct,
            "date_range": self.date_range,
            "notes": self.notes,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class DataLineageTracker:
    """
    Tracks data lineage throughout the pipeline.

    Use this to:
    1. Record which data sources returned real vs synthetic data
    2. Log warnings when fallbacks are used
    3. Include lineage information in model outputs
    4. Generate lineage reports
    """

    def __init__(self):
        self.records: dict[str, DataSourceRecord] = {}
        self._warnings: list[str] = []
        self._critical_issues: list[str] = []

    def record_source(
        self,
        source_name: str,
        status: DataSourceStatus,
        quality: DataQualityLevel,
        rows: int = 0,
        columns: int = 0,
        missing_pct: float = 0.0,
        date_range: tuple[str, str] | None = None,
        notes: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DataSourceRecord:
        """
        Record a data source fetch.

        Args:
            source_name: Identifier for the data source
            status: Whether data is real, synthetic, etc.
            quality: Quality assessment
            rows: Number of rows in the data
            columns: Number of columns
            missing_pct: Percentage of missing values
            date_range: (start_date, end_date) tuple
            notes: Additional notes
            metadata: Additional metadata

        Returns:
            The created DataSourceRecord
        """
        warnings = []

        # Automatically generate warnings based on status
        if status == DataSourceStatus.SYNTHETIC:
            warning = f"SYNTHETIC DATA: {source_name} is using placeholder data, not real observations"
            warnings.append(warning)
            self._critical_issues.append(warning)
            logger.warning(warning)

        if status == DataSourceStatus.PARTIAL:
            warning = f"PARTIAL DATA: {source_name} has significant gaps ({missing_pct:.1f}% missing)"
            warnings.append(warning)
            self._warnings.append(warning)
            logger.warning(warning)

        if quality == DataQualityLevel.POOR:
            warning = f"POOR QUALITY: {source_name} data quality is poor"
            warnings.append(warning)
            self._warnings.append(warning)
            logger.warning(warning)

        if quality == DataQualityLevel.UNUSABLE:
            warning = f"UNUSABLE: {source_name} data should not be used for analysis"
            warnings.append(warning)
            self._critical_issues.append(warning)
            logger.error(warning)

        record = DataSourceRecord(
            source_name=source_name,
            status=status,
            quality=quality,
            timestamp=datetime.now(),
            rows=rows,
            columns=columns,
            missing_pct=missing_pct,
            date_range=date_range,
            notes=notes or [],
            warnings=warnings,
            metadata=metadata or {},
        )

        self.records[source_name] = record
        return record

    def record_hardcoded_values(
        self,
        variable_name: str,
        values: dict[str, float],
        source_citation: str | None = None,
    ) -> None:
        """
        Record that hardcoded values are being used instead of real data.

        Args:
            variable_name: Name of the variable (e.g., "E_oil_r")
            values: The hardcoded values being used
            source_citation: Citation for where values came from (if any)
        """
        warning = f"HARDCODED VALUES: {variable_name} uses hardcoded values, not measured data"
        if source_citation:
            warning += f" (source: {source_citation})"
        else:
            warning += " (NO SOURCE CITED)"
            self._critical_issues.append(f"UNDOCUMENTED: {variable_name} has no source citation")

        self._warnings.append(warning)
        logger.warning(warning)

        self.record_source(
            source_name=variable_name,
            status=DataSourceStatus.SYNTHETIC,
            quality=DataQualityLevel.POOR,
            notes=[
                "Values are hardcoded, not from empirical data",
                f"Source citation: {source_citation or 'NONE'}",
                f"Values: {values}",
            ],
            metadata={"hardcoded_values": values},
        )

    def has_synthetic_data(self) -> bool:
        """Check if any data source is synthetic."""
        return any(
            r.status == DataSourceStatus.SYNTHETIC
            for r in self.records.values()
        )

    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return len(self._critical_issues) > 0

    def get_warnings(self) -> list[str]:
        """Get all warnings."""
        return self._warnings.copy()

    def get_critical_issues(self) -> list[str]:
        """Get all critical issues."""
        return self._critical_issues.copy()

    def generate_report(self) -> str:
        """Generate a human-readable lineage report."""
        lines = []
        lines.append("=" * 70)
        lines.append("DATA LINEAGE REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")

        # Critical issues section
        if self._critical_issues:
            lines.append("CRITICAL ISSUES:")
            lines.append("-" * 50)
            for issue in self._critical_issues:
                lines.append(f"  [X] {issue}")
            lines.append("")

        # Summary table
        lines.append("DATA SOURCES:")
        lines.append("-" * 70)
        lines.append(f"{'Source':<25} {'Status':<12} {'Quality':<12} {'Rows':>8}")
        lines.append("-" * 70)

        for name, record in sorted(self.records.items()):
            status_indicator = {
                DataSourceStatus.REAL: "[OK]",
                DataSourceStatus.CACHED: "[OK]",
                DataSourceStatus.SYNTHETIC: "[!!]",
                DataSourceStatus.PARTIAL: "[??]",
                DataSourceStatus.FAILED: "[XX]",
                DataSourceStatus.UNKNOWN: "[??]",
            }.get(record.status, "[??]")

            lines.append(
                f"{status_indicator} {name:<21} "
                f"{record.status.value:<12} "
                f"{record.quality.value:<12} "
                f"{record.rows:>8}"
            )

        lines.append("-" * 70)
        lines.append("")

        # Warnings section
        if self._warnings:
            lines.append("WARNINGS:")
            lines.append("-" * 50)
            for warning in self._warnings:
                lines.append(f"  ! {warning}")
            lines.append("")

        # Detailed records
        lines.append("DETAILED RECORDS:")
        lines.append("-" * 70)

        for name, record in sorted(self.records.items()):
            lines.append(f"\n{name}:")
            lines.append(f"  Status: {record.status.value}")
            lines.append(f"  Quality: {record.quality.value}")
            lines.append(f"  Rows: {record.rows}, Columns: {record.columns}")
            lines.append(f"  Missing: {record.missing_pct:.1f}%")
            if record.date_range:
                lines.append(f"  Date range: {record.date_range[0]} to {record.date_range[1]}")
            if record.notes:
                lines.append("  Notes:")
                for note in record.notes:
                    lines.append(f"    - {note}")
            if record.warnings:
                lines.append("  Warnings:")
                for warning in record.warnings:
                    lines.append(f"    ! {warning}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert lineage to dictionary for serialization."""
        return {
            "generated": datetime.now().isoformat(),
            "has_synthetic_data": self.has_synthetic_data(),
            "has_critical_issues": self.has_critical_issues(),
            "critical_issues": self._critical_issues,
            "warnings": self._warnings,
            "sources": {
                name: record.to_dict()
                for name, record in self.records.items()
            },
        }

    def save(self, filepath: str | Path) -> None:
        """Save lineage report to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved data lineage to {filepath}")

    def save_report(self, filepath: str | Path) -> None:
        """Save human-readable report to text file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            f.write(self.generate_report())

        logger.info(f"Saved lineage report to {filepath}")


# Global tracker instance
_global_tracker: DataLineageTracker | None = None


def get_tracker() -> DataLineageTracker:
    """Get or create the global lineage tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = DataLineageTracker()
    return _global_tracker


def reset_tracker() -> None:
    """Reset the global lineage tracker."""
    global _global_tracker
    _global_tracker = None


def record_source(*args, **kwargs) -> DataSourceRecord:
    """Convenience function to record a source on the global tracker."""
    return get_tracker().record_source(*args, **kwargs)


def record_hardcoded(*args, **kwargs) -> None:
    """Convenience function to record hardcoded values on the global tracker."""
    get_tracker().record_hardcoded_values(*args, **kwargs)


def print_lineage_report() -> None:
    """Print the global lineage report."""
    print(get_tracker().generate_report())


def check_data_quality() -> bool:
    """
    Check if data quality is sufficient for analysis.

    Returns:
        True if data is usable, False if critical issues exist
    """
    tracker = get_tracker()

    if tracker.has_critical_issues():
        logger.error("CRITICAL DATA QUALITY ISSUES DETECTED")
        for issue in tracker.get_critical_issues():
            logger.error(f"  {issue}")
        return False

    if tracker.has_synthetic_data():
        logger.warning("WARNING: Analysis includes synthetic/placeholder data")
        return True  # Usable but with caveats

    return True
