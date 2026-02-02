"""
Frequency Alignment Layer.

Aligns mixed-frequency data (daily FX, monthly CPI, quarterly income)
for consistent estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
import numpy as np

from shared.agentic.dag.parser import NodeSpec

logger = logging.getLogger(__name__)


class FrequencyMismatch(Exception):
    """Exception raised when frequencies cannot be aligned."""

    def __init__(self, message: str, source_freq: str, target_freq: str):
        self.source_freq = source_freq
        self.target_freq = target_freq
        super().__init__(f"{message}: {source_freq} → {target_freq}")


@dataclass
class AlignmentWarning:
    """Warning about alignment issues."""

    node_id: str
    message: str
    severity: Literal["INFO", "WARNING", "ERROR"]


@dataclass
class AlignedDataset:
    """Result of frequency alignment."""

    data: pd.DataFrame
    target_frequency: str
    source_frequencies: dict[str, str]  # node_id -> original frequency
    alignment_methods: dict[str, str]  # node_id -> method used
    date_range: tuple[str, str]
    n_periods: int

    def summary(self) -> str:
        """Generate summary."""
        lines = [
            f"Aligned Dataset: {self.target_frequency}",
            f"Date range: {self.date_range[0]} to {self.date_range[1]}",
            f"Periods: {self.n_periods}",
            "",
            "Alignment by node:",
        ]
        for node_id, orig_freq in self.source_frequencies.items():
            method = self.alignment_methods.get(node_id, "direct")
            lines.append(f"  {node_id}: {orig_freq} → {self.target_frequency} ({method})")

        return "\n".join(lines)


@dataclass
class AlignmentReport:
    """Report on alignment process."""

    target_frequency: str
    nodes_aligned: list[str]
    warnings: list[AlignmentWarning] = field(default_factory=list)
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_frequency": self.target_frequency,
            "nodes_aligned": self.nodes_aligned,
            "warnings": [
                {"node_id": w.node_id, "message": w.message, "severity": w.severity}
                for w in self.warnings
            ],
            "success": self.success,
        }


# Frequency hierarchy (from finest to coarsest)
FREQUENCY_ORDER = ["daily", "monthly", "quarterly", "annual"]


def get_frequency_rank(freq: str) -> int:
    """Get rank of frequency (lower = finer)."""
    try:
        return FREQUENCY_ORDER.index(freq)
    except ValueError:
        return -1  # Unknown frequency


def can_aggregate(source_freq: str, target_freq: str) -> bool:
    """Check if source can be aggregated to target."""
    source_rank = get_frequency_rank(source_freq)
    target_rank = get_frequency_rank(target_freq)
    return source_rank >= 0 and target_rank >= 0 and source_rank <= target_rank


class FrequencyAligner:
    """
    Aligns mixed-frequency nodes for estimation.

    Handles:
    - Aggregation of higher-frequency data (daily → quarterly)
    - Publication lag handling
    - Deflation at correct timing
    - Seasonal adjustment
    """

    def __init__(self, target_frequency: str = "quarterly"):
        """
        Initialize aligner.

        Args:
            target_frequency: Target frequency for alignment
        """
        self.target_frequency = target_frequency
        self.warnings: list[AlignmentWarning] = []

    def align(
        self,
        data: dict[str, pd.DataFrame],
        nodes: list[NodeSpec],
    ) -> AlignedDataset:
        """
        Align multiple data series to target frequency.

        Args:
            data: Dictionary of node_id -> DataFrame with date index
            nodes: Node specifications with frequency info

        Returns:
            AlignedDataset with aligned data
        """
        self.warnings = []
        aligned_series = {}
        source_frequencies = {}
        alignment_methods = {}

        for node in nodes:
            if node.id not in data:
                logger.warning(f"No data for node {node.id}")
                continue

            series_data = data[node.id]
            source_freq = node.frequency

            if source_freq == "static":
                # Static data doesn't need alignment
                aligned_series[node.id] = series_data
                source_frequencies[node.id] = "static"
                alignment_methods[node.id] = "static (no alignment)"
                continue

            # Determine aggregation method
            agg_config = node.aggregation
            method = agg_config.method if agg_config else "average"

            # Align to target frequency
            try:
                aligned = self._align_series(
                    series_data,
                    source_freq,
                    self.target_frequency,
                    method,
                    node.id,
                )
                aligned_series[node.id] = aligned
                source_frequencies[node.id] = source_freq
                alignment_methods[node.id] = method

                # Apply publication lag if specified
                if node.release_lag:
                    aligned = self._apply_release_lag(
                        aligned,
                        node.release_lag.periods,
                        node.release_lag.lag_unit,
                        node.id,
                    )
                    aligned_series[node.id] = aligned
                    alignment_methods[node.id] += f" + lag({node.release_lag.periods})"

                # Apply deflation if specified
                if agg_config and agg_config.deflator:
                    if agg_config.deflator in aligned_series:
                        deflator = aligned_series[agg_config.deflator]
                        aligned = self._apply_deflation(
                            aligned,
                            deflator,
                            agg_config.deflator_timing,
                            node.id,
                        )
                        aligned_series[node.id] = aligned
                        alignment_methods[node.id] += f" + deflate({agg_config.deflator})"

                # Apply seasonal adjustment if specified
                if agg_config and agg_config.seasonal_adjust:
                    aligned = self._seasonal_adjust(aligned, node.id)
                    aligned_series[node.id] = aligned
                    alignment_methods[node.id] += " + sa"

            except FrequencyMismatch as e:
                self.warnings.append(AlignmentWarning(
                    node_id=node.id,
                    message=str(e),
                    severity="ERROR",
                ))
                continue

        # Combine into single DataFrame
        if not aligned_series:
            raise ValueError("No series could be aligned")

        combined = pd.DataFrame(aligned_series)

        # Get date range
        date_range = (
            str(combined.index.min()),
            str(combined.index.max()),
        )

        return AlignedDataset(
            data=combined,
            target_frequency=self.target_frequency,
            source_frequencies=source_frequencies,
            alignment_methods=alignment_methods,
            date_range=date_range,
            n_periods=len(combined),
        )

    def _align_series(
        self,
        data: pd.DataFrame,
        source_freq: str,
        target_freq: str,
        method: str,
        node_id: str,
    ) -> pd.DataFrame:
        """Align a single series to target frequency."""
        if source_freq == target_freq:
            return data

        if not can_aggregate(source_freq, target_freq):
            raise FrequencyMismatch(
                f"Cannot aggregate {node_id}",
                source_freq,
                target_freq,
            )

        # Determine pandas resample rule
        resample_rules = {
            "daily": "D",
            "monthly": "ME",
            "quarterly": "QE",
            "annual": "YE",
        }

        target_rule = resample_rules.get(target_freq)
        if not target_rule:
            raise FrequencyMismatch(
                f"Unknown target frequency for {node_id}",
                source_freq,
                target_freq,
            )

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception:
                self.warnings.append(AlignmentWarning(
                    node_id=node_id,
                    message="Could not convert index to datetime",
                    severity="WARNING",
                ))
                return data

        # Resample
        resampler = data.resample(target_rule)

        if method == "end_of_period":
            aligned = resampler.last()
        elif method == "sum":
            aligned = resampler.sum()
        elif method == "last":
            aligned = resampler.last()
        else:  # average (default)
            aligned = resampler.mean()

        self.warnings.append(AlignmentWarning(
            node_id=node_id,
            message=f"Aggregated from {source_freq} to {target_freq} using {method}",
            severity="INFO",
        ))

        return aligned

    def _apply_release_lag(
        self,
        data: pd.DataFrame,
        periods: int,
        lag_unit: str,
        node_id: str,
    ) -> pd.DataFrame:
        """Apply publication lag."""
        # For now, just shift the data
        # In reality, this would handle the timing of when data becomes available
        self.warnings.append(AlignmentWarning(
            node_id=node_id,
            message=f"Applied release lag of {periods} {lag_unit}(s)",
            severity="INFO",
        ))

        return data.shift(periods)

    def _apply_deflation(
        self,
        data: pd.DataFrame,
        deflator: pd.DataFrame,
        timing: str,
        node_id: str,
    ) -> pd.DataFrame:
        """Apply deflation using price index."""
        if timing == "lagged":
            deflator = deflator.shift(1)

        # Ensure same index
        common_idx = data.index.intersection(deflator.index)
        if len(common_idx) < len(data):
            self.warnings.append(AlignmentWarning(
                node_id=node_id,
                message=f"Deflator missing for {len(data) - len(common_idx)} periods",
                severity="WARNING",
            ))

        # Deflate
        deflated = data.loc[common_idx] / deflator.loc[common_idx] * 100

        return deflated

    def _seasonal_adjust(
        self,
        data: pd.DataFrame,
        node_id: str,
    ) -> pd.DataFrame:
        """Apply simple seasonal adjustment."""
        try:
            # Simple X-11 style adjustment using moving average
            # In production, would use statsmodels.tsa.seasonal
            seasonal = data.rolling(window=4, center=True).mean()
            adjusted = data / seasonal

            self.warnings.append(AlignmentWarning(
                node_id=node_id,
                message="Applied seasonal adjustment (moving average)",
                severity="INFO",
            ))

            return adjusted
        except Exception as e:
            self.warnings.append(AlignmentWarning(
                node_id=node_id,
                message=f"Seasonal adjustment failed: {e}",
                severity="WARNING",
            ))
            return data

    def validate_alignment(
        self,
        edge_treatment: str,
        edge_outcome: str,
        aligned_data: AlignedDataset,
    ) -> AlignmentReport:
        """
        Validate alignment for a specific edge.

        Args:
            edge_treatment: Treatment node ID
            edge_outcome: Outcome node ID
            aligned_data: The aligned dataset

        Returns:
            AlignmentReport with validation results
        """
        warnings = self.warnings.copy()

        # Check both variables are present
        missing = []
        if edge_treatment not in aligned_data.data.columns:
            missing.append(edge_treatment)
        if edge_outcome not in aligned_data.data.columns:
            missing.append(edge_outcome)

        if missing:
            warnings.append(AlignmentWarning(
                node_id=",".join(missing),
                message="Variables missing from aligned data",
                severity="ERROR",
            ))

            return AlignmentReport(
                target_frequency=self.target_frequency,
                nodes_aligned=list(aligned_data.source_frequencies.keys()),
                warnings=warnings,
                success=False,
            )

        # Check for sufficient overlap
        treatment_data = aligned_data.data[edge_treatment]
        outcome_data = aligned_data.data[edge_outcome]

        valid_mask = ~(treatment_data.isna() | outcome_data.isna())
        n_valid = valid_mask.sum()

        if n_valid < 30:
            warnings.append(AlignmentWarning(
                node_id=f"{edge_treatment},{edge_outcome}",
                message=f"Only {n_valid} overlapping observations",
                severity="WARNING",
            ))

        return AlignmentReport(
            target_frequency=self.target_frequency,
            nodes_aligned=list(aligned_data.source_frequencies.keys()),
            warnings=warnings,
            success=True,
        )


def align_for_edge(
    data: dict[str, pd.DataFrame],
    nodes: list[NodeSpec],
    target_frequency: str = "quarterly",
) -> AlignedDataset:
    """
    Convenience function to align data for edge estimation.

    Args:
        data: Dictionary of node_id -> DataFrame
        nodes: Node specifications
        target_frequency: Target frequency

    Returns:
        AlignedDataset
    """
    aligner = FrequencyAligner(target_frequency)
    return aligner.align(data, nodes)
