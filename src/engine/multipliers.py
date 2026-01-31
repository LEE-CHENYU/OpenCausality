"""
Multiplier storage and management.

Stores estimated coefficients from shift-share and local projection models
for use in scenario simulation.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Multiplier:
    """A single multiplier estimate."""

    name: str
    coefficient: float
    std_error: float
    exposure: str
    shock: str
    horizon: int = 0  # 0 for contemporaneous, >0 for LP horizons
    model_type: str = "shift_share"
    specification: str = "baseline"
    estimated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MultiplierSet:
    """Collection of multipliers from a model estimation."""

    name: str
    multipliers: list[Multiplier]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, exposure: str, shock: str, horizon: int = 0) -> Multiplier | None:
        """Get a specific multiplier."""
        for m in self.multipliers:
            if m.exposure == exposure and m.shock == shock and m.horizon == horizon:
                return m
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "multipliers": [asdict(m) for m in self.multipliers],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiplierSet":
        """Create from dictionary."""
        multipliers = [Multiplier(**m) for m in data["multipliers"]]
        return cls(
            name=data["name"],
            multipliers=multipliers,
            metadata=data.get("metadata", {}),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        records = []
        for m in self.multipliers:
            records.append(asdict(m))
        return pd.DataFrame(records)


class MultiplierStore:
    """Storage and retrieval for multiplier estimates."""

    def __init__(self, storage_path: Path | None = None):
        settings = get_settings()
        self.storage_path = storage_path or (settings.project_root / settings.output_dir / "multipliers.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, MultiplierSet] = {}
        self._load()

    def _load(self) -> None:
        """Load multipliers from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                for name, mset_data in data.items():
                    self._cache[name] = MultiplierSet.from_dict(mset_data)
                logger.info(f"Loaded {len(self._cache)} multiplier sets")
            except Exception as e:
                logger.warning(f"Failed to load multipliers: {e}")

    def save(self) -> None:
        """Save multipliers to storage."""
        data = {name: mset.to_dict() for name, mset in self._cache.items()}
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self._cache)} multiplier sets")

    def add(self, multiplier_set: MultiplierSet) -> None:
        """Add or update a multiplier set."""
        self._cache[multiplier_set.name] = multiplier_set
        self.save()

    def get(self, name: str) -> MultiplierSet | None:
        """Get a multiplier set by name."""
        return self._cache.get(name)

    def list_sets(self) -> list[str]:
        """List all multiplier set names."""
        return list(self._cache.keys())

    def get_multiplier(
        self,
        set_name: str,
        exposure: str,
        shock: str,
        horizon: int = 0,
    ) -> Multiplier | None:
        """Get a specific multiplier from a set."""
        mset = self.get(set_name)
        if mset is None:
            return None
        return mset.get(exposure, shock, horizon)

    def from_shift_share_results(
        self,
        results: dict[str, Any],
        set_name: str = "shift_share",
    ) -> MultiplierSet:
        """
        Create multiplier set from shift-share regression results.

        Args:
            results: Dictionary of RegressionResult objects
            set_name: Name for the multiplier set

        Returns:
            MultiplierSet
        """
        multipliers = []

        for spec_name, result in results.items():
            # Parse interaction term names to extract exposure and shock
            for param_name in result.params.index:
                if "_x_" in param_name:
                    parts = param_name.split("_x_")
                    if len(parts) == 2:
                        exposure = parts[0]
                        shock = parts[1] + "_shock"

                        multipliers.append(
                            Multiplier(
                                name=param_name,
                                coefficient=result.params[param_name],
                                std_error=result.std_errors[param_name],
                                exposure=exposure,
                                shock=shock,
                                horizon=0,
                                model_type="shift_share",
                                specification=spec_name,
                            )
                        )

        mset = MultiplierSet(
            name=set_name,
            multipliers=multipliers,
            metadata={
                "source": "shift_share",
                "n_specs": len(results),
            },
        )

        self.add(mset)
        return mset

    def from_local_projections(
        self,
        irf_results: dict[str, Any],
        set_name: str = "local_projections",
    ) -> MultiplierSet:
        """
        Create multiplier set from local projection IRF results.

        Args:
            irf_results: Dictionary of IRFResult objects
            set_name: Name for the multiplier set

        Returns:
            MultiplierSet
        """
        multipliers = []

        for var_name, irf in irf_results.items():
            # Parse variable name
            if "_x_" in var_name:
                parts = var_name.split("_x_")
                exposure = parts[0]
                shock = parts[1] + "_shock"
            else:
                exposure = "unknown"
                shock = var_name

            # Create multiplier for each horizon
            for i, h in enumerate(irf.horizons):
                multipliers.append(
                    Multiplier(
                        name=f"{var_name}_h{h}",
                        coefficient=irf.coefficients[i],
                        std_error=irf.std_errors[i],
                        exposure=exposure,
                        shock=shock,
                        horizon=h,
                        model_type="local_projection",
                        specification="baseline",
                    )
                )

        mset = MultiplierSet(
            name=set_name,
            multipliers=multipliers,
            metadata={
                "source": "local_projections",
                "max_horizon": max(m.horizon for m in multipliers) if multipliers else 0,
            },
        )

        self.add(mset)
        return mset


def get_multiplier_store() -> MultiplierStore:
    """Get or create the global multiplier store."""
    return MultiplierStore()
