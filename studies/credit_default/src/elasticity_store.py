"""
Elasticity storage for credit default study.

Stores estimated income-default elasticities by segment for use in
scenario simulation and stress testing.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Elasticity:
    """A single elasticity estimate."""

    segment: str  # e.g., "formal_workers", "near_retirees"
    coefficient: float  # Income-default elasticity
    std_error: float
    pvalue: float
    conf_int: tuple[float, float]

    # Metadata
    design: str  # "mw_diff_discs" or "pension_rdd"
    outcome: str  # "dpd30", etc.
    n_obs: int
    first_stage_f: float | None = None  # For IV designs

    # Timing
    estimated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Interpretation
    interpretation: str = ""
    external_validity_caveats: list[str] = field(default_factory=list)


@dataclass
class ElasticitySet:
    """Collection of elasticities from a study."""

    name: str
    elasticities: list[Elasticity]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, segment: str) -> Elasticity | None:
        """Get elasticity for a segment."""
        for e in self.elasticities:
            if e.segment == segment:
                return e
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "elasticities": [asdict(e) for e in self.elasticities],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ElasticitySet":
        """Create from dictionary."""
        elasticities = [Elasticity(**e) for e in data["elasticities"]]
        return cls(
            name=data["name"],
            elasticities=elasticities,
            metadata=data.get("metadata", {}),
        )


class ElasticityStore:
    """
    Storage for income-default elasticity estimates.

    Stores estimates by segment for use in scenario simulation.
    """

    def __init__(self, storage_path: Path | None = None):
        """
        Initialize store.

        Args:
            storage_path: Path to JSON storage file
        """
        self.storage_path = storage_path or Path(
            "studies/credit_default/outputs/elasticities.json"
        )
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, ElasticitySet] = {}
        self._load()

    def _load(self) -> None:
        """Load elasticities from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                for name, eset_data in data.items():
                    self._cache[name] = ElasticitySet.from_dict(eset_data)
                logger.info(f"Loaded {len(self._cache)} elasticity sets")
            except Exception as e:
                logger.warning(f"Failed to load elasticities: {e}")

    def save(self) -> None:
        """Save elasticities to storage."""
        data = {name: eset.to_dict() for name, eset in self._cache.items()}
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self._cache)} elasticity sets")

    def add(self, elasticity_set: ElasticitySet) -> None:
        """Add or update an elasticity set."""
        self._cache[elasticity_set.name] = elasticity_set
        self.save()

    def get(self, name: str) -> ElasticitySet | None:
        """Get an elasticity set by name."""
        return self._cache.get(name)

    def get_elasticity(
        self,
        set_name: str,
        segment: str,
    ) -> Elasticity | None:
        """Get a specific elasticity."""
        eset = self.get(set_name)
        if eset is None:
            return None
        return eset.get(segment)

    def store_from_mw_result(
        self,
        result: Any,  # DiffInDiscsResult
        segment: str = "formal_workers",
        set_name: str = "credit_default",
    ) -> None:
        """
        Store elasticity from MW diff-in-discs result.

        Args:
            result: DiffInDiscsResult
            segment: Segment name
            set_name: Elasticity set name
        """
        # Convert MW effect to income elasticity
        # Assuming MW increase is ~21.4%, the elasticity is:
        # elasticity = beta / (delta_income_pct)
        mw_pct_increase = 0.214  # (85000 - 70000) / 70000
        income_elasticity = result.coefficient / mw_pct_increase

        elasticity = Elasticity(
            segment=segment,
            coefficient=income_elasticity,
            std_error=result.std_error / mw_pct_increase,
            pvalue=result.pvalue,
            conf_int=(
                result.conf_int[0] / mw_pct_increase,
                result.conf_int[1] / mw_pct_increase,
            ),
            design="mw_diff_discs",
            outcome=result.outcome,
            n_obs=result.n_obs,
            interpretation=(
                f"1% decrease in income increases {result.outcome} probability by "
                f"{abs(income_elasticity)*100:.3f} percentage points"
            ),
            external_validity_caveats=[
                "LATE for formal workers whose income was raised by MW policy",
                "May not generalize to informal workers",
                "Estimated for workers earning near old minimum wage",
            ],
        )

        # Get or create set
        eset = self.get(set_name)
        if eset is None:
            eset = ElasticitySet(name=set_name, elasticities=[])

        # Replace or add
        existing = [e for e in eset.elasticities if e.segment != segment]
        existing.append(elasticity)
        eset.elasticities = existing

        self.add(eset)

    def store_from_pension_result(
        self,
        result: Any,  # FuzzyRDDResult
        segment: str = "near_retirees",
        set_name: str = "credit_default",
        avg_pension_income_pct: float = 0.30,  # Pension as % of pre-retirement income
    ) -> None:
        """
        Store elasticity from pension RDD result.

        Args:
            result: FuzzyRDDResult
            segment: Segment name
            set_name: Elasticity set name
            avg_pension_income_pct: Average pension as fraction of income
        """
        # Convert pension effect to income elasticity
        # Assuming pension adds ~30% to income on average
        income_elasticity = result.coefficient / avg_pension_income_pct

        elasticity = Elasticity(
            segment=segment,
            coefficient=income_elasticity,
            std_error=result.std_error / avg_pension_income_pct,
            pvalue=result.pvalue,
            conf_int=(
                result.conf_int[0] / avg_pension_income_pct,
                result.conf_int[1] / avg_pension_income_pct,
            ),
            design="pension_rdd",
            outcome="dpd30",  # Assuming
            n_obs=result.n_obs,
            first_stage_f=result.first_stage_f,
            interpretation=(
                f"1% income increase (from pension) changes default probability by "
                f"{income_elasticity*100:.3f} percentage points"
            ),
            external_validity_caveats=[
                "LATE at pension eligibility threshold",
                "Applies to near-retirees receiving first pension",
                "May not generalize to younger borrowers",
                "Income source is pension (recurring, stable)",
            ],
        )

        # Get or create set
        eset = self.get(set_name)
        if eset is None:
            eset = ElasticitySet(name=set_name, elasticities=[])

        # Replace or add
        existing = [e for e in eset.elasticities if e.segment != segment]
        existing.append(elasticity)
        eset.elasticities = existing

        self.add(eset)

    def summary(self) -> str:
        """Generate summary of stored elasticities."""
        lines = []
        lines.append("=" * 60)
        lines.append("STORED INCOME-DEFAULT ELASTICITIES")
        lines.append("=" * 60)

        for name, eset in self._cache.items():
            lines.append(f"\nSet: {name}")
            lines.append("-" * 40)

            for e in eset.elasticities:
                sig = "*" if e.pvalue < 0.05 else ""
                lines.append(f"  {e.segment}:")
                lines.append(f"    Elasticity: {e.coefficient:.4f}{sig}")
                lines.append(f"    Design: {e.design}")
                lines.append(f"    N obs: {e.n_obs:,}")
                if e.first_stage_f:
                    lines.append(f"    First-stage F: {e.first_stage_f:.1f}")

        return "\n".join(lines)


def get_elasticity_store() -> ElasticityStore:
    """Get or create the elasticity store."""
    return ElasticityStore()
