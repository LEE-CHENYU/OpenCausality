"""
Artifact Store.

Manages storage and retrieval of EdgeCards and other artifacts
produced during DAG estimation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import yaml

from shared.agentic.output.edge_card import EdgeCard

logger = logging.getLogger(__name__)


@dataclass
class ArtifactMetadata:
    """Metadata for a stored artifact."""

    artifact_id: str
    artifact_type: str
    edge_id: str | None
    created_at: datetime
    spec_hash: str
    file_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "edge_id": self.edge_id,
            "created_at": self.created_at.isoformat(),
            "spec_hash": self.spec_hash,
            "file_path": str(self.file_path),
        }


class ArtifactStore:
    """
    Stores and retrieves estimation artifacts.

    Provides:
    - EdgeCard storage (YAML/JSON)
    - Version tracking
    - Query by edge ID
    - Caching
    """

    def __init__(self, base_path: Path | str):
        """
        Initialize artifact store.

        Args:
            base_path: Base directory for artifact storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.edge_cards_path = self.base_path / "edge_cards"
        self.data_cards_path = self.base_path / "data_cards"
        self.model_specs_path = self.base_path / "model_specs"

        for path in [self.edge_cards_path, self.data_cards_path, self.model_specs_path]:
            path.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._edge_cards: dict[str, EdgeCard] = {}
        self._metadata: dict[str, ArtifactMetadata] = {}

    def save_edge_card(
        self,
        card: EdgeCard,
        format: str = "yaml",
    ) -> Path:
        """
        Save an EdgeCard to storage.

        Args:
            card: The EdgeCard to save
            format: Output format ("yaml" or "json")

        Returns:
            Path to saved file
        """
        filename = f"{card.edge_id}.{format}"
        filepath = self.edge_cards_path / filename

        if format == "yaml":
            content = card.to_yaml()
        else:
            content = card.to_json()

        with open(filepath, "w") as f:
            f.write(content)

        # Update cache
        self._edge_cards[card.edge_id] = card

        # Store metadata
        self._metadata[card.edge_id] = ArtifactMetadata(
            artifact_id=f"edge_card:{card.edge_id}",
            artifact_type="edge_card",
            edge_id=card.edge_id,
            created_at=card.created_at,
            spec_hash=card.spec_hash,
            file_path=filepath,
        )

        logger.info(f"Saved EdgeCard for {card.edge_id} to {filepath}")
        return filepath

    def load_edge_card(self, edge_id: str) -> EdgeCard | None:
        """
        Load an EdgeCard from storage.

        Args:
            edge_id: The edge ID

        Returns:
            EdgeCard if found, None otherwise
        """
        # Check cache first
        if edge_id in self._edge_cards:
            return self._edge_cards[edge_id]

        # Try to load from file
        for ext in ["yaml", "json"]:
            filepath = self.edge_cards_path / f"{edge_id}.{ext}"
            if filepath.exists():
                try:
                    with open(filepath) as f:
                        if ext == "yaml":
                            data = yaml.safe_load(f)
                        else:
                            data = json.load(f)

                    card = self._dict_to_edge_card(data)
                    self._edge_cards[edge_id] = card
                    return card

                except Exception as e:
                    logger.error(f"Failed to load EdgeCard {edge_id}: {e}")
                    return None

        return None

    def _dict_to_edge_card(self, data: dict) -> EdgeCard:
        """Convert dictionary to EdgeCard."""
        from shared.agentic.output.edge_card import (
            Estimates,
            DiagnosticResult,
            Interpretation,
            FailureFlags,
            CounterfactualApplicability,
        )
        from shared.agentic.output.provenance import (
            DataProvenance,
            SpecDetails,
        )

        # Parse estimates
        estimates = None
        if data.get("estimates"):
            est = data["estimates"]
            estimates = Estimates(
                point=est["point"],
                se=est["se"],
                ci_95=tuple(est["ci_95"]),
                pvalue=est.get("pvalue"),
            )

        # Parse diagnostics
        diagnostics = {}
        for name, diag_data in data.get("diagnostics", {}).items():
            diagnostics[name] = DiagnosticResult(
                name=name,
                passed=diag_data["passed"],
                value=diag_data.get("value"),
                threshold=diag_data.get("threshold"),
                pvalue=diag_data.get("pvalue"),
                message=diag_data.get("message", ""),
            )

        # Parse interpretation
        interp_data = data.get("interpretation", {})
        interpretation = Interpretation(
            estimand=interp_data.get("estimand", ""),
            is_not=interp_data.get("is_not", ""),
            channels=interp_data.get("channels", []),
            population=interp_data.get("population", ""),
            conditions=interp_data.get("conditions", ""),
        )

        # Parse failure flags
        flags_data = data.get("failure_flags", {})
        failure_flags = FailureFlags(
            weak_identification=flags_data.get("weak_identification", False),
            potential_bad_control=flags_data.get("potential_bad_control", False),
            mechanical_identity_risk=flags_data.get("mechanical_identity_risk", False),
            regime_break_detected=flags_data.get("regime_break_detected", False),
            small_sample=flags_data.get("small_sample", False),
            high_missing_rate=flags_data.get("high_missing_rate", False),
        )

        # Parse counterfactual
        cf_data = data.get("counterfactual", {})
        counterfactual = CounterfactualApplicability(
            supports_shock_path=cf_data.get("supports_shock_path", True),
            supports_policy_intervention=cf_data.get("supports_policy_intervention", False),
            intervention_note=cf_data.get("intervention_note", ""),
            external_validity=cf_data.get("external_validity", ""),
        )

        # Parse spec details
        spec_data = data.get("spec_details", {})
        spec_details = SpecDetails(
            design=spec_data.get("design", ""),
            controls=spec_data.get("controls", []),
            instruments=spec_data.get("instruments", []),
            fixed_effects=spec_data.get("fixed_effects", []),
            se_method=spec_data.get("se_method", "cluster"),
        )

        return EdgeCard(
            edge_id=data["edge_id"],
            dag_version_hash=data.get("dag_version_hash", ""),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            spec_hash=data.get("spec_hash", ""),
            spec_details=spec_details,
            estimates=estimates,
            diagnostics=diagnostics,
            interpretation=interpretation,
            failure_flags=failure_flags,
            counterfactual=counterfactual,
            credibility_rating=data.get("credibility_rating", "D"),
            credibility_score=data.get("credibility_score", 0.0),
            is_precisely_null=data.get("is_precisely_null", False),
            null_equivalence_bound=data.get("null_equivalence_bound"),
        )

    def get_all_edge_cards(self) -> list[EdgeCard]:
        """Get all stored EdgeCards."""
        cards = []

        for filepath in self.edge_cards_path.glob("*.yaml"):
            edge_id = filepath.stem
            card = self.load_edge_card(edge_id)
            if card:
                cards.append(card)

        for filepath in self.edge_cards_path.glob("*.json"):
            edge_id = filepath.stem
            if edge_id not in [c.edge_id for c in cards]:
                card = self.load_edge_card(edge_id)
                if card:
                    cards.append(card)

        return cards

    def get_edge_cards_by_rating(self, rating: str) -> list[EdgeCard]:
        """Get EdgeCards with a specific rating."""
        return [c for c in self.get_all_edge_cards() if c.credibility_rating == rating]

    def has_edge_card(self, edge_id: str) -> bool:
        """Check if an EdgeCard exists."""
        if edge_id in self._edge_cards:
            return True

        for ext in ["yaml", "json"]:
            if (self.edge_cards_path / f"{edge_id}.{ext}").exists():
                return True

        return False

    def delete_edge_card(self, edge_id: str) -> bool:
        """Delete an EdgeCard."""
        deleted = False

        # Remove from cache
        if edge_id in self._edge_cards:
            del self._edge_cards[edge_id]

        if edge_id in self._metadata:
            del self._metadata[edge_id]

        # Remove files
        for ext in ["yaml", "json"]:
            filepath = self.edge_cards_path / f"{edge_id}.{ext}"
            if filepath.exists():
                filepath.unlink()
                deleted = True

        return deleted

    def clear(self) -> None:
        """Clear all artifacts."""
        self._edge_cards.clear()
        self._metadata.clear()

        for filepath in self.edge_cards_path.glob("*"):
            filepath.unlink()

    def summary(self) -> str:
        """Generate summary of stored artifacts."""
        cards = self.get_all_edge_cards()

        lines = [
            "=" * 60,
            "ARTIFACT STORE SUMMARY",
            "=" * 60,
            f"Base path: {self.base_path}",
            f"EdgeCards: {len(cards)}",
            "",
        ]

        if cards:
            lines.append("By rating:")
            for rating in ["A", "B", "C", "D"]:
                count = len([c for c in cards if c.credibility_rating == rating])
                if count > 0:
                    lines.append(f"  {rating}: {count}")

            lines.append("")
            lines.append("EdgeCards:")
            for card in cards[:10]:  # Show first 10
                lines.append(
                    f"  {card.edge_id}: {card.credibility_rating} "
                    f"({card.credibility_score:.2f})"
                )
            if len(cards) > 10:
                lines.append(f"  ... and {len(cards) - 10} more")

        lines.append("=" * 60)
        return "\n".join(lines)
