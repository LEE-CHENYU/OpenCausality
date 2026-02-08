"""
Confirmation Freeze Validator.

Prevents specification drift between EXPLORATION and CONFIRMATION modes
by snapshotting the DAG, design assignments, and data state at the end
of exploration and validating them before confirmation begins.

This is the core p-hacking prevention mechanism: it ensures that
CONFIRMATION mode runs with exactly the spec that was agreed upon
during EXPLORATION, not a silently-modified variant.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default path to the freeze manifest schema/template
DEFAULT_SCHEMA_PATH = Path("config/agentic/confirmation_freeze.yaml")


@dataclass
class FreezeManifest:
    """Snapshot of the state at the end of EXPLORATION."""

    dag_hash: str
    design_assignments_hash: str
    data_hash: str
    frozen_at: str  # ISO timestamp
    frozen_by: str = ""
    exploration_run_id: str = ""
    exploration_window: tuple[str, str] = ("", "")
    confirmation_window: tuple[str, str] = ("", "")
    overlap_policy: str = "FORBIDDEN"
    edgecard_fields_required: list[str] = field(
        default_factory=lambda: ["spec_hash", "data_hash", "sample_window"],
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dag_hash": self.dag_hash,
            "design_assignments_hash": self.design_assignments_hash,
            "data_hash": self.data_hash,
            "frozen_at": self.frozen_at,
            "frozen_by": self.frozen_by,
            "exploration_run_id": self.exploration_run_id,
            "exploration_window": list(self.exploration_window),
            "confirmation_window": list(self.confirmation_window),
            "overlap_policy": self.overlap_policy,
            "edgecard_fields_required": self.edgecard_fields_required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FreezeManifest:
        ew = data.get("exploration_window", ["", ""])
        cw = data.get("confirmation_window", ["", ""])
        return cls(
            dag_hash=data["dag_hash"],
            design_assignments_hash=data["design_assignments_hash"],
            data_hash=data["data_hash"],
            frozen_at=data["frozen_at"],
            frozen_by=data.get("frozen_by", ""),
            exploration_run_id=data.get("exploration_run_id", ""),
            exploration_window=(ew[0], ew[1]) if len(ew) >= 2 else ("", ""),
            confirmation_window=(cw[0], cw[1]) if len(cw) >= 2 else ("", ""),
            overlap_policy=data.get("overlap_policy", "FORBIDDEN"),
            edgecard_fields_required=data.get(
                "edgecard_fields_required",
                ["spec_hash", "data_hash", "sample_window"],
            ),
        )

    def save(self, path: Path) -> None:
        """Persist manifest to YAML."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                {"freeze_manifest": self.to_dict()},
                f, sort_keys=False, allow_unicode=True, default_flow_style=False,
            )
        logger.info(f"Freeze manifest saved to {path}")

    @classmethod
    def load(cls, path: Path) -> FreezeManifest:
        """Load manifest from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data["freeze_manifest"])


class FreezeValidator:
    """Validates that the current state matches a frozen manifest.

    Used to gate entry into CONFIRMATION mode, ensuring no specification
    drift has occurred since the manifest was created.
    """

    def __init__(self, schema_path: Path = DEFAULT_SCHEMA_PATH):
        self.schema_path = schema_path

    def create_manifest(self, agent_loop: Any) -> FreezeManifest:
        """Snapshot current state into a freeze manifest at end of EXPLORATION.

        Args:
            agent_loop: The AgentLoop instance after exploration completes.

        Returns:
            A populated FreezeManifest.
        """
        # Compute design assignments hash from current edge cards
        design_hash = self._compute_design_assignments_hash(agent_loop)

        holdout = agent_loop.config.holdout_period or ("", "")

        manifest = FreezeManifest(
            dag_hash=agent_loop._dag_hash,
            design_assignments_hash=design_hash,
            data_hash=agent_loop._data_hash,
            frozen_at=datetime.now().isoformat(),
            frozen_by="agent_loop",
            exploration_run_id=agent_loop.run_id,
            confirmation_window=holdout,
            overlap_policy="FORBIDDEN",
        )

        return manifest

    def validate_before_confirmation(
        self,
        agent_loop: Any,
        manifest: FreezeManifest,
    ) -> list[str]:
        """Gate CONFIRMATION mode entry. Returns list of violations (empty = OK).

        Checks:
        1. DAG hash matches current DAG
        2. Design assignments haven't changed
        3. Data hash matches (no new data slipped in)
        4. Sample windows don't overlap (for TS edges with FORBIDDEN policy)
        5. Frozen timestamp is in the past
        """
        violations: list[str] = []

        # 1. DAG hash
        current_dag_hash = agent_loop._dag_hash
        if current_dag_hash != manifest.dag_hash:
            violations.append(
                f"DAG hash mismatch: frozen={manifest.dag_hash[:12]}... "
                f"current={current_dag_hash[:12]}... "
                f"The DAG was modified after exploration freeze."
            )

        # 2. Design assignments
        current_design_hash = self._compute_design_assignments_hash(agent_loop)
        if current_design_hash != manifest.design_assignments_hash:
            violations.append(
                "Design assignments hash mismatch: "
                f"frozen={manifest.design_assignments_hash[:12]}... "
                f"current={current_design_hash[:12]}... "
                "Edge designs were modified after exploration freeze."
            )

        # 3. Data hash
        current_data_hash = agent_loop._data_hash
        if current_data_hash and manifest.data_hash:
            if current_data_hash != manifest.data_hash:
                violations.append(
                    f"Data hash mismatch: frozen={manifest.data_hash[:12]}... "
                    f"current={current_data_hash[:12]}... "
                    f"Data catalog changed after exploration freeze."
                )

        # 4. Sample window overlap (FORBIDDEN policy for TS edges)
        if manifest.overlap_policy == "FORBIDDEN":
            overlap = self._check_window_overlap(
                manifest.exploration_window,
                manifest.confirmation_window,
            )
            if overlap:
                violations.append(
                    f"Sample window overlap detected: exploration ends "
                    f"{manifest.exploration_window[1]}, confirmation starts "
                    f"{manifest.confirmation_window[0]}. "
                    f"Overlap policy is FORBIDDEN."
                )

        # 5. Frozen timestamp
        if manifest.frozen_at:
            try:
                frozen_time = datetime.fromisoformat(manifest.frozen_at)
                if frozen_time > datetime.now():
                    violations.append(
                        f"Freeze timestamp is in the future: {manifest.frozen_at}. "
                        f"Clock skew or tampered manifest."
                    )
            except ValueError:
                violations.append(
                    f"Invalid freeze timestamp: {manifest.frozen_at}"
                )

        return violations

    def _compute_design_assignments_hash(self, agent_loop: Any) -> str:
        """Compute hash of current design assignments from edge cards."""
        edge_cards = agent_loop.artifact_store.get_all_edge_cards()
        assignments = {}
        for card in sorted(edge_cards, key=lambda c: c.edge_id):
            design = card.spec_details.design if card.spec_details else ""
            spec_hash = card.spec_hash or ""
            assignments[card.edge_id] = {
                "design": design,
                "spec_hash": spec_hash,
            }
        content = json.dumps(assignments, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _check_window_overlap(
        self,
        window_a: tuple[str, str],
        window_b: tuple[str, str],
    ) -> bool:
        """Check if two date windows overlap. Returns True if overlapping."""
        if not all(window_a) or not all(window_b):
            return False
        try:
            a_end = datetime.fromisoformat(window_a[1])
            b_start = datetime.fromisoformat(window_b[0])
            return a_end >= b_start
        except ValueError:
            return False
