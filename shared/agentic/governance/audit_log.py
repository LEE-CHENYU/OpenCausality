"""
Audit Log.

Maintains a complete ledger of all specification changes for:
- Reproducibility
- P-hacking prevention
- Compliance/governance
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from shared.agentic.governance.whitelist import (
    RefinementWhitelist,
    Refinement,
    GovernanceViolation,
)

logger = logging.getLogger(__name__)


@dataclass
class Hashes:
    """Hash values for reproducibility."""

    dag: str
    data: str
    spec: str
    result: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "dag": self.dag,
            "data": self.data,
            "spec": self.spec,
            "result": self.result,
        }


@dataclass
class ChangeRecord:
    """Record of a single change."""

    type: str  # "INITIAL", "REFINEMENT", "CONFIRMATION"
    edge_id: str
    field: str | None = None
    old_value: Any = None
    new_value: Any = None
    reason: str = ""
    approved_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "edge_id": self.edge_id,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "approved_by": self.approved_by,
        }


@dataclass
class ResultDelta:
    """Change in results after a refinement."""

    credibility_before: float
    credibility_after: float
    improvement: float

    def to_dict(self) -> dict[str, float]:
        return {
            "credibility_before": self.credibility_before,
            "credibility_after": self.credibility_after,
            "improvement": self.improvement,
        }


@dataclass
class LedgerEntry:
    """A single entry in the audit ledger."""

    timestamp: datetime
    run_id: str
    iteration: int
    hashes: Hashes
    change: ChangeRecord
    mode: str  # "EXPLORATION" or "CONFIRMATION"
    holdout_period: tuple[str, str] | None = None
    result_delta: ResultDelta | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "iteration": self.iteration,
            "hashes": self.hashes.to_dict(),
            "change": self.change.to_dict(),
            "mode": self.mode,
            "holdout_period": list(self.holdout_period) if self.holdout_period else None,
            "result_delta": self.result_delta.to_dict() if self.result_delta else None,
        }

    def to_jsonl(self) -> str:
        """Convert to JSON Lines format."""
        return json.dumps(self.to_dict(), default=str)


class AuditLog:
    """
    Full ledger of all specification changes with governance enforcement.

    Maintains immutable record of:
    - All spec changes with timestamps
    - Hashes for reproducibility
    - Whitelist approvals
    - Result deltas
    """

    def __init__(
        self,
        whitelist: RefinementWhitelist | None = None,
        log_path: Path | None = None,
    ):
        """
        Initialize audit log.

        Args:
            whitelist: Refinement whitelist for governance
            log_path: Path to persist ledger (JSONL format)
        """
        self.whitelist = whitelist or RefinementWhitelist()
        self.log_path = log_path

        self.entries: list[LedgerEntry] = []
        self.current_run_id: str = ""
        self.iteration: int = 0
        self.mode: str = "EXPLORATION"

        # Current hashes
        self.dag_hash: str = ""
        self.data_hash: str = ""

    def set_run_context(
        self,
        run_id: str,
        dag_hash: str,
        data_hash: str,
        mode: str = "EXPLORATION",
    ) -> None:
        """
        Set the current run context.

        Args:
            run_id: Unique run identifier
            dag_hash: Hash of the DAG specification
            data_hash: Hash of the combined data
            mode: "EXPLORATION" or "CONFIRMATION"
        """
        self.current_run_id = run_id
        self.dag_hash = dag_hash
        self.data_hash = data_hash
        self.mode = mode
        self.iteration = 0

    def log_initial(self, edge_id: str, spec_hash: str) -> LedgerEntry:
        """
        Log initial specification for an edge.

        Args:
            edge_id: The edge ID
            spec_hash: Hash of the initial spec

        Returns:
            The logged entry
        """
        entry = LedgerEntry(
            timestamp=datetime.now(),
            run_id=self.current_run_id,
            iteration=self.iteration,
            hashes=Hashes(
                dag=self.dag_hash,
                data=self.data_hash,
                spec=spec_hash,
            ),
            change=ChangeRecord(
                type="INITIAL",
                edge_id=edge_id,
                reason="Initial specification",
            ),
            mode=self.mode,
        )

        self._add_entry(entry)
        return entry

    def log_refinement(
        self,
        refinement: Refinement,
        new_spec_hash: str,
        result_delta: ResultDelta | None = None,
    ) -> LedgerEntry:
        """
        Log a refinement with governance check.

        Args:
            refinement: The refinement to log
            new_spec_hash: Hash of the new spec
            result_delta: Change in results (optional)

        Returns:
            The logged entry

        Raises:
            GovernanceViolation: If refinement not in whitelist
        """
        # Governance check
        if not self.whitelist.allows(refinement):
            raise GovernanceViolation(refinement)

        rule_id = self.whitelist.get_rule_id(refinement)
        refinement.approved_by = rule_id
        refinement.new_spec_hash = new_spec_hash

        entry = LedgerEntry(
            timestamp=datetime.now(),
            run_id=self.current_run_id,
            iteration=self.iteration,
            hashes=Hashes(
                dag=self.dag_hash,
                data=self.data_hash,
                spec=new_spec_hash,
            ),
            change=ChangeRecord(
                type="REFINEMENT",
                edge_id=refinement.edge_id,
                field=refinement.field,
                old_value=refinement.old_value,
                new_value=refinement.new_value,
                reason=refinement.reason,
                approved_by=rule_id,
            ),
            mode=self.mode,
            result_delta=result_delta,
        )

        self._add_entry(entry)
        return entry

    def log_refinements(self, refinements: list[Refinement]) -> list[LedgerEntry]:
        """
        Log multiple refinements.

        Args:
            refinements: List of refinements to log

        Returns:
            List of logged entries

        Raises:
            GovernanceViolation: If any refinement not in whitelist
        """
        entries = []
        for r in refinements:
            entry = self.log_refinement(r, r.new_spec_hash or "")
            entries.append(entry)
        return entries

    def log_confirmation(
        self,
        edge_id: str,
        spec_hash: str,
        holdout_period: tuple[str, str],
        result_hash: str,
    ) -> LedgerEntry:
        """
        Log a confirmation run (frozen spec, holdout data).

        Args:
            edge_id: The edge ID
            spec_hash: Hash of the frozen spec
            holdout_period: Start and end of holdout period
            result_hash: Hash of results

        Returns:
            The logged entry
        """
        entry = LedgerEntry(
            timestamp=datetime.now(),
            run_id=self.current_run_id,
            iteration=self.iteration,
            hashes=Hashes(
                dag=self.dag_hash,
                data=self.data_hash,
                spec=spec_hash,
                result=result_hash,
            ),
            change=ChangeRecord(
                type="CONFIRMATION",
                edge_id=edge_id,
                reason="Confirmation run with frozen spec",
            ),
            mode="CONFIRMATION",
            holdout_period=holdout_period,
        )

        self._add_entry(entry)
        return entry

    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self.iteration += 1

    def _add_entry(self, entry: LedgerEntry) -> None:
        """Add entry to ledger and persist if path set."""
        self.entries.append(entry)

        if self.log_path:
            self._persist_entry(entry)

    def _persist_entry(self, entry: LedgerEntry) -> None:
        """Persist entry to JSONL file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_path, "a") as f:
            f.write(entry.to_jsonl() + "\n")

    def get_full_ledger(self) -> list[LedgerEntry]:
        """Get all entries."""
        return self.entries.copy()

    def get_entries_for_edge(self, edge_id: str) -> list[LedgerEntry]:
        """Get all entries for a specific edge."""
        return [e for e in self.entries if e.change.edge_id == edge_id]

    def get_entries_for_run(self, run_id: str) -> list[LedgerEntry]:
        """Get all entries for a specific run."""
        return [e for e in self.entries if e.run_id == run_id]

    def get_refinements_for_edge(self, edge_id: str) -> list[ChangeRecord]:
        """Get all refinements for a specific edge."""
        return [
            e.change for e in self.entries
            if e.change.edge_id == edge_id and e.change.type == "REFINEMENT"
        ]

    def count_refinements(self) -> int:
        """Count total refinements."""
        return sum(1 for e in self.entries if e.change.type == "REFINEMENT")

    def get_total_improvement(self, edge_id: str) -> float:
        """Get total credibility improvement for an edge."""
        entries = self.get_entries_for_edge(edge_id)
        return sum(
            e.result_delta.improvement
            for e in entries
            if e.result_delta is not None
        )

    def summary(self) -> str:
        """Generate summary of audit log."""
        lines = [
            "=" * 60,
            "AUDIT LOG SUMMARY",
            "=" * 60,
            f"Run ID: {self.current_run_id}",
            f"Mode: {self.mode}",
            f"Iteration: {self.iteration}",
            f"Total entries: {len(self.entries)}",
            f"Refinements: {self.count_refinements()}",
            "",
        ]

        # Group by edge
        edges = set(e.change.edge_id for e in self.entries)
        for edge_id in sorted(edges):
            edge_entries = self.get_entries_for_edge(edge_id)
            refinements = [e for e in edge_entries if e.change.type == "REFINEMENT"]
            lines.append(f"Edge {edge_id}:")
            lines.append(f"  Initial + {len(refinements)} refinements")

            for r in refinements[:3]:  # Show first 3
                lines.append(
                    f"    {r.change.field}: {r.change.old_value} → {r.change.new_value}"
                )
            if len(refinements) > 3:
                lines.append(f"    ... and {len(refinements) - 3} more")

        lines.append("=" * 60)
        return "\n".join(lines)

    def iter_entries(self) -> Iterator[LedgerEntry]:
        """Iterate over all entries."""
        yield from self.entries

    @classmethod
    def load(cls, path: Path, whitelist: RefinementWhitelist | None = None) -> 'AuditLog':
        """
        Load audit log from JSONL file.

        Args:
            path: Path to JSONL file
            whitelist: Optional whitelist to use

        Returns:
            Loaded AuditLog
        """
        log = cls(whitelist=whitelist, log_path=path)

        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        entry = LedgerEntry(
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            run_id=data["run_id"],
                            iteration=data["iteration"],
                            hashes=Hashes(**data["hashes"]),
                            change=ChangeRecord(**data["change"]),
                            mode=data["mode"],
                            holdout_period=tuple(data["holdout_period"]) if data.get("holdout_period") else None,
                            result_delta=ResultDelta(**data["result_delta"]) if data.get("result_delta") else None,
                        )
                        log.entries.append(entry)

        return log
