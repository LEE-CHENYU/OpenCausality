"""
Cross-Run Reducer: Build global state.json from per-run JSONL files.

The reducer maintains a persistent view of all issues across runs,
enabling cross-run issue types like MULTIPLE_TESTING_DRIFT and
CROSS_EVIDENCE_CONFLICT.

Baseline rule: Compare to last CONFIRMATION run with matching dag_hash.
If no prior CONFIRMATION exists, compare to the most recent EXPLORATION run.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.agentic.issues.issue_ledger import Issue, IssueLedger

logger = logging.getLogger(__name__)


class CrossRunState:
    """
    Global issue state built from all per-run JSONL files.

    state.json structure:
    {
        "last_updated": "2026-02-05T14:00:00Z",
        "baseline_run_id": "abc123",
        "baseline_mode": "CONFIRMATION",
        "baseline_dag_hash": "sha256:...",
        "issues": {
            "UNIT_MISSING:shock_to_cor_kspi": {
                "status": "CLOSED",
                "opened_run": "run001",
                "closed_run": "run003",
                "resolution": "auto_fix:add_edge_units"
            },
            ...
        }
    }
    """

    def __init__(self, state_path: Path | None = None):
        self.state_path = state_path or Path("outputs/agentic/issues/state.json")
        self.last_updated: str = ""
        self.baseline_run_id: str = ""
        self.baseline_mode: str = ""
        self.baseline_dag_hash: str = ""
        self.issues: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        """Load state from disk."""
        if self.state_path.exists():
            with open(self.state_path) as f:
                data = json.load(f)
            self.last_updated = data.get("last_updated", "")
            self.baseline_run_id = data.get("baseline_run_id", "")
            self.baseline_mode = data.get("baseline_mode", "")
            self.baseline_dag_hash = data.get("baseline_dag_hash", "")
            self.issues = data.get("issues", {})

    def save(self) -> None:
        """Save state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "baseline_run_id": self.baseline_run_id,
            "baseline_mode": self.baseline_mode,
            "baseline_dag_hash": self.baseline_dag_hash,
            "issues": self.issues,
        }
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def apply_run(self, run_issues: list[Issue], run_id: str) -> None:
        """Apply a run's issues to the global state."""
        for issue in run_issues:
            key = issue.issue_key

            if issue.is_open:
                if key not in self.issues:
                    self.issues[key] = {
                        "status": "OPEN",
                        "opened_run": run_id,
                        "severity": issue.severity,
                        "rule_id": issue.rule_id,
                        "message": issue.message,
                    }
            else:
                # Issue was closed in this run
                if key in self.issues:
                    self.issues[key]["status"] = "CLOSED"
                    self.issues[key]["closed_run"] = run_id
                    self.issues[key]["resolution"] = issue.closed_reason or "unknown"

    def set_baseline(self, run_id: str, mode: str, dag_hash: str) -> None:
        """Set the baseline run for cross-run comparisons."""
        self.baseline_run_id = run_id
        self.baseline_mode = mode
        self.baseline_dag_hash = dag_hash

    def get_open_issues(self) -> dict[str, dict[str, Any]]:
        """Get all globally open issues."""
        return {k: v for k, v in self.issues.items() if v.get("status") == "OPEN"}

    def get_open_by_severity(self, severity: str) -> dict[str, dict[str, Any]]:
        """Get open issues of a specific severity."""
        return {
            k: v for k, v in self.issues.items()
            if v.get("status") == "OPEN" and v.get("severity") == severity
        }


class CrossRunReducer:
    """
    Reducer that builds global state from per-run JSONL files.

    Usage:
        reducer = CrossRunReducer(issues_dir=Path("outputs/agentic/issues"))
        reducer.reduce()  # Reads all .jsonl, builds state.json
    """

    def __init__(self, issues_dir: Path | None = None):
        self.issues_dir = issues_dir or Path("outputs/agentic/issues")
        self.state = CrossRunState(self.issues_dir / "state.json")

    def reduce(self) -> CrossRunState:
        """Build global state from all per-run JSONL files."""
        self.state.load()

        # Find all JSONL files
        jsonl_files = sorted(self.issues_dir.glob("*.jsonl"))

        ledger = IssueLedger()
        for jsonl_path in jsonl_files:
            run_id = jsonl_path.stem
            run_issues = ledger.load_from_file(jsonl_path)
            self.state.apply_run(run_issues, run_id)

        self.state.save()
        return self.state

    def reduce_incremental(self, run_id: str, run_issues: list[Issue]) -> CrossRunState:
        """Apply a single run's issues to the global state (fast path)."""
        self.state.load()
        self.state.apply_run(run_issues, run_id)
        self.state.save()
        return self.state

    def detect_cross_run_issues(
        self,
        current_run_id: str,
        current_edge_cards: dict[str, Any],
        ledger: IssueLedger,
    ) -> list[Issue]:
        """Detect issues that span multiple runs."""
        detected = []
        self.state.load()

        # CROSS_EVIDENCE_CONFLICT: check if current estimates conflict with baseline
        # This would compare signs across runs for the same edge
        # (Implemented as a placeholder - actual comparison requires baseline cards)

        return detected
