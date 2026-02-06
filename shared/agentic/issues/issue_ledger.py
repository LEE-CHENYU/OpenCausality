"""
Issue Ledger: Append-Only Issue Log.

Provides persistent issue tracking across agent loop runs.
Each run writes issues to a per-run JSONL file. The cross-run
reducer (see cross_run_reducer.py) builds global state from
all per-run files.

Core principle: Issues are the primary mechanism for communicating
problems. CRITICAL issues block CONFIRMATION mode. Issues that
require human input pause the loop.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


@dataclass
class Issue:
    """A single issue entry in the ledger."""

    run_id: str
    timestamp: str
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    rule_id: str
    scope: Literal["edge", "node", "dag", "run", "cross_run"]
    message: str
    edge_id: str | None = None
    node_id: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    auto_fixable: bool = False
    suggested_fix: dict[str, Any] | None = None
    requires_human: bool = False
    status: Literal["OPEN", "CLOSED", "WONT_FIX"] = "OPEN"
    closed_by: str | None = None
    closed_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Issue:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def issue_key(self) -> str:
        """Unique key for deduplication: rule_id:edge_id or rule_id:node_id."""
        target = self.edge_id or self.node_id or "global"
        return f"{self.rule_id}:{target}"

    @property
    def is_critical(self) -> bool:
        return self.severity == "CRITICAL"

    @property
    def is_open(self) -> bool:
        return self.status == "OPEN"

    def close(self, reason: str, closed_by: str = "auto") -> None:
        self.status = "CLOSED"
        self.closed_reason = reason
        self.closed_by = closed_by


class IssueLedger:
    """
    Append-only issue log with per-run JSONL persistence.

    Usage:
        ledger = IssueLedger(run_id="abc123", output_dir=Path("outputs/agentic/issues"))
        ledger.add(Issue(severity="CRITICAL", rule_id="UNIT_MISSING", ...))
        ledger.flush()  # Writes to <run_id>.jsonl
    """

    def __init__(
        self,
        run_id: str | None = None,
        output_dir: Path | None = None,
    ):
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.output_dir = output_dir or Path("outputs/agentic/issues")
        self.issues: list[Issue] = []
        self._flushed_count = 0

    @property
    def jsonl_path(self) -> Path:
        return self.output_dir / f"{self.run_id}.jsonl"

    def add(self, issue: Issue) -> Issue:
        """Add an issue to the ledger. Returns the issue for chaining."""
        issue.run_id = self.run_id
        if not issue.timestamp:
            issue.timestamp = datetime.now(timezone.utc).isoformat()
        self.issues.append(issue)
        logger.info(
            f"Issue [{issue.severity}] {issue.rule_id}: {issue.message}"
            + (f" (edge: {issue.edge_id})" if issue.edge_id else "")
        )
        return issue

    def add_from_rule(
        self,
        rule_id: str,
        severity: str,
        message: str,
        scope: str = "edge",
        edge_id: str | None = None,
        node_id: str | None = None,
        evidence: dict | None = None,
        auto_fixable: bool = False,
        suggested_fix: dict | None = None,
        requires_human: bool = False,
    ) -> Issue:
        """Convenience method to add an issue from rule parameters."""
        issue = Issue(
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity=severity,
            rule_id=rule_id,
            scope=scope,
            message=message,
            edge_id=edge_id,
            node_id=node_id,
            evidence=evidence or {},
            auto_fixable=auto_fixable,
            suggested_fix=suggested_fix,
            requires_human=requires_human,
        )
        return self.add(issue)

    def close_issue(self, issue_key: str, reason: str, closed_by: str = "auto") -> bool:
        """Close an open issue by its key. Returns True if found and closed."""
        for issue in reversed(self.issues):
            if issue.issue_key == issue_key and issue.is_open:
                issue.close(reason, closed_by)
                return True
        return False

    def get_open_issues(self) -> list[Issue]:
        """Get all open issues."""
        return [i for i in self.issues if i.is_open]

    def get_critical_open(self) -> list[Issue]:
        """Get all open CRITICAL issues."""
        return [i for i in self.issues if i.is_open and i.is_critical]

    def get_issues_requiring_human(self) -> list[Issue]:
        """Get open issues that require human decision."""
        return [i for i in self.issues if i.is_open and i.requires_human]

    def get_auto_fixable(self) -> list[Issue]:
        """Get open issues that can be auto-fixed."""
        return [i for i in self.issues if i.is_open and i.auto_fixable]

    def get_issues_for_edge(self, edge_id: str) -> list[Issue]:
        """Get all issues for a specific edge."""
        return [i for i in self.issues if i.edge_id == edge_id]

    def has_critical_open(self) -> bool:
        """Check if any CRITICAL issues are open."""
        return any(i.is_open and i.is_critical for i in self.issues)

    def has_human_required(self) -> bool:
        """Check if any issues need human input."""
        return any(i.is_open and i.requires_human for i in self.issues)

    def flush(self) -> Path:
        """Write unflushed issues to the per-run JSONL file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        new_issues = self.issues[self._flushed_count:]
        if new_issues:
            with open(self.jsonl_path, "a") as f:
                for issue in new_issues:
                    f.write(issue.to_jsonl() + "\n")
            self._flushed_count = len(self.issues)
        return self.jsonl_path

    def load_from_file(self, path: Path) -> list[Issue]:
        """Load issues from a JSONL file."""
        issues = []
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        issues.append(Issue.from_dict(data))
        return issues

    def summary(self) -> str:
        """Generate a summary of current issues."""
        open_issues = self.get_open_issues()
        by_severity: dict[str, int] = {}
        for issue in open_issues:
            by_severity[issue.severity] = by_severity.get(issue.severity, 0) + 1

        lines = [
            f"Issue Ledger: {len(self.issues)} total, {len(open_issues)} open",
        ]
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = by_severity.get(sev, 0)
            if count > 0:
                lines.append(f"  {sev}: {count}")

        human_count = len(self.get_issues_requiring_human())
        if human_count > 0:
            lines.append(f"  Requires human: {human_count}")

        return "\n".join(lines)
