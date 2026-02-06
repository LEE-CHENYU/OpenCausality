"""
LinkageTask Definition.

Defines the task structure for edge estimation with:
- Rich status states for dependency tracking
- Data and artifact dependencies
- Credibility scoring support
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    """Status states for a LinkageTask."""

    NEW = "NEW"                       # Just created
    WAITING_DATA = "WAITING_DATA"     # Blocked on data fetch
    WAITING_ARTIFACT = "WAITING_ARTIFACT"  # Blocked on upstream task
    READY = "READY"                   # All deps satisfied, can run
    RUNNING = "RUNNING"               # Currently executing
    DONE = "DONE"                     # Completed successfully
    DONE_SUGGESTIVE = "DONE_SUGGESTIVE"  # Estimated but not credible
    BLOCKED_ID = "BLOCKED_ID"         # Not identifiable with available data
    FAILED = "FAILED"                 # Execution failed


class TaskPriority(Enum):
    """Priority levels for task scheduling."""

    CRITICAL = 1     # On direct path to target
    HIGH = 2         # Provides instrument/artifact for critical task
    MEDIUM = 3       # Creates reusable artifact
    LOW = 4          # Supporting task
    BACKGROUND = 5   # "Download everything" tasks


@dataclass
class DataDependency:
    """A data dependency for a task."""

    node_id: str
    connector: str
    dataset: str
    series: str
    satisfied: bool = False
    data_ref: str | None = None  # Path or cache key when satisfied


@dataclass
class ArtifactDependency:
    """An artifact dependency from another task."""

    source_edge_id: str
    artifact_type: str  # e.g., "imported_inflation", "first_stage_residuals"
    satisfied: bool = False
    artifact_ref: str | None = None  # Path when satisfied


@dataclass
class LinkageTask:
    """
    A task for estimating a causal edge.

    Tracks:
    - Edge identification
    - Data and artifact dependencies
    - Execution status
    - Results and credibility
    """

    # Edge identification
    edge_id: str
    treatment: str
    outcome: str
    estimand: str

    # Design selection
    candidate_designs: list[str] = field(default_factory=list)
    selected_design: str | None = None

    # Dependencies
    required_data_assets: set[str] = field(default_factory=set)
    required_derived_features: set[str] = field(default_factory=set)
    required_upstream_artifacts: set[str] = field(default_factory=set)

    # Detailed dependencies for tracking
    data_dependencies: list[DataDependency] = field(default_factory=list)
    artifact_dependencies: list[ArtifactDependency] = field(default_factory=list)

    # Status
    status: TaskStatus = TaskStatus.NEW
    priority: TaskPriority = TaskPriority.MEDIUM
    blocked_reason: str | None = None

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Results
    results_ref: str | None = None  # Path to EdgeCard
    critique_ref: str | None = None  # Path to critique
    credibility_score: float | None = None
    credibility_rating: str | None = None  # A/B/C/D

    # Re-estimation tracking
    _prev_credibility_score: float | None = None
    reset_reason: str | None = None

    # Lock for idempotency
    _locked_by: str | None = None

    def is_blocked(self) -> bool:
        """Check if task is blocked on dependencies."""
        return self.status in {
            TaskStatus.WAITING_DATA,
            TaskStatus.WAITING_ARTIFACT,
            TaskStatus.BLOCKED_ID,
        }

    def is_done(self) -> bool:
        """Check if task has completed (successfully or not)."""
        return self.status in {
            TaskStatus.DONE,
            TaskStatus.DONE_SUGGESTIVE,
            TaskStatus.BLOCKED_ID,
            TaskStatus.FAILED,
        }

    def is_successful(self) -> bool:
        """Check if task completed with credible results."""
        return self.status == TaskStatus.DONE

    def can_run(self) -> bool:
        """Check if task can be executed."""
        return self.status == TaskStatus.READY

    def check_data_dependencies(self) -> bool:
        """Check if all data dependencies are satisfied."""
        return all(d.satisfied for d in self.data_dependencies)

    def check_artifact_dependencies(self) -> bool:
        """Check if all artifact dependencies are satisfied."""
        return all(d.satisfied for d in self.artifact_dependencies)

    def update_status(self) -> TaskStatus:
        """
        Update status based on dependency state.

        Returns the new status.
        """
        if self.status in {TaskStatus.DONE, TaskStatus.DONE_SUGGESTIVE,
                          TaskStatus.BLOCKED_ID, TaskStatus.FAILED}:
            # Terminal states don't change
            return self.status

        if self.status == TaskStatus.RUNNING:
            # Running tasks stay running
            return self.status

        # Check dependencies
        if not self.check_data_dependencies():
            self.status = TaskStatus.WAITING_DATA
        elif not self.check_artifact_dependencies():
            self.status = TaskStatus.WAITING_ARTIFACT
        else:
            self.status = TaskStatus.READY

        return self.status

    def mark_data_satisfied(self, node_id: str, data_ref: str) -> None:
        """Mark a data dependency as satisfied."""
        for dep in self.data_dependencies:
            if dep.node_id == node_id:
                dep.satisfied = True
                dep.data_ref = data_ref
        self.update_status()

    def mark_artifact_satisfied(self, edge_id: str, artifact_ref: str) -> None:
        """Mark an artifact dependency as satisfied."""
        for dep in self.artifact_dependencies:
            if dep.source_edge_id == edge_id:
                dep.satisfied = True
                dep.artifact_ref = artifact_ref
        self.update_status()

    def start(self) -> None:
        """Mark task as started."""
        if self.status != TaskStatus.READY:
            raise ValueError(f"Cannot start task in status {self.status}")
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

    def complete(
        self,
        results_ref: str,
        credibility_score: float,
        credibility_rating: str,
    ) -> None:
        """Mark task as completed successfully."""
        self.status = TaskStatus.DONE if credibility_score >= 0.6 else TaskStatus.DONE_SUGGESTIVE
        self.completed_at = datetime.now()
        self.results_ref = results_ref
        self.credibility_score = credibility_score
        self.credibility_rating = credibility_rating

    def block(self, reason: str, block_type: TaskStatus = TaskStatus.BLOCKED_ID) -> None:
        """Mark task as blocked."""
        self.status = block_type
        self.blocked_reason = reason
        self.completed_at = datetime.now()

    def fail(self, reason: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.blocked_reason = reason
        self.completed_at = datetime.now()

    def acquire_lock(self, locker_id: str) -> bool:
        """
        Acquire exclusive lock for this task.

        Returns True if lock acquired, False if already locked.
        """
        if self._locked_by is None:
            self._locked_by = locker_id
            return True
        return self._locked_by == locker_id

    def release_lock(self, locker_id: str) -> bool:
        """
        Release lock on this task.

        Returns True if released, False if not owner.
        """
        if self._locked_by == locker_id:
            self._locked_by = None
            return True
        return False

    def reset(self, reason: str = "") -> None:
        """Reset a terminal task for re-estimation."""
        self._prev_credibility_score = self.credibility_score
        self.status = TaskStatus.READY
        self.started_at = None
        self.completed_at = None
        self.results_ref = None
        self.critique_ref = None
        self.credibility_score = None
        self.credibility_rating = None
        self.blocked_reason = None
        self.reset_reason = reason

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "edge_id": self.edge_id,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "estimand": self.estimand,
            "candidate_designs": self.candidate_designs,
            "selected_design": self.selected_design,
            "status": self.status.value,
            "priority": self.priority.value,
            "blocked_reason": self.blocked_reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results_ref": self.results_ref,
            "credibility_score": self.credibility_score,
            "credibility_rating": self.credibility_rating,
            "reset_reason": self.reset_reason,
            "prev_credibility_score": self._prev_credibility_score,
        }


def create_task_from_edge(
    edge_id: str,
    from_node: str,
    to_node: str,
    estimand_type: str,
    allowed_designs: list[str],
    required_nodes: list[str],
    upstream_edges: list[str] | None = None,
) -> LinkageTask:
    """
    Create a LinkageTask from edge specification.

    Args:
        edge_id: Edge identifier
        from_node: Treatment node ID
        to_node: Outcome node ID
        estimand_type: Type of estimand (ATE, elasticity, etc.)
        allowed_designs: List of allowed designs
        required_nodes: Nodes needed for estimation (treatment, outcome, controls)
        upstream_edges: Edges that produce required artifacts

    Returns:
        Configured LinkageTask
    """
    task = LinkageTask(
        edge_id=edge_id,
        treatment=from_node,
        outcome=to_node,
        estimand=estimand_type,
        candidate_designs=allowed_designs,
        required_data_assets=set(required_nodes),
    )

    # Create data dependencies for each required node
    for node_id in required_nodes:
        task.data_dependencies.append(DataDependency(
            node_id=node_id,
            connector="",  # Will be filled when data is fetched
            dataset="",
            series="",
        ))

    # Create artifact dependencies for upstream edges
    if upstream_edges:
        for upstream_edge in upstream_edges:
            task.artifact_dependencies.append(ArtifactDependency(
                source_edge_id=upstream_edge,
                artifact_type="edge_card",
            ))
            task.required_upstream_artifacts.add(upstream_edge)

    task.update_status()
    return task
