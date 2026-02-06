"""
Dependency-Aware Task Queue.

Manages task scheduling with:
- Dependency graph tracking
- Priority-based ordering
- Status updates and blocking
- Idempotency via locks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Iterator

from shared.agentic.dag.parser import DAGSpec, EdgeSpec
from shared.agentic.queue.task import (
    LinkageTask,
    TaskStatus,
    TaskPriority,
    DataDependency,
    ArtifactDependency,
    create_task_from_edge,
)
from shared.agentic.queue.priority import PriorityComputer, sort_tasks_by_priority

logger = logging.getLogger(__name__)


@dataclass
class QueueStats:
    """Statistics about the task queue."""

    total: int = 0
    pending: int = 0
    ready: int = 0
    running: int = 0
    completed: int = 0
    blocked: int = 0
    failed: int = 0

    def summary(self) -> str:
        return (
            f"Total: {self.total} | "
            f"Ready: {self.ready} | Running: {self.running} | "
            f"Completed: {self.completed} | Blocked: {self.blocked}"
        )


class TaskQueue:
    """
    Dependency-aware task queue for DAG estimation.

    Handles:
    - Task creation from DAG specification
    - Dependency tracking (data and artifacts)
    - Priority-based scheduling
    - Status updates and propagation
    - Idempotent operations via locks
    """

    def __init__(
        self,
        dag: DAGSpec,
        target_node: str | None = None,
    ):
        """
        Initialize task queue from DAG.

        Args:
            dag: The DAG specification
            target_node: Override target node (uses dag.metadata.target_node if None)
        """
        self.dag = dag
        self.target_node = target_node or dag.metadata.target_node

        # Task storage
        self._tasks: dict[str, LinkageTask] = {}

        # Dependency graph
        self._edge_to_task: dict[str, str] = {}  # edge_id -> task edge_id
        self._data_waiters: dict[str, set[str]] = {}  # node_id -> task edge_ids waiting

        # Priority computer
        self._priority_computer = PriorityComputer(dag)

        # Thread safety (RLock for reentrant locking in mark_complete -> mark_artifact_available)
        self._lock = RLock()

        # Asset locks for idempotency
        self._asset_locks: dict[str, str] = {}  # asset_id -> locker_id

        # Build tasks from DAG
        self._build_tasks()

    def _build_tasks(self) -> None:
        """Build tasks from DAG edges."""
        for edge in self.dag.edges:
            # Collect required nodes
            required_nodes = [edge.from_node, edge.to_node]
            required_nodes.extend(edge.required_adjustments)

            # Find upstream edges (edges whose outcome is this edge's treatment)
            upstream_edges = []
            for other_edge in self.dag.edges:
                if other_edge.to_node == edge.from_node:
                    upstream_edges.append(other_edge.id)

            # Create task
            task = create_task_from_edge(
                edge_id=edge.id,
                from_node=edge.from_node,
                to_node=edge.to_node,
                estimand_type=edge.estimand.type,
                allowed_designs=edge.allowed_designs,
                required_nodes=required_nodes,
                upstream_edges=upstream_edges if upstream_edges else None,
            )

            # Compute priority
            task.priority = self._priority_computer.compute_priority(task)

            # Store task
            self._tasks[edge.id] = task
            self._edge_to_task[edge.id] = edge.id

            # Track data waiters
            for node_id in required_nodes:
                if node_id not in self._data_waiters:
                    self._data_waiters[node_id] = set()
                self._data_waiters[node_id].add(edge.id)

    def add(self, task: LinkageTask) -> None:
        """Add a task to the queue."""
        with self._lock:
            self._tasks[task.edge_id] = task
            self._edge_to_task[task.edge_id] = task.edge_id

            # Track data waiters
            for dep in task.data_dependencies:
                if dep.node_id not in self._data_waiters:
                    self._data_waiters[dep.node_id] = set()
                self._data_waiters[dep.node_id].add(task.edge_id)

    def get(self, edge_id: str) -> LinkageTask | None:
        """Get a task by edge ID."""
        return self._tasks.get(edge_id)

    def get_ready(self) -> list[LinkageTask]:
        """
        Get all tasks ready to run, sorted by priority.

        Returns:
            List of ready tasks, highest priority first
        """
        with self._lock:
            ready = [t for t in self._tasks.values() if t.status == TaskStatus.READY]
            return sort_tasks_by_priority(ready, self.dag)

    def get_next(self) -> LinkageTask | None:
        """
        Get the highest priority ready task.

        Returns:
            The next task to run, or None if no tasks ready
        """
        ready = self.get_ready()
        return ready[0] if ready else None

    def get_all(self) -> list[LinkageTask]:
        """Get all tasks."""
        return list(self._tasks.values())

    def get_by_status(self, status: TaskStatus) -> list[LinkageTask]:
        """Get all tasks with a specific status."""
        return [t for t in self._tasks.values() if t.status == status]

    def mark_data_available(self, node_id: str, data_ref: str) -> list[str]:
        """
        Mark data for a node as available.

        Updates all tasks waiting on this data.

        Args:
            node_id: The node ID whose data is available
            data_ref: Reference to the data (path or cache key)

        Returns:
            List of edge IDs that became ready
        """
        newly_ready = []

        with self._lock:
            waiting_tasks = self._data_waiters.get(node_id, set())

            for edge_id in waiting_tasks:
                task = self._tasks.get(edge_id)
                if task:
                    old_status = task.status
                    task.mark_data_satisfied(node_id, data_ref)

                    if old_status != TaskStatus.READY and task.status == TaskStatus.READY:
                        newly_ready.append(edge_id)
                        logger.info(f"Task {edge_id} became ready")

        return newly_ready

    def mark_artifact_available(self, source_edge_id: str, artifact_ref: str) -> list[str]:
        """
        Mark an artifact from an edge as available.

        Updates all tasks waiting on this artifact.

        Args:
            source_edge_id: The edge that produced the artifact
            artifact_ref: Reference to the artifact

        Returns:
            List of edge IDs that became ready
        """
        newly_ready = []

        with self._lock:
            for task in self._tasks.values():
                for dep in task.artifact_dependencies:
                    if dep.source_edge_id == source_edge_id:
                        old_status = task.status
                        task.mark_artifact_satisfied(source_edge_id, artifact_ref)

                        if old_status != TaskStatus.READY and task.status == TaskStatus.READY:
                            newly_ready.append(task.edge_id)
                            logger.info(f"Task {task.edge_id} became ready")

        return newly_ready

    def update_status(self, edge_id: str, status: TaskStatus) -> None:
        """Update a task's status."""
        with self._lock:
            task = self._tasks.get(edge_id)
            if task:
                task.status = status

    def mark_blocked(
        self,
        edge_id: str,
        reason: str,
        block_type: TaskStatus = TaskStatus.BLOCKED_ID,
    ) -> None:
        """Mark a task as blocked."""
        with self._lock:
            task = self._tasks.get(edge_id)
            if task:
                task.block(reason, block_type)
                logger.warning(f"Task {edge_id} blocked: {reason}")

    def mark_complete(
        self,
        edge_id: str,
        results_ref: str,
        credibility_score: float,
        credibility_rating: str,
    ) -> None:
        """Mark a task as complete."""
        with self._lock:
            task = self._tasks.get(edge_id)
            if task:
                task.complete(results_ref, credibility_score, credibility_rating)
                logger.info(
                    f"Task {edge_id} completed with credibility "
                    f"{credibility_rating} ({credibility_score:.2f})"
                )

                # Propagate artifact availability
                self.mark_artifact_available(edge_id, results_ref)

    def reset_task(self, edge_id: str, reason: str = "") -> bool:
        """Reset a completed/failed/blocked task for re-estimation."""
        with self._lock:
            task = self._tasks.get(edge_id)
            if task and task.is_done():
                task.reset(reason)
                return True
            return False

    def get_prev_credibility_scores(self) -> dict[str, float]:
        """Get previous iteration credibility scores for delta tracking."""
        return {
            eid: t._prev_credibility_score
            for eid, t in self._tasks.items()
            if t._prev_credibility_score is not None
        }

    def mark_failed(self, edge_id: str, reason: str) -> None:
        """Mark a task as failed."""
        with self._lock:
            task = self._tasks.get(edge_id)
            if task:
                task.fail(reason)
                logger.error(f"Task {edge_id} failed: {reason}")

    def acquire_lock(self, asset_id: str, locker_id: str) -> bool:
        """
        Acquire a lock on an asset (for idempotency).

        Args:
            asset_id: The asset to lock
            locker_id: ID of the locker

        Returns:
            True if lock acquired, False if already locked
        """
        with self._lock:
            if asset_id in self._asset_locks:
                return self._asset_locks[asset_id] == locker_id
            self._asset_locks[asset_id] = locker_id
            return True

    def release_lock(self, asset_id: str, locker_id: str) -> bool:
        """
        Release a lock on an asset.

        Args:
            asset_id: The asset to unlock
            locker_id: ID of the locker

        Returns:
            True if released, False if not owner
        """
        with self._lock:
            if self._asset_locks.get(asset_id) == locker_id:
                del self._asset_locks[asset_id]
                return True
            return False

    def stats(self) -> QueueStats:
        """Get queue statistics."""
        stats = QueueStats(total=len(self._tasks))

        for task in self._tasks.values():
            if task.status == TaskStatus.NEW:
                stats.pending += 1
            elif task.status == TaskStatus.READY:
                stats.ready += 1
            elif task.status == TaskStatus.RUNNING:
                stats.running += 1
            elif task.status in {TaskStatus.DONE, TaskStatus.DONE_SUGGESTIVE}:
                stats.completed += 1
            elif task.status in {TaskStatus.WAITING_DATA, TaskStatus.WAITING_ARTIFACT,
                                TaskStatus.BLOCKED_ID}:
                stats.blocked += 1
            elif task.status == TaskStatus.FAILED:
                stats.failed += 1

        return stats

    def is_complete(self) -> bool:
        """Check if all tasks are done (completed, blocked, or failed)."""
        return all(t.is_done() for t in self._tasks.values())

    def get_critical_tasks(self) -> list[LinkageTask]:
        """Get tasks on the critical path to target."""
        critical_edges = self._priority_computer.get_critical_edges()
        return [self._tasks[e] for e in critical_edges if e in self._tasks]

    def get_blocking_issues(self) -> dict[str, str]:
        """Get all blocked tasks and their reasons."""
        return {
            t.edge_id: t.blocked_reason or "Unknown"
            for t in self._tasks.values()
            if t.is_blocked() and t.blocked_reason
        }

    def iter_by_priority(self) -> Iterator[LinkageTask]:
        """Iterate over all tasks in priority order."""
        sorted_tasks = sort_tasks_by_priority(list(self._tasks.values()), self.dag)
        yield from sorted_tasks

    def summary(self) -> str:
        """Generate a summary of queue state."""
        stats = self.stats()
        lines = [
            "=" * 60,
            "TASK QUEUE SUMMARY",
            "=" * 60,
            stats.summary(),
            "",
            "TASKS BY STATUS:",
        ]

        for status in TaskStatus:
            tasks = self.get_by_status(status)
            if tasks:
                lines.append(f"\n  {status.value}:")
                for task in tasks[:5]:  # Show first 5
                    priority = task.priority.name
                    lines.append(f"    [{priority}] {task.edge_id}: {task.treatment} → {task.outcome}")
                if len(tasks) > 5:
                    lines.append(f"    ... and {len(tasks) - 5} more")

        if blocking := self.get_blocking_issues():
            lines.append("\nBLOCKING ISSUES:")
            for edge_id, reason in blocking.items():
                lines.append(f"  {edge_id}: {reason}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


def create_queue_from_dag(
    dag: DAGSpec,
    target_node: str | None = None,
) -> TaskQueue:
    """
    Create a task queue from a DAG specification.

    Convenience function that initializes the queue and
    computes all priorities.

    Args:
        dag: The DAG specification
        target_node: Override target node

    Returns:
        Configured TaskQueue
    """
    return TaskQueue(dag, target_node)
