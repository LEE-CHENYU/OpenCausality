"""
Priority Computation for Task Scheduling.

Computes task priorities based on:
1. Path to target node (critical edges)
2. Artifact reusability
3. Data download budgets
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.agentic.dag.parser import DAGSpec, EdgeSpec
from shared.agentic.queue.task import LinkageTask, TaskPriority


@dataclass
class PriorityConfig:
    """Configuration for priority computation."""

    # Weight for distance from target (closer = higher priority)
    target_distance_weight: float = 2.0

    # Weight for number of dependent edges
    dependents_weight: float = 1.0

    # Weight for reusable artifacts
    artifact_reuse_weight: float = 0.5


class PriorityComputer:
    """
    Computes task priorities based on DAG structure.

    Priority rules:
    1. CRITICAL: Edges on direct path to target_node
    2. HIGH: Edges that provide artifacts for critical edges
    3. MEDIUM: Edges that create reusable artifacts
    4. LOW: All other edges
    5. BACKGROUND: Data download tasks
    """

    def __init__(self, dag: DAGSpec, config: PriorityConfig | None = None):
        self.dag = dag
        self.config = config or PriorityConfig()
        self.target_node = dag.metadata.target_node

        # Computed path info
        self._target_path_edges: set[str] = set()
        self._edge_distances: dict[str, int] = {}
        self._edge_dependents: dict[str, set[str]] = {}

        self._compute_path_info()

    def _compute_path_info(self) -> None:
        """Compute path information from all nodes to target."""
        if not self.target_node:
            return

        # BFS backward from target to find all paths
        visited = {self.target_node}
        queue = [(self.target_node, 0)]
        node_distances = {self.target_node: 0}

        while queue:
            current_node, distance = queue.pop(0)

            # Find edges pointing to this node
            for edge in self.dag.get_edges_to(current_node):
                self._target_path_edges.add(edge.id)
                self._edge_distances[edge.id] = distance

                if edge.from_node not in visited:
                    visited.add(edge.from_node)
                    node_distances[edge.from_node] = distance + 1
                    queue.append((edge.from_node, distance + 1))

        # Compute edge dependents (edges that depend on each edge)
        for edge in self.dag.edges:
            self._edge_dependents[edge.id] = set()

        for edge in self.dag.edges:
            # Find edges that use this edge's outcome as treatment or control
            for other_edge in self.dag.edges:
                if other_edge.id == edge.id:
                    continue

                # If other edge's treatment is this edge's outcome
                if other_edge.from_node == edge.to_node:
                    self._edge_dependents[edge.id].add(other_edge.id)

                # If other edge requires this edge's outcome as adjustment
                if edge.to_node in other_edge.required_adjustments:
                    self._edge_dependents[edge.id].add(other_edge.id)

    def compute_priority(self, task: LinkageTask) -> TaskPriority:
        """
        Compute priority for a task.

        Args:
            task: The task to prioritize

        Returns:
            TaskPriority level
        """
        edge_id = task.edge_id

        # Critical: on direct path to target
        if edge_id in self._target_path_edges:
            distance = self._edge_distances.get(edge_id, float('inf'))
            if distance <= 1:
                return TaskPriority.CRITICAL
            elif distance <= 3:
                return TaskPriority.HIGH

        # High: provides artifacts for critical edges
        dependents = self._edge_dependents.get(edge_id, set())
        critical_dependents = dependents & self._target_path_edges
        if critical_dependents:
            return TaskPriority.HIGH

        # Medium: has multiple dependents (reusable)
        if len(dependents) >= 2:
            return TaskPriority.MEDIUM

        # Low: everything else
        return TaskPriority.LOW

    def compute_score(self, task: LinkageTask) -> float:
        """
        Compute numeric priority score for sorting.

        Higher score = higher priority.

        Args:
            task: The task to score

        Returns:
            Numeric priority score
        """
        edge_id = task.edge_id
        config = self.config
        score = 0.0

        # Distance to target (closer = higher)
        if edge_id in self._edge_distances:
            distance = self._edge_distances[edge_id]
            max_distance = max(self._edge_distances.values()) if self._edge_distances else 1
            score += config.target_distance_weight * (1 - distance / max(max_distance, 1))

        # Number of dependents
        dependents = self._edge_dependents.get(edge_id, set())
        score += config.dependents_weight * len(dependents) / max(len(self.dag.edges), 1)

        # Artifact reuse potential (IV edges produce reusable first-stage)
        edge = self.dag.get_edge(edge_id)
        if edge and "IV_2SLS" in edge.allowed_designs:
            score += config.artifact_reuse_weight

        return score

    def get_critical_edges(self) -> set[str]:
        """Get all edges on the path to target."""
        return self._target_path_edges.copy()

    def get_edge_distance(self, edge_id: str) -> int | None:
        """Get distance from edge to target (number of hops)."""
        return self._edge_distances.get(edge_id)

    def get_edge_dependents(self, edge_id: str) -> set[str]:
        """Get edges that depend on this edge."""
        return self._edge_dependents.get(edge_id, set()).copy()


def sort_tasks_by_priority(
    tasks: list[LinkageTask],
    dag: DAGSpec,
) -> list[LinkageTask]:
    """
    Sort tasks by priority (highest first).

    Args:
        tasks: List of tasks to sort
        dag: DAG specification for priority computation

    Returns:
        Sorted list of tasks
    """
    computer = PriorityComputer(dag)

    # Compute priority and score for each task
    for task in tasks:
        task.priority = computer.compute_priority(task)

    # Sort by priority level, then by score within level
    def sort_key(task: LinkageTask) -> tuple[int, float]:
        score = computer.compute_score(task)
        return (task.priority.value, -score)

    return sorted(tasks, key=sort_key)
