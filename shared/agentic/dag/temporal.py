"""
Temporal DAG Expansion.

Expands a static DAG into a temporal graph for cycle detection.
Each node becomes multiple nodes (node_t, node_{t+1}, ...) based on
the lag structure of edges.

This allows detection of cycles that would otherwise be hidden
in contemporaneous representations (e.g., FX → CPI → Policy → FX
becomes valid when lags are properly specified).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from shared.agentic.dag.parser import DAGSpec, EdgeSpec, NodeSpec


@dataclass
class TemporalNode:
    """A node at a specific time period."""

    base_id: str  # Original node ID
    time_offset: int  # Time offset from reference (0 = current)

    @property
    def id(self) -> str:
        """Unique ID for this temporal node."""
        if self.time_offset == 0:
            return f"{self.base_id}_t"
        elif self.time_offset > 0:
            return f"{self.base_id}_t+{self.time_offset}"
        else:
            return f"{self.base_id}_t{self.time_offset}"

    def __hash__(self):
        return hash((self.base_id, self.time_offset))

    def __eq__(self, other):
        if not isinstance(other, TemporalNode):
            return False
        return self.base_id == other.base_id and self.time_offset == other.time_offset


@dataclass
class TemporalEdge:
    """An edge between temporal nodes."""

    from_node: TemporalNode
    to_node: TemporalNode
    original_edge_id: str

    @property
    def id(self) -> str:
        """Unique ID for this temporal edge."""
        return f"{self.from_node.id}→{self.to_node.id}"

    def __hash__(self):
        return hash((self.from_node, self.to_node))

    def __eq__(self, other):
        if not isinstance(other, TemporalEdge):
            return False
        return self.from_node == other.from_node and self.to_node == other.to_node


@dataclass
class TemporalDAG:
    """
    Temporal expansion of a DAG.

    Represents the DAG across time periods to enable proper
    cycle detection and temporal ordering.
    """

    nodes: set[TemporalNode] = field(default_factory=set)
    edges: set[TemporalEdge] = field(default_factory=set)
    max_horizon: int = 12  # Maximum time horizon to expand

    # Index for fast lookup
    _adjacency: dict[TemporalNode, set[TemporalNode]] = field(
        default_factory=dict, repr=False
    )

    def add_node(self, node: TemporalNode) -> None:
        """Add a temporal node."""
        self.nodes.add(node)
        if node not in self._adjacency:
            self._adjacency[node] = set()

    def add_edge(self, edge: TemporalEdge) -> None:
        """Add a temporal edge."""
        self.add_node(edge.from_node)
        self.add_node(edge.to_node)
        self.edges.add(edge)
        self._adjacency[edge.from_node].add(edge.to_node)

    def get_successors(self, node: TemporalNode) -> set[TemporalNode]:
        """Get all nodes reachable from this node in one step."""
        return self._adjacency.get(node, set())

    def has_cycle(self) -> bool:
        """
        Detect if the temporal DAG has a cycle.

        Uses DFS with coloring:
        - WHITE (0): unvisited
        - GRAY (1): visiting (in current DFS path)
        - BLACK (2): finished (all descendants visited)

        A cycle exists if we encounter a GRAY node.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in self.nodes}

        def dfs(node: TemporalNode) -> bool:
            """Returns True if cycle found."""
            color[node] = GRAY

            for successor in self.get_successors(node):
                if color[successor] == GRAY:
                    # Back edge found - cycle!
                    return True
                if color[successor] == WHITE:
                    if dfs(successor):
                        return True

            color[node] = BLACK
            return False

        for node in self.nodes:
            if color[node] == WHITE:
                if dfs(node):
                    return True

        return False

    def find_cycle(self) -> list[TemporalNode] | None:
        """
        Find and return a cycle if one exists.

        Returns a list of nodes forming the cycle, or None.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in self.nodes}
        parent = {node: None for node in self.nodes}

        def dfs(node: TemporalNode) -> TemporalNode | None:
            """Returns the node where cycle was detected, or None."""
            color[node] = GRAY

            for successor in self.get_successors(node):
                if color[successor] == GRAY:
                    # Found cycle - return the node
                    parent[successor] = node
                    return successor
                if color[successor] == WHITE:
                    parent[successor] = node
                    result = dfs(successor)
                    if result is not None:
                        return result

            color[node] = BLACK
            return None

        for node in self.nodes:
            if color[node] == WHITE:
                cycle_node = dfs(node)
                if cycle_node is not None:
                    # Reconstruct cycle
                    cycle = [cycle_node]
                    current = parent[cycle_node]
                    while current != cycle_node:
                        cycle.append(current)
                        current = parent[current]
                    cycle.append(cycle_node)
                    return list(reversed(cycle))

        return None

    def topological_order(self) -> list[TemporalNode] | None:
        """
        Return nodes in topological order, or None if cycle exists.
        """
        if self.has_cycle():
            return None

        # Kahn's algorithm
        in_degree = {node: 0 for node in self.nodes}
        for edge in self.edges:
            in_degree[edge.to_node] += 1

        queue = [node for node in self.nodes if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for successor in self.get_successors(node):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        return result if len(result) == len(self.nodes) else None


def _convert_lag_to_periods(
    lag: int,
    lag_unit: str,
    base_unit: str = "quarter"
) -> int:
    """
    Convert a lag to the base time unit.

    Args:
        lag: Number of lag units
        lag_unit: Unit of the lag (day, month, quarter, year)
        base_unit: Base unit for the temporal DAG

    Returns:
        Lag in base units
    """
    # Conversion factors to quarters (our default base)
    to_quarters = {
        "day": 1 / 90,  # Approximate
        "month": 1 / 3,
        "quarter": 1,
        "year": 4,
    }

    from_quarters = {
        "day": 90,
        "month": 3,
        "quarter": 1,
        "year": 0.25,
    }

    # Convert to quarters, then to base unit
    quarters = lag * to_quarters.get(lag_unit, 1)
    return int(quarters * from_quarters.get(base_unit, 1))


def expand_temporal_dag(
    dag: DAGSpec,
    max_horizon: int = 12,
    base_unit: Literal["day", "month", "quarter", "year"] = "quarter",
) -> TemporalDAG:
    """
    Expand a static DAG into a temporal DAG.

    Each edge X → Y with lag L becomes:
    X_t → Y_{t+L}

    For contemporaneous edges (lag=0, contemporaneous=True):
    X_t → Y_t

    Args:
        dag: The static DAG specification
        max_horizon: Maximum time horizon to expand
        base_unit: Base time unit for the expansion

    Returns:
        TemporalDAG with expanded nodes and edges
    """
    temporal = TemporalDAG(max_horizon=max_horizon)

    # Add all nodes at t=0
    for node in dag.nodes:
        temporal.add_node(TemporalNode(base_id=node.id, time_offset=0))

    # Process each edge
    for edge in dag.edges:
        timing = edge.timing

        if timing.contemporaneous:
            # X_t → Y_t
            from_node = TemporalNode(base_id=edge.from_node, time_offset=0)
            to_node = TemporalNode(base_id=edge.to_node, time_offset=0)
            temporal.add_edge(TemporalEdge(
                from_node=from_node,
                to_node=to_node,
                original_edge_id=edge.id,
            ))
        else:
            # X_t → Y_{t+lag}
            lag_periods = _convert_lag_to_periods(
                timing.lag,
                timing.lag_unit,
                base_unit,
            )

            # Create edges for multiple time periods
            for t in range(max_horizon - lag_periods):
                from_node = TemporalNode(base_id=edge.from_node, time_offset=t)
                to_node = TemporalNode(base_id=edge.to_node, time_offset=t + lag_periods)
                temporal.add_edge(TemporalEdge(
                    from_node=from_node,
                    to_node=to_node,
                    original_edge_id=edge.id,
                ))

        # Handle anticipation effects (leads)
        if timing.max_anticipation > 0:
            for lead in range(1, timing.max_anticipation + 1):
                for t in range(lead, max_horizon):
                    from_node = TemporalNode(base_id=edge.from_node, time_offset=t)
                    to_node = TemporalNode(base_id=edge.to_node, time_offset=t - lead)
                    temporal.add_edge(TemporalEdge(
                        from_node=from_node,
                        to_node=to_node,
                        original_edge_id=edge.id,
                    ))

    return temporal


def check_temporal_cycle(dag: DAGSpec, max_horizon: int = 12) -> tuple[bool, list[str] | None]:
    """
    Check if the DAG has temporal cycles.

    Args:
        dag: The DAG specification
        max_horizon: Maximum horizon for temporal expansion

    Returns:
        Tuple of (has_cycle, cycle_description)
        If has_cycle is True, cycle_description lists the nodes in the cycle
    """
    temporal = expand_temporal_dag(dag, max_horizon)

    if temporal.has_cycle():
        cycle = temporal.find_cycle()
        if cycle:
            # Convert temporal nodes back to readable description
            description = [f"{n.base_id} at t{'+' if n.time_offset >= 0 else ''}{n.time_offset}"
                          for n in cycle]
            return True, description
        return True, ["Unknown cycle detected"]

    return False, None
