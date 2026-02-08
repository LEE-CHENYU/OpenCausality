"""
Discovery Agent.

Runs causal discovery algorithms (PC, GES, FCI, NOTEARS) and
compares results against an existing DAGSpec.

CRITICAL: Discovery outputs are proposals only, never auto-merged into DAG.
All results must go through HITL review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from shared.agentic.agents.paper_dag_extractor import ProposedEdge
from shared.agentic.dag.parser import DAGSpec

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredEdge:
    """A single edge discovered by a causal discovery algorithm."""

    from_node: str
    to_node: str
    edge_type: Literal["directed", "undirected", "bidirected"] = "directed"
    weight: float = 1.0
    in_existing_dag: bool = False
    contradicts_dag: bool = False


@dataclass
class DiscoveryResult:
    """Result of a causal discovery run."""

    algorithm: str
    edges_discovered: list[DiscoveredEdge] = field(default_factory=list)
    adjacency_matrix: np.ndarray | None = None
    node_names: list[str] = field(default_factory=list)
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DiscoveryAgent:
    """Runs causal discovery algorithms and compares with existing DAGs.

    Usage:
        agent = DiscoveryAgent()
        result = agent.run_pc(data, alpha=0.05, node_names=["X", "Y", "Z"])
        comparison = agent.compare_with_dag(result, existing_dag)
        proposals = agent.to_proposed_edges(result)
    """

    def run_pc(
        self,
        data: pd.DataFrame | np.ndarray,
        alpha: float = 0.05,
        node_names: list[str] | None = None,
    ) -> DiscoveryResult:
        """Run the PC algorithm for constraint-based discovery.

        Args:
            data: Data matrix (observations x variables).
            alpha: Significance level for conditional independence tests.
            node_names: Variable names. If None, uses column names or X0, X1, ...
        """
        try:
            from causallearn.search.ConstraintBased.PC import pc
        except ImportError:
            raise ImportError(
                "causal-learn is required for PC algorithm. "
                "Install with: pip install causal-learn"
            )

        arr, names = self._prepare_data(data, node_names)
        cg = pc(arr, alpha=alpha, node_names=names)

        edges = self._extract_edges(cg.G.graph, names)
        return DiscoveryResult(
            algorithm="PC",
            edges_discovered=edges,
            adjacency_matrix=cg.G.graph.copy(),
            node_names=names,
            metadata={"alpha": alpha, "n_obs": arr.shape[0]},
        )

    def run_ges(
        self,
        data: pd.DataFrame | np.ndarray,
        node_names: list[str] | None = None,
        score_func: str = "local_score_BIC",
    ) -> DiscoveryResult:
        """Run the GES (Greedy Equivalence Search) algorithm.

        Args:
            data: Data matrix.
            node_names: Variable names.
            score_func: Score function name.
        """
        try:
            from causallearn.search.ScoreBased.GES import ges
        except ImportError:
            raise ImportError(
                "causal-learn is required for GES algorithm. "
                "Install with: pip install causal-learn"
            )

        arr, names = self._prepare_data(data, node_names)
        result = ges(arr, score_func=score_func, node_names=names)

        graph_matrix = result["G"].graph
        score = result.get("score", None)

        edges = self._extract_edges(graph_matrix, names)
        return DiscoveryResult(
            algorithm="GES",
            edges_discovered=edges,
            adjacency_matrix=graph_matrix.copy(),
            node_names=names,
            score=float(score) if score is not None else None,
            metadata={"score_func": score_func, "n_obs": arr.shape[0]},
        )

    def run_fci(
        self,
        data: pd.DataFrame | np.ndarray,
        alpha: float = 0.05,
        node_names: list[str] | None = None,
    ) -> DiscoveryResult:
        """Run the FCI algorithm (handles latent confounders).

        Args:
            data: Data matrix.
            alpha: Significance level.
            node_names: Variable names.
        """
        try:
            from causallearn.search.ConstraintBased.FCI import fci
            from causallearn.utils.cit import fisherz
        except ImportError:
            raise ImportError(
                "causal-learn is required for FCI algorithm. "
                "Install with: pip install causal-learn"
            )

        arr, names = self._prepare_data(data, node_names)
        G, edges_result = fci(arr, independence_test_method=fisherz, alpha=alpha, node_names=names)

        graph_matrix = G.graph
        edges = self._extract_edges(graph_matrix, names, fci_mode=True)
        return DiscoveryResult(
            algorithm="FCI",
            edges_discovered=edges,
            adjacency_matrix=graph_matrix.copy(),
            node_names=names,
            metadata={"alpha": alpha, "n_obs": arr.shape[0], "handles_latents": True},
        )

    def run_notears(
        self,
        data: pd.DataFrame | np.ndarray,
        lambda1: float = 0.1,
        node_names: list[str] | None = None,
    ) -> DiscoveryResult:
        """Run the NOTEARS algorithm for continuous structure learning.

        Args:
            data: Data matrix.
            lambda1: L1 penalty for sparsity.
            node_names: Variable names.
        """
        try:
            from castle.algorithms import NotearsNonlinear, Notears
        except ImportError:
            raise ImportError(
                "gCastle is required for NOTEARS algorithm. "
                "Install with: pip install gcastle"
            )

        arr, names = self._prepare_data(data, node_names)

        model = Notears()
        model.learn(arr)
        W = model.causal_matrix

        # Threshold small weights
        threshold = 0.05
        W[np.abs(W) < threshold] = 0

        edges = []
        n = W.shape[0]
        for i in range(n):
            for j in range(n):
                if W[i, j] != 0:
                    edges.append(DiscoveredEdge(
                        from_node=names[i],
                        to_node=names[j],
                        edge_type="directed",
                        weight=float(abs(W[i, j])),
                    ))

        return DiscoveryResult(
            algorithm="NOTEARS",
            edges_discovered=edges,
            adjacency_matrix=W.copy(),
            node_names=names,
            metadata={"lambda1": lambda1, "n_obs": arr.shape[0]},
        )

    def compare_with_dag(
        self,
        result: DiscoveryResult,
        dag: DAGSpec,
    ) -> dict[str, list[DiscoveredEdge]]:
        """Compare discovery result with an existing DAG.

        Returns a dict with four categories:
            - ``"confirmed"``: Edges found in both discovery and DAG.
            - ``"contradicted"``: Edges where discovery found the reverse direction.
            - ``"novel"``: Edges found by discovery but not in DAG.
            - ``"missing"``: Edges in DAG but not found by discovery.
        """
        # Build set of existing directed edges
        existing_edges = {(e.from_node, e.to_node) for e in dag.edges}
        existing_reverse = {(e.to_node, e.from_node) for e in dag.edges}

        confirmed: list[DiscoveredEdge] = []
        contradicted: list[DiscoveredEdge] = []
        novel: list[DiscoveredEdge] = []

        discovered_pairs = set()

        for edge in result.edges_discovered:
            if edge.edge_type != "directed":
                continue

            pair = (edge.from_node, edge.to_node)
            discovered_pairs.add(pair)

            if pair in existing_edges:
                edge.in_existing_dag = True
                confirmed.append(edge)
            elif pair in existing_reverse:
                edge.contradicts_dag = True
                contradicted.append(edge)
            else:
                novel.append(edge)

        # Missing: edges in DAG not discovered
        missing = []
        for e in dag.edges:
            pair = (e.from_node, e.to_node)
            if pair not in discovered_pairs:
                missing.append(DiscoveredEdge(
                    from_node=e.from_node,
                    to_node=e.to_node,
                    edge_type="directed",
                    in_existing_dag=True,
                ))

        return {
            "confirmed": confirmed,
            "contradicted": contradicted,
            "novel": novel,
            "missing": missing,
        }

    @staticmethod
    def to_proposed_edges(result: DiscoveryResult) -> list[ProposedEdge]:
        """Convert discovery result to ProposedEdge objects for HITL review.

        Reuses the ProposedEdge dataclass from paper_dag_extractor.py.
        """
        proposals = []
        for edge in result.edges_discovered:
            if edge.edge_type != "directed":
                continue
            proposals.append(ProposedEdge(
                from_node=edge.from_node,
                to_node=edge.to_node,
                edge_id=f"disc_{result.algorithm}_{edge.from_node}_to_{edge.to_node}",
                edge_type="causal",
                match_confidence=edge.weight,
                is_existing=edge.in_existing_dag,
            ))
        return proposals

    @staticmethod
    def _prepare_data(
        data: pd.DataFrame | np.ndarray,
        node_names: list[str] | None,
    ) -> tuple[np.ndarray, list[str]]:
        """Convert data to numpy array and extract/generate node names."""
        if isinstance(data, pd.DataFrame):
            names = node_names or list(data.columns)
            arr = data[names].values.astype(float)
        else:
            arr = np.asarray(data, dtype=float)
            names = node_names or [f"X{i}" for i in range(arr.shape[1])]
        return arr, names

    @staticmethod
    def _extract_edges(
        graph_matrix: np.ndarray,
        names: list[str],
        fci_mode: bool = False,
    ) -> list[DiscoveredEdge]:
        """Extract edges from a causal-learn adjacency matrix.

        causal-learn encoding:
            - graph[i,j] = -1, graph[j,i] = 1  →  i -> j (directed)
            - graph[i,j] = -1, graph[j,i] = -1  →  i -- j (undirected)
            - graph[i,j] = 1,  graph[j,i] = 1   →  i <-> j (bidirected, FCI)
        """
        edges: list[DiscoveredEdge] = []
        n = graph_matrix.shape[0]
        seen = set()

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                if graph_matrix[i, j] == -1 and graph_matrix[j, i] == 1:
                    # i -> j
                    edges.append(DiscoveredEdge(
                        from_node=names[i],
                        to_node=names[j],
                        edge_type="directed",
                    ))
                elif graph_matrix[i, j] == -1 and graph_matrix[j, i] == -1:
                    # Undirected: i -- j (only add once)
                    key = (min(i, j), max(i, j))
                    if key not in seen:
                        seen.add(key)
                        edges.append(DiscoveredEdge(
                            from_node=names[i],
                            to_node=names[j],
                            edge_type="undirected",
                        ))
                elif fci_mode and graph_matrix[i, j] == 1 and graph_matrix[j, i] == 1:
                    # Bidirected: i <-> j (only add once)
                    key = (min(i, j), max(i, j))
                    if key not in seen:
                        seen.add(key)
                        edges.append(DiscoveredEdge(
                            from_node=names[i],
                            to_node=names[j],
                            edge_type="bidirected",
                        ))

        return edges
