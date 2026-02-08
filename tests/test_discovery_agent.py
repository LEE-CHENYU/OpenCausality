"""
Tests for the Discovery Agent.

Tests PC/GES skeleton recovery and compare_with_dag classification.
Requires causal-learn to be installed. Skip if not available.
"""

from __future__ import annotations

import pytest

import numpy as np

from shared.agentic.dag.parser import DAGMetadata, DAGSpec, EdgeSpec, NodeSpec
from tests.fixtures.synthetic_dgp import make_discovery_dgp

try:
    import causallearn
    HAS_CAUSALLEARN = True
except ImportError:
    HAS_CAUSALLEARN = False


@pytest.fixture
def discovery_data():
    """3-node DAG data: X -> Z -> Y."""
    return make_discovery_dgp(n=500)


@pytest.fixture
def reference_dag() -> DAGSpec:
    """Known DAG: X -> Z -> Y."""
    return DAGSpec(
        metadata=DAGMetadata(name="ref"),
        nodes=[
            NodeSpec(id="X", name="X"),
            NodeSpec(id="Z", name="Z"),
            NodeSpec(id="Y", name="Y"),
        ],
        edges=[
            EdgeSpec(id="x_to_z", from_node="X", to_node="Z"),
            EdgeSpec(id="z_to_y", from_node="Z", to_node="Y"),
        ],
    )


@pytest.mark.skipif(not HAS_CAUSALLEARN, reason="causal-learn not installed")
class TestPCAlgorithm:
    def test_skeleton_recovery(self, discovery_data):
        from shared.agentic.agents.discovery_agent import DiscoveryAgent

        df, truth = discovery_data
        agent = DiscoveryAgent()
        result = agent.run_pc(df, alpha=0.05)

        assert result.algorithm == "PC"
        assert len(result.node_names) == 3

        # Check that the skeleton is recovered (at least X-Z and Z-Y adjacency)
        edge_pairs = set()
        for e in result.edges_discovered:
            edge_pairs.add((e.from_node, e.to_node))
            if e.edge_type == "undirected":
                edge_pairs.add((e.to_node, e.from_node))

        # At minimum, X and Z should be adjacent, Z and Y should be adjacent
        has_xz = ("X", "Z") in edge_pairs or ("Z", "X") in edge_pairs
        has_zy = ("Z", "Y") in edge_pairs or ("Y", "Z") in edge_pairs
        assert has_xz, f"Expected X-Z adjacency, got edges: {edge_pairs}"
        assert has_zy, f"Expected Z-Y adjacency, got edges: {edge_pairs}"


@pytest.mark.skipif(not HAS_CAUSALLEARN, reason="causal-learn not installed")
class TestGESAlgorithm:
    def test_skeleton_recovery(self, discovery_data):
        from shared.agentic.agents.discovery_agent import DiscoveryAgent

        df, _ = discovery_data
        agent = DiscoveryAgent()
        result = agent.run_ges(df)

        assert result.algorithm == "GES"
        assert len(result.edges_discovered) >= 1


class TestCompareWithDAG:
    def test_confirmed_edges(self, reference_dag):
        from shared.agentic.agents.discovery_agent import (
            DiscoveryAgent,
            DiscoveredEdge,
            DiscoveryResult,
        )

        # Discovery found the same edges
        disc_result = DiscoveryResult(
            algorithm="test",
            edges_discovered=[
                DiscoveredEdge(from_node="X", to_node="Z"),
                DiscoveredEdge(from_node="Z", to_node="Y"),
            ],
            node_names=["X", "Z", "Y"],
        )

        agent = DiscoveryAgent()
        comparison = agent.compare_with_dag(disc_result, reference_dag)

        assert len(comparison["confirmed"]) == 2
        assert len(comparison["contradicted"]) == 0
        assert len(comparison["novel"]) == 0
        assert len(comparison["missing"]) == 0

    def test_contradicted_edges(self, reference_dag):
        from shared.agentic.agents.discovery_agent import (
            DiscoveryAgent,
            DiscoveredEdge,
            DiscoveryResult,
        )

        # Discovery found reversed direction
        disc_result = DiscoveryResult(
            algorithm="test",
            edges_discovered=[
                DiscoveredEdge(from_node="Z", to_node="X"),  # reversed!
                DiscoveredEdge(from_node="Z", to_node="Y"),
            ],
            node_names=["X", "Z", "Y"],
        )

        agent = DiscoveryAgent()
        comparison = agent.compare_with_dag(disc_result, reference_dag)

        assert len(comparison["contradicted"]) == 1
        assert comparison["contradicted"][0].from_node == "Z"
        assert comparison["contradicted"][0].to_node == "X"
        assert len(comparison["missing"]) == 1  # X->Z missing

    def test_novel_edges(self, reference_dag):
        from shared.agentic.agents.discovery_agent import (
            DiscoveryAgent,
            DiscoveredEdge,
            DiscoveryResult,
        )

        # Discovery found an extra edge
        disc_result = DiscoveryResult(
            algorithm="test",
            edges_discovered=[
                DiscoveredEdge(from_node="X", to_node="Z"),
                DiscoveredEdge(from_node="Z", to_node="Y"),
                DiscoveredEdge(from_node="X", to_node="Y"),  # novel!
            ],
            node_names=["X", "Z", "Y"],
        )

        agent = DiscoveryAgent()
        comparison = agent.compare_with_dag(disc_result, reference_dag)

        assert len(comparison["novel"]) == 1
        assert comparison["novel"][0].from_node == "X"
        assert comparison["novel"][0].to_node == "Y"


class TestToProposedEdges:
    def test_conversion(self):
        from shared.agentic.agents.discovery_agent import (
            DiscoveryAgent,
            DiscoveredEdge,
            DiscoveryResult,
        )

        result = DiscoveryResult(
            algorithm="PC",
            edges_discovered=[
                DiscoveredEdge(from_node="A", to_node="B"),
                DiscoveredEdge(from_node="B", to_node="C", edge_type="undirected"),
            ],
            node_names=["A", "B", "C"],
        )

        proposals = DiscoveryAgent.to_proposed_edges(result)

        # Only directed edges should be converted
        assert len(proposals) == 1
        assert proposals[0].from_node == "A"
        assert proposals[0].to_node == "B"
        assert proposals[0].edge_id.startswith("disc_PC_")
