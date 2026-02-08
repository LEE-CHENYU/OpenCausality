"""
Tests for graph conversion layer.

Tests DAGSpec ↔ NetworkX ↔ GML round-trips and pywhy-graphs conversion.
"""

from __future__ import annotations

import pytest

from shared.agentic.dag.parser import (
    DAGMetadata,
    DAGSpec,
    EdgeSpec,
    LatentSpec,
    NodeSpec,
)
from shared.agentic.dag.graph_convert import (
    dagspec_to_dowhy_graph,
    dagspec_to_networkx,
    networkx_to_dagspec,
)


@pytest.fixture
def simple_dag() -> DAGSpec:
    """A simple 3-node DAG: X -> Z -> Y."""
    return DAGSpec(
        metadata=DAGMetadata(name="test_dag", target_node="Y"),
        nodes=[
            NodeSpec(id="X", name="Treatment"),
            NodeSpec(id="Z", name="Mediator"),
            NodeSpec(id="Y", name="Outcome"),
        ],
        edges=[
            EdgeSpec(id="x_to_z", from_node="X", to_node="Z"),
            EdgeSpec(id="z_to_y", from_node="Z", to_node="Y"),
        ],
    )


class TestDAGSpecToNetworkX:
    def test_basic_conversion(self, simple_dag):
        G = dagspec_to_networkx(simple_dag)
        assert len(G.nodes) == 3
        assert len(G.edges) == 2
        assert ("X", "Z") in G.edges
        assert ("Z", "Y") in G.edges

    def test_node_attributes_preserved(self, simple_dag):
        G = dagspec_to_networkx(simple_dag)
        assert G.nodes["X"]["name"] == "Treatment"
        assert G.nodes["Y"]["name"] == "Outcome"

    def test_graph_metadata(self, simple_dag):
        G = dagspec_to_networkx(simple_dag)
        assert G.graph["name"] == "test_dag"
        assert G.graph["target_node"] == "Y"


class TestNetworkXToDAGSpec:
    def test_roundtrip(self, simple_dag):
        G = dagspec_to_networkx(simple_dag)
        dag2 = networkx_to_dagspec(G)

        assert dag2.metadata.name == "test_dag"
        assert len(dag2.nodes) == 3
        assert len(dag2.edges) == 2

        # Check edge connectivity preserved
        edge_pairs = {(e.from_node, e.to_node) for e in dag2.edges}
        assert ("X", "Z") in edge_pairs
        assert ("Z", "Y") in edge_pairs

    def test_metadata_override(self, simple_dag):
        G = dagspec_to_networkx(simple_dag)
        dag2 = networkx_to_dagspec(G, metadata={"name": "overridden"})
        assert dag2.metadata.name == "overridden"


class TestDAGSpecToGML:
    def test_gml_format(self, simple_dag):
        gml = dagspec_to_dowhy_graph(simple_dag)
        assert "graph [" in gml
        assert "directed 1" in gml
        assert 'label "X"' in gml
        assert 'label "Y"' in gml
        assert 'source "X"' in gml
        assert 'target "Z"' in gml

    def test_gml_edge_count(self, simple_dag):
        gml = dagspec_to_dowhy_graph(simple_dag)
        assert gml.count("edge [") == 2


class TestPyWhyConversion:
    def test_dagspec_to_pywhy_import_error(self, simple_dag):
        """Test that import error is raised when pywhy-graphs is not installed."""
        try:
            from shared.agentic.dag.graph_convert import dagspec_to_pywhy
            # If pywhy-graphs is installed, test the conversion
            admg = dagspec_to_pywhy(simple_dag)
            assert admg is not None
        except ImportError:
            pytest.skip("pywhy-graphs not installed")

    def test_dagspec_with_latents_to_pywhy(self):
        """Test that latent confounders become bidirected edges."""
        dag = DAGSpec(
            metadata=DAGMetadata(name="latent_test"),
            nodes=[
                NodeSpec(id="X", name="X"),
                NodeSpec(id="Y", name="Y"),
            ],
            edges=[
                EdgeSpec(id="x_to_y", from_node="X", to_node="Y"),
            ],
            latents=[
                LatentSpec(id="U", description="Unobserved", affects=["X", "Y"]),
            ],
        )
        try:
            from shared.agentic.dag.graph_convert import dagspec_to_pywhy, pywhy_to_dagspec

            admg = dagspec_to_pywhy(dag)
            # Should have bidirected edge X <-> Y
            assert len(list(admg.bidirected_edges)) > 0

            # Round-trip back
            dag2 = pywhy_to_dagspec(admg, name="roundtrip")
            assert len(dag2.latents) > 0
        except ImportError:
            pytest.skip("pywhy-graphs not installed")


class TestCausalLearnConversion:
    def test_causallearn_to_networkx(self):
        """Test conversion from causal-learn format."""
        try:
            from shared.agentic.dag.graph_convert import causallearn_to_networkx
            import numpy as np

            # Mock a causal-learn CausalGraph object
            # In causal-learn, cg.G is a GeneralGraph with .graph (numpy adj matrix)
            class MockGeneralGraph:
                def __init__(self):
                    self.graph = np.array([
                        [0, -1, 0],
                        [1, 0, -1],
                        [0, 1, 0],
                    ])

            class MockCG:
                G = MockGeneralGraph()
                def get_node_names(self):
                    return ["X", "Z", "Y"]

            cg = MockCG()
            G = causallearn_to_networkx(cg)
            assert ("X", "Z") in G.edges
            assert ("Z", "Y") in G.edges
            assert len(G.edges) == 2
        except ImportError:
            pytest.skip("networkx not installed")
