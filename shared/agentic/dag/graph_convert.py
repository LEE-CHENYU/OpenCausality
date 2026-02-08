"""
Graph Conversion Layer.

Converts between DAGSpec and external graph formats (NetworkX, GML, pywhy-graphs).
All external imports are lazy so these remain optional dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from shared.agentic.dag.parser import DAGMetadata, DAGSpec, EdgeSpec, LatentSpec, NodeSpec

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)


def dagspec_to_networkx(dag: DAGSpec) -> "nx.DiGraph":
    """Convert a DAGSpec to a NetworkX DiGraph, preserving node/edge attributes."""
    import networkx as nx

    G = nx.DiGraph()

    for node in dag.nodes:
        G.add_node(
            node.id,
            name=node.name,
            unit=node.unit,
            frequency=node.frequency,
            type=node.type,
            observed=node.observed,
            latent=node.latent,
        )

    for edge in dag.edges:
        G.add_edge(
            edge.from_node,
            edge.to_node,
            id=edge.id,
            edge_type=edge.get_edge_type(),
            allowed_designs=edge.allowed_designs,
        )

    # Store DAG metadata as graph-level attributes
    G.graph["name"] = dag.metadata.name
    G.graph["description"] = dag.metadata.description
    G.graph["target_node"] = dag.metadata.target_node

    return G


def networkx_to_dagspec(G: "nx.DiGraph", metadata: dict[str, Any] | None = None) -> DAGSpec:
    """Convert a NetworkX DiGraph back to a DAGSpec.

    Args:
        G: NetworkX directed graph.
        metadata: Optional metadata dict override. If not provided, uses graph attributes.
    """
    meta = metadata or {}
    dag_metadata = DAGMetadata(
        name=meta.get("name", G.graph.get("name", "converted_dag")),
        description=meta.get("description", G.graph.get("description", "")),
        target_node=meta.get("target_node", G.graph.get("target_node", "")),
    )

    nodes = []
    for node_id, attrs in G.nodes(data=True):
        nodes.append(NodeSpec(
            id=str(node_id),
            name=attrs.get("name", str(node_id)),
            unit=attrs.get("unit", "level"),
            frequency=attrs.get("frequency", "monthly"),
            type=attrs.get("type", "continuous"),
            observed=attrs.get("observed", True),
            latent=attrs.get("latent", False),
        ))

    edges = []
    for i, (u, v, attrs) in enumerate(G.edges(data=True)):
        edge_id = attrs.get("id", f"{u}_to_{v}")
        edges.append(EdgeSpec(
            id=edge_id,
            from_node=str(u),
            to_node=str(v),
            edge_type=attrs.get("edge_type", "causal"),
            allowed_designs=attrs.get("allowed_designs", ["PANEL_FE_BACKDOOR"]),
        ))

    return DAGSpec(metadata=dag_metadata, nodes=nodes, edges=edges)


def dagspec_to_dowhy_graph(dag: DAGSpec) -> str:
    """Convert a DAGSpec to a GML string for DoWhy's CausalModel(graph=...).

    DoWhy expects a GML-formatted string describing the causal graph.
    """
    G = dagspec_to_networkx(dag)

    # DoWhy wants a clean graph with only node names
    import networkx as nx

    clean = nx.DiGraph()
    for node_id in G.nodes:
        clean.add_node(node_id)
    for u, v in G.edges:
        clean.add_edge(u, v)

    # Generate GML string
    lines = ["graph [", "  directed 1"]
    for node_id in clean.nodes:
        lines.append(f'  node [ id "{node_id}" label "{node_id}" ]')
    for u, v in clean.edges:
        lines.append(f'  edge [ source "{u}" target "{v}" ]')
    lines.append("]")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# pywhy-graphs extensions (Phase 4C)
# ──────────────────────────────────────────────────────────────────────


def dagspec_to_pywhy(dag: DAGSpec) -> Any:
    """Convert a DAGSpec to a pywhy-graphs ADMG.

    Bidirected edges are added for latent confounders:
    if latent L affects [X, Y], a bidirected edge X <-> Y is created.
    """
    try:
        from pywhy_graphs import ADMG
    except ImportError:
        raise ImportError(
            "pywhy-graphs is required for ADMG conversion. "
            "Install with: pip install pywhy-graphs"
        )

    import networkx as nx

    # Directed edges
    directed = nx.DiGraph()
    for node in dag.nodes:
        if not node.latent:
            directed.add_node(node.id)
    for edge in dag.edges:
        directed.add_edge(edge.from_node, edge.to_node)

    # Bidirected edges from latent confounders
    bidirected = nx.Graph()
    for latent in dag.latents:
        affected = latent.affects
        for i in range(len(affected)):
            for j in range(i + 1, len(affected)):
                bidirected.add_edge(affected[i], affected[j])

    admg = ADMG(
        incoming_directed_edges=directed,
        incoming_bidirected_edges=bidirected,
    )
    return admg


def pywhy_to_dagspec(admg: Any, name: str = "pywhy_dag") -> DAGSpec:
    """Convert a pywhy-graphs ADMG back to a DAGSpec.

    Bidirected edges are converted to latent confounders.
    """
    metadata = DAGMetadata(name=name)

    # Extract nodes
    node_ids = set()
    for u, v in admg.directed_edges:
        node_ids.add(u)
        node_ids.add(v)
    for u, v in admg.bidirected_edges:
        node_ids.add(u)
        node_ids.add(v)

    nodes = [NodeSpec(id=nid, name=nid) for nid in sorted(node_ids)]

    # Directed edges
    edges = []
    for u, v in admg.directed_edges:
        edges.append(EdgeSpec(
            id=f"{u}_to_{v}",
            from_node=u,
            to_node=v,
        ))

    # Bidirected edges → latent confounders
    latents = []
    for i, (u, v) in enumerate(admg.bidirected_edges):
        latents.append(LatentSpec(
            id=f"U_{u}_{v}",
            description=f"Unobserved confounder between {u} and {v}",
            affects=[u, v],
        ))

    return DAGSpec(metadata=metadata, nodes=nodes, edges=edges, latents=latents)


def causallearn_to_networkx(cg: Any) -> "nx.DiGraph":
    """Convert a causal-learn CausalGraph to a NetworkX DiGraph.

    This bridges discovery algorithm output to our DAGSpec pipeline.
    Undirected edges are dropped (only directed edges are kept).
    """
    import networkx as nx

    G = nx.DiGraph()
    # causal-learn: cg.G is a GeneralGraph with .graph (numpy adjacency matrix)
    graph = cg.G.graph if hasattr(cg.G, "graph") else cg.G
    node_names = (
        cg.get_node_names()
        if hasattr(cg, "get_node_names")
        else [f"X{i}" for i in range(graph.shape[0])]
    )

    for name in node_names:
        G.add_node(name)

    n = graph.shape[0]
    for i in range(n):
        for j in range(n):
            # In causal-learn: graph[i,j] = -1 and graph[j,i] = 1 means i -> j
            if graph[i, j] == -1 and graph[j, i] == 1:
                G.add_edge(node_names[i], node_names[j])

    return G
