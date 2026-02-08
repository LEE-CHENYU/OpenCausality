#!/usr/bin/env python3
"""
Build DAG Visualization HTML.

Reads a DAG YAML, optional edge cards, and optional state.json,
then generates a self-contained D3.js interactive graph.

Usage:
    python scripts/build_dag_viz.py config/agentic/dags/kspi_k2_full.yaml
    python scripts/build_dag_viz.py config/agentic/dags/kspi_k2_full.yaml --cards outputs/agentic/cards/edge_cards/ -o /tmp/viz.html
    python scripts/build_dag_viz.py config/agentic/dags/kspi_k2_full.yaml --llm-annotate
"""

from __future__ import annotations

import argparse
import html as _html
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Default paths ────────────────────────────────────────────────────────────

DEFAULT_CARDS_DIR = PROJECT_ROOT / "outputs" / "agentic" / "cards" / "edge_cards"
DEFAULT_STATE = PROJECT_ROOT / "outputs" / "agentic" / "issues" / "state.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "agentic" / "dag_visualization.html"

# ── Edge type styling ────────────────────────────────────────────────────────

EDGE_TYPE_COLORS = {
    "causal": "#2563eb",
    "reaction_function": "#059669",
    "bridge": "#9333ea",
    "identity": "#9333ea",
    "mechanical": "#9333ea",
    "immutable": "#d97706",
}

EDGE_TYPE_LABELS = {
    "causal": "Causal",
    "reaction_function": "Reaction Function",
    "bridge": "Accounting Bridge",
    "identity": "Identity",
    "mechanical": "Mechanical",
    "immutable": "Immutable (Validated)",
}


# ── Data loading ─────────────────────────────────────────────────────────────


def load_dag(dag_path: Path) -> dict:
    """Load DAG YAML and return raw dict."""
    with open(dag_path) as f:
        return yaml.safe_load(f) or {}


def load_edge_card(card_path: Path) -> dict | None:
    """Load edge card YAML and extract viz-relevant fields."""
    if not card_path.exists():
        return None
    with open(card_path) as f:
        card = yaml.safe_load(f)
    if not card:
        return None

    estimates = card.get("estimates", {})
    diagnostics_raw = card.get("diagnostics", {})
    identification = card.get("identification", {})
    spec = card.get("spec_details", {})
    failure_flags_raw = card.get("failure_flags", {})
    cf = card.get("counterfactual_block") or {}

    diagnostics = []
    for diag in diagnostics_raw.values():
        if isinstance(diag, dict):
            diagnostics.append({
                "name": diag.get("name", ""),
                "passed": bool(diag.get("passed", False)),
            })

    failure_flags = [k for k, v in failure_flags_raw.items() if v]

    return {
        "point": estimates.get("point"),
        "se": estimates.get("se"),
        "pvalue": estimates.get("pvalue"),
        "design": spec.get("design"),
        "rating": card.get("credibility_rating"),
        "score": card.get("credibility_score"),
        "claim_level": identification.get("claim_level"),
        "all_pass": bool(card.get("all_diagnostics_pass", False)),
        "diagnostics": diagnostics,
        "failure_flags": failure_flags,
        "id_risks": identification.get("risks", {}),
        "cf_shock_allowed": cf.get("shock_scenario_allowed"),
        "cf_policy_allowed": cf.get("policy_intervention_allowed"),
        "cf_reason_blocked": cf.get("reason_blocked", ""),
    }


def load_issues(state_path: Path) -> dict[str, list[dict]]:
    """Load open issues grouped by edge_id."""
    if not state_path.exists():
        return {}
    with open(state_path) as f:
        state = json.load(f)

    by_edge: dict[str, list[dict]] = {}
    for key, issue in state.get("issues", {}).items():
        if issue.get("status") != "OPEN":
            continue
        parts = key.split(":", 1)
        edge_id = parts[1] if len(parts) > 1 else key
        by_edge.setdefault(edge_id, []).append({
            "rule_id": issue.get("rule_id", ""),
            "severity": issue.get("severity", ""),
            "message": issue.get("message", ""),
        })
    return by_edge


def build_nodes_data(dag: dict) -> list[dict]:
    """Build node data for D3."""
    nodes = []
    for i, node in enumerate(dag.get("nodes", [])):
        nid = node.get("id", "")
        nodes.append({
            "id": nid,
            "label": _html.escape(node.get("name", nid)),
            "description": _html.escape(node.get("description", "")),
            "unit": _html.escape(node.get("unit", "")),
            "frequency": node.get("frequency", ""),
            "observed": node.get("observed", True),
            "latent": node.get("latent", False),
        })
    return nodes


def build_edges_data(
    dag: dict,
    cards_dir: Path | None,
    issues: dict[str, list[dict]],
    annotations: dict[str, str] | None = None,
) -> list[dict]:
    """Build edge data for D3, enriched with edge card info."""
    edges = []
    for edge in dag.get("edges", []):
        eid = edge.get("id", "")
        from_node = edge.get("from", "")
        to_node = edge.get("to", "")

        # Determine edge type
        edge_type = edge.get("edge_type", "")
        if not edge_type:
            status = edge.get("edge_status", "")
            type_map = {
                "IMMUTABLE": "immutable",
                "IDENTITY": "identity",
                "ESTIMABLE_REDUCED_FORM": "causal",
                "NEEDS_CONNECTOR": "causal",
            }
            edge_type = type_map.get(status, "causal")

        interp = edge.get("interpretation") or {}
        plaus = (edge.get("acceptance_criteria") or {}).get("plausibility") or {}

        edge_data: dict[str, Any] = {
            "id": eid,
            "source": from_node,
            "target": to_node,
            "edge_type": edge_type,
            "expected_sign": plaus.get("expected_sign", ""),
            "interpretation": _html.escape(str(interp.get("is", ""))),
            "notes": _html.escape(str(edge.get("notes", ""))),
        }

        # Enrich with edge card data
        if cards_dir:
            card = load_edge_card(cards_dir / f"{eid}.yaml")
            if card:
                edge_data.update(card)

        # Enrich with issues
        edge_issues = issues.get(eid, [])
        edge_data["issues"] = edge_issues
        edge_data["has_critical_issue"] = any(
            i.get("severity") == "CRITICAL" for i in edge_issues
        )

        # LLM annotation
        if annotations and eid in annotations:
            edge_data["llm_annotation"] = _html.escape(annotations[eid])

        edges.append(edge_data)
    return edges


def build_latents_data(dag: dict) -> list[dict]:
    """Build latent confounder data."""
    latents = []
    for lat in dag.get("latents", []):
        latents.append({
            "id": lat.get("id", ""),
            "description": _html.escape(lat.get("description", "")),
            "affects": lat.get("affects", []),
        })
    return latents


# ── LLM annotations ─────────────────────────────────────────────────────────


def generate_llm_annotations(
    dag: dict, cards_dir: Path | None,
) -> dict[str, str]:
    """Generate LLM annotations for each edge."""
    try:
        from shared.llm.client import get_llm_client
        from shared.llm.prompts import EDGE_ANNOTATION_SYSTEM
    except ImportError:
        print("  Warning: LLM client not available, skipping annotations")
        return {}

    client = get_llm_client()
    annotations: dict[str, str] = {}

    for edge in dag.get("edges", []):
        eid = edge.get("id", "")
        from_node = edge.get("from", "")
        to_node = edge.get("to", "")
        edge_type = edge.get("edge_type", "causal")

        card_context = ""
        if cards_dir:
            card = load_edge_card(cards_dir / f"{eid}.yaml")
            if card:
                card_context = (
                    f"Estimate: {card.get('point')}, SE: {card.get('se')}, "
                    f"p-value: {card.get('pvalue')}, "
                    f"Claim level: {card.get('claim_level')}, "
                    f"Rating: {card.get('rating')}"
                )

        user_msg = (
            f"Edge: {from_node} -> {to_node} (id: {eid})\n"
            f"Type: {edge_type}\n"
            f"Notes: {edge.get('notes', '')}\n"
            f"{card_context}"
        )

        try:
            annotation = client.complete(
                system=EDGE_ANNOTATION_SYSTEM,
                user=user_msg,
                max_tokens=200,
            )
            annotations[eid] = annotation.strip()
            print(f"  Annotated: {eid}")
        except Exception as e:
            print(f"  Warning: LLM annotation failed for {eid}: {e}")

    return annotations


# ── HTML template ────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>__DAG_TITLE__</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background: #fff;
            color: #000;
        }
        #graph { width: 100vw; height: 100vh; }
        .node circle {
            fill: #fff;
            stroke: #000;
            stroke-width: 1.5px;
            cursor: grab;
        }
        .node circle:hover { stroke-width: 2.5px; }
        .node circle.latent { stroke-dasharray: 4, 3; }
        .node circle.id-risk { stroke: #c00; stroke-width: 2.5px; }
        .node text {
            font-size: 9px;
            font-weight: 400;
            pointer-events: none;
            text-anchor: middle;
        }
        .link {
            fill: none;
            stroke-opacity: 0.7;
        }
        .link:hover { stroke-opacity: 1; stroke-width: 3px !important; }
        .link.null { stroke-dasharray: 4, 3; stroke-opacity: 0.4; }
        .link.critical-issue { stroke: #f59e0b; stroke-width: 3px; }
        .label-bg {
            fill: white;
            stroke: none;
        }
        .link-label {
            font-size: 9px;
            font-weight: 500;
            fill: #333;
            pointer-events: none;
            text-anchor: middle;
            dominant-baseline: middle;
        }
        .tooltip {
            position: absolute;
            background: #fff;
            border: 1px solid #000;
            padding: 12px 16px;
            font-size: 11px;
            pointer-events: none;
            opacity: 0;
            max-width: 380px;
            line-height: 1.5;
            z-index: 1000;
        }
        .tooltip.visible { opacity: 1; }
        .tooltip h3 { font-size: 12px; font-weight: 600; margin-bottom: 8px; border-bottom: 1px solid #000; padding-bottom: 6px; }
        .tooltip-row { display: flex; justify-content: space-between; margin: 3px 0; gap: 12px; }
        .tooltip-label { color: #666; white-space: nowrap; }
        .tooltip-value { font-weight: 500; font-variant-numeric: tabular-nums; text-align: right; }
        .tooltip-value.null { color: #999; }
        .tooltip-annotation { margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; color: #444; font-style: italic; }
        .tooltip-issues { margin-top: 6px; padding-top: 6px; border-top: 1px solid #eee; }
        .tooltip-issue { font-size: 10px; margin: 2px 0; }
        .tooltip-issue.critical { color: #c00; font-weight: 600; }
        .tooltip-issue.high { color: #a16207; }
        .controls {
            position: fixed;
            top: 20px;
            left: 20px;
            font-size: 11px;
            background: rgba(255,255,255,0.95);
            padding: 16px;
            border: 1px solid #000;
            z-index: 100;
            max-width: 220px;
        }
        .controls h1 { font-size: 14px; font-weight: 600; margin-bottom: 4px; }
        .controls .subtitle { color: #666; margin-bottom: 12px; font-size: 10px; }
        .control-row { margin: 8px 0; }
        .control-row label { color: #666; margin-right: 8px; }
        .control-row select {
            border: 1px solid #000;
            background: #fff;
            padding: 4px 8px;
            font-size: 11px;
            font-family: inherit;
        }
        .legend { margin-top: 12px; padding-top: 12px; border-top: 1px solid #ddd; }
        .legend-title { font-weight: 600; font-size: 10px; margin-bottom: 6px; }
        .legend-item { display: flex; align-items: center; margin: 4px 0; font-size: 10px; }
        .legend-line { width: 20px; height: 2px; margin-right: 8px; }
        .legend-line.dashed { border-top: 2px dashed; height: 0; }
        .stats {
            position: fixed;
            bottom: 20px;
            left: 20px;
            font-size: 10px;
            color: #666;
        }
        .stats span { margin-right: 16px; }
        .stats strong { color: #000; }
        /* Pitfall sidebar */
        .pitfall-sidebar {
            position: fixed;
            top: 20px;
            right: 20px;
            width: 280px;
            max-height: calc(100vh - 40px);
            overflow-y: auto;
            background: rgba(255,255,255,0.95);
            border: 1px solid #000;
            padding: 16px;
            font-size: 11px;
            z-index: 100;
        }
        .pitfall-sidebar h2 { font-size: 13px; font-weight: 600; margin-bottom: 8px; }
        .pitfall-item { padding: 6px 0; border-bottom: 1px solid #eee; }
        .pitfall-item:last-child { border-bottom: none; }
        .pitfall-edge { font-weight: 600; font-size: 10px; }
        .pitfall-severity { font-size: 9px; font-weight: 600; padding: 1px 5px; border: 1px solid; display: inline-block; margin-left: 4px; }
        .pitfall-severity.CRITICAL { background: #000; color: #fff; border-color: #000; }
        .pitfall-severity.HIGH { background: #fff; color: #000; border-color: #000; }
        .pitfall-severity.MEDIUM { background: #eee; color: #000; border-color: #ccc; }
        .pitfall-severity.LOW { background: #fff; color: #999; border-color: #ccc; }
        .pitfall-msg { font-size: 10px; color: #555; margin-top: 2px; }
        .pitfall-empty { color: #999; font-style: italic; }
    </style>
</head>
<body>
    <svg id="graph"></svg>
    <div class="controls">
        <h1 id="dagTitle">__DAG_TITLE__</h1>
        <div class="subtitle" id="dagSubtitle"></div>
        <div class="control-row">
            <label>Edge type</label>
            <select id="typeFilter">
                <option value="all">All</option>
            </select>
        </div>
        <div class="control-row">
            <label>Rating</label>
            <select id="ratingFilter">
                <option value="all">All</option>
                <option value="A">A</option>
                <option value="B">B</option>
                <option value="C">C</option>
                <option value="D">D</option>
            </select>
        </div>
        <div class="control-row">
            <label><input type="checkbox" id="showNull" checked> Show null</label>
        </div>
        <div class="control-row">
            <label><input type="checkbox" id="showLabels" checked> Show labels</label>
        </div>
        <div class="legend" id="legendContainer">
            <div class="legend-title">Edge Types</div>
        </div>
    </div>
    <div class="stats">
        <span>Edges: <strong id="statEdges">0</strong></span>
        <span>Nodes: <strong id="statNodes">0</strong></span>
        <span>A: <strong id="statA">0</strong></span>
        <span>B: <strong id="statB">0</strong></span>
        <span>C: <strong id="statC">0</strong></span>
        <span>Issues: <strong id="statIssues">0</strong></span>
    </div>
    <div class="pitfall-sidebar" id="pitfallSidebar">
        <h2>Open Issues</h2>
        <div id="pitfallList"></div>
    </div>
    <div class="tooltip" id="tooltip"></div>

    <script>
// ── INJECTED DATA ────────────────────────────────────────────────────────
const NODES = __NODES_JSON__;
const EDGES = __EDGES_JSON__;
const LATENTS = __LATENTS_JSON__;
const METADATA = __METADATA_JSON__;
const EDGE_TYPE_COLORS = __EDGE_TYPE_COLORS__;
const EDGE_TYPE_LABELS = __EDGE_TYPE_LABELS__;

// ── Setup ────────────────────────────────────────────────────────────────
document.getElementById('dagSubtitle').textContent =
    NODES.length + ' nodes, ' + EDGES.length + ' edges';
document.getElementById('statNodes').textContent = NODES.length;

// Compute node layers via topological distance from roots
var inDegree = {};
NODES.forEach(function(n) { inDegree[n.id] = 0; });
EDGES.forEach(function(e) { inDegree[e.target] = (inDegree[e.target] || 0) + 1; });

var layers = {};
var visited = {};
function computeLayer(nodeId) {
    if (visited[nodeId]) return layers[nodeId] || 0;
    visited[nodeId] = true;
    var incoming = EDGES.filter(function(e) { return e.target === nodeId; });
    if (incoming.length === 0) { layers[nodeId] = 0; return 0; }
    var maxParent = 0;
    incoming.forEach(function(e) {
        var pLayer = computeLayer(e.source);
        if (pLayer + 1 > maxParent) maxParent = pLayer + 1;
    });
    layers[nodeId] = maxParent;
    return maxParent;
}
NODES.forEach(function(n) { computeLayer(n.id); });
NODES.forEach(function(n) { n.layer = layers[n.id] || 0; });

// Populate type filter
var typeSet = {};
EDGES.forEach(function(e) { typeSet[e.edge_type || 'causal'] = true; });
var typeFilter = document.getElementById('typeFilter');
Object.keys(typeSet).sort().forEach(function(t) {
    var opt = document.createElement('option');
    opt.value = t;
    opt.textContent = EDGE_TYPE_LABELS[t] || t;
    typeFilter.appendChild(opt);
});

// Build legend
var legendContainer = document.getElementById('legendContainer');
Object.keys(typeSet).sort().forEach(function(t) {
    var color = EDGE_TYPE_COLORS[t] || '#2563eb';
    var div = document.createElement('div');
    div.className = 'legend-item';
    div.innerHTML = '<div class="legend-line" style="background:' + color + '"></div>' +
        (EDGE_TYPE_LABELS[t] || t);
    legendContainer.appendChild(div);
});
// Null/insig legend
var nullLeg = document.createElement('div');
nullLeg.className = 'legend-item';
nullLeg.innerHTML = '<div class="legend-line dashed" style="border-color:#999"></div>Null / Insig.';
legendContainer.appendChild(nullLeg);

// Build pitfall sidebar
var pitfallList = document.getElementById('pitfallList');
var SEVERITY_ORDER = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3};
var edgesWithIssues = EDGES.filter(function(e) { return e.issues && e.issues.length > 0; });
edgesWithIssues.sort(function(a, b) {
    var aMax = Math.min.apply(null, a.issues.map(function(i) { return SEVERITY_ORDER[i.severity] || 99; }));
    var bMax = Math.min.apply(null, b.issues.map(function(i) { return SEVERITY_ORDER[i.severity] || 99; }));
    return aMax - bMax;
});
if (edgesWithIssues.length === 0) {
    pitfallList.innerHTML = '<div class="pitfall-empty">No open issues.</div>';
} else {
    edgesWithIssues.forEach(function(e) {
        e.issues.forEach(function(issue) {
            var div = document.createElement('div');
            div.className = 'pitfall-item';
            div.innerHTML = '<div><span class="pitfall-edge">' + e.id + '</span>' +
                '<span class="pitfall-severity ' + issue.severity + '">' + issue.severity + '</span></div>' +
                '<div class="pitfall-msg">' + escHtml(issue.message) + '</div>';
            pitfallList.appendChild(div);
        });
    });
}

function escHtml(s) {
    if (!s) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Helpers ──────────────────────────────────────────────────────────────
function formatBeta(beta) {
    if (beta === null || beta === undefined) return '--';
    if (beta === 0) return '0';
    var abs = Math.abs(beta);
    if (abs >= 10000) return (beta / 1000).toFixed(0) + 'k';
    if (abs >= 1000) return (beta / 1000).toFixed(1) + 'k';
    if (abs >= 100) return beta.toFixed(0);
    if (abs >= 10) return beta.toFixed(1);
    if (abs >= 1) return beta.toFixed(1);
    if (abs >= 0.01) return beta.toFixed(2);
    if (abs >= 0.001) return beta.toFixed(3);
    return beta.toExponential(1);
}
function fmtP(p) {
    if (p === null || p === undefined) return '--';
    if (p < 0.0001) return p.toExponential(2);
    return p.toFixed(4);
}

function getEdgeColor(d) {
    var isNull = d.pvalue !== null && d.pvalue !== undefined && d.pvalue >= 0.05;
    if (isNull) return '#999';
    return EDGE_TYPE_COLORS[d.edge_type] || '#2563eb';
}
function isEdgeNull(d) {
    return d.pvalue !== null && d.pvalue !== undefined && d.pvalue >= 0.05;
}

// ── D3 Graph ─────────────────────────────────────────────────────────────
var svg = d3.select('#graph');
var width = window.innerWidth;
var height = window.innerHeight;
svg.attr('width', width).attr('height', height);

// Arrow markers
var defs = svg.append('defs');
var defaultColors = Object.values(EDGE_TYPE_COLORS).concat(['#999']);
var colorSet = {};
defaultColors.forEach(function(c) { colorSet[c] = true; });
Object.keys(colorSet).forEach(function(color) {
    defs.append('marker')
        .attr('id', 'arrow-' + color.replace('#', ''))
        .attr('viewBox', '0 -3 6 6')
        .attr('refX', 18)
        .attr('refY', 0)
        .attr('markerWidth', 5)
        .attr('markerHeight', 5)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-3L6,0L0,3')
        .attr('fill', color);
});

var g = svg.append('g');
svg.call(d3.zoom().scaleExtent([0.3, 3]).on('zoom', function(e) { g.attr('transform', e.transform); }));

var simulation = d3.forceSimulation(NODES)
    .force('link', d3.forceLink(EDGES).id(function(d) { return d.id; }).distance(120).strength(0.3))
    .force('charge', d3.forceManyBody().strength(-400))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('y', d3.forceY(function(d) { return 80 + d.layer * 90; }).strength(0.7))
    .force('x', d3.forceX(width / 2).strength(0.02))
    .force('collision', d3.forceCollide(40));

var linkGroup = g.append('g');
var links = linkGroup.selectAll('path')
    .data(EDGES)
    .enter()
    .append('path')
    .attr('class', function(d) { return 'link' + (isEdgeNull(d) ? ' null' : '') + (d.has_critical_issue ? ' critical-issue' : ''); })
    .attr('stroke', function(d) { return getEdgeColor(d); })
    .attr('stroke-width', function(d) { return d.pvalue !== null && d.pvalue < 0.05 ? 2.5 : 1.5; })
    .attr('marker-end', function(d) { return 'url(#arrow-' + getEdgeColor(d).replace('#', '') + ')'; })
    .on('mouseover', showEdgeTooltip)
    .on('mouseout', hideTooltip);

var labelGroup = g.append('g');
var labelBgs = labelGroup.selectAll('rect')
    .data(EDGES)
    .enter()
    .append('rect')
    .attr('class', 'label-bg')
    .attr('rx', 2)
    .attr('ry', 2);

var linkLabels = labelGroup.selectAll('text')
    .data(EDGES)
    .enter()
    .append('text')
    .attr('class', 'link-label')
    .text(function(d) { return formatBeta(d.point); });

var nodeGroup = g.append('g');
var node = nodeGroup.selectAll('g')
    .data(NODES)
    .enter()
    .append('g')
    .attr('class', 'node')
    .call(d3.drag()
        .on('start', function(e, d) { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', function(e, d) { d.fx = e.x; d.fy = e.y; })
        .on('end', function(e, d) { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }))
    .on('mouseover', showNodeTooltip)
    .on('mouseout', hideTooltip);

// Check if node has ID risks from any edge
var nodeIdRisks = {};
EDGES.forEach(function(e) {
    if (e.id_risks && Object.keys(e.id_risks).length > 0) {
        nodeIdRisks[e.source] = true;
        nodeIdRisks[e.target] = true;
    }
});

node.append('circle')
    .attr('r', 18)
    .attr('class', function(d) {
        var cls = '';
        if (d.latent) cls += ' latent';
        if (nodeIdRisks[d.id]) cls += ' id-risk';
        return cls;
    });
node.append('text').attr('dy', 4).text(function(d) { return d.label; });

var tooltip = d3.select('#tooltip');

function showEdgeTooltip(e, d) {
    var html = '<h3>' + d.source.id.replace(/_/g, ' ') + ' &rarr; ' + d.target.id.replace(/_/g, ' ') + '</h3>';
    html += '<div class="tooltip-row"><span class="tooltip-label">Edge ID</span><span class="tooltip-value">' + d.id + '</span></div>';
    html += '<div class="tooltip-row"><span class="tooltip-label">Type</span><span class="tooltip-value">' + (EDGE_TYPE_LABELS[d.edge_type] || d.edge_type) + '</span></div>';
    if (d.point !== null && d.point !== undefined) {
        html += '<div class="tooltip-row"><span class="tooltip-label">&beta;</span><span class="tooltip-value">' + d.point.toFixed(4) + '</span></div>';
    }
    if (d.se !== null && d.se !== undefined) {
        html += '<div class="tooltip-row"><span class="tooltip-label">SE</span><span class="tooltip-value">' + d.se.toFixed(4) + '</span></div>';
    }
    if (d.pvalue !== null && d.pvalue !== undefined) {
        html += '<div class="tooltip-row"><span class="tooltip-label">p-value</span><span class="tooltip-value ' + (isEdgeNull(d) ? 'null' : '') + '">' + fmtP(d.pvalue) + '</span></div>';
    }
    if (d.claim_level) html += '<div class="tooltip-row"><span class="tooltip-label">Claim</span><span class="tooltip-value">' + d.claim_level + '</span></div>';
    if (d.rating) html += '<div class="tooltip-row"><span class="tooltip-label">Rating</span><span class="tooltip-value">' + d.rating + '</span></div>';
    if (d.design) html += '<div class="tooltip-row"><span class="tooltip-label">Design</span><span class="tooltip-value">' + d.design + '</span></div>';
    if (d.diagnostics && d.diagnostics.length > 0) {
        var passed = d.diagnostics.filter(function(x) { return x.passed; }).length;
        html += '<div class="tooltip-row"><span class="tooltip-label">Diagnostics</span><span class="tooltip-value">' + passed + '/' + d.diagnostics.length + ' pass</span></div>';
    }
    if (d.interpretation) html += '<div class="tooltip-row"><span class="tooltip-label">Interpretation</span><span class="tooltip-value" style="text-align:left;font-weight:400">' + d.interpretation + '</span></div>';
    // LLM annotation
    if (d.llm_annotation) {
        html += '<div class="tooltip-annotation">' + d.llm_annotation + '</div>';
    }
    // Issues
    if (d.issues && d.issues.length > 0) {
        html += '<div class="tooltip-issues">';
        d.issues.forEach(function(issue) {
            html += '<div class="tooltip-issue ' + (issue.severity || '').toLowerCase() + '">[' + issue.severity + '] ' + escHtml(issue.message) + '</div>';
        });
        html += '</div>';
    }
    tooltip.html(html).style('left', (e.pageX + 15) + 'px').style('top', (e.pageY - 15) + 'px').classed('visible', true);
}

function showNodeTooltip(e, d) {
    var html = '<h3>' + d.label + '</h3>';
    html += '<div class="tooltip-row"><span class="tooltip-label">ID</span><span class="tooltip-value">' + d.id + '</span></div>';
    if (d.description) html += '<div class="tooltip-row"><span class="tooltip-label">Description</span><span class="tooltip-value" style="text-align:left;font-weight:400">' + d.description + '</span></div>';
    if (d.unit) html += '<div class="tooltip-row"><span class="tooltip-label">Unit</span><span class="tooltip-value">' + d.unit + '</span></div>';
    if (d.frequency) html += '<div class="tooltip-row"><span class="tooltip-label">Frequency</span><span class="tooltip-value">' + d.frequency + '</span></div>';
    if (d.latent) html += '<div class="tooltip-row"><span class="tooltip-label">Observed</span><span class="tooltip-value null">Latent (unobserved)</span></div>';
    tooltip.html(html).style('left', (e.pageX + 15) + 'px').style('top', (e.pageY - 15) + 'px').classed('visible', true);
}

function hideTooltip() { tooltip.classed('visible', false); }

function getLabelPos(d) {
    var dx = d.target.x - d.source.x;
    var dy = d.target.y - d.source.y;
    var dist = Math.sqrt(dx*dx + dy*dy) || 1;
    var t = 0.4;
    var mx = d.source.x + dx * t;
    var my = d.source.y + dy * t;
    var offset = 12;
    var px = -dy / dist * offset;
    var py = dx / dist * offset;
    return { x: mx + px, y: my + py };
}

simulation.on('tick', function() {
    links.attr('d', function(d) {
        var dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
        var dr = Math.sqrt(dx * dx + dy * dy) * 1.8;
        return 'M' + d.source.x + ',' + d.source.y + 'A' + dr + ',' + dr + ' 0 0,1 ' + d.target.x + ',' + d.target.y;
    });
    linkLabels.each(function(d) {
        var pos = getLabelPos(d);
        d3.select(this).attr('x', pos.x).attr('y', pos.y);
    });
    labelBgs.each(function(d, i) {
        var pos = getLabelPos(d);
        var textNode = linkLabels.nodes()[i];
        var bbox = textNode.getBBox();
        d3.select(this)
            .attr('x', pos.x - bbox.width/2 - 2)
            .attr('y', pos.y - bbox.height/2 - 1)
            .attr('width', bbox.width + 4)
            .attr('height', bbox.height + 2);
    });
    node.attr('transform', function(d) { return 'translate(' + d.x + ',' + d.y + ')'; });
});

// ── Filters ──────────────────────────────────────────────────────────────
function applyFilters() {
    var typeVal = document.getElementById('typeFilter').value;
    var ratVal = document.getElementById('ratingFilter').value;
    var showNullVal = document.getElementById('showNull').checked;
    var showLabelsVal = document.getElementById('showLabels').checked;

    var isVisible = function(d) {
        if (typeVal !== 'all' && d.edge_type !== typeVal) return false;
        if (ratVal !== 'all' && d.rating !== ratVal) return false;
        if (!showNullVal && isEdgeNull(d)) return false;
        return true;
    };

    links.style('display', function(d) { return isVisible(d) ? null : 'none'; });
    linkLabels.style('display', function(d) { return (isVisible(d) && showLabelsVal) ? null : 'none'; });
    labelBgs.style('display', function(d) { return (isVisible(d) && showLabelsVal) ? null : 'none'; });

    var visible = EDGES.filter(isVisible);
    document.getElementById('statEdges').textContent = visible.length;
    document.getElementById('statA').textContent = visible.filter(function(e) { return e.rating === 'A'; }).length;
    document.getElementById('statB').textContent = visible.filter(function(e) { return e.rating === 'B'; }).length;
    document.getElementById('statC').textContent = visible.filter(function(e) { return e.rating === 'C'; }).length;
    var totalIssues = visible.reduce(function(s, e) { return s + (e.issues ? e.issues.length : 0); }, 0);
    document.getElementById('statIssues').textContent = totalIssues;
}

document.getElementById('typeFilter').addEventListener('change', applyFilters);
document.getElementById('ratingFilter').addEventListener('change', applyFilters);
document.getElementById('showNull').addEventListener('change', applyFilters);
document.getElementById('showLabels').addEventListener('change', applyFilters);

applyFilters();
    </script>
</body>
</html>"""


# ── Build ────────────────────────────────────────────────────────────────────


def build(
    dag_path: Path,
    cards_dir: Path | None = None,
    state_path: Path | None = None,
    output_path: Path | None = None,
    llm_annotate: bool = False,
) -> Path:
    """Build the DAG visualization HTML."""
    dag = load_dag(dag_path)
    metadata = dag.get("metadata", {})
    title = _html.escape(metadata.get("name", dag_path.stem))

    # Load optional data
    issues = load_issues(state_path) if state_path and state_path.exists() else {}

    # LLM annotations
    annotations = None
    if llm_annotate:
        print("Generating LLM annotations...")
        annotations = generate_llm_annotations(dag, cards_dir)
        print(f"  {len(annotations)} annotations generated")

    # Build data arrays
    nodes = build_nodes_data(dag)
    edges = build_edges_data(dag, cards_dir, issues, annotations)
    latents = build_latents_data(dag)

    print(f"  {len(nodes)} nodes, {len(edges)} edges, {len(latents)} latents")
    issue_count = sum(len(issues.get(e["id"], [])) for e in edges)
    print(f"  {issue_count} open issues across edges")

    # Build HTML
    html = HTML_TEMPLATE
    html = html.replace("__DAG_TITLE__", title)
    html = html.replace("__NODES_JSON__", json.dumps(nodes, indent=2))
    html = html.replace("__EDGES_JSON__", json.dumps(edges, indent=2))
    html = html.replace("__LATENTS_JSON__", json.dumps(latents, indent=2))
    html = html.replace("__METADATA_JSON__", json.dumps(metadata, indent=2))
    html = html.replace("__EDGE_TYPE_COLORS__", json.dumps(EDGE_TYPE_COLORS))
    html = html.replace("__EDGE_TYPE_LABELS__", json.dumps(EDGE_TYPE_LABELS))

    # Write output
    if output_path is None:
        output_path = DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"  Written: {output_path}")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build DAG Visualization HTML from a DAG YAML."
    )
    parser.add_argument(
        "dag_path",
        type=Path,
        help="Path to DAG YAML specification file",
    )
    parser.add_argument(
        "--cards",
        type=Path,
        default=None,
        help="Path to edge_cards directory (default: outputs/agentic/cards/edge_cards)",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Path to state.json for open issues",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output HTML file path",
    )
    parser.add_argument(
        "--llm-annotate",
        action="store_true",
        help="Generate LLM annotations for edge tooltips",
    )
    args = parser.parse_args()

    cards_dir = args.cards or DEFAULT_CARDS_DIR
    state_path = args.state or DEFAULT_STATE

    print(f"Building DAG visualization from {args.dag_path}...")
    build(args.dag_path, cards_dir, state_path, args.output, args.llm_annotate)
    print("Done.")


if __name__ == "__main__":
    main()
