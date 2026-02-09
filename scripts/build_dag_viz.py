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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Default paths ────────────────────────────────────────────────────────────

DEFAULT_CARDS_DIR = PROJECT_ROOT / "outputs" / "agentic" / "cards" / "edge_cards"
DEFAULT_STATE = PROJECT_ROOT / "outputs" / "agentic" / "issues" / "state.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "agentic" / "dag_visualization.html"
DEFAULT_ACTIONS = PROJECT_ROOT / "config" / "agentic" / "hitl_actions.yaml"
DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "agentic" / "issue_registry.yaml"

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

    n_obs = None
    eff_obs = diagnostics_raw.get("effective_obs", {})
    if isinstance(eff_obs, dict) and eff_obs.get("value") is not None:
        n_obs = int(eff_obs["value"])

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
        "n_obs": n_obs,
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
            "issue_key": key,
            "rule_id": issue.get("rule_id", ""),
            "severity": issue.get("severity", ""),
            "message": issue.get("message", ""),
        })
    return by_edge


def load_actions(actions_path: Path) -> dict:
    """Load hitl_actions.yaml with HTML-escaped tooltips."""
    if not actions_path.exists():
        return {}
    with open(actions_path) as f:
        data = yaml.safe_load(f)
    actions = data.get("actions", {})
    for rule_id, action_cfg in actions.items():
        for opt in action_cfg.get("options", []):
            if "tooltip" in opt:
                opt["tooltip"] = _html.escape(opt["tooltip"])
    return actions


def load_rule_info(registry_path: Path) -> dict:
    """Load issue_registry.yaml -> {rule_id: {description, explanation, guidance}}."""
    if not registry_path.exists():
        return {}
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    rules = {}
    for rule in data.get("rules", []):
        rid = rule.get("rule_id")
        if rid:
            rules[rid] = {
                "description": _html.escape(rule.get("description", "")),
                "explanation": _html.escape(rule.get("explanation", "")),
                "guidance": _html.escape(rule.get("guidance", "")),
            }
    return rules


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

        /* Proposal banner */
        .proposal-banner {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 36px;
            background: #000;
            color: #fff;
            font-size: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            z-index: 200;
            letter-spacing: 0.5px;
        }
        .banner-sub {
            font-weight: 400;
            font-size: 10px;
            color: #aaa;
        }

        .controls {
            position: fixed;
            top: 50px;
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
            top: 50px;
            right: 20px;
            width: 340px;
            max-height: calc(100vh - 70px);
            overflow-y: auto;
            background: rgba(255,255,255,0.95);
            border: 1px solid #000;
            font-size: 11px;
            z-index: 100;
            display: flex;
            flex-direction: column;
        }
        .pitfall-header {
            padding: 12px 16px 8px;
            border-bottom: 1px solid #ddd;
        }
        .pitfall-header h2 { font-size: 13px; font-weight: 600; margin-bottom: 6px; }
        .pitfall-progress { margin-bottom: 4px; }
        .pitfall-progress-bar {
            width: 100%;
            height: 6px;
            border: 1px solid #000;
            background: #fff;
        }
        .pitfall-progress-fill {
            height: 100%;
            background: #000;
            transition: width 0.3s;
        }
        .pitfall-progress-label {
            font-size: 10px;
            color: #666;
            margin-top: 2px;
        }
        #pitfallList {
            flex: 1;
            overflow-y: auto;
            padding: 0 16px;
        }
        .pitfall-item {
            border-bottom: 1px solid #eee;
            cursor: pointer;
        }
        .pitfall-item:last-child { border-bottom: none; }
        .pitfall-item:hover { background: #f8f8f8; }
        .pitfall-item.active {
            border-left: 3px solid #2563eb;
            background: #f0f4ff;
        }
        .pitfall-item.resolved {
            opacity: 0.5;
        }
        .pitfall-item.resolved .pitfall-msg { text-decoration: line-through; }
        .pitfall-item-header {
            padding: 8px 0 4px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .pitfall-edge { font-weight: 600; font-size: 10px; }
        .pitfall-severity { font-size: 9px; font-weight: 600; padding: 1px 5px; border: 1px solid; display: inline-block; }
        .pitfall-severity.CRITICAL { background: #000; color: #fff; border-color: #000; }
        .pitfall-severity.HIGH { background: #fff; color: #000; border-color: #000; }
        .pitfall-severity.MEDIUM { background: #eee; color: #000; border-color: #ccc; }
        .pitfall-severity.LOW { background: #fff; color: #999; border-color: #ccc; }
        .pitfall-resolved-check { color: #080; font-weight: 700; font-size: 11px; display: none; }
        .pitfall-item.resolved .pitfall-resolved-check { display: inline; }
        .pitfall-msg { font-size: 10px; color: #555; margin-bottom: 4px; }
        .pitfall-empty { color: #999; font-style: italic; padding: 12px 0; }

        /* Collapsible decision panel */
        .pitfall-decision {
            display: none;
            padding: 8px 0 10px;
            border-top: 1px solid #eee;
            font-size: 10px;
            line-height: 1.6;
        }
        .pitfall-decision.open { display: block; }
        .pitfall-decision .pd-label {
            font-weight: 600;
            color: #444;
            margin-top: 6px;
            margin-bottom: 2px;
        }
        .pitfall-decision .pd-text {
            color: #555;
            margin-bottom: 4px;
        }
        .pitfall-decision .pd-guidance {
            background: #f8f8f8;
            border: 1px solid #ddd;
            padding: 4px 8px;
            margin-bottom: 6px;
        }
        .pitfall-decision select {
            width: 100%;
            border: 1px solid #000;
            background: #fff;
            padding: 3px 6px;
            font-size: 10px;
            font-family: inherit;
        }
        .pitfall-decision select.decided {
            background: #000;
            color: #fff;
        }
        .pitfall-decision .pd-tooltip-area {
            font-size: 9px;
            color: #666;
            min-height: 14px;
            margin-top: 2px;
            padding: 2px 4px;
            background: #fafafa;
            border: 1px solid #eee;
        }
        .pitfall-decision input[type="text"] {
            width: 100%;
            border: 1px solid #ccc;
            padding: 3px 6px;
            font-size: 10px;
            font-family: inherit;
            margin-top: 4px;
        }
        .pitfall-decision input[type="text"]:focus {
            border-color: #000;
            outline: none;
        }
        .pitfall-decision .pd-llm-guidance {
            background: #f0f4ff;
            border-left: 3px solid #2563eb;
            padding: 6px 10px;
            margin-bottom: 6px;
            font-size: 10px;
            line-height: 1.6;
            color: #333;
        }
        .pitfall-decision .pd-llm-guidance-title {
            font-weight: 600;
            color: #1e40af;
            margin-bottom: 2px;
        }
        .pd-static-ref { font-size: 9px; color: #888; margin-bottom: 4px; }
        .pd-static-ref summary { cursor: pointer; font-weight: 500; }
        .pitfall-decision .pd-buttons {
            display: flex;
            gap: 6px;
            margin-top: 6px;
        }
        .pitfall-decision .pd-btn {
            border: 1px solid #000;
            background: #fff;
            padding: 3px 12px;
            font-size: 10px;
            font-family: inherit;
            cursor: pointer;
        }
        .pitfall-decision .pd-btn:hover { background: #000; color: #fff; }
        .pitfall-decision .pd-btn.pd-btn-confirm { background: #000; color: #fff; }
        .pitfall-decision .pd-btn.pd-btn-confirm:hover { background: #333; }

        /* Sidebar footer */
        .pitfall-footer {
            padding: 10px 16px;
            border-top: 1px solid #000;
        }
        .pitfall-footer label {
            font-size: 10px;
            color: #666;
            display: block;
            margin-bottom: 4px;
        }
        .pitfall-footer input[type="text"] {
            width: 100%;
            border: 1px solid #000;
            padding: 4px 8px;
            font-size: 10px;
            font-family: inherit;
            margin-bottom: 6px;
        }
        .pitfall-footer .btn-export {
            width: 100%;
            border: 1px solid #000;
            background: #000;
            color: #fff;
            padding: 5px 12px;
            font-size: 11px;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
        }
        .pitfall-footer .btn-export:hover { background: #333; }

        /* Clean edge entries (no issues) */
        .pitfall-divider {
            padding: 8px 0 4px;
            font-size: 10px;
            font-weight: 600;
            color: #999;
            border-top: 1px solid #ddd;
            margin-top: 4px;
        }
        .pitfall-item.clean-edge .pitfall-rating {
            font-size: 9px;
            font-weight: 600;
            padding: 1px 5px;
            border: 1px solid #059669;
            color: #059669;
            display: inline-block;
        }
        .pitfall-item.clean-edge .pitfall-stats {
            font-size: 9px;
            color: #888;
            margin-bottom: 4px;
        }

        /* Edge highlight animation */
        @keyframes edgeFlash {
            0%, 100% { stroke-opacity: 1; }
            50% { stroke-opacity: 0.3; }
        }
        .link.highlighted {
            stroke: #2563eb !important;
            stroke-width: 4px !important;
            stroke-opacity: 1;
            animation: edgeFlash 1s ease-in-out 3;
        }
        .link.edge-resolved {
            stroke-dasharray: 6, 4;
            stroke-opacity: 0.3;
            stroke-width: 1px !important;
        }
    </style>
</head>
<body>
    <div class="proposal-banner">
        DRAFT PROPOSAL &mdash; Requires analyst review before use
        <span class="banner-sub">Resolve flagged issues or export decisions to proceed</span>
    </div>
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
        <div class="pitfall-header">
            <h2>Open Issues</h2>
            <div class="pitfall-progress">
                <div class="pitfall-progress-bar"><div class="pitfall-progress-fill" id="pitfallProgressFill"></div></div>
                <div class="pitfall-progress-label" id="pitfallProgressLabel">0 / 0 resolved</div>
            </div>
        </div>
        <div id="pitfallList"></div>
        <div class="pitfall-footer">
            <label>Analyst ID</label>
            <input type="text" id="analystId" placeholder="Your name or initials">
            <button class="btn-export" id="exportBtn">Export Decisions</button>
        </div>
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
const RULE_DESCRIPTIONS = __RULES_JSON__;
const ACTIONS_BY_RULE = __ACTIONS_JSON__;
const GENERATED_AT = "__GENERATED_AT__";
const ISSUE_GUIDANCE = __ISSUE_GUIDANCE_JSON__;

// ── Helpers ──────────────────────────────────────────────────────────────
function escHtml(s) {
    if (!s) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderMarkdown(s) {
    if (!s) return '';
    var h = escHtml(s);
    // Bold: **text** or __text__
    h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/__(.+?)__/g, '<strong>$1</strong>');
    // Italic: *text* or _text_ (but not inside already-converted strong)
    h = h.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
    h = h.replace(/(?<!_)_(?!_)(.+?)(?<!_)_(?!_)/g, '<em>$1</em>');
    // Inline code: `text`
    h = h.replace(/`([^`]+)`/g, '<code style="background:#e8e8e8;padding:1px 3px;font-size:9px;">$1</code>');
    // Bullet lists: lines starting with - or *
    h = h.replace(/^[\-\*]\s+(.+)$/gm, '<li>$1</li>');
    h = h.replace(/((?:<li>.*<\/li>\s*)+)/g, '<ul style="margin:4px 0 4px 16px;padding:0;">$1</ul>');
    // Numbered lists: lines starting with 1. 2. etc
    h = h.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');
    // Line breaks: double newline -> paragraph break, single newline -> <br>
    h = h.replace(/\n\n+/g, '</p><p style="margin:4px 0;">');
    h = h.replace(/\n/g, '<br>');
    h = '<p style="margin:4px 0;">' + h + '</p>';
    return h;
}

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

// ── Decision state ───────────────────────────────────────────────────────
var decisions = {}; // issue_key -> {value, rationale}

// Flatten all issues across edges for the sidebar
var allIssues = [];
var SEVERITY_ORDER = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3};
EDGES.forEach(function(e) {
    if (e.issues && e.issues.length > 0) {
        e.issues.forEach(function(issue) {
            allIssues.push({
                edge_id: e.id,
                issue_key: issue.issue_key || (issue.rule_id + ':' + e.id),
                rule_id: issue.rule_id,
                severity: issue.severity,
                message: issue.message
            });
        });
    }
});
allIssues.sort(function(a, b) {
    var sa = SEVERITY_ORDER[a.severity] || 99;
    var sb = SEVERITY_ORDER[b.severity] || 99;
    if (sa !== sb) return sa - sb;
    return a.edge_id.localeCompare(b.edge_id);
});
var totalIssues = allIssues.length;

function countResolved() {
    var c = 0;
    for (var k in decisions) {
        if (decisions[k] && decisions[k].value) c++;
    }
    return c;
}

function updateProgress() {
    var resolved = countResolved();
    var pct = totalIssues > 0 ? (resolved / totalIssues * 100) : 0;
    document.getElementById('pitfallProgressFill').style.width = pct + '%';
    document.getElementById('pitfallProgressLabel').textContent = resolved + ' / ' + totalIssues + ' resolved';
}

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
var nullLeg = document.createElement('div');
nullLeg.className = 'legend-item';
nullLeg.innerHTML = '<div class="legend-line dashed" style="border-color:#999"></div>Null / Insig.';
legendContainer.appendChild(nullLeg);

// ── Build pitfall sidebar ────────────────────────────────────────────────
var pitfallList = document.getElementById('pitfallList');

if (allIssues.length === 0) {
    pitfallList.innerHTML = '<div class="pitfall-empty">No open issues.</div>';
} else {
    allIssues.forEach(function(issue) {
        var ruleInfo = RULE_DESCRIPTIONS[issue.rule_id] || {};
        var actCfg = ACTIONS_BY_RULE[issue.rule_id] || ACTIONS_BY_RULE['_default'] || {};
        var options = actCfg.options || [];

        var div = document.createElement('div');
        div.className = 'pitfall-item';
        div.setAttribute('data-issue-key', issue.issue_key);
        div.setAttribute('data-edge-id', issue.edge_id);

        // Build options HTML
        var optHtml = '<option value="">-- select action --</option>';
        options.forEach(function(o) {
            optHtml += '<option value="' + escHtml(o.value) + '" data-tooltip="' + escHtml(o.tooltip || '') + '">' + escHtml(o.label) + '</option>';
        });

        var llmText = ISSUE_GUIDANCE[issue.issue_key] || '';
        var llmHtml = llmText ?
            '<div class="pd-llm-guidance"><div class="pd-llm-guidance-title">AI Analysis</div>' +
            renderMarkdown(llmText) + '</div>' : '';

        // Build static text (Why it matters + Decision guidance)
        var staticHtml = '';
        if (ruleInfo.explanation || ruleInfo.guidance) {
            var staticContent =
                (ruleInfo.explanation ? '<div class="pd-label">Why it matters:</div><div class="pd-text">' + ruleInfo.explanation + '</div>' : '') +
                (ruleInfo.guidance ? '<div class="pd-label">Decision guidance:</div><div class="pd-guidance">' + ruleInfo.guidance + '</div>' : '');
            if (llmText) {
                // Collapse when AI Analysis is present
                staticHtml = '<details class="pd-static-ref"><summary>Reference</summary>' + staticContent + '</details>';
            } else {
                staticHtml = staticContent;
            }
        }

        div.innerHTML =
            '<div class="pitfall-item-header" data-header="1">' +
                '<span class="pitfall-edge">' + escHtml(issue.edge_id) + '</span>' +
                '<span class="pitfall-severity ' + issue.severity + '">' + issue.severity + '</span>' +
                '<span class="pitfall-resolved-check">&#10003;</span>' +
            '</div>' +
            '<div class="pitfall-msg">' + escHtml(issue.message) + '</div>' +
            '<div class="pitfall-decision" data-decision-key="' + escHtml(issue.issue_key) + '">' +
                llmHtml +
                staticHtml +
                '<div class="pd-label">Action:</div>' +
                '<select data-key="' + escHtml(issue.issue_key) + '">' + optHtml + '</select>' +
                '<div class="pd-tooltip-area" data-tip-for="' + escHtml(issue.issue_key) + '"></div>' +
                '<input type="text" data-rationale="' + escHtml(issue.issue_key) + '" placeholder="Justification (optional)...">' +
                '<div class="pd-buttons">' +
                    '<button class="pd-btn pd-btn-confirm" data-confirm="' + escHtml(issue.issue_key) + '">Confirm</button>' +
                    '<button class="pd-btn" data-defer="' + escHtml(issue.issue_key) + '">Defer</button>' +
                '</div>' +
            '</div>';

        pitfallList.appendChild(div);
    });
}

// ── Add clean-edge entries (edges without issues) ────────────────────────
var edgesWithIssues = {};
allIssues.forEach(function(i) { edgesWithIssues[i.edge_id] = true; });

var cleanEdges = EDGES.filter(function(e) { return !edgesWithIssues[e.id]; });
if (cleanEdges.length > 0) {
    var divider = document.createElement('div');
    divider.className = 'pitfall-divider';
    divider.textContent = 'Assessed Edges (' + cleanEdges.length + ')';
    pitfallList.appendChild(divider);

    cleanEdges.forEach(function(edge) {
        var div = document.createElement('div');
        div.className = 'pitfall-item clean-edge';
        div.setAttribute('data-edge-id', edge.id);

        var ratingHtml = edge.rating ? '<span class="pitfall-rating">' + escHtml(edge.rating) + '</span>' : '';
        var statsHtml = '';
        if (edge.point !== null && edge.point !== undefined) {
            statsHtml = '<div class="pitfall-stats">' +
                '\u03B2=' + formatBeta(edge.point) +
                (edge.pvalue !== null && edge.pvalue !== undefined ? '  p=' + fmtP(edge.pvalue) : '') +
                (edge.design ? '  ' + escHtml(edge.design) : '') +
                '</div>';
        }

        var annText = edge.llm_annotation || '';
        var annHtml = annText ?
            '<div class="pd-llm-guidance"><div class="pd-llm-guidance-title">AI Analysis</div>' +
            renderMarkdown(annText) + '</div>' : '';

        div.innerHTML =
            '<div class="pitfall-item-header" data-header="1">' +
                '<span class="pitfall-edge">' + escHtml(edge.id) + '</span>' +
                ratingHtml +
            '</div>' +
            statsHtml +
            '<div class="pitfall-decision" data-decision-key="edge:' + escHtml(edge.id) + '">' +
                annHtml +
            '</div>';

        pitfallList.appendChild(div);
    });
}

updateProgress();

// ── Click-to-highlight + expand ──────────────────────────────────────────
var currentHighlightEdge = null;

function highlightEdge(edgeId) {
    // Remove previous highlight
    links.classed('highlighted', false);
    // Apply highlight to matching link
    links.each(function(d) {
        if (d.id === edgeId) {
            d3.select(this).classed('highlighted', true);
        }
    });
    currentHighlightEdge = edgeId;

    // Pan/zoom to edge midpoint
    var edgeData = EDGES.find(function(e) { return e.id === edgeId; });
    if (edgeData && edgeData.source && edgeData.target) {
        var sx = edgeData.source.x || edgeData.source;
        var sy = edgeData.source.y || edgeData.source;
        var tx = edgeData.target.x || edgeData.target;
        var ty = edgeData.target.y || edgeData.target;
        if (typeof sx === 'number' && typeof tx === 'number') {
            var mx = (sx + tx) / 2;
            var my = (sy + ty) / 2;
            var svgEl = document.getElementById('graph');
            var w = svgEl.clientWidth;
            var h = svgEl.clientHeight;
            var t = d3.zoomIdentity.translate(w/2 - mx, h/2 - my);
            svg.transition().duration(500).call(zoomBehavior.transform, t);
        }
    }
}

function markEdgeResolved(edgeId) {
    // Check if ALL issues for this edge are resolved
    var edgeIssues = allIssues.filter(function(i) { return i.edge_id === edgeId; });
    var allResolved = edgeIssues.every(function(i) {
        return decisions[i.issue_key] && decisions[i.issue_key].value;
    });
    if (allResolved) {
        links.each(function(d) {
            if (d.id === edgeId) {
                d3.select(this).classed('edge-resolved', true).classed('critical-issue', false).classed('highlighted', false);
            }
        });
    }
}

// Event delegation on pitfall list
pitfallList.addEventListener('click', function(e) {
    var target = e.target;

    // Confirm button
    if (target.hasAttribute('data-confirm')) {
        e.stopPropagation();
        var key = target.getAttribute('data-confirm');
        if (!decisions[key] || !decisions[key].value) {
            alert('Please select an action before confirming.');
            return;
        }
        var item = pitfallList.querySelector('[data-issue-key="' + key + '"]');
        if (item) {
            item.classList.add('resolved');
            var sel = item.querySelector('select[data-key="' + key + '"]');
            if (sel) sel.classList.add('decided');
        }
        var edgeId = null;
        allIssues.forEach(function(i) { if (i.issue_key === key) edgeId = i.edge_id; });
        if (edgeId) markEdgeResolved(edgeId);
        updateProgress();
        return;
    }

    // Defer button
    if (target.hasAttribute('data-defer')) {
        e.stopPropagation();
        var key2 = target.getAttribute('data-defer');
        var item2 = pitfallList.querySelector('[data-issue-key="' + key2 + '"]');
        if (item2) {
            item2.classList.remove('active');
            var panel = item2.querySelector('.pitfall-decision');
            if (panel) panel.classList.remove('open');
        }
        links.classed('highlighted', false);
        return;
    }

    // Don't toggle when clicking inside decision panel controls
    if (target.tagName === 'SELECT' || target.tagName === 'INPUT' || target.tagName === 'OPTION') return;

    // Find the pitfall-item
    var pitfallItem = target.closest('.pitfall-item');
    if (!pitfallItem) return;

    // Toggle active state
    var wasActive = pitfallItem.classList.contains('active');

    // Deactivate all
    pitfallList.querySelectorAll('.pitfall-item').forEach(function(el) {
        el.classList.remove('active');
        var p = el.querySelector('.pitfall-decision');
        if (p) p.classList.remove('open');
    });
    links.classed('highlighted', false);

    if (!wasActive) {
        pitfallItem.classList.add('active');
        var decPanel = pitfallItem.querySelector('.pitfall-decision');
        if (decPanel) decPanel.classList.add('open');
        var eid = pitfallItem.getAttribute('data-edge-id');
        if (eid) highlightEdge(eid);
    }
});

// Action dropdown change
pitfallList.addEventListener('change', function(e) {
    if (e.target.tagName !== 'SELECT' || !e.target.hasAttribute('data-key')) return;
    var key = e.target.getAttribute('data-key');
    var val = e.target.value;
    if (!decisions[key]) decisions[key] = {value: '', rationale: ''};
    decisions[key].value = val;

    if (val) {
        e.target.classList.add('decided');
    } else {
        e.target.classList.remove('decided');
    }

    // Show tooltip for selected option
    var tipArea = pitfallList.querySelector('[data-tip-for="' + key + '"]');
    if (tipArea) {
        var selectedOpt = e.target.options[e.target.selectedIndex];
        tipArea.textContent = selectedOpt ? (selectedOpt.getAttribute('data-tooltip') || '') : '';
    }
});

// Rationale input
pitfallList.addEventListener('input', function(e) {
    if (e.target.tagName !== 'INPUT' || !e.target.hasAttribute('data-rationale')) return;
    var key = e.target.getAttribute('data-rationale');
    if (!decisions[key]) decisions[key] = {value: '', rationale: ''};
    decisions[key].rationale = e.target.value;
});

// ── Export ────────────────────────────────────────────────────────────────
document.getElementById('exportBtn').addEventListener('click', function() {
    var analystId = document.getElementById('analystId').value.trim();
    var now = new Date().toISOString();
    var decisionList = [];

    for (var key in decisions) {
        var dec = decisions[key];
        if (!dec.value) continue;
        var parts = key.split(':', 1);
        var ruleId = parts[0] || '';
        var edgeId = key.substring(ruleId.length + 1);
        var actCfg = ACTIONS_BY_RULE[ruleId] || ACTIONS_BY_RULE['_default'] || {};
        decisionList.push({
            issue_key: key,
            edge_id: edgeId,
            rule_id: ruleId,
            action: actCfg.action_type || 'unknown',
            value: dec.value,
            rationale: dec.rationale || '',
            status: 'CLOSED',
            source: 'dag_visualization'
        });
    }

    var output = {
        generated_at: now,
        analyst_id: analystId || 'anonymous',
        panel_built_at: GENERATED_AT,
        decisions: decisionList
    };

    var blob = new Blob([JSON.stringify(output, null, 2)], {type: 'application/json'});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    var dateStr = now.slice(0,10).replace(/-/g,'');
    a.href = url;
    a.download = 'dag_viz_decisions_' + dateStr + '.json';
    a.click();
    URL.revokeObjectURL(url);
});

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
var zoomBehavior = d3.zoom().scaleExtent([0.3, 3]).on('zoom', function(e) { g.attr('transform', e.transform); });
svg.call(zoomBehavior);

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
    if (d.llm_annotation) {
        html += '<div class="tooltip-annotation">' + d.llm_annotation + '</div>';
    }
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
    var visibleIssues = visible.reduce(function(s, e) { return s + (e.issues ? e.issues.length : 0); }, 0);
    document.getElementById('statIssues').textContent = visibleIssues;
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
    actions_path: Path | None = None,
    registry_path: Path | None = None,
) -> Path:
    """Build the DAG visualization HTML."""
    dag = load_dag(dag_path)
    metadata = dag.get("metadata", {})
    title = _html.escape(metadata.get("name", dag_path.stem))

    # Load optional data
    issues = load_issues(state_path) if state_path and state_path.exists() else {}

    # Load HITL actions and rule registry
    act_path = actions_path or DEFAULT_ACTIONS
    reg_path = registry_path or DEFAULT_REGISTRY
    actions = load_actions(act_path)
    rules = load_rule_info(reg_path)
    print(f"  {len(actions)} action rules, {len(rules)} rule descriptions loaded")

    # LLM annotations via shared cache
    from shared.llm.guidance_cache import load_cache, generate_and_cache

    cache_dir = PROJECT_ROOT / "outputs" / "agentic" / "llm_cache"
    cache = load_cache(cache_dir)
    if cache is None and llm_annotate:
        print("Generating LLM annotations (shared cache)...")
        cache = generate_and_cache(
            state_path or DEFAULT_STATE, cards_dir or DEFAULT_CARDS_DIR,
            dag_path, cache_dir,
        )
    elif cache is not None:
        print("  Loaded LLM annotations from cache")

    annotations = cache["edge_annotations"] if cache else None
    issue_guidance: dict[str, str] = cache["issue_guidance"] if cache else {}

    # Build data arrays
    nodes = build_nodes_data(dag)
    edges = build_edges_data(dag, cards_dir, issues, annotations)
    latents = build_latents_data(dag)

    print(f"  {len(nodes)} nodes, {len(edges)} edges, {len(latents)} latents")
    issue_count = sum(len(issues.get(e["id"], [])) for e in edges)
    print(f"  {issue_count} open issues across edges")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build HTML
    html = HTML_TEMPLATE
    html = html.replace("__DAG_TITLE__", title)
    html = html.replace("__NODES_JSON__", json.dumps(nodes, indent=2))
    html = html.replace("__EDGES_JSON__", json.dumps(edges, indent=2))
    html = html.replace("__LATENTS_JSON__", json.dumps(latents, indent=2))
    html = html.replace("__METADATA_JSON__", json.dumps(metadata, indent=2))
    html = html.replace("__EDGE_TYPE_COLORS__", json.dumps(EDGE_TYPE_COLORS))
    html = html.replace("__EDGE_TYPE_LABELS__", json.dumps(EDGE_TYPE_LABELS))
    html = html.replace("__RULES_JSON__", json.dumps(rules, indent=2))
    html = html.replace("__ACTIONS_JSON__", json.dumps(actions, indent=2))
    html = html.replace("__GENERATED_AT__", generated_at)
    html = html.replace("__ISSUE_GUIDANCE_JSON__", json.dumps(issue_guidance, indent=2))

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
    parser.add_argument(
        "--actions-file",
        type=Path,
        default=None,
        help="Path to hitl_actions.yaml (default: config/agentic/hitl_actions.yaml)",
    )
    parser.add_argument(
        "--registry-file",
        type=Path,
        default=None,
        help="Path to issue_registry.yaml (default: config/agentic/issue_registry.yaml)",
    )
    args = parser.parse_args()

    cards_dir = args.cards or DEFAULT_CARDS_DIR
    state_path = args.state or DEFAULT_STATE

    print(f"Building DAG visualization from {args.dag_path}...")
    build(
        args.dag_path, cards_dir, state_path, args.output, args.llm_annotate,
        actions_path=args.actions_file, registry_path=args.registry_file,
    )
    print("Done.")


if __name__ == "__main__":
    main()
