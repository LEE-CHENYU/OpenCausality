#!/usr/bin/env python3
"""
Build HITL Resolution Panel HTML.

Reads state.json, edge card YAMLs, hitl_actions.yaml, and issue_registry.yaml,
then generates a self-contained hitl_panel.html with all data injected.

Usage:
    python scripts/build_hitl_panel.py
    python scripts/build_hitl_panel.py --state-file outputs/agentic/issues/state.json
    python scripts/build_hitl_panel.py --output-dir outputs/agentic
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Default paths ────────────────────────────────────────────────────────────

DEFAULT_STATE = PROJECT_ROOT / "outputs" / "agentic" / "issues" / "state.json"
DEFAULT_CARDS_DIR = PROJECT_ROOT / "outputs" / "agentic" / "cards" / "edge_cards"
DEFAULT_ACTIONS = PROJECT_ROOT / "config" / "agentic" / "hitl_actions.yaml"
DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "agentic" / "issue_registry.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "agentic"
DEFAULT_DAG = PROJECT_ROOT / "config" / "agentic" / "dags" / "kspi_k2_full.yaml"


# ── Data extraction ─────────────────────────────────────────────────────────


def load_state(state_path: Path) -> tuple[dict, int]:
    """Return (open_issues dict, closed_count)."""
    with open(state_path) as f:
        state = json.load(f)

    all_issues = state.get("issues", {})
    open_issues = {}
    closed_count = 0
    for key, issue in all_issues.items():
        if issue.get("status") == "OPEN":
            open_issues[key] = issue
        else:
            closed_count += 1
    return open_issues, closed_count


def extract_edge_id(issue_key: str) -> str:
    """Extract edge_id from 'RULE_ID:edge_id' key."""
    parts = issue_key.split(":", 1)
    return parts[1] if len(parts) > 1 else issue_key


def load_edge_summary(card_path: Path) -> dict | None:
    """Extract panel-relevant fields from an edge card YAML."""
    if not card_path.exists():
        return None
    with open(card_path) as f:
        card = yaml.safe_load(f)
    if not card:
        return None

    estimates = card.get("estimates", {})
    diagnostics_raw = card.get("diagnostics", {})
    failure_flags_raw = card.get("failure_flags", {})
    panel_dims = (card.get("data_provenance") or {}).get("panel_dimensions")
    identification = card.get("identification", {})
    spec = card.get("spec_details", {})

    # Build diagnostics list
    diagnostics = []
    for diag in diagnostics_raw.values():
        if isinstance(diag, dict):
            diagnostics.append({
                "name": diag.get("name", ""),
                "passed": bool(diag.get("passed", False)),
            })

    # Build failure flags (true only)
    failure_flags = [k for k, v in failure_flags_raw.items() if v]

    # Panel dimensions
    panel = None
    if panel_dims:
        panel = {
            "n_units": panel_dims.get("n_units"),
            "n_periods": panel_dims.get("n_periods"),
            "balance": panel_dims.get("balance"),
        }

    # Effective obs: try diagnostics first, fall back to combined_row_count
    n_obs = None
    eff_obs = diagnostics_raw.get("effective_obs", {})
    if isinstance(eff_obs, dict) and eff_obs.get("value") is not None:
        n_obs = int(eff_obs["value"])
    elif (card.get("data_provenance") or {}).get("combined_row_count"):
        n_obs = int(card["data_provenance"]["combined_row_count"])

    # Interpretation block
    interp = card.get("interpretation") or {}
    # Counterfactual block
    cf = card.get("counterfactual_block") or {}
    # Propagation role
    prop = card.get("propagation_role") or {}

    return {
        "point": estimates.get("point"),
        "se": estimates.get("se"),
        "pvalue": estimates.get("pvalue"),
        "design": spec.get("design"),
        "rating": card.get("credibility_rating"),
        "score": card.get("credibility_score"),
        "claim_level": identification.get("claim_level"),
        "all_pass": bool(card.get("all_diagnostics_pass", False)),
        "n_obs": n_obs,
        "diagnostics": diagnostics,
        "failure_flags": failure_flags,
        "panel": panel,
        # Additional fields for decision guides
        "estimand": interp.get("estimand", ""),
        "is_not": interp.get("is_not", []),
        "allowed_uses": interp.get("allowed_uses", []),
        "forbidden_uses": interp.get("forbidden_uses", []),
        "cf_shock_allowed": cf.get("shock_scenario_allowed"),
        "cf_policy_allowed": cf.get("policy_intervention_allowed"),
        "cf_reason_blocked": cf.get("reason_blocked", ""),
        "id_risks": identification.get("risks", {}),
        "untestable_assumptions": identification.get("untestable_assumptions", []),
        "testable_passed": identification.get("testable_threats_passed", []),
        "testable_failed": identification.get("testable_threats_failed", []),
        "propagation_role": prop.get("role", ""),
    }


def load_actions(actions_path: Path) -> dict:
    """Load hitl_actions.yaml → actions dict with HTML-escaped tooltips."""
    import html as _html

    with open(actions_path) as f:
        data = yaml.safe_load(f)
    actions = data.get("actions", {})
    # HTML-escape tooltip content for safe injection
    for rule_id, action_cfg in actions.items():
        for opt in action_cfg.get("options", []):
            if "tooltip" in opt:
                opt["tooltip"] = _html.escape(opt["tooltip"])
    return actions


def load_rule_info(registry_path: Path) -> dict:
    """Load issue_registry.yaml → {rule_id: {description, explanation, guidance}}."""
    import html as _html

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


def load_dag_edges(dag_path: Path) -> dict:
    """Load edge metadata from DAG YAML → {edge_id: {from, to, edge_type, expected_sign, interpretation, notes}}."""
    import html as _html

    if not dag_path.exists():
        print(f"  Warning: DAG file not found: {dag_path}")
        return {}
    with open(dag_path) as f:
        dag = yaml.safe_load(f)
    if not dag:
        return {}
    result = {}
    for edge in dag.get("edges", []):
        eid = edge.get("id")
        if not eid:
            continue
        ac = edge.get("acceptance_criteria") or {}
        plaus = ac.get("plausibility") or {}
        stab = ac.get("stability") or {}
        interp = edge.get("interpretation") or {}
        result[eid] = {
            "from": edge.get("from", ""),
            "to": edge.get("to", ""),
            "edge_type": edge.get("edge_type", ""),
            "expected_sign": plaus.get("expected_sign", ""),
            "magnitude_range": plaus.get("magnitude_range"),
            "regime_split_date": stab.get("regime_split_date", ""),
            "interpretation": _html.escape(str(interp.get("is", ""))),
            "is_not": [_html.escape(str(x)) for x in (interp.get("is_not") or [] if isinstance(interp.get("is_not"), list) else [interp.get("is_not", "")] if interp.get("is_not") else [])],
            "allowed_uses": interp.get("allowed_uses", []),
            "forbidden_uses": interp.get("forbidden_uses", []),
            "notes": _html.escape(str(edge.get("notes", ""))),
        }
    return result


def load_dag_nodes(dag_path: Path) -> dict:
    """Load node metadata from DAG YAML → {node_id: {name, unit, description}}."""
    import html as _html

    if not dag_path.exists():
        return {}
    with open(dag_path) as f:
        dag = yaml.safe_load(f)
    if not dag:
        return {}
    result = {}
    for node in dag.get("nodes", []):
        nid = node.get("id")
        if not nid:
            continue
        result[nid] = {
            "name": _html.escape(str(node.get("name", nid))),
            "unit": _html.escape(str(node.get("unit", ""))),
            "description": _html.escape(str(node.get("description", ""))),
        }
    return result


# ── HTML template ────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HITL Resolution Panel</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background: #fff;
            color: #000;
            font-size: 12px;
            line-height: 1.5;
        }

        /* Header */
        .header {
            padding: 24px 32px 16px;
            border-bottom: 1px solid #000;
        }
        .header h1 {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .header .subtitle {
            color: #666;
            font-size: 11px;
            margin-bottom: 12px;
        }
        .stats-row {
            display: flex;
            gap: 24px;
            align-items: center;
            flex-wrap: wrap;
        }
        .stat {
            font-size: 11px;
        }
        .stat strong { font-weight: 600; }
        .stat .critical { color: #000; font-weight: 700; }
        .stat .high { color: #555; font-weight: 600; }

        /* Controls bar */
        .controls-bar {
            padding: 10px 32px;
            border-bottom: 1px solid #000;
            display: flex;
            gap: 16px;
            align-items: center;
            background: #fafafa;
            font-size: 11px;
        }
        .controls-bar label {
            color: #666;
            margin-right: 4px;
        }
        .controls-bar select, .controls-bar input[type="text"] {
            border: 1px solid #000;
            background: #fff;
            padding: 4px 8px;
            font-size: 11px;
            font-family: inherit;
        }
        .controls-bar input[type="text"] {
            width: 200px;
        }

        /* Progress bar */
        .progress-container {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-left: auto;
        }
        .progress-bar {
            width: 120px;
            height: 8px;
            border: 1px solid #000;
            background: #fff;
        }
        .progress-fill {
            height: 100%;
            background: #000;
            transition: width 0.3s;
        }
        .progress-label {
            font-size: 11px;
            font-weight: 500;
            min-width: 60px;
        }

        /* Sections */
        .section {
            padding: 16px 32px;
            border-bottom: 1px solid #ddd;
        }
        .section:last-of-type {
            border-bottom: 1px solid #000;
        }
        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }
        .section-title {
            font-size: 13px;
            font-weight: 600;
        }
        .section-count {
            font-size: 11px;
            color: #666;
        }
        .severity-badge {
            display: inline-block;
            font-size: 10px;
            font-weight: 600;
            padding: 1px 6px;
            border: 1px solid #000;
            margin-left: 8px;
        }
        .severity-badge.critical {
            background: #000;
            color: #fff;
        }
        .severity-badge.high {
            background: #fff;
            color: #000;
        }
        .severity-badge.medium {
            background: #eee;
            color: #000;
        }
        .severity-badge.low {
            background: #fff;
            color: #999;
        }

        /* Bulk action */
        .bulk-action {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
            padding: 8px 12px;
            border: 1px solid #ddd;
            background: #fafafa;
        }
        .bulk-action label { color: #666; font-size: 11px; }
        .bulk-action select {
            border: 1px solid #000;
            background: #fff;
            padding: 3px 6px;
            font-size: 11px;
            font-family: inherit;
        }
        .btn-sm {
            border: 1px solid #000;
            background: #fff;
            padding: 3px 10px;
            font-size: 11px;
            font-family: inherit;
            cursor: pointer;
        }
        .btn-sm:hover { background: #000; color: #fff; }

        /* Table */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }
        th {
            text-align: left;
            font-weight: 600;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 6px 8px;
            border-bottom: 1px solid #000;
            color: #666;
        }
        td {
            padding: 8px 8px;
            border-bottom: 1px solid #eee;
            vertical-align: top;
        }
        tr:hover td { background: #fafafa; }
        tr.resolved td { opacity: 0.45; }

        .edge-name {
            font-weight: 500;
            white-space: nowrap;
        }
        .mono {
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-size: 10px;
        }
        .diag-pass { color: #000; }
        .diag-fail { color: #000; font-weight: 700; text-decoration: underline; }

        .claim-tag {
            display: inline-block;
            font-size: 9px;
            font-weight: 600;
            padding: 1px 5px;
            border: 1px solid #000;
            white-space: nowrap;
        }
        .rating-tag {
            display: inline-block;
            font-size: 10px;
            font-weight: 700;
            width: 18px;
            text-align: center;
            border: 1px solid #000;
        }

        .flag-list {
            font-size: 10px;
            color: #666;
        }
        .flag-list .flag {
            display: inline-block;
            margin-right: 4px;
            padding: 0 3px;
            border: 1px solid #ccc;
            font-size: 9px;
        }

        /* Action cells */
        td select {
            border: 1px solid #000;
            background: #fff;
            padding: 3px 6px;
            font-size: 11px;
            font-family: inherit;
            width: 100%;
        }
        td select.decided {
            background: #000;
            color: #fff;
        }
        td input[type="text"] {
            border: 1px solid #ccc;
            padding: 3px 6px;
            font-size: 10px;
            font-family: inherit;
            width: 100%;
            margin-top: 4px;
        }
        td input[type="text"]:focus {
            border-color: #000;
            outline: none;
        }

        /* Details toggle */
        .details-row td {
            padding: 0 8px 8px 8px;
            border-bottom: 1px solid #ddd;
        }
        .details-content {
            display: none;
            padding: 8px 12px;
            background: #fafafa;
            border: 1px solid #eee;
            font-size: 10px;
            line-height: 1.6;
        }
        .details-content.open { display: block; }
        .detail-row {
            display: flex;
            justify-content: space-between;
            max-width: 400px;
        }
        .detail-label { color: #666; }
        .detail-value { font-weight: 500; font-variant-numeric: tabular-nums; }
        .toggle-btn {
            cursor: pointer;
            color: #666;
            font-size: 10px;
            text-decoration: underline;
        }
        .toggle-btn:hover { color: #000; }

        /* Footer */
        .footer {
            padding: 16px 32px;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .footer label {
            font-size: 11px;
            color: #666;
        }
        .footer input[type="text"] {
            border: 1px solid #000;
            padding: 6px 10px;
            font-size: 11px;
            font-family: inherit;
            width: 220px;
        }
        .btn-export {
            border: 1px solid #000;
            background: #000;
            color: #fff;
            padding: 6px 20px;
            font-size: 12px;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
        }
        .btn-export:hover { background: #333; }
        .btn-export:disabled {
            background: #ccc;
            border-color: #ccc;
            cursor: not-allowed;
        }
        .footer .timestamp {
            font-size: 10px;
            color: #999;
            margin-left: auto;
        }

        /* Welcome banner */
        .welcome-banner {
            padding: 16px 32px;
            background: #f8f8f8;
            border-bottom: 1px solid #ddd;
            font-size: 11px;
            line-height: 1.7;
            color: #333;
        }
        .welcome-banner h2 { font-size: 13px; font-weight: 600; margin-bottom: 8px; }
        .welcome-banner p { margin-bottom: 6px; }

        /* Rule info box */
        .rule-info {
            margin-bottom: 12px;
            padding: 10px 14px;
            background: #f8f8f8;
            border-left: 3px solid #000;
        }
        .rule-description { font-size: 11px; font-weight: 500; margin-bottom: 4px; }
        .rule-extra { display: none; }
        .rule-extra.open { display: block; }
        .rule-explanation {
            font-size: 11px; color: #444; line-height: 1.6; margin-bottom: 6px; margin-top: 6px;
        }
        .rule-guidance {
            font-size: 11px; color: #222; padding: 6px 10px;
            background: #fff; border: 1px solid #ddd;
        }
        .rule-guidance strong { font-weight: 600; }
        .why-toggle {
            cursor: pointer; color: #666; font-size: 10px;
            text-decoration: underline; margin-left: 8px;
        }
        .why-toggle:hover { color: #000; }

        /* Action tooltip info box */
        .action-info {
            font-size: 10px; color: #555; margin-top: 4px;
            padding: 4px 6px; background: #f8f8f8; border: 1px solid #eee;
            min-height: 16px;
        }

        /* Decision guide */
        .decision-guide {
            margin-bottom: 10px;
            padding: 10px 14px;
            background: #fffbf0;
            border-left: 3px solid #c8a000;
            font-size: 11px;
            line-height: 1.7;
        }
        .decision-guide-title {
            font-weight: 600;
            font-size: 11px;
            margin-bottom: 6px;
            color: #7a6400;
        }
        .decision-guide .risk-tag {
            display: inline-block;
            font-size: 9px;
            padding: 1px 5px;
            border: 1px solid #c8a000;
            margin-right: 3px;
            background: #fff8e0;
        }
        .decision-guide .risk-tag.high { border-color: #c00; background: #fff0f0; }
        .decision-guide .sign-match { color: #080; }
        .decision-guide .sign-mismatch { color: #c00; font-weight: 600; }
        .decision-guide .implication {
            margin-top: 6px; padding: 4px 8px;
            background: #fff; border: 1px solid #e8e0c0;
        }

        /* Hidden */
        .hidden { display: none !important; }
    </style>
</head>
<body>

<div class="header">
    <h1>HITL Resolution Panel</h1>
    <div class="subtitle">Human-in-the-Loop Issue Resolution &mdash; KSPI K2 Agentic Pipeline</div>
    <div class="stats-row" id="statsRow"></div>
</div>

<div class="welcome-banner">
    <h2>How to Use This Panel</h2>
    <p>This panel presents issues detected by the OpenCausality governance system that require
    your expert judgment. Each section groups issues by rule type and severity. Your role is to
    review each flagged issue, understand its implications for causal inference, and select an
    appropriate resolution action.</p>
    <p>For each issue, click <strong>[Why?]</strong> to see why it matters and how to decide.
    Select an action from the dropdown and optionally add a rationale. Your decisions will be
    logged in the audit trail for reproducibility.</p>
    <p>When all issues are resolved, click <strong>Export Decisions</strong> to save your
    decisions as a JSON file that can be fed back into the pipeline.</p>
</div>

<div class="controls-bar">
    <label>Severity</label>
    <select id="filterSeverity">
        <option value="all">All</option>
        <option value="CRITICAL">CRITICAL</option>
        <option value="HIGH">HIGH</option>
        <option value="MEDIUM">MEDIUM</option>
        <option value="LOW">LOW</option>
    </select>
    <label>Rule</label>
    <select id="filterRule">
        <option value="all">All rules</option>
    </select>
    <label>Status</label>
    <select id="filterStatus">
        <option value="all">All</option>
        <option value="pending">Pending</option>
        <option value="resolved">Resolved</option>
    </select>
    <div class="progress-container">
        <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
        <span class="progress-label" id="progressLabel">0 / 0</span>
    </div>
</div>

<div id="sectionsContainer"></div>

<div class="footer">
    <label>Analyst ID</label>
    <input type="text" id="analystId" placeholder="Your name or initials">
    <button class="btn-export" id="exportBtn">Export Decisions</button>
    <span class="timestamp" id="genTimestamp"></span>
</div>

<script>
// ── INJECTED DATA (replaced at build time) ──────────────────────────────
const ISSUES = __ISSUES_JSON__;
const CLOSED_ISSUES_COUNT = __CLOSED_COUNT__;
const EDGES = __EDGES_JSON__;
const ACTIONS_BY_RULE = __ACTIONS_JSON__;
const RULE_DESCRIPTIONS = __RULES_JSON__;
const GENERATED_AT = "__GENERATED_AT__";
const DAG_EDGES = __DAG_EDGES_JSON__;
const DAG_NODES = __DAG_NODES_JSON__;

// Severity sort order
const SEVERITY_ORDER = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3};

// Known rule-specific column configs
const RULE_COLUMNS = {
    SIGNIFICANT_BUT_NOT_IDENTIFIED: {
        headers: '<th>Edge</th><th>p-value</th><th>Point (SE)</th><th>Claim</th><th>Rating</th><th>Design</th><th>Diagnostics</th><th>Action</th>',
        colspan: 8,
        renderRow: function(item, edge, actCfg, isResolved, decVal) {
            return `
                <td>
                    <span class="edge-name">${item.edge_id}</span>
                    <span class="toggle-btn" data-toggle="${item.key}">[+]</span>
                    ${edge.failure_flags && edge.failure_flags.length > 0 ?
                        '<br><span class="flag-list">' + edge.failure_flags.map(f => '<span class="flag">'+f+'</span>').join('') + '</span>' : ''}
                </td>
                <td class="mono">${fmtP(edge.pvalue)}</td>
                <td class="mono">${fmtNum(edge.point)} (${fmtNum(edge.se)})</td>
                <td><span class="claim-tag">${edge.claim_level || '--'}</span></td>
                <td><span class="rating-tag">${edge.rating || '--'}</span></td>
                <td style="font-size:10px">${edge.design || '--'}</td>
                <td style="font-size:10px">${edge.diagnostics ? diagSummary(edge.diagnostics) : '--'}</td>
                <td>${renderAction(item.key, actCfg, isResolved, decVal)}</td>
            `;
        }
    },
    RATING_DIAGNOSTICS_CONFLICT: {
        headers: '<th>Edge</th><th>Rating</th><th>Score</th><th>Failed diagnostics</th><th>N obs</th><th>Action</th>',
        colspan: 6,
        renderRow: function(item, edge, actCfg, isResolved, decVal) {
            var failedDiags = (edge.diagnostics || []).filter(function(d){return !d.passed;});
            return `
                <td>
                    <span class="edge-name">${item.edge_id}</span>
                    <span class="toggle-btn" data-toggle="${item.key}">[+]</span>
                </td>
                <td><span class="rating-tag">${edge.rating || '--'}</span></td>
                <td class="mono">${edge.score ? edge.score.toFixed(3) : '--'}</td>
                <td style="font-size:10px">${failedDiags.map(function(d){return '<span class="diag-fail">\u2717 '+d.name+'</span>';}).join(', ') || 'none'}</td>
                <td class="mono">${edge.n_obs || '--'}</td>
                <td>${renderAction(item.key, actCfg, isResolved, decVal)}</td>
            `;
        }
    },
    LOO_INSTABILITY: {
        headers: '<th>Edge</th><th>Message</th><th>N units</th><th>Rating</th><th>Score</th><th>Action</th>',
        colspan: 6,
        renderRow: function(item, edge, actCfg, isResolved, decVal) {
            var nUnits = edge.panel ? edge.panel.n_units : (edge.n_obs || '--');
            return `
                <td>
                    <span class="edge-name">${item.edge_id}</span>
                    <span class="toggle-btn" data-toggle="${item.key}">[+]</span>
                </td>
                <td style="font-size:10px;max-width:200px">${item.message}</td>
                <td class="mono">${nUnits}</td>
                <td><span class="rating-tag">${edge.rating || '--'}</span></td>
                <td class="mono">${edge.score ? edge.score.toFixed(3) : '--'}</td>
                <td>${renderAction(item.key, actCfg, isResolved, decVal)}</td>
            `;
        }
    }
};

// Default (generic) columns for unknown rules
var DEFAULT_COLUMNS = {
    headers: '<th>Edge</th><th>Message</th><th>Rating</th><th>Score</th><th>Action</th>',
    colspan: 5,
    renderRow: function(item, edge, actCfg, isResolved, decVal) {
        return `
            <td>
                <span class="edge-name">${item.edge_id}</span>
                <span class="toggle-btn" data-toggle="${item.key}">[+]</span>
            </td>
            <td style="font-size:10px;max-width:250px">${item.message}</td>
            <td><span class="rating-tag">${edge.rating || '--'}</span></td>
            <td class="mono">${edge.score ? edge.score.toFixed(3) : '--'}</td>
            <td>${renderAction(item.key, actCfg, isResolved, decVal)}</td>
        `;
    }
};

// ── DAG helpers ───────────────────────────────────────────────────────
function edgeLabel(edgeId) {
    var de = DAG_EDGES[edgeId];
    if (!de) return edgeId;
    var fromName = (DAG_NODES[de.from] || {}).name || de.from;
    var toName = (DAG_NODES[de.to] || {}).name || de.to;
    return fromName + ' \u2192 ' + toName;
}

function edgeInterpretation(edgeId) {
    var de = DAG_EDGES[edgeId];
    return de ? (de.interpretation || '') : '';
}

function escHtml(s) {
    if (!s) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function signLabel(val) {
    if (val === null || val === undefined) return 'unknown';
    return val > 0 ? 'positive' : val < 0 ? 'negative' : 'zero';
}

function checkSignConsistency(edge, dagEdge) {
    if (!dagEdge || !dagEdge.expected_sign || edge.point === null || edge.point === undefined) return null;
    var actual = edge.point > 0 ? 'positive' : edge.point < 0 ? 'negative' : 'zero';
    var expected = dagEdge.expected_sign;
    return actual === expected;
}

function renderRiskTags(risks) {
    if (!risks || typeof risks !== 'object') return '';
    var html = '';
    for (var k in risks) {
        var level = risks[k];
        var cls = (level === 'high') ? ' high' : '';
        html += '<span class="risk-tag' + cls + '">' + escHtml(k.replace(/_/g,' ')) + ' ' + escHtml(level) + '</span>';
    }
    return html;
}

// ── Decision Guides (per-rule template functions) ─────────────────────
var DECISION_GUIDES = {
    SIGNIFICANT_BUT_NOT_IDENTIFIED: function(item, edge, dagEdge) {
        var label = edgeLabel(item.edge_id);
        var interp = edgeInterpretation(item.edge_id);
        var signOk = checkSignConsistency(edge, dagEdge);
        var signHtml = '';
        if (signOk !== null) {
            var expected = dagEdge.expected_sign || '?';
            var actual = signLabel(edge.point);
            if (signOk) {
                signHtml = '<div>Expected sign: <strong>' + escHtml(expected) + '</strong>. ' +
                    'Estimated: <span class="sign-match"><strong>' + escHtml(actual) + '</strong> \u2713 consistent</span></div>';
            } else {
                signHtml = '<div>Expected sign: <strong>' + escHtml(expected) + '</strong>. ' +
                    'Estimated: <span class="sign-mismatch"><strong>' + escHtml(actual) + '</strong> \u2717 inconsistent</span></div>';
            }
        }

        var claimHtml = '';
        var cl = edge.claim_level || '';
        if (cl === 'REDUCED_FORM') {
            claimHtml = '<div>Current claim: <strong>REDUCED_FORM</strong> \u2014 already acknowledged as reduced-form. Consider upgrading if you have IV/RDD/DiD evidence.</div>';
        } else if (cl === 'BLOCKED_ID') {
            claimHtml = '<div>Current claim: <strong>BLOCKED_ID</strong> \u2014 identification is currently blocked. Resolve identification concern before using in counterfactuals.</div>';
        } else {
            claimHtml = '<div>Current claim: <strong>' + escHtml(cl || 'none') + '</strong> \u2014 no claim level set. This is the most urgent: significant result with no acknowledgement.</div>';
        }

        var risksHtml = renderRiskTags(edge.id_risks);

        var cfHtml = '';
        if (edge.cf_reason_blocked) {
            cfHtml = '<div class="implication"><strong>Counterfactual status:</strong> ' + escHtml(edge.cf_reason_blocked) + '</div>';
        }

        var implHtml = '<div class="implication"><strong>If you select IDENTIFIED_CAUSAL:</strong> This edge will be used in shock and policy counterfactuals. Requires documented identification strategy (IV, RDD, DiD).</div>' +
            '<div class="implication"><strong>If you select REDUCED_FORM:</strong> Informative but excluded from causal counterfactuals. Reduced-form edges cannot support policy intervention claims.</div>' +
            '<div class="implication"><strong>If you select BLOCKED_ID:</strong> Edge remains blocked. No counterfactual propagation until resolved.</div>';

        return '<div class="decision-guide">' +
            '<div class="decision-guide-title">Decision Guide: ' + escHtml(label) + '</div>' +
            (interp ? '<div>' + escHtml(interp) + '</div>' : '') +
            signHtml +
            claimHtml +
            (risksHtml ? '<div style="margin-top:4px">Identification risks: ' + risksHtml + '</div>' : '') +
            cfHtml +
            implHtml +
            '</div>';
    },

    LOO_INSTABILITY: function(item, edge, dagEdge) {
        var label = edgeLabel(item.edge_id);
        var interp = edgeInterpretation(item.edge_id);

        // Extract influential unit from message
        var unitMatch = (item.message || '').match(/excluding[:\s]+(\S+)/i);
        var unitName = unitMatch ? unitMatch[1] : 'unknown unit';

        var panelHtml = '';
        if (edge.panel) {
            var nUnits = edge.panel.n_units || '?';
            var nPeriods = edge.panel.n_periods || '?';
            var fraction = edge.panel.n_units ? (1 / edge.panel.n_units * 100).toFixed(1) : '?';
            panelHtml = '<div>Panel: <strong>' + nUnits + ' units \u00d7 ' + nPeriods + ' periods</strong>. ' +
                'Dropping <strong>' + escHtml(unitName) + '</strong> removes ~' + fraction + '% of cross-sectional variation.</div>';
        } else {
            panelHtml = '<div>Influential unit: <strong>' + escHtml(unitName) + '</strong></div>';
        }

        var implHtml = '<div class="implication"><strong>Suggested actions:</strong> ' +
            '(1) Winsorize extreme values from this unit. ' +
            '(2) Trim/exclude and report both results. ' +
            '(3) Accept with caveat noting sensitivity to this unit.</div>';

        return '<div class="decision-guide">' +
            '<div class="decision-guide-title">Decision Guide: ' + escHtml(label) + '</div>' +
            (interp ? '<div>' + escHtml(interp) + '</div>' : '') +
            '<div>' + escHtml(item.message) + '</div>' +
            panelHtml +
            implHtml +
            '</div>';
    },

    RATING_DIAGNOSTICS_CONFLICT: function(item, edge, dagEdge) {
        var label = edgeLabel(item.edge_id);
        var interp = edgeInterpretation(item.edge_id);

        var failedDiags = (edge.diagnostics || []).filter(function(d){return !d.passed;});
        var diagNames = {
            'effective_obs': 'Tests whether the sample has enough observations for reliable inference',
            'ts_leads_test': 'Tests whether future values predict current treatment (anticipation/reverse causality)',
            'sign_consistency': 'Tests whether the estimated sign matches the expected direction from theory',
            'ts_residual_autocorr': 'Tests for autocorrelation in residuals that could bias standard errors',
            'ts_hac_sensitivity': 'Tests whether HAC standard error corrections materially change inference',
            'ts_lag_sensitivity': 'Tests whether results are robust to different lag specifications',
            'ts_regime_stability': 'Tests for structural breaks in the estimated relationship',
            'ts_shock_support': 'Tests whether the shock variable has sufficient variation',
            'hac_bandwidth': 'Checks Newey-West bandwidth selection'
        };

        var failedHtml = '';
        if (failedDiags.length > 0) {
            failedHtml = '<div style="margin-top:4px"><strong>Failed diagnostics:</strong><ul style="margin:2px 0 0 16px;padding:0">';
            failedDiags.forEach(function(d) {
                var desc = diagNames[d.name] || 'Diagnostic test';
                failedHtml += '<li><span class="diag-fail">\u2717 ' + escHtml(d.name) + '</span> \u2014 ' + escHtml(desc) + '</li>';
            });
            failedHtml += '</ul></div>';
        }

        var ratingHtml = '<div>Current rating: <strong>' + escHtml(edge.rating || '--') + '</strong> (score: ' +
            (edge.score ? edge.score.toFixed(3) : '--') + ') but <strong>' + failedDiags.length +
            '</strong> diagnostic(s) failed.</div>';

        var implHtml = '<div class="implication"><strong>Options:</strong> ' +
            '(1) Downgrade rating to match diagnostic evidence. ' +
            '(2) Re-estimate with different specification to resolve failures. ' +
            '(3) Accept with caveat, documenting why the rating override is justified.</div>';

        return '<div class="decision-guide">' +
            '<div class="decision-guide-title">Decision Guide: ' + escHtml(label) + '</div>' +
            (interp ? '<div>' + escHtml(interp) + '</div>' : '') +
            ratingHtml +
            failedHtml +
            implHtml +
            '</div>';
    },

    REGIME_INSTABILITY: function(item, edge, dagEdge) {
        var label = edgeLabel(item.edge_id);
        var interp = edgeInterpretation(item.edge_id);

        var splitDate = (dagEdge && dagEdge.regime_split_date) ? dagEdge.regime_split_date : '';
        var splitHtml = splitDate ?
            '<div>DAG regime split date: <strong>' + escHtml(splitDate) + '</strong></div>' : '';

        var implHtml = '<div class="implication"><strong>Options:</strong> ' +
            '(1) Split sample at the break date and estimate separately. ' +
            '(2) Restrict counterfactual scope to post-break regime only. ' +
            '(3) Use a regime-switching model that accounts for the structural change.</div>';

        return '<div class="decision-guide">' +
            '<div class="decision-guide-title">Decision Guide: ' + escHtml(label) + '</div>' +
            (interp ? '<div>' + escHtml(interp) + '</div>' : '') +
            '<div>' + escHtml(item.message) + '</div>' +
            splitHtml +
            implHtml +
            '</div>';
    }
};

function defaultDecisionGuide(item, edge, dagEdge) {
    var label = edgeLabel(item.edge_id);
    var interp = edgeInterpretation(item.edge_id);
    var signOk = checkSignConsistency(edge, dagEdge);
    var signHtml = '';
    if (signOk !== null) {
        var expected = dagEdge.expected_sign || '?';
        var actual = signLabel(edge.point);
        if (signOk) {
            signHtml = '<div>Sign: <span class="sign-match">' + escHtml(actual) + ' (expected ' + escHtml(expected) + ') \u2713</span></div>';
        } else {
            signHtml = '<div>Sign: <span class="sign-mismatch">' + escHtml(actual) + ' (expected ' + escHtml(expected) + ') \u2717</span></div>';
        }
    }
    var risksHtml = renderRiskTags(edge.id_risks);
    return '<div class="decision-guide">' +
        '<div class="decision-guide-title">Decision Guide: ' + escHtml(label) + '</div>' +
        (interp ? '<div>' + escHtml(interp) + '</div>' : '') +
        '<div>Estimate: <strong>' + fmtNum(edge.point) + '</strong> (SE ' + fmtNum(edge.se) + ', p=' + fmtP(edge.pvalue) + '). ' +
        'Claim: <strong>' + escHtml(edge.claim_level || 'none') + '</strong>. Rating: <strong>' + escHtml(edge.rating || '--') + '</strong>.</div>' +
        signHtml +
        (risksHtml ? '<div style="margin-top:4px">Identification risks: ' + risksHtml + '</div>' : '') +
        '<div style="margin-top:4px"><strong>Issue:</strong> ' + escHtml(item.message) + '</div>' +
        '</div>';
}

// ── State ─────────────────────────────────────────────────────────────
var decisions = {}; // key -> {value, rationale}

// ── Helpers ───────────────────────────────────────────────────────────
function extractEdgeId(issueKey) {
    var parts = issueKey.split(':');
    return parts.slice(1).join(':');
}

function fmtP(p) {
    if (p === null || p === undefined) return '--';
    if (p < 0.0001) return p.toExponential(2);
    return p.toFixed(4);
}

function fmtNum(n, decimals) {
    if (n === null || n === undefined) return '--';
    if (Math.abs(n) > 1000) return n.toFixed(1);
    if (Math.abs(n) < 0.01 && n !== 0) return n.toExponential(2);
    return n.toFixed(decimals !== undefined ? decimals : 4);
}

function diagSummary(diags) {
    var passed = diags.filter(function(d){return d.passed;}).length;
    return passed + '/' + diags.length + ' pass';
}

function diagDetail(diags) {
    return diags.map(function(d) {
        return '<span class="' + (d.passed ? 'diag-pass' : 'diag-fail') + '">' +
               (d.passed ? '\u2713' : '\u2717') + ' ' + d.name + '</span>';
    }).join(', ');
}

function renderAction(key, actCfg, isResolved, decVal) {
    var opts = '<option value="">-- select --</option>';
    var tooltips = {};
    if (actCfg && actCfg.options) {
        actCfg.options.forEach(function(o) {
            var sel = (isResolved && decVal === o.value) ? ' selected' : '';
            var tip = o.tooltip ? ' title="' + o.tooltip.replace(/"/g, '&quot;') + '"' : '';
            opts += '<option value="' + o.value + '"' + sel + tip + '>' + o.label + '</option>';
            if (o.tooltip) tooltips[o.value] = o.tooltip;
        });
    }
    var rat = (decisions[key] && decisions[key].rationale) || '';
    var currentTip = (isResolved && tooltips[decVal]) ? tooltips[decVal] : '';
    return '<select data-key="' + key + '" data-tooltips=\'' + JSON.stringify(tooltips).replace(/'/g, "&#39;") + '\'>' + opts + '</select>' +
           '<div class="action-info" data-tip-for="' + key + '">' + currentTip + '</div>' +
           '<input type="text" data-rationale="' + key + '" placeholder="Rationale..." value="' + rat.replace(/"/g, '&quot;') + '">';
}

function updateProgress() {
    var openKeys = Object.keys(ISSUES);
    var total = openKeys.length;
    var resolved = openKeys.filter(function(k){return decisions[k] && decisions[k].value;}).length;
    var pct = total > 0 ? (resolved / total * 100) : 0;
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('progressLabel').textContent = resolved + ' / ' + total + ' resolved';
}

function updateStats() {
    var openKeys = Object.keys(ISSUES);
    var total = openKeys.length + CLOSED_ISSUES_COUNT;
    var sevCounts = {};
    openKeys.forEach(function(k) {
        var s = ISSUES[k].severity;
        sevCounts[s] = (sevCounts[s] || 0) + 1;
    });
    var html = '<span class="stat"><strong>' + openKeys.length + '</strong> open</span>' +
               '<span class="stat">&middot; <strong>' + CLOSED_ISSUES_COUNT + '</strong> closed</span>' +
               '<span class="stat">&middot; <strong>' + total + '</strong> total</span>' +
               '<span class="stat" style="margin-left:12px">';
    var parts = [];
    ['CRITICAL','HIGH','MEDIUM','LOW'].forEach(function(s) {
        if (sevCounts[s]) parts.push('<span class="' + s.toLowerCase() + '">' + sevCounts[s] + ' ' + s + '</span>');
    });
    html += parts.join(' &middot; ') + '</span>';
    document.getElementById('statsRow').innerHTML = html;
}

// ── Group issues by rule_id ───────────────────────────────────────────
function groupByRule() {
    var groups = {};
    for (var key in ISSUES) {
        var issue = ISSUES[key];
        var rule = issue.rule_id;
        if (!groups[rule]) groups[rule] = [];
        groups[rule].push({key: key, severity: issue.severity, rule_id: rule,
                          message: issue.message, edge_id: extractEdgeId(key)});
    }
    return groups;
}

// ── Render ────────────────────────────────────────────────────────────
function render() {
    var container = document.getElementById('sectionsContainer');
    container.innerHTML = '';
    var groups = groupByRule();

    // Populate rule filter
    var ruleFilter = document.getElementById('filterRule');
    var existingRules = [];
    for (var i = 0; i < ruleFilter.options.length; i++) existingRules.push(ruleFilter.options[i].value);
    for (var rule in groups) {
        if (existingRules.indexOf(rule) === -1) {
            var opt = document.createElement('option');
            opt.value = rule;
            opt.textContent = rule;
            ruleFilter.appendChild(opt);
        }
    }

    var sevFilter = document.getElementById('filterSeverity').value;
    var ruleFilterVal = document.getElementById('filterRule').value;
    var statusFilter = document.getElementById('filterStatus').value;

    // Sort rules: by severity of first item (CRITICAL first), then alphabetical
    var ruleKeys = Object.keys(groups);
    ruleKeys.sort(function(a, b) {
        var sevA = SEVERITY_ORDER[groups[a][0].severity] || 99;
        var sevB = SEVERITY_ORDER[groups[b][0].severity] || 99;
        if (sevA !== sevB) return sevA - sevB;
        return a.localeCompare(b);
    });

    ruleKeys.forEach(function(rule) {
        if (ruleFilterVal !== 'all' && ruleFilterVal !== rule) return;

        var items = groups[rule];

        // Apply severity filter
        if (sevFilter !== 'all') {
            items = items.filter(function(i){return i.severity === sevFilter;});
        }

        // Apply status filter
        if (statusFilter === 'pending') {
            items = items.filter(function(i){return !decisions[i.key] || !decisions[i.key].value;});
        } else if (statusFilter === 'resolved') {
            items = items.filter(function(i){return decisions[i.key] && decisions[i.key].value;});
        }

        if (items.length === 0) return;

        var severity = items[0].severity;
        var actCfg = ACTIONS_BY_RULE[rule] || ACTIONS_BY_RULE['_default'];
        var colCfg = RULE_COLUMNS[rule] || DEFAULT_COLUMNS;

        var section = document.createElement('div');
        section.className = 'section';
        var ruleInfo = RULE_DESCRIPTIONS[rule] || {};
        var desc = (typeof ruleInfo === 'string') ? ruleInfo : (ruleInfo.description || '');
        var explanation = (typeof ruleInfo === 'object') ? (ruleInfo.explanation || '') : '';
        var guidance = (typeof ruleInfo === 'object') ? (ruleInfo.guidance || '') : '';
        var extraId = 'ruleextra-' + rule.replace(/[^a-zA-Z0-9]/g, '_');

        section.innerHTML =
            '<div class="section-header">' +
                '<div>' +
                    '<span class="section-title">' + rule + '</span>' +
                    '<span class="severity-badge ' + severity.toLowerCase() + '">' + severity + '</span>' +
                    '<span class="section-count">' + items.length + ' edge' + (items.length !== 1 ? 's' : '') + '</span>' +
                '</div>' +
            '</div>' +
            '<div class="rule-info">' +
                '<div class="rule-description">' + desc +
                    (explanation ? '<span class="why-toggle" data-toggle-extra="' + extraId + '">[Why?]</span>' : '') +
                '</div>' +
                (explanation ? '<div class="rule-extra" id="' + extraId + '">' +
                    '<div class="rule-explanation">' + explanation + '</div>' +
                    (guidance ? '<div class="rule-guidance"><strong>Decision guidance:</strong> ' + guidance + '</div>' : '') +
                '</div>' : '') +
            '</div>';

        // Bulk action (for sections with >3 items)
        if (items.length > 3 && actCfg) {
            var bulk = document.createElement('div');
            bulk.className = 'bulk-action';
            var optHtml = '<option value="">-- select --</option>';
            actCfg.options.forEach(function(o) { optHtml += '<option value="' + o.value + '">' + o.label + '</option>'; });
            bulk.innerHTML =
                '<label>Bulk action:</label>' +
                '<select id="bulk-' + rule + '">' + optHtml + '</select>' +
                '<button class="btn-sm" data-rule="' + rule + '">Apply to all unresolved</button>';
            (function(r, itms) {
                bulk.querySelector('button').addEventListener('click', function() {
                    var val = document.getElementById('bulk-' + r).value;
                    if (!val) return;
                    itms.forEach(function(item) {
                        if (!decisions[item.key] || !decisions[item.key].value) {
                            decisions[item.key] = {value: val, rationale: (decisions[item.key] && decisions[item.key].rationale) || ''};
                            var sel = document.querySelector('select[data-key="' + item.key + '"]');
                            if (sel) { sel.value = val; sel.classList.add('decided'); }
                            var row = document.querySelector('tr[data-key="' + item.key + '"]');
                            if (row) row.classList.add('resolved');
                        }
                    });
                    updateProgress();
                });
            })(rule, items);
            section.appendChild(bulk);
        }

        // Build table
        var table = document.createElement('table');
        table.innerHTML = '<thead><tr>' + colCfg.headers + '</tr></thead><tbody></tbody>';
        var tbody = table.querySelector('tbody');

        items.forEach(function(item) {
            var edge = EDGES[item.edge_id] || {};
            var isResolved = decisions[item.key] && decisions[item.key].value;
            var decVal = isResolved ? decisions[item.key].value : '';
            var tr = document.createElement('tr');
            tr.setAttribute('data-key', item.key);
            if (isResolved) tr.classList.add('resolved');

            tr.innerHTML = colCfg.renderRow(item, edge, actCfg, isResolved, decVal);
            tbody.appendChild(tr);

            // Details row with decision guide
            var detailTr = document.createElement('tr');
            detailTr.className = 'details-row';
            var safeId = item.key.replace(/[^a-zA-Z0-9]/g, '_');
            var dagEdge = DAG_EDGES[item.edge_id] || {};
            var guideFn = DECISION_GUIDES[item.rule_id] || defaultDecisionGuide;
            var guideHtml = guideFn(item, edge, dagEdge);
            detailTr.innerHTML = '<td colspan="' + colCfg.colspan + '">' +
                '<div class="details-content" id="detail-' + safeId + '">' +
                    guideHtml +
                    '<div class="detail-row"><span class="detail-label">Edge ID</span><span class="detail-value">' + item.edge_id + '</span></div>' +
                    '<div class="detail-row"><span class="detail-label">Design</span><span class="detail-value">' + (edge.design || '--') + '</span></div>' +
                    '<div class="detail-row"><span class="detail-label">Point (SE)</span><span class="detail-value">' + fmtNum(edge.point) + ' (' + fmtNum(edge.se) + ')</span></div>' +
                    '<div class="detail-row"><span class="detail-label">p-value</span><span class="detail-value">' + fmtP(edge.pvalue) + '</span></div>' +
                    '<div class="detail-row"><span class="detail-label">Claim level</span><span class="detail-value">' + (edge.claim_level || '--') + '</span></div>' +
                    '<div class="detail-row"><span class="detail-label">Rating / Score</span><span class="detail-value">' + (edge.rating || '--') + ' / ' + (edge.score ? edge.score.toFixed(3) : '--') + '</span></div>' +
                    '<div class="detail-row"><span class="detail-label">N obs</span><span class="detail-value">' + (edge.n_obs || '--') + '</span></div>' +
                    (edge.panel ? '<div class="detail-row"><span class="detail-label">Panel</span><span class="detail-value">' + edge.panel.n_units + ' units x ' + edge.panel.n_periods + ' periods (' + edge.panel.balance + ')</span></div>' : '') +
                    '<div class="detail-row"><span class="detail-label">All diagnostics pass</span><span class="detail-value">' + (edge.all_pass ? 'Yes' : 'No') + '</span></div>' +
                    '<div style="margin-top:4px">' + (edge.diagnostics ? diagDetail(edge.diagnostics) : '--') + '</div>' +
                    (edge.failure_flags && edge.failure_flags.length > 0 ? '<div class="detail-row" style="margin-top:4px"><span class="detail-label">Flags</span><span class="detail-value">' + edge.failure_flags.join(', ') + '</span></div>' : '') +
                    '<div class="detail-row" style="margin-top:4px"><span class="detail-label">Issue</span><span class="detail-value">' + item.message + '</span></div>' +
                '</div>' +
            '</td>';
            tbody.appendChild(detailTr);
        });

        section.appendChild(table);
        container.appendChild(section);
    });

    // Attach event listeners
    container.querySelectorAll('select[data-key]').forEach(function(sel) {
        sel.addEventListener('change', function() {
            var key = this.getAttribute('data-key');
            if (!decisions[key]) decisions[key] = {value: '', rationale: ''};
            decisions[key].value = this.value;
            var row = document.querySelector('tr[data-key="' + key + '"]');
            if (this.value) {
                this.classList.add('decided');
                if (row) row.classList.add('resolved');
            } else {
                this.classList.remove('decided');
                if (row) row.classList.remove('resolved');
            }
            // Update tooltip display
            try {
                var tooltips = JSON.parse(this.getAttribute('data-tooltips') || '{}');
                var tipEl = document.querySelector('[data-tip-for="' + key + '"]');
                if (tipEl) tipEl.textContent = tooltips[this.value] || '';
            } catch(e) {}
            updateProgress();
        });
    });

    container.querySelectorAll('input[data-rationale]').forEach(function(inp) {
        inp.addEventListener('input', function() {
            var key = this.getAttribute('data-rationale');
            if (!decisions[key]) decisions[key] = {value: '', rationale: ''};
            decisions[key].rationale = this.value;
        });
    });

    container.querySelectorAll('.toggle-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            var key = this.getAttribute('data-toggle');
            var id = 'detail-' + key.replace(/[^a-zA-Z0-9]/g, '_');
            var el = document.getElementById(id);
            if (el) {
                el.classList.toggle('open');
                this.textContent = el.classList.contains('open') ? '[-]' : '[+]';
            }
        });
    });

    // Why? toggle for rule explanation
    container.querySelectorAll('.why-toggle').forEach(function(btn) {
        btn.addEventListener('click', function() {
            var id = this.getAttribute('data-toggle-extra');
            var el = document.getElementById(id);
            if (el) {
                el.classList.toggle('open');
                this.textContent = el.classList.contains('open') ? '[Hide]' : '[Why?]';
            }
        });
    });

    // Restore decided styles
    for (var key in decisions) {
        if (decisions[key].value) {
            var sel = container.querySelector('select[data-key="' + key + '"]');
            if (sel) { sel.value = decisions[key].value; sel.classList.add('decided'); }
        }
    }

    updateProgress();
}

// ── Export ────────────────────────────────────────────────────────────
document.getElementById('exportBtn').addEventListener('click', function() {
    var analystId = document.getElementById('analystId').value.trim();
    var now = new Date().toISOString();
    var decisionList = [];

    for (var key in decisions) {
        var dec = decisions[key];
        if (!dec.value) continue;
        var issue = ISSUES[key];
        var edgeId = extractEdgeId(key);
        var actCfg = ACTIONS_BY_RULE[issue.rule_id] || ACTIONS_BY_RULE['_default'];
        decisionList.push({
            issue_key: key,
            edge_id: edgeId,
            rule_id: issue.rule_id,
            action: actCfg ? actCfg.action_type : 'unknown',
            value: dec.value,
            rationale: dec.rationale || '',
            status: 'CLOSED'
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
    a.download = 'hitl_decisions_' + dateStr + '.json';
    a.click();
    URL.revokeObjectURL(url);
});

// ── Filter listeners ──────────────────────────────────────────────────
document.getElementById('filterSeverity').addEventListener('change', render);
document.getElementById('filterRule').addEventListener('change', render);
document.getElementById('filterStatus').addEventListener('change', render);

// ── Init ──────────────────────────────────────────────────────────────
updateStats();
document.getElementById('genTimestamp').textContent = 'Generated: ' + GENERATED_AT;
render();

</script>
</body>
</html>"""


# ── Build ────────────────────────────────────────────────────────────────────


def build(
    state_path: Path,
    cards_dir: Path,
    actions_path: Path,
    registry_path: Path,
    output_dir: Path,
    dag_path: Path | None = None,
) -> Path:
    """Build the HITL panel HTML and return the output path."""
    if dag_path is None:
        dag_path = DEFAULT_DAG

    # 1. Load issues
    open_issues, closed_count = load_state(state_path)
    print(f"  {len(open_issues)} open issues, {closed_count} closed")

    # 2. Collect edge IDs from open issues
    edge_ids = set()
    for key in open_issues:
        edge_ids.add(extract_edge_id(key))

    # 3. Load edge summaries
    edges = {}
    for eid in sorted(edge_ids):
        card_path = cards_dir / f"{eid}.yaml"
        summary = load_edge_summary(card_path)
        if summary:
            edges[eid] = summary
        else:
            print(f"  Warning: no edge card for {eid}")
            edges[eid] = {
                "point": None, "se": None, "pvalue": None,
                "design": None, "rating": None, "score": None,
                "claim_level": None, "all_pass": False, "n_obs": None,
                "diagnostics": [], "failure_flags": [], "panel": None,
                "estimand": "", "is_not": [], "allowed_uses": [],
                "forbidden_uses": [], "cf_shock_allowed": None,
                "cf_policy_allowed": None, "cf_reason_blocked": "",
                "id_risks": {}, "untestable_assumptions": [],
                "testable_passed": [], "testable_failed": [],
                "propagation_role": "",
            }
    print(f"  {len(edges)} edge cards loaded")

    # 4. Load actions config
    actions = load_actions(actions_path)
    print(f"  {len(actions)} action rules loaded")

    # 5. Load rule info (description + explanation + guidance)
    rules = load_rule_info(registry_path)
    print(f"  {len(rules)} rule descriptions loaded")

    # 6. Load DAG edge/node metadata
    dag_edges = load_dag_edges(dag_path)
    dag_nodes = load_dag_nodes(dag_path)
    print(f"  {len(dag_edges)} DAG edges, {len(dag_nodes)} DAG nodes loaded")

    # 7. Build HTML via placeholder replacement
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html = HTML_TEMPLATE
    html = html.replace("__ISSUES_JSON__", json.dumps(open_issues, indent=2))
    html = html.replace("__EDGES_JSON__", json.dumps(edges, indent=2))
    html = html.replace("__ACTIONS_JSON__", json.dumps(actions, indent=2))
    html = html.replace("__RULES_JSON__", json.dumps(rules, indent=2))
    html = html.replace("__DAG_EDGES_JSON__", json.dumps(dag_edges, indent=2))
    html = html.replace("__DAG_NODES_JSON__", json.dumps(dag_nodes, indent=2))
    html = html.replace("__CLOSED_COUNT__", str(closed_count))
    html = html.replace("__GENERATED_AT__", generated_at)

    # 8. Write output
    output_path = output_dir / "hitl_panel.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"  Written: {output_path}")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build HITL Resolution Panel HTML from state.json and edge cards."
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE,
        help="Path to state.json (default: outputs/agentic/issues/state.json)",
    )
    parser.add_argument(
        "--cards-dir",
        type=Path,
        default=DEFAULT_CARDS_DIR,
        help="Path to edge_cards directory (default: outputs/agentic/cards/edge_cards)",
    )
    parser.add_argument(
        "--actions-file",
        type=Path,
        default=DEFAULT_ACTIONS,
        help="Path to hitl_actions.yaml (default: config/agentic/hitl_actions.yaml)",
    )
    parser.add_argument(
        "--registry-file",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Path to issue_registry.yaml (default: config/agentic/issue_registry.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for hitl_panel.html (default: outputs/agentic)",
    )
    parser.add_argument(
        "--dag-file",
        type=Path,
        default=DEFAULT_DAG,
        help="Path to DAG YAML for edge/node context (default: config/agentic/dags/kspi_k2_full.yaml)",
    )
    args = parser.parse_args()

    print("Building HITL panel...")
    build(args.state_file, args.cards_dir, args.actions_file, args.registry_file, args.output_dir, args.dag_file)
    print("Done.")


if __name__ == "__main__":
    main()
