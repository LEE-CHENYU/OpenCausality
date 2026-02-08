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

            // Details row
            var detailTr = document.createElement('tr');
            detailTr.className = 'details-row';
            var safeId = item.key.replace(/[^a-zA-Z0-9]/g, '_');
            detailTr.innerHTML = '<td colspan="' + colCfg.colspan + '">' +
                '<div class="details-content" id="detail-' + safeId + '">' +
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
) -> Path:
    """Build the HITL panel HTML and return the output path."""
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
            }
    print(f"  {len(edges)} edge cards loaded")

    # 4. Load actions config
    actions = load_actions(actions_path)
    print(f"  {len(actions)} action rules loaded")

    # 5. Load rule info (description + explanation + guidance)
    rules = load_rule_info(registry_path)
    print(f"  {len(rules)} rule descriptions loaded")

    # 6. Build HTML via placeholder replacement
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html = HTML_TEMPLATE
    html = html.replace("__ISSUES_JSON__", json.dumps(open_issues, indent=2))
    html = html.replace("__EDGES_JSON__", json.dumps(edges, indent=2))
    html = html.replace("__ACTIONS_JSON__", json.dumps(actions, indent=2))
    html = html.replace("__RULES_JSON__", json.dumps(rules, indent=2))
    html = html.replace("__CLOSED_COUNT__", str(closed_count))
    html = html.replace("__GENERATED_AT__", generated_at)

    # 7. Write output
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
    args = parser.parse_args()

    print("Building HITL panel...")
    build(args.state_file, args.cards_dir, args.actions_file, args.registry_file, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
