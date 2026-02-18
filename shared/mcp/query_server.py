"""
OpenCausality MCP Query Server — stdio transport.

Exposes DAG propagation, falsification, edge inspection, and health-check
tools to Claude Code via the Model Context Protocol.

Usage:
    python -m shared.mcp.query_server
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server state (populated by load_dag)
#
# IMPORTANT: Each call to load_dag fully replaces all state — the DAG, edge
# cards, TSGuard results, issue ledger, and propagation engine are all scoped
# to the most recently loaded DAG.  Loading a different DAG discards the
# previous state entirely.  There is no cross-DAG state sharing.
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "dag": None,
    "edge_cards": {},
    "tsguard_results": {},
    "issue_ledger": None,
    "engine": None,
    "mode": "REDUCED_FORM",
    "dag_path": None,
}

VALID_MODES = {"STRUCTURAL", "REDUCED_FORM", "DESCRIPTIVE"}


def _require_dag() -> str | None:
    """Return an error message if DAG is not loaded, else None."""
    if _state["dag"] is None:
        return "DAG not loaded. Call load_dag first."
    return None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _safe_float(v: float) -> float | str:
    """Convert NaN/Inf to string for JSON safety."""
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    return round(v, 6)


def _edge_in_path_to_dict(e: Any) -> dict:
    return {
        "edge_id": e.edge_id,
        "from": e.from_node,
        "to": e.to_node,
        "coefficient": _safe_float(e.coefficient),
        "se": _safe_float(e.se),
        "role": e.role,
        "claim_level": e.claim_level,
        "treatment_unit": e.treatment_unit.kind,
        "treatment_scale": e.treatment_unit.scale,
    }


def _propagation_result_to_dict(result: Any) -> dict:
    paths = []
    for p in result.paths:
        paths.append({
            "total_effect": _safe_float(p.total_effect),
            "total_se": _safe_float(p.total_se),
            "ci_lower": _safe_float(p.ci_95[0]),
            "ci_upper": _safe_float(p.ci_95[1]),
            "se_method": p.se_method,
            "is_blocked": p.is_blocked,
            "blocked_reasons": p.blocked_reasons,
            "warnings": p.warnings,
            "edges": [_edge_in_path_to_dict(e) for e in p.edges],
        })
    blocked = [
        {"edge_id": b.edge_id, "from": b.from_node, "to": b.to_node, "reason": b.reason}
        for b in result.blocked_edges
    ]
    return {
        "paths": paths,
        "blocked_edges": blocked,
        "warnings": result.warnings,
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _get_dag_output_dir(dag_path: Path) -> Path:
    """Derive per-DAG output directory from the DAG YAML path.

    Mirrors the logic in scripts/cli.py:get_dag_output_dir() so that the
    MCP server reads from the same artifact directories as the CLI.
    The returned path is absolute (anchored to project root) so that it
    works regardless of the MCP server's working directory.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    slug = dag_path.stem  # e.g. "kspi_k2_full"
    return project_root / "outputs" / "agentic" / slug


def _do_load_dag(dag_path: str) -> dict:
    """Load and initialise DAG, edge cards, TSGuard, issues, and engine."""
    # Resolve path relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    dp = Path(dag_path)
    if not dp.is_absolute():
        dp = project_root / dp

    if not dp.exists():
        return {"error": f"DAG file not found: {dp}"}

    from shared.agentic.dag.parser import parse_dag
    dag = parse_dag(dp)
    _state["dag"] = dag
    _state["dag_path"] = dp

    # Per-DAG artifact directory (matches CLI output layout)
    dag_out = _get_dag_output_dir(dp)

    # Edge cards — per-DAG: outputs/agentic/{dag_stem}/cards/edge_cards/
    cards_dir = dag_out / "cards" / "edge_cards"
    edge_cards: dict[str, Any] = {}
    if cards_dir.exists():
        from shared.agentic.artifact_store import ArtifactStore
        store = ArtifactStore(cards_dir.parent)
        for card in store.get_all_edge_cards():
            edge_cards[card.edge_id] = card
    _state["edge_cards"] = edge_cards

    # TSGuard results — per-DAG: outputs/agentic/{dag_stem}/tsguard/
    import yaml
    tsguard_dir = dag_out / "tsguard"
    tsguard_results: dict[str, Any] = {}
    if tsguard_dir.exists():
        from shared.agentic.ts_guard import TSGuardResult
        for f in tsguard_dir.glob("*.yaml"):
            try:
                with open(f) as fh:
                    data = yaml.safe_load(fh)
                if data and "edge_id" in data:
                    result = TSGuardResult(
                        edge_id=data["edge_id"],
                        dynamics_risk=data.get("dynamics_risk", {}),
                        diagnostics_results=data.get("diagnostics_results", {}),
                        claim_level_cap=data.get("claim_level_cap"),
                        counterfactual_blocked=data.get("counterfactual_blocked", False),
                    )
                    tsguard_results[data["edge_id"]] = result
            except Exception:
                pass
    _state["tsguard_results"] = tsguard_results

    # Issue ledger — per-DAG: outputs/agentic/{dag_stem}/issues/
    issues_dir = dag_out / "issues"
    issue_ledger = None
    if issues_dir.exists():
        from shared.agentic.issues.issue_ledger import IssueLedger
        jsonl_files = sorted(issues_dir.glob("*.jsonl"))
        if jsonl_files:
            ledger = IssueLedger()
            issues = ledger.load_from_file(jsonl_files[-1])
            ledger.issues = issues
            issue_ledger = ledger
    _state["issue_ledger"] = issue_ledger

    # Propagation engine
    from shared.agentic.propagation import PropagationEngine
    engine = PropagationEngine(
        dag=dag,
        edge_cards=edge_cards,
        tsguard_results=tsguard_results,
        issue_ledger=issue_ledger,
    )
    _state["engine"] = engine

    return {
        "name": dag.metadata.name,
        "target": dag.metadata.target_node or "",
        "nodes": [
            {"id": n.id, "name": n.name, "frequency": n.frequency, "type": n.type}
            for n in dag.nodes
        ],
        "edges": [
            {
                "id": e.id,
                "from": e.from_node,
                "to": e.to_node,
                "type": e.get_edge_type(),
                "claim": edge_cards[e.id].identification.claim_level
                if e.id in edge_cards else "",
            }
            for e in dag.edges
        ],
        "n_cards": len(edge_cards),
        "n_issues": len(issue_ledger.issues) if issue_ledger else 0,
        "cards_not_on_dag": [
            cid for cid in edge_cards if cid not in {e.id for e in dag.edges}
        ],
    }


def _do_list_nodes() -> list[dict]:
    dag = _state["dag"]
    return [
        {"id": n.id, "name": n.name, "frequency": n.frequency, "type": n.type}
        for n in dag.nodes
    ]


def _do_list_edges(mode: str | None) -> list[dict]:
    mode = mode or _state["mode"]
    engine = _state["engine"]
    return engine.get_edges_summary(mode)


def _do_switch_mode(mode: str) -> dict:
    mode = mode.upper()
    if mode not in VALID_MODES:
        return {"error": f"Invalid mode: {mode}. Must be one of {sorted(VALID_MODES)}."}
    previous = _state["mode"]
    _state["mode"] = mode
    return {"previous_mode": previous, "new_mode": mode}


def _do_propagate(
    source: str,
    target: str,
    magnitude: float,
    unit: str,
    mode: str | None,
    scenario_type: str,
) -> dict:
    mode = mode or _state["mode"]
    engine = _state["engine"]
    dag = _state["dag"]

    # Default target to DAG target
    if not target:
        target = dag.metadata.target_node or ""
    if not target:
        return {"error": "No target node specified and DAG has no default target."}

    result = engine.find_all_paths(
        source=source,
        target=target,
        mode=mode,
        scenario_type=scenario_type,
    )
    out = _propagation_result_to_dict(result)
    out["source"] = source
    out["target"] = target
    out["magnitude"] = magnitude
    out["unit"] = unit
    out["mode"] = mode
    out["scenario_type"] = scenario_type

    # Scale effects by magnitude for convenience
    for p in out["paths"]:
        if not p["is_blocked"] and magnitude != 0:
            p["scaled_effect"] = _safe_float(p["total_effect"] * magnitude)
            scaled_se = p["total_se"] * abs(magnitude)
            p["scaled_se"] = _safe_float(scaled_se)
            eff = p["total_effect"] * magnitude
            p["scaled_ci_lower"] = _safe_float(eff - 1.96 * scaled_se)
            p["scaled_ci_upper"] = _safe_float(eff + 1.96 * scaled_se)

    return out


def _do_find_paths(source: str, target: str, mode: str | None) -> dict:
    mode = mode or _state["mode"]
    engine = _state["engine"]
    result = engine.find_all_paths(source=source, target=target, mode=mode)
    return _propagation_result_to_dict(result)


def _do_target_contributors(target: str, mode: str | None) -> list[dict]:
    mode = mode or _state["mode"]
    engine = _state["engine"]
    node_ids = engine.get_all_node_ids()

    contributors = []
    for src in node_ids:
        if src == target:
            continue
        try:
            result = engine.find_all_paths(source=src, target=target, mode=mode)
            for path in result.paths:
                if not path.is_blocked:
                    contributors.append({
                        "source": src,
                        "effect": _safe_float(path.total_effect),
                        "se": _safe_float(path.total_se),
                        "ci_lower": _safe_float(path.ci_95[0]),
                        "ci_upper": _safe_float(path.ci_95[1]),
                        "path": " -> ".join(e.edge_id for e in path.edges),
                    })
        except Exception:
            continue

    # Sort by absolute effect
    contributors.sort(key=lambda c: abs(c["effect"]) if isinstance(c["effect"], (int, float)) else 0, reverse=True)
    for i, c in enumerate(contributors, 1):
        c["rank"] = i
    return contributors


def _do_inspect_edge(edge_id: str) -> dict:
    edge_cards = _state["edge_cards"]
    card = edge_cards.get(edge_id)
    if card is None:
        available = list(edge_cards.keys())[:20]
        return {"error": f"No edge card for '{edge_id}'.", "available": available}

    result: dict[str, Any] = {
        "edge_id": edge_id,
        "credibility_rating": card.credibility_rating,
        "credibility_score": _safe_float(card.credibility_score),
    }

    # Estimates
    if card.estimates:
        est = card.estimates
        result["estimates"] = {
            "point": _safe_float(est.point),
            "se": _safe_float(est.se),
            "ci_95": [_safe_float(est.ci_95[0]), _safe_float(est.ci_95[1])],
            "pvalue": _safe_float(est.pvalue) if est.pvalue is not None else None,
            "treatment_unit": est.treatment_unit,
            "outcome_unit": est.outcome_unit,
        }

    # Identification
    result["identification"] = card.identification.to_dict()

    # Counterfactual
    result["counterfactual"] = card.counterfactual_block.to_dict()

    # TSGuard
    ts = _state["tsguard_results"].get(edge_id)
    if ts:
        result["tsguard"] = {
            "has_any_high_risk": ts.has_any_high_risk,
            "dynamics_risk": ts.dynamics_risk,
            "counterfactual_blocked": ts.counterfactual_blocked,
        }

    # Open issues
    issue_ledger = _state["issue_ledger"]
    if issue_ledger:
        edge_issues = [
            i for i in issue_ledger.issues
            if i.edge_id == edge_id and i.is_open
        ]
        if edge_issues:
            result["open_issues"] = [
                {
                    "severity": iss.severity,
                    "rule_id": iss.rule_id,
                    "message": iss.message,
                }
                for iss in edge_issues
            ]

    # Literature
    lit = card.literature
    if lit.total_results > 0:
        result["literature"] = {
            "total_results": lit.total_results,
            "search_status": lit.search_status,
            "supporting": lit.supporting[:5],
            "challenging": lit.challenging[:5],
        }

    return result


def _do_get_identification() -> list[dict]:
    dag = _state["dag"]
    edge_cards = _state["edge_cards"]
    rows = []
    for edge in dag.edges:
        card = edge_cards.get(edge.id)
        if card is None:
            rows.append({"edge_id": edge.id, "claim_level": "no card", "shock_cf": False, "policy_cf": False, "risks": {}})
            continue
        rows.append({
            "edge_id": edge.id,
            "claim_level": card.identification.claim_level or "-",
            "shock_cf": card.counterfactual_block.shock_scenario_allowed,
            "policy_cf": card.counterfactual_block.policy_intervention_allowed,
            "risks": {k: v for k, v in card.identification.risks.items() if v in ("medium", "high")},
        })
    return rows


def _do_compare_modes() -> list[dict]:
    engine = _state["engine"]
    dag = _state["dag"]
    modes = ["STRUCTURAL", "REDUCED_FORM", "DESCRIPTIVE"]

    # Cache summaries
    summaries_by_mode: dict[str, dict[str, dict]] = {}
    for m in modes:
        sums = engine.get_edges_summary(m)
        summaries_by_mode[m] = {s["edge_id"]: s for s in sums}

    rows = []
    for edge in dag.edges:
        row: dict[str, Any] = {"edge_id": edge.id}
        for m in modes:
            s = summaries_by_mode[m].get(edge.id)
            if s:
                row[m] = {"allowed": s["allowed"], "role": s["role"]}
            else:
                row[m] = {"allowed": False, "role": "-"}
        rows.append(row)
    return rows


def _do_run_placebo(max_tests: int, alpha: float) -> dict:
    from shared.agentic.falsification import PlaceboFalsifier

    dag = _state["dag"]
    issue_ledger = _state["issue_ledger"]

    falsifier = PlaceboFalsifier(
        dag=dag,
        max_tests=max_tests,
        alpha=alpha,
        issue_ledger=issue_ledger,
    )
    results = falsifier.run()

    tested = sum(1 for r in results if r.data_status != "missing")
    failed = sum(1 for r in results if not r.passed and r.data_status != "missing")
    skipped = sum(1 for r in results if r.data_status == "missing")

    return {
        "tested": tested,
        "failed": failed,
        "skipped": skipped,
        "results": [
            {
                "from": r.from_node,
                "to": r.to_node,
                "conditioning_set": r.conditioning_set,
                "coefficient": _safe_float(r.coefficient),
                "se": _safe_float(r.se),
                "pvalue": _safe_float(r.pvalue),
                "n_obs": r.n_obs,
                "passed": r.passed,
                "data_status": r.data_status,
                "message": r.message,
            }
            for r in results
        ],
    }


def _do_dag_doctor() -> dict:
    dag = _state["dag"]
    edge_cards = _state["edge_cards"]
    issue_ledger = _state["issue_ledger"]
    tsguard_results = _state["tsguard_results"]

    edges_without_cards = [e.id for e in dag.edges if e.id not in edge_cards]

    return {
        "dag_name": dag.metadata.name,
        "target_node": dag.metadata.target_node or "",
        "nodes": len(dag.nodes),
        "edges": len(dag.edges),
        "cards": len(edge_cards),
        "card_coverage": f"{len(edge_cards)}/{len(dag.edges)}",
        "tsguard_results": len(tsguard_results),
        "issues_total": len(issue_ledger.issues) if issue_ledger else 0,
        "issues_critical": len(issue_ledger.get_critical_open()) if issue_ledger else 0,
        "edges_without_cards": edges_without_cards,
        "cards_not_on_dag": [
            cid for cid in edge_cards if cid not in {e.id for e in dag.edges}
        ],
        "current_mode": _state["mode"],
    }


# ---------------------------------------------------------------------------
# MCP Server setup
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[Tool] = [
    Tool(
        name="load_dag",
        description=(
            "Load a DAG YAML specification file and initialise the query engine. "
            "Must be called before any other tool. Loads edge cards, TSGuard results, "
            "and the issue ledger from the project outputs directory."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "dag_path": {
                    "type": "string",
                    "description": "Path to DAG YAML file (absolute or relative to project root).",
                },
            },
            "required": ["dag_path"],
        },
    ),
    Tool(
        name="list_nodes",
        description="List all nodes in the loaded DAG with their id, name, frequency, and type.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="list_edges",
        description=(
            "List all edges in the DAG with their propagation role, claim level, "
            "and whether they are allowed in the given query mode."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Query mode (STRUCTURAL, REDUCED_FORM, DESCRIPTIVE). Defaults to current mode.",
                },
            },
        },
    ),
    Tool(
        name="switch_mode",
        description=(
            "Switch the active query mode. Modes control which edges are allowed in propagation: "
            "STRUCTURAL (strictest, mechanism-based), REDUCED_FORM (statistical associations), "
            "DESCRIPTIVE (all edges including correlations)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["STRUCTURAL", "REDUCED_FORM", "DESCRIPTIVE"],
                    "description": "The mode to switch to.",
                },
            },
            "required": ["mode"],
        },
    ),
    Tool(
        name="propagate_shock",
        description=(
            "Propagate a shock scenario through the DAG from source to target. "
            "Computes causal effect paths with full guardrails (mode gating, "
            "counterfactual eligibility, TSGuard, issue ledger, unit compatibility). "
            "Use for questions like 'What if oil drops 30%?'"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source node ID."},
                "target": {"type": "string", "description": "Target node ID. Defaults to DAG target."},
                "magnitude": {"type": "number", "description": "Shock magnitude (e.g. -0.30 for -30%)."},
                "unit": {"type": "string", "description": "Unit of shock (pct, pp, sd, bps). Defaults to sd."},
                "mode": {"type": "string", "description": "Query mode override. Defaults to current mode."},
            },
            "required": ["source", "magnitude"],
        },
    ),
    Tool(
        name="propagate_policy",
        description=(
            "Propagate a policy intervention scenario through the DAG. "
            "Policy counterfactuals require STRUCTURAL mode and IDENTIFIED_CAUSAL claim level. "
            "Use for questions like 'What if the central bank raises rates 100bps?'"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source node ID (policy instrument)."},
                "target": {"type": "string", "description": "Target node ID. Defaults to DAG target."},
                "magnitude": {"type": "number", "description": "Policy change magnitude."},
                "unit": {"type": "string", "description": "Unit (pct, pp, sd, bps). Defaults to sd."},
                "mode": {"type": "string", "description": "Query mode override. Defaults to current mode."},
            },
            "required": ["source", "magnitude"],
        },
    ),
    Tool(
        name="find_paths",
        description=(
            "Find all propagation paths between two nodes in the DAG. "
            "Shows which paths are allowed and which are blocked, with reasons."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source node ID."},
                "target": {"type": "string", "description": "Target node ID."},
                "mode": {"type": "string", "description": "Query mode override."},
            },
            "required": ["source", "target"],
        },
    ),
    Tool(
        name="target_contributors",
        description=(
            "List all source nodes with active paths to a target node, "
            "ranked by absolute effect size. Shows which variables are the "
            "biggest contributors/drivers."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Target node ID."},
                "mode": {"type": "string", "description": "Query mode override."},
            },
            "required": ["target"],
        },
    ),
    Tool(
        name="inspect_edge",
        description=(
            "Get full details of an edge card: credibility rating, estimates, "
            "identification claim level, counterfactual eligibility, TSGuard flags, "
            "open issues, and literature citations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "edge_id": {"type": "string", "description": "Edge ID (e.g. 'brent_to_usdkzt')."},
            },
            "required": ["edge_id"],
        },
    ),
    Tool(
        name="get_identification",
        description=(
            "Get identification summary for all edges: claim level, "
            "shock/policy counterfactual eligibility, and medium/high risks."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="compare_modes",
        description=(
            "Compare edge permissions across all three query modes. "
            "Shows which edges are allowed and their roles in each mode."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="run_placebo",
        description=(
            "Run placebo falsification tests on null links (missing edges) in the DAG. "
            "Tests the Markov property: for non-edges A->B, regresses B on A "
            "conditioning on parents(B). Significant results suggest missing edges."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "max_tests": {
                    "type": "integer",
                    "description": "Maximum number of null links to test. Defaults to 20.",
                },
                "alpha": {
                    "type": "number",
                    "description": "Significance threshold. Defaults to 0.05.",
                },
            },
        },
    ),
    Tool(
        name="dag_doctor",
        description=(
            "Run a health check on the loaded DAG. Reports node/edge counts, "
            "card coverage, TSGuard results, issue counts, and edges without cards."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
]


async def serve() -> None:
    server = Server("opencausality-query")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOL_DEFINITIONS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            result = _dispatch_tool(name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            logger.exception(f"Tool {name} failed")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def _dispatch_tool(name: str, arguments: dict) -> Any:
    """Synchronous dispatch to tool implementations."""

    if name == "load_dag":
        return _do_load_dag(arguments["dag_path"])

    # All other tools require DAG to be loaded
    err = _require_dag()
    if err:
        return {"error": err}

    if name == "list_nodes":
        return _do_list_nodes()
    elif name == "list_edges":
        return _do_list_edges(arguments.get("mode"))
    elif name == "switch_mode":
        return _do_switch_mode(arguments["mode"])
    elif name == "propagate_shock":
        return _do_propagate(
            source=arguments["source"],
            target=arguments.get("target", ""),
            magnitude=arguments["magnitude"],
            unit=arguments.get("unit", "sd"),
            mode=arguments.get("mode"),
            scenario_type="shock",
        )
    elif name == "propagate_policy":
        return _do_propagate(
            source=arguments["source"],
            target=arguments.get("target", ""),
            magnitude=arguments["magnitude"],
            unit=arguments.get("unit", "sd"),
            mode=arguments.get("mode"),
            scenario_type="policy",
        )
    elif name == "find_paths":
        return _do_find_paths(
            source=arguments["source"],
            target=arguments["target"],
            mode=arguments.get("mode"),
        )
    elif name == "target_contributors":
        return _do_target_contributors(
            target=arguments["target"],
            mode=arguments.get("mode"),
        )
    elif name == "inspect_edge":
        return _do_inspect_edge(arguments["edge_id"])
    elif name == "get_identification":
        return _do_get_identification()
    elif name == "compare_modes":
        return _do_compare_modes()
    elif name == "run_placebo":
        return _do_run_placebo(
            max_tests=arguments.get("max_tests", 20),
            alpha=arguments.get("alpha", 0.05),
        )
    elif name == "dag_doctor":
        return _do_dag_doctor()
    else:
        return {"error": f"Unknown tool: {name}"}


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    asyncio.run(serve())


if __name__ == "__main__":
    main()
