#!/usr/bin/env python3
"""
Generate a DAG from Natural Language Narrative.

Reads a narrative text, extracts causal claims via LLM, matches to an existing
DAG, proposes new nodes for unmatched concepts, and writes a new DAG YAML.

Usage:
    python scripts/generate_narrative_dag.py
    python scripts/generate_narrative_dag.py --narrative examples/kaspi_narrative.txt
    python scripts/generate_narrative_dag.py --base-dag config/agentic/dags/kspi_k2_full.yaml --out output.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_NARRATIVE = PROJECT_ROOT / "examples" / "kaspi_narrative.txt"
DEFAULT_BASE_DAG = PROJECT_ROOT / "config" / "agentic" / "dags" / "kspi_k2_full.yaml"
DEFAULT_OUT = PROJECT_ROOT / "config" / "agentic" / "dags" / "kspi_k2_narrative.yaml"

NEW_NODE_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "unit": {"type": "string"},
                    "frequency": {"type": "string"},
                    "type": {"type": "string"},
                },
                "required": ["id", "name", "description"],
            },
        },
    },
}


@dataclass
class NarrativePaper:
    """Simple wrapper to satisfy PaperDAGExtractor's paper interface."""
    title: str = "Narrative Text"
    authors: list[str] = field(default_factory=lambda: ["analyst"])
    excerpt: str = ""
    doi: str = ""
    year: str = ""


def _canonical_id(name: str) -> str:
    """Generate a canonical node ID from a human-readable name."""
    return re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")


def _sanitize_edge_id(raw_id: str) -> str:
    """Sanitize edge ID to snake_case (no spaces, arrows, parentheses).

    Converts ``->`` to ``_to_``, strips parens / special chars,
    collapses consecutive underscores.
    """
    cleaned = raw_id.replace("->", "_to_").replace("→", "_to_")
    cleaned = re.sub(r"[^a-z0-9_]", "_", cleaned.lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _dedup_id(candidate: str, existing_ids: set[str]) -> str:
    """Return a unique ID, appending _2 suffix on collision."""
    if candidate not in existing_ids:
        return candidate
    return f"{candidate}_2"


def _generate_new_nodes(
    llm: Any, new_node_names: list[str], existing_ids: set[str],
) -> list[Any]:
    """Use LLM to generate NodeSpec definitions for unmatched concepts."""
    from shared.agentic.dag.parser import NodeSpec

    existing_list = ", ".join(sorted(existing_ids))
    names_list = "\n".join(f"  - {n}" for n in new_node_names)
    system_prompt = (
        "You are an econometric DAG designer. Given a list of new variable "
        "concepts that do not exist in the current DAG, generate node "
        "definitions. Each node needs an id (snake_case), a human-readable "
        "name, a description, unit (e.g. level, percent, index, count), "
        "frequency (daily/monthly/quarterly/annual), and type "
        "(continuous/binary/count/share/index/exposure).\n\n"
        "IMPORTANT: Do NOT re-propose any of these existing node IDs:\n"
        f"  {existing_list}"
    )
    user_prompt = f"Generate node definitions for these new concepts:\n{names_list}"
    try:
        result = llm.complete_structured(system_prompt, user_prompt, NEW_NODE_SCHEMA)
    except Exception as e:
        logger.warning(f"LLM new-node generation failed: {e}")
        result = {"nodes": []}

    nodes: list[NodeSpec] = []
    for nd in result.get("nodes", []):
        raw_id = nd.get("id", _canonical_id(nd.get("name", "")))
        node_id = _dedup_id(_canonical_id(raw_id), existing_ids)
        existing_ids.add(node_id)
        nodes.append(NodeSpec(
            id=node_id, name=nd.get("name", node_id),
            description=nd.get("description", ""),
            unit=nd.get("unit", "level"), frequency=nd.get("frequency", "monthly"),
            type=nd.get("type", "continuous"), observed=True,
            tags=["narrative_generated"],
        ))
        logger.info(f"Generated new node: {node_id} ({nd.get('name', '')})")
    return nodes


def _build_dag(base_dag: Any, proposed_edges: list[Any], new_nodes: list[Any]) -> Any:
    """Build a new DAGSpec from proposed edges and new nodes."""
    from shared.agentic.dag.parser import (
        DAGMetadata, DAGSpec, EdgeSpec, EdgeTiming, EdgeInterpretation,
    )

    new_node_map = {n.id: n for n in new_nodes}

    # Resolve empty from_node/to_node using requires_new_nodes
    for pe in proposed_edges:
        if not pe.to_node and pe.requires_new_nodes:
            candidate = _canonical_id(pe.requires_new_nodes[0])
            if candidate in new_node_map:
                pe.to_node = candidate
        if not pe.from_node and pe.requires_new_nodes:
            candidate = _canonical_id(pe.requires_new_nodes[-1])
            if candidate in new_node_map:
                pe.from_node = candidate

    # Drop edges with unresolved empty endpoints
    proposed_edges = [pe for pe in proposed_edges if pe.from_node and pe.to_node]

    referenced_ids: set[str] = set()
    for pe in proposed_edges:
        # Handle multi-input from_node like "ppop_kspi + cor_kspi"
        for part in pe.from_node.split(" + "):
            referenced_ids.add(part.strip())
        referenced_ids.add(pe.to_node)

    existing_node_map = {n.id: n for n in base_dag.nodes}

    # Transitively include nodes referenced by depends_on / identity
    def _collect_deps(nid: str, collected: set[str]) -> None:
        if nid not in existing_node_map:
            return
        node = existing_node_map[nid]
        for dep in getattr(node, "depends_on", []) or []:
            if dep not in collected:
                collected.add(dep)
                _collect_deps(dep, collected)

    all_ids = set(referenced_ids)
    for nid in list(referenced_ids):
        _collect_deps(nid, all_ids)

    nodes = [existing_node_map[nid] for nid in all_ids if nid in existing_node_map]
    for nid in all_ids:
        if nid not in existing_node_map and nid in new_node_map:
            nodes.append(new_node_map[nid])

    # Build node-unit lookup for auto-populating unit_specification
    all_node_map = {**existing_node_map, **new_node_map}
    def _get_unit_spec(from_id: str, to_id: str) -> dict[str, str]:
        t_unit = getattr(all_node_map.get(from_id), "unit", "") or ""
        o_unit = getattr(all_node_map.get(to_id), "unit", "") or ""
        return {"treatment_unit": t_unit, "outcome_unit": o_unit}

    base_edge_map: dict[tuple[str, str], Any] = {
        (e.from_node, e.to_node): e for e in base_dag.edges
    }
    edges: list[EdgeSpec] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for pe in proposed_edges:
        key = (pe.from_node, pe.to_node)
        if pe.is_existing and key in base_edge_map:
            existing_edge = base_edge_map[key]
            # Backfill unit_specification if missing on existing edge
            if not existing_edge.unit_specification:
                existing_edge.unit_specification = _get_unit_spec(
                    pe.from_node, pe.to_node
                )
            edges.append(existing_edge)
            seen_ids.add(existing_edge.id)
            seen_pairs.add(key)
        else:
            edge_id = _sanitize_edge_id(
                pe.edge_id or f"{pe.from_node}_to_{pe.to_node}"
            )
            # Regenerate ID if it doesn't reflect actual from/to nodes
            if pe.from_node and pe.to_node:
                if pe.from_node not in edge_id or pe.to_node not in edge_id:
                    edge_id = f"{pe.from_node}_to_{pe.to_node}"
            if edge_id in seen_ids:
                edge_id = f"{edge_id}_nl"
            seen_ids.add(edge_id)
            seen_pairs.add(key)
            evidence_notes = "; ".join(
                f"{c.treatment}->{c.outcome} ({c.direction})" for c in pe.evidence
            )
            edges.append(EdgeSpec(
                id=edge_id, from_node=pe.from_node, to_node=pe.to_node,
                edge_type=pe.edge_type, notes=f"NL-extracted: {evidence_notes}",
                unit_specification=_get_unit_spec(pe.from_node, pe.to_node),
            ))

    # --- Auto-create identity edges for derived nodes ---
    node_id_set = {n.id for n in nodes}
    for node in nodes:
        if not getattr(node, "derived", False):
            continue
        identity = getattr(node, "identity", None)
        if not identity:
            continue
        deps = getattr(identity, "depends_on", None) or getattr(node, "depends_on", [])
        for dep_id in deps or []:
            if dep_id not in node_id_set:
                continue
            pair = (dep_id, node.id)
            if pair in seen_pairs:
                continue
            edge_id = f"{dep_id}_to_{node.id}"
            if edge_id in seen_ids:
                continue
            # Copy from expert DAG if available, otherwise create minimal identity edge
            if pair in base_edge_map:
                edges.append(base_edge_map[pair])
                seen_ids.add(base_edge_map[pair].id)
            else:
                seen_ids.add(edge_id)
                formula_name = getattr(identity, "name", "") or ""
                edges.append(EdgeSpec(
                    id=edge_id, from_node=dep_id, to_node=node.id,
                    edge_type="identity", edge_status="IDENTITY",
                    allowed_designs=[],
                    timing=EdgeTiming(lag=0, lag_unit="quarter",
                                     contemporaneous=True, max_anticipation=0),
                    interpretation=EdgeInterpretation(
                        is_field=f"Identity component: {formula_name}",
                        is_not=["Causal relationship"],
                        allowed_uses=["scenario_only"],
                    ),
                    notes=f"Auto-created identity edge from {formula_name}",
                    unit_specification=_get_unit_spec(dep_id, node.id),
                ))
            seen_pairs.add(pair)

    base_name = base_dag.metadata.name
    n_matched = sum(1 for p in proposed_edges if p.is_existing)
    metadata = DAGMetadata(
        name=f"{base_name}_narrative",
        description=(
            f"DAG generated from narrative text, based on {base_name}. "
            f"Contains {len(edges)} edges ({n_matched} matched, "
            f"{len(edges) - n_matched} new)."
        ),
        target_node=base_dag.metadata.target_node,
        created=datetime.now().strftime("%Y-%m-%d"),
        owner="narrative_pipeline",
        tags=["narrative", "auto-generated"],
    )
    return DAGSpec(
        metadata=metadata, nodes=sorted(nodes, key=lambda n: n.id),
        edges=edges, schema_version=base_dag.schema_version,
    )


def _print_comparison(base_dag: Any, new_dag: Any) -> None:
    """Print a Rich table comparing base DAG and NL-generated DAG."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    base_pairs = {(e.from_node, e.to_node): e for e in base_dag.edges}
    new_pairs = {(e.from_node, e.to_node): e for e in new_dag.edges}
    all_pairs = sorted(set(base_pairs.keys()) | set(new_pairs.keys()))

    table = Table(title="DAG Comparison: Manual vs NL-Generated", show_lines=True)
    table.add_column("Edge (from -> to)", style="cyan", min_width=30)
    table.add_column("Manual DAG", justify="center", min_width=16)
    table.add_column("NL-Generated", justify="center", min_width=16)
    table.add_column("Match?", justify="center", min_width=18)

    stats = {"structural": 0, "manual_only": 0, "nl_only": 0}
    for pair in all_pairs:
        label = f"{pair[0]} -> {pair[1]}"
        in_base, in_new = pair in base_pairs, pair in new_pairs
        if in_base:
            base_cell = f"[green]yes[/green] ({base_pairs[pair].get_edge_type()})"
        else:
            base_cell = "[dim]no[/dim]"
        if in_new:
            new_cell = f"[green]yes[/green] ({new_pairs[pair].get_edge_type()})"
        else:
            new_cell = "[dim]no[/dim]"
        if in_base and in_new:
            match_cell = "[bold green]structural match[/bold green]"
            stats["structural"] += 1
        elif in_base:
            match_cell = "[yellow]manual only[/yellow]"
            stats["manual_only"] += 1
        else:
            match_cell = "[blue]NL only[/blue]"
            stats["nl_only"] += 1
        table.add_row(label, base_cell, new_cell, match_cell)

    console.print()
    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold]  {stats['structural']} structural matches  |  "
        f"{stats['manual_only']} manual-only  |  {stats['nl_only']} NL-only\n"
    )


def generate(
    narrative_path: Path = DEFAULT_NARRATIVE,
    base_dag_path: Path = DEFAULT_BASE_DAG,
    output_path: Path = DEFAULT_OUT,
) -> Path:
    """Generate a DAG from narrative text using LLM extraction.

    Args:
        narrative_path: Path to narrative text file.
        base_dag_path: Path to base DAG YAML for node matching.
        output_path: Output path for generated DAG YAML.

    Returns:
        Path to the generated DAG YAML.
    """
    # Load narrative
    if not narrative_path.exists():
        raise FileNotFoundError(f"Narrative file not found: {narrative_path}")
    narrative_text = narrative_path.read_text(encoding="utf-8")
    logger.info(f"Loaded narrative ({len(narrative_text)} chars) from {narrative_path}")

    # Load base DAG
    from shared.agentic.dag.parser import parse_dag
    base_dag = parse_dag(base_dag_path)
    logger.info(f"Loaded base DAG '{base_dag.metadata.name}': "
                f"{len(base_dag.nodes)} nodes, {len(base_dag.edges)} edges")

    # Init LLM and extractor
    from shared.llm.client import get_llm_client
    from shared.agentic.agents.paper_dag_extractor import PaperDAGExtractor

    llm = get_llm_client()
    extractor = PaperDAGExtractor(llm, base_dag)
    paper = NarrativePaper(excerpt=narrative_text)

    # Stage 1: Extract causal claims
    logger.info("Stage 1: Extracting causal claims from narrative...")
    claims = extractor.extract_causal_claims([paper])
    logger.info(f"Extracted {len(claims)} causal claims")
    if not claims:
        logger.warning("No causal claims found in narrative.")
        return output_path

    # Stage 2: Match claims to existing DAG nodes
    logger.info("Stage 2: Matching claims to existing DAG nodes...")
    proposed_edges = extractor.match_to_dag(claims)
    logger.info(f"Proposed {len(proposed_edges)} edges "
                f"({sum(1 for e in proposed_edges if e.is_existing)} matched existing)")

    # Stage 3: Generate new node definitions for unmatched concepts
    all_new_names: list[str] = []
    for pe in proposed_edges:
        all_new_names.extend(pe.requires_new_nodes)
    existing_ids = {n.id for n in base_dag.nodes}
    unique_new_names = sorted(
        n for n in set(all_new_names) if _canonical_id(n) not in existing_ids
    )
    new_nodes: list[Any] = []
    if unique_new_names:
        logger.info(f"Stage 3: Generating {len(unique_new_names)} new node definitions...")
        new_nodes = _generate_new_nodes(llm, unique_new_names, existing_ids)
        logger.info(f"Generated {len(new_nodes)} new nodes")
    else:
        logger.info("Stage 3: No new nodes required.")

    # Build and write DAG
    logger.info("Building narrative DAG...")
    new_dag = _build_dag(base_dag, proposed_edges, new_nodes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(new_dag.to_dict(), f, sort_keys=False, allow_unicode=True,
                  default_flow_style=False)
    logger.info(f"Wrote narrative DAG to {output_path}")

    # Validate round-trip
    validated = parse_dag(output_path)
    logger.info(f"Validation passed: {len(validated.nodes)} nodes, "
                f"{len(validated.edges)} edges")

    # Print comparison table
    _print_comparison(base_dag, new_dag)

    # Post-generation repair pass
    logger.info("Running post-generation repair pass...")
    with open(output_path) as f:
        raw_dag = yaml.safe_load(f)
    repaired = repair_narrative_dag(raw_dag, base_dag_path)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(repaired, f, sort_keys=False, allow_unicode=True,
                  default_flow_style=False)
    logger.info("Repair pass complete.")

    return output_path


# =========================================================================
# Post-generation repair
# =========================================================================

def repair_narrative_dag(
    dag_dict: dict,
    base_dag_path: Path = DEFAULT_BASE_DAG,
) -> dict:
    """Post-generation repair pass that fixes known issues in narrative DAGs.

    Works on the raw YAML dict so it can be applied to any existing DAG file.

    Fixes applied in order:
      1. Double-logging formula correction (strip log() when node has transforms: [log])
      2. Missing identity edges (auto-create from derived node depends_on)
      3. Sink-node repair (borrow outgoing edges from base DAG)
      4. Edge ID sanitization (snake_case, no arrows/spaces/parens) — runs LAST so
         borrowed edges also get normalized, ensuring idempotency
      5. Scope-consistency assumption (auto-generate when cross-scope edges exist)

    Args:
        dag_dict: Raw narrative DAG loaded via yaml.safe_load.
        base_dag_path: Path to expert/base DAG for edge borrowing.

    Returns:
        Repaired DAG dict (deep copy; original is not mutated).
    """
    import copy

    dag = copy.deepcopy(dag_dict)

    # Load base DAG for edge borrowing
    base_dag: dict | None = None
    if base_dag_path.exists():
        with open(base_dag_path) as f:
            base_dag = yaml.safe_load(f)

    nodes: dict[str, dict] = {n["id"]: n for n in dag.get("nodes", [])}
    edges: list[dict] = dag.get("edges", [])

    # ── 1. Double-logging formula correction ─────────────────────────
    for node_id, node in nodes.items():
        identity = node.get("identity")
        if not identity:
            continue
        formula = identity.get("formula", "")
        deps = identity.get("depends_on") or node.get("depends_on", [])
        for dep in deps:
            dep_node = nodes.get(dep)
            if not dep_node:
                continue
            if "log" in dep_node.get("transforms", []):
                # diff(log(X)) → diff(X)
                formula = formula.replace(f"diff(log({dep}))", f"diff({dep})")
                # log(X) → X  (standalone)
                formula = formula.replace(f"log({dep})", dep)
        identity["formula"] = formula

    # ── 2. Missing identity edges ─────────────────────────────────────
    edge_pairs: set[tuple[str, str]] = {
        (e["from"], e["to"]) for e in edges
    }
    base_edge_map: dict[tuple[str, str], dict] = {}
    if base_dag:
        base_edge_map = {
            (e["from"], e["to"]): e for e in base_dag.get("edges", [])
        }

    for node_id, node in nodes.items():
        if not node.get("derived"):
            continue
        identity = node.get("identity")
        if not identity:
            continue
        deps = identity.get("depends_on") or node.get("depends_on", [])
        for dep in deps:
            if dep not in nodes:
                continue
            if (dep, node_id) in edge_pairs:
                continue
            # Prefer copying from base DAG (preserves rich metadata)
            import copy as _copy
            if (dep, node_id) in base_edge_map:
                edges.append(_copy.deepcopy(base_edge_map[(dep, node_id)]))
            else:
                edge_id = f"{dep}_to_{node_id}"
                edges.append({
                    "id": edge_id,
                    "from": dep,
                    "to": node_id,
                    "edge_type": "identity",
                    "edge_status": "IDENTITY",
                    "timing": {
                        "lag": 0, "lag_unit": "quarter",
                        "contemporaneous": True, "max_anticipation": 0,
                    },
                    "allowed_designs": [],
                    "interpretation": {
                        "is": f"Identity component: {identity.get('name', '')}",
                        "is_not": ["Causal relationship"],
                        "allowed_uses": ["scenario_only"],
                    },
                    "unit_specification": {
                        "treatment_unit": nodes.get(dep, {}).get("unit", ""),
                        "outcome_unit": node.get("unit", ""),
                    },
                    "notes": f"Auto-created identity edge from {identity.get('name', '')}",
                })
            edge_pairs.add((dep, node_id))

    # ── 3. Sink-node repair (borrow outgoing edges from base DAG) ─────
    if base_dag:
        target = dag.get("metadata", {}).get("target_node", "")
        edge_pairs_current = {(e["from"], e["to"]) for e in edges}
        outgoing: dict[str, int] = {n: 0 for n in nodes}
        incoming_set: dict[str, int] = {n: 0 for n in nodes}
        for e in edges:
            fn, tn = e.get("from", ""), e.get("to", "")
            if fn in outgoing:
                outgoing[fn] += 1
            if tn in incoming_set:
                incoming_set[tn] += 1

        for node_id in nodes:
            if node_id == target:
                continue
            if incoming_set[node_id] > 0 and outgoing[node_id] == 0:
                # Sink node — look for outgoing edges in base DAG
                for base_edge in base_dag.get("edges", []):
                    bf, bt = base_edge["from"], base_edge["to"]
                    if bf == node_id and bt in nodes and (bf, bt) not in edge_pairs_current:
                        import copy as _copy
                        edges.append(_copy.deepcopy(base_edge))
                        edge_pairs_current.add((bf, bt))
                        outgoing[node_id] += 1

    # ── 4. Edge ID sanitization + misleading ID regeneration ──────────
    # Runs AFTER edge additions so borrowed edges also get normalized.
    seen_ids: set[str] = set()
    for edge in edges:
        new_id = _sanitize_edge_id(edge["id"])
        from_node = edge.get("from", "")
        to_node = edge.get("to", "")
        from_lower = from_node.lower()
        to_lower = to_node.lower()
        # Regenerate if ID doesn't contain both from/to (case-insensitive)
        if from_lower and to_lower:
            if from_lower not in new_id or to_lower not in new_id:
                new_id = _sanitize_edge_id(f"{from_node}_to_{to_node}")
        # Collision guard
        if new_id in seen_ids:
            new_id = f"{new_id}_nl"
        seen_ids.add(new_id)
        edge["id"] = new_id

    # ── 5. Scope-consistency assumption ───────────────────────────────
    assumptions: list[dict] = dag.get("assumptions", []) or []
    assumption_ids = {a["id"] for a in assumptions}

    if "scope_consistency" not in assumption_ids:
        cross_scope_edges: list[str] = []
        for edge in edges:
            from_scope = nodes.get(edge["from"], {}).get("scope", "")
            to_scope = nodes.get(edge["to"], {}).get("scope", "")
            if from_scope and to_scope and from_scope != to_scope:
                cross_scope_edges.append(edge["id"])
        if cross_scope_edges:
            assumptions.append({
                "id": "scope_consistency",
                "description": (
                    "Transmission of global shocks to Kazakhstan-specific outcomes "
                    "preserves relative comparisons across regions within KZ."
                ),
                "applies_to_edges": cross_scope_edges,
                "testable": False,
            })

    dag["assumptions"] = assumptions
    dag["edges"] = edges
    return dag


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a DAG from a narrative text using LLM extraction.",
    )
    parser.add_argument("--narrative", type=Path, default=DEFAULT_NARRATIVE,
                        help="Path to narrative text file.")
    parser.add_argument("--base-dag", type=Path, default=DEFAULT_BASE_DAG,
                        help="Path to base DAG YAML for node matching.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output path for generated DAG YAML.")
    parser.add_argument("--repair", type=Path, default=None,
                        help="Repair an existing narrative DAG YAML (skip generation).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        if args.repair:
            logger.info(f"Repairing existing DAG: {args.repair}")
            with open(args.repair) as f:
                raw = yaml.safe_load(f)
            repaired = repair_narrative_dag(raw, args.base_dag)
            out = args.out or args.repair
            with open(out, "w", encoding="utf-8") as f:
                yaml.dump(repaired, f, sort_keys=False, allow_unicode=True,
                          default_flow_style=False)
            logger.info(f"Wrote repaired DAG to {out}")
        else:
            generate(args.narrative, args.base_dag, args.out)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
