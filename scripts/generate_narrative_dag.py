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
    from shared.agentic.dag.parser import DAGMetadata, DAGSpec, EdgeSpec

    referenced_ids: set[str] = set()
    for pe in proposed_edges:
        referenced_ids.add(pe.from_node)
        referenced_ids.add(pe.to_node)

    existing_node_map = {n.id: n for n in base_dag.nodes}
    nodes = [existing_node_map[nid] for nid in referenced_ids if nid in existing_node_map]
    new_node_map = {n.id: n for n in new_nodes}
    for nid in referenced_ids:
        if nid not in existing_node_map and nid in new_node_map:
            nodes.append(new_node_map[nid])

    base_edge_map: dict[tuple[str, str], Any] = {
        (e.from_node, e.to_node): e for e in base_dag.edges
    }
    edges: list[EdgeSpec] = []
    seen_ids: set[str] = set()
    for pe in proposed_edges:
        key = (pe.from_node, pe.to_node)
        if pe.is_existing and key in base_edge_map:
            edges.append(base_edge_map[key])
            seen_ids.add(base_edge_map[key].id)
        else:
            edge_id = pe.edge_id or f"{pe.from_node}_to_{pe.to_node}"
            if edge_id in seen_ids:
                edge_id = f"{edge_id}_nl"
            seen_ids.add(edge_id)
            evidence_notes = "; ".join(
                f"{c.treatment}->{c.outcome} ({c.direction})" for c in pe.evidence
            )
            edges.append(EdgeSpec(
                id=edge_id, from_node=pe.from_node, to_node=pe.to_node,
                edge_type=pe.edge_type, notes=f"NL-extracted: {evidence_notes}",
            ))

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
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load narrative
    if not args.narrative.exists():
        logger.error(f"Narrative file not found: {args.narrative}")
        sys.exit(1)
    narrative_text = args.narrative.read_text(encoding="utf-8")
    logger.info(f"Loaded narrative ({len(narrative_text)} chars) from {args.narrative}")

    # Load base DAG
    from shared.agentic.dag.parser import parse_dag
    base_dag = parse_dag(args.base_dag)
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
        logger.warning("No causal claims found in narrative. Exiting.")
        sys.exit(0)

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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.dump(new_dag.to_dict(), f, sort_keys=False, allow_unicode=True,
                  default_flow_style=False)
    logger.info(f"Wrote narrative DAG to {args.out}")

    # Validate round-trip
    try:
        validated = parse_dag(args.out)
        logger.info(f"Validation passed: {len(validated.nodes)} nodes, "
                    f"{len(validated.edges)} edges")
    except Exception as e:
        logger.error(f"Validation failed on written DAG: {e}")
        sys.exit(1)

    # Print comparison table
    _print_comparison(base_dag, new_dag)


if __name__ == "__main__":
    main()
