#!/usr/bin/env python3
"""
Run DataScout to download missing data for orphan DAG nodes.

Identifies orphan nodes (no incoming edges) with fetchable connectors
(fred, bns), invokes DataScout to download them, and updates YAML
provenance for successfully-fetched nodes.

Run with: PYTHONPATH=. python scripts/run_data_scout.py
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_path=False)],
)
logger = logging.getLogger(__name__)

DAG_PATH = Path("config/agentic/dags/kspi_k2_full.yaml")
FETCHABLE_CONNECTORS = {"fred", "bns"}


def find_orphan_nodes(dag):
    """Find nodes with no incoming edges that use fetchable connectors."""
    # Collect all edge targets
    edge_targets = {e.to_node for e in dag.edges}

    orphans = []
    for node in dag.nodes:
        if node.id in edge_targets:
            continue
        if not node.source or not node.source.preferred:
            continue
        connector = node.source.preferred[0].connector
        if connector in FETCHABLE_CONNECTORS:
            orphans.append(node.id)

    return orphans


def update_yaml_provenance(dag_path: Path, succeeded_ids: set[str]) -> int:
    """Update provenance in the YAML for successfully-fetched nodes.

    Returns:
        Number of nodes updated.
    """
    if not succeeded_ids:
        return 0

    with open(dag_path) as f:
        raw = yaml.safe_load(f)

    updated = 0
    for node_dict in raw.get("nodes", []):
        if node_dict.get("id") in succeeded_ids:
            prov = node_dict.get("provenance", {})
            connector = ""
            dataset = ""
            src_list = node_dict.get("source", {}).get("preferred", [])
            if src_list:
                connector = src_list[0].get("connector", "")
                dataset = src_list[0].get("dataset", "")

            prov["source"] = "data_scout"
            prov["added_at"] = date.today().isoformat()
            prov["connector"] = connector
            prov["dataset"] = dataset
            node_dict["provenance"] = prov
            updated += 1

    if updated:
        with open(dag_path, "w") as f:
            yaml.dump(raw, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Run DataScout for orphan DAG nodes with fetchable connectors"
    )
    parser.add_argument(
        "--dag", type=Path, default=DAG_PATH, help="Path to DAG YAML"
    )
    parser.add_argument(
        "--budget-mb", type=int, default=100, help="Download budget in MB"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List orphan nodes without downloading"
    )
    parser.add_argument(
        "--no-provenance-update", action="store_true",
        help="Skip YAML provenance update after download"
    )
    args = parser.parse_args()

    from shared.agentic.dag.parser import parse_dag
    from shared.agentic.agents.data_scout import DataScout

    console.print("[bold green]DataScout: Orphan Node Downloader[/bold green]")
    console.print("=" * 50)

    # Load DAG
    dag = parse_dag(args.dag)
    console.print(f"Loaded DAG: {dag.metadata.name} ({len(dag.nodes)} nodes, {len(dag.edges)} edges)")

    # Find orphan nodes with fetchable connectors
    orphan_ids = find_orphan_nodes(dag)

    if not orphan_ids:
        console.print("[yellow]No orphan nodes with fetchable connectors found.[/yellow]")
        return

    # Display orphan nodes
    node_map = {n.id: n for n in dag.nodes}
    table = Table(title="Orphan Nodes with Fetchable Connectors")
    table.add_column("Node ID", style="cyan")
    table.add_column("Name")
    table.add_column("Connector", style="green")
    table.add_column("Dataset")
    table.add_column("Series")

    for nid in orphan_ids:
        node = node_map[nid]
        src = node.source.preferred[0]
        table.add_row(nid, node.name, src.connector, src.dataset, src.series)

    console.print(table)
    console.print(f"\nFound {len(orphan_ids)} orphan nodes to fetch.\n")

    if args.dry_run:
        console.print("[yellow]Dry run — skipping downloads.[/yellow]")
        return

    # Run DataScout
    scout = DataScout(budget_mb=args.budget_mb)
    report = scout.download_missing(dag, orphan_ids)

    # Display results
    results_table = Table(title="Download Results")
    results_table.add_column("Node ID", style="cyan")
    results_table.add_column("Connector")
    results_table.add_column("Status")
    results_table.add_column("Size", justify="right")
    results_table.add_column("Rows", justify="right")
    results_table.add_column("Error", style="red")

    for r in report.results:
        if r.success:
            status = "[green]OK[/green]"
        elif r.error == "static":
            status = "[dim]skipped[/dim]"
        else:
            status = "[red]FAIL[/red]"

        size_str = f"{r.size_bytes / 1024:.1f} KB" if r.size_bytes else "-"
        rows_str = str(r.row_count) if r.row_count else "-"
        err_str = (r.error[:50] + "...") if len(r.error) > 50 else r.error

        results_table.add_row(r.node_id, r.connector, status, size_str, rows_str, err_str)

    console.print(results_table)

    console.print(
        f"\n[bold]Summary:[/bold] {report.downloaded} downloaded, "
        f"{report.skipped} skipped, {report.failed} failed "
        f"({report.total_bytes / 1024:.1f} KB used of "
        f"{report.budget_bytes / 1024 / 1024:.0f} MB budget)"
    )

    # Generate guidance for failures
    guidance = scout.generate_user_guidance(report, dag)
    if guidance:
        console.print("\n[bold yellow]Manual action needed:[/bold yellow]")
        for msg in guidance:
            console.print(f"  {msg}")

    # Update YAML provenance for successes
    if not args.no_provenance_update:
        succeeded = {r.node_id for r in report.results if r.success}
        if succeeded:
            n_updated = update_yaml_provenance(args.dag, succeeded)
            console.print(
                f"\n[green]Updated provenance for {n_updated} nodes "
                f"in {args.dag}[/green]"
            )

    # Exit code: 0 if all succeeded, 1 if any failed
    if report.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
