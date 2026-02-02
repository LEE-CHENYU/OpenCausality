"""
Unified CLI for Kazakhstan econometric research.

Usage:
    kzresearch list-studies
    kzresearch welfare <command>
    kzresearch credit <command>
    kzresearch dag run <path>
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="kzresearch",
    help="Kazakhstan Econometric Research Platform",
    no_args_is_help=True,
)
dag_app = typer.Typer(
    name="dag",
    help="DAG-based agentic estimation commands",
)
app.add_typer(dag_app, name="dag")

console = Console()


# Import study-specific apps
try:
    from studies.household_welfare.src.cli import app as welfare_app
    app.add_typer(welfare_app, name="welfare", help="Household welfare study commands")
except ImportError:
    pass

try:
    from studies.credit_default.src.cli import app as credit_app
    app.add_typer(credit_app, name="credit", help="Credit default study commands")
except ImportError:
    pass

try:
    from studies.fx_passthrough.src.cli import app as passthrough_app
    app.add_typer(passthrough_app, name="passthrough", help="FX passthrough study commands")
except ImportError:
    pass


@app.command("list-studies")
def list_studies():
    """List all available research studies."""
    table = Table(title="Kazakhstan Econometric Research Studies")

    table.add_column("Study", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Status", style="green")
    table.add_column("CLI", style="yellow")

    table.add_row(
        "household_welfare",
        "Oil price shocks → household income",
        "Active",
        "kzresearch welfare",
    )
    table.add_row(
        "credit_default",
        "Income changes → credit default risk",
        "Active",
        "kzresearch credit",
    )
    table.add_row(
        "fx_passthrough",
        "FX → inflation → income → expenditure",
        "Active",
        "kzresearch passthrough",
    )

    console.print(table)


@app.command("info")
def info():
    """Show information about the research platform."""
    console.print("\n[bold cyan]Kazakhstan Econometric Research Platform[/bold cyan]\n")
    console.print("Version: 0.3.0")
    console.print("\n[bold]Studies:[/bold]")
    console.print("  1. [cyan]household_welfare[/cyan] - Oil shocks → household income")
    console.print("  2. [cyan]credit_default[/cyan] - Income → credit default")
    console.print("  3. [cyan]fx_passthrough[/cyan] - FX → inflation → expenditure\n")
    console.print("[bold]Shared Infrastructure:[/bold]")
    console.print("  - shared/data/ - Data clients (BNS, FRED, exchange rate, CPI)")
    console.print("  - shared/model/ - Inference, event study, small-N methods")
    console.print("  - shared/engine/ - Scenario simulation base")
    console.print("  - shared/agentic/ - DAG-based agentic estimation\n")
    console.print("[bold]Usage:[/bold]")
    console.print("  kzresearch list-studies       List all studies")
    console.print("  kzresearch welfare --help     Household welfare commands")
    console.print("  kzresearch credit --help      Credit default commands")
    console.print("  kzresearch passthrough --help FX passthrough commands")
    console.print("  kzresearch dag run <path>     Run a DAG specification\n")


# ============================================================================
# DAG Commands
# ============================================================================

@dag_app.command("run")
def dag_run(
    dag_path: Path = typer.Argument(
        ...,
        help="Path to DAG YAML specification file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for results",
    ),
    mode: str = typer.Option(
        "EXPLORATION",
        "--mode", "-m",
        help="Run mode: EXPLORATION or CONFIRMATION",
    ),
    max_iterations: int = typer.Option(
        3,
        "--max-iterations",
        help="Maximum iterations in exploration mode",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose output",
    ),
):
    """
    Run a DAG specification through the agentic estimation loop.

    Example:
        kzresearch dag run config/agentic/dags/example_kz_welfare.yaml

    The command will:
    1. Parse and validate the DAG
    2. Catalog data availability
    3. Select designs for each edge
    4. Run estimations and produce EdgeCards
    5. Evaluate credibility and iterate (in EXPLORATION mode)
    """
    import logging

    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    console.print(f"\n[bold cyan]Running DAG: {dag_path}[/bold cyan]\n")

    try:
        from shared.agentic.dag.parser import parse_dag
        from shared.agentic.dag.validator import validate_dag
        from shared.agentic.agent_loop import AgentLoop, AgentLoopConfig

        # Parse DAG
        console.print("[dim]Parsing DAG...[/dim]")
        dag = parse_dag(dag_path)
        console.print(f"  DAG: [cyan]{dag.metadata.name}[/cyan]")
        console.print(f"  Nodes: {len(dag.nodes)}")
        console.print(f"  Edges: {len(dag.edges)}")
        console.print(f"  Target: {dag.metadata.target_node}")

        # Validate DAG
        console.print("\n[dim]Validating DAG...[/dim]")
        validation = validate_dag(dag, raise_on_error=False)

        if not validation.is_valid:
            console.print("[bold red]DAG validation failed![/bold red]")
            for issue in validation.errors():
                console.print(f"  [red]ERROR[/red] {issue.location}: {issue.message}")
            raise typer.Exit(1)

        if validation.warnings():
            for issue in validation.warnings():
                console.print(f"  [yellow]WARN[/yellow] {issue.location}: {issue.message}")

        console.print("  [green]Validation passed[/green]")

        # Configure loop
        config = AgentLoopConfig(
            mode=mode,
            max_iterations=max_iterations,
            output_dir=output_dir or Path("outputs/agentic"),
        )

        # Run loop
        console.print(f"\n[dim]Running agent loop (mode={mode})...[/dim]\n")
        loop = AgentLoop(dag, config)
        report = loop.run()

        # Display results
        console.print("\n" + "=" * 60)
        console.print("[bold green]RUN COMPLETE[/bold green]")
        console.print("=" * 60)

        # Summary table
        table = Table(title="Edge Results")
        table.add_column("Edge", style="cyan")
        table.add_column("Design", style="white")
        table.add_column("Estimate", style="white")
        table.add_column("Rating", style="yellow")
        table.add_column("Score", style="white")

        for summary in report.edge_summaries:
            est = f"{summary.estimate:.4f}" if summary.estimate else "-"
            table.add_row(
                summary.edge_id,
                summary.design,
                est,
                summary.credibility_rating,
                f"{summary.credibility_score:.2f}",
            )

        console.print(table)

        # Blocked edges
        if report.blocked_edges:
            console.print("\n[bold red]Blocked Edges:[/bold red]")
            for edge_id, reason in report.blocked_edges.items():
                console.print(f"  {edge_id}: {reason}")

        # Output location
        console.print(f"\nResults saved to: [cyan]{config.output_dir}[/cyan]")

    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@dag_app.command("validate")
def dag_validate(
    dag_path: Path = typer.Argument(
        ...,
        help="Path to DAG YAML specification file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
):
    """
    Validate a DAG specification without running estimation.

    Checks:
    - Referential integrity (edge endpoints exist)
    - Temporal cycles
    - Identity dependencies
    - Forbidden controls consistency
    """
    console.print(f"\n[bold cyan]Validating DAG: {dag_path}[/bold cyan]\n")

    try:
        from shared.agentic.dag.parser import parse_dag
        from shared.agentic.dag.validator import validate_dag

        dag = parse_dag(dag_path)
        console.print(f"  DAG: [cyan]{dag.metadata.name}[/cyan]")

        validation = validate_dag(dag, raise_on_error=False)

        if validation.errors():
            console.print("\n[bold red]Errors:[/bold red]")
            for issue in validation.errors():
                console.print(f"  {issue.location}: {issue.message}")

        if validation.warnings():
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for issue in validation.warnings():
                console.print(f"  {issue.location}: {issue.message}")

        if validation.forbidden_controls:
            console.print("\n[dim]Forbidden Controls (per edge):[/dim]")
            for edge_id, result in validation.forbidden_controls.items():
                if result.total_forbidden:
                    console.print(f"  {edge_id}: {sorted(result.total_forbidden)}")

        if validation.is_valid:
            console.print("\n[bold green]DAG is valid[/bold green]")
        else:
            console.print("\n[bold red]DAG validation failed[/bold red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@dag_app.command("list")
def dag_list(
    dags_dir: Path = typer.Option(
        Path("config/agentic/dags"),
        "--dir", "-d",
        help="Directory containing DAG YAML files",
    ),
):
    """List available DAG specifications."""
    console.print(f"\n[bold cyan]DAG Specifications in {dags_dir}[/bold cyan]\n")

    if not dags_dir.exists():
        console.print(f"[yellow]Directory does not exist: {dags_dir}[/yellow]")
        return

    dag_files = list(dags_dir.glob("*.yaml")) + list(dags_dir.glob("*.yml"))

    if not dag_files:
        console.print("[dim]No DAG files found[/dim]")
        return

    try:
        from shared.agentic.dag.parser import parse_dag

        table = Table(title="Available DAGs")
        table.add_column("File", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Nodes", style="white")
        table.add_column("Edges", style="white")
        table.add_column("Target", style="yellow")

        for dag_file in sorted(dag_files):
            try:
                dag = parse_dag(dag_file)
                table.add_row(
                    dag_file.name,
                    dag.metadata.name,
                    str(len(dag.nodes)),
                    str(len(dag.edges)),
                    dag.metadata.target_node or "-",
                )
            except Exception as e:
                table.add_row(
                    dag_file.name,
                    f"[red]Error: {e}[/red]",
                    "-", "-", "-",
                )

        console.print(table)

    except ImportError:
        console.print("[yellow]Could not import DAG parser[/yellow]")
        for dag_file in sorted(dag_files):
            console.print(f"  {dag_file.name}")


if __name__ == "__main__":
    app()
