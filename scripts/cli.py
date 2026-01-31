"""
Unified CLI for Kazakhstan econometric research.

Usage:
    kzresearch list-studies
    kzresearch welfare <command>
    kzresearch credit <command>
"""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="kzresearch",
    help="Kazakhstan Econometric Research Platform",
    no_args_is_help=True,
)
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

    console.print(table)


@app.command("info")
def info():
    """Show information about the research platform."""
    console.print("\n[bold cyan]Kazakhstan Econometric Research Platform[/bold cyan]\n")
    console.print("Version: 0.2.0")
    console.print("\n[bold]Studies:[/bold]")
    console.print("  1. [cyan]household_welfare[/cyan] - Oil shocks → household income")
    console.print("  2. [cyan]credit_default[/cyan] - Income → credit default\n")
    console.print("[bold]Shared Infrastructure:[/bold]")
    console.print("  - shared/data/ - Data clients (BNS, FRED, policy calendar)")
    console.print("  - shared/model/ - Inference, event study, diagnostics")
    console.print("  - shared/engine/ - Scenario simulation base\n")
    console.print("[bold]Usage:[/bold]")
    console.print("  kzresearch list-studies    List all studies")
    console.print("  kzresearch welfare --help  Household welfare commands")
    console.print("  kzresearch credit --help   Credit default commands\n")


if __name__ == "__main__":
    app()
