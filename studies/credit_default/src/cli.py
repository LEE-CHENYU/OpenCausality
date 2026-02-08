"""
CLI for credit default study.

Usage:
    opencausality credit check-confounds
    opencausality credit build-panel
    opencausality credit estimate mw-diff-discs
    opencausality credit estimate pension-rdd
    opencausality credit diagnostics
    opencausality credit simulate
    opencausality credit stress-test
    opencausality credit run-mvp
"""

from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(
    name="credit",
    help="Credit Default Sensitivity Study",
    no_args_is_help=True,
)
console = Console()


@app.command("check-confounds")
def check_confounds(
    origination_cutoff: str = typer.Option(
        "2023-12-01",
        help="Latest origination date (YYYY-MM-DD)",
    ),
    outcome_start: str = typer.Option(
        "2024-01-01",
        help="Outcome window start (YYYY-MM-DD)",
    ),
    outcome_end: str = typer.Option(
        "2024-05-31",
        help="Outcome window end (YYYY-MM-DD)",
    ),
    treatment_date: str = typer.Option(
        "2024-01-01",
        help="Treatment date (YYYY-MM-DD)",
    ),
):
    """Check for policy confounds in the analysis window."""
    from studies.credit_default.src.confound_checks import ConfoundChecker

    console.print("\n[bold cyan]Policy Confound Check[/bold cyan]\n")

    # Parse dates
    orig_cutoff = date.fromisoformat(origination_cutoff)
    out_start = date.fromisoformat(outcome_start)
    out_end = date.fromisoformat(outcome_end)
    treat_date = date.fromisoformat(treatment_date)

    checker = ConfoundChecker()
    results = checker.run_all_checks(orig_cutoff, out_start, out_end, treat_date)

    # Display results
    all_passed = all(r.passed for r in results.values())

    for name, result in results.items():
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(f"{name}: {status}")
        console.print(f"  {result.description}")

        if result.confounds_found:
            for c in result.confounds_found:
                console.print(f"  [red]• {c}[/red]")

        if result.recommendations:
            for r in result.recommendations:
                console.print(f"  [yellow]→ {r}[/yellow]")

    console.print()
    if all_passed:
        console.print("[bold green]All confound checks passed.[/bold green]\n")
    else:
        console.print("[bold red]Some confound checks failed. Review recommendations.[/bold red]\n")


@app.command("build-panel")
def build_panel(
    origination_before: str = typer.Option(
        "2023-12-01",
        help="Restrict to loans originated before this date",
    ),
    outcome_window: str = typer.Option(
        "2024-01-01:2024-05-31",
        help="Outcome window (start:end)",
    ),
    output_path: Path = typer.Option(
        Path("studies/credit_default/outputs/loan_panel.parquet"),
        help="Output path for panel",
    ),
):
    """Build loan-month panel for estimation."""
    console.print("\n[bold cyan]Building Loan Panel[/bold cyan]\n")

    console.print(f"Origination cutoff: {origination_before}")
    console.print(f"Outcome window: {outcome_window}")

    # Parse outcome window
    start, end = outcome_window.split(":")

    console.print("\n[yellow]Note: This requires internal loan data.[/yellow]")
    console.print("Panel builder would load from configured data source.\n")

    # In production, would call:
    # from studies.credit_default.src.panel_data import LoanPanelBuilder
    # builder = LoanPanelBuilder()
    # panel = builder.build_panel(...)

    console.print("[dim]Panel building not implemented without data source.[/dim]\n")


# Subcommand for estimation
estimate_app = typer.Typer(help="Estimation commands")
app.add_typer(estimate_app, name="estimate")


@estimate_app.command("mw-diff-discs")
def estimate_mw(
    outcome: str = typer.Option("dpd30", help="Outcome variable"),
    horizon: int = typer.Option(3, help="Outcome horizon in months"),
    bandwidth: float = typer.Option(None, help="Bandwidth around cutoff"),
    polynomial: int = typer.Option(1, help="Polynomial order"),
):
    """Estimate minimum wage diff-in-discontinuities."""
    console.print("\n[bold cyan]MW Difference-in-Discontinuities Estimation[/bold cyan]\n")

    console.print(f"Outcome: {outcome}")
    console.print(f"Horizon: {horizon} months")
    console.print(f"Bandwidth: {'auto' if bandwidth is None else bandwidth}")
    console.print(f"Polynomial order: {polynomial}")

    console.print("\n[yellow]Note: Requires loan panel data.[/yellow]")

    # Show specification
    console.print("\n[dim]Specification:[/dim]")
    console.print("[dim]Y_it = α + β₁(Post) + β₂(Below_c) + β₃(Post × Below_c)[/dim]")
    console.print("[dim]     + f(wage - c) + g(wage - c) × Post + X'δ + ε[/dim]")
    console.print("[dim]Key coefficient: β₃[/dim]\n")


@estimate_app.command("pension-rdd")
def estimate_pension(
    outcome: str = typer.Option("dpd30", help="Outcome variable"),
    bandwidth: str = typer.Option("optimal", help="Bandwidth (optimal or numeric)"),
    kernel: str = typer.Option("triangular", help="Kernel type"),
):
    """Estimate pension eligibility fuzzy RDD."""
    console.print("\n[bold cyan]Pension Fuzzy RDD Estimation[/bold cyan]\n")

    console.print(f"Outcome: {outcome}")
    console.print(f"Bandwidth: {bandwidth}")
    console.print(f"Kernel: {kernel}")
    console.print(f"Cutoffs: Men=63, Women=61")

    console.print("\n[yellow]Note: Requires loan panel with age and pension data.[/yellow]")

    # Show specification
    console.print("\n[dim]First stage:  PensionInflow = γ₀ + γ₁(Age ≥ c) + f(Age - c) + X'δ + ν[/dim]")
    console.print("[dim]Second stage: Default = α + β(PensionInflow_hat) + f(Age - c) + X'δ + ε[/dim]")
    console.print("[dim]Key coefficient: β (LATE)[/dim]\n")


@app.command("diagnostics")
def diagnostics():
    """Run diagnostic tests for identification designs."""
    console.print("\n[bold cyan]Diagnostic Tests[/bold cyan]\n")

    table = Table(title="Required Diagnostics")

    table.add_column("Test", style="cyan")
    table.add_column("Purpose", style="white")
    table.add_column("Criterion", style="yellow")

    table.add_row("Pre-trends", "Parallel trends assumption", "Event study + joint F")
    table.add_row("First-stage F", "Instrument strength", "F > 10")
    table.add_row("Balance test", "Covariate randomization", "No jumps at cutoff")
    table.add_row("McCrary", "No manipulation", "No density discontinuity")
    table.add_row("Placebo cutoffs", "Design validity", "No effect at false cutoffs")

    console.print(table)
    console.print("\n[yellow]Note: Requires estimated models to run diagnostics.[/yellow]\n")


@app.command("simulate")
def simulate(
    income_change: float = typer.Option(
        -0.10,
        help="Income change as decimal (e.g., -0.10 for -10%)",
    ),
    segment: str = typer.Option(
        "formal_workers",
        help="Borrower segment",
    ),
    baseline_rate: float = typer.Option(
        0.05,
        help="Baseline default rate",
    ),
):
    """Simulate income shock effect on default rates."""
    console.print("\n[bold cyan]Income Shock Simulation[/bold cyan]\n")

    console.print(f"Income change: {income_change*100:+.1f}%")
    console.print(f"Segment: {segment}")
    console.print(f"Baseline rate: {baseline_rate*100:.1f}%")

    console.print("\n[yellow]Note: Requires stored elasticities from estimation.[/yellow]")
    console.print("[dim]Run estimation first to populate elasticity store.[/dim]\n")

    # Would call:
    # from studies.credit_default.src.scenario_simulator import CreditScenarioSimulator
    # simulator = CreditScenarioSimulator(baseline_default_rate=baseline_rate)
    # result = simulator.simulate_income_shock(income_change)


@app.command("stress-test")
def stress_test(
    scenario: str = typer.Option(
        "moderate_recession",
        help="Stress scenario (mild_recession, moderate_recession, severe_recession, oil_shock)",
    ),
    lgd: float = typer.Option(
        0.50,
        help="Loss given default",
    ),
):
    """Run portfolio stress test."""
    console.print("\n[bold cyan]Portfolio Stress Test[/bold cyan]\n")

    console.print(f"Scenario: {scenario}")
    console.print(f"LGD: {lgd*100:.0f}%")

    # Show available scenarios
    table = Table(title="Available Stress Scenarios")

    table.add_column("Scenario", style="cyan")
    table.add_column("Severity", style="yellow")
    table.add_column("Description", style="white")

    table.add_row("mild_recession", "Mild", "5% income decline")
    table.add_row("moderate_recession", "Moderate", "10% income decline")
    table.add_row("severe_recession", "Severe", "20% income decline")
    table.add_row("oil_shock", "Severe", "Oil price collapse")

    console.print(table)
    console.print("\n[yellow]Note: Requires elasticities and portfolio data.[/yellow]\n")


@app.command("run-mvp")
def run_mvp():
    """Run full MVP analysis pipeline."""
    console.print("\n[bold cyan]Credit Default Study MVP Pipeline[/bold cyan]\n")

    steps = [
        ("1. Check policy confounds", "Verify analysis window"),
        ("2. Build loan panel", "Construct loan-month panel"),
        ("3. Estimate MW diff-in-discs", "Primary identification"),
        ("4. Estimate pension RDD", "Secondary identification"),
        ("5. Run diagnostics", "Validate designs"),
        ("6. Store elasticities", "Save for simulation"),
        ("7. Run simulations", "Scenario analysis"),
    ]

    table = Table(title="MVP Pipeline Steps")
    table.add_column("Step", style="cyan")
    table.add_column("Description", style="white")

    for step, desc in steps:
        table.add_row(step, desc)

    console.print(table)

    console.print("\n[bold]Primary Outcome:[/bold] DPD30 at 3-month horizon")
    console.print("[bold]Sample:[/bold] Loans originated before Dec 2023, outcomes Jan-May 2024")

    console.print("\n[yellow]Run individual commands to execute each step.[/yellow]\n")


@app.command("event-study")
def event_study(
    pre_periods: int = typer.Option(6, help="Number of pre-treatment periods"),
    post_periods: int = typer.Option(5, help="Number of post-treatment periods"),
    outcome: str = typer.Option("dpd30", help="Outcome variable"),
):
    """Generate event study for pre-trends visualization."""
    console.print("\n[bold cyan]Event Study Analysis[/bold cyan]\n")

    console.print(f"Pre-treatment periods: {pre_periods}")
    console.print(f"Post-treatment periods: {post_periods}")
    console.print(f"Outcome: {outcome}")

    console.print("\n[yellow]Note: Requires panel data with treatment timing.[/yellow]")
    console.print("[dim]Event study helps visualize pre-trends for parallel trends assessment.[/dim]\n")


if __name__ == "__main__":
    app()
