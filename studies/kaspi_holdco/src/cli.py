"""
CLI commands for Kaspi.kz Holding Company Capital Adequacy Study.

Entry point: opencausality kaspi (or standalone kzkaspi)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from studies.kaspi_holdco.src.bank_state import BankState, BankScenario
from studies.kaspi_holdco.src.bank_stress import (
    simulate_bank_stress,
    run_bank_stress_grid,
    BankStressResult,
)
from studies.kaspi_holdco.src.holdco_state import HoldCoState, HoldCoScenario
from studies.kaspi_holdco.src.holdco_simulator import (
    simulate_holdco_12m,
    scenario_grid,
    HoldCoStressTester,
)
from studies.kaspi_holdco.src.bank_integration import (
    PassthroughConfig,
    build_holdco_scenario_from_bank_stress,
)
from studies.kaspi_holdco.src.stress_scenarios import (
    FY2024_BANK_STATE,
    FY2024_HOLDCO_STATE,
    FY2024_BASELINE_SCENARIO,
    FY2024_ANNUAL_NET_INCOME,
    BANK_STRESS_SCENARIOS,
    HOLDCO_SCENARIOS,
    list_bank_scenarios,
    list_holdco_scenarios,
)

app = typer.Typer(
    name="kzkaspi",
    help="Kaspi.kz Holding Company Capital Adequacy Analysis",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with rich output."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


# =============================================================================
# STATE INSPECTION
# =============================================================================


@app.command()
def state() -> None:
    """Show bank and holdco baseline state (FY2024)."""
    setup_logging()

    console.print(Panel("[bold]Kaspi.kz FY2024 State[/bold]", style="blue"))

    # Bank state
    console.print("\n[bold cyan]Bank State[/bold cyan]")
    console.print(FY2024_BANK_STATE.summary())

    # Holdco state
    console.print("\n[bold cyan]HoldCo State[/bold cyan]")
    console.print(FY2024_HOLDCO_STATE.summary())

    # Key metrics table
    table = Table(title="Key Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Threshold", style="yellow")

    table.add_row("Bank K1-2", f"{FY2024_BANK_STATE.k1_2_ratio:.1%}", "min 10.5%")
    table.add_row("Bank K2", f"{FY2024_BANK_STATE.k2_ratio:.1%}", "min 12.0%")
    table.add_row("K2 Headroom", f"{FY2024_BANK_STATE.k2_headroom:.1f} bn", "-")
    table.add_row(
        "HoldCo Runway",
        f"{FY2024_HOLDCO_STATE.months_of_runway_standalone:.0f} months",
        "-",
    )
    table.add_row("HoldCo Cash", f"{FY2024_HOLDCO_STATE.cash:.1f} bn", "-")

    console.print(table)


# =============================================================================
# BANK-LEVEL STRESS TESTING
# =============================================================================


@app.command("bank-stress")
def bank_stress(
    scenario: str = typer.Option(
        "moderate",
        "--scenario",
        "-s",
        help=f"Scenario name: {', '.join(list_bank_scenarios())}",
    ),
) -> None:
    """Run a single bank stress scenario."""
    setup_logging()

    if scenario not in BANK_STRESS_SCENARIOS:
        console.print(f"[red]Unknown scenario: {scenario}[/red]")
        console.print(f"Available: {', '.join(list_bank_scenarios())}")
        raise typer.Exit(1)

    sc = BANK_STRESS_SCENARIOS[scenario]
    result = simulate_bank_stress(FY2024_BANK_STATE, sc)

    console.print(Panel(f"[bold]Bank Stress: {scenario}[/bold]", style="blue"))
    console.print(result.summary())

    # Interpretation
    if result.support_needed:
        console.print("\n[bold red]SUPPORT NEEDED[/bold red]")
        if result.capital_shortfall > 0:
            console.print(f"  Capital shortfall: {result.capital_shortfall:.1f} bn")
        if result.liquidity_gap > 0:
            console.print(f"  Liquidity gap: {result.liquidity_gap:.1f} bn")
    else:
        console.print("\n[bold green]Bank remains viable[/bold green]")


@app.command("bank-stress-grid")
def bank_stress_grid(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path",
    ),
) -> None:
    """Run bank stress across parameter grid."""
    setup_logging()

    console.print(Panel("[bold]Bank Stress Grid[/bold]", style="blue"))

    df = run_bank_stress_grid(FY2024_BANK_STATE, FY2024_ANNUAL_NET_INCOME)

    # Summary stats
    total = len(df)
    support_needed = df["support_needed"].sum()
    console.print(f"\nTotal scenarios: {total}")
    console.print(f"Support needed: {support_needed} ({support_needed/total:.0%})")

    # Key thresholds
    viable = df[~df["support_needed"]]
    if not viable.empty:
        max_credit_loss = viable["credit_loss_rate"].max()
        max_rwa_mult = viable["rwa_multiplier"].max()
        console.print(f"\nMax viable credit loss: {max_credit_loss:.1%}")
        console.print(f"Max viable RWA multiplier: {max_rwa_mult:.0%}")

    # Show sample of critical scenarios
    critical = df[df["support_needed"]].nsmallest(5, "capital_shortfall")
    if not critical.empty:
        console.print("\n[bold]Critical scenarios (smallest shortfall):[/bold]")
        table = Table()
        table.add_column("Scenario")
        table.add_column("K2 After", justify="right")
        table.add_column("Shortfall", justify="right")

        for _, row in critical.iterrows():
            table.add_row(
                row["scenario"],
                f"{row['k2_after']:.1%}",
                f"{row['capital_shortfall']:.1f}",
            )
        console.print(table)

    if output:
        df.to_csv(output, index=False)
        console.print(f"\n[green]Saved to {output}[/green]")


# =============================================================================
# HOLDCO STRESS TESTING
# =============================================================================


@app.command("holdco-simulate")
def holdco_simulate(
    scenario: str = typer.Option(
        "baseline",
        "--scenario",
        "-s",
        help=f"Scenario name: {', '.join(list_holdco_scenarios())}",
    ),
    monthly: bool = typer.Option(False, "--monthly", "-m", help="Show monthly detail"),
) -> None:
    """Run holdco cash flow simulation."""
    setup_logging()

    if scenario not in HOLDCO_SCENARIOS:
        console.print(f"[red]Unknown scenario: {scenario}[/red]")
        console.print(f"Available: {', '.join(list_holdco_scenarios())}")
        raise typer.Exit(1)

    sc = HOLDCO_SCENARIOS[scenario]

    if monthly:
        result, monthly_df = simulate_holdco_12m(FY2024_HOLDCO_STATE, sc, monthly_detail=True)
    else:
        result = simulate_holdco_12m(FY2024_HOLDCO_STATE, sc)

    console.print(Panel(f"[bold]HoldCo Simulation: {scenario}[/bold]", style="blue"))
    console.print(result.summary())

    if monthly:
        console.print("\n[bold]Monthly Cash Flow:[/bold]")
        table = Table()
        table.add_column("Month", justify="right")
        table.add_column("Balance", justify="right")
        table.add_column("Inflows", justify="right")
        table.add_column("Fixed", justify="right")
        table.add_column("Discretionary", justify="right")

        for _, row in monthly_df.iterrows():
            table.add_row(
                str(int(row["month"])),
                f"{row['cash_balance']:.1f}",
                f"{row['inflows']:.1f}",
                f"{row['fixed_costs']:.1f}",
                f"{row['discretionary']:.1f}",
            )
        console.print(table)


@app.command("holdco-sensitivity")
def holdco_sensitivity(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path",
    ),
) -> None:
    """Run holdco sensitivity grid (bank dividend × capital call)."""
    setup_logging()

    console.print(Panel("[bold]HoldCo Sensitivity Grid[/bold]", style="blue"))

    df = scenario_grid(FY2024_HOLDCO_STATE)

    # Summary stats
    total = len(df)
    needs_funding = df["needs_external_funding"].sum()
    console.print(f"\nTotal scenarios: {total}")
    console.print(f"Needs funding: {needs_funding} ({needs_funding/total:.0%})")

    # Find threshold
    tester = HoldCoStressTester(FY2024_HOLDCO_STATE)
    capacity = tester.capacity_analysis()

    console.print("\n[bold]Capital Call Capacity:[/bold]")
    table = Table()
    table.add_column("Scenario")
    table.add_column("Max Capital Call", justify="right")

    for name, value in capacity.items():
        table.add_row(name.replace("_", " ").title(), f"{value:.0f} bn")
    console.print(table)

    if output:
        df.to_csv(output, index=False)
        console.print(f"\n[green]Saved to {output}[/green]")


# =============================================================================
# FULL PASSTHROUGH (BANK → HOLDCO)
# =============================================================================


@app.command("full-stress")
def full_stress(
    gamma: float = typer.Option(1.0, "--gamma", "-g", help="Dividend payout curvature"),
    multiplier: float = typer.Option(
        1.0, "--multiplier", "-m", help="Capital call multiplier"
    ),
) -> None:
    """Run full bank → holdco passthrough stress test."""
    setup_logging()

    console.print(Panel("[bold]Full Bank → HoldCo Stress[/bold]", style="blue"))

    config = PassthroughConfig(gamma=gamma, capital_call_multiplier=multiplier)

    console.print(f"Parameters: gamma={gamma}, multiplier={multiplier}")
    console.print()

    table = Table(title="Bank → HoldCo Passthrough")
    table.add_column("Scenario")
    table.add_column("K2 After", justify="right")
    table.add_column("Shortfall", justify="right")
    table.add_column("Div Payout", justify="right")
    table.add_column("Bank Div", justify="right")
    table.add_column("Capital Call", justify="right")
    table.add_column("Ending Cash", justify="right")
    table.add_column("Status")

    for name, bank_sc in BANK_STRESS_SCENARIOS.items():
        # Run bank stress
        bank_result = simulate_bank_stress(FY2024_BANK_STATE, bank_sc)

        # Translate to holdco scenario
        holdco_sc = build_holdco_scenario_from_bank_stress(
            bank_result, FY2024_BASELINE_SCENARIO, config
        )

        # Run holdco simulation
        holdco_result = simulate_holdco_12m(FY2024_HOLDCO_STATE, holdco_sc)

        # Calculate payout fraction for display
        from studies.kaspi_holdco.src.bank_integration import (
            compute_dividend_payout_fraction,
        )

        payout_frac = compute_dividend_payout_fraction(bank_result.k2_after, config)

        status = "[red]NEEDS FUNDING[/red]" if holdco_result.needs_external_funding else "[green]Viable[/green]"

        table.add_row(
            name,
            f"{bank_result.k2_after:.1%}",
            f"{bank_result.capital_shortfall:.0f}",
            f"{payout_frac:.0%}",
            f"{holdco_sc.dividend_from_bank:.0f}",
            f"{holdco_sc.capital_injection_to_bank:.0f}",
            f"{holdco_result.ending_cash:.0f}",
            status,
        )

    console.print(table)


@app.command("full-stress-grid")
def full_stress_grid(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path",
    ),
) -> None:
    """Run comprehensive bank → holdco stress grid."""
    setup_logging()

    console.print(Panel("[bold]Full Stress Grid[/bold]", style="blue"))

    # Run bank stress grid
    bank_df = run_bank_stress_grid(FY2024_BANK_STATE, FY2024_ANNUAL_NET_INCOME)

    # For each bank result, translate to holdco
    config = PassthroughConfig()
    results = []

    for _, row in bank_df.iterrows():
        # Create bank result from row
        bank_result = BankStressResult(
            scenario_name=row["scenario"],
            k1_2_before=row["k1_2_before"],
            k1_2_after=row["k1_2_after"],
            k2_before=row["k2_before"],
            k2_after=row["k2_after"],
            capital_shortfall=row["capital_shortfall"],
            liquidity_gap=row.get("liquidity_gap", 0),
            support_needed=row["support_needed"],
            retained_earnings=row["retained_earnings"],
            credit_losses=row["credit_losses"],
            fire_sale_losses=row["fire_sale_losses"],
            mtm_losses=row["mtm_losses"],
            total_outflows=0,
            repo_used=0,
            securities_sold=0,
            rwa_after=row["rwa_after"],
            total_capital_after=row["total_capital_after"],
        )

        # Translate to holdco
        holdco_sc = build_holdco_scenario_from_bank_stress(
            bank_result, FY2024_BASELINE_SCENARIO, config
        )

        # Run holdco simulation
        holdco_result = simulate_holdco_12m(FY2024_HOLDCO_STATE, holdco_sc)

        results.append(
            {
                "scenario": row["scenario"],
                "profit_multiplier": row["profit_multiplier"],
                "credit_loss_rate": row["credit_loss_rate"],
                "rwa_multiplier": row["rwa_multiplier"],
                "retail_run_rate": row["retail_run_rate"],
                "k2_after": row["k2_after"],
                "capital_shortfall": row["capital_shortfall"],
                "bank_dividend": holdco_sc.dividend_from_bank,
                "capital_call": holdco_sc.capital_injection_to_bank,
                "holdco_ending_cash": holdco_result.ending_cash,
                "holdco_needs_funding": holdco_result.needs_external_funding,
            }
        )

    df = pd.DataFrame(results)

    # Summary
    total = len(df)
    needs_funding = df["holdco_needs_funding"].sum()
    console.print(f"\nTotal scenarios: {total}")
    console.print(f"HoldCo needs funding: {needs_funding} ({needs_funding/total:.0%})")

    # Critical thresholds
    viable = df[~df["holdco_needs_funding"]]
    if not viable.empty:
        console.print("\n[bold]Viable Scenario Bounds:[/bold]")
        console.print(f"  Max credit loss: {viable['credit_loss_rate'].max():.1%}")
        console.print(f"  Max RWA multiplier: {viable['rwa_multiplier'].max():.0%}")
        console.print(f"  Max capital call: {viable['capital_call'].max():.0f} bn")

    if output:
        df.to_csv(output, index=False)
        console.print(f"\n[green]Saved to {output}[/green]")


# =============================================================================
# REPORTS
# =============================================================================


@app.command()
def summary() -> None:
    """Executive summary with key thresholds."""
    setup_logging()

    console.print(Panel("[bold]Kaspi.kz Capital Adequacy Summary[/bold]", style="blue"))

    # Bank state
    console.print("\n[bold cyan]1. Bank Regulatory Position[/bold cyan]")
    console.print(f"  K1-2 Ratio: {FY2024_BANK_STATE.k1_2_ratio:.1%} (min 10.5%)")
    console.print(f"  K2 Ratio: {FY2024_BANK_STATE.k2_ratio:.1%} (min 12.0%)")
    console.print(f"  K2 Headroom: {FY2024_BANK_STATE.k2_headroom:.1f} bn")
    console.print(f"  RWA: {FY2024_BANK_STATE.rwa:.0f} bn")

    # Holdco state
    console.print("\n[bold cyan]2. HoldCo Position[/bold cyan]")
    console.print(f"  Cash: {FY2024_HOLDCO_STATE.cash:.1f} bn")
    console.print(f"  Annual Fixed Costs: {FY2024_HOLDCO_STATE.annual_fixed_costs:.1f} bn")
    console.print(
        f"  Standalone Runway: {FY2024_HOLDCO_STATE.months_of_runway_standalone:.0f} months"
    )

    # Capital call capacity
    console.print("\n[bold cyan]3. Capital Call Capacity[/bold cyan]")
    tester = HoldCoStressTester(FY2024_HOLDCO_STATE)
    capacity = tester.capacity_analysis()
    for name, value in capacity.items():
        console.print(f"  {name.replace('_', ' ').title()}: {value:.0f} bn")

    # Key scenarios
    console.print("\n[bold cyan]4. Stress Scenario Results[/bold cyan]")
    config = PassthroughConfig()

    for name in ["mild", "moderate", "severe", "oil_crisis"]:
        bank_sc = BANK_STRESS_SCENARIOS[name]
        bank_result = simulate_bank_stress(FY2024_BANK_STATE, bank_sc)
        holdco_sc = build_holdco_scenario_from_bank_stress(
            bank_result, FY2024_BASELINE_SCENARIO, config
        )
        holdco_result = simulate_holdco_12m(FY2024_HOLDCO_STATE, holdco_sc)

        status = "NEEDS FUNDING" if holdco_result.needs_external_funding else "Viable"
        console.print(
            f"  {name.title()}: K2={bank_result.k2_after:.1%}, "
            f"Shortfall={bank_result.capital_shortfall:.0f}, "
            f"HoldCo Cash={holdco_result.ending_cash:.0f} [{status}]"
        )

    # Key findings
    console.print("\n[bold cyan]5. Key Findings[/bold cyan]")
    console.print("  - Bank K2 buffer is tight (70 bps above minimum)")
    console.print("  - NBK RWA ~45% higher than Basel III (consumer risk weights)")
    console.print(f"  - HoldCo can absorb ~{capacity['no_shareholder_dividend']:.0f} bn capital call")
    console.print("    (if shareholder dividends suspended)")
    console.print("  - Severe stress (5% credit loss, 15% RWA inflation) requires support")


@app.command()
def export(
    output: Path = typer.Option(
        Path("kaspi_holdco_results.json"),
        "--output",
        "-o",
        help="Output JSON path",
    ),
) -> None:
    """Export all results to JSON."""
    setup_logging()

    console.print(Panel("[bold]Exporting Results[/bold]", style="blue"))

    config = PassthroughConfig()
    tester = HoldCoStressTester(FY2024_HOLDCO_STATE)

    results = {
        "bank_state": {
            "k1_2_ratio": FY2024_BANK_STATE.k1_2_ratio,
            "k2_ratio": FY2024_BANK_STATE.k2_ratio,
            "k2_headroom": FY2024_BANK_STATE.k2_headroom,
            "rwa": FY2024_BANK_STATE.rwa,
            "total_capital": FY2024_BANK_STATE.total_capital,
        },
        "holdco_state": {
            "cash": FY2024_HOLDCO_STATE.cash,
            "annual_fixed_costs": FY2024_HOLDCO_STATE.annual_fixed_costs,
            "standalone_runway_months": FY2024_HOLDCO_STATE.months_of_runway_standalone,
        },
        "capital_call_capacity": tester.capacity_analysis(),
        "stress_scenarios": {},
    }

    for name, bank_sc in BANK_STRESS_SCENARIOS.items():
        bank_result = simulate_bank_stress(FY2024_BANK_STATE, bank_sc)
        holdco_sc = build_holdco_scenario_from_bank_stress(
            bank_result, FY2024_BASELINE_SCENARIO, config
        )
        holdco_result = simulate_holdco_12m(FY2024_HOLDCO_STATE, holdco_sc)

        results["stress_scenarios"][name] = {
            "bank": bank_result.to_dict(),
            "holdco": holdco_result.to_dict(),
        }

    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"[green]Exported to {output}[/green]")


if __name__ == "__main__":
    app()
