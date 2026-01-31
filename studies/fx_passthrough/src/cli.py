"""
CLI for FX Passthrough Study.

Usage:
    kzresearch passthrough fetch-data [source]
    kzresearch passthrough build-cpi-panel
    kzresearch passthrough build-income-series
    kzresearch passthrough estimate [block]
    kzresearch passthrough falsification
    kzresearch passthrough run-full-chain
"""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(
    name="passthrough",
    help="FX-to-Expenditure Causal Chain Study",
)
console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with rich output."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


# =============================================================================
# Data Commands
# =============================================================================

@app.command("fetch-data")
def fetch_data(
    source: str = typer.Argument(
        "all",
        help="Data source: nbk, bns-cpi, bns-income, import-intensity, all",
    ),
    save: bool = typer.Option(True, help="Save raw data to disk"),
):
    """Fetch data from specified source."""
    setup_logging()

    from studies.fx_passthrough.src.national_panel import FXPassthroughPipeline

    pipeline = FXPassthroughPipeline()

    if source == "all":
        console.print("[bold]Fetching all FX passthrough data sources...[/bold]")
        data = pipeline.fetch_all_fx_data()
        console.print(f"Fetched {len(data)} data sources")

        if save:
            for name, df in data.items():
                if not df.empty:
                    console.print(f"  {name}: {len(df)} rows")

    elif source == "nbk":
        console.print("[bold]Fetching NBK exchange rates...[/bold]")
        df = pipeline.exchange_rate_client.fetch_with_cache(tier=1)
        console.print(f"Fetched {len(df)} observations")
        if save:
            pipeline.exchange_rate_client.save_all_raw()

    elif source == "bns-cpi":
        console.print("[bold]Fetching BNS CPI categories...[/bold]")
        df = pipeline.cpi_client.fetch_with_cache()
        console.print(f"Fetched {len(df)} observations")
        if save:
            pipeline.cpi_client.save_all_raw()

    elif source == "bns-income":
        console.print("[bold]Fetching BNS national income...[/bold]")
        df = pipeline.national_income_client.fetch_quarterly()
        console.print(f"Fetched {len(df)} observations")
        if save:
            pipeline.national_income_client.save_all_raw()

    elif source == "import-intensity":
        console.print("[bold]Fetching import intensity...[/bold]")
        df = pipeline.import_intensity_client.fetch_with_cache()
        console.print(f"Fetched {len(df)} category estimates")
        if save:
            pipeline.import_intensity_client.save_all_raw()

    else:
        console.print(f"[red]Unknown source: {source}[/red]")
        console.print("Available: nbk, bns-cpi, bns-income, import-intensity, all")
        raise typer.Exit(1)


# =============================================================================
# Panel Building Commands
# =============================================================================

@app.command("build-cpi-panel")
def build_cpi_panel(
    start_date: str = typer.Option("2010-01-01", help="Start date"),
    end_date: Optional[str] = typer.Option(None, help="End date"),
    output: Optional[Path] = typer.Option(None, help="Output path"),
):
    """Build CPI category panel for Block A."""
    setup_logging()

    from studies.fx_passthrough.src.national_panel import NationalPanelBuilder

    builder = NationalPanelBuilder()

    console.print(f"[bold]Building CPI panel from {start_date}...[/bold]")
    panel = builder.build_cpi_panel(start_date, end_date)

    if output:
        panel.to_parquet(output)
        console.print(f"Saved to {output}")
    else:
        paths = builder.save_panels({"cpi_panel": panel})
        console.print(f"Saved to {paths['cpi_panel']}")

    console.print(f"\nPanel shape: {panel.shape}")
    console.print(f"Categories: {panel['category'].nunique()}")
    console.print(f"Months: {panel['date'].nunique()}")

    builder.print_quality_summary()


@app.command("build-income-series")
def build_income_series(
    start_date: str = typer.Option("2010-01-01", help="Start date"),
    end_date: Optional[str] = typer.Option(None, help="End date"),
    frequency: str = typer.Option("quarterly", help="Frequency: monthly, quarterly"),
    output: Optional[Path] = typer.Option(None, help="Output path"),
):
    """Build national income time series for Blocks B-D."""
    setup_logging()

    from studies.fx_passthrough.src.national_panel import NationalPanelBuilder

    builder = NationalPanelBuilder()

    console.print(f"[bold]Building income series from {start_date}...[/bold]")
    series = builder.build_income_series(start_date, end_date, frequency)

    if output:
        series.to_parquet(output)
        console.print(f"Saved to {output}")
    else:
        paths = builder.save_panels({"income_series": series})
        console.print(f"Saved to {paths['income_series']}")

    console.print(f"\nSeries shape: {series.shape}")
    console.print(f"Columns: {list(series.columns)}")

    builder.print_quality_summary()


@app.command("build-expenditure-series")
def build_expenditure_series(
    start_date: str = typer.Option("2010-01-01", help="Start date"),
    end_date: Optional[str] = typer.Option(None, help="End date"),
    output: Optional[Path] = typer.Option(None, help="Output path"),
):
    """Build expenditure series for Block E."""
    setup_logging()

    from studies.fx_passthrough.src.national_panel import NationalPanelBuilder

    builder = NationalPanelBuilder()

    console.print(f"[bold]Building expenditure series from {start_date}...[/bold]")
    series = builder.build_expenditure_series(start_date, end_date)

    if output:
        series.to_parquet(output)
    else:
        paths = builder.save_panels({"expenditure_series": series})
        console.print(f"Saved to {paths['expenditure_series']}")

    console.print(f"\nSeries shape: {series.shape}")


# =============================================================================
# Estimation Commands
# =============================================================================

@app.command("estimate")
def estimate(
    block: str = typer.Argument(
        "all",
        help="Block to estimate: block-a, block-b, block-c, block-d, block-e, all",
    ),
    exclude_admin: bool = typer.Option(True, help="Exclude admin prices (Block A)"),
    max_horizon: int = typer.Option(12, help="Maximum horizon"),
    outcome: Optional[str] = typer.Option(None, help="Outcome variable (Block B)"),
):
    """Estimate causal chain blocks."""
    setup_logging()

    import pandas as pd
    from config.settings import get_settings

    settings = get_settings()
    data_dir = settings.project_root / settings.processed_data_dir / "fx_passthrough"

    if block in ["block-a", "all"]:
        console.print("[bold]Estimating Block A: CPI Pass-Through...[/bold]")

        cpi_path = data_dir / "cpi_panel.parquet"
        if not cpi_path.exists():
            console.print("[red]CPI panel not found. Run 'build-cpi-panel' first.[/red]")
            raise typer.Exit(1)

        cpi_panel = pd.read_parquet(cpi_path)

        from studies.fx_passthrough.src.cpi_pass_through import (
            CPIPassThroughModel,
            CPIPassThroughSpec,
        )

        spec = CPIPassThroughSpec(
            exclude_admin_prices=exclude_admin,
            max_horizon=max_horizon,
        )

        model = CPIPassThroughModel(cpi_panel)
        result = model.fit(spec)

        console.print(result.summary())

    if block in ["block-b", "all"]:
        console.print("\n[bold]Estimating Block B: Income LP-IV...[/bold]")

        income_path = data_dir / "income_series.parquet"
        if not income_path.exists():
            console.print("[red]Income series not found. Run 'build-income-series' first.[/red]")
            if block != "all":
                raise typer.Exit(1)
        else:
            income_data = pd.read_parquet(income_path)

            from studies.fx_passthrough.src.income_lp_iv import (
                IncomeLPIVModel,
                IncomeLPIVSpec,
            )

            if outcome:
                spec = IncomeLPIVSpec(outcome=outcome, max_horizon=max_horizon)
                model = IncomeLPIVModel(income_data)
                result = model.fit(spec)
                console.print(result.summary())
            else:
                model = IncomeLPIVModel(income_data)
                results = model.fit_all_outcomes()
                for name, result in results.items():
                    console.print(result.summary())

    if block in ["block-c", "all"]:
        console.print("\n[bold]Estimating Block C: Real Income Decomposition...[/bold]")
        console.print("[yellow]Block C requires Blocks A and B results.[/yellow]")
        console.print("Run 'run-full-chain' for complete analysis.")

    if block in ["block-d", "all"]:
        console.print("\n[bold]Estimating Block D: Transfer Mechanism...[/bold]")

        income_path = data_dir / "income_series.parquet"
        if income_path.exists():
            income_data = pd.read_parquet(income_path)

            from studies.fx_passthrough.src.transfer_mechanism import (
                TransferMechanismModel,
            )

            model = TransferMechanismModel(income_data)
            results = model.fit_all()

            for name, result in results.items():
                console.print(result.summary())
        else:
            console.print("[red]Income series not found.[/red]")

    if block in ["block-e", "all"]:
        console.print("\n[bold]Estimating Block E: Expenditure Response...[/bold]")

        exp_path = data_dir / "expenditure_series.parquet"
        if not exp_path.exists():
            exp_path = data_dir / "income_series.parquet"

        if exp_path.exists():
            exp_data = pd.read_parquet(exp_path)

            from studies.fx_passthrough.src.expenditure_response import (
                ExpenditureLPIVModel,
            )

            model = ExpenditureLPIVModel(exp_data)
            result = model.fit()
            console.print(result.summary())
        else:
            console.print("[red]Expenditure/income data not found.[/red]")


# =============================================================================
# Falsification Commands
# =============================================================================

@app.command("falsification")
def falsification(
    permutation_n: int = typer.Option(1000, help="Number of permutations"),
    bootstrap_n: int = typer.Option(999, help="Number of bootstrap iterations"),
):
    """Run falsification tests."""
    setup_logging()

    import pandas as pd
    from config.settings import get_settings

    settings = get_settings()
    data_dir = settings.project_root / settings.processed_data_dir / "fx_passthrough"

    console.print("[bold]Running Falsification Tests[/bold]")
    console.print("=" * 50)

    cpi_path = data_dir / "cpi_panel.parquet"
    if not cpi_path.exists():
        console.print("[red]CPI panel not found. Run 'build-cpi-panel' first.[/red]")
        raise typer.Exit(1)

    cpi_panel = pd.read_parquet(cpi_path)

    from studies.fx_passthrough.src.cpi_pass_through import (
        CPIPassThroughModel,
        CPIPassThroughSpec,
    )

    spec = CPIPassThroughSpec(
        permutation_iterations=permutation_n,
        bootstrap_iterations=bootstrap_n,
    )

    model = CPIPassThroughModel(cpi_panel)

    # Run all falsification tests
    results = model.run_all_falsification(spec)

    # Display results
    table = Table(title="Falsification Test Results")
    table.add_column("Test", style="cyan")
    table.add_column("Result", style="white")
    table.add_column("Pass", style="green")

    # Pre-trends
    pre_trends = results.get("pre_trends", {})
    table.add_row(
        "Pre-trends (leads)",
        f"Joint p = {pre_trends.get('joint_pvalue', 'N/A'):.4f}" if pre_trends.get('joint_pvalue') else "N/A",
        "[green]PASS[/green]" if pre_trends.get("pre_trends_pass") else "[red]FAIL[/red]",
    )

    # Admin prices
    admin = results.get("admin_prices", {})
    table.add_row(
        "Admin prices",
        f"β = {admin.get('admin_coefficient', 'N/A'):.4f}" if admin.get('admin_coefficient') else "N/A",
        "[green]PASS[/green]" if admin.get("admin_test_pass") else "[red]FAIL[/red]",
    )

    console.print(table)

    if results.get("all_tests_pass"):
        console.print("\n[bold green]All falsification tests PASSED[/bold green]")
    else:
        console.print("\n[bold red]Some falsification tests FAILED[/bold red]")


@app.command("structural-break")
def structural_break(
    break_date: str = typer.Option("2015-08", help="Structural break date"),
):
    """Run structural break analysis (pre/post tenge float)."""
    setup_logging()

    import pandas as pd
    from config.settings import get_settings

    settings = get_settings()
    data_dir = settings.project_root / settings.processed_data_dir / "fx_passthrough"

    console.print(f"[bold]Structural Break Analysis: {break_date}[/bold]")

    cpi_path = data_dir / "cpi_panel.parquet"
    if not cpi_path.exists():
        console.print("[red]CPI panel not found.[/red]")
        raise typer.Exit(1)

    from studies.fx_passthrough.src.causal_chain import (
        CausalChainAnalysis,
        CausalChainConfig,
    )

    cpi_panel = pd.read_parquet(cpi_path)

    config = CausalChainConfig(structural_break_date=break_date)
    analysis = CausalChainAnalysis(cpi_panel=cpi_panel, config=config)

    results = analysis.run_structural_break()

    console.print(f"\nBreak date: {results.get('break_date')}")

    if "pre_period" in results:
        console.print("\n[bold]Pre-float period:[/bold]")
        pre = results["pre_period"]
        if "beta" in pre:
            console.print(f"  β = {pre['beta']:.4f} (SE: {pre['se']:.4f})")
            console.print(f"  p-value: {pre['pvalue']:.4f}")
            console.print(f"  N obs: {pre['n_obs']}")
        else:
            console.print(f"  Error: {pre.get('error')}")

    if "post_period" in results:
        console.print("\n[bold]Post-float period:[/bold]")
        post = results["post_period"]
        if "beta" in post:
            console.print(f"  β = {post['beta']:.4f} (SE: {post['se']:.4f})")
            console.print(f"  p-value: {post['pvalue']:.4f}")
            console.print(f"  N obs: {post['n_obs']}")
        else:
            console.print(f"  Error: {post.get('error')}")

    if "beta_change" in results:
        console.print(f"\nβ change: {results['beta_change']:.4f}")
        console.print(f"β change (%): {results.get('beta_change_pct', 'N/A'):.1f}%")


# =============================================================================
# Full Chain Command
# =============================================================================

@app.command("run-full-chain")
def run_full_chain(
    save_results: bool = typer.Option(True, help="Save results to files"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory"),
):
    """Run complete causal chain analysis."""
    setup_logging()

    import pandas as pd
    from config.settings import get_settings

    settings = get_settings()
    data_dir = settings.project_root / settings.processed_data_dir / "fx_passthrough"

    console.print("[bold cyan]FX PASSTHROUGH CAUSAL CHAIN ANALYSIS[/bold cyan]")
    console.print("=" * 60)

    # Load data
    panels = {}
    for name in ["cpi_panel", "income_series", "expenditure_series"]:
        path = data_dir / f"{name}.parquet"
        if path.exists():
            panels[name] = pd.read_parquet(path)
            console.print(f"Loaded {name}: {len(panels[name])} rows")
        else:
            console.print(f"[yellow]Warning: {name} not found[/yellow]")

    if not panels:
        console.print("[red]No data found. Run build commands first.[/red]")
        raise typer.Exit(1)

    from studies.fx_passthrough.src.causal_chain import (
        CausalChainAnalysis,
        CausalChainConfig,
    )

    config = CausalChainConfig()
    if output_dir:
        config.output_dir = output_dir

    analysis = CausalChainAnalysis(
        cpi_panel=panels.get("cpi_panel"),
        income_data=panels.get("income_series"),
        expenditure_data=panels.get("expenditure_series"),
        config=config,
    )

    summary = analysis.run_full_chain()

    console.print("\n" + summary.summary())

    if save_results:
        paths = analysis.save_results()
        console.print(f"\nResults saved to: {list(paths.values())}")


# =============================================================================
# Utility Commands
# =============================================================================

@app.command("info")
def info():
    """Show study information."""
    console.print("\n[bold cyan]FX Passthrough Study[/bold cyan]")
    console.print("\nResearch Question:")
    console.print("  How do exchange rate shocks affect household welfare through")
    console.print("  inflation, income, and expenditure channels?\n")

    console.print("Causal Chain:")
    console.print("  FX -> Inflation -> Income/Transfers -> Real Income -> Expenditure\n")

    table = Table(title="Block Structure")
    table.add_column("Block", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Method", style="green")

    table.add_row("A", "CPI Pass-Through", "Category DiD")
    table.add_row("B", "Income Response", "LP-IV")
    table.add_row("C", "Real Income", "Accounting Identity")
    table.add_row("D", "Transfer Mechanism", "IV Tests")
    table.add_row("E", "Expenditure Response", "LP-IV")

    console.print(table)

    console.print("\nCommands:")
    console.print("  fetch-data          Fetch raw data")
    console.print("  build-cpi-panel     Build CPI panel for Block A")
    console.print("  build-income-series Build income series for Blocks B-D")
    console.print("  estimate            Estimate blocks")
    console.print("  falsification       Run falsification tests")
    console.print("  structural-break    Pre/post 2015 analysis")
    console.print("  run-full-chain      Run complete analysis")


if __name__ == "__main__":
    app()
