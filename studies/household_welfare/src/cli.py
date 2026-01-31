"""
CLI for Kazakhstan Household Welfare Study.

Usage:
    kzwelfare fetch-data <source>
    kzwelfare build-panel
    kzwelfare estimate <spec>
    kzwelfare local-projections
    kzwelfare simulate <scenario>
"""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="welfare",
    help="Kazakhstan Household Welfare Causal Econometric Model",
)
console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with rich output."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def fetch_data(
    source: str = typer.Argument(..., help="Data source: bns, fred, baumeister, all"),
    save: bool = typer.Option(True, help="Save raw data to disk"),
):
    """Fetch data from specified source."""
    setup_logging()

    from studies.household_welfare.src.data_pipeline import WelfareDataPipeline

    pipeline = WelfareDataPipeline()

    if source == "all":
        console.print("[bold]Fetching all data sources...[/bold]")
        if save:
            paths = pipeline.save_all_raw()
            console.print(f"Saved {sum(len(p) for p in paths.values())} files")
        else:
            data = pipeline.fetch_all_raw()
            console.print(f"Fetched data from {len(data)} sources")
    elif source == "bns":
        console.print("[bold]Fetching BNS data...[/bold]")
        paths = pipeline.bns_client.save_all_raw() if save else pipeline.bns_client.fetch_all()
        console.print(f"Fetched {len(paths)} BNS datasets")
    elif source == "fred":
        console.print("[bold]Fetching FRED data...[/bold]")
        paths = pipeline.fred_client.save_all_raw() if save else pipeline.fred_client.fetch_all_quarterly()
        console.print(f"Fetched {len(paths)} FRED series")
    elif source == "baumeister":
        console.print("[bold]Fetching Baumeister shocks...[/bold]")
        df = pipeline.baumeister_loader.fetch()
        if save:
            pipeline.baumeister_loader.save_raw(df)
        console.print(f"Fetched {len(df)} observations")
    else:
        console.print(f"[red]Unknown source: {source}[/red]")
        raise typer.Exit(1)


@app.command()
def build_panel(
    start_year: int = typer.Option(2010, help="Panel start year"),
    end_year: int = typer.Option(2024, help="Panel end year"),
    output: Optional[Path] = typer.Option(None, help="Output path"),
):
    """Build analysis panel from all data sources."""
    setup_logging()

    from studies.household_welfare.src.data_pipeline import WelfareDataPipeline

    pipeline = WelfareDataPipeline()

    console.print(f"[bold]Building panel from {start_year} to {end_year}...[/bold]")

    panel = pipeline.build_panel(start_year=start_year, end_year=end_year)

    if output:
        panel.to_parquet(output)
        console.print(f"Saved panel to {output}")
    else:
        path = pipeline.save_panel(panel)
        console.print(f"Saved panel to {path}")

    console.print(f"Panel shape: {panel.shape}")
    pipeline.print_quality_summary()


@app.command()
def estimate(
    spec: str = typer.Argument("baseline", help="Specification: baseline, auxiliary, all"),
    panel_path: Optional[Path] = typer.Option(None, help="Path to panel data"),
):
    """Estimate shift-share regression."""
    setup_logging()

    import pandas as pd
    from studies.household_welfare.src.shift_share import (
        ShiftShareModel,
        MAIN_SPEC,
        ROBUSTNESS_SPEC,
    )

    # Load panel
    if panel_path:
        panel = pd.read_parquet(panel_path)
    else:
        from config.settings import get_settings
        settings = get_settings()
        panel_path = settings.project_root / settings.processed_data_dir / "panel.parquet"
        if not panel_path.exists():
            console.print("[red]Panel not found. Run 'build-panel' first.[/red]")
            raise typer.Exit(1)
        panel = pd.read_parquet(panel_path)

    console.print(f"[bold]Estimating {spec} specification...[/bold]")

    model = ShiftShareModel(panel)

    if spec == "baseline":
        model.fit(MAIN_SPEC)
    elif spec == "auxiliary" or spec == "robustness":
        model.fit(ROBUSTNESS_SPEC)
    elif spec == "all":
        model.fit_all_specs()
    else:
        console.print(f"[red]Unknown specification: {spec}[/red]")
        raise typer.Exit(1)

    console.print(model.summary())


@app.command()
def local_projections(
    max_horizon: int = typer.Option(12, help="Maximum horizon (quarters)"),
    panel_path: Optional[Path] = typer.Option(None, help="Path to panel data"),
):
    """Estimate local projections for dynamic IRFs."""
    setup_logging()

    import pandas as pd
    from studies.household_welfare.src.local_projections import LocalProjections

    # Load panel
    if panel_path:
        panel = pd.read_parquet(panel_path)
    else:
        from config.settings import get_settings
        settings = get_settings()
        panel_path = settings.project_root / settings.processed_data_dir / "panel.parquet"
        if not panel_path.exists():
            console.print("[red]Panel not found. Run 'build-panel' first.[/red]")
            raise typer.Exit(1)
        panel = pd.read_parquet(panel_path)

    console.print(f"[bold]Estimating local projections (h=0 to {max_horizon})...[/bold]")

    lp = LocalProjections(panel)
    lp.fit(max_horizon=max_horizon)

    console.print(lp.summary())


@app.command()
def simulate(
    scenario: str = typer.Argument(..., help="Scenario name"),
):
    """Simulate a scenario."""
    setup_logging()

    from studies.household_welfare.src.simulator import WelfareSimulator

    console.print(f"[bold]Simulating scenario: {scenario}[/bold]")

    simulator = WelfareSimulator()

    # Check for predefined scenarios
    if scenario == "oil_supply_disruption":
        result = simulator.simulate_oil_shock(-0.10)  # 10% oil price drop
    elif scenario == "oil_boom":
        result = simulator.simulate_oil_shock(0.20)  # 20% oil price increase
    else:
        console.print(f"[yellow]Unknown scenario: {scenario}[/yellow]")
        console.print("Available scenarios: oil_supply_disruption, oil_boom")
        raise typer.Exit(1)

    console.print(result.summary())


@app.command()
def quality_report(
    panel_path: Optional[Path] = typer.Option(None, help="Path to panel data"),
):
    """Generate data quality report."""
    setup_logging()

    import pandas as pd

    # Load panel
    if panel_path:
        panel = pd.read_parquet(panel_path)
    else:
        from config.settings import get_settings
        settings = get_settings()
        panel_path = settings.project_root / settings.processed_data_dir / "panel.parquet"
        if not panel_path.exists():
            console.print("[red]Panel not found. Run 'build-panel' first.[/red]")
            raise typer.Exit(1)
        panel = pd.read_parquet(panel_path)

    console.print("[bold]Data Quality Report[/bold]")
    console.print(f"\nPanel shape: {panel.shape}")
    console.print(f"Columns: {list(panel.columns)}")

    # Missing values
    console.print("\nMissing Values:")
    missing = panel.isnull().sum()
    for col, count in missing[missing > 0].items():
        pct = count / len(panel) * 100
        console.print(f"  {col}: {count} ({pct:.1f}%)")

    # Date range
    if "quarter" in panel.columns:
        console.print(f"\nDate range: {panel['quarter'].min()} to {panel['quarter'].max()}")

    # Regions
    if "region" in panel.columns:
        console.print(f"\nRegions: {panel['region'].nunique()}")
        console.print(f"  {list(panel['region'].unique())}")


if __name__ == "__main__":
    app()
