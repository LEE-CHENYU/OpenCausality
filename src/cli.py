"""
CLI for Kazakhstan Household Welfare Model.
"""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="kzwelfare",
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

    from src.data.data_pipeline import DataPipeline

    pipeline = DataPipeline()

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

    from src.data.data_pipeline import DataPipeline

    pipeline = DataPipeline()

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
    from src.model.shift_share import ShiftShareModel, BASELINE_SPEC, AUXILIARY_SPEC

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
        model.fit(BASELINE_SPEC)
    elif spec == "auxiliary":
        model.fit(AUXILIARY_SPEC)
    elif spec == "all":
        model.fit_all_specs()
    else:
        console.print(f"[red]Unknown specification: {spec}[/red]")
        raise typer.Exit(1)

    console.print(model.summary())

    # Save multipliers
    from src.engine.multipliers import get_multiplier_store
    store = get_multiplier_store()
    store.from_shift_share_results(model.results)
    console.print("\nMultipliers saved.")


@app.command()
def local_projections(
    max_horizon: int = typer.Option(12, help="Maximum horizon (quarters)"),
    panel_path: Optional[Path] = typer.Option(None, help="Path to panel data"),
):
    """Estimate local projections for dynamic IRFs."""
    setup_logging()

    import pandas as pd
    from src.model.local_projections import LocalProjections, LocalProjectionSpec

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

    spec = LocalProjectionSpec(max_horizon=max_horizon)
    lp = LocalProjections(panel)
    lp.fit(spec)

    console.print(lp.summary())

    # Save multipliers
    from src.engine.multipliers import get_multiplier_store
    store = get_multiplier_store()
    store.from_local_projections(lp.irf_results)
    console.print("\nIRF multipliers saved.")


@app.command()
def falsification(
    panel_path: Optional[Path] = typer.Option(None, help="Path to panel data"),
):
    """Run falsification tests."""
    setup_logging()

    import pandas as pd
    from src.model.falsification import FalsificationTests

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

    console.print("[bold]Running falsification tests...[/bold]")

    tests = FalsificationTests(panel)
    tests.run_all()

    console.print(tests.summary())


@app.command()
def simulate(
    scenario: str = typer.Argument(..., help="Scenario name"),
    multipliers: str = typer.Option("shift_share", help="Multiplier set to use"),
):
    """Simulate a scenario."""
    setup_logging()

    from src.engine.simulator import run_scenario

    console.print(f"[bold]Simulating scenario: {scenario}[/bold]")

    result = run_scenario(scenario, multipliers)

    console.print(result.summary())

    # Save results
    df = result.to_dataframe()
    from config.settings import get_settings
    settings = get_settings()
    output_path = settings.project_root / settings.output_dir / f"simulation_{scenario}.csv"
    df.to_csv(output_path, index=False)
    console.print(f"\nResults saved to {output_path}")


@app.command()
def backtest(
    episode: str = typer.Argument(..., help="Historical episode: oil_collapse_2014, pandemic_2020, energy_crisis_2022"),
    panel_path: Optional[Path] = typer.Option(None, help="Path to panel data"),
):
    """Backtest against historical episode."""
    setup_logging()

    import pandas as pd
    from src.engine.shock_paths import get_historical_scenario
    from src.engine.simulator import ScenarioSimulator

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

    console.print(f"[bold]Backtesting: {episode}[/bold]")

    scenario = get_historical_scenario(episode)
    simulator = ScenarioSimulator()
    metrics = simulator.backtest(scenario, panel)

    console.print(f"\nBacktest Results: {episode}")
    console.print(f"Quarters: {metrics['quarters']}")

    if "aggregate" in metrics:
        console.print(f"\nAggregate Metrics:")
        console.print(f"  Mean RMSE: {metrics['aggregate']['mean_rmse']:.4f}")
        console.print(f"  Mean Correlation: {metrics['aggregate']['mean_correlation']:.4f}")
        console.print(f"  Mean Bias: {metrics['aggregate']['mean_bias']:.4f}")


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
