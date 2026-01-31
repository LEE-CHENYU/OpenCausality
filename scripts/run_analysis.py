#!/usr/bin/env python3
"""
Main analysis script for Kazakhstan Household Welfare Model.

This script runs the complete analysis pipeline:
1. Fetch and process data
2. Build panel
3. Estimate models
4. Run falsification tests
5. Generate scenario simulations
"""

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# Setup logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)


def run_data_pipeline(save_raw: bool = True) -> None:
    """Fetch and process all data."""
    from src.data.data_pipeline import DataPipeline

    console.print("[bold blue]Step 1: Data Collection[/bold blue]")

    pipeline = DataPipeline()

    if save_raw:
        console.print("Saving raw data...")
        paths = pipeline.save_all_raw()
        console.print(f"Saved {sum(len(p) for p in paths.values())} files")

    console.print("Building panel...")
    panel = pipeline.build_panel()
    pipeline.save_panel(panel)

    console.print(f"Panel shape: {panel.shape}")
    pipeline.print_quality_summary()

    return panel


def run_estimation(panel):
    """Estimate shift-share and local projection models."""
    from src.model.shift_share import ShiftShareModel, BASELINE_SPEC
    from src.model.local_projections import LocalProjections, LocalProjectionSpec
    from src.engine.multipliers import get_multiplier_store

    console.print("\n[bold blue]Step 2: Model Estimation[/bold blue]")

    # Shift-share
    console.print("Fitting shift-share model...")
    model = ShiftShareModel(panel)
    model.fit(BASELINE_SPEC)
    console.print(model.summary())

    # Save multipliers
    store = get_multiplier_store()
    store.from_shift_share_results(model.results)

    # Local projections
    console.print("\nFitting local projections...")
    lp = LocalProjections(panel)
    spec = LocalProjectionSpec(max_horizon=8)
    lp.fit(spec)
    console.print(lp.summary())

    # Save LP multipliers
    store.from_local_projections(lp.irf_results)

    return model, lp


def run_falsification(panel):
    """Run falsification tests."""
    from src.model.falsification import FalsificationTests

    console.print("\n[bold blue]Step 3: Falsification Tests[/bold blue]")

    tests = FalsificationTests(panel)
    tests.run_all()

    console.print(tests.summary())

    return tests.results


def run_scenarios():
    """Run scenario simulations."""
    from src.engine.simulator import run_scenario

    console.print("\n[bold blue]Step 4: Scenario Simulations[/bold blue]")

    scenarios = [
        "oil_collapse_2014",
        "pandemic_2020",
        "oil_supply_disruption",
    ]

    results = {}
    for scenario_name in scenarios:
        console.print(f"\nSimulating: {scenario_name}")
        result = run_scenario(scenario_name)
        results[scenario_name] = result
        console.print(result.summary())

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Kazakhstan Welfare Analysis")
    parser.add_argument("--skip-data", action="store_true", help="Skip data collection")
    parser.add_argument("--skip-estimation", action="store_true", help="Skip estimation")
    parser.add_argument("--skip-falsification", action="store_true", help="Skip falsification")
    parser.add_argument("--skip-scenarios", action="store_true", help="Skip scenarios")
    args = parser.parse_args()

    console.print("[bold green]Kazakhstan Household Welfare Model[/bold green]")
    console.print("=" * 50)

    # Step 1: Data
    if not args.skip_data:
        panel = run_data_pipeline()
    else:
        import pandas as pd
        from config.settings import get_settings
        settings = get_settings()
        panel_path = settings.project_root / settings.processed_data_dir / "panel.parquet"
        if panel_path.exists():
            panel = pd.read_parquet(panel_path)
        else:
            console.print("[red]No panel found. Run without --skip-data first.[/red]")
            return

    # Step 2: Estimation
    if not args.skip_estimation:
        model, lp = run_estimation(panel)

    # Step 3: Falsification
    if not args.skip_falsification:
        falsification_results = run_falsification(panel)

    # Step 4: Scenarios
    if not args.skip_scenarios:
        scenario_results = run_scenarios()

    console.print("\n[bold green]Analysis complete![/bold green]")


if __name__ == "__main__":
    main()
