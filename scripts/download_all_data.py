#!/usr/bin/env python3
"""
Systematic data download script with status tracking.

Downloads all data sources required for research analysis and tracks
download status in data/metadata/download_status.json.

Run with: PYTHONPATH=. python scripts/download_all_data.py [--verify]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Setup logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_path=False)],
)
logger = logging.getLogger(__name__)


# Data source configuration
DATA_SOURCES = {
    # FRED data (already downloaded)
    "fred_igrea": {
        "path": "data/raw/fred/IGREA.parquet",
        "fetcher": "FREDClient",
        "series": "IGREA",
        "description": "Kilian Global Real Economic Activity Index",
        "optional": False,
    },
    "fred_vix": {
        "path": "data/raw/fred/VIXCLS.parquet",
        "fetcher": "FREDClient",
        "series": "VIXCLS",
        "description": "CBOE Volatility Index",
        "optional": False,
    },
    "fred_brent": {
        "path": "data/raw/fred/DCOILBRENTEU.parquet",
        "fetcher": "FREDClient",
        "series": "DCOILBRENTEU",
        "description": "Brent Crude Oil Price",
        "optional": False,
    },
    # Baumeister shocks
    "baumeister_shocks": {
        "path": "data/raw/baumeister_shocks/shocks.parquet",
        "fetcher": "BaumeisterLoader",
        "description": "Baumeister-Hamilton Oil Supply Shocks",
        "optional": False,
    },
    # NBK Exchange Rate
    "nbk_usd_kzt": {
        "path": "data/raw/nbk/usd_kzt.parquet",
        "fetcher": "ExchangeRateClient",
        "tier": 1,
        "description": "NBK USD/KZT Exchange Rate",
        "optional": False,
    },
    # BNS Kazakhstan data
    "bns_cpi_categories": {
        "path": "data/raw/kazakhstan_bns/cpi_categories.parquet",
        "fetcher": "BNSCPICategoriesClient",
        "description": "BNS CPI by COICOP Category",
        "optional": False,
    },
    "bns_national_income": {
        "path": "data/raw/kazakhstan_bns/national_income.parquet",
        "fetcher": "BNSNationalIncomeClient",
        "description": "BNS National Income by Region",
        "optional": False,
    },
    # World Bank (optional, annual only)
    "worldbank_reer": {
        "path": "data/raw/worldbank/reer_kz.parquet",
        "fetcher": "ExchangeRateClient",
        "tier": 3,
        "description": "World Bank REER (annual, robustness only)",
        "optional": True,
    },
}


class DownloadStatusTracker:
    """Track download status for all data sources."""

    def __init__(self, status_file: Path):
        self.status_file = status_file
        self.status: dict[str, Any] = self._load_status()

    def _load_status(self) -> dict[str, Any]:
        """Load existing status file or create new."""
        if self.status_file.exists():
            with open(self.status_file) as f:
                return json.load(f)
        return {
            "version": "1.0",
            "last_updated": None,
            "sources": {},
        }

    def save(self) -> None:
        """Save status to file."""
        self.status["last_updated"] = datetime.now().isoformat()
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, "w") as f:
            json.dump(self.status, f, indent=2)

    def update_source(
        self,
        source_name: str,
        status: str,
        n_rows: int = 0,
        date_range: tuple[str, str] | None = None,
        error: str | None = None,
    ) -> None:
        """Update status for a data source."""
        self.status["sources"][source_name] = {
            "status": status,
            "n_rows": n_rows,
            "date_range": list(date_range) if date_range else None,
            "last_download": datetime.now().isoformat(),
            "error": error,
        }

    def get_source_status(self, source_name: str) -> dict[str, Any] | None:
        """Get status for a specific source."""
        return self.status["sources"].get(source_name)

    def has_synthetic_data(self) -> bool:
        """Check if any source has synthetic data."""
        for source in self.status["sources"].values():
            if source.get("status") == "synthetic":
                return True
        return False


def download_fred_series(series_id: str, output_path: Path) -> tuple[pd.DataFrame, str | None]:
    """Download a FRED series."""
    from fredapi import Fred
    from config.settings import get_settings

    settings = get_settings()
    if not settings.fred_api_key:
        return pd.DataFrame(), "No FRED API key configured"

    fred = Fred(api_key=settings.fred_api_key)
    data = fred.get_series(series_id, observation_start="1960-01-01")

    if data is None or len(data) == 0:
        return pd.DataFrame(), f"No data returned for {series_id}"

    df = pd.DataFrame({
        "date": data.index,
        "value": data.values,
    })
    df["series"] = series_id

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    return df, None


def download_baumeister_shocks(output_path: Path) -> tuple[pd.DataFrame, str | None]:
    """Download Baumeister oil shocks."""
    from shared.data.baumeister import BaumeisterLoader

    loader = BaumeisterLoader()
    df = loader.fetch_with_cache()

    if df.empty:
        return pd.DataFrame(), "No data returned from Baumeister"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    return df, None


def download_exchange_rate(tier: int, output_path: Path) -> tuple[pd.DataFrame, str | None]:
    """Download exchange rate data."""
    from shared.data.exchange_rate import ExchangeRateClient

    client = ExchangeRateClient()
    df = client.fetch(tier=tier)

    if df.empty:
        return pd.DataFrame(), f"No data returned from tier {tier}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    return df, None


def download_bns_cpi(output_path: Path) -> tuple[pd.DataFrame, str | None]:
    """Download BNS CPI categories."""
    from shared.data.bns_cpi_categories import BNSCPICategoriesClient

    client = BNSCPICategoriesClient()
    df = client.fetch()

    if df.empty:
        return pd.DataFrame(), "No data returned from BNS CPI API"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    return df, None


def download_bns_income(output_path: Path) -> tuple[pd.DataFrame, str | None]:
    """Download BNS national income."""
    from shared.data.bns_national_income import BNSNationalIncomeClient

    client = BNSNationalIncomeClient()
    df = client.fetch()

    if df.empty:
        return pd.DataFrame(), "No data returned from BNS income API"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    return df, None


def download_source(source_name: str, config: dict[str, Any], project_root: Path) -> tuple[pd.DataFrame, str | None]:
    """Download a single data source."""
    output_path = project_root / config["path"]
    fetcher = config["fetcher"]

    try:
        if fetcher == "FREDClient":
            return download_fred_series(config["series"], output_path)
        elif fetcher == "BaumeisterLoader":
            return download_baumeister_shocks(output_path)
        elif fetcher == "ExchangeRateClient":
            return download_exchange_rate(config.get("tier", 1), output_path)
        elif fetcher == "BNSCPICategoriesClient":
            return download_bns_cpi(output_path)
        elif fetcher == "BNSNationalIncomeClient":
            return download_bns_income(output_path)
        else:
            return pd.DataFrame(), f"Unknown fetcher: {fetcher}"
    except Exception as e:
        return pd.DataFrame(), str(e)


def verify_existing_data(project_root: Path, tracker: DownloadStatusTracker) -> None:
    """Verify existing data files and update status."""
    console.print("\n[bold blue]Verifying existing data files...[/bold blue]")

    for source_name, config in DATA_SOURCES.items():
        file_path = project_root / config["path"]

        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                n_rows = len(df)

                # Try to get date range
                date_range = None
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    dates = df["date"].dropna()
                    if len(dates) > 0:
                        date_range = (
                            dates.min().strftime("%Y-%m-%d"),
                            dates.max().strftime("%Y-%m-%d"),
                        )

                tracker.update_source(
                    source_name,
                    status="downloaded",
                    n_rows=n_rows,
                    date_range=date_range,
                )
                logger.info(f"  {source_name}: {n_rows} rows")

            except Exception as e:
                tracker.update_source(
                    source_name,
                    status="error",
                    error=f"Failed to read: {e}",
                )
                logger.warning(f"  {source_name}: Error reading file - {e}")
        else:
            tracker.update_source(
                source_name,
                status="not_downloaded",
            )
            logger.info(f"  {source_name}: Not downloaded")


def download_all(project_root: Path, tracker: DownloadStatusTracker, force: bool = False) -> None:
    """Download all data sources."""
    console.print("\n[bold blue]Downloading data sources...[/bold blue]")

    for source_name, config in DATA_SOURCES.items():
        file_path = project_root / config["path"]

        # Skip if already downloaded and not forcing
        if file_path.exists() and not force:
            current_status = tracker.get_source_status(source_name)
            if current_status and current_status.get("status") == "downloaded":
                logger.info(f"  {source_name}: Already downloaded, skipping")
                continue

        logger.info(f"  {source_name}: Downloading...")

        df, error = download_source(source_name, config, project_root)

        if error:
            if config.get("optional"):
                logger.warning(f"  {source_name}: Failed (optional) - {error}")
                tracker.update_source(source_name, status="failed_optional", error=error)
            else:
                logger.error(f"  {source_name}: FAILED - {error}")
                tracker.update_source(source_name, status="failed", error=error)
        else:
            # Get date range
            date_range = None
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                dates = df["date"].dropna()
                if len(dates) > 0:
                    date_range = (
                        dates.min().strftime("%Y-%m-%d"),
                        dates.max().strftime("%Y-%m-%d"),
                    )

            tracker.update_source(
                source_name,
                status="downloaded",
                n_rows=len(df),
                date_range=date_range,
            )
            logger.info(f"  {source_name}: Downloaded {len(df)} rows")


def print_status_table(tracker: DownloadStatusTracker) -> None:
    """Print a summary table of download status."""
    table = Table(title="Data Download Status")
    table.add_column("Source", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Rows", justify="right")
    table.add_column("Date Range")
    table.add_column("Error", style="red")

    for source_name, config in DATA_SOURCES.items():
        status_info = tracker.get_source_status(source_name) or {}
        status = status_info.get("status", "not_attempted")
        n_rows = status_info.get("n_rows", 0)
        date_range = status_info.get("date_range")
        error = status_info.get("error")

        # Style status
        if status == "downloaded":
            status_display = "[green]downloaded[/green]"
        elif status == "synthetic":
            status_display = "[bold red]SYNTHETIC[/bold red]"
        elif status == "failed":
            status_display = "[red]failed[/red]"
        elif status == "failed_optional":
            status_display = "[yellow]failed (optional)[/yellow]"
        else:
            status_display = f"[dim]{status}[/dim]"

        date_str = f"{date_range[0]} to {date_range[1]}" if date_range else "-"
        error_str = (error[:40] + "...") if error and len(error) > 40 else (error or "-")

        table.add_row(
            source_name,
            status_display,
            str(n_rows) if n_rows else "-",
            date_str,
            error_str,
        )

    console.print(table)


def download_data(
    force: bool = False,
    verify_only: bool = False,
    source_filter: str | None = None,
) -> None:
    """Download research data sources with status tracking.

    Args:
        force: Force re-download all sources.
        verify_only: Only verify existing data, don't download.
        source_filter: Only process sources matching this name.
    """
    from config.settings import get_settings
    settings = get_settings()
    project_root = settings.project_root

    status_file = project_root / "data/metadata/download_status.json"
    tracker = DownloadStatusTracker(status_file)

    console.print("[bold green]Research Data Download Tool[/bold green]")
    console.print("=" * 50)

    if verify_only:
        verify_existing_data(project_root, tracker)
    else:
        verify_existing_data(project_root, tracker)
        if source_filter:
            # Filter DATA_SOURCES to only matching
            filtered = {k: v for k, v in DATA_SOURCES.items() if source_filter in k}
            if not filtered:
                console.print(f"[red]No sources matching '{source_filter}'[/red]")
                return
            original = dict(DATA_SOURCES)
            DATA_SOURCES.clear()
            DATA_SOURCES.update(filtered)
            download_all(project_root, tracker, force=force)
            DATA_SOURCES.clear()
            DATA_SOURCES.update(original)
        else:
            download_all(project_root, tracker, force=force)

    tracker.save()
    logger.info(f"\nStatus saved to: {status_file}")
    print_status_table(tracker)

    if tracker.has_synthetic_data():
        console.print("\n[bold red]WARNING: Synthetic data detected![/bold red]")
        console.print("Run with --force to re-download real data.")

    required_failures = []
    for source_name, config in DATA_SOURCES.items():
        if not config.get("optional"):
            status_info = tracker.get_source_status(source_name)
            if status_info and status_info.get("status") == "failed":
                required_failures.append(source_name)

    if required_failures:
        console.print(f"\n[bold red]ERROR: {len(required_failures)} required sources failed[/bold red]")
        for name in required_failures:
            console.print(f"  - {name}")
    else:
        console.print("\n[bold green]All required data sources are available![/bold green]")


def main():
    """Run data download with status tracking."""
    parser = argparse.ArgumentParser(description="Download all research data sources")
    parser.add_argument("--verify", action="store_true", help="Verify existing data only")
    parser.add_argument("--force", action="store_true", help="Force re-download all sources")
    parser.add_argument("--status", action="store_true", help="Show status only")
    parser.add_argument("--source", type=str, default=None, help="Filter by source name")
    args = parser.parse_args()

    if args.status:
        from config.settings import get_settings
        settings = get_settings()
        status_file = settings.project_root / "data/metadata/download_status.json"
        tracker = DownloadStatusTracker(status_file)
        print_status_table(tracker)
        return

    download_data(force=args.force, verify_only=args.verify, source_filter=args.source)


if __name__ == "__main__":
    main()
