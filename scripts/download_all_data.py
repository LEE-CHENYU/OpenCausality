#!/usr/bin/env python3
"""
Comprehensive data download script for archiving all available API data.

This script downloads as much data as possible from all available sources:
- FRED: Extensive set of economic indicators
- Baumeister: Oil structural shocks
- BNS Kazakhstan: Regional statistics

Run with: PYTHONPATH=. python scripts/download_all_data.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler

# Setup logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_path=False)],
)
logger = logging.getLogger(__name__)

# Backup directory
BACKUP_DIR = Path("/Users/lichenyu/econometric-research/data/backup")


class DataManifest:
    """Track downloaded data with metadata."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict[str, Any]] = []

    def add(
        self,
        source: str,
        series: str,
        filepath: Path,
        rows: int,
        date_range: tuple[str, str] | None = None,
        notes: str = "",
    ):
        self.entries.append({
            "source": source,
            "series": series,
            "filepath": str(filepath.relative_to(BACKUP_DIR)),
            "rows": rows,
            "date_range": date_range,
            "notes": notes,
            "downloaded_at": datetime.now().isoformat(),
        })

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "manifest_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_datasets": len(self.entries),
                "datasets": self.entries,
            }, f, indent=2)
        logger.info(f"Saved manifest with {len(self.entries)} entries to {self.path}")


def download_fred_data(manifest: DataManifest) -> None:
    """Download comprehensive FRED data."""
    from fredapi import Fred

    fred_dir = BACKUP_DIR / "fred"
    fred_dir.mkdir(parents=True, exist_ok=True)

    # Get API key
    from config.settings import get_settings
    settings = get_settings()

    if not settings.fred_api_key:
        logger.error("No FRED API key configured")
        return

    fred = Fred(api_key=settings.fred_api_key)

    # Comprehensive list of FRED series to download
    series_list = {
        # Oil & Energy
        "DCOILBRENTEU": "Brent Crude Oil Price",
        "DCOILWTICO": "WTI Crude Oil Price",
        "GASREGW": "US Regular Gasoline Prices",
        "DHHNGSP": "Henry Hub Natural Gas Price",
        "PNRGINDEXM": "Global Price of Energy Index",

        # Global Activity & Trade
        "IGREA": "Kilian Global Real Economic Activity Index",
        "GSCPI": "Global Supply Chain Pressure Index",
        "WTISPLC": "Spot Crude Oil Price: WTI",

        # Financial Conditions
        "VIXCLS": "CBOE Volatility Index (VIX)",
        "STLFSI4": "St. Louis Fed Financial Stress Index",
        "NFCI": "Chicago Fed Financial Conditions Index",
        "TEDRATE": "TED Spread",
        "BAMLH0A0HYM2": "ICE BofA US High Yield Index",

        # Exchange Rates
        "DEXUSEU": "US/Euro Exchange Rate",
        "DEXUSUK": "US/UK Exchange Rate",
        "DEXJPUS": "Japan/US Exchange Rate",
        "DEXCHUS": "China/US Exchange Rate",
        "DEXKZUS": "Kazakhstan Tenge/US Dollar (if available)",
        "DTWEXBGS": "Trade Weighted US Dollar Index: Broad",

        # Interest Rates
        "FEDFUNDS": "Federal Funds Rate",
        "DGS10": "10-Year Treasury Constant Maturity",
        "DGS2": "2-Year Treasury Constant Maturity",
        "T10Y2Y": "10-Year Minus 2-Year Treasury Spread",
        "DFEDTARU": "Federal Funds Target Range Upper",

        # Inflation
        "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
        "CPILFESL": "Core CPI (Less Food and Energy)",
        "PCEPI": "Personal Consumption Expenditures Price Index",
        "PCEPILFE": "Core PCE Price Index",
        "MICH": "University of Michigan Inflation Expectation",
        "T5YIE": "5-Year Breakeven Inflation Rate",

        # Labor Market
        "UNRATE": "Unemployment Rate",
        "PAYEMS": "Total Nonfarm Payrolls",
        "ICSA": "Initial Jobless Claims",
        "JTSJOL": "Job Openings: Total Nonfarm",

        # GDP & Output
        "GDPC1": "Real GDP",
        "A191RL1Q225SBEA": "Real GDP Growth Rate",
        "INDPRO": "Industrial Production Index",
        "CAPACITY": "Capacity Utilization",

        # Monetary Aggregates
        "M2SL": "M2 Money Stock",
        "BOGMBASE": "Monetary Base",
        "WALCL": "Fed Total Assets",

        # Commodity Prices
        "PPIACO": "Producer Price Index: All Commodities",
        "PCOPPUSDM": "Global Copper Price",
        "GOLDAMGBD228NLBM": "Gold Fixing Price",
        "PALLFNFINDEXM": "All Commodities Index",
        "PFOODINDEXM": "Food Price Index",
        "PMETAINDEXM": "Metals Price Index",

        # Global Economy
        "GEPUCURRENT": "Global Economic Policy Uncertainty Index",
        "USEPUINDXD": "US Economic Policy Uncertainty Index",

        # Consumer & Business Sentiment
        "UMCSENT": "University of Michigan Consumer Sentiment",
        "CSCICP03USM665S": "Consumer Confidence Index",
        "BSCICP03USM665S": "Business Confidence Index",

        # Housing
        "CSUSHPISA": "Case-Shiller Home Price Index",
        "HOUST": "Housing Starts",
        "PERMIT": "Building Permits",

        # Credit Spreads
        "BAMLC0A0CM": "ICE BofA US Corporate Index",
        "BAMLC0A4CBBB": "ICE BofA BBB US Corporate Index",
        "BAMLH0A0HYM2EY": "ICE BofA High Yield Effective Yield",
    }

    console.print(f"\n[bold blue]Downloading {len(series_list)} FRED series...[/bold blue]")

    for series_id, description in series_list.items():
        try:
            data = fred.get_series(series_id, observation_start="1960-01-01")

            if data is not None and len(data) > 0:
                df = pd.DataFrame({
                    "date": data.index,
                    "value": data.values,
                })
                df["series"] = series_id

                filepath = fred_dir / f"{series_id}.parquet"
                df.to_parquet(filepath, index=False)

                date_range = (
                    df["date"].min().strftime("%Y-%m-%d"),
                    df["date"].max().strftime("%Y-%m-%d"),
                )

                manifest.add(
                    source="FRED",
                    series=series_id,
                    filepath=filepath,
                    rows=len(df),
                    date_range=date_range,
                    notes=description,
                )

                logger.info(f"  {series_id}: {len(df)} obs ({date_range[0]} to {date_range[1]})")
            else:
                logger.warning(f"  {series_id}: No data returned")

        except Exception as e:
            logger.warning(f"  {series_id}: Failed - {e}")


def download_baumeister_data(manifest: DataManifest) -> None:
    """Download Baumeister structural oil shocks."""
    import httpx
    import io

    baumeister_dir = BACKUP_DIR / "baumeister"
    baumeister_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold blue]Downloading Baumeister oil shocks...[/bold blue]")

    # Known Google Drive URLs
    datasets = {
        "oil_supply_shocks": "https://drive.google.com/uc?export=download&id=1OsA8btgm2rmDucUFngiLkwv4uywTDmya",
        "oil_demand_shocks": "https://drive.google.com/uc?export=download&id=1neFXLrIvGwggebQRwjmtrWK-dfQZ9NH8",
    }

    client = httpx.Client(follow_redirects=True, timeout=60)

    for name, url in datasets.items():
        try:
            response = client.get(url)
            response.raise_for_status()

            # Parse Excel (skip metadata row)
            df = pd.read_excel(io.BytesIO(response.content), skiprows=1)

            # Standardize columns
            df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "shock_value"})
            df = df.iloc[:, :2]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])

            filepath = baumeister_dir / f"{name}.parquet"
            df.to_parquet(filepath, index=False)

            date_range = (
                df["date"].min().strftime("%Y-%m-%d"),
                df["date"].max().strftime("%Y-%m-%d"),
            )

            manifest.add(
                source="Baumeister",
                series=name,
                filepath=filepath,
                rows=len(df),
                date_range=date_range,
                notes="Baumeister-Hamilton (2019) AER structural shocks",
            )

            logger.info(f"  {name}: {len(df)} obs ({date_range[0]} to {date_range[1]})")

        except Exception as e:
            logger.error(f"  {name}: Failed - {e}")


def download_bns_data(manifest: DataManifest) -> None:
    """Download Kazakhstan BNS data by exploring iblock IDs."""
    import httpx
    import io

    bns_dir = BACKUP_DIR / "bns"
    bns_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold blue]Downloading Kazakhstan BNS data...[/bold blue]")

    # Known working iblock IDs and their descriptions
    known_ids = {
        48953: "Per capita monetary income (quarterly)",
        48510: "Per capita monetary income (annual)",
        469805: "Household expenditure structure",
    }

    # Also try IDs in ranges around known ones
    id_ranges = [
        range(48500, 48560),  # Around income data
        range(48900, 48960),
        range(469800, 469810),
    ]

    # Collect all IDs to try
    all_ids = set(known_ids.keys())
    for r in id_ranges:
        all_ids.update(r)

    client = httpx.Client(follow_redirects=True, timeout=30)
    base_url = "https://stat.gov.kz"

    successful = 0
    for iblock_id in sorted(all_ids):
        try:
            # Try CSV endpoint first
            url = f"{base_url}/api/iblock/element/{iblock_id}/csv/file/en/"
            response = client.get(url)

            if response.status_code == 200 and len(response.content) > 100:
                # Parse CSV
                df = pd.read_csv(io.BytesIO(response.content), sep="\t")

                if len(df) > 0 and len(df.columns) > 1:
                    # Get description from NAM column if available
                    desc = known_ids.get(iblock_id, "")
                    if "NAM" in df.columns and len(df) > 0:
                        desc = str(df["NAM"].iloc[0])[:100]

                    filepath = bns_dir / f"iblock_{iblock_id}.parquet"
                    df.to_parquet(filepath, index=False)

                    manifest.add(
                        source="BNS_Kazakhstan",
                        series=f"iblock_{iblock_id}",
                        filepath=filepath,
                        rows=len(df),
                        notes=desc,
                    )

                    successful += 1
                    if iblock_id in known_ids:
                        logger.info(f"  iblock/{iblock_id}: {len(df)} rows - {known_ids[iblock_id]}")
                    else:
                        logger.info(f"  iblock/{iblock_id}: {len(df)} rows (discovered)")

        except Exception as e:
            pass  # Silently skip failed IDs

    logger.info(f"  Total BNS datasets: {successful}")


def download_world_bank_data(manifest: DataManifest) -> None:
    """Download World Bank indicators for Kazakhstan."""
    console.print("\n[bold blue]Downloading World Bank data...[/bold blue]")

    try:
        import wbgapi as wb

        wb_dir = BACKUP_DIR / "worldbank"
        wb_dir.mkdir(parents=True, exist_ok=True)

        # Key indicators for Kazakhstan
        indicators = {
            "NY.GDP.MKTP.CD": "GDP (current US$)",
            "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
            "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
            "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
            "SL.UEM.TOTL.ZS": "Unemployment, total (% of labor force)",
            "NE.EXP.GNFS.ZS": "Exports of goods and services (% of GDP)",
            "NE.IMP.GNFS.ZS": "Imports of goods and services (% of GDP)",
            "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment, net inflows (% of GDP)",
            "PA.NUS.FCRF": "Official exchange rate (LCU per US$)",
            "GC.REV.XGRT.GD.ZS": "Revenue, excluding grants (% of GDP)",
            "EG.ELC.ACCS.ZS": "Access to electricity (% of population)",
            "SI.POV.GINI": "Gini index",
            "SI.POV.NAHC": "Poverty headcount ratio at national poverty lines",
        }

        for ind_id, description in indicators.items():
            try:
                df = wb.data.DataFrame(ind_id, "KAZ", time=range(1990, 2025))
                if df is not None and len(df) > 0:
                    df = df.reset_index()
                    filepath = wb_dir / f"{ind_id.replace('.', '_')}.parquet"
                    df.to_parquet(filepath, index=False)

                    manifest.add(
                        source="WorldBank",
                        series=ind_id,
                        filepath=filepath,
                        rows=len(df),
                        notes=f"Kazakhstan: {description}",
                    )

                    logger.info(f"  {ind_id}: {len(df)} obs")

            except Exception as e:
                logger.warning(f"  {ind_id}: Failed - {e}")

    except ImportError:
        logger.warning("  wbgapi not installed, skipping World Bank data")


def main():
    """Run comprehensive data download."""
    console.print("[bold green]Comprehensive Data Archive Download[/bold green]")
    console.print("=" * 50)

    # Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize manifest
    manifest = DataManifest(BACKUP_DIR / "manifest.json")

    # Download from all sources
    download_fred_data(manifest)
    download_baumeister_data(manifest)
    download_bns_data(manifest)
    download_world_bank_data(manifest)

    # Save manifest
    manifest.save()

    # Summary
    console.print("\n[bold green]Download Complete![/bold green]")
    console.print(f"Total datasets archived: {len(manifest.entries)}")
    console.print(f"Location: {BACKUP_DIR}")


if __name__ == "__main__":
    main()
