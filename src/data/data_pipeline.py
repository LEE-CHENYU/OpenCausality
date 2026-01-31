"""
Data pipeline orchestration.

Coordinates fetching, processing, and panel construction from all data sources.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import get_settings
from src.data.kazakhstan_bns import KazakhstanBNSClient, BNSDataType
from src.data.fred_client import FREDClient
from src.data.baumeister_loader import BaumeisterLoader

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report on data quality issues."""

    source: str
    total_rows: int
    missing_values: dict[str, int]
    date_range: tuple[str, str] | None
    structural_breaks: list[str]
    warnings: list[str]
    timestamp: datetime


class DataPipeline:
    """Orchestrates data collection and panel construction."""

    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self.bns_client = KazakhstanBNSClient()
        self.fred_client = FREDClient()
        self.baumeister_loader = BaumeisterLoader()
        self._quality_reports: list[DataQualityReport] = []

    def fetch_all_raw(self) -> dict[str, Any]:
        """Fetch all raw data from all sources."""
        results = {}

        # Fetch BNS data
        logger.info("Fetching Kazakhstan BNS data...")
        try:
            results["bns"] = self.bns_client.fetch_all()
        except Exception as e:
            logger.error(f"Failed to fetch BNS data: {e}")
            results["bns"] = {}

        # Fetch FRED data
        logger.info("Fetching FRED data...")
        try:
            results["fred"] = self.fred_client.fetch_all_quarterly()
        except Exception as e:
            logger.error(f"Failed to fetch FRED data: {e}")
            results["fred"] = {}

        # Fetch Baumeister shocks
        logger.info("Fetching Baumeister oil shocks...")
        try:
            results["baumeister"] = self.baumeister_loader.get_shocks_for_panel()
        except Exception as e:
            logger.error(f"Failed to fetch Baumeister data: {e}")
            results["baumeister"] = pd.DataFrame()

        return results

    def save_all_raw(self) -> dict[str, list[Path]]:
        """Fetch and save all raw data."""
        paths: dict[str, list[Path]] = {"bns": [], "fred": [], "baumeister": []}

        # Save BNS data
        try:
            bns_paths = self.bns_client.save_all_raw()
            paths["bns"] = list(bns_paths.values())
        except Exception as e:
            logger.error(f"Failed to save BNS data: {e}")

        # Save FRED data
        try:
            fred_paths = self.fred_client.save_all_raw()
            paths["fred"] = list(fred_paths.values())
        except Exception as e:
            logger.error(f"Failed to save FRED data: {e}")

        # Save Baumeister data
        try:
            path = self.baumeister_loader.save_raw()
            paths["baumeister"] = [path]
        except Exception as e:
            logger.error(f"Failed to save Baumeister data: {e}")

        return paths

    def build_panel(
        self,
        start_year: int = 2010,
        end_year: int = 2024,
    ) -> pd.DataFrame:
        """
        Build the main analysis panel.

        Args:
            start_year: Panel start year
            end_year: Panel end year

        Returns:
            Panel DataFrame with region-quarter observations
        """
        from src.model.panel_data import PanelBuilder

        # Fetch all data
        raw_data = self.fetch_all_raw()

        # Build panel
        builder = PanelBuilder()
        panel = builder.build(
            bns_data=raw_data.get("bns", {}),
            fred_data=raw_data.get("fred", {}),
            baumeister_data=raw_data.get("baumeister", pd.DataFrame()),
            start_year=start_year,
            end_year=end_year,
        )

        # Generate quality report
        self._generate_quality_report(panel, "panel")

        return panel

    def save_panel(
        self,
        panel: pd.DataFrame | None = None,
        filename: str = "panel.parquet",
    ) -> Path:
        """Save the analysis panel."""
        if panel is None:
            panel = self.build_panel()

        settings = get_settings()
        output_dir = settings.project_root / settings.processed_data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename
        panel.to_parquet(filepath, index=False)

        logger.info(f"Saved panel to {filepath}")
        return filepath

    def _generate_quality_report(
        self, df: pd.DataFrame, source: str
    ) -> DataQualityReport:
        """Generate data quality report for a DataFrame."""
        warnings = []

        # Check for missing values
        missing = df.isnull().sum().to_dict()
        missing = {k: v for k, v in missing.items() if v > 0}

        # Check date range
        date_cols = [c for c in df.columns if "date" in c.lower() or "quarter" in c.lower()]
        date_range = None
        if date_cols:
            date_col = date_cols[0]
            date_range = (
                str(df[date_col].min()),
                str(df[date_col].max()),
            )

        # Check for potential structural breaks
        structural_breaks = []
        reform_years = [2018, 2022]
        for year in reform_years:
            if "year" in df.columns and year in df["year"].values:
                structural_breaks.append(f"Region reform in {year}")

        # Generate warnings
        if missing:
            warnings.append(f"Missing values in columns: {list(missing.keys())}")

        pct_missing = df.isnull().mean().mean() * 100
        if pct_missing > 5:
            warnings.append(f"High overall missing rate: {pct_missing:.1f}%")

        report = DataQualityReport(
            source=source,
            total_rows=len(df),
            missing_values=missing,
            date_range=date_range,
            structural_breaks=structural_breaks,
            warnings=warnings,
            timestamp=datetime.now(),
        )

        self._quality_reports.append(report)
        return report

    def get_quality_reports(self) -> list[DataQualityReport]:
        """Get all data quality reports."""
        return self._quality_reports

    def print_quality_summary(self) -> None:
        """Print summary of data quality reports."""
        if not self._quality_reports:
            print("No quality reports generated yet.")
            return

        for report in self._quality_reports:
            print(f"\n{'='*60}")
            print(f"Data Quality Report: {report.source}")
            print(f"{'='*60}")
            print(f"Total rows: {report.total_rows:,}")
            print(f"Date range: {report.date_range}")
            print(f"Timestamp: {report.timestamp}")

            if report.missing_values:
                print("\nMissing values:")
                for col, count in report.missing_values.items():
                    pct = count / report.total_rows * 100
                    print(f"  - {col}: {count:,} ({pct:.1f}%)")

            if report.structural_breaks:
                print("\nStructural breaks:")
                for break_info in report.structural_breaks:
                    print(f"  - {break_info}")

            if report.warnings:
                print("\nWarnings:")
                for warning in report.warnings:
                    print(f"  ⚠ {warning}")


def run_pipeline(save: bool = True) -> pd.DataFrame:
    """Run the complete data pipeline."""
    pipeline = DataPipeline()

    # Build panel
    logger.info("Building analysis panel...")
    panel = pipeline.build_panel()

    # Print quality summary
    pipeline.print_quality_summary()

    # Save if requested
    if save:
        pipeline.save_panel(panel)

    return panel
