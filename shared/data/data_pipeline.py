"""
Data pipeline orchestration for shared data infrastructure.

Coordinates fetching, processing, and caching from all data sources.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import get_settings
from shared.data.kazakhstan_bns import KazakhstanBNSClient, BNSDataType
from shared.data.fred_client import FREDClient

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


class SharedDataPipeline:
    """
    Orchestrates shared data collection.

    This is the shared infrastructure used by all studies.
    Study-specific pipelines should extend this with their own data sources.
    """

    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self.bns_client = KazakhstanBNSClient()
        self.fred_client = FREDClient()
        self._quality_reports: list[DataQualityReport] = []

    def fetch_bns_data(self) -> dict[BNSDataType, pd.DataFrame]:
        """Fetch all BNS data."""
        logger.info("Fetching Kazakhstan BNS data...")
        try:
            return self.bns_client.fetch_all()
        except Exception as e:
            logger.error(f"Failed to fetch BNS data: {e}")
            return {}

    def fetch_fred_data(self, start_date: str = "2000-01-01") -> dict[str, pd.DataFrame]:
        """Fetch all FRED data."""
        logger.info("Fetching FRED data...")
        try:
            return self.fred_client.fetch_all_quarterly(start_date=start_date)
        except Exception as e:
            logger.error(f"Failed to fetch FRED data: {e}")
            return {}

    def save_bns_raw(self) -> dict[BNSDataType, Path]:
        """Fetch and save all BNS data."""
        try:
            return self.bns_client.save_all_raw()
        except Exception as e:
            logger.error(f"Failed to save BNS data: {e}")
            return {}

    def save_fred_raw(self, start_date: str = "2000-01-01") -> dict[str, Path]:
        """Fetch and save all FRED data."""
        try:
            return self.fred_client.save_all_raw(start_date=start_date)
        except Exception as e:
            logger.error(f"Failed to save FRED data: {e}")
            return {}

    def generate_quality_report(
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
                    print(f"  ! {warning}")
