"""
Loader for Baumeister-Hamilton structural oil shocks.

Christiane Baumeister's updated structural oil supply and demand shocks
(Feb 1975 – May 2025) provide usable shock series for backtesting
2014-15 oil collapse, 2020 pandemic, and 2022 energy disruptions.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import io

import pandas as pd

from config.settings import get_settings
from src.data.base import HTTPDataSource, DataSourceMetadata

logger = logging.getLogger(__name__)


@dataclass
class ShockSeries:
    """Description of a structural shock series."""

    name: str
    column: str
    description: str
    unit: str


# Expected shock series in Baumeister data
SHOCK_SERIES = {
    "oil_supply_shock": ShockSeries(
        name="oil_supply_shock",
        column="oil_supply",  # Will be mapped from actual column name
        description="Structural oil supply shock",
        unit="standard deviations",
    ),
    "aggregate_demand_shock": ShockSeries(
        name="aggregate_demand_shock",
        column="aggregate_demand",
        description="Aggregate demand shock",
        unit="standard deviations",
    ),
    "oil_specific_demand_shock": ShockSeries(
        name="oil_specific_demand_shock",
        column="oil_specific_demand",
        description="Oil-specific demand shock",
        unit="standard deviations",
    ),
    "oil_inventory_demand_shock": ShockSeries(
        name="oil_inventory_demand_shock",
        column="oil_inventory_demand",
        description="Oil inventory demand shock",
        unit="standard deviations",
    ),
}


class BaumeisterLoader(HTTPDataSource):
    """
    Loader for Baumeister-Hamilton structural oil shocks.

    Primary source: Christiane Baumeister's research page
    https://sites.google.com/site/cjsbaumeister/datasets
    Data updated through September 2025.
    """

    # Google Drive direct download URLs (current as of Jan 2026)
    # From https://sites.google.com/site/cjsbaumeister/datasets
    SUPPLY_SHOCK_URL = "https://drive.google.com/uc?export=download&id=1OsA8btgm2rmDucUFngiLkwv4uywTDmya"
    DEMAND_SHOCK_URL = "https://drive.google.com/uc?export=download&id=1neFXLrIvGwggebQRwjmtrWK-dfQZ9NH8"

    # Direct download URLs (primary)
    DIRECT_URLS = [
        SUPPLY_SHOCK_URL,
        DEMAND_SHOCK_URL,
    ]

    # Fallback URLs
    DATA_URLS = [
        "https://sites.google.com/site/cjsbaumeister/datasets",
    ]

    @property
    def source_name(self) -> str:
        return "baumeister_shocks"

    def fetch(self, **kwargs: Any) -> pd.DataFrame:
        """
        Fetch Baumeister structural oil shocks.

        Returns:
            DataFrame with date and shock columns
        """
        # Try fetching supply and demand shocks separately from Google Drive
        supply_df = None
        demand_df = None

        try:
            logger.info("Fetching oil supply shocks from Google Drive...")
            supply_df = self._fetch_from_url(self.SUPPLY_SHOCK_URL)
            if not supply_df.empty:
                logger.info(f"Fetched supply shocks: {len(supply_df)} rows")
        except Exception as e:
            logger.warning(f"Supply shock download failed: {e}")

        try:
            logger.info("Fetching oil demand shocks from Google Drive...")
            demand_df = self._fetch_from_url(self.DEMAND_SHOCK_URL)
            if not demand_df.empty:
                logger.info(f"Fetched demand shocks: {len(demand_df)} rows")
        except Exception as e:
            logger.warning(f"Demand shock download failed: {e}")

        # Merge if we got both
        if supply_df is not None and not supply_df.empty:
            # Rename shock_value to appropriate name
            supply_df = supply_df.rename(columns={'shock_value': 'oil_supply_shock'})

            if demand_df is not None and not demand_df.empty:
                demand_df = demand_df.rename(columns={'shock_value': 'aggregate_demand_shock'})
                # Merge on date
                df = pd.merge(supply_df, demand_df, on="date", how="outer")
            else:
                df = supply_df
            return df
        elif demand_df is not None and not demand_df.empty:
            demand_df = demand_df.rename(columns={'shock_value': 'aggregate_demand_shock'})
            return demand_df

        # Try discovery from data pages as fallback
        for page_url in self.DATA_URLS:
            try:
                logger.info(f"Discovering data links at: {page_url}")
                df = self._discover_and_fetch(page_url)
                if not df.empty:
                    return df
            except Exception as e:
                logger.debug(f"Discovery failed: {e}")

        # No silent fallback - fail loudly
        raise ValueError(
            "CRITICAL: Could not fetch Baumeister oil shock data. "
            "All download attempts failed. "
            "The analysis requires real structural oil shocks, not synthetic data. "
            "Options: (1) Check network connectivity, (2) Manually download from "
            "https://sites.google.com/site/cjsbaumeister/datasets and place in data/backup/baumeister/, "
            "(3) Check if Google Drive URLs have changed."
        )

    def _fetch_from_url(self, url: str) -> pd.DataFrame:
        """Fetch data from a direct URL."""
        response = self.client.get(url)
        response.raise_for_status()

        content = response.content

        # Try parsing as Excel first (Google Drive files are often Excel)
        try:
            # Skip first row which has metadata, use row 1 as header
            df = pd.read_excel(io.BytesIO(content), skiprows=1)
            # Clean up column names
            df = df.rename(columns={df.columns[0]: 'date'})
            # Keep only first two columns (date and shock values)
            df = df.iloc[:, :2]
            df.columns = ['date', 'shock_value']
            # Convert date column
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            return df
        except Exception as e:
            logger.debug(f"Excel parsing failed: {e}")

        # Try parsing as CSV
        try:
            df = pd.read_csv(io.BytesIO(content))
            return self._standardize_columns(df)
        except Exception as e:
            logger.debug(f"CSV parsing failed: {e}")

        raise ValueError("Could not parse response as Excel or CSV")

    def _discover_and_fetch(self, page_url: str) -> pd.DataFrame:
        """Discover data links on a page and fetch."""
        from bs4 import BeautifulSoup

        response = self.client.get(page_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Look for links to data files
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            text = anchor.get_text(strip=True).lower()

            # Look for oil shock data
            if any(
                kw in text for kw in ["shock", "oil", "structural", "monthly"]
            ) and any(href.endswith(ext) for ext in [".xlsx", ".xls", ".csv", ".zip"]):

                full_url = (
                    href if href.startswith("http") else f"{page_url.rsplit('/', 1)[0]}/{href}"
                )
                try:
                    return self._fetch_from_url(full_url)
                except Exception as e:
                    logger.debug(f"Failed to fetch {full_url}: {e}")

        return pd.DataFrame()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names from various Baumeister data formats."""
        # Common column name patterns
        col_mapping = {
            # Date columns
            "date": "date",
            "month": "date",
            "time": "date",
            "period": "date",
            # Shock columns (various naming conventions)
            "oil_supply": "oil_supply_shock",
            "supply_shock": "oil_supply_shock",
            "oil supply shock": "oil_supply_shock",
            "aggregate_demand": "aggregate_demand_shock",
            "demand_shock": "aggregate_demand_shock",
            "global demand shock": "aggregate_demand_shock",
            "oil_specific": "oil_specific_demand_shock",
            "oil-specific demand": "oil_specific_demand_shock",
            "speculative_demand": "oil_inventory_demand_shock",
            "inventory_demand": "oil_inventory_demand_shock",
        }

        # Rename columns
        df = df.copy()
        for old, new in col_mapping.items():
            for col in df.columns:
                if old.lower() in col.lower():
                    df = df.rename(columns={col: new})
                    break

        # Ensure date column exists and is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            # Try to construct date from year/month columns
            if "year" in df.columns and "month" in df.columns:
                df["date"] = pd.to_datetime(
                    df["year"].astype(str) + "-" + df["month"].astype(str) + "-01"
                )

        return df

    def _generate_placeholder_data_for_tests_only(self) -> pd.DataFrame:
        """
        Generate placeholder data FOR UNIT TESTS ONLY.

        WARNING: This method should NEVER be called in production code.
        It generates deterministic random data (seed=42) that has no
        relationship to real oil market shocks. Using this data for
        actual analysis would produce meaningless results.

        This method exists solely to allow unit tests to run without
        network access.
        """
        import numpy as np

        # Generate monthly dates from 1975-02 to 2025-05
        dates = pd.date_range(start="1975-02-01", end="2025-05-01", freq="MS")

        # Generate random walk shocks (for placeholder only)
        np.random.seed(42)
        n = len(dates)

        df = pd.DataFrame(
            {
                "date": dates,
                "oil_supply_shock": np.random.randn(n) * 0.5,
                "aggregate_demand_shock": np.random.randn(n) * 0.5,
                "oil_specific_demand_shock": np.random.randn(n) * 0.5,
                "oil_inventory_demand_shock": np.random.randn(n) * 0.5,
            }
        )

        logger.error(
            "USING SYNTHETIC SHOCK DATA - THIS IS FOR TESTS ONLY. "
            "Results from this data are MEANINGLESS."
        )
        return df

    def resample_to_quarterly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample monthly shocks to quarterly."""
        if df.empty:
            return df

        df = df.set_index("date")

        # Aggregate shocks by quarter (mean)
        shock_cols = [c for c in df.columns if "shock" in c.lower()]
        quarterly = df[shock_cols].resample("QE").mean()

        result = quarterly.reset_index()
        # Convert datetime to quarter string format (e.g., "2010Q1")
        result["quarter"] = (
            result["date"].dt.year.astype(str)
            + "Q"
            + result["date"].dt.quarter.astype(str)
        )
        result = result.drop(columns=["date"])

        return result

    def get_shocks_for_panel(
        self,
        start_date: str = "2010-01-01",
        end_date: str | None = None,
        frequency: str = "Q",
    ) -> pd.DataFrame:
        """
        Get structural shocks ready for panel merge.

        Args:
            start_date: Start date
            end_date: End date (default: today)
            frequency: Q for quarterly, M for monthly

        Returns:
            DataFrame with quarter/month and shock columns
        """
        df = self.fetch_with_cache()

        if df.empty:
            return df

        # Filter date range
        df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        # Resample if needed
        if frequency == "Q":
            df = self.resample_to_quarterly(df)
        else:
            df = df.rename(columns={"date": "month"})

        return df

    def save_raw(self, df: pd.DataFrame | None = None, filename: str = "shocks.parquet") -> Path:
        """Save raw shock data."""
        if df is None:
            df = self.fetch()
        return super().save_raw(df, filename)
