"""
BNS CPI Categories by COICOP division.

Fetches monthly Consumer Price Index data broken down by COICOP categories
from Kazakhstan Bureau of National Statistics.

COICOP Categories:
01 - Food and non-alcoholic beverages
02 - Alcoholic beverages, tobacco
03 - Clothing and footwear
04 - Housing, water, electricity, gas
05 - Furnishings, household equipment
06 - Health
07 - Transport
08 - Communications
09 - Recreation and culture
10 - Education
11 - Restaurants and hotels
12 - Miscellaneous goods and services
"""

import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from config.settings import get_settings
from shared.data.base import HTTPDataSource

logger = logging.getLogger(__name__)


# COICOP category metadata
COICOP_CATEGORIES = {
    "01": {
        "name": "Food and non-alcoholic beverages",
        "name_short": "Food",
        "tradable": True,
        "admin_price": False,
    },
    "02": {
        "name": "Alcoholic beverages, tobacco",
        "name_short": "Alcohol/Tobacco",
        "tradable": True,
        "admin_price": False,
    },
    "03": {
        "name": "Clothing and footwear",
        "name_short": "Clothing",
        "tradable": True,
        "admin_price": False,
    },
    "04": {
        "name": "Housing, water, electricity, gas",
        "name_short": "Housing/Utilities",
        "tradable": False,
        "admin_price": True,
    },
    "05": {
        "name": "Furnishings, household equipment",
        "name_short": "Furnishings",
        "tradable": True,
        "admin_price": False,
    },
    "06": {
        "name": "Health",
        "name_short": "Health",
        "tradable": False,
        "admin_price": True,
    },
    "07": {
        "name": "Transport",
        "name_short": "Transport",
        "tradable": True,  # Vehicles, fuel
        "admin_price": False,  # Some transport (07.3) is admin
    },
    "08": {
        "name": "Communications",
        "name_short": "Communications",
        "tradable": False,
        "admin_price": True,
    },
    "09": {
        "name": "Recreation and culture",
        "name_short": "Recreation",
        "tradable": True,
        "admin_price": False,
    },
    "10": {
        "name": "Education",
        "name_short": "Education",
        "tradable": False,
        "admin_price": True,
    },
    "11": {
        "name": "Restaurants and hotels",
        "name_short": "Restaurants/Hotels",
        "tradable": False,
        "admin_price": False,
    },
    "12": {
        "name": "Miscellaneous goods and services",
        "name_short": "Miscellaneous",
        "tradable": True,
        "admin_price": False,
    },
}


@dataclass
class CPICategoryMetadata:
    """Metadata for CPI category data."""

    n_categories: int = 12
    n_months: int = 0
    start_date: str | None = None
    end_date: str | None = None
    admin_categories: list[str] = field(default_factory=lambda: ["04", "06", "08", "10"])


class BNSCPICategoriesClient(HTTPDataSource):
    """
    Client for BNS CPI data by COICOP category.

    Fetches monthly CPI indices disaggregated by category for
    Block A pass-through analysis.
    """

    @property
    def source_name(self) -> str:
        return "bns_cpi_categories"

    def __init__(self, cache_dir: Path | None = None):
        super().__init__(cache_dir)
        settings = get_settings()
        self.base_url = settings.bns_base_url
        self._metadata: CPICategoryMetadata | None = None

    def fetch(self, **kwargs: Any) -> pd.DataFrame:
        """
        Fetch CPI data by COICOP category.

        Returns:
            DataFrame with columns:
                - date: Month (datetime)
                - category: COICOP code (01-12)
                - category_name: Category name
                - cpi_index: CPI index level
                - inflation_mom: Month-over-month inflation
                - inflation_yoy: Year-over-year inflation
                - tradable: Whether category is tradable
                - admin_price: Whether category has admin prices
        """
        # Try BNS API first
        try:
            df = self._fetch_bns_cpi_api()
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"BNS CPI API failed: {e}")

        # Fallback: try BNS statistics page
        try:
            df = self._fetch_bns_cpi_page()
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"BNS CPI page scraping failed: {e}")

        # Final fallback: load from local cache if available
        settings = get_settings()
        local_path = settings.project_root / "data/raw/kazakhstan_bns/cpi_categories.parquet"
        if local_path.exists():
            logger.info(f"Loading cached CPI data from {local_path}")
            return pd.read_parquet(local_path)

        raise ValueError(
            "CPI category data unavailable.\n"
            "BNS API failed and no local cache found.\n"
            "Please download CPI by category data manually from:\n"
            "https://stat.gov.kz/en/industries/prices/stat-official-ind-prices/"
        )

    def _fetch_bns_cpi_api(self) -> pd.DataFrame:
        """Fetch CPI data via BNS API."""
        # BNS iblock ID for CPI by category (may need updating)
        iblock_id = 49140  # Consumer Price Index statistics

        csv_url = f"{self.base_url}/api/iblock/element/{iblock_id}/csv/file/en/"
        logger.info(f"Fetching BNS CPI categories from iblock {iblock_id}")

        response = self.client.get(csv_url)
        response.raise_for_status()

        df = pd.read_csv(io.BytesIO(response.content), sep="\t")

        return self._standardize_cpi_data(df)

    def _fetch_bns_cpi_page(self) -> pd.DataFrame:
        """Fetch CPI data by scraping BNS page."""
        page_url = f"{self.base_url}/en/industries/prices/stat-official-ind-prices/"

        response = self.client.get(page_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Find links to CPI data files
        data_links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            text = anchor.get_text(strip=True).lower()

            if "cpi" in text or "consumer price" in text or "iblock" in href:
                if "/api/iblock/" in href or href.endswith((".xlsx", ".csv")):
                    full_url = href if href.startswith("http") else f"{self.base_url}{href}"
                    data_links.append((full_url, text))

        for url, text in data_links:
            try:
                logger.info(f"Trying CPI link: {text}")
                response = self.client.get(url)
                response.raise_for_status()

                # Try to parse as Excel or CSV
                content = response.content
                try:
                    df = pd.read_excel(io.BytesIO(content))
                except Exception:
                    df = pd.read_csv(io.BytesIO(content), sep="\t")

                if not df.empty:
                    return self._standardize_cpi_data(df)
            except Exception as e:
                logger.debug(f"Failed to fetch {url}: {e}")
                continue

        raise RuntimeError("No valid CPI data links found")

    def _standardize_cpi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize CPI data to common format.

        Expected output columns:
            date, category, category_name, cpi_index, inflation_mom, inflation_yoy,
            tradable, admin_price
        """
        import numpy as np

        # Identify period and category columns
        period_col = None
        value_col = None
        category_col = None

        for col in df.columns:
            col_lower = col.lower()
            if "period" in col_lower or "month" in col_lower or "dat" in col_lower:
                period_col = col
            elif "val" in col_lower or "index" in col_lower or "cpi" in col_lower:
                value_col = col
            elif "category" in col_lower or "group" in col_lower or "coicop" in col_lower:
                category_col = col

        if period_col is None:
            # Try to identify by examining data patterns
            for col in df.columns:
                sample = df[col].dropna().astype(str).iloc[:5] if len(df) > 0 else []
                if any("2020" in s or "2021" in s or "2022" in s for s in sample):
                    period_col = col
                    break

        if value_col is None:
            # Look for numeric columns that look like indices
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    sample = df[col].dropna()
                    if len(sample) > 0 and 50 < sample.mean() < 200:  # CPI range
                        value_col = col
                        break

        # Build standardized dataframe
        records = []

        if period_col and value_col:
            # Parse period to date
            df["_period"] = df[period_col].astype(str)

            # Extract year and month
            for _, row in df.iterrows():
                period = str(row[period_col])
                value = row[value_col]

                # Parse period (formats: 202001, 2020-01, Jan 2020, etc.)
                date = self._parse_period(period)
                if date is None:
                    continue

                # Get category if available
                cat_code = "00"  # Default: total
                if category_col and pd.notna(row.get(category_col)):
                    cat_code = self._parse_category(str(row[category_col]))

                records.append({
                    "date": date,
                    "category": cat_code,
                    "cpi_index": float(value) if pd.notna(value) else np.nan,
                })

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records)

        # Add category metadata
        result["category_name"] = result["category"].map(
            lambda c: COICOP_CATEGORIES.get(c, {}).get("name", "Unknown")
        )
        result["tradable"] = result["category"].map(
            lambda c: COICOP_CATEGORIES.get(c, {}).get("tradable", True)
        )
        result["admin_price"] = result["category"].map(
            lambda c: COICOP_CATEGORIES.get(c, {}).get("admin_price", False)
        )

        # Sort and compute inflation rates
        result = result.sort_values(["category", "date"]).reset_index(drop=True)

        # Month-over-month inflation by category
        result["inflation_mom"] = result.groupby("category")["cpi_index"].pct_change()

        # Year-over-year inflation
        result["inflation_yoy"] = result.groupby("category")["cpi_index"].pct_change(12)

        # Update metadata
        self._metadata = CPICategoryMetadata(
            n_categories=result["category"].nunique(),
            n_months=result["date"].nunique(),
            start_date=result["date"].min().strftime("%Y-%m-%d"),
            end_date=result["date"].max().strftime("%Y-%m-%d"),
        )

        return result

    def _parse_period(self, period: str) -> datetime | None:
        """Parse period string to datetime."""
        import re

        period = str(period).strip()

        # YYYYMM format
        if re.match(r"^\d{6}$", period):
            try:
                return datetime.strptime(period, "%Y%m")
            except ValueError:
                pass

        # YYYY-MM format
        if re.match(r"^\d{4}-\d{2}$", period):
            try:
                return datetime.strptime(period, "%Y-%m")
            except ValueError:
                pass

        # YYYY/MM format
        if re.match(r"^\d{4}/\d{2}$", period):
            try:
                return datetime.strptime(period, "%Y/%m")
            except ValueError:
                pass

        # Month Year format (Jan 2020)
        try:
            return datetime.strptime(period, "%b %Y")
        except ValueError:
            pass

        return None

    def _parse_category(self, cat_str: str) -> str:
        """Parse category string to COICOP code."""
        import re

        cat_str = str(cat_str).strip()

        # Direct code match (01, 02, etc.)
        if re.match(r"^\d{2}$", cat_str):
            return cat_str

        # Code with description (01 - Food...)
        match = re.match(r"^(\d{2})\s*[-:]", cat_str)
        if match:
            return match.group(1)

        # Name-based matching
        cat_lower = cat_str.lower()
        for code, meta in COICOP_CATEGORIES.items():
            if meta["name"].lower() in cat_lower or cat_lower in meta["name"].lower():
                return code

        return "00"  # Unknown

    def fetch_headline_cpi(self) -> pd.DataFrame:
        """Fetch headline (total) CPI only."""
        df = self.fetch_with_cache()
        return df[df["category"] == "00"]

    def fetch_by_category(self, category_code: str) -> pd.DataFrame:
        """Fetch CPI for a specific category."""
        df = self.fetch_with_cache()
        return df[df["category"] == category_code]

    def fetch_tradables(self) -> pd.DataFrame:
        """Fetch CPI for tradable categories only."""
        df = self.fetch_with_cache()
        return df[df["tradable"] == True]  # noqa: E712

    def fetch_non_admin(self) -> pd.DataFrame:
        """Fetch CPI for non-administered price categories."""
        df = self.fetch_with_cache()
        return df[df["admin_price"] == False]  # noqa: E712

    def build_cpi_panel(self) -> pd.DataFrame:
        """
        Build panel data for Block A CPI pass-through analysis.

        Returns:
            Panel with category-month observations ready for DiD estimation
        """
        df = self.fetch_with_cache()

        # Filter to non-admin categories for main analysis
        panel = df[df["admin_price"] == False].copy()  # noqa: E712

        # Add category index for panel regression
        panel["category_idx"] = pd.Categorical(panel["category"]).codes

        # Add month string (YYYY-MM)
        panel["month"] = panel["date"].dt.strftime("%Y-%m")

        # Add year-month numeric for time FE
        panel["time_idx"] = panel["date"].dt.year * 100 + panel["date"].dt.month

        return panel

    def save_all_raw(self) -> Path:
        """Fetch and save CPI data."""
        df = self.fetch_with_cache()
        return self.save_raw(df, "cpi_categories.parquet")
