"""
BNS National Income Aggregates.

Fetches national-level income data from Kazakhstan Bureau of National Statistics
for use in LP-IV income response analysis (Block B).

Source: BNS "Expenses and incomes of the population"

Available series:
- Nominal monetary income (monthly, quarterly, annual)
- Income structure: wages, self-employment, social transfers, other
- Transfer breakdown: pensions, benefits, TSA, housing assistance

Note: "Nominal monetary income" != "Disposable income after taxes"
This is the BNS macro estimate of monetary resources.
"""

import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from bs4 import BeautifulSoup

from config.settings import get_settings
from shared.data.base import HTTPDataSource

logger = logging.getLogger(__name__)


@dataclass
class NationalIncomeMetadata:
    """Metadata for national income data."""

    frequency: str = "monthly"
    start_date: str | None = None
    end_date: str | None = None
    n_periods: int = 0
    components: list[str] = field(default_factory=list)
    source: str = "BNS Expenses and Incomes"


class BNSNationalIncomeClient(HTTPDataSource):
    """
    Client for BNS national income aggregates.

    Fetches national-level income time series for Block B LP-IV analysis.
    """

    @property
    def source_name(self) -> str:
        return "bns_national_income"

    def __init__(self, cache_dir: Path | None = None):
        super().__init__(cache_dir)
        settings = get_settings()
        self.base_url = settings.bns_base_url
        self._metadata: NationalIncomeMetadata | None = None

    def fetch(self, **kwargs: Any) -> pd.DataFrame:
        """
        Fetch national income data.

        Args:
            frequency: 'monthly', 'quarterly', or 'annual'

        Returns:
            DataFrame with columns:
                - date: Period date
                - nominal_income: Total nominal monetary income
                - wage_income: Wage component
                - transfer_income: Social transfer component
                - self_employment_income: Self-employment income
                - other_income: Other income sources
                - pension_income: Pension transfers
                - benefit_income: Social benefits
        """
        frequency = kwargs.get("frequency", "quarterly")

        # Try BNS API first
        try:
            df = self._fetch_bns_income_api(frequency)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"BNS income API failed: {e}")

        # Fallback: BNS page scraping
        try:
            df = self._fetch_bns_income_page(frequency)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"BNS income page scraping failed: {e}")

        # Final fallback: local cache
        settings = get_settings()
        local_path = settings.project_root / "data/raw/kazakhstan_bns/national_income.parquet"
        if local_path.exists():
            logger.info(f"Loading cached income data from {local_path}")
            return pd.read_parquet(local_path)

        raise ValueError(
            "National income data unavailable.\n"
            "BNS API failed and no local cache found.\n"
            "Please download income data manually from:\n"
            "https://stat.gov.kz/en/industries/labor-and-income/stat-life/"
        )

    def _fetch_bns_income_api(self, frequency: str) -> pd.DataFrame:
        """Fetch income data via BNS API."""
        # BNS iblock IDs for income data
        iblock_ids = {
            "monthly": 48954,  # Monthly income statistics
            "quarterly": 48953,  # Quarterly income per capita
            "annual": 48510,  # Annual income
        }

        iblock_id = iblock_ids.get(frequency, iblock_ids["quarterly"])

        csv_url = f"{self.base_url}/api/iblock/element/{iblock_id}/csv/file/en/"
        logger.info(f"Fetching BNS national income from iblock {iblock_id}")

        response = self.client.get(csv_url)
        response.raise_for_status()

        df = pd.read_csv(io.BytesIO(response.content), sep="\t")

        return self._standardize_income_data(df, frequency)

    def _fetch_bns_income_page(self, frequency: str) -> pd.DataFrame:
        """Fetch income data by scraping BNS page."""
        page_url = f"{self.base_url}/en/industries/labor-and-income/stat-life/"

        response = self.client.get(page_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Find links to income data files
        data_links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            text = anchor.get_text(strip=True).lower()

            if ("income" in text or "monetary" in text) and "expense" not in text:
                if "/api/iblock/" in href or href.endswith((".xlsx", ".csv")):
                    full_url = href if href.startswith("http") else f"{self.base_url}{href}"
                    data_links.append((full_url, text))

        for url, text in data_links:
            try:
                logger.info(f"Trying income link: {text}")
                response = self.client.get(url)
                response.raise_for_status()

                content = response.content
                try:
                    df = pd.read_excel(io.BytesIO(content))
                except Exception:
                    df = pd.read_csv(io.BytesIO(content), sep="\t")

                if not df.empty:
                    return self._standardize_income_data(df, frequency)
            except Exception as e:
                logger.debug(f"Failed to fetch {url}: {e}")
                continue

        raise RuntimeError("No valid income data links found")

    def _standardize_income_data(
        self,
        df: pd.DataFrame,
        frequency: str,
    ) -> pd.DataFrame:
        """
        Standardize income data to common format.

        Args:
            df: Raw BNS data
            frequency: Data frequency

        Returns:
            Standardized DataFrame
        """
        import numpy as np

        # Try to identify columns by pattern matching
        records = []

        # Get column mapping
        period_col = None
        value_cols = {}

        for col in df.columns:
            col_lower = col.lower()
            if "period" in col_lower or "month" in col_lower or "quarter" in col_lower:
                period_col = col
            elif "val" in col_lower or df[col].dtype in [np.float64, np.int64]:
                # Try to categorize by examining column name
                if "wage" in col_lower or "labor" in col_lower:
                    value_cols["wage_income"] = col
                elif "transfer" in col_lower or "social" in col_lower:
                    value_cols["transfer_income"] = col
                elif "pension" in col_lower:
                    value_cols["pension_income"] = col
                elif "benefit" in col_lower:
                    value_cols["benefit_income"] = col
                elif "self" in col_lower or "enterprise" in col_lower:
                    value_cols["self_employment_income"] = col
                elif "total" in col_lower or "monetary" in col_lower:
                    value_cols["nominal_income"] = col
                elif not value_cols.get("nominal_income"):
                    # Default numeric column to nominal income
                    value_cols["nominal_income"] = col

        if period_col is None:
            # Look for date-like column
            for col in df.columns:
                if df[col].dtype == object:
                    sample = df[col].dropna().astype(str).iloc[:5] if len(df) > 0 else []
                    if any("2020" in s or "2021" in s or "Q" in s for s in sample):
                        period_col = col
                        break

        if period_col is None:
            logger.warning("Could not identify period column in income data")
            return pd.DataFrame()

        # Build records
        for _, row in df.iterrows():
            period = str(row[period_col])
            date = self._parse_period(period, frequency)

            if date is None:
                continue

            record = {"date": date}
            for key, col in value_cols.items():
                value = row.get(col)
                if pd.notna(value):
                    # Clean value (remove spaces, commas)
                    if isinstance(value, str):
                        value = value.replace(" ", "").replace(",", ".")
                    try:
                        record[key] = float(value)
                    except (ValueError, TypeError):
                        pass

            if "nominal_income" in record:
                records.append(record)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records)

        # Ensure all income columns exist
        for col in ["nominal_income", "wage_income", "transfer_income",
                    "self_employment_income", "other_income"]:
            if col not in result.columns:
                result[col] = np.nan

        # Compute other_income as residual if possible
        if result["other_income"].isna().all():
            components = ["wage_income", "transfer_income", "self_employment_income"]
            component_sum = result[components].sum(axis=1)
            result["other_income"] = result["nominal_income"] - component_sum
            result.loc[result["other_income"] < 0, "other_income"] = np.nan

        # Sort by date
        result = result.sort_values("date").reset_index(drop=True)

        # Add frequency-appropriate period columns
        if frequency == "quarterly":
            result["quarter"] = (
                result["date"].dt.year.astype(str) + "Q" +
                result["date"].dt.quarter.astype(str)
            )
        elif frequency == "monthly":
            result["month"] = result["date"].dt.strftime("%Y-%m")

        # Update metadata
        self._metadata = NationalIncomeMetadata(
            frequency=frequency,
            start_date=result["date"].min().strftime("%Y-%m-%d"),
            end_date=result["date"].max().strftime("%Y-%m-%d"),
            n_periods=len(result),
            components=list(value_cols.keys()),
        )

        return result

    def _parse_period(
        self,
        period: str,
        frequency: str,
    ) -> datetime | None:
        """Parse period string to datetime."""
        import re

        period = str(period).strip()

        if frequency == "quarterly":
            # YYYY Q# or YYYYQ#
            match = re.match(r"(\d{4})\s*Q(\d)", period)
            if match:
                year, q = int(match.group(1)), int(match.group(2))
                month = (q - 1) * 3 + 1
                return datetime(year, month, 1)

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

        # Annual: YYYY
        if re.match(r"^\d{4}$", period):
            try:
                return datetime(int(period), 1, 1)
            except ValueError:
                pass

        return None

    def fetch_quarterly(self) -> pd.DataFrame:
        """Fetch quarterly national income data."""
        return self.fetch_with_cache(frequency="quarterly")

    def fetch_monthly(self) -> pd.DataFrame:
        """Fetch monthly national income data."""
        return self.fetch_with_cache(frequency="monthly")

    def fetch_annual(self) -> pd.DataFrame:
        """Fetch annual national income data."""
        return self.fetch_with_cache(frequency="annual")

    def compute_income_growth(
        self,
        df: pd.DataFrame,
        log: bool = True,
    ) -> pd.DataFrame:
        """
        Compute income growth rates.

        Args:
            df: Income DataFrame
            log: If True, compute log differences

        Returns:
            DataFrame with growth rates added
        """
        import numpy as np

        df = df.copy()
        income_cols = ["nominal_income", "wage_income", "transfer_income"]

        for col in income_cols:
            if col not in df.columns:
                continue

            if log:
                df[f"{col}_log"] = np.log(df[col])
                df[f"{col}_growth"] = df[f"{col}_log"].diff()
            else:
                df[f"{col}_growth"] = df[col].pct_change()

        return df

    def compute_income_shares(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute income component shares.

        Args:
            df: Income DataFrame

        Returns:
            DataFrame with share columns added
        """
        df = df.copy()

        components = ["wage_income", "transfer_income", "self_employment_income", "other_income"]

        for col in components:
            if col in df.columns and "nominal_income" in df.columns:
                df[f"{col}_share"] = df[col] / df["nominal_income"]

        return df

    def build_income_panel(self) -> pd.DataFrame:
        """
        Build panel data for Block B income response analysis.

        Returns:
            Time series panel with income data ready for LP-IV estimation
        """
        df = self.fetch_quarterly()

        # Compute growth rates
        df = self.compute_income_growth(df)

        # Compute shares
        df = self.compute_income_shares(df)

        # Add time index for regression
        df["time_idx"] = df["date"].dt.year * 10 + df["date"].dt.quarter

        return df

    def save_all_raw(self) -> dict[str, Path]:
        """Fetch and save all income data."""
        paths = {}

        for freq in ["monthly", "quarterly", "annual"]:
            try:
                df = self.fetch(frequency=freq)
                if not df.empty:
                    path = self.save_raw(df, f"national_income_{freq}.parquet")
                    paths[freq] = path
            except Exception as e:
                logger.warning(f"Failed to save {freq} income data: {e}")

        return paths
