"""
IMF Financial Soundness Indicators client.

Fetches standardized FSI data including NPL ratios from IMF.

Data source: https://data.imf.org/FSI
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


# IMF FSI API endpoints
IMF_FSI_BASE = "https://data.imf.org/api/v1"

# Kazakhstan country code
KAZAKHSTAN_CODE = "KZ"

# Key FSI indicator codes
FSI_INDICATORS = {
    "npl_ratio": "FSANL_PT",  # Non-performing loans to total gross loans
    "capital_ratio": "FSERA_PT",  # Regulatory capital to risk-weighted assets
    "roa": "FSKRAR_PT",  # Return on assets
    "roe": "FSKRER_PT",  # Return on equity
    "liquid_assets": "FSKLAC_PT",  # Liquid assets to total assets
    "provisions": "FSKAP_PT",  # Provisions to NPLs
}


@dataclass
class FSIDataPoint:
    """A single FSI data point."""
    indicator: str
    country: str
    date: str
    value: float
    unit: str = "percent"


class IMFFSIClient:
    """
    Client for IMF Financial Soundness Indicators.

    Provides access to standardized FSI data including:
    - NPL ratio (non-performing loans to gross loans)
    - Capital adequacy ratios
    - Profitability metrics
    - Liquidity ratios
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
    ):
        """Initialize IMF FSI client."""
        self.base_url = IMF_FSI_BASE
        self.raw_data_dir = Path("data/raw/imf_fsi")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_npl_ratio(
        self,
        country: str = "KZ",
        start_year: int = 2000,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """
        Fetch NPL ratio time series for a country.

        Note: IMF FSI API access may require registration or
        may not be fully public. This method also supports
        loading from pre-downloaded files.

        Args:
            country: ISO country code
            start_year: Start year
            end_year: End year

        Returns:
            DataFrame with date and npl_ratio columns
        """
        # Try to load from cached file first
        cache_file = self.raw_data_dir / f"npl_ratio_{country}.csv"
        if cache_file.exists():
            logger.info(f"Loading cached NPL data from {cache_file}")
            df = pd.read_csv(cache_file)
            df["date"] = pd.to_datetime(df["date"])
            return df

        # Try World Bank data as fallback (more reliably accessible)
        logger.info("Attempting to fetch from World Bank API...")
        return self._fetch_worldbank_npl(country, start_year, end_year)

    def _fetch_worldbank_npl(
        self,
        country: str,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """
        Fetch NPL ratio from World Bank API.

        World Bank indicator: FB.AST.NPER.ZS
        """
        indicator = "FB.AST.NPER.ZS"
        # World Bank API expects ISO3 country codes (kaz for Kazakhstan)
        country_map = {"KZ": "kaz", "kz": "kaz"}
        country_code = country_map.get(country, country.lower())
        url = (
            f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
            f"?date={start_year}:{end_year}&format=json&per_page=100"
        )

        try:
            response = httpx.get(url, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            if len(data) < 2 or data[1] is None:
                logger.warning("No data returned from World Bank API")
                return pd.DataFrame(columns=["date", "npl_ratio"])

            records = []
            for item in data[1]:
                if item["value"] is not None:
                    records.append({
                        "date": f"{item['date']}-12-31",  # Annual, use year-end
                        "npl_ratio": float(item["value"]),
                        "country": item["country"]["id"],
                        "source": "World Bank",
                    })

            df = pd.DataFrame(records)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")

                # Cache the result
                cache_file = self.raw_data_dir / f"npl_ratio_{country}.csv"
                df.to_csv(cache_file, index=False)
                logger.info(f"Cached NPL data to {cache_file}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch World Bank NPL data: {e}")
            return pd.DataFrame(columns=["date", "npl_ratio"])

    def fetch_all_fsi(
        self,
        country: str = "KZ",
        start_year: int = 2000,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """
        Fetch all available FSI indicators for a country.

        Returns:
            DataFrame with date and indicator columns
        """
        # Start with NPL ratio
        df = self.fetch_npl_ratio(country, start_year, end_year)

        # Could add other indicators here if API access is available
        return df

    def get_kazakhstan_npl_series(self) -> pd.DataFrame:
        """
        Get Kazakhstan NPL ratio time series.

        Convenience method that handles caching and fallbacks.

        Returns:
            DataFrame with date and npl_ratio columns
        """
        return self.fetch_npl_ratio(country="KZ")

    def save_npl_data(self, df: pd.DataFrame, country: str = "KZ") -> Path:
        """Save NPL data to cache."""
        cache_file = self.raw_data_dir / f"npl_ratio_{country}.csv"
        df.to_csv(cache_file, index=False)
        logger.info(f"Saved NPL data to {cache_file}")
        return cache_file


def get_imf_fsi_client() -> IMFFSIClient:
    """Get IMF FSI client instance."""
    return IMFFSIClient()


def fetch_kazakhstan_npl() -> pd.DataFrame:
    """
    Convenience function to fetch Kazakhstan NPL data.

    Returns:
        DataFrame with date and npl_ratio columns
    """
    client = IMFFSIClient()
    return client.get_kazakhstan_npl_series()
