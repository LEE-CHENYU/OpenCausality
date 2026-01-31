"""
FRED (Federal Reserve Economic Data) client.

Fetches:
- IGREA (Kilian global economic activity index)
- VIXCLS (VIX volatility index)
- DCOILBRENTEU (Brent crude oil prices)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from fredapi import Fred

from config.settings import get_settings
from shared.data.base import DataSource, DataSourceMetadata

logger = logging.getLogger(__name__)


class FREDSeries(Enum):
    """FRED series we need."""

    IGREA = "IGREA"  # Kilian global real economic activity index
    VIXCLS = "VIXCLS"  # VIX volatility index
    DCOILBRENTEU = "DCOILBRENTEU"  # Brent crude oil price


@dataclass
class SeriesConfig:
    """Configuration for a FRED series."""

    series_id: str
    frequency: str  # D, M, Q, A
    aggregation: str  # mean, last, sum
    transform: str | None  # None, log_return, innovation
    description: str


SERIES_CONFIGS: dict[FREDSeries, SeriesConfig] = {
    FREDSeries.IGREA: SeriesConfig(
        series_id="IGREA",
        frequency="M",
        aggregation="mean",
        transform="innovation",
        description="Kilian Global Real Economic Activity Index",
    ),
    FREDSeries.VIXCLS: SeriesConfig(
        series_id="VIXCLS",
        frequency="D",
        aggregation="mean",
        transform="innovation",
        description="CBOE Volatility Index (VIX)",
    ),
    FREDSeries.DCOILBRENTEU: SeriesConfig(
        series_id="DCOILBRENTEU",
        frequency="D",
        aggregation="mean",
        transform="log_return",
        description="Crude Oil Prices: Brent - Europe",
    ),
}


class FREDClient(DataSource):
    """Client for FRED economic data."""

    @property
    def source_name(self) -> str:
        return "fred"

    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None):
        super().__init__(cache_dir)
        settings = get_settings()
        self.api_key = api_key or settings.fred_api_key
        if not self.api_key:
            logger.warning(
                "No FRED API key provided. Set FRED_API_KEY environment variable."
            )
        self._fred = None

    @property
    def fred(self) -> Fred:
        """Lazy-loaded FRED client."""
        if self._fred is None:
            if not self.api_key:
                raise ValueError("FRED API key required")
            self._fred = Fred(api_key=self.api_key)
        return self._fred

    def fetch(
        self,
        series: FREDSeries | str,
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch a FRED series.

        Args:
            series: FRED series identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with date index and series values
        """
        if isinstance(series, FREDSeries):
            series_id = series.value
        else:
            series_id = series

        logger.info(f"Fetching FRED series: {series_id}")

        # Default date range
        if start_date is None:
            start_date = "2000-01-01"
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Fetch from FRED
        data = self.fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
        )

        # Convert to DataFrame
        df = pd.DataFrame({"date": data.index, "value": data.values})
        df["series"] = series_id
        df["date"] = pd.to_datetime(df["date"])

        # Record metadata
        self._metadata.append(
            DataSourceMetadata(
                source_name=self.source_name,
                fetch_time=datetime.now(),
                url=f"https://fred.stlouisfed.org/series/{series_id}",
                row_count=len(df),
                columns=df.columns.tolist(),
                date_range=(
                    df["date"].min().strftime("%Y-%m-%d"),
                    df["date"].max().strftime("%Y-%m-%d"),
                ),
            )
        )

        return df

    def fetch_and_resample(
        self,
        series: FREDSeries,
        target_freq: str = "Q",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch and resample to target frequency.

        Args:
            series: FRED series
            target_freq: Target frequency (Q, M, A)
            start_date: Start date
            end_date: End date

        Returns:
            Resampled DataFrame
        """
        config = SERIES_CONFIGS.get(series)
        if config is None:
            raise ValueError(f"Unknown series: {series}")

        df = self.fetch_with_cache(
            series=series, start_date=start_date, end_date=end_date
        )

        # Set date as index
        df = df.set_index("date")

        # Resample to target frequency
        # Map deprecated frequency aliases to new ones
        freq_map = {"Q": "QE", "M": "ME", "A": "YE"}
        resample_freq = freq_map.get(target_freq, target_freq)

        if config.aggregation == "mean":
            resampled = df["value"].resample(resample_freq).mean()
        elif config.aggregation == "last":
            resampled = df["value"].resample(resample_freq).last()
        elif config.aggregation == "sum":
            resampled = df["value"].resample(resample_freq).sum()
        else:
            resampled = df["value"].resample(resample_freq).mean()

        result = pd.DataFrame({"date": resampled.index, "value": resampled.values})
        result["series"] = series.value

        return result

    def compute_innovation(self, df: pd.DataFrame, ar_order: int = 1) -> pd.DataFrame:
        """
        Compute AR innovation (residual) for a series.

        Args:
            df: DataFrame with 'value' column
            ar_order: AR lag order

        Returns:
            DataFrame with innovation column added
        """
        import statsmodels.api as sm

        values = df["value"].dropna()

        # Fit AR model
        model = sm.tsa.AutoReg(values, lags=ar_order, old_names=False)
        result = model.fit()

        # Get residuals (innovations)
        innovations = result.resid

        # Align with original DataFrame
        df = df.copy()
        df["innovation"] = pd.Series(innovations.values, index=innovations.index)

        return df

    def compute_log_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute log returns for a series.

        Args:
            df: DataFrame with 'value' column

        Returns:
            DataFrame with log_return column added
        """
        import numpy as np

        df = df.copy()
        df["log_return"] = np.log(df["value"]).diff()
        return df

    def fetch_global_activity(
        self, freq: str = "Q", start_date: str | None = None
    ) -> pd.DataFrame:
        """Fetch IGREA and compute innovation."""
        df = self.fetch_and_resample(
            FREDSeries.IGREA, target_freq=freq, start_date=start_date
        )
        return self.compute_innovation(df)

    def fetch_vix(
        self, freq: str = "Q", start_date: str | None = None
    ) -> pd.DataFrame:
        """Fetch VIX and compute innovation."""
        df = self.fetch_and_resample(
            FREDSeries.VIXCLS, target_freq=freq, start_date=start_date
        )
        return self.compute_innovation(df)

    def fetch_brent(
        self, freq: str = "Q", start_date: str | None = None
    ) -> pd.DataFrame:
        """Fetch Brent oil price and compute log returns."""
        df = self.fetch_and_resample(
            FREDSeries.DCOILBRENTEU, target_freq=freq, start_date=start_date
        )
        return self.compute_log_return(df)

    def fetch_all_quarterly(
        self, start_date: str = "2000-01-01"
    ) -> dict[str, pd.DataFrame]:
        """Fetch all series resampled to quarterly."""
        return {
            "global_activity": self.fetch_global_activity(start_date=start_date),
            "vix": self.fetch_vix(start_date=start_date),
            "brent": self.fetch_brent(start_date=start_date),
        }

    def save_all_raw(self, start_date: str = "2000-01-01") -> dict[str, Path]:
        """Fetch and save all raw FRED data."""
        paths = {}
        for series in FREDSeries:
            try:
                df = self.fetch(series, start_date=start_date)
                path = self.save_raw(df, f"{series.value}.parquet")
                paths[series.value] = path
            except Exception as e:
                logger.error(f"Failed to fetch {series.value}: {e}")
        return paths
