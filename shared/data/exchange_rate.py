"""
Three-tier exchange rate fetcher for Kazakhstan.

Tier 1: NBK (National Bank of Kazakhstan) - USD/KZT official rate
Tier 2: IMF EER (Effective Exchange Rates) - requires auth
Tier 3: World Bank REER - annual only, robustness

NBK Data Source: https://nationalbank.kz/en/exchangerates/ezhednevnye-oficialnye-rynochnye-kursy-valyut
"""

import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from config.settings import get_settings
from shared.data.base import HTTPDataSource

logger = logging.getLogger(__name__)


@dataclass
class ExchangeRateMetadata:
    """Metadata for exchange rate data."""

    source: str
    tier: int
    frequency: str
    start_date: str | None = None
    end_date: str | None = None
    n_obs: int = 0
    currency_pair: str = "USD/KZT"
    notes: str = ""


class ExchangeRateClient(HTTPDataSource):
    """
    Three-tier exchange rate fetcher for shared infrastructure.

    Tiers:
        1. NBK USD/KZT - Primary, daily/monthly
        2. IMF EER - NEER/REER, monthly (requires auth)
        3. World Bank REER - Annual (robustness only)
    """

    @property
    def source_name(self) -> str:
        return "exchange_rate"

    def __init__(self, cache_dir: Path | None = None):
        super().__init__(cache_dir)
        settings = get_settings()
        self.nbk_base_url = "https://nationalbank.kz"
        self._metadata: list[ExchangeRateMetadata] = []

    def fetch(self, **kwargs: Any) -> pd.DataFrame:
        """
        Fetch exchange rate data using three-tier fallback.

        Args:
            tier: Specific tier to fetch from (1, 2, or 3)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: date, rate, source
        """
        tier = kwargs.get("tier")
        start_date = kwargs.get("start_date", "2000-01-01")
        end_date = kwargs.get("end_date", datetime.now().strftime("%Y-%m-%d"))

        if tier is not None:
            if tier == 1:
                return self._fetch_nbk_usd_kzt(start_date, end_date)
            elif tier == 2:
                return self._fetch_imf_eer()
            elif tier == 3:
                return self._fetch_worldbank_reer()
            else:
                raise ValueError(f"Invalid tier: {tier}. Must be 1, 2, or 3.")

        # Auto-fallback through tiers
        # Tier 1: NBK USD/KZT
        try:
            df = self._fetch_nbk_usd_kzt(start_date, end_date)
            if not df.empty:
                logger.info(f"Successfully fetched {len(df)} NBK exchange rates")
                return df
        except Exception as e:
            logger.warning(f"NBK fetch failed: {e}")

        # Tier 2: IMF EER (if auth available)
        try:
            df = self._fetch_imf_eer()
            if not df.empty:
                logger.info(f"Successfully fetched {len(df)} IMF EER rates")
                return df
        except Exception as e:
            logger.warning(f"IMF EER fetch failed: {e}")

        # Tier 3: World Bank (annual only)
        try:
            df = self._fetch_worldbank_reer()
            if not df.empty:
                logger.info(f"Successfully fetched {len(df)} World Bank REER rates")
                return df
        except Exception as e:
            logger.warning(f"World Bank REER fetch failed: {e}")

        raise ValueError(
            "All exchange rate fetch tiers failed.\n"
            "Please check network connectivity and try again."
        )

    def _fetch_nbk_usd_kzt(
        self,
        start_date: str = "2000-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch USD/KZT official rate from National Bank of Kazakhstan.

        NBK provides daily rates via their API/data portal.
        Uses the get_rates.cfm endpoint to fetch monthly data points.
        """
        import httpx
        from xml.etree import ElementTree as ET

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Fetching NBK USD/KZT rates from {start_date} to {end_date}")

        # Create a client with longer timeout for NBK API
        nbk_client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=30.0),
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; KazakhstanResearch/1.0)"},
        )

        rates = []

        try:
            # Generate monthly date points to fetch
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Create monthly date range (first of each month)
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="MS")

            api_url = f"{self.nbk_base_url}/rss/get_rates.cfm"

            for fetch_date in date_range:
                try:
                    # Format date as DD.MM.YYYY for NBK API
                    fdate = fetch_date.strftime("%d.%m.%Y")
                    response = nbk_client.get(api_url, params={"fdate": fdate})

                    if response.status_code != 200:
                        continue

                    root = ET.fromstring(response.content)

                    # Get date from root element
                    date_elem = root.find("date")
                    rate_date = None
                    if date_elem is not None and date_elem.text:
                        try:
                            rate_date = pd.to_datetime(date_elem.text, format="%d.%m.%Y")
                        except ValueError:
                            rate_date = fetch_date

                    # Find USD rate
                    for item in root.findall(".//item"):
                        title_elem = item.find("title")
                        desc_elem = item.find("description")

                        if title_elem is not None and title_elem.text == "USD":
                            if desc_elem is not None and desc_elem.text:
                                try:
                                    rate = float(desc_elem.text.strip())
                                    rates.append({
                                        "date": rate_date or fetch_date,
                                        "rate": rate,
                                    })
                                except (ValueError, TypeError):
                                    pass
                            break

                except Exception as e:
                    logger.debug(f"Failed to fetch rate for {fetch_date}: {e}")
                    continue

            if rates:
                df = pd.DataFrame(rates)
                df["source"] = "nbk"
                df["currency_pair"] = "USD/KZT"
                df = df.sort_values("date").reset_index(drop=True)
                df = df.drop_duplicates(subset=["date"], keep="last")

                self._metadata.append(ExchangeRateMetadata(
                    source="NBK",
                    tier=1,
                    frequency="monthly",
                    start_date=df["date"].min().strftime("%Y-%m-%d"),
                    end_date=df["date"].max().strftime("%Y-%m-%d"),
                    n_obs=len(df),
                ))

                logger.info(f"Fetched {len(df)} monthly NBK exchange rates")
                return df[["date", "rate", "source", "currency_pair"]]

        except Exception as e:
            logger.warning(f"NBK monthly fetch failed: {e}")

        finally:
            nbk_client.close()

        # Fallback to XML/RSS endpoint for current rates
        try:
            rss_url = f"{self.nbk_base_url}/rss/rates_all.xml"
            response = self.client.get(rss_url)
            response.raise_for_status()

            # Parse XML
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)

            rates = []
            for item in root.findall(".//item"):
                title = item.find("title")
                if title is not None and title.text and "USD" in title.text:
                    desc = item.find("description")
                    pub_date = item.find("pubDate")
                    if desc is not None and pub_date is not None:
                        try:
                            rate = float(desc.text.strip())
                            date = pd.to_datetime(pub_date.text)
                            rates.append({"date": date, "rate": rate})
                        except (ValueError, TypeError):
                            continue

            if rates:
                df = pd.DataFrame(rates)
                df["source"] = "nbk"
                df["currency_pair"] = "USD/KZT"
                df = df.sort_values("date").reset_index(drop=True)

                self._metadata.append(ExchangeRateMetadata(
                    source="NBK",
                    tier=1,
                    frequency="daily",
                    start_date=df["date"].min().strftime("%Y-%m-%d"),
                    end_date=df["date"].max().strftime("%Y-%m-%d"),
                    n_obs=len(df),
                    notes="Limited to recent rates from RSS feed",
                ))

                return df[["date", "rate", "source", "currency_pair"]]

        except Exception as e:
            logger.debug(f"NBK RSS feed failed: {e}")

        # Final fallback: try historical CSV if available locally
        settings = get_settings()
        local_path = settings.project_root / "data/raw/nbk/usd_kzt.parquet"
        if local_path.exists():
            logger.info(f"Loading cached NBK data from {local_path}")
            return pd.read_parquet(local_path)

        raise RuntimeError("NBK exchange rate fetch failed")

    def _fetch_imf_eer(self) -> pd.DataFrame:
        """
        Fetch NEER/REER from IMF International Financial Statistics.

        Requires IMF data portal access (may need authentication).
        """
        # IMF IFS API for Kazakhstan NEER/REER
        # Series: ENDA_XDC_EUR_NEER, ENDA_XDC_EUR_REER
        imf_url = "https://data.imf.org/api/v1/data/IFS"

        params = {
            "countries": "KZ",
            "indicators": "ENDA_XDC_EUR_NEER,ENDA_XDC_EUR_REER",
            "startPeriod": "2000",
            "format": "json",
        }

        try:
            response = self.client.get(imf_url, params=params)
            response.raise_for_status()

            data = response.json()
            # Parse IMF SDMX-JSON format
            if "dataSets" in data and len(data["dataSets"]) > 0:
                observations = data["dataSets"][0].get("observations", {})
                # Transform to DataFrame
                records = []
                for key, value in observations.items():
                    # Parse key (time series index)
                    records.append({
                        "date": pd.Timestamp(key),
                        "rate": value[0],
                    })

                df = pd.DataFrame(records)
                df["source"] = "imf_eer"
                df["currency_pair"] = "NEER"

                self._metadata.append(ExchangeRateMetadata(
                    source="IMF EER",
                    tier=2,
                    frequency="monthly",
                    start_date=df["date"].min().strftime("%Y-%m-%d"),
                    end_date=df["date"].max().strftime("%Y-%m-%d"),
                    n_obs=len(df),
                ))

                return df

        except Exception as e:
            logger.warning(f"IMF EER API failed: {e}")

        raise RuntimeError("IMF EER fetch requires authentication or is unavailable")

    def _fetch_worldbank_reer(self) -> pd.DataFrame:
        """
        Fetch Real Effective Exchange Rate from World Bank.

        Indicator: PX.REX.REER (Real effective exchange rate index)
        Frequency: Annual
        """
        import httpx

        wb_url = "https://api.worldbank.org/v2/country/KZ/indicator/PX.REX.REER"
        params = {
            "format": "json",
            "date": "2000:2026",
            "per_page": 100,
        }

        # Use longer timeout for World Bank API (known to be slow)
        wb_client = httpx.Client(
            timeout=httpx.Timeout(90.0, connect=30.0),
            follow_redirects=True,
        )

        try:
            response = wb_client.get(wb_url, params=params)
            response.raise_for_status()

            data = response.json()
            if len(data) > 1 and data[1]:
                records = []
                for obs in data[1]:
                    if obs.get("value") is not None:
                        records.append({
                            "date": pd.Timestamp(f"{obs['date']}-01-01"),
                            "rate": float(obs["value"]),
                        })

                df = pd.DataFrame(records)
                df["source"] = "worldbank"
                df["currency_pair"] = "REER"
                df = df.sort_values("date").reset_index(drop=True)

                self._metadata.append(ExchangeRateMetadata(
                    source="World Bank",
                    tier=3,
                    frequency="annual",
                    start_date=df["date"].min().strftime("%Y-%m-%d"),
                    end_date=df["date"].max().strftime("%Y-%m-%d"),
                    n_obs=len(df),
                    notes="REER index, robustness check only",
                ))

                return df

        except Exception as e:
            logger.warning(f"World Bank API failed: {e}")
        finally:
            wb_client.close()

        raise RuntimeError("World Bank REER fetch failed")

    def fetch_nbk(
        self,
        start_date: str = "2000-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Convenience method to fetch NBK data only."""
        return self._fetch_nbk_usd_kzt(start_date, end_date)

    def fetch_with_fallback(self) -> tuple[pd.DataFrame, str]:
        """
        Fetch exchange rates with full fallback chain.

        Returns:
            Tuple of (DataFrame, source_description)
        """
        try:
            df = self.fetch_with_cache(tier=1)
            return df, "nbk_usd_kzt"
        except Exception:
            pass

        try:
            df = self.fetch_with_cache(tier=2)
            return df, "imf_eer"
        except Exception:
            pass

        try:
            df = self.fetch_with_cache(tier=3)
            return df, "worldbank_reer"
        except Exception:
            pass

        raise ValueError("All exchange rate tiers failed")

    def compute_fx_change(
        self,
        df: pd.DataFrame,
        method: Literal["log_return", "pct_change"] = "log_return",
    ) -> pd.DataFrame:
        """
        Compute exchange rate changes for shock construction.

        Args:
            df: DataFrame with date and rate columns
            method: 'log_return' or 'pct_change'

        Returns:
            DataFrame with additional fx_change column
        """
        import numpy as np

        df = df.copy()
        df = df.sort_values("date")

        if method == "log_return":
            df["fx_change"] = np.log(df["rate"]) - np.log(df["rate"].shift(1))
        else:
            df["fx_change"] = df["rate"].pct_change()

        return df

    def aggregate_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily rates to monthly (end of period).

        Args:
            df: DataFrame with daily rates

        Returns:
            Monthly DataFrame
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Resample to month-end
        monthly = df.groupby(pd.Grouper(freq="ME")).agg({
            "rate": "last",
            "source": "first",
            "currency_pair": "first",
        }).reset_index()

        monthly = monthly.dropna(subset=["rate"])

        return monthly

    def aggregate_to_quarterly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate to quarterly (end of period).

        Args:
            df: DataFrame with rates

        Returns:
            Quarterly DataFrame with quarter column
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Resample to quarter-end
        quarterly = df.groupby(pd.Grouper(freq="QE")).agg({
            "rate": "last",
            "source": "first",
            "currency_pair": "first",
        }).reset_index()

        quarterly = quarterly.dropna(subset=["rate"])

        # Add quarter string
        quarterly["quarter"] = (
            quarterly["date"].dt.year.astype(str) + "Q" +
            quarterly["date"].dt.quarter.astype(str)
        )

        return quarterly

    def save_all_raw(self) -> dict[str, Path]:
        """Fetch and save all available tiers."""
        paths = {}
        settings = get_settings()

        for tier in [1, 3]:  # Skip tier 2 (requires auth)
            try:
                df = self.fetch(tier=tier)
                if not df.empty:
                    tier_name = {1: "nbk", 2: "imf_eer", 3: "worldbank"}[tier]
                    filename = f"usd_kzt_{tier_name}.parquet"
                    path = self.save_raw(df, filename)
                    paths[tier_name] = path
            except Exception as e:
                logger.warning(f"Failed to save tier {tier}: {e}")

        return paths
