"""
Data pipeline for credit quality study.

Combines:
- NBK credit quality data (overdue rates)
- IMF/World Bank NPL ratios
- External shock series (oil, VIX, global activity)
- Macro controls (CPI, policy rate, exchange rate)

Outputs a unified panel for local projections estimation.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.data.nbk_credit import NBKCreditClient
from shared.data.imf_fsi import IMFFSIClient, fetch_kazakhstan_npl
from shared.data.fred_client import FREDClient

logger = logging.getLogger(__name__)


class CreditDataPipeline:
    """
    Data pipeline for credit quality analysis.

    Fetches and combines:
    1. Credit quality outcomes (NBK, IMF, World Bank)
    2. External shocks (oil, VIX, global demand)
    3. Macro controls (inflation, rates, FX)
    """

    def __init__(
        self,
        raw_data_dir: Path | None = None,
        processed_data_dir: Path | None = None,
    ):
        """
        Initialize pipeline.

        Args:
            raw_data_dir: Directory for raw data files
            processed_data_dir: Directory for processed panel
        """
        self.raw_data_dir = raw_data_dir or Path("data/raw")
        self.processed_data_dir = processed_data_dir or Path("data/processed")

        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize clients
        self.nbk_client = NBKCreditClient()
        self.imf_client = IMFFSIClient()
        self.fred_client = FREDClient()

    def fetch_npl_data(self) -> pd.DataFrame:
        """
        Fetch NPL ratio from available sources.

        Tries in order:
        1. IMF/World Bank API
        2. Cached local file

        Returns:
            DataFrame with date and npl_ratio
        """
        logger.info("Fetching NPL data...")

        # Try IMF/World Bank
        df = fetch_kazakhstan_npl()

        if df.empty:
            logger.warning("Could not fetch NPL data from API")
            # Check for cached file
            cache_path = self.raw_data_dir / "imf_fsi" / "npl_ratio_KZ.csv"
            if cache_path.exists():
                df = pd.read_csv(cache_path)
                df["date"] = pd.to_datetime(df["date"])

        if not df.empty:
            logger.info(f"NPL data: {len(df)} observations, {df['date'].min()} to {df['date'].max()}")

        return df

    def fetch_oil_shocks(self) -> pd.DataFrame:
        """
        Fetch oil price/shock data.

        For now, uses Brent price returns as shock proxy.
        Could be upgraded to structural shocks (Baumeister-Hamilton).

        Returns:
            DataFrame with date, brent_price, brent_return, oil_shock
        """
        logger.info("Fetching oil data...")

        try:
            # Use the FREDClient's fetch_brent method which handles resampling
            brent_df = self.fred_client.fetch_brent(freq="Q", start_date="2000-01-01")

            if brent_df is not None and not brent_df.empty:
                df = brent_df.copy()
                df = df.rename(columns={"value": "brent_price", "log_return": "brent_return"})

                # Simple shock: standardized returns
                if "brent_return" in df.columns:
                    returns = df["brent_return"].dropna()
                    df["oil_shock"] = (
                        df["brent_return"] - returns.mean()
                    ) / returns.std()

                logger.info(f"Oil data: {len(df)} observations")
                return df

        except Exception as e:
            logger.error(f"Failed to fetch oil data: {e}")

        return pd.DataFrame(columns=["date", "brent_price", "brent_return", "oil_shock"])

    def fetch_vix(self) -> pd.DataFrame:
        """
        Fetch VIX (global risk) data.

        Returns:
            DataFrame with date, vix, vix_innovation
        """
        logger.info("Fetching VIX data...")

        try:
            # Use the FREDClient's fetch_vix method
            vix_df = self.fred_client.fetch_vix(freq="Q", start_date="2000-01-01")

            if vix_df is not None and not vix_df.empty:
                df = vix_df.copy()
                df = df.rename(columns={"value": "vix", "innovation": "vix_innovation"})

                logger.info(f"VIX data: {len(df)} observations")
                return df

        except Exception as e:
            logger.error(f"Failed to fetch VIX data: {e}")

        return pd.DataFrame(columns=["date", "vix", "vix_innovation"])

    def fetch_macro_controls(self) -> pd.DataFrame:
        """
        Fetch macro control variables for Kazakhstan.

        Returns:
            DataFrame with date and available macro variables
        """
        logger.info("Fetching macro controls...")

        dfs = []

        # Try to fetch global activity as a control
        try:
            igrea_df = self.fred_client.fetch_global_activity(freq="Q", start_date="2000-01-01")
            if igrea_df is not None and not igrea_df.empty:
                df = igrea_df.copy()
                df = df.rename(columns={"value": "global_activity", "innovation": "global_activity_shock"})
                dfs.append(df[["date", "global_activity", "global_activity_shock"]])
                logger.info(f"Global activity data: {len(df)} obs")
        except Exception as e:
            logger.warning(f"Could not fetch global activity: {e}")

        if dfs:
            # Merge all
            result = dfs[0]
            for df in dfs[1:]:
                result = result.merge(df, on="date", how="outer")
            return result

        return pd.DataFrame(columns=["date"])

    def build_panel(
        self,
        frequency: str = "quarterly",
        start_year: int = 2008,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """
        Build unified analysis panel.

        Args:
            frequency: 'monthly' or 'quarterly'
            start_year: Panel start year
            end_year: Panel end year

        Returns:
            DataFrame with all variables aligned by date
        """
        logger.info(f"Building {frequency} panel from {start_year} to {end_year}")

        # Fetch all data
        npl = self.fetch_npl_data()
        oil = self.fetch_oil_shocks()
        vix = self.fetch_vix()
        macro = self.fetch_macro_controls()

        # Resample to desired frequency
        def resample_to_freq(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
            if df.empty:
                return df

            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])

            # Only keep numeric columns plus date
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df = df[[date_col] + numeric_cols]

            df = df.set_index(date_col)

            if frequency == "quarterly":
                # Resample to quarter-end, taking mean
                df = df.resample("QE").mean(numeric_only=True)
            elif frequency == "monthly":
                df = df.resample("ME").mean(numeric_only=True)

            return df.reset_index()

        npl = resample_to_freq(npl)
        oil = resample_to_freq(oil)
        vix = resample_to_freq(vix)
        macro = resample_to_freq(macro)

        # Start with date skeleton
        if frequency == "quarterly":
            dates = pd.date_range(
                start=f"{start_year}-03-31",
                end=f"{end_year}-12-31",
                freq="QE",
            )
        else:
            dates = pd.date_range(
                start=f"{start_year}-01-31",
                end=f"{end_year}-12-31",
                freq="ME",
            )

        panel = pd.DataFrame({"date": dates})

        # Merge all data
        for df, name in [(npl, "npl"), (oil, "oil"), (vix, "vix"), (macro, "macro")]:
            if not df.empty and "date" in df.columns:
                panel = panel.merge(df, on="date", how="left")
                logger.info(f"Merged {name}: {len(df)} obs")

        # Filter date range
        panel = panel[
            (panel["date"].dt.year >= start_year) &
            (panel["date"].dt.year <= end_year)
        ]

        # Add derived variables
        if "npl_ratio" in panel.columns:
            panel["npl_change"] = panel["npl_ratio"].diff()
            panel["npl_pct_change"] = panel["npl_ratio"].pct_change()

        # Rename shock columns for clarity
        if "oil_shock" in panel.columns:
            panel["oil_supply_shock"] = panel["oil_shock"]  # Simplified naming

        logger.info(f"Panel built: {len(panel)} observations, {panel.shape[1]} columns")
        logger.info(f"Columns: {list(panel.columns)}")

        # Report missing data
        missing = panel.isnull().sum()
        if missing.any():
            logger.warning(f"Missing data:\n{missing[missing > 0]}")

        return panel

    def save_panel(self, panel: pd.DataFrame, filename: str = "credit_panel.parquet") -> Path:
        """Save panel to parquet."""
        path = self.processed_data_dir / filename
        panel.to_parquet(path, index=False)
        logger.info(f"Saved panel to {path}")
        return path

    def load_panel(self, filename: str = "credit_panel.parquet") -> pd.DataFrame:
        """Load panel from parquet."""
        path = self.processed_data_dir / filename
        if path.exists():
            return pd.read_parquet(path)
        raise FileNotFoundError(f"Panel not found at {path}")

    def get_data_summary(self, panel: pd.DataFrame) -> str:
        """Generate data summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("CREDIT QUALITY PANEL DATA SUMMARY")
        lines.append("=" * 60)

        lines.append(f"\nObservations: {len(panel)}")
        lines.append(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
        lines.append(f"Columns: {len(panel.columns)}")

        lines.append("\nVariables:")
        for col in panel.columns:
            if col != "date":
                non_null = panel[col].notna().sum()
                pct = non_null / len(panel) * 100
                lines.append(f"  {col}: {non_null} obs ({pct:.0f}% non-missing)")

        # Key statistics
        if "npl_ratio" in panel.columns:
            npl = panel["npl_ratio"].dropna()
            lines.append(f"\nNPL Ratio Statistics:")
            lines.append(f"  Mean: {npl.mean():.2f}%")
            lines.append(f"  Std: {npl.std():.2f}%")
            lines.append(f"  Min: {npl.min():.2f}%")
            lines.append(f"  Max: {npl.max():.2f}%")

        return "\n".join(lines)


def build_credit_panel(
    frequency: str = "quarterly",
    start_year: int = 2008,
    end_year: int = 2025,
) -> pd.DataFrame:
    """
    Convenience function to build credit quality panel.

    Args:
        frequency: 'monthly' or 'quarterly'
        start_year: Start year
        end_year: End year

    Returns:
        Analysis panel DataFrame
    """
    pipeline = CreditDataPipeline()
    return pipeline.build_panel(
        frequency=frequency,
        start_year=start_year,
        end_year=end_year,
    )
