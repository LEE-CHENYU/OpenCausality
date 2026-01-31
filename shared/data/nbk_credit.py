"""
National Bank of Kazakhstan credit data client.

Fetches loan portfolio quality and credit statistics from NBK.

Data sources:
- Loan portfolio structure and quality (monthly, 2005-present)
- Loans to economy (monthly, 1996-present)
- Financial Soundness Indicators (quarterly)
"""

import io
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


# NBK data URLs (these may need updating if NBK changes their structure)
NBK_URLS = {
    "loan_portfolio_quality": "https://nationalbank.kz/en/news/banks-performance/rubrics/2186",
    "loans_to_economy": "https://nationalbank.kz/en/news/kredity-bankovskogo-sektora-ekonomike",
    "fsi": "https://nationalbank.kz/en/page/indikatory-finansovoy-ustoychivosti",
}

# Direct file download patterns (NBK uses file IDs)
# These need to be discovered from the website
NBK_FILE_IDS = {
    "loan_portfolio_quality": None,  # Will be discovered
    "loans_to_economy": None,
}


@dataclass
class CreditQualityData:
    """Parsed credit quality data from NBK."""

    # Time series of overdue rates
    overdue_30_plus: pd.Series  # % of loans 30+ days overdue
    overdue_60_plus: pd.Series  # % of loans 60+ days overdue
    overdue_90_plus: pd.Series  # % of loans 90+ days overdue (NPL proxy)

    # By borrower type
    consumer_overdue: pd.Series | None = None
    corporate_overdue: pd.Series | None = None

    # Total loans for context
    total_loans: pd.Series | None = None

    # Metadata
    source: str = "NBK"
    frequency: str = "monthly"
    last_updated: str | None = None


class NBKCreditClient:
    """
    Client for National Bank of Kazakhstan credit statistics.

    Fetches:
    - Loan portfolio quality (overdue by days)
    - Loans to economy (consumer, corporate breakdown)
    - Financial Soundness Indicators

    Note: Does not inherit from HTTPDataSource to avoid abstract method issues.
    NBK data requires manual download or specific file handling.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        raw_data_dir: Path | None = None,
    ):
        """
        Initialize NBK credit client.

        Args:
            cache_dir: Directory for caching downloaded files
            raw_data_dir: Directory for saving raw data
        """
        self.base_url = "https://nationalbank.kz"
        self.raw_data_dir = raw_data_dir or Path("data/raw/nbk_credit")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_loan_portfolio_quality(
        self,
        file_path: Path | None = None,
    ) -> pd.DataFrame:
        """
        Fetch loan portfolio quality data.

        If file_path is provided, reads from local file.
        Otherwise attempts to download from NBK.

        Args:
            file_path: Path to local Excel file (optional)

        Returns:
            DataFrame with monthly credit quality metrics
        """
        if file_path and file_path.exists():
            logger.info(f"Reading loan portfolio quality from {file_path}")
            return self._parse_loan_portfolio_excel(file_path)

        # Try to download
        logger.info("Attempting to download loan portfolio quality data from NBK...")

        # NBK requires navigating their website to get file URLs
        # For now, provide instructions for manual download
        raise NotImplementedError(
            "Automatic download from NBK not yet implemented. "
            "Please manually download the Excel file from:\n"
            f"  {NBK_URLS['loan_portfolio_quality']}\n"
            "Look for: 'Information on the structure and quality of the loan portfolio'\n"
            "Then call this method with file_path=Path('path/to/file.xlsx')"
        )

    def _parse_loan_portfolio_excel(self, file_path: Path) -> pd.DataFrame:
        """
        Parse NBK loan portfolio quality Excel file.

        The Excel structure typically has:
        - Multiple sheets (by year or category)
        - Rows for different metrics (total loans, overdue by days)
        - Columns for months

        Args:
            file_path: Path to Excel file

        Returns:
            Tidy DataFrame with columns: date, metric, value
        """
        logger.info(f"Parsing {file_path}")

        # Read all sheets
        xlsx = pd.ExcelFile(file_path)

        all_data = []

        for sheet_name in xlsx.sheet_names:
            try:
                df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
                parsed = self._parse_loan_quality_sheet(df, sheet_name)
                if parsed is not None:
                    all_data.append(parsed)
            except Exception as e:
                logger.warning(f"Could not parse sheet {sheet_name}: {e}")

        if not all_data:
            raise ValueError(f"Could not parse any sheets from {file_path}")

        combined = pd.concat(all_data, ignore_index=True)

        # Ensure date column
        if "date" in combined.columns:
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.sort_values("date")

        return combined

    def _parse_loan_quality_sheet(
        self,
        df: pd.DataFrame,
        sheet_name: str,
    ) -> pd.DataFrame | None:
        """
        Parse a single sheet from loan portfolio quality file.

        NBK files have varying structures. This method attempts to
        identify the data layout and extract relevant metrics.
        """
        # Look for header row with dates
        date_row = None
        for i, row in df.iterrows():
            # Check if row contains date-like values
            date_count = sum(
                1 for val in row
                if isinstance(val, (datetime, pd.Timestamp)) or
                (isinstance(val, str) and any(m in str(val).lower() for m in
                    ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                     'jul', 'aug', 'sep', 'oct', 'nov', 'dec']))
            )
            if date_count >= 3:
                date_row = i
                break

        if date_row is None:
            logger.debug(f"No date header found in sheet {sheet_name}")
            return None

        # Set header
        df.columns = df.iloc[date_row]
        df = df.iloc[date_row + 1:].reset_index(drop=True)

        # Try to identify metric column (usually first)
        metric_col = df.columns[0]
        date_cols = df.columns[1:]

        # Melt to long format
        melted = df.melt(
            id_vars=[metric_col],
            value_vars=date_cols,
            var_name="date",
            value_name="value",
        )
        melted = melted.rename(columns={metric_col: "metric"})

        # Clean
        melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
        melted = melted.dropna(subset=["value"])

        return melted

    def fetch_loans_to_economy(
        self,
        file_path: Path | None = None,
    ) -> pd.DataFrame:
        """
        Fetch loans to economy data (consumer, mortgage, corporate breakdown).

        Args:
            file_path: Path to local Excel file (optional)

        Returns:
            DataFrame with monthly loan volumes by category
        """
        if file_path and file_path.exists():
            logger.info(f"Reading loans to economy from {file_path}")
            return self._parse_loans_to_economy_excel(file_path)

        raise NotImplementedError(
            "Automatic download from NBK not yet implemented. "
            "Please manually download the Excel file from:\n"
            f"  {NBK_URLS['loans_to_economy']}\n"
            "Look for: 'Loans from banking sector to economy'\n"
            "Then call this method with file_path=Path('path/to/file.xlsx')"
        )

    def _parse_loans_to_economy_excel(self, file_path: Path) -> pd.DataFrame:
        """Parse NBK loans to economy Excel file."""
        logger.info(f"Parsing {file_path}")

        xlsx = pd.ExcelFile(file_path)

        all_data = []

        for sheet_name in xlsx.sheet_names:
            try:
                df = pd.read_excel(xlsx, sheet_name=sheet_name)
                # Attempt standard parsing
                if len(df.columns) > 2:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Could not parse sheet {sheet_name}: {e}")

        if not all_data:
            raise ValueError(f"Could not parse any sheets from {file_path}")

        # Return first valid sheet for now
        return all_data[0]

    def fetch_fsi(
        self,
        file_path: Path | None = None,
    ) -> pd.DataFrame:
        """
        Fetch Financial Soundness Indicators.

        Args:
            file_path: Path to local Excel file (optional)

        Returns:
            DataFrame with quarterly FSI metrics
        """
        if file_path and file_path.exists():
            logger.info(f"Reading FSI from {file_path}")
            return pd.read_excel(file_path)

        raise NotImplementedError(
            "Automatic download from NBK not yet implemented. "
            "Please manually download the Excel file from:\n"
            f"  {NBK_URLS['fsi']}\n"
            "Look for: 'Financial Soundness Indicators' Excel file\n"
            "Then call this method with file_path=Path('path/to/file.xlsx')"
        )

    def build_credit_quality_panel(
        self,
        loan_quality_path: Path | None = None,
        loans_economy_path: Path | None = None,
        fsi_path: Path | None = None,
    ) -> pd.DataFrame:
        """
        Build unified credit quality panel from available sources.

        Combines:
        - Loan portfolio quality (overdue rates)
        - Loans to economy (volumes)
        - FSI (standardized ratios)

        Args:
            loan_quality_path: Path to loan quality Excel
            loans_economy_path: Path to loans to economy Excel
            fsi_path: Path to FSI Excel

        Returns:
            Monthly panel with credit quality metrics
        """
        dfs = []

        # Loan quality (primary source for overdue rates)
        if loan_quality_path and loan_quality_path.exists():
            quality_df = self.fetch_loan_portfolio_quality(loan_quality_path)
            quality_df["source"] = "loan_quality"
            dfs.append(quality_df)

        # Loans to economy (for volumes/growth)
        if loans_economy_path and loans_economy_path.exists():
            loans_df = self.fetch_loans_to_economy(loans_economy_path)
            loans_df["source"] = "loans_economy"
            dfs.append(loans_df)

        # FSI (for standardized NPL ratio)
        if fsi_path and fsi_path.exists():
            fsi_df = self.fetch_fsi(fsi_path)
            fsi_df["source"] = "fsi"
            dfs.append(fsi_df)

        if not dfs:
            raise ValueError("No data files provided")

        # For now, return the primary loan quality data
        # Full merge logic would depend on actual file structures
        return dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)

    def get_download_instructions(self) -> str:
        """Get instructions for manually downloading NBK data."""
        return """
NBK Credit Data Download Instructions
=====================================

The National Bank of Kazakhstan provides Excel files that must be
downloaded manually from their website.

1. LOAN PORTFOLIO QUALITY (Primary - overdue rates by days)
   URL: https://nationalbank.kz/en/news/banks-performance/rubrics/2186
   File: "Information on the structure and quality of the loan portfolio"
   Format: Excel (.xlsx)
   Frequency: Monthly
   Coverage: 2005-present

   Save to: data/raw/nbk_credit/loan_portfolio_quality.xlsx

2. LOANS TO ECONOMY (Volume breakdown)
   URL: https://nationalbank.kz/en/news/kredity-bankovskogo-sektora-ekonomike
   File: "Loans from banking sector to economy"
   Format: Excel (.xlsx)
   Frequency: Monthly
   Coverage: 1996-present

   Save to: data/raw/nbk_credit/loans_to_economy.xlsx

3. FINANCIAL SOUNDNESS INDICATORS (Standardized FSI)
   URL: https://nationalbank.kz/en/page/indikatory-finansovoy-ustoychivosti
   File: "Financial Soundness Indicators" (latest quarter)
   Format: Excel (.xlsx)
   Frequency: Quarterly

   Save to: data/raw/nbk_credit/fsi_latest.xlsx

After downloading, use:
    client = NBKCreditClient()
    df = client.fetch_loan_portfolio_quality(
        file_path=Path("data/raw/nbk_credit/loan_portfolio_quality.xlsx")
    )
"""


def get_nbk_credit_client() -> NBKCreditClient:
    """Get NBK credit client instance."""
    return NBKCreditClient()
