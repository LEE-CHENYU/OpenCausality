"""
Kazakhstan Bureau of National Statistics (BNS) data client.

Three-tier fallback strategy:
1. Known iblock element IDs -> direct file download
2. Known docIds -> getFile endpoint
3. HTML link discovery -> parse BNS pages to find correct iblock links

Additionally supports loading from alternative sources when BNS API is unavailable:
- USGS Mineral Industry Reports
- EITI Kazakhstan Reports
- stat.gov.kz GRP Publications (when main API fails)
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from config.settings import get_settings
from shared.data.base import HTTPDataSource, DataSourceMetadata

logger = logging.getLogger(__name__)


# Alternative data source paths
ALTERNATIVE_SOURCES_DIR = Path("data/raw/alternative_sources")
MINING_SHARES_FILE = ALTERNATIVE_SOURCES_DIR / "mining_shares.csv"
GRP_COMPOSITION_FILE = ALTERNATIVE_SOURCES_DIR / "grp_composition.csv"


class BNSDataType(Enum):
    """Types of BNS data we need."""

    INCOME_PER_CAPITA = "income_per_capita"
    EXPENDITURE_STRUCTURE = "expenditure_structure"
    MINING_SHARES = "mining_shares"
    GRP_BY_REGION = "grp_by_region"
    EMPLOYMENT = "employment"
    # DAG node dataset mappings
    HOUSEHOLD_INCOME = "household_income"
    HOUSEHOLD_BUDGET = "household_budget"
    REGIONAL_ACCOUNTS = "regional_accounts"
    PRICES = "prices"


@dataclass
class BNSEndpoint:
    """Configuration for a BNS data endpoint."""

    data_type: BNSDataType
    iblock_id: int | None = None
    doc_id: str | None = None
    discovery_url: str | None = None
    file_pattern: str | None = None
    description: str = ""


# Known endpoints - these may need updating as BNS site changes
KNOWN_ENDPOINTS: dict[BNSDataType, BNSEndpoint] = {
    BNSDataType.INCOME_PER_CAPITA: BNSEndpoint(
        data_type=BNSDataType.INCOME_PER_CAPITA,
        iblock_id=48953,  # Quarterly per capita income
        discovery_url="/en/industries/labor-and-income/stat-life/",
        file_pattern=r"monetary.*income.*region|income.*capita",
        description="Per-capita monetary income by region (quarterly)",
    ),
    BNSDataType.EXPENDITURE_STRUCTURE: BNSEndpoint(
        data_type=BNSDataType.EXPENDITURE_STRUCTURE,
        iblock_id=469805,  # Household expenses and income
        discovery_url="/en/industries/labor-and-income/stat-life/",
        file_pattern=r"expenditure.*structure|expense|household",
        description="Household expenditure structure (quarterly)",
    ),
    BNSDataType.MINING_SHARES: BNSEndpoint(
        data_type=BNSDataType.MINING_SHARES,
        discovery_url="/en/industries/business-statistics/stat-industry/",
        file_pattern=r"mining|extractive.*industr",
        description="Mining sector shares by region (annual)",
    ),
    BNSDataType.GRP_BY_REGION: BNSEndpoint(
        data_type=BNSDataType.GRP_BY_REGION,
        iblock_id=48510,  # Annual income data
        discovery_url="/en/industries/economy/national-accounts/",
        file_pattern=r"gross.*regional.*product|grp.*region",
        description="GRP by region (annual)",
    ),
    BNSDataType.EMPLOYMENT: BNSEndpoint(
        data_type=BNSDataType.EMPLOYMENT,
        discovery_url="/en/industries/labor-and-income/stat-occupancy/",
        file_pattern=r"employment.*region|labor.*force|employed",
        description="Employment by region (quarterly)",
    ),
    # DAG node dataset mappings
    BNSDataType.HOUSEHOLD_INCOME: BNSEndpoint(
        data_type=BNSDataType.HOUSEHOLD_INCOME,
        discovery_url="/en/industries/labor-and-income/stat-life/",
        file_pattern=r"wage.*income|transfer.*income|monetary.*income|household.*income",
        description="Household income by source (wages, transfers) by region (quarterly)",
    ),
    BNSDataType.HOUSEHOLD_BUDGET: BNSEndpoint(
        data_type=BNSDataType.HOUSEHOLD_BUDGET,
        discovery_url="/en/industries/labor-and-income/stat-life/",
        file_pattern=r"consumption.*expenditure|household.*budget|household.*expense",
        description="Household consumption expenditure by region (quarterly)",
    ),
    BNSDataType.REGIONAL_ACCOUNTS: BNSEndpoint(
        data_type=BNSDataType.REGIONAL_ACCOUNTS,
        discovery_url="/en/industries/economy/national-accounts/",
        file_pattern=r"import.*share|regional.*account|trade.*region",
        description="Regional accounts including import shares (annual)",
    ),
    BNSDataType.PRICES: BNSEndpoint(
        data_type=BNSDataType.PRICES,
        discovery_url="/en/industries/prices/stat-consumer-prices/",
        file_pattern=r"consumer.*price|cpi|price.*index",
        description="Consumer price indices by category (monthly)",
    ),
}


class KazakhstanBNSClient(HTTPDataSource):
    """Client for Kazakhstan Bureau of National Statistics data."""

    @property
    def source_name(self) -> str:
        return "kazakhstan_bns"

    def __init__(self, cache_dir: Path | None = None):
        super().__init__(cache_dir)
        settings = get_settings()
        self.base_url = settings.bns_base_url

    def fetch(self, data_type: BNSDataType, **kwargs: Any) -> pd.DataFrame:
        """
        Fetch data from BNS using three-tier fallback.

        Args:
            data_type: Type of data to fetch

        Returns:
            DataFrame with the requested data
        """
        endpoint = KNOWN_ENDPOINTS.get(data_type)
        if endpoint is None:
            raise ValueError(f"Unknown data type: {data_type}")

        # Tier 1: Try known iblock ID
        if endpoint.iblock_id:
            try:
                return self._fetch_iblock(endpoint.iblock_id)
            except Exception as e:
                logger.warning(f"Iblock fetch failed: {e}, trying next tier")

        # Tier 2: Try known doc ID
        if endpoint.doc_id:
            try:
                return self._fetch_getfile(endpoint.doc_id)
            except Exception as e:
                logger.warning(f"GetFile fetch failed: {e}, trying next tier")

        # Tier 3: Link discovery
        if endpoint.discovery_url:
            return self._fetch_via_discovery(endpoint)

        raise RuntimeError(f"No valid fetch method for {data_type}")

    def _fetch_iblock(self, element_id: int) -> pd.DataFrame:
        """Fetch file directly via iblock API (prefer CSV for structured data)."""
        import io

        # Try CSV endpoint first (better structured)
        csv_url = f"{self.base_url}/api/iblock/element/{element_id}/csv/file/en/"
        logger.info(f"Fetching iblock element {element_id} (CSV)")

        try:
            response = self.client.get(csv_url)
            response.raise_for_status()
            df = pd.read_csv(io.BytesIO(response.content), sep='\t')
            return self._standardize_bns_data(df)
        except Exception as e:
            logger.debug(f"CSV fetch failed: {e}, trying Excel")

        # Fallback to Excel
        url = f"{self.base_url}/api/iblock/element/{element_id}/file/en/"
        logger.info(f"Fetching iblock element {element_id} (Excel)")

        response = self.client.get(url)
        response.raise_for_status()

        return self._parse_excel_response(response.content)

    def _standardize_bns_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize BNS CSV data to common format."""
        # Find region column (contains 'kamalog' in Kazakh)
        region_col = None
        for col in df.columns:
            if 'каталог' in col.lower() or 'region' in col.lower():
                region_col = col
                break

        if region_col is None:
            return df

        # Rename columns
        df = df.rename(columns={
            region_col: 'region_raw',
            'VAL': 'value',
            'PERIOD': 'period',
            'DAT': 'date_raw',
        })

        # Parse value (remove space separator)
        if 'value' in df.columns:
            df['value'] = df['value'].astype(str).str.replace(' ', '').str.replace(',', '.')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Parse period to quarter (YYYYMM -> YYYYQ#)
        if 'period' in df.columns:
            df['period'] = df['period'].astype(str)
            df['year'] = df['period'].str[:4].astype(int)
            df['month'] = df['period'].str[4:6].astype(int)
            df['q'] = ((df['month'] - 1) // 3) + 1
            df['quarter'] = df['year'].astype(str) + 'Q' + df['q'].astype(str)

        # Standardize region names
        df['region'] = df['region_raw'].apply(self._normalize_bns_region)

        # Filter out national totals
        df = df[~df['region_raw'].str.contains('REPUBLIC|KAZAKHSTAN', case=False, na=False)]

        return df

    def _normalize_bns_region(self, region: str) -> str:
        """Normalize BNS region names to canonical form."""
        if pd.isna(region):
            return region

        region = str(region).strip().upper()

        # BNS region name mappings
        bns_mapping = {
            'AKMOLA REGION': 'Akmola',
            'AKTOBE REGION': 'Aktobe',
            'ALMATY REGION': 'Almaty Region',
            'ALMATY CITY': 'Almaty City',
            'ASTANA CITY': 'Astana',
            'ATYRAU REGION': 'Atyrau',
            'BATYS-KAZAKHSTAN REGION': 'West Kazakhstan',
            'BATYS KAZAKHSTAN REGION': 'West Kazakhstan',
            'WEST KAZAKHSTAN REGION': 'West Kazakhstan',
            'ZHAMBYL REGION': 'Jambyl',
            'KARAGANDY REGION': 'Karaganda',
            'KARAGANDA REGION': 'Karaganda',
            'KOSTANAY REGION': 'Kostanay',
            'KYZYLORDA REGION': 'Kyzylorda',
            'MANGYSTAU REGION': 'Mangystau',
            'PAVLODAR REGION': 'Pavlodar',
            'SOLTUSTIK KAZAKHSTAN REGION': 'North Kazakhstan',
            'NORTH KAZAKHSTAN REGION': 'North Kazakhstan',
            'SHYGYS KAZAKHSTAN REGION': 'East Kazakhstan',
            'EAST KAZAKHSTAN REGION': 'East Kazakhstan',
            'SOUTH-KAZAKHSTAN REGION': 'South Kazakhstan',
            'SOUTH KAZAKHSTAN REGION': 'South Kazakhstan',
            'TURKISTAN REGION': 'South Kazakhstan',  # Harmonize
            'SHYMKENT CITY': 'South Kazakhstan',  # Harmonize
            'ABAY REGION': 'East Kazakhstan',  # Harmonize 2022 split
            'ZHETISU REGION': 'Almaty Region',  # Harmonize 2022 split
            'ULYTAU REGION': 'Karaganda',  # Harmonize 2022 split
        }

        return bns_mapping.get(region, region)

    def _fetch_getfile(self, doc_id: str) -> pd.DataFrame:
        """Fetch file via getFile endpoint."""
        url = f"{self.base_url}/api/getFile/"
        params = {"docId": doc_id}
        logger.info(f"Fetching via getFile: {doc_id}")

        response = self.client.get(url, params=params)
        response.raise_for_status()

        return self._parse_excel_response(response.content)

    def _fetch_via_discovery(self, endpoint: BNSEndpoint) -> pd.DataFrame:
        """Discover and fetch data via HTML link parsing."""
        if not endpoint.discovery_url:
            raise ValueError("No discovery URL configured")

        discovery_url = f"{self.base_url}{endpoint.discovery_url}"
        logger.info(f"Discovering links at {discovery_url}")

        response = self.client.get(discovery_url)
        response.raise_for_status()

        # Parse HTML to find data links
        soup = BeautifulSoup(response.text, "html.parser")
        links = self._find_data_links(soup, endpoint.file_pattern)

        if not links:
            raise RuntimeError(f"No data links found at {discovery_url}")

        # Try each discovered link
        for link_url, link_text in links:
            try:
                logger.info(f"Trying discovered link: {link_text}")
                return self._fetch_discovered_link(link_url)
            except Exception as e:
                logger.warning(f"Failed to fetch {link_url}: {e}")
                continue

        raise RuntimeError(f"All discovered links failed for {endpoint.data_type}")

    def _find_data_links(
        self, soup: BeautifulSoup, pattern: str | None
    ) -> list[tuple[str, str]]:
        """Find data download links in BNS page."""
        links = []

        # Look for links to Excel/CSV files
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            text = anchor.get_text(strip=True).lower()

            # Check if it's a data file link
            is_data_link = any(
                [
                    "/api/iblock/" in href,
                    "/api/getFile/" in href,
                    href.endswith((".xlsx", ".xls", ".csv")),
                    "download" in href.lower(),
                ]
            )

            if is_data_link:
                # Apply pattern filter if provided
                if pattern:
                    if not re.search(pattern, text, re.IGNORECASE):
                        continue

                full_url = href if href.startswith("http") else f"{self.base_url}{href}"
                links.append((full_url, text))

        # Sort by relevance (exact matches first)
        return links

    def _fetch_discovered_link(self, url: str) -> pd.DataFrame:
        """Fetch data from a discovered link."""
        response = self.client.get(url)
        response.raise_for_status()
        return self._parse_excel_response(response.content)

    def _parse_excel_response(self, content: bytes) -> pd.DataFrame:
        """Parse Excel/CSV content from response."""
        import io

        # Try Excel first
        try:
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
            return df
        except Exception:
            pass

        # Try older Excel format
        try:
            df = pd.read_excel(io.BytesIO(content), engine="xlrd")
            return df
        except Exception:
            pass

        # Try CSV
        try:
            df = pd.read_csv(io.BytesIO(content))
            return df
        except Exception:
            pass

        raise ValueError("Could not parse response as Excel or CSV")

    def fetch_income_by_region(self) -> pd.DataFrame:
        """Fetch per-capita monetary income by region."""
        return self.fetch_with_cache(data_type=BNSDataType.INCOME_PER_CAPITA)

    def fetch_expenditure_structure(self) -> pd.DataFrame:
        """Fetch household expenditure structure."""
        return self.fetch_with_cache(data_type=BNSDataType.EXPENDITURE_STRUCTURE)

    def fetch_mining_shares(self) -> pd.DataFrame:
        """Fetch mining sector shares by region."""
        return self.fetch_with_cache(data_type=BNSDataType.MINING_SHARES)

    def fetch_grp_by_region(self) -> pd.DataFrame:
        """Fetch GRP by region."""
        return self.fetch_with_cache(data_type=BNSDataType.GRP_BY_REGION)

    def fetch_employment(self) -> pd.DataFrame:
        """Fetch employment by region."""
        return self.fetch_with_cache(data_type=BNSDataType.EMPLOYMENT)

    def fetch_all(self) -> dict[BNSDataType, pd.DataFrame]:
        """Fetch all available data types."""
        results = {}
        for data_type in BNSDataType:
            try:
                results[data_type] = self.fetch_with_cache(data_type=data_type)
                logger.info(f"Successfully fetched {data_type.value}")
            except Exception as e:
                logger.error(f"Failed to fetch {data_type.value}: {e}")
                results[data_type] = pd.DataFrame()
        return results

    def save_all_raw(self) -> dict[BNSDataType, Path]:
        """Fetch and save all raw data."""
        paths = {}
        for data_type, df in self.fetch_all().items():
            if not df.empty:
                path = self.save_raw(df, f"{data_type.value}.parquet")
                paths[data_type] = path
        return paths

    def load_alternative_mining_shares(self) -> pd.DataFrame:
        """
        Load mining shares from alternative sources (USGS, EITI, stat.gov.kz GRP).

        Used when BNS API is unavailable. Data is sourced from:
        - USGS Mineral Industry Reports (2022)
        - EITI Kazakhstan Reports (2020-2021)
        - stat.gov.kz GRP Publications (2023)

        Returns:
            DataFrame with columns: region, mining_share, source, source_url, year, notes

        Raises:
            FileNotFoundError: If alternative sources file not found
        """
        if not MINING_SHARES_FILE.exists():
            raise FileNotFoundError(
                f"Alternative mining shares file not found: {MINING_SHARES_FILE}\n"
                "Please create this file with regional mining shares data.\n"
                "See data/raw/alternative_sources/README.md for format."
            )

        logger.info(f"Loading alternative mining shares from {MINING_SHARES_FILE}")
        df = pd.read_csv(MINING_SHARES_FILE)

        # Validate required columns
        required_cols = ["region", "mining_share", "source"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Alternative mining shares file missing required columns: {missing}"
            )

        logger.info(
            f"Loaded {len(df)} regions from alternative sources. "
            f"Sources: {df['source'].unique().tolist()}"
        )

        return df

    def load_alternative_grp_composition(self) -> pd.DataFrame:
        """
        Load GRP composition from alternative sources.

        Returns:
            DataFrame with regional GRP data including oil sector percentages
        """
        if not GRP_COMPOSITION_FILE.exists():
            raise FileNotFoundError(
                f"Alternative GRP composition file not found: {GRP_COMPOSITION_FILE}"
            )

        logger.info(f"Loading alternative GRP composition from {GRP_COMPOSITION_FILE}")
        df = pd.read_csv(GRP_COMPOSITION_FILE)

        return df

    def fetch_mining_shares_with_fallback(self) -> tuple[pd.DataFrame, str]:
        """
        Try to fetch mining shares from BNS, fall back to alternative sources.

        Returns:
            Tuple of (DataFrame, source_description)
            source_description indicates whether data is from BNS or alternative sources
        """
        # Try BNS first
        try:
            df = self.fetch_mining_shares()
            if not df.empty:
                logger.info("Successfully fetched mining shares from BNS API")
                return df, "bns_api"
        except Exception as e:
            logger.warning(f"BNS mining shares fetch failed: {e}")

        # Fall back to alternative sources
        try:
            alt_df = self.load_alternative_mining_shares()
            logger.info("Using alternative mining shares (USGS/EITI/stat.gov.kz)")

            # Transform to match expected format
            df = alt_df[["region", "mining_share"]].copy()
            df["year"] = alt_df["year"].iloc[0] if "year" in alt_df.columns else 2022

            # Build source description
            sources = alt_df["source"].unique().tolist()
            source_desc = f"alternative_sources: {', '.join(sources)}"

            return df, source_desc

        except FileNotFoundError:
            raise ValueError(
                "CRITICAL: No mining sector data available.\n"
                "BNS API failed and no alternative sources found.\n"
                "\nTo resolve:\n"
                "1. Download USGS Kazakhstan report: https://pubs.usgs.gov/myb/vol3/2022/myb3-2022-kazakhstan.pdf\n"
                "2. Download EITI Kazakhstan report: https://eiti.org/countries/kazakhstan\n"
                "3. Create data/raw/alternative_sources/mining_shares.csv with regional shares\n"
                "4. See data/raw/alternative_sources/README.md for format requirements"
            )
