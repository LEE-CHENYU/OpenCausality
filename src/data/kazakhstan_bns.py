"""
Kazakhstan Bureau of National Statistics (BNS) data client.

Three-tier fallback strategy:
1. Known iblock element IDs → direct file download
2. Known docIds → getFile endpoint
3. HTML link discovery → parse BNS pages to find correct iblock links
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
from src.data.base import HTTPDataSource, DataSourceMetadata

logger = logging.getLogger(__name__)


class BNSDataType(Enum):
    """Types of BNS data we need."""

    INCOME_PER_CAPITA = "income_per_capita"
    EXPENDITURE_STRUCTURE = "expenditure_structure"
    MINING_SHARES = "mining_shares"
    GRP_BY_REGION = "grp_by_region"
    EMPLOYMENT = "employment"


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
        discovery_url="/official/industry/14/statistic/5",
        file_pattern=r"monetary.*income.*region",
        description="Per-capita monetary income by region (quarterly)",
    ),
    BNSDataType.EXPENDITURE_STRUCTURE: BNSEndpoint(
        data_type=BNSDataType.EXPENDITURE_STRUCTURE,
        discovery_url="/official/industry/14/statistic/5",
        file_pattern=r"expenditure.*structure|structure.*expenditure",
        description="Household expenditure structure (quarterly)",
    ),
    BNSDataType.MINING_SHARES: BNSEndpoint(
        data_type=BNSDataType.MINING_SHARES,
        discovery_url="/official/industry/11/statistic/7",
        file_pattern=r"mining|extractive.*industr",
        description="Mining sector shares by region (annual)",
    ),
    BNSDataType.GRP_BY_REGION: BNSEndpoint(
        data_type=BNSDataType.GRP_BY_REGION,
        discovery_url="/official/industry/11/statistic/6",
        file_pattern=r"gross.*regional.*product|grp.*region",
        description="GRP by region (annual)",
    ),
    BNSDataType.EMPLOYMENT: BNSEndpoint(
        data_type=BNSDataType.EMPLOYMENT,
        discovery_url="/official/industry/13/statistic/8",
        file_pattern=r"employment.*region|labor.*force",
        description="Employment by region (quarterly)",
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
        """Fetch file directly via iblock API."""
        url = f"{self.base_url}/api/iblock/element/{element_id}/file/en/"
        logger.info(f"Fetching iblock element {element_id}")

        response = self.client.get(url)
        response.raise_for_status()

        return self._parse_excel_response(response.content)

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
