"""
Import Intensity Calculator for COICOP categories.

Provides import share estimates for CPI categories used in Block A
pass-through analysis. Import intensity (s_c) is the key exposure
variable for the DiD design.

Sources (fallback tiers):
1. BNS Input-Output tables (if available)
2. UN COMTRADE + COICOP mapping
3. Binary tradable/non-tradable classification (default)

Key Assumption: s_c is predetermined (fixed using pre-period data).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from config.settings import get_settings
from shared.data.base import DataSource

logger = logging.getLogger(__name__)


# Default import shares by COICOP category
# Based on typical emerging market import penetration patterns
# and Kazakhstan-specific trade structure
DEFAULT_IMPORT_SHARES: dict[str, float] = {
    # High import intensity (>50%)
    "03": 0.80,  # Clothing and footwear - largely imported
    "02": 0.60,  # Alcohol/tobacco - significant imports
    "09": 0.55,  # Recreation and culture - electronics, imports

    # Medium import intensity (20-50%)
    "01": 0.40,  # Food - partial import dependence
    "05": 0.45,  # Furnishings - mixed domestic/import
    "12": 0.35,  # Miscellaneous - varies widely
    "07": 0.40,  # Transport - vehicles imported, fuel domestic

    # Low import intensity (<20%)
    "04": 0.10,  # Housing/utilities - mostly domestic, admin
    "06": 0.15,  # Health - domestic services + imported pharma
    "08": 0.10,  # Communications - services domestic
    "10": 0.05,  # Education - domestic service
    "11": 0.05,  # Restaurants/hotels - domestic service
}


@dataclass
class ImportIntensityMetadata:
    """Metadata for import intensity data."""

    source: str
    year: int | None = None
    n_categories: int = 12
    method: str = "default"
    notes: str = ""


@dataclass
class ImportIntensityResult:
    """Import intensity result for a category."""

    category: str
    import_share: float
    source: str
    confidence: Literal["high", "medium", "low"]
    year: int | None = None
    notes: str = ""


class ImportIntensityClient(DataSource):
    """
    Import intensity calculator for COICOP categories.

    Provides import share (s_c) estimates for Block A CPI pass-through.
    Uses three-tier fallback:
    1. BNS Input-Output tables
    2. UN COMTRADE mapping
    3. Default binary tradable classification
    """

    @property
    def source_name(self) -> str:
        return "import_intensity"

    def __init__(self, cache_dir: Path | None = None):
        super().__init__(cache_dir)
        settings = get_settings()
        self._metadata: ImportIntensityMetadata | None = None

    def fetch(self, **kwargs: Any) -> pd.DataFrame:
        """
        Fetch import intensity data for COICOP categories.

        Args:
            method: 'io_table', 'comtrade', or 'default'
            year: Reference year for IO tables

        Returns:
            DataFrame with columns:
                - category: COICOP code
                - import_share: Import intensity (0-1)
                - source: Data source
                - confidence: Confidence level
                - tradable: Binary tradable indicator
        """
        method = kwargs.get("method", "auto")
        year = kwargs.get("year", 2019)  # Pre-COVID reference

        if method == "auto":
            # Try tiers in order
            try:
                return self._fetch_io_table(year)
            except Exception as e:
                logger.debug(f"IO table fetch failed: {e}")

            try:
                return self._fetch_comtrade()
            except Exception as e:
                logger.debug(f"COMTRADE fetch failed: {e}")

            # Default fallback
            return self._get_default_shares()

        elif method == "io_table":
            return self._fetch_io_table(year)
        elif method == "comtrade":
            return self._fetch_comtrade()
        elif method == "default":
            return self._get_default_shares()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _fetch_io_table(self, year: int) -> pd.DataFrame:
        """
        Fetch import shares from BNS Input-Output tables.

        IO tables provide the most accurate import penetration by sector,
        which can be mapped to COICOP categories.
        """
        settings = get_settings()

        # Check for local IO table data
        io_path = settings.project_root / "data/raw/kazakhstan_bns/io_tables.parquet"
        if io_path.exists():
            logger.info(f"Loading IO table data from {io_path}")
            io_df = pd.read_parquet(io_path)
            return self._map_io_to_coicop(io_df, year)

        # Try BNS API for IO tables (limited availability)
        # BNS publishes IO tables every 5 years approximately
        raise RuntimeError(
            "IO table data not available locally.\n"
            "Please download from BNS National Accounts section:\n"
            "https://stat.gov.kz/en/industries/economy/national-accounts/"
        )

    def _map_io_to_coicop(
        self,
        io_df: pd.DataFrame,
        year: int,
    ) -> pd.DataFrame:
        """
        Map Input-Output sector import shares to COICOP categories.

        IO table sectors (ISIC-based) must be mapped to COICOP consumption categories.
        """
        # ISIC to COICOP approximate mapping
        isic_to_coicop = {
            "A": ["01"],  # Agriculture -> Food
            "C10-C12": ["01", "02"],  # Food/beverage manufacturing
            "C13-C15": ["03"],  # Textiles -> Clothing
            "C19-C23": ["07"],  # Chemicals, fuel -> Transport
            "C26-C28": ["05", "09"],  # Electronics -> Furnishings, Recreation
            "C29-C30": ["07"],  # Vehicles -> Transport
            "D": ["04"],  # Electricity -> Housing
            "E": ["04"],  # Water -> Housing
            "G": ["01", "03", "12"],  # Retail -> Food, Clothing, Misc
            "H": ["07"],  # Transport services
            "I": ["11"],  # Accommodation -> Hotels/Restaurants
            "J": ["08"],  # Communications
            "Q": ["06"],  # Health
            "P": ["10"],  # Education
            "R": ["09"],  # Recreation services
        }

        # Filter to reference year
        if "year" in io_df.columns:
            io_df = io_df[io_df["year"] == year]

        # Compute import share by sector: imports / (domestic + imports)
        if "imports" in io_df.columns and "domestic" in io_df.columns:
            io_df["import_share"] = (
                io_df["imports"] / (io_df["domestic"] + io_df["imports"])
            )
        elif "import_share" not in io_df.columns:
            raise ValueError("IO table missing required columns")

        # Map to COICOP categories
        records = []
        for coicop_code in DEFAULT_IMPORT_SHARES.keys():
            # Find matching ISIC sectors
            matching_shares = []
            for isic, coicop_list in isic_to_coicop.items():
                if coicop_code in coicop_list:
                    sector_row = io_df[io_df["sector"].str.contains(isic, na=False)]
                    if not sector_row.empty:
                        matching_shares.append(sector_row["import_share"].mean())

            if matching_shares:
                import_share = sum(matching_shares) / len(matching_shares)
                confidence = "high"
            else:
                # Fall back to default
                import_share = DEFAULT_IMPORT_SHARES[coicop_code]
                confidence = "low"

            records.append({
                "category": coicop_code,
                "import_share": import_share,
                "source": "io_table",
                "confidence": confidence,
                "year": year,
            })

        result = pd.DataFrame(records)
        result["tradable"] = result["import_share"] > 0.2

        self._metadata = ImportIntensityMetadata(
            source="BNS IO Tables",
            year=year,
            n_categories=len(result),
            method="io_table",
        )

        return result

    def _fetch_comtrade(self) -> pd.DataFrame:
        """
        Fetch import data from UN COMTRADE and map to COICOP.

        Uses HS codes to estimate import penetration by product category,
        then maps to COICOP divisions.
        """
        # UN COMTRADE API (requires API key for bulk queries)
        # For now, check if we have local COMTRADE extracts

        settings = get_settings()
        comtrade_path = settings.project_root / "data/raw/import_intensity/comtrade_kz.parquet"

        if comtrade_path.exists():
            logger.info(f"Loading COMTRADE data from {comtrade_path}")
            df = pd.read_parquet(comtrade_path)
            return self._process_comtrade(df)

        raise RuntimeError(
            "COMTRADE data not available locally.\n"
            "Please download Kazakhstan import data from:\n"
            "https://comtradeplus.un.org/"
        )

    def _process_comtrade(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process COMTRADE data to COICOP import shares."""
        # HS code to COICOP mapping (simplified)
        hs_to_coicop = {
            "01-24": "01",  # Food products -> Food
            "22": "02",  # Beverages
            "24": "02",  # Tobacco
            "50-63": "03",  # Textiles/clothing
            "64-67": "03",  # Footwear
            "94": "05",  # Furniture
            "30": "06",  # Pharmaceuticals
            "87": "07",  # Vehicles
            "27": "07",  # Fuel (transport)
            "85": "08",  # Electronics (comms)
            "95-97": "09",  # Recreation goods
        }

        # Aggregate imports by mapped COICOP
        records = []
        for coicop_code in DEFAULT_IMPORT_SHARES.keys():
            # Find matching HS chapters
            matching_imports = df[df["coicop_mapped"] == coicop_code]["import_value"].sum()
            matching_domestic = df[df["coicop_mapped"] == coicop_code]["domestic_value"].sum()

            if matching_imports + matching_domestic > 0:
                import_share = matching_imports / (matching_imports + matching_domestic)
                confidence = "medium"
            else:
                import_share = DEFAULT_IMPORT_SHARES[coicop_code]
                confidence = "low"

            records.append({
                "category": coicop_code,
                "import_share": import_share,
                "source": "comtrade",
                "confidence": confidence,
            })

        result = pd.DataFrame(records)
        result["tradable"] = result["import_share"] > 0.2

        return result

    def _get_default_shares(self) -> pd.DataFrame:
        """
        Return default import shares based on typical emerging market patterns.

        These are calibrated estimates - not data-driven.
        Used when IO tables and trade data unavailable.
        """
        records = []

        for category, share in DEFAULT_IMPORT_SHARES.items():
            records.append({
                "category": category,
                "import_share": share,
                "source": "default",
                "confidence": "low",
                "tradable": share > 0.2,
            })

        result = pd.DataFrame(records)

        self._metadata = ImportIntensityMetadata(
            source="Default estimates",
            method="default",
            n_categories=len(result),
            notes="Calibrated estimates based on typical emerging market patterns",
        )

        return result

    def get_category_share(self, category: str) -> float:
        """Get import share for a single category."""
        df = self.fetch_with_cache()
        row = df[df["category"] == category]
        if row.empty:
            return DEFAULT_IMPORT_SHARES.get(category, 0.3)
        return row["import_share"].iloc[0]

    def get_tradable_categories(self) -> list[str]:
        """Get list of tradable (high import intensity) categories."""
        df = self.fetch_with_cache()
        return df[df["tradable"]]["category"].tolist()

    def get_nontradable_categories(self) -> list[str]:
        """Get list of non-tradable categories."""
        df = self.fetch_with_cache()
        return df[~df["tradable"]]["category"].tolist()

    def compute_weighted_share(
        self,
        weights: dict[str, float],
    ) -> float:
        """
        Compute weighted average import share.

        Args:
            weights: Dictionary mapping category code to weight (e.g., CPI weight)

        Returns:
            Weighted average import share
        """
        df = self.fetch_with_cache()
        df = df.set_index("category")

        total_weight = 0
        weighted_sum = 0

        for cat, weight in weights.items():
            if cat in df.index:
                weighted_sum += df.loc[cat, "import_share"] * weight
                total_weight += weight

        if total_weight == 0:
            return 0

        return weighted_sum / total_weight

    def build_exposure_panel(
        self,
        cpi_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build exposure panel for Block A analysis.

        Merges import intensity with CPI category data to create
        the s_c exposure variable for DiD estimation.

        Args:
            cpi_data: CPI panel with category column

        Returns:
            Panel with import_share (s_c) merged in
        """
        import_shares = self.fetch_with_cache()

        # Merge import shares into CPI panel
        result = cpi_data.merge(
            import_shares[["category", "import_share", "tradable"]],
            on="category",
            how="left",
        )

        # Rename for clarity
        result = result.rename(columns={"import_share": "s_c"})

        return result

    def save_to_yaml(self, output_path: Path | None = None) -> Path:
        """
        Save import shares to YAML for documentation.

        Args:
            output_path: Output file path

        Returns:
            Path to saved file
        """
        import yaml

        settings = get_settings()
        if output_path is None:
            output_path = settings.project_root / "data/metadata/import_shares.yaml"

        df = self.fetch_with_cache()

        data = {
            "version": "1.0",
            "source": self._metadata.source if self._metadata else "default",
            "method": self._metadata.method if self._metadata else "default",
            "categories": {},
        }

        for _, row in df.iterrows():
            data["categories"][row["category"]] = {
                "import_share": float(row["import_share"]),
                "tradable": bool(row["tradable"]),
                "confidence": row.get("confidence", "low"),
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        return output_path

    def save_all_raw(self) -> Path:
        """Fetch and save import intensity data."""
        df = self.fetch_with_cache()
        return self.save_raw(df, "coicop_shares.parquet")
