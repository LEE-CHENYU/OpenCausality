"""
Panel data construction with region crosswalk.

Implements stable geography approach by aggregating new regions
back into parent pre-split regions.

Supports fallback to alternative data sources when BNS API is unavailable.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.data.data_lineage import (
    record_source,
    DataSourceStatus,
    DataQualityLevel,
)

logger = logging.getLogger(__name__)

# Alternative data source paths
ALTERNATIVE_SOURCES_DIR = Path("data/raw/alternative_sources")
MINING_SHARES_FILE = ALTERNATIVE_SOURCES_DIR / "mining_shares.csv"
CYCLICAL_PROXY_FILE = ALTERNATIVE_SOURCES_DIR / "cyclical_proxy.csv"


# Region crosswalk for stable geography
# Map new regions to their parent pre-split regions
REGION_CROSSWALK = {
    # 2022 splits
    "Abay": "East Kazakhstan",
    "Zhetysu": "Almaty Region",
    "Ulytau": "Karaganda",
    # 2018 reform
    "Turkestan": "South Kazakhstan",
    "Shymkent": "South Kazakhstan",
}

# Alternate spellings and transliterations
REGION_NAME_NORMALIZATION = {
    # Kazakh/Russian spellings
    "Almaty region": "Almaty Region",
    "Almaty oblast": "Almaty Region",
    "Алматинская область": "Almaty Region",
    "East Kazakhstan region": "East Kazakhstan",
    "East Kazakhstan oblast": "East Kazakhstan",
    "Восточно-Казахстанская область": "East Kazakhstan",
    "Karaganda region": "Karaganda",
    "Karaganda oblast": "Karaganda",
    "Карагандинская область": "Karaganda",
    "South Kazakhstan region": "South Kazakhstan",
    "South Kazakhstan oblast": "South Kazakhstan",
    "Южно-Казахстанская область": "South Kazakhstan",
    # Cities of republican significance
    "Almaty city": "Almaty City",
    "Almaty": "Almaty City",
    "г. Алматы": "Almaty City",
    "Astana city": "Astana",
    "Astana": "Astana",
    "Nur-Sultan": "Astana",
    "г. Астана": "Astana",
    "г. Нур-Султан": "Astana",
    # Other regions
    "Akmola region": "Akmola",
    "Akmola oblast": "Akmola",
    "Aktobe region": "Aktobe",
    "Aktobe oblast": "Aktobe",
    "Atyrau region": "Atyrau",
    "Atyrau oblast": "Atyrau",
    "West Kazakhstan region": "West Kazakhstan",
    "West Kazakhstan oblast": "West Kazakhstan",
    "Jambyl region": "Jambyl",
    "Jambyl oblast": "Jambyl",
    "Zhambyl region": "Jambyl",
    "Kostanay region": "Kostanay",
    "Kostanay oblast": "Kostanay",
    "Kyzylorda region": "Kyzylorda",
    "Kyzylorda oblast": "Kyzylorda",
    "Mangystau region": "Mangystau",
    "Mangystau oblast": "Mangystau",
    "North Kazakhstan region": "North Kazakhstan",
    "North Kazakhstan oblast": "North Kazakhstan",
    "Pavlodar region": "Pavlodar",
    "Pavlodar oblast": "Pavlodar",
}

# Canonical region list (post-harmonization)
CANONICAL_REGIONS = [
    "Akmola",
    "Aktobe",
    "Almaty City",
    "Almaty Region",
    "Astana",
    "Atyrau",
    "East Kazakhstan",
    "Jambyl",
    "Karaganda",
    "Kostanay",
    "Kyzylorda",
    "Mangystau",
    "North Kazakhstan",
    "Pavlodar",
    "South Kazakhstan",
    "West Kazakhstan",
]


@dataclass
class ExposureConfig:
    """Configuration for computing an exposure variable."""

    name: str
    source_column: str
    baseline_start: int
    baseline_end: int
    aggregation: str = "mean"


class PanelBuilder:
    """Builds analysis panel from raw data sources."""

    def __init__(self):
        settings = get_settings()
        self.settings = settings

    def normalize_region_name(self, region: str) -> str:
        """Normalize region name to canonical form."""
        if pd.isna(region):
            return region

        region = str(region).strip()

        # Check normalization map
        if region in REGION_NAME_NORMALIZATION:
            return REGION_NAME_NORMALIZATION[region]

        # Try case-insensitive match
        for old, new in REGION_NAME_NORMALIZATION.items():
            if old.lower() == region.lower():
                return new

        return region

    def harmonize_region(self, region: str) -> str:
        """Map region to stable geography (pre-split parent)."""
        normalized = self.normalize_region_name(region)

        if normalized in REGION_CROSSWALK:
            return REGION_CROSSWALK[normalized]

        return normalized

    def create_quarter_id(self, year: int, quarter: int) -> str:
        """Create quarter identifier string."""
        return f"{year}Q{quarter}"

    def parse_quarter_id(self, quarter_id: str) -> tuple[int, int]:
        """Parse quarter identifier to (year, quarter)."""
        year = int(quarter_id[:4])
        quarter = int(quarter_id[-1])
        return year, quarter

    def build(
        self,
        bns_data: dict[Any, pd.DataFrame],
        fred_data: dict[str, pd.DataFrame],
        baumeister_data: pd.DataFrame,
        start_year: int = 2010,
        end_year: int = 2024,
    ) -> pd.DataFrame:
        """
        Build the analysis panel.

        Args:
            bns_data: Dictionary of BNS DataFrames
            fred_data: Dictionary of FRED DataFrames
            baumeister_data: Baumeister shocks DataFrame
            start_year: Panel start year
            end_year: Panel end year

        Returns:
            Panel DataFrame
        """
        # Create panel skeleton
        panel = self._create_skeleton(start_year, end_year)

        # Add outcome variable (income)
        panel = self._add_income(panel, bns_data)

        # Add exposure variables
        panel = self._add_exposures(panel, bns_data)

        # Add shock variables
        panel = self._add_shocks(panel, fred_data, baumeister_data)

        # Create interaction terms
        panel = self._create_interactions(panel)

        return panel

    def _create_skeleton(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Create panel skeleton with all region-quarter combinations."""
        quarters = []
        for year in range(start_year, end_year + 1):
            for q in range(1, 5):
                quarters.append(self.create_quarter_id(year, q))

        # Create all combinations
        rows = []
        for region in CANONICAL_REGIONS:
            for quarter in quarters:
                year, q = self.parse_quarter_id(quarter)
                rows.append(
                    {
                        "region": region,
                        "quarter": quarter,
                        "year": year,
                        "q": q,
                    }
                )

        panel = pd.DataFrame(rows)

        # Create numeric identifiers for fixed effects
        panel["region_id"] = pd.Categorical(panel["region"]).codes
        panel["quarter_id"] = pd.Categorical(panel["quarter"]).codes

        return panel

    def _add_income(
        self, panel: pd.DataFrame, bns_data: dict[Any, pd.DataFrame]
    ) -> pd.DataFrame:
        """Add income outcome variable."""
        from src.data.kazakhstan_bns import BNSDataType

        income_df = bns_data.get(BNSDataType.INCOME_PER_CAPITA, pd.DataFrame())

        if income_df.empty:
            raise ValueError(
                "CRITICAL: No income data available. "
                "Cannot proceed without real BNS income data. "
                "Run 'kzwelfare fetch-data bns' to download data, or check BNS API status."
            )

        # Process and merge income data
        income_df = self._process_bns_income(income_df)

        if income_df.empty:
            raise ValueError(
                "CRITICAL: Processed income data is empty after cleaning. "
                "Check BNS data format or column names. "
                f"Raw data had columns: {list(bns_data.get('INCOME_PER_CAPITA', pd.DataFrame()).columns)}"
            )

        panel = panel.merge(income_df, on=["region", "quarter"], how="left")

        # Compute log income
        if "income_pc" in panel.columns:
            panel["log_income_pc"] = np.log(panel["income_pc"].clip(lower=1))

        return panel

    def _process_bns_income(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process BNS income data for panel merge."""
        df = df.copy()

        # Check for required columns (from standardized BNS data)
        if "region" not in df.columns or "quarter" not in df.columns:
            logger.warning(f"BNS income data missing required columns. Found: {list(df.columns)}")
            return pd.DataFrame()

        # Rename value column to income_pc
        if "value" in df.columns:
            df["income_pc"] = df["value"]

        if "income_pc" not in df.columns:
            logger.warning("No income value column found in BNS data")
            return pd.DataFrame()

        # Apply region harmonization
        df["region"] = df["region"].apply(self.harmonize_region)

        # Aggregate split regions (mean income for regions that were combined)
        df = (
            df.groupby(["region", "quarter"])
            .agg({"income_pc": "mean"})  # Mean for combined regions
                .reset_index()
            )

        return df

    def _add_exposures(
        self, panel: pd.DataFrame, bns_data: dict[Any, pd.DataFrame]
    ) -> pd.DataFrame:
        """Add pre-period exposure variables."""
        settings = get_settings()
        baseline_start = settings.baseline_start_year
        baseline_end = settings.baseline_end_year

        # Oil exposure: mining share
        panel = self._compute_oil_exposure(panel, bns_data, baseline_start, baseline_end)

        # Debt exposure: debt repayment share
        panel = self._compute_debt_exposure(panel, bns_data, baseline_start, baseline_end)

        # Cyclical exposure: cyclical employment share
        panel = self._compute_cyclical_exposure(panel, bns_data, baseline_start, baseline_end)

        return panel

    def _compute_oil_exposure(
        self,
        panel: pd.DataFrame,
        bns_data: dict[Any, pd.DataFrame],
        baseline_start: int,
        baseline_end: int,
    ) -> pd.DataFrame:
        """
        Compute oil/mining exposure by region.

        First tries BNS data, then falls back to alternative sources
        (USGS/EITI/stat.gov.kz GRP publications).
        """
        from src.data.kazakhstan_bns import BNSDataType

        mining_df = bns_data.get(BNSDataType.MINING_SHARES, pd.DataFrame())
        data_source = "bns_api"

        # If BNS data unavailable, try alternative sources
        if mining_df.empty:
            logger.warning("BNS mining data unavailable, trying alternative sources")
            mining_df, data_source = self._load_alternative_mining_shares()

        if mining_df.empty:
            raise ValueError(
                "CRITICAL: No mining sector data available from any source. "
                "Cannot compute oil exposure (E_oil_r) without real regional mining share data. "
                "The shift-share identification requires measured exposures, not hardcoded values. "
                "\nOptions:\n"
                "1. Download USGS report: https://pubs.usgs.gov/myb/vol3/2022/myb3-2022-kazakhstan.pdf\n"
                "2. Download EITI report: https://eiti.org/countries/kazakhstan\n"
                "3. Create data/raw/alternative_sources/mining_shares.csv\n"
                "See data/raw/alternative_sources/README.md for format."
            )

        # Process and compute exposure
        mining_df = mining_df.copy()
        if "region" in mining_df.columns:
            mining_df["region"] = mining_df["region"].apply(self.harmonize_region)

        # For alternative sources, use mining_share directly (already a share)
        if "mining_share" in mining_df.columns:
            if "year" in mining_df.columns:
                # Use baseline period if year data available
                baseline = mining_df[
                    (mining_df["year"] >= baseline_start)
                    & (mining_df["year"] <= baseline_end)
                ]
                if not baseline.empty:
                    exposure = baseline.groupby("region")["mining_share"].mean()
                else:
                    # Use all available data as structural approximation
                    logger.info(
                        f"No data in baseline period ({baseline_start}-{baseline_end}), "
                        "using all available data as structural approximation"
                    )
                    exposure = mining_df.groupby("region")["mining_share"].mean()
            else:
                # No year column - use as static exposure
                exposure = mining_df.set_index("region")["mining_share"]

            panel["E_oil_r"] = panel["region"].map(exposure).fillna(0)

            # Record data lineage
            record_source(
                source_name="E_oil_r (oil exposure)",
                status=DataSourceStatus.REAL if data_source == "bns_api" else DataSourceStatus.CACHED,
                quality=DataQualityLevel.GOOD if data_source == "bns_api" else DataQualityLevel.FAIR,
                rows=len(mining_df),
                notes=[
                    f"Data source: {data_source}",
                    f"Regions with data: {len(exposure)}",
                    f"Regions in panel: {panel['region'].nunique()}",
                ],
            )

            logger.info(
                f"Computed E_oil_r from {data_source}. "
                f"Range: [{panel['E_oil_r'].min():.3f}, {panel['E_oil_r'].max():.3f}]"
            )

        return panel

    def _load_alternative_mining_shares(self) -> tuple[pd.DataFrame, str]:
        """
        Load mining shares from alternative sources.

        Returns:
            Tuple of (DataFrame, source_description)
        """
        if not MINING_SHARES_FILE.exists():
            logger.warning(f"Alternative mining shares file not found: {MINING_SHARES_FILE}")
            return pd.DataFrame(), ""

        logger.info(f"Loading alternative mining shares from {MINING_SHARES_FILE}")
        df = pd.read_csv(MINING_SHARES_FILE)

        # Validate required columns
        if "region" not in df.columns or "mining_share" not in df.columns:
            logger.error("Alternative mining shares file missing required columns")
            return pd.DataFrame(), ""

        # Build source description
        sources = df["source"].unique().tolist() if "source" in df.columns else ["unknown"]
        source_desc = f"alternative_sources: {', '.join(sources)}"

        logger.info(
            f"Loaded {len(df)} regions from alternative sources. "
            f"Sources: {sources}"
        )

        return df, source_desc

    def _compute_debt_exposure(
        self,
        panel: pd.DataFrame,
        bns_data: dict[Any, pd.DataFrame],
        baseline_start: int,
        baseline_end: int,
    ) -> pd.DataFrame:
        """Compute debt repayment exposure by region."""
        from src.data.kazakhstan_bns import BNSDataType

        expenditure_df = bns_data.get(BNSDataType.EXPENDITURE_STRUCTURE, pd.DataFrame())

        if expenditure_df.empty:
            raise ValueError(
                "CRITICAL: No expenditure data available. "
                "Cannot compute debt exposure (E_debt_r) without real household expenditure data. "
                "Run 'kzwelfare fetch-data bns' or check BNS API status."
            )

        # Process and compute baseline average
        expenditure_df = expenditure_df.copy()
        if "region" in expenditure_df.columns:
            expenditure_df["region"] = expenditure_df["region"].apply(self.harmonize_region)

        # Extract debt share if available
        if "debt_share" not in expenditure_df.columns:
            raise ValueError(
                f"CRITICAL: Expenditure data missing 'debt_share' column. "
                f"Available columns: {list(expenditure_df.columns)}. "
                "Cannot compute debt exposure without debt repayment share data."
            )
        if "year" not in expenditure_df.columns:
            raise ValueError(
                f"CRITICAL: Expenditure data missing 'year' column. "
                f"Available columns: {list(expenditure_df.columns)}."
            )

        baseline = expenditure_df[
            (expenditure_df["year"] >= baseline_start)
            & (expenditure_df["year"] <= baseline_end)
        ]
        if baseline.empty:
            raise ValueError(
                f"CRITICAL: No expenditure data in baseline period ({baseline_start}-{baseline_end}). "
                f"Data year range: {expenditure_df['year'].min()}-{expenditure_df['year'].max()}"
            )
        exposure = baseline.groupby("region")["debt_share"].mean()
        panel["E_debt_r"] = panel["region"].map(exposure)

        # Check for missing regions
        missing = panel[panel["E_debt_r"].isna()]["region"].unique()
        if len(missing) > 0:
            raise ValueError(
                f"CRITICAL: Debt exposure missing for regions: {list(missing)}. "
                "Cannot proceed with incomplete exposure data."
            )

        return panel

    def _compute_cyclical_exposure(
        self,
        panel: pd.DataFrame,
        bns_data: dict[Any, pd.DataFrame],
        baseline_start: int,
        baseline_end: int,
    ) -> pd.DataFrame:
        """
        Compute cyclical employment exposure by region.

        STUDY DESIGN (v4):
        - Main spec uses oil exposure only (E_oil_r)
        - Cyclical exposure is OPTIONAL, used for robustness checks
        - If BNS employment data unavailable, uses GRP-based proxy

        Returns panel with:
        - E_cyc_r: True cyclical exposure (if BNS data available)
        - E_cyc_proxy_r: GRP-based proxy (for robustness checks)
        """
        from src.data.kazakhstan_bns import BNSDataType

        employment_df = bns_data.get(BNSDataType.EMPLOYMENT, pd.DataFrame())

        # Try BNS employment data first
        if not employment_df.empty:
            employment_df = employment_df.copy()
            if "region" in employment_df.columns:
                employment_df["region"] = employment_df["region"].apply(self.harmonize_region)

            if "cyclical_share" in employment_df.columns and "year" in employment_df.columns:
                baseline = employment_df[
                    (employment_df["year"] >= baseline_start)
                    & (employment_df["year"] <= baseline_end)
                ]
                if not baseline.empty:
                    exposure = baseline.groupby("region")["cyclical_share"].mean()
                    panel["E_cyc_r"] = panel["region"].map(exposure)
                    logger.info("Computed E_cyc_r from BNS employment data")

        # Always add GRP-based proxy for robustness checks
        panel = self._add_cyclical_proxy(panel)

        # Log status
        if "E_cyc_r" not in panel.columns:
            logger.warning(
                "No BNS employment data available. "
                "Main spec uses oil exposure only (E_oil_r). "
                "E_cyc_proxy_r available for robustness checks."
            )

        return panel

    def _add_cyclical_proxy(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Add GRP-based cyclical proxy for robustness checks.

        This is NOT true cyclical employment exposure. Use only to verify
        that the oil exposure coefficient (β) is stable when controlling
        for (noisy) cyclical exposure.
        """
        if not CYCLICAL_PROXY_FILE.exists():
            logger.warning(f"Cyclical proxy file not found: {CYCLICAL_PROXY_FILE}")
            return panel

        logger.info(f"Loading cyclical proxy from {CYCLICAL_PROXY_FILE}")
        proxy_df = pd.read_csv(CYCLICAL_PROXY_FILE)

        if "region" not in proxy_df.columns or "cyclical_proxy" not in proxy_df.columns:
            logger.error("Cyclical proxy file missing required columns")
            return panel

        # Harmonize region names
        proxy_df["region"] = proxy_df["region"].apply(self.harmonize_region)

        # Map to panel
        proxy_map = proxy_df.set_index("region")["cyclical_proxy"]
        panel["E_cyc_proxy_r"] = panel["region"].map(proxy_map).fillna(0.5)

        # Record data lineage
        record_source(
            source_name="E_cyc_proxy_r (cyclical proxy)",
            status=DataSourceStatus.CACHED,
            quality=DataQualityLevel.FAIR,
            rows=len(proxy_df),
            notes=[
                "GRP-based proxy, NOT true employment data",
                "Use for robustness checks only",
                "Source: stat.gov.kz GRP + structural estimates",
            ],
        )

        logger.info(
            f"Added E_cyc_proxy_r (GRP-based). "
            f"Range: [{panel['E_cyc_proxy_r'].min():.2f}, {panel['E_cyc_proxy_r'].max():.2f}]"
        )

        return panel

    def _add_shocks(
        self,
        panel: pd.DataFrame,
        fred_data: dict[str, pd.DataFrame],
        baumeister_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add shock time series."""
        # Add Baumeister structural shocks
        if not baumeister_data.empty:
            baumeister_data = baumeister_data.copy()

            # Ensure quarter column exists and is string type
            if "quarter" in baumeister_data.columns:
                baumeister_data["quarter"] = baumeister_data["quarter"].astype(str)
            elif "date" in baumeister_data.columns:
                # Convert date to quarter string
                baumeister_data["quarter"] = (
                    baumeister_data["date"].dt.year.astype(str)
                    + "Q"
                    + baumeister_data["date"].dt.quarter.astype(str)
                )

            shock_cols = [c for c in baumeister_data.columns if "shock" in c.lower()]
            if shock_cols:
                # Aggregate to unique quarters (in case of duplicates)
                merge_data = baumeister_data.groupby("quarter")[shock_cols].mean().reset_index()
                panel = panel.merge(merge_data, on="quarter", how="left")

        # Add FRED-based shocks
        for name, df in fred_data.items():
            if df.empty:
                continue

            df = df.copy()

            # Convert date to quarter
            if "date" in df.columns:
                df["quarter"] = (
                    df["date"].dt.year.astype(str)
                    + "Q"
                    + df["date"].dt.quarter.astype(str)
                )

            # Get the value column
            if "innovation" in df.columns:
                value_col = "innovation"
            elif "log_return" in df.columns:
                value_col = "log_return"
            elif "value" in df.columns:
                value_col = "value"
            else:
                continue

            # Rename and merge
            shock_name = f"{name}_shock"
            merge_df = df[["quarter", value_col]].rename(columns={value_col: shock_name})
            panel = panel.merge(merge_df, on="quarter", how="left")

        return panel

    def _create_interactions(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Create exposure × shock interaction terms.

        Main spec (v4): Uses E_oil_r × oil shocks only
        Robustness: Adds E_cyc_proxy_r × global activity
        """
        # === MAIN SPECIFICATION: Oil exposure interactions ===

        # Oil exposure × oil supply shock (PRIMARY)
        if "E_oil_r" in panel.columns and "oil_supply_shock" in panel.columns:
            panel["E_oil_x_supply"] = panel["E_oil_r"] * panel["oil_supply_shock"]

        # Oil exposure × oil demand shock
        if "E_oil_r" in panel.columns and "aggregate_demand_shock" in panel.columns:
            panel["E_oil_x_demand"] = panel["E_oil_r"] * panel["aggregate_demand_shock"]

        # Oil exposure × VIX
        if "E_oil_r" in panel.columns and "vix_shock" in panel.columns:
            panel["E_oil_x_vix"] = panel["E_oil_r"] * panel["vix_shock"]

        # === ROBUSTNESS: Cyclical exposure interactions ===

        # True cyclical exposure × global activity (if BNS data available)
        if "E_cyc_r" in panel.columns and "global_activity_shock" in panel.columns:
            panel["E_cyc_x_activity"] = panel["E_cyc_r"] * panel["global_activity_shock"]

        # Cyclical PROXY × global activity (for robustness checks)
        if "E_cyc_proxy_r" in panel.columns and "global_activity_shock" in panel.columns:
            panel["E_cyc_proxy_x_activity"] = panel["E_cyc_proxy_r"] * panel["global_activity_shock"]

        return panel


def test_region_harmonization():
    """Test that region harmonization works correctly."""
    builder = PanelBuilder()

    # Test crosswalk
    assert builder.harmonize_region("Abay") == "East Kazakhstan"
    assert builder.harmonize_region("Zhetysu") == "Almaty Region"
    assert builder.harmonize_region("Ulytau") == "Karaganda"
    assert builder.harmonize_region("Turkestan") == "South Kazakhstan"
    assert builder.harmonize_region("Shymkent") == "South Kazakhstan"

    # Test that canonical regions are unchanged
    for region in CANONICAL_REGIONS:
        assert builder.harmonize_region(region) == region

    print("Region harmonization tests passed!")


if __name__ == "__main__":
    test_region_harmonization()
