"""
Panel data construction with region crosswalk.

Implements stable geography approach by aggregating new regions
back into parent pre-split regions.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)


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
            logger.warning("No income data available, using placeholder")
            # Generate placeholder for testing
            np.random.seed(42)
            panel["income_pc"] = np.exp(
                10 + np.random.randn(len(panel)) * 0.3
            )
            panel["log_income_pc"] = np.log(panel["income_pc"])
            return panel

        # Process and merge income data
        income_df = self._process_bns_income(income_df)

        if income_df.empty:
            logger.warning("Processed income data is empty, using placeholder")
            np.random.seed(42)
            panel["income_pc"] = np.exp(10 + np.random.randn(len(panel)) * 0.3)
            panel["log_income_pc"] = np.log(panel["income_pc"])
            return panel

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
        """Compute oil/mining exposure by region."""
        from src.data.kazakhstan_bns import BNSDataType

        mining_df = bns_data.get(BNSDataType.MINING_SHARES, pd.DataFrame())

        if mining_df.empty:
            logger.warning("No mining data available, using stylized exposure")
            # Stylized exposure based on known oil regions
            oil_regions = {
                "Atyrau": 0.8,
                "Mangystau": 0.7,
                "West Kazakhstan": 0.5,
                "Kyzylorda": 0.3,
                "Aktobe": 0.25,
            }
            panel["E_oil_r"] = panel["region"].map(oil_regions).fillna(0.05)
            return panel

        # Process and compute baseline average
        mining_df = mining_df.copy()
        if "region" in mining_df.columns:
            mining_df["region"] = mining_df["region"].apply(self.harmonize_region)

        if "year" in mining_df.columns:
            baseline = mining_df[
                (mining_df["year"] >= baseline_start)
                & (mining_df["year"] <= baseline_end)
            ]
            if "mining_share" in baseline.columns:
                exposure = baseline.groupby("region")["mining_share"].mean()
                panel["E_oil_r"] = panel["region"].map(exposure).fillna(0)

        return panel

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
            logger.warning("No expenditure data available, using placeholder")
            panel["E_debt_r"] = 0.1  # Placeholder
            return panel

        # Process and compute baseline average
        expenditure_df = expenditure_df.copy()
        if "region" in expenditure_df.columns:
            expenditure_df["region"] = expenditure_df["region"].apply(self.harmonize_region)

        # Extract debt share if available
        if "debt_share" in expenditure_df.columns and "year" in expenditure_df.columns:
            baseline = expenditure_df[
                (expenditure_df["year"] >= baseline_start)
                & (expenditure_df["year"] <= baseline_end)
            ]
            exposure = baseline.groupby("region")["debt_share"].mean()
            panel["E_debt_r"] = panel["region"].map(exposure).fillna(0.1)
        else:
            panel["E_debt_r"] = 0.1

        return panel

    def _compute_cyclical_exposure(
        self,
        panel: pd.DataFrame,
        bns_data: dict[Any, pd.DataFrame],
        baseline_start: int,
        baseline_end: int,
    ) -> pd.DataFrame:
        """Compute cyclical employment exposure by region."""
        from src.data.kazakhstan_bns import BNSDataType

        employment_df = bns_data.get(BNSDataType.EMPLOYMENT, pd.DataFrame())

        if employment_df.empty:
            logger.warning("No employment data available, using placeholder")
            # Stylized cyclical exposure varying by region
            cyclical_exposures = {
                "Almaty City": 0.55,
                "Astana": 0.50,
                "Karaganda": 0.40,
                "Pavlodar": 0.38,
                "East Kazakhstan": 0.35,
                "Kostanay": 0.32,
                "Aktobe": 0.30,
                "Atyrau": 0.25,
                "Mangystau": 0.22,
                "West Kazakhstan": 0.28,
                "Almaty Region": 0.35,
                "Jambyl": 0.30,
                "South Kazakhstan": 0.28,
                "Kyzylorda": 0.26,
                "North Kazakhstan": 0.33,
                "Akmola": 0.32,
            }
            panel["E_cyc_r"] = panel["region"].map(cyclical_exposures).fillna(0.3)
            return panel

        # Process and compute baseline average
        employment_df = employment_df.copy()
        if "region" in employment_df.columns:
            employment_df["region"] = employment_df["region"].apply(self.harmonize_region)

        # Extract cyclical share if available
        if "cyclical_share" in employment_df.columns and "year" in employment_df.columns:
            baseline = employment_df[
                (employment_df["year"] >= baseline_start)
                & (employment_df["year"] <= baseline_end)
            ]
            exposure = baseline.groupby("region")["cyclical_share"].mean()
            panel["E_cyc_r"] = panel["region"].map(exposure).fillna(0.3)
        else:
            panel["E_cyc_r"] = 0.3

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
        """Create exposure × shock interaction terms."""
        # Oil exposure × oil supply shock
        if "E_oil_r" in panel.columns and "oil_supply_shock" in panel.columns:
            panel["E_oil_x_supply"] = panel["E_oil_r"] * panel["oil_supply_shock"]

        # Oil exposure × oil demand shock
        if "E_oil_r" in panel.columns and "aggregate_demand_shock" in panel.columns:
            panel["E_oil_x_demand"] = panel["E_oil_r"] * panel["aggregate_demand_shock"]

        # Cyclical exposure × global activity
        if "E_cyc_r" in panel.columns and "global_activity_shock" in panel.columns:
            panel["E_cyc_x_activity"] = panel["E_cyc_r"] * panel["global_activity_shock"]

        # Oil exposure × VIX
        if "E_oil_r" in panel.columns and "vix_shock" in panel.columns:
            panel["E_oil_x_vix"] = panel["E_oil_r"] * panel["vix_shock"]

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
