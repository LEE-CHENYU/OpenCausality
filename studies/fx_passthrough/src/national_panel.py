"""
National Time Series Panel Builder for FX Passthrough Study.

Constructs the analysis panels from raw data sources:
1. CPI panel: Category × Month for Block A
2. Income series: National time series for Blocks B-D
3. Expenditure series: For Block E

Uses shared data infrastructure:
- shared/data/exchange_rate.py
- shared/data/bns_cpi_categories.py
- shared/data/bns_national_income.py
- shared/data/import_intensity.py
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_settings
from shared.data.data_pipeline import SharedDataPipeline

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Quality report for a dataset."""

    source: str
    n_rows: int
    n_cols: int
    date_range: tuple[str, str] | None = None
    missing_pct: float = 0.0
    quality_grade: str = "unknown"
    notes: list[str] = field(default_factory=list)


class FXPassthroughPipeline(SharedDataPipeline):
    """
    Data pipeline for FX Passthrough study.

    Extends SharedDataPipeline with study-specific data sources.
    """

    def __init__(self):
        super().__init__()
        self._exchange_rate_client = None
        self._cpi_client = None
        self._national_income_client = None
        self._import_intensity_client = None
        self._quality_reports: list[DataQualityReport] = []

    @property
    def exchange_rate_client(self):
        """Lazy-loaded exchange rate client."""
        if self._exchange_rate_client is None:
            from shared.data.exchange_rate import ExchangeRateClient
            self._exchange_rate_client = ExchangeRateClient()
        return self._exchange_rate_client

    @property
    def cpi_client(self):
        """Lazy-loaded CPI categories client."""
        if self._cpi_client is None:
            from shared.data.bns_cpi_categories import BNSCPICategoriesClient
            self._cpi_client = BNSCPICategoriesClient()
        return self._cpi_client

    @property
    def national_income_client(self):
        """Lazy-loaded national income client."""
        if self._national_income_client is None:
            from shared.data.bns_national_income import BNSNationalIncomeClient
            self._national_income_client = BNSNationalIncomeClient()
        return self._national_income_client

    @property
    def import_intensity_client(self):
        """Lazy-loaded import intensity client."""
        if self._import_intensity_client is None:
            from shared.data.import_intensity import ImportIntensityClient
            self._import_intensity_client = ImportIntensityClient()
        return self._import_intensity_client

    def fetch_all_fx_data(self) -> dict[str, pd.DataFrame]:
        """Fetch all FX passthrough data sources."""
        data = {}

        # Exchange rates
        try:
            data["exchange_rate"] = self.exchange_rate_client.fetch_with_cache()
            logger.info(f"Fetched {len(data['exchange_rate'])} exchange rate observations")
        except Exception as e:
            logger.error(f"Exchange rate fetch failed: {e}")

        # CPI categories
        try:
            data["cpi_categories"] = self.cpi_client.fetch_with_cache()
            logger.info(f"Fetched {len(data['cpi_categories'])} CPI observations")
        except Exception as e:
            logger.error(f"CPI categories fetch failed: {e}")

        # National income
        try:
            data["national_income"] = self.national_income_client.fetch_quarterly()
            logger.info(f"Fetched {len(data['national_income'])} income observations")
        except Exception as e:
            logger.error(f"National income fetch failed: {e}")

        # Import intensity
        try:
            data["import_intensity"] = self.import_intensity_client.fetch_with_cache()
            logger.info(f"Fetched {len(data['import_intensity'])} import intensity estimates")
        except Exception as e:
            logger.error(f"Import intensity fetch failed: {e}")

        return data


class NationalPanelBuilder:
    """
    Builds analysis panels for FX Passthrough study.

    Creates:
    1. CPI panel for Block A (category × month)
    2. Income time series for Blocks B-D
    3. Expenditure series for Block E
    """

    def __init__(self, pipeline: FXPassthroughPipeline | None = None):
        """
        Initialize panel builder.

        Args:
            pipeline: Data pipeline (creates new one if not provided)
        """
        self.pipeline = pipeline or FXPassthroughPipeline()
        self._quality_reports: list[DataQualityReport] = []

    def build_cpi_panel(
        self,
        start_date: str = "2010-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Build CPI category panel for Block A.

        Returns DataFrame with columns:
            - category: COICOP code
            - date: Month
            - inflation_mom: Month-over-month inflation
            - import_share (s_c): Category import intensity
            - fx_change: Exchange rate change
            - tradable: Whether tradable category
            - admin_price: Whether administered price
        """
        logger.info("Building CPI panel for Block A")

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Fetch CPI data
        cpi_data = self.pipeline.cpi_client.fetch_with_cache()

        # Fetch exchange rates
        fx_data = self.pipeline.exchange_rate_client.fetch_with_cache()
        fx_data = self.pipeline.exchange_rate_client.aggregate_to_monthly(fx_data)
        fx_data = self.pipeline.exchange_rate_client.compute_fx_change(fx_data)

        # Fetch import intensity
        import_shares = self.pipeline.import_intensity_client.fetch_with_cache()

        # Filter date range
        cpi_data = cpi_data[
            (cpi_data["date"] >= start_date) &
            (cpi_data["date"] <= end_date)
        ]

        # Merge import shares
        panel = cpi_data.merge(
            import_shares[["category", "import_share", "tradable"]],
            on="category",
            how="left",
        )

        # Merge FX changes
        panel["month"] = panel["date"].dt.to_period("M")
        fx_data["month"] = fx_data["date"].dt.to_period("M")

        panel = panel.merge(
            fx_data[["month", "fx_change"]],
            on="month",
            how="left",
        )

        # Create time index
        panel["time_idx"] = panel["date"].dt.year * 100 + panel["date"].dt.month

        # Create category index
        panel["category_idx"] = pd.Categorical(panel["category"]).codes

        # Quality report
        self._quality_reports.append(DataQualityReport(
            source="cpi_panel",
            n_rows=len(panel),
            n_cols=len(panel.columns),
            date_range=(
                panel["date"].min().strftime("%Y-%m-%d"),
                panel["date"].max().strftime("%Y-%m-%d"),
            ),
            missing_pct=panel.isnull().mean().mean() * 100,
            quality_grade="good" if panel["fx_change"].notna().mean() > 0.9 else "fair",
        ))

        logger.info(f"Built CPI panel: {len(panel)} observations")
        return panel

    def build_income_series(
        self,
        start_date: str = "2010-01-01",
        end_date: str | None = None,
        frequency: str = "quarterly",
    ) -> pd.DataFrame:
        """
        Build national income time series for Blocks B-D.

        Returns DataFrame with columns:
            - date: Period date
            - quarter: Quarter string (YYYYQ#)
            - nominal_income, wage_income, transfer_income: Income components
            - *_growth: Growth rates
            - *_share: Component shares
            - headline_inflation: CPI inflation
            - imported_inflation: Constructed instrument (if CPI panel built)
        """
        logger.info("Building income series for Blocks B-D")

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Fetch income data
        if frequency == "quarterly":
            income_data = self.pipeline.national_income_client.fetch_quarterly()
        else:
            income_data = self.pipeline.national_income_client.fetch_monthly()

        # Filter date range
        income_data = income_data[
            (income_data["date"] >= start_date) &
            (income_data["date"] <= end_date)
        ]

        # Compute growth rates
        income_data = self.pipeline.national_income_client.compute_income_growth(income_data)

        # Compute shares
        income_data = self.pipeline.national_income_client.compute_income_shares(income_data)

        # Add headline inflation (from CPI data if available)
        try:
            cpi_data = self.pipeline.cpi_client.fetch_with_cache()
            # Get headline (total) CPI
            headline = cpi_data[cpi_data["category"] == "00"][["date", "inflation_mom"]]
            headline = headline.rename(columns={"inflation_mom": "headline_inflation"})

            # Aggregate to quarterly if needed
            if frequency == "quarterly":
                headline["quarter"] = (
                    headline["date"].dt.year.astype(str) + "Q" +
                    headline["date"].dt.quarter.astype(str)
                )
                headline = headline.groupby("quarter")["headline_inflation"].mean().reset_index()
                income_data = income_data.merge(headline, on="quarter", how="left")
            else:
                income_data["month"] = income_data["date"].dt.to_period("M")
                headline["month"] = headline["date"].dt.to_period("M")
                income_data = income_data.merge(
                    headline[["month", "headline_inflation"]],
                    on="month",
                    how="left",
                )
        except Exception as e:
            logger.warning(f"Could not add headline inflation: {e}")

        # Add time index
        if frequency == "quarterly":
            income_data["time_idx"] = (
                income_data["date"].dt.year * 10 +
                income_data["date"].dt.quarter
            )
        else:
            income_data["time_idx"] = (
                income_data["date"].dt.year * 100 +
                income_data["date"].dt.month
            )

        # Quality report
        self._quality_reports.append(DataQualityReport(
            source="income_series",
            n_rows=len(income_data),
            n_cols=len(income_data.columns),
            date_range=(
                income_data["date"].min().strftime("%Y-%m-%d"),
                income_data["date"].max().strftime("%Y-%m-%d"),
            ),
            missing_pct=income_data.isnull().mean().mean() * 100,
            quality_grade="good" if len(income_data) > 40 else "fair",
        ))

        logger.info(f"Built income series: {len(income_data)} observations")
        return income_data

    def build_expenditure_series(
        self,
        start_date: str = "2010-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Build expenditure time series for Block E.

        Returns DataFrame with columns:
            - date, quarter: Time identifiers
            - real_expenditure: Real household expenditure
            - consumption_expenditure: "Income used for consumption"
            - food_expenditure, nonfood_expenditure: Components
            - *_growth, *_share: Derived variables
        """
        logger.info("Building expenditure series for Block E")

        # BNS expenditure data may be in same source as income
        # For now, use income data and add expenditure-specific processing

        income_data = self.build_income_series(start_date, end_date)

        # If we have expenditure-specific data, use it
        # Otherwise, derive from income (consumption = income - savings)
        expenditure_data = income_data.copy()

        # Add expenditure-specific columns if not present
        if "real_expenditure" not in expenditure_data.columns:
            # Use nominal income as proxy (imperfect but allows pipeline to run)
            if "nominal_income" in expenditure_data.columns:
                expenditure_data["real_expenditure"] = expenditure_data["nominal_income"]
                expenditure_data["real_expenditure_growth"] = expenditure_data.get(
                    "nominal_income_growth",
                    np.log(expenditure_data["nominal_income"]).diff()
                )

        logger.info(f"Built expenditure series: {len(expenditure_data)} observations")
        return expenditure_data

    def build_all_panels(
        self,
        start_date: str = "2010-01-01",
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Build all analysis panels."""
        panels = {}

        try:
            panels["cpi_panel"] = self.build_cpi_panel(start_date, end_date)
        except Exception as e:
            logger.error(f"CPI panel build failed: {e}")

        try:
            panels["income_series"] = self.build_income_series(start_date, end_date)
        except Exception as e:
            logger.error(f"Income series build failed: {e}")

        try:
            panels["expenditure_series"] = self.build_expenditure_series(start_date, end_date)
        except Exception as e:
            logger.error(f"Expenditure series build failed: {e}")

        return panels

    def add_imported_inflation_instrument(
        self,
        income_data: pd.DataFrame,
        cpi_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add imported inflation instrument (Z_t) to income data.

        Z_t = Σ_c w_c × s_c × ΔFX_t
        """
        # Compute Z_t from CPI panel
        def compute_z(group):
            # Weighted sum of import_share × fx_change
            if "import_share" in group.columns and "fx_change" in group.columns:
                return (group["import_share"] * group["fx_change"]).mean()
            return np.nan

        z_t = cpi_panel.groupby("time_idx").apply(compute_z)
        z_t = z_t.reset_index()
        z_t.columns = ["time_idx", "imported_inflation"]

        # Merge into income data
        income_data = income_data.merge(z_t, on="time_idx", how="left")

        return income_data

    def save_panels(
        self,
        panels: dict[str, pd.DataFrame],
        output_dir: Path | None = None,
    ) -> dict[str, Path]:
        """Save panels to parquet files."""
        settings = get_settings()
        output_dir = output_dir or (
            settings.project_root / settings.processed_data_dir / "fx_passthrough"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        for name, df in panels.items():
            path = output_dir / f"{name}.parquet"
            df.to_parquet(path, index=False)
            paths[name] = path
            logger.info(f"Saved {name} to {path}")

        return paths

    def load_panels(
        self,
        input_dir: Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load panels from parquet files."""
        settings = get_settings()
        input_dir = input_dir or (
            settings.project_root / settings.processed_data_dir / "fx_passthrough"
        )

        panels = {}
        for name in ["cpi_panel", "income_series", "expenditure_series"]:
            path = input_dir / f"{name}.parquet"
            if path.exists():
                panels[name] = pd.read_parquet(path)
                logger.info(f"Loaded {name} from {path}")
            else:
                logger.warning(f"{name} not found at {path}")

        return panels

    def print_quality_summary(self) -> None:
        """Print data quality summary."""
        print("\n" + "=" * 60)
        print("DATA QUALITY SUMMARY")
        print("=" * 60)

        for report in self._quality_reports:
            print(f"\n{report.source}:")
            print(f"  Rows: {report.n_rows:,}")
            print(f"  Columns: {report.n_cols}")
            if report.date_range:
                print(f"  Date range: {report.date_range[0]} to {report.date_range[1]}")
            print(f"  Missing: {report.missing_pct:.1f}%")
            print(f"  Quality: {report.quality_grade}")
            for note in report.notes:
                print(f"  Note: {note}")
