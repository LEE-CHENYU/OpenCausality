"""
Data pipeline for household welfare study.

Coordinates fetching, processing, and panel construction from all data sources.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import get_settings
from shared.data.data_pipeline import SharedDataPipeline
from studies.household_welfare.src.panel_data import PanelBuilder

logger = logging.getLogger(__name__)


class WelfareDataPipeline(SharedDataPipeline):
    """
    Data pipeline for household welfare study.

    Extends SharedDataPipeline with Baumeister shock loader
    and study-specific panel construction.
    """

    def __init__(self):
        super().__init__()
        self._baumeister_loader = None

    @property
    def baumeister_loader(self):
        """Lazy-loaded Baumeister loader."""
        if self._baumeister_loader is None:
            # Import here to avoid circular imports
            from src.data.baumeister_loader import BaumeisterLoader
            self._baumeister_loader = BaumeisterLoader()
        return self._baumeister_loader

    def fetch_baumeister_shocks(self) -> pd.DataFrame:
        """Fetch Baumeister oil shocks."""
        logger.info("Fetching Baumeister oil shocks...")
        try:
            return self.baumeister_loader.get_shocks_for_panel()
        except Exception as e:
            logger.error(f"Failed to fetch Baumeister data: {e}")
            return pd.DataFrame()

    def fetch_all_raw(self) -> dict[str, Any]:
        """Fetch all raw data for welfare study."""
        results = {}

        # Fetch BNS data
        results["bns"] = self.fetch_bns_data()

        # Fetch FRED data
        results["fred"] = self.fetch_fred_data()

        # Fetch Baumeister shocks
        results["baumeister"] = self.fetch_baumeister_shocks()

        return results

    def save_all_raw(self) -> dict[str, list[Path]]:
        """Fetch and save all raw data."""
        paths: dict[str, list[Path]] = {"bns": [], "fred": [], "baumeister": []}

        # Save BNS data
        bns_paths = self.save_bns_raw()
        paths["bns"] = list(bns_paths.values())

        # Save FRED data
        fred_paths = self.save_fred_raw()
        paths["fred"] = list(fred_paths.values())

        # Save Baumeister data
        try:
            path = self.baumeister_loader.save_raw()
            paths["baumeister"] = [path]
        except Exception as e:
            logger.error(f"Failed to save Baumeister data: {e}")

        return paths

    def build_panel(
        self,
        start_year: int = 2010,
        end_year: int = 2024,
    ) -> pd.DataFrame:
        """
        Build the main analysis panel.

        Args:
            start_year: Panel start year
            end_year: Panel end year

        Returns:
            Panel DataFrame with region-quarter observations
        """
        # Fetch all data
        raw_data = self.fetch_all_raw()

        # Build panel
        builder = PanelBuilder()
        panel = builder.build(
            bns_data=raw_data.get("bns", {}),
            fred_data=raw_data.get("fred", {}),
            baumeister_data=raw_data.get("baumeister", pd.DataFrame()),
            start_year=start_year,
            end_year=end_year,
        )

        # Generate quality report
        self.generate_quality_report(panel, "welfare_panel")

        return panel

    def save_panel(
        self,
        panel: pd.DataFrame | None = None,
        filename: str = "panel.parquet",
    ) -> Path:
        """Save the analysis panel."""
        if panel is None:
            panel = self.build_panel()

        settings = get_settings()
        output_dir = settings.project_root / "studies" / "household_welfare" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename
        panel.to_parquet(filepath, index=False)

        logger.info(f"Saved panel to {filepath}")
        return filepath


def run_pipeline(save: bool = True) -> pd.DataFrame:
    """Run the complete data pipeline for welfare study."""
    pipeline = WelfareDataPipeline()

    # Build panel
    logger.info("Building welfare analysis panel...")
    panel = pipeline.build_panel()

    # Print quality summary
    pipeline.print_quality_summary()

    # Save if requested
    if save:
        pipeline.save_panel(panel)

    return panel
