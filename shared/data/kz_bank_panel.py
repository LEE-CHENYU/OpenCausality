"""
Multi-Bank Panel Data Connector for Kazakhstan Banks.

Loads bank-level KPIs from per-bank JSON files and constructs
panel datasets for shift-share (Exposure × Shock) estimation.

Banks:
- Kaspi Bank (kaspi): 2011-2024, digital retail
- Halyk Bank (halyk): 2011-2024, universal
- ForteBank (forte): 2017-2024, universal
- Bank CenterCredit (bcc): 2015-2024, universal

Design: Time FE absorb common shocks; identification comes from
cross-bank variation in predetermined exposure to the shock.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Base data directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
KZ_BANKS_DIR = RAW_DIR / "kz_banks"
KSPI_DIR = RAW_DIR / "kspi"

# Bank file mapping
BANK_FILES = {
    "kaspi": KSPI_DIR / "kaspi_bank_extended_kpis.json",
    "halyk": KZ_BANKS_DIR / "halyk_bank_quarterly.json",
    "forte": KZ_BANKS_DIR / "fortebank_quarterly.json",
    "bcc": KZ_BANKS_DIR / "bcc_quarterly.json",
}

EXPOSURE_FILE = KZ_BANKS_DIR / "bank_exposures.json"


@dataclass
class BankQuarterlyKPIs:
    """Quarterly KPIs for a single bank observation."""

    bank_id: str
    quarter: str  # Format: "2024Q3"
    frequency: str = "quarterly"  # "quarterly", "semiannual", "annual"
    estimation_eligible: bool = True
    is_interpolated: bool = False

    # KPIs
    net_loans: float | None = None
    deposits: float | None = None
    npl_ratio: float | None = None
    cor: float | None = None
    total_capital: float | None = None
    rwa: float | None = None
    deposit_cost: float | None = None
    ppop: float | None = None
    net_income: float | None = None

    # Metadata
    extraction_source: str = ""
    extraction_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "quarter": self.quarter,
            "frequency": self.frequency,
            "estimation_eligible": self.estimation_eligible,
            "is_interpolated": self.is_interpolated,
            "net_loans": self.net_loans,
            "deposits": self.deposits,
            "npl_ratio": self.npl_ratio,
            "cor": self.cor,
            "total_capital": self.total_capital,
            "rwa": self.rwa,
            "deposit_cost": self.deposit_cost,
            "ppop": self.ppop,
            "net_income": self.net_income,
            "extraction_source": self.extraction_source,
            "extraction_confidence": self.extraction_confidence,
        }


def _quarterly_label_to_date(label: str) -> pd.Timestamp:
    """Convert '2024Q3' to Timestamp at quarter start."""
    year = int(label[:4])
    q = int(label[-1])
    month = (q - 1) * 3 + 1
    return pd.Timestamp(year=year, month=month, day=1)


class KZBankPanelClient:
    """
    Multi-bank panel data client for Kazakhstan banks.

    Loads per-bank JSON files and constructs panel datasets
    with Exposure × Shock interaction columns for shift-share estimation.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or KZ_BANKS_DIR
        self._bank_data: dict[str, pd.DataFrame] = {}
        self._exposures: dict[str, dict[str, float]] | None = None
        self._metadata: dict[str, dict] = {}

    def fetch(self, bank_ids: list[str] | None = None) -> pd.DataFrame:
        """
        Fetch panel data in long format.

        Args:
            bank_ids: List of bank IDs to include (default: all)

        Returns:
            DataFrame with columns: bank_id, quarter, date, and KPI columns.
            Indexed by (bank_id, date) MultiIndex.
        """
        bank_ids = bank_ids or list(BANK_FILES.keys())
        all_records = []

        for bank_id in bank_ids:
            df = self._load_bank(bank_id)
            if df is not None and not df.empty:
                df = df.copy()
                df["bank_id"] = bank_id
                all_records.append(df)

        if not all_records:
            logger.warning("No bank data loaded")
            return pd.DataFrame()

        panel = pd.concat(all_records, ignore_index=False)
        panel = panel.reset_index()

        # Ensure date column
        if "date" not in panel.columns and "quarter" in panel.columns:
            panel["date"] = panel["quarter"].apply(_quarterly_label_to_date)

        panel = panel.set_index(["bank_id", "date"]).sort_index()
        return panel

    def build_panel_for_edge(
        self,
        outcome_kpi: str,
        shock_series: pd.Series,
        exposure_var: str,
        max_horizon: int = 2,
    ) -> pd.DataFrame:
        """
        Build panel dataset with Exposure × Shock interaction column.

        Constructs the regression-ready panel:
            y_{b,t+h}, Exposure_b × Shock_t, bank FE dummies, time FE dummies

        Args:
            outcome_kpi: Name of the outcome KPI column (e.g., "npl_ratio")
            shock_series: The common shock series (e.g., ΔCPI), monthly/quarterly
            exposure_var: Exposure variable name from bank_exposures.json
            max_horizon: Maximum LP horizon

        Returns:
            DataFrame with columns: bank_id, date, outcome, shock,
            exposure, interaction, and horizon-shifted outcomes
        """
        panel = self.fetch()
        exposures = self.load_exposures()

        if panel.empty:
            return pd.DataFrame()

        # Filter to estimation-eligible observations
        if "estimation_eligible" in panel.columns:
            panel = panel[panel["estimation_eligible"] == True]  # noqa: E712

        # Ensure outcome column exists
        if outcome_kpi not in panel.columns:
            logger.warning(f"Outcome KPI '{outcome_kpi}' not found in panel")
            return pd.DataFrame()

        # Aggregate shock to match panel frequency (quarterly for annual banks)
        shock = shock_series.copy()
        if not isinstance(shock.index, pd.DatetimeIndex):
            shock.index = pd.to_datetime(shock.index)

        # Resample to quarterly if monthly
        gap = shock.index.to_series().diff().median()
        if gap is not pd.NaT and gap < pd.Timedelta(days=60):
            shock = shock.resample("QS").mean().dropna()

        # Build regression panel
        records = []
        for (bank_id, date), row in panel.iterrows():
            if bank_id not in exposures:
                continue
            bank_exp = exposures[bank_id]
            if exposure_var not in bank_exp:
                continue

            exposure = bank_exp[exposure_var]

            # Find closest shock value
            shock_val = None
            if date in shock.index:
                shock_val = shock[date]
            else:
                # Try nearest date within tolerance
                closest = shock.index[shock.index.get_indexer([date], method="nearest")]
                if len(closest) > 0:
                    delta = abs(closest[0] - date)
                    if delta < pd.Timedelta(days=100):
                        shock_val = shock[closest[0]]

            if shock_val is None or np.isnan(shock_val):
                continue

            outcome_val = row.get(outcome_kpi)
            if outcome_val is None or (isinstance(outcome_val, float) and np.isnan(outcome_val)):
                continue

            records.append({
                "bank_id": bank_id,
                "date": date,
                "outcome": outcome_val,
                "shock": shock_val,
                "exposure": exposure,
                "interaction": exposure * shock_val,
                "quarter": row.get("quarter", ""),
                "frequency": row.get("frequency", "unknown"),
            })

        if not records:
            logger.warning("No valid panel observations after merging")
            return pd.DataFrame()

        result = pd.DataFrame(records)

        # Add horizon-shifted outcomes
        for bank_id in result["bank_id"].unique():
            mask = result["bank_id"] == bank_id
            bank_subset = result[mask].sort_values("date")
            for h in range(1, max_horizon + 1):
                result.loc[mask, f"outcome_h{h}"] = (
                    bank_subset["outcome"].shift(-h).values
                )

        return result

    def load_exposures(self) -> dict[str, dict[str, float]]:
        """
        Load baseline bank exposure variables from bank_exposures.json.

        Returns:
            Dict mapping bank_id to exposure variables dict.
        """
        if self._exposures is not None:
            return self._exposures

        if not EXPOSURE_FILE.exists():
            logger.warning(f"Exposure file not found: {EXPOSURE_FILE}")
            return {}

        with open(EXPOSURE_FILE) as f:
            data = json.load(f)

        self._exposures = {}
        for bank_id, bank_data in data.get("banks", {}).items():
            self._exposures[bank_id] = {
                k: v for k, v in bank_data.items()
                if isinstance(v, (int, float))
            }

        return self._exposures

    def validate_panel_balance(self) -> dict[str, Any]:
        """
        Check panel balance and coverage consistency.

        Returns:
            Report dict with balance info, coverage per bank,
            and definition consistency notes.
        """
        panel = self.fetch()
        if panel.empty:
            return {"balanced": False, "error": "No data loaded"}

        report: dict[str, Any] = {
            "n_banks": 0,
            "n_total_obs": len(panel),
            "banks": {},
            "balanced": True,
            "common_periods": [],
        }

        # Per-bank coverage
        bank_periods: dict[str, set] = {}
        for bank_id in panel.index.get_level_values("bank_id").unique():
            bank_df = panel.loc[bank_id]
            n_obs = len(bank_df)
            dates = bank_df.index.tolist() if isinstance(bank_df.index, pd.DatetimeIndex) else []
            quarters = bank_df["quarter"].tolist() if "quarter" in bank_df.columns else []

            report["banks"][bank_id] = {
                "n_obs": n_obs,
                "start": str(min(quarters)) if quarters else "?",
                "end": str(max(quarters)) if quarters else "?",
                "metadata": self._metadata.get(bank_id, {}),
            }
            bank_periods[bank_id] = set(quarters)

        report["n_banks"] = len(report["banks"])

        # Check balance
        if bank_periods:
            common = set.intersection(*bank_periods.values())
            report["common_periods"] = sorted(common)
            obs_counts = [len(p) for p in bank_periods.values()]
            report["balanced"] = len(set(obs_counts)) == 1

        return report

    def get_definition_consistency(self, kpi: str) -> dict[str, Any]:
        """
        Check if KPI definitions are consistent across banks.

        Args:
            kpi: KPI name (e.g., "npl_ratio", "total_capital")

        Returns:
            Dict with definition per bank and consistency flag.
        """
        definitions = {}
        definition_fields = {
            "npl_ratio": "npl_definition",
            "total_capital": "capital_definition",
            "rwa": "rwa_definition",
        }

        field_name = definition_fields.get(kpi, "")

        for bank_id, meta in self._metadata.items():
            if field_name and field_name in meta:
                definitions[bank_id] = meta[field_name]
            else:
                definitions[bank_id] = "not specified"

        unique_defs = set(definitions.values()) - {"not specified"}
        consistent = len(unique_defs) <= 1

        return {
            "kpi": kpi,
            "definitions": definitions,
            "consistent": consistent,
            "unique_definitions": list(unique_defs),
        }

    def _load_bank(self, bank_id: str) -> pd.DataFrame | None:
        """Load data for a single bank from its JSON file."""
        if bank_id in self._bank_data:
            return self._bank_data[bank_id]

        file_path = BANK_FILES.get(bank_id)
        if file_path is None or not file_path.exists():
            logger.warning(f"No data file for bank: {bank_id}")
            return None

        try:
            with open(file_path) as f:
                data = json.load(f)

            # Store metadata
            self._metadata[bank_id] = data.get("metadata", {})

            records = data.get("quarters", [])
            if not records:
                return None

            df = pd.DataFrame(records)
            df["date"] = df["quarter"].apply(_quarterly_label_to_date)
            df = df.set_index("date").sort_index()

            self._bank_data[bank_id] = df
            logger.info(f"Loaded {len(df)} observations for bank {bank_id}")
            return df

        except Exception as e:
            logger.error(f"Error loading bank {bank_id}: {e}")
            return None
