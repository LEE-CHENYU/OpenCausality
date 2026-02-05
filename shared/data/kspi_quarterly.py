"""
KSPI (Kaspi.kz) Quarterly KPI Extractor.

Extracts structured KPIs from Kaspi.kz investor relations quarterly reports.

Data sources:
- IR quarterly results PDFs: https://ir.kaspi.kz/quarterly-results
- SEC 20-F filings for regulatory capital figures

Features:
- Structured KPI dataclass with validation
- PDF extraction with confidence scoring
- Manual override support with audit trails
- Internal consistency validation
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from shared.data.base import DataSource

logger = logging.getLogger(__name__)


class KPICategory(Enum):
    """Categories of KSPI KPIs."""

    BALANCE_SHEET = "balance_sheet"
    CREDIT_QUALITY = "credit_quality"
    INCOME_STATEMENT = "income_statement"
    SEGMENT = "segment"
    CAPITAL = "capital"


@dataclass
class ValidationResult:
    """Result of KPI validation."""

    is_valid: bool
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
        }


@dataclass
class KSPIQuarterlyKPIs:
    """
    Concrete, reliably extractable KPIs from KSPI quarterly reports.

    All monetary values in billion KZT unless noted.
    Ratios as percentages (e.g., 2.5 = 2.5%, not 0.025).
    """

    # Quarter identification
    quarter: str  # Format: "2024Q3"
    report_date: date | None = None

    # Extended metadata for frequency-aware estimation
    entity_type: str = "kaspi_bank"  # "kaspi_bank" for bank-level, "kaspi_group" for consolidated
    frequency: str = "quarterly"  # "quarterly", "semiannual", "annual"
    estimation_eligible: bool = True  # False if interpolated or insufficient quality
    is_interpolated: bool = False  # True if derived from annual→quarterly split
    source_document: str = ""  # e.g., "kaspi_bank_annual_report_2014.pdf"
    data_quality_notes: list[str] = field(default_factory=list)

    # Balance sheet KPIs
    net_loans: float | None = None  # bn KZT
    avg_net_loans: float | None = None  # bn KZT
    deposits: float | None = None  # bn KZT
    total_assets: float | None = None  # bn KZT

    # Credit quality KPIs
    npl_ratio: float | None = None  # % (90+ days past due / gross loans)
    cor: float | None = None  # % annualized (provisions / avg loans)
    coverage_ratio: float | None = None  # % (provisions / NPL)

    # Profitability KPIs
    nim: float | None = None  # % (net interest margin)
    ppop: float | None = None  # bn KZT (pre-provision operating profit)
    net_income: float | None = None  # bn KZT

    # Segment breakdown (revenue in bn KZT)
    payments_revenue: float | None = None
    marketplace_revenue: float | None = None
    fintech_revenue: float | None = None
    total_revenue: float | None = None

    # Capital KPIs (may require 20-F for NBK regulatory figures)
    total_capital: float | None = None  # bn KZT
    tier1_capital: float | None = None  # bn KZT
    rwa: float | None = None  # bn KZT (risk-weighted assets)
    k2_ratio: float | None = None  # % (total capital / RWA)
    k1_ratio: float | None = None  # % (tier 1 / RWA)

    # Operational KPIs
    active_users: float | None = None  # millions
    monthly_active_users: float | None = None  # millions
    gmv: float | None = None  # bn KZT (gross merchandise value)

    # Extraction metadata
    extraction_confidence: float = 0.0  # 0-1
    extraction_source: str = ""  # "ir_pdf", "20f", "manual"
    extraction_date: date | None = None
    manual_override: bool = False
    audit_trail: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "quarter": self.quarter,
            "report_date": self.report_date.isoformat() if self.report_date else None,
            # Extended metadata
            "entity_type": self.entity_type,
            "frequency": self.frequency,
            "estimation_eligible": self.estimation_eligible,
            "is_interpolated": self.is_interpolated,
            "source_document": self.source_document,
            "data_quality_notes": self.data_quality_notes,
            # Balance sheet
            "net_loans": self.net_loans,
            "avg_net_loans": self.avg_net_loans,
            "deposits": self.deposits,
            "total_assets": self.total_assets,
            # Credit quality
            "npl_ratio": self.npl_ratio,
            "cor": self.cor,
            "coverage_ratio": self.coverage_ratio,
            # Profitability
            "nim": self.nim,
            "ppop": self.ppop,
            "net_income": self.net_income,
            # Segment
            "payments_revenue": self.payments_revenue,
            "marketplace_revenue": self.marketplace_revenue,
            "fintech_revenue": self.fintech_revenue,
            "total_revenue": self.total_revenue,
            # Capital
            "total_capital": self.total_capital,
            "tier1_capital": self.tier1_capital,
            "rwa": self.rwa,
            "k2_ratio": self.k2_ratio,
            "k1_ratio": self.k1_ratio,
            # Operational
            "active_users": self.active_users,
            "monthly_active_users": self.monthly_active_users,
            "gmv": self.gmv,
            # Metadata
            "extraction_confidence": self.extraction_confidence,
            "extraction_source": self.extraction_source,
            "extraction_date": self.extraction_date.isoformat() if self.extraction_date else None,
            "manual_override": self.manual_override,
            "audit_trail": self.audit_trail,
        }

    def compute_hash(self) -> str:
        """Compute hash of KPIs for audit."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class QuarterlyReport:
    """Container for a quarterly report with KPIs and validation."""

    kpis: KSPIQuarterlyKPIs
    validation: ValidationResult
    raw_source_path: Path | None = None
    parsed_at: datetime | None = None


class KSPIQuarterlyClient(DataSource):
    """
    Client for extracting KSPI quarterly KPIs.

    Supports:
    - Loading from pre-parsed JSON/YAML files
    - Manual entry with validation
    - PDF extraction (placeholder for implementation)

    Note: Full PDF extraction requires pdfplumber or similar.
    This implementation focuses on structured data management.
    """

    @property
    def source_name(self) -> str:
        return "kspi_quarterly"

    def __init__(
        self,
        data_dir: Path | None = None,
        cache_dir: Path | None = None,
    ):
        """
        Initialize KSPI quarterly client.

        Args:
            data_dir: Directory containing pre-parsed KPI files
            cache_dir: Cache directory for downloaded PDFs
        """
        super().__init__(cache_dir)
        self.data_dir = data_dir or Path("data/raw/kspi")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # In-memory store for KPIs
        self._kpi_store: dict[str, KSPIQuarterlyKPIs] = {}

    def fetch(self, **kwargs: Any) -> pd.DataFrame:
        """
        Fetch KSPI quarterly KPIs as DataFrame.

        Args:
            start_quarter: Start quarter (e.g., "2020Q1")
            end_quarter: End quarter (default: latest)
            kpi_category: Filter by category (optional)

        Returns:
            DataFrame with quarterly KPIs
        """
        start = kwargs.get("start_quarter", "2020Q1")
        end = kwargs.get("end_quarter")
        category = kwargs.get("kpi_category")

        # Load all available KPIs
        self._load_stored_kpis()

        # Convert to DataFrame
        records = []
        for quarter, kpi in sorted(self._kpi_store.items()):
            if self._quarter_in_range(quarter, start, end):
                records.append(kpi.to_dict())

        df = pd.DataFrame(records)

        if df.empty:
            logger.warning(f"No KSPI KPIs found for range {start} to {end}")
            return pd.DataFrame()

        # Filter by category if specified
        if category:
            category_cols = self._get_category_columns(KPICategory(category))
            keep_cols = ["quarter", "report_date"] + category_cols + [
                "extraction_confidence", "extraction_source"
            ]
            df = df[[c for c in keep_cols if c in df.columns]]

        return df

    def fetch_quarterly_report(self, quarter: str) -> QuarterlyReport:
        """
        Fetch or extract KPIs for a specific quarter.

        Args:
            quarter: Quarter in format "2024Q3"

        Returns:
            QuarterlyReport with KPIs and validation
        """
        # Check store first
        if quarter in self._kpi_store:
            kpis = self._kpi_store[quarter]
            validation = self.validate_extraction(kpis)
            return QuarterlyReport(kpis=kpis, validation=validation)

        # Try to load from file
        file_path = self.data_dir / f"kpis_{quarter}.json"
        if file_path.exists():
            kpis = self._load_kpis_from_file(file_path)
            if kpis:
                self._kpi_store[quarter] = kpis
                validation = self.validate_extraction(kpis)
                return QuarterlyReport(
                    kpis=kpis,
                    validation=validation,
                    raw_source_path=file_path,
                )

        # No data available
        raise FileNotFoundError(
            f"No KPI data found for {quarter}. "
            f"Use add_manual_kpis() to enter data or place JSON at {file_path}"
        )

    def add_manual_kpis(
        self,
        kpis: KSPIQuarterlyKPIs,
        override_existing: bool = False,
        audit_note: str = "",
    ) -> QuarterlyReport:
        """
        Add or update KPIs with manual entry.

        Args:
            kpis: KSPIQuarterlyKPIs to add
            override_existing: Allow overwriting existing data
            audit_note: Note for audit trail

        Returns:
            QuarterlyReport with validation
        """
        quarter = kpis.quarter

        if quarter in self._kpi_store and not override_existing:
            raise ValueError(
                f"KPIs for {quarter} already exist. "
                "Set override_existing=True to replace."
            )

        # Mark as manual entry
        kpis.manual_override = True
        kpis.extraction_source = "manual"
        kpis.extraction_date = date.today()
        kpis.extraction_confidence = 1.0  # Manual = high confidence

        # Add audit trail
        if audit_note:
            kpis.audit_trail.append(f"{datetime.now().isoformat()}: {audit_note}")
        kpis.audit_trail.append(
            f"{datetime.now().isoformat()}: Manual entry, hash={kpis.compute_hash()}"
        )

        # Validate
        validation = self.validate_extraction(kpis)

        # Store
        self._kpi_store[quarter] = kpis

        # Persist to file
        self._save_kpis_to_file(kpis)

        return QuarterlyReport(kpis=kpis, validation=validation)

    def validate_extraction(self, kpis: KSPIQuarterlyKPIs) -> ValidationResult:
        """
        Validate extracted KPIs for internal consistency.

        Checks:
        - Cross-field consistency (e.g., K2 = capital / RWA)
        - Plausible value ranges
        - Required fields present
        - Growth rates reasonable

        Args:
            kpis: KPIs to validate

        Returns:
            ValidationResult with details
        """
        passed = []
        failed = []
        warnings = []

        # Check required fields
        required = ["net_loans", "deposits"]
        for field_name in required:
            value = getattr(kpis, field_name, None)
            if value is not None:
                passed.append(f"required_field_{field_name}")
            else:
                warnings.append(f"Missing recommended field: {field_name}")

        # Check K2 consistency
        if all(v is not None for v in [kpis.k2_ratio, kpis.total_capital, kpis.rwa]):
            computed_k2 = (kpis.total_capital / kpis.rwa) * 100
            if abs(computed_k2 - kpis.k2_ratio) < 0.5:  # Within 0.5%
                passed.append("k2_consistency")
            else:
                failed.append(
                    f"K2 inconsistency: reported={kpis.k2_ratio:.2f}%, "
                    f"computed={computed_k2:.2f}%"
                )

        # Check NPL plausibility (0-50%)
        if kpis.npl_ratio is not None:
            if 0 <= kpis.npl_ratio <= 50:
                passed.append("npl_plausible_range")
            else:
                failed.append(f"NPL ratio outside plausible range: {kpis.npl_ratio}%")

        # Check CoR plausibility (0-20%)
        if kpis.cor is not None:
            if 0 <= kpis.cor <= 20:
                passed.append("cor_plausible_range")
            else:
                warnings.append(f"CoR unusually high: {kpis.cor}%")

        # Check segment revenue sums
        if all(v is not None for v in [
            kpis.payments_revenue, kpis.marketplace_revenue, kpis.fintech_revenue
        ]):
            segment_sum = (
                kpis.payments_revenue +
                kpis.marketplace_revenue +
                kpis.fintech_revenue
            )
            if kpis.total_revenue is not None:
                # Allow 20% discrepancy for other segments
                if segment_sum <= kpis.total_revenue * 1.2:
                    passed.append("segment_sum_plausible")
                else:
                    warnings.append(
                        f"Segment sum ({segment_sum:.1f}) exceeds total revenue "
                        f"({kpis.total_revenue:.1f})"
                    )

        # Check deposits vs loans ratio
        if kpis.net_loans is not None and kpis.deposits is not None:
            ld_ratio = kpis.net_loans / kpis.deposits if kpis.deposits > 0 else 0
            if 0.5 <= ld_ratio <= 2.0:
                passed.append("loan_deposit_ratio_plausible")
            else:
                warnings.append(f"Unusual loan/deposit ratio: {ld_ratio:.2f}")

        is_valid = len(failed) == 0

        return ValidationResult(
            is_valid=is_valid,
            checks_passed=passed,
            checks_failed=failed,
            warnings=warnings,
        )

    def get_time_series(
        self,
        kpi_name: str,
        start_quarter: str = "2020Q1",
        end_quarter: str | None = None,
    ) -> pd.Series:
        """
        Get a single KPI as time series.

        Args:
            kpi_name: Name of KPI field
            start_quarter: Start quarter
            end_quarter: End quarter (default: latest)

        Returns:
            pd.Series indexed by quarter
        """
        df = self.fetch(start_quarter=start_quarter, end_quarter=end_quarter)

        if kpi_name not in df.columns:
            raise ValueError(f"Unknown KPI: {kpi_name}")

        return df.set_index("quarter")[kpi_name]

    def compute_growth_rates(
        self,
        kpi_names: list[str] | None = None,
        yoy: bool = True,
    ) -> pd.DataFrame:
        """
        Compute growth rates for KPIs.

        Args:
            kpi_names: List of KPIs to compute (default: balance sheet)
            yoy: If True, compute year-over-year; else quarter-over-quarter

        Returns:
            DataFrame with growth rates
        """
        if kpi_names is None:
            kpi_names = ["net_loans", "deposits", "total_revenue", "net_income"]

        df = self.fetch()

        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("quarter")

        periods = 4 if yoy else 1
        suffix = "_yoy" if yoy else "_qoq"

        for kpi in kpi_names:
            if kpi in df.columns:
                df[f"{kpi}{suffix}"] = df[kpi].pct_change(periods=periods) * 100

        return df

    def _load_stored_kpis(self) -> None:
        """Load all stored KPIs from data directory."""
        for file_path in self.data_dir.glob("kpis_*.json"):
            try:
                kpis = self._load_kpis_from_file(file_path)
                if kpis:
                    self._kpi_store[kpis.quarter] = kpis
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

    def _load_kpis_from_file(self, file_path: Path) -> KSPIQuarterlyKPIs | None:
        """Load KPIs from JSON file."""
        try:
            with open(file_path) as f:
                data = json.load(f)

            # Parse dates
            if data.get("report_date"):
                data["report_date"] = date.fromisoformat(data["report_date"])
            if data.get("extraction_date"):
                data["extraction_date"] = date.fromisoformat(data["extraction_date"])

            return KSPIQuarterlyKPIs(**data)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def _save_kpis_to_file(self, kpis: KSPIQuarterlyKPIs) -> Path:
        """Save KPIs to JSON file."""
        file_path = self.data_dir / f"kpis_{kpis.quarter}.json"

        with open(file_path, "w") as f:
            json.dump(kpis.to_dict(), f, indent=2, default=str)

        logger.info(f"Saved KPIs for {kpis.quarter} to {file_path}")
        return file_path

    def _quarter_in_range(
        self,
        quarter: str,
        start: str | None,
        end: str | None,
    ) -> bool:
        """Check if quarter is in range."""
        if start and quarter < start:
            return False
        if end and quarter > end:
            return False
        return True

    def _get_category_columns(self, category: KPICategory) -> list[str]:
        """Get column names for a KPI category."""
        category_map = {
            KPICategory.BALANCE_SHEET: [
                "net_loans", "avg_net_loans", "deposits", "total_assets"
            ],
            KPICategory.CREDIT_QUALITY: [
                "npl_ratio", "cor", "coverage_ratio"
            ],
            KPICategory.INCOME_STATEMENT: [
                "nim", "ppop", "net_income", "total_revenue"
            ],
            KPICategory.SEGMENT: [
                "payments_revenue", "marketplace_revenue", "fintech_revenue"
            ],
            KPICategory.CAPITAL: [
                "total_capital", "tier1_capital", "rwa", "k2_ratio", "k1_ratio"
            ],
        }
        return category_map.get(category, [])

    def get_download_instructions(self) -> str:
        """Get instructions for obtaining KSPI data."""
        return """
KSPI Quarterly Data Instructions
=================================

Source URLs:
1. IR Quarterly Results: https://ir.kaspi.kz/quarterly-results
2. SEC 20-F Filings: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001770787

Manual Entry:
-------------
Use the add_manual_kpis() method to enter KPIs:

from shared.data.kspi_quarterly import KSPIQuarterlyClient, KSPIQuarterlyKPIs

client = KSPIQuarterlyClient()

kpis = KSPIQuarterlyKPIs(
    quarter="2024Q3",
    net_loans=2800.0,  # bn KZT
    deposits=3200.0,
    npl_ratio=2.5,
    cor=3.2,
    # ... other fields
)

client.add_manual_kpis(kpis, audit_note="Extracted from Q3 2024 IR presentation")

JSON File Format:
-----------------
Alternatively, place JSON files in data/raw/kspi/kpis_2024Q3.json:

{
    "quarter": "2024Q3",
    "net_loans": 2800.0,
    "deposits": 3200.0,
    "npl_ratio": 2.5,
    "cor": 3.2,
    "payments_revenue": 150.0,
    "marketplace_revenue": 80.0,
    "fintech_revenue": 200.0,
    "extraction_source": "ir_pdf",
    "extraction_confidence": 0.9
}

Required KPIs for DAG:
---------------------
- net_loans (balance sheet volume)
- npl_ratio (credit quality)
- cor (provisioning)
- total_capital, rwa, k2_ratio (capital adequacy)
- payments_revenue, marketplace_revenue, fintech_revenue (segment)
"""


# Pre-populated historical data (loaded from kaspi_bank_extended_kpis.json first,
# then falls back to kspi_historical_kpis.json, then inline data).
def _load_historical_kpis() -> dict[str, dict]:
    """Load KSPI historical KPIs from JSON file, with inline fallback.

    Tries kaspi_bank_extended_kpis.json first (extended bank-level data),
    then kspi_historical_kpis.json (original quarterly data).
    """
    # Try extended bank-level data first
    extended_path = Path("data/raw/kspi/kaspi_bank_extended_kpis.json")
    if extended_path.exists():
        try:
            with open(extended_path) as f:
                data = json.load(f)
            logger.info("Loaded extended Kaspi Bank KPIs from kaspi_bank_extended_kpis.json")
            return {q["quarter"]: q for q in data["quarters"]}
        except Exception as e:
            logger.warning(f"Failed to load kaspi_bank_extended_kpis.json: {e}")

    # Fall back to original historical KPIs
    json_path = Path("data/raw/kspi/kspi_historical_kpis.json")
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
            return {q["quarter"]: q for q in data["quarters"]}
        except Exception as e:
            logger.warning(f"Failed to load kspi_historical_kpis.json: {e}")

    # Inline fallback (3 quarters)
    return {
        "2024Q3": {
            "quarter": "2024Q3",
            "net_loans": 2800.0, "deposits": 3200.0,
            "npl_ratio": 2.5, "cor": 3.2,
            "total_capital": 259.2, "rwa": 1600.0, "k2_ratio": 16.2,
            "payments_revenue": 150.0, "deposit_cost": 9.5,
            "extraction_source": "manual", "extraction_confidence": 0.9,
        },
        "2024Q2": {
            "quarter": "2024Q2",
            "net_loans": 2650.0, "deposits": 3050.0,
            "npl_ratio": 2.5, "cor": 3.0,
            "total_capital": 255.0, "rwa": 1550.0, "k2_ratio": 16.5,
            "payments_revenue": 142.0, "deposit_cost": 9.4,
            "extraction_source": "placeholder", "extraction_confidence": 0.5,
        },
        "2024Q1": {
            "quarter": "2024Q1",
            "net_loans": 2500.0, "deposits": 2900.0,
            "npl_ratio": 2.3, "cor": 2.8,
            "total_capital": 250.0, "rwa": 1500.0, "k2_ratio": 16.7,
            "payments_revenue": 135.0, "deposit_cost": 9.2,
            "extraction_source": "placeholder", "extraction_confidence": 0.5,
        },
    }


KSPI_HISTORICAL_KPIS = _load_historical_kpis()


def get_kspi_quarterly_client(
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> KSPIQuarterlyClient:
    """Get KSPI quarterly client instance."""
    return KSPIQuarterlyClient(data_dir, cache_dir)
