"""
Data Ingest Pipeline — Drop-and-Process for data/raw/.

Users drop raw data files (CSV, Excel, Parquet, JSON) into data/raw/
and this module automatically detects, profiles, standardizes, and
registers them for use in DAG estimation.

Workflow:
    IngestPipeline.scan()          -> find new/changed files
    IngestPipeline.ingest_file()   -> profile + standardize -> parquet
    IngestPipeline.register_loaders() -> register NODE_LOADERS
    IngestPipeline.get_virtual_nodes() -> create NodeSpec entries
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Base data directory (mirrors dynamic_loader.py)
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INGESTED_DIR = PROCESSED_DIR / "ingested"
MANIFEST_PATH = INGESTED_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    name: str
    dtype: str  # "numeric", "datetime", "string", "boolean"
    non_null_count: int
    null_count: int
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "name": self.name,
            "dtype": self.dtype,
            "non_null_count": self.non_null_count,
            "null_count": self.null_count,
        }
        if self.mean is not None:
            d["mean"] = self.mean
        if self.std is not None:
            d["std"] = self.std
        if self.min_val is not None:
            d["min_val"] = self.min_val
        if self.max_val is not None:
            d["max_val"] = self.max_val
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ColumnProfile:
        return cls(
            name=d["name"],
            dtype=d["dtype"],
            non_null_count=d["non_null_count"],
            null_count=d["null_count"],
            mean=d.get("mean"),
            std=d.get("std"),
            min_val=d.get("min_val"),
            max_val=d.get("max_val"),
        )


@dataclass
class DatasetProfile:
    file_id: str
    original_path: str
    format: str  # "csv", "xlsx", "parquet", "json"
    rows: int
    columns: list[ColumnProfile]
    date_column: str | None = None
    frequency: str | None = None  # daily/monthly/quarterly/annual/irregular/None
    date_range: tuple[str, str] | None = None
    value_columns: list[str] = field(default_factory=list)
    primary_value_column: str | None = None
    output_path: str = ""
    ingested_at: str = ""
    file_hash: str = ""
    file_size: int = 0
    mtime: float = 0.0
    n_unique_dates: int = 0
    n_duplicate_dates: int = 0
    sidecar: dict | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "file_id": self.file_id,
            "original_path": self.original_path,
            "format": self.format,
            "rows": self.rows,
            "columns": [c.to_dict() for c in self.columns],
            "date_column": self.date_column,
            "frequency": self.frequency,
            "date_range": list(self.date_range) if self.date_range else None,
            "value_columns": self.value_columns,
            "primary_value_column": self.primary_value_column,
            "output_path": self.output_path,
            "ingested_at": self.ingested_at,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "mtime": self.mtime,
            "n_unique_dates": self.n_unique_dates,
            "n_duplicate_dates": self.n_duplicate_dates,
            "sidecar": self.sidecar,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DatasetProfile:
        date_range = tuple(d["date_range"]) if d.get("date_range") else None
        return cls(
            file_id=d["file_id"],
            original_path=d["original_path"],
            format=d["format"],
            rows=d["rows"],
            columns=[ColumnProfile.from_dict(c) for c in d.get("columns", [])],
            date_column=d.get("date_column"),
            frequency=d.get("frequency"),
            date_range=date_range,
            value_columns=d.get("value_columns", []),
            primary_value_column=d.get("primary_value_column"),
            output_path=d.get("output_path", ""),
            ingested_at=d.get("ingested_at", ""),
            file_hash=d.get("file_hash", ""),
            file_size=d.get("file_size", 0),
            mtime=d.get("mtime", 0.0),
            n_unique_dates=d.get("n_unique_dates", 0),
            n_duplicate_dates=d.get("n_duplicate_dates", 0),
            sidecar=d.get("sidecar"),
        )


@dataclass
class IngestManifest:
    manifest_version: str = "1.0"
    updated_at: str = ""
    datasets: list[DatasetProfile] = field(default_factory=list)

    # Internal index for O(1) lookup
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._rebuild_index()

    def _rebuild_index(self):
        self._index = {ds.file_id: i for i, ds in enumerate(self.datasets)}

    def get_dataset(self, file_id: str) -> DatasetProfile | None:
        idx = self._index.get(file_id)
        return self.datasets[idx] if idx is not None else None

    def is_stale(self, file_id: str, file_size: int, mtime: float, file_hash: str) -> bool:
        """Fast staleness check: size+mtime first, hash only if needed."""
        ds = self.get_dataset(file_id)
        if ds is None:
            return True
        if ds.file_size != file_size or ds.mtime != mtime:
            return ds.file_hash != file_hash
        return False

    def upsert(self, profile: DatasetProfile):
        idx = self._index.get(profile.file_id)
        if idx is not None:
            self.datasets[idx] = profile
        else:
            self.datasets.append(profile)
            self._index[profile.file_id] = len(self.datasets) - 1

    def to_dict(self) -> dict:
        return {
            "manifest_version": self.manifest_version,
            "updated_at": self.updated_at,
            "datasets": [ds.to_dict() for ds in self.datasets],
        }

    @classmethod
    def from_dict(cls, d: dict) -> IngestManifest:
        manifest = cls(
            manifest_version=d.get("manifest_version", "1.0"),
            updated_at=d.get("updated_at", ""),
            datasets=[DatasetProfile.from_dict(ds) for ds in d.get("datasets", [])],
        )
        return manifest


# ---------------------------------------------------------------------------
# Module-level loader (picklable via functools.partial)
# ---------------------------------------------------------------------------

def load_ingested_series(path: str, col: str, node_id: str) -> pd.Series:
    """Load a single series from an ingested parquet file.

    This is a top-level function so functools.partial(load_ingested_series, ...)
    is picklable for multiprocessing compatibility.
    """
    from shared.engine.dynamic_loader import _load_series_from_file
    return _load_series_from_file(Path(path), col, node_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify_column(col_name: str) -> str:
    """Slugify a column name for use as a node_id component.

    "GDP growth (%)" -> "gdp_growth_pct"
    """
    s = col_name.lower()
    s = s.replace("%", "pct").replace("$", "usd").replace("€", "eur")
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _compute_file_hash(path: Path) -> str:
    """SHA256 of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_sidecar(file_path: Path) -> dict | None:
    """Load optional sidecar YAML for a data file."""
    import yaml

    for ext in (".yaml", ".yml"):
        sidecar_path = file_path.with_suffix(ext)
        if sidecar_path.exists():
            with open(sidecar_path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    return None


def _read_raw_file(path: Path, sidecar: dict | None = None) -> pd.DataFrame:
    """Read a raw data file into a DataFrame with robust format handling."""
    suffix = path.suffix.lower()
    sidecar = sidecar or {}

    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(path, sep=sep, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot decode {path} with any supported encoding")

    elif suffix in (".xlsx", ".xls"):
        sheet = sidecar.get("sheet_name", 0)
        return pd.read_excel(path, sheet_name=sheet)

    elif suffix == ".parquet":
        return pd.read_parquet(path)

    elif suffix in (".json", ".jsonl"):
        is_jsonl = suffix == ".jsonl" or sidecar.get("json_lines", False)
        if is_jsonl:
            return pd.read_json(path, lines=True)
        try:
            return pd.read_json(path)
        except ValueError:
            # Nested objects — try json_normalize
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.json_normalize(data)
            elif isinstance(data, dict):
                # Try to find a list inside
                for v in data.values():
                    if isinstance(v, list):
                        return pd.json_normalize(v)
            return pd.json_normalize(data)

    raise ValueError(f"Unsupported file format: {suffix}")


def _infer_date_column(df: pd.DataFrame, sidecar: dict | None = None) -> tuple[pd.DataFrame, str | None]:
    """Detect and parse date column, returning (df_with_datetime_index, date_col_name).

    Reuses heuristic from dynamic_loader.py:_load_series_from_file.
    """
    sidecar = sidecar or {}

    # 1. Sidecar override
    if "date_column" in sidecar:
        col = sidecar["date_column"]
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.set_index(col).sort_index()
            return df, col

    # 2. Exact "date" column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date").sort_index()
        return df, "date"

    # 3. "quarter" column
    if "quarter" in df.columns:
        try:
            df["_date"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp()
            df = df.set_index("_date").sort_index()
            return df, "quarter"
        except Exception:
            pass

    # 4. year + month columns
    if "year" in df.columns and "month" in df.columns:
        try:
            df["_date"] = pd.to_datetime(
                df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
            )
            df = df.set_index("_date").sort_index()
            return df, "year+month"
        except Exception:
            pass

    # 5. Regex search for date-like columns
    date_pattern = re.compile(r"(date|time|period|year)", re.IGNORECASE)
    for col in df.columns:
        if date_pattern.search(col):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:
                    df[col] = parsed
                    df = df.set_index(col).sort_index()
                    return df, col
            except Exception:
                continue

    # 6. Try parsing index as datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            parsed_idx = pd.to_datetime(df.index, errors="coerce")
            if parsed_idx.notna().sum() > len(df) * 0.5:
                df.index = parsed_idx
                df = df.sort_index()
                return df, "(index)"
        except Exception:
            pass

    return df, None


def _infer_frequency(df: pd.DataFrame) -> str | None:
    """Infer time series frequency from a DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    idx = df.index.dropna()
    if len(idx) < 3:
        return None

    # Try pandas infer_freq
    try:
        freq = pd.infer_freq(idx)
        if freq:
            freq_upper = freq.upper()
            if "D" in freq_upper and "M" not in freq_upper:
                return "daily"
            if freq_upper.startswith("M") or freq_upper.startswith("MS"):
                return "monthly"
            if freq_upper.startswith("Q") or freq_upper.startswith("QS"):
                return "quarterly"
            if freq_upper.startswith("A") or freq_upper.startswith("Y") or freq_upper.startswith("AS") or freq_upper.startswith("YS"):
                return "annual"
    except Exception:
        pass

    # Fallback: median gap
    diffs = pd.Series(idx).diff().dropna()
    if len(diffs) == 0:
        return None
    median_gap = diffs.median().days
    gap_std = diffs.dt.days.std()
    gap_mean = diffs.dt.days.mean()

    # CV check for irregular
    if gap_mean > 0 and (gap_std / gap_mean) > 0.5:
        return "irregular"

    if median_gap < 3:
        return "daily"
    elif median_gap < 35:
        return "monthly"
    elif median_gap < 120:
        return "quarterly"
    else:
        return "annual"


def _identify_value_columns(df: pd.DataFrame) -> list[str]:
    """Identify numeric value columns, excluding ID-like columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    value_cols = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        # Skip ID-like columns: high cardinality + monotonic + integer-only
        is_integer = (s == s.astype(int)).all() if len(s) > 0 else False
        is_monotonic = s.is_monotonic_increasing or s.is_monotonic_decreasing
        uniqueness = s.nunique() / len(s) if len(s) > 0 else 0
        if is_integer and is_monotonic and uniqueness > 0.9 and len(s) > 5:
            continue
        value_cols.append(col)
    return value_cols


def _profile_column(s: pd.Series) -> ColumnProfile:
    """Profile a single column."""
    non_null = int(s.notna().sum())
    null = int(s.isna().sum())

    if pd.api.types.is_numeric_dtype(s):
        dtype = "numeric"
        desc = s.describe()
        return ColumnProfile(
            name=s.name,
            dtype=dtype,
            non_null_count=non_null,
            null_count=null,
            mean=float(desc.get("mean", 0)) if non_null > 0 else None,
            std=float(desc.get("std", 0)) if non_null > 1 else None,
            min_val=float(desc.get("min", 0)) if non_null > 0 else None,
            max_val=float(desc.get("max", 0)) if non_null > 0 else None,
        )
    elif pd.api.types.is_datetime64_any_dtype(s):
        dtype = "datetime"
    elif pd.api.types.is_bool_dtype(s):
        dtype = "boolean"
    else:
        dtype = "string"

    return ColumnProfile(name=s.name, dtype=dtype, non_null_count=non_null, null_count=null)


# ---------------------------------------------------------------------------
# IngestPipeline
# ---------------------------------------------------------------------------

class IngestPipeline:
    """Core pipeline: scan raw dir, profile, standardize, register loaders."""

    KNOWN_MANAGED_DIRS = {
        "fred", "baumeister_shocks", "kazakhstan_bns", "kspi",
        "kz_banks", "nbk", "nbk_credit", "alternative_sources",
        "imf_eer", "imf_fsi", "import_intensity",
    }
    SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}

    def __init__(
        self,
        raw_dir: Path | None = None,
        processed_dir: Path | None = None,
    ):
        self.raw_dir = raw_dir or RAW_DIR
        self.processed_dir = processed_dir or INGESTED_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> IngestManifest:
        """Load manifest from disk or create empty."""
        if MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH) as f:
                    data = json.load(f)
                return IngestManifest.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
        return IngestManifest()

    def _save_manifest(self):
        """Atomically save manifest to disk."""
        self.manifest.updated_at = datetime.now(timezone.utc).isoformat()
        tmp_path = MANIFEST_PATH.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.manifest.to_dict(), f, indent=2)
        os.rename(str(tmp_path), str(MANIFEST_PATH))

    def _make_file_id(self, path: Path) -> str:
        """Generate file_id from relative path, collision-safe.

        data/raw/my_data.csv        -> "my_data"
        data/raw/macro/my_data.csv  -> "macro__my_data"
        data/raw/a/b/c.csv          -> "a__b__c"
        """
        rel = path.relative_to(self.raw_dir)
        parts = list(rel.parts)
        stem = Path(parts[-1]).stem
        parts[-1] = stem
        # Slugify each part individually, then join with __
        slugified = []
        for part in parts:
            s = re.sub(r"[^a-z0-9_]", "_", part.lower())
            s = re.sub(r"_+", "_", s).strip("_")
            if s:
                slugified.append(s)
        return "__".join(slugified)

    def scan(self) -> list[Path]:
        """Find new or changed files in raw_dir outside managed directories."""
        if not self.raw_dir.exists():
            return []

        candidates = []
        for path in self.raw_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            # Skip sidecar YAML files
            if path.suffix.lower() in (".yaml", ".yml"):
                continue

            # Skip managed directories (check first path component)
            try:
                rel = path.relative_to(self.raw_dir)
                first_dir = rel.parts[0] if len(rel.parts) > 1 else None
                if first_dir and first_dir.lower() in self.KNOWN_MANAGED_DIRS:
                    continue
            except ValueError:
                continue

            # Check staleness
            file_id = self._make_file_id(path)
            stat = path.stat()
            file_hash = _compute_file_hash(path)
            if self.manifest.is_stale(file_id, stat.st_size, stat.st_mtime, file_hash):
                candidates.append(path)

        return sorted(candidates)

    def ingest_file(self, path: Path, force: bool = False) -> DatasetProfile:
        """Profile, standardize, and register a single file."""
        file_id = self._make_file_id(path)
        stat = path.stat()
        file_hash = _compute_file_hash(path)

        # Skip if not stale (unless forced)
        if not force and not self.manifest.is_stale(file_id, stat.st_size, stat.st_mtime, file_hash):
            existing = self.manifest.get_dataset(file_id)
            if existing:
                return existing

        logger.info(f"Ingesting: {path} -> {file_id}")

        # Load sidecar
        sidecar = _load_sidecar(path)

        # Read raw file
        df = _read_raw_file(path, sidecar)

        # Infer date column and set index
        df, date_col = _infer_date_column(df, sidecar)

        # Infer frequency
        frequency = None
        if sidecar and "frequency" in sidecar:
            frequency = sidecar["frequency"]
        elif isinstance(df.index, pd.DatetimeIndex):
            frequency = _infer_frequency(df)

        # Date range and duplicate stats
        date_range = None
        n_unique = 0
        n_dup = 0
        if isinstance(df.index, pd.DatetimeIndex):
            valid_idx = df.index.dropna()
            if len(valid_idx) > 0:
                date_range = (str(valid_idx.min().date()), str(valid_idx.max().date()))
                n_unique = valid_idx.nunique()
                n_dup = len(valid_idx) - n_unique

        # Handle duplicate dates: group by date and take mean for numerics
        if isinstance(df.index, pd.DatetimeIndex) and df.index.duplicated().any():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df = df.groupby(df.index)[numeric_cols].mean()

        # Identify value columns
        value_columns = _identify_value_columns(df)
        if sidecar and "value_column" in sidecar:
            vc = sidecar["value_column"]
            if vc in df.columns:
                primary_value = vc
            else:
                primary_value = value_columns[0] if value_columns else None
        else:
            primary_value = value_columns[0] if value_columns else None

        # Profile columns
        col_profiles = []
        for col in df.columns:
            col_profiles.append(_profile_column(df[col]))

        # Determine output format suffix
        suffix = path.suffix.lower()
        fmt_map = {
            ".csv": "csv", ".tsv": "csv", ".xlsx": "xlsx", ".xls": "xlsx",
            ".parquet": "parquet", ".json": "json", ".jsonl": "json",
        }

        # Write standardized parquet (atomic)
        output_path = self.processed_dir / f"{file_id}.parquet"
        tmp_path = output_path.with_suffix(".parquet.tmp")
        df.to_parquet(str(tmp_path))
        os.rename(str(tmp_path), str(output_path))

        # Build profile
        profile = DatasetProfile(
            file_id=file_id,
            original_path=str(path),
            format=fmt_map.get(suffix, suffix.lstrip(".")),
            rows=len(df),
            columns=col_profiles,
            date_column=date_col,
            frequency=frequency,
            date_range=date_range,
            value_columns=value_columns,
            primary_value_column=primary_value,
            output_path=str(output_path),
            ingested_at=datetime.now(timezone.utc).isoformat(),
            file_hash=file_hash,
            file_size=stat.st_size,
            mtime=stat.st_mtime,
            n_unique_dates=n_unique,
            n_duplicate_dates=n_dup,
            sidecar=sidecar,
        )

        # Update manifest
        self.manifest.upsert(profile)
        self._save_manifest()

        return profile

    def ingest_all(self, force: bool = False) -> list[DatasetProfile]:
        """Scan and ingest all new/changed files."""
        if force:
            # Re-scan everything, not just stale
            files = []
            if self.raw_dir.exists():
                for path in self.raw_dir.rglob("*"):
                    if not path.is_file():
                        continue
                    if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                        continue
                    if path.suffix.lower() in (".yaml", ".yml"):
                        continue
                    try:
                        rel = path.relative_to(self.raw_dir)
                        first_dir = rel.parts[0] if len(rel.parts) > 1 else None
                        if first_dir and first_dir.lower() in self.KNOWN_MANAGED_DIRS:
                            continue
                    except ValueError:
                        continue
                    files.append(path)
            files = sorted(files)
        else:
            files = self.scan()

        results = []
        for path in files:
            try:
                profile = self.ingest_file(path, force=force)
                results.append(profile)
            except Exception as e:
                logger.warning(f"Failed to ingest {path}: {e}")
        return results

    def get_virtual_nodes(self) -> list:
        """Create virtual DAG nodes from all ingested datasets."""
        from shared.agentic.dag.parser import NodeSpec, DataSourceConfig, SourceEntry

        nodes = []
        for ds in self.manifest.datasets:
            for col_name in ds.value_columns:
                col_slug = _slugify_column(col_name)
                node_id = f"{ds.file_id}__{col_slug}"
                # Map inferred frequency; default to "monthly" for unknown/irregular
                freq = ds.frequency if ds.frequency in ("daily", "monthly", "quarterly", "annual") else "monthly"
                node = NodeSpec(
                    id=node_id,
                    name=f"{col_name} ({ds.file_id})",
                    frequency=freq,
                    type="continuous",
                    observed=True,
                    source=DataSourceConfig(preferred=[
                        SourceEntry(connector="ingested", dataset=ds.file_id, series=col_name)
                    ]),
                )
                nodes.append(node)
        return nodes

    def register_loaders(self) -> int:
        """Register NODE_LOADERS for all ingested datasets."""
        from shared.engine.data_assembler import NODE_LOADERS

        count = 0
        for ds in self.manifest.datasets:
            for col_name in ds.value_columns:
                col_slug = _slugify_column(col_name)
                node_id = f"{ds.file_id}__{col_slug}"
                if node_id not in NODE_LOADERS:
                    NODE_LOADERS[node_id] = partial(
                        load_ingested_series,
                        ds.output_path, col_name, node_id,
                    )
                    count += 1
        return count


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def auto_ingest(quiet: bool = True) -> IngestPipeline:
    """One-call auto-ingest: scan, ingest new files, register loaders."""
    pipeline = IngestPipeline()
    new_files = pipeline.scan()
    if new_files:
        results = pipeline.ingest_all()
        if not quiet:
            logger.info(f"Auto-ingested {len(results)} file(s)")
    pipeline.register_loaders()
    return pipeline
