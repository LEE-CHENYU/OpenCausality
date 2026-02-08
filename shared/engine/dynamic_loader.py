"""
Dynamic Data Loader Factory.

Reads DAG node source specs and dynamically builds data loaders,
so edges with properly declared source metadata "just work" without
manual NODE_LOADERS / EDGE_NODE_MAP entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from shared.agentic.dag.parser import DAGSpec, EdgeSpec, NodeSpec

logger = logging.getLogger(__name__)

# Base data directory (mirrors data_assembler.py)
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
BACKUP_DIR = DATA_DIR / "backup"
PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Connector strategy definitions
# ---------------------------------------------------------------------------

@dataclass
class ConnectorStrategy:
    """Defines how to resolve local files (and optional API fallback) for a connector."""

    name: str
    local_path_templates: list[str] = field(default_factory=list)
    api_fallback: str | None = None  # module path for fallback client
    single_file: bool = False  # if True, series is a column in a single file
    notes: str = ""


# Each connector maps to ordered local file path candidates + optional API fallback.
_CONNECTOR_STRATEGIES: dict[str, ConnectorStrategy] = {
    "fred": ConnectorStrategy(
        name="fred",
        local_path_templates=[
            "data/raw/fred/{series}.parquet",
            "data/backup/fred/{series}.parquet",
        ],
        api_fallback="shared.data.fred_client.FREDClient",
    ),
    "bns": ConnectorStrategy(
        name="bns",
        local_path_templates=[
            "data/raw/kazakhstan_bns/{dataset}.parquet",
            "data/raw/kazakhstan_bns/{series}.parquet",
            "data/processed/fx_passthrough/{series}.parquet",
        ],
        api_fallback="shared.data.kazakhstan_bns.KazakhstanBNSClient",
        notes="Also tries fuzzy directory scan as last resort",
    ),
    "baumeister": ConnectorStrategy(
        name="baumeister",
        local_path_templates=[
            "data/raw/baumeister_shocks/shocks.parquet",
        ],
        single_file=True,
        notes="Single file; resolve column by series name",
    ),
    "kspi_quarterly": ConnectorStrategy(
        name="kspi_quarterly",
        local_path_templates=[],
        notes="Delegates to existing _load_kspi_quarterly()",
    ),
    "nbk": ConnectorStrategy(
        name="nbk",
        local_path_templates=[
            "data/raw/nbk/{series}.parquet",
            "data/raw/nbk/usd_kzt.parquet",
        ],
    ),
    "world_bank": ConnectorStrategy(
        name="world_bank",
        local_path_templates=[
            "data/backup/worldbank/{series}.parquet",
        ],
    ),
    "ingested": ConnectorStrategy(
        name="ingested",
        local_path_templates=[
            "data/processed/ingested/{dataset}.parquet",
        ],
        single_file=True,
        notes="Auto-ingested user data; series resolves to column within dataset parquet",
    ),
}


# ---------------------------------------------------------------------------
# Transform implementations
# ---------------------------------------------------------------------------

def _ar1_innovation(s: pd.Series) -> pd.Series:
    """Extract AR(1) innovation (reuse from data_assembler)."""
    from shared.engine.data_assembler import _ar1_innovation as _assembler_ar1
    return _assembler_ar1(s)


def _transform_log(s: pd.Series) -> pd.Series:
    """Log transform with floor clipping."""
    return np.log(s.clip(lower=1e-6))


def _transform_diff(s: pd.Series) -> pd.Series:
    """First difference."""
    return s.diff().dropna()


def _transform_log_return(s: pd.Series) -> pd.Series:
    """Log-return: log().diff()."""
    return np.log(s.clip(lower=1e-6)).diff().dropna()


def _transform_aggregate_mean_to_quarter(s: pd.Series) -> pd.Series:
    """Resample to quarterly mean (reuse pattern from data_assembler)."""
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.resample("QS").mean().dropna()


def _transform_aggregate_mean_to_month(s: pd.Series) -> pd.Series:
    """Resample to monthly mean (reuse pattern from data_assembler)."""
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.resample("MS").mean().dropna()


_TRANSFORM_FNS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "innovation": _ar1_innovation,
    "log": _transform_log,
    "diff": _transform_diff,
    "log_return": _transform_log_return,
    "aggregate_mean_to_quarter": _transform_aggregate_mean_to_quarter,
    "aggregate_mean_to_month": _transform_aggregate_mean_to_month,
}


# ---------------------------------------------------------------------------
# File-loading helpers
# ---------------------------------------------------------------------------

def _load_series_from_file(
    path: Path,
    series_hint: str,
    node_id: str,
) -> pd.Series:
    """Load a single series from a parquet or CSV file.

    Resolution order for column selection:
    1. Exact match on series_hint
    2. Case-insensitive match
    3. Substring match
    4. Column named 'value'
    5. First numeric column
    """
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    # Set DatetimeIndex
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif "quarter" in df.columns:
        # Handle quarterly labels like '2015Q3'
        df["_date"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp()
        df = df.set_index("_date").sort_index()
    elif "year" in df.columns and "month" in df.columns:
        df["_date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
        )
        df = df.set_index("_date").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # Column resolution
    col = _resolve_column(df, series_hint)
    s = df[col].dropna()

    # If there are duplicate index entries (panel data), aggregate by mean
    if isinstance(s.index, pd.DatetimeIndex) and s.index.duplicated().any():
        s = s.groupby(s.index).mean()

    s.name = node_id
    return s


def _resolve_column(df: pd.DataFrame, hint: str) -> str:
    """Resolve which column to use from a DataFrame."""
    cols = list(df.columns)

    # 1. Exact match
    if hint in cols:
        return hint

    # 2. Case-insensitive match
    lower_map = {c.lower(): c for c in cols}
    if hint.lower() in lower_map:
        return lower_map[hint.lower()]

    # 3. Substring match
    for c in cols:
        if hint.lower() in c.lower():
            return c

    # 4. Column named 'value'
    if "value" in lower_map:
        return lower_map["value"]

    # 5. First numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return numeric_cols[0]

    # Last resort: first column
    return cols[0]


def _apply_transforms(
    series: pd.Series,
    transform_names: list[str],
) -> pd.Series:
    """Apply transforms in declared order, skipping unknown with warning."""
    for name in transform_names:
        fn = _TRANSFORM_FNS.get(name)
        if fn is None:
            logger.warning(f"Unknown transform '{name}', skipping")
            continue
        series = fn(series)
    return series


# ---------------------------------------------------------------------------
# Fuzzy file matching helpers
# ---------------------------------------------------------------------------

_CONNECTOR_SCAN_DIRS: dict[str, list[Path]] = {
    "fred": [RAW_DIR / "fred", BACKUP_DIR / "fred"],
    "bns": [RAW_DIR / "kazakhstan_bns", PROCESSED_DIR / "fx_passthrough"],
    "baumeister": [RAW_DIR / "baumeister_shocks"],
    "nbk": [RAW_DIR / "nbk"],
    "world_bank": [BACKUP_DIR / "worldbank"],
    "ingested": [PROCESSED_DIR / "ingested"],
}


def _get_scan_dirs(connector: str) -> list[Path]:
    """Get directories to scan for fuzzy file matching."""
    return _CONNECTOR_SCAN_DIRS.get(connector, [])


def _fuzzy_find_file(
    directory: Path,
    series: str,
    dataset: str,
    node_id: str,
) -> Path | None:
    """Find a parquet file in a directory by fuzzy-matching keywords.

    Tries matching tokens from the node_id, series, and dataset names
    against available filenames.
    """
    if not directory.exists():
        return None

    parquets = list(directory.glob("*.parquet"))
    if not parquets:
        return None

    # Build keyword tokens from series, dataset, and node_id
    keywords = set()
    for term in (series, dataset, node_id):
        for token in term.lower().replace("_", " ").split():
            if len(token) >= 3:  # skip tiny tokens
                keywords.add(token)

    # Score each file by number of keyword matches
    best_path: Path | None = None
    best_score = 0
    for p in parquets:
        stem = p.stem.lower()
        score = sum(1 for kw in keywords if kw in stem)
        if score > best_score:
            best_score = score
            best_path = p

    # Require at least 1 keyword match
    if best_score >= 1:
        return best_path
    return None


# ---------------------------------------------------------------------------
# DynamicLoaderFactory
# ---------------------------------------------------------------------------

class DynamicLoaderFactory:
    """Bridges DAG source specs to data loading.

    Reads node.source.preferred[0] for connector/dataset/series,
    looks up a connector strategy, and builds closures that:
      1. Try local file paths in order
      2. Try API fallback if available
      3. Apply declared transforms

    Dynamically registers results into NODE_LOADERS and EDGE_NODE_MAP.
    """

    def __init__(self) -> None:
        self._connector_strategies = dict(_CONNECTOR_STRATEGIES)
        self._transform_fns = dict(_TRANSFORM_FNS)

    def try_build_loader(self, node: NodeSpec) -> Callable[[], pd.Series] | None:
        """Build a loader closure for a DAG node from its source spec.

        Returns None if no strategy exists or no source spec is available.
        """
        if not node.source or not node.source.preferred:
            return None

        source = node.source.preferred[0]
        connector = source.connector
        dataset = source.dataset
        series = source.series
        node_id = node.id
        transforms = list(node.transforms)

        strategy = self._connector_strategies.get(connector)
        if strategy is None:
            logger.debug(f"No connector strategy for '{connector}' (node {node_id})")
            return None

        # Special case: kspi_quarterly delegates to existing loader
        if connector == "kspi_quarterly":
            return self._build_kspi_loader(node_id, series, transforms)

        def _loader(
            _strategy=strategy,
            _dataset=dataset,
            _series=series,
            _node_id=node_id,
            _transforms=transforms,
            _connector=connector,
        ) -> pd.Series:
            # Try local file paths
            for template in _strategy.local_path_templates:
                path = Path(template.format(
                    series=_series,
                    dataset=_dataset,
                ))
                if path.exists():
                    logger.debug(f"DynamicLoader [{_node_id}]: loading from {path}")
                    s = _load_series_from_file(path, _series, _node_id)
                    return _apply_transforms(s, _transforms)

            # Try fuzzy directory scan: search for parquet files containing
            # keywords from the series or dataset name
            for scan_dir in _get_scan_dirs(_connector):
                match = _fuzzy_find_file(scan_dir, _series, _dataset, _node_id)
                if match is not None:
                    logger.debug(f"DynamicLoader [{_node_id}]: fuzzy match {match}")
                    s = _load_series_from_file(match, _series, _node_id)
                    return _apply_transforms(s, _transforms)

            # Try API fallback
            if _strategy.api_fallback:
                s = self._try_api_fallback(
                    _strategy.api_fallback, _dataset, _series, _node_id,
                )
                if s is not None:
                    return _apply_transforms(s, _transforms)

            raise FileNotFoundError(
                f"DynamicLoader [{_node_id}]: no data found for "
                f"connector={_connector}, dataset={_dataset}, series={_series}"
            )

        return _loader

    def _build_kspi_loader(
        self,
        node_id: str,
        series: str,
        transforms: list[str],
    ) -> Callable[[], pd.Series]:
        """Build a loader that delegates to _load_kspi_quarterly()."""
        def _loader(
            _series=series,
            _node_id=node_id,
            _transforms=transforms,
        ) -> pd.Series:
            from shared.engine.data_assembler import _load_kspi_quarterly
            df = _load_kspi_quarterly()
            col = _resolve_column(df, _series)
            s = df[col].dropna().rename(_node_id)
            return _apply_transforms(s, _transforms)
        return _loader

    @staticmethod
    def _try_api_fallback(
        module_path: str,
        dataset: str,
        series: str,
        node_id: str,
    ) -> pd.Series | None:
        """Try loading via API client as a fallback."""
        try:
            parts = module_path.rsplit(".", 1)
            mod_path, cls_name = parts[0], parts[1]
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            client = cls()
            # Try common client API patterns
            for method_name in ("get_series", "fetch_series", "download"):
                method = getattr(client, method_name, None)
                if method is not None:
                    s = method(series)
                    if isinstance(s, pd.Series) and len(s) > 0:
                        s.name = node_id
                        return s
                    elif isinstance(s, pd.DataFrame) and len(s) > 0:
                        col = _resolve_column(s, series)
                        result = s[col].dropna()
                        result.name = node_id
                        return result
        except Exception as e:
            logger.debug(f"API fallback failed for {node_id}: {e}")
        return None

    def register_edge_from_spec(
        self,
        edge: EdgeSpec,
        dag: DAGSpec,
    ) -> bool:
        """Register a single edge's loaders and mapping.

        Returns True if both nodes are loadable and the edge is registered.
        """
        from shared.engine.data_assembler import NODE_LOADERS, EDGE_NODE_MAP

        from_node = dag.get_node(edge.from_node)
        to_node = dag.get_node(edge.to_node)

        if from_node is None or to_node is None:
            logger.warning(
                f"DynamicLoader: edge {edge.id} references missing node(s): "
                f"from={edge.from_node}, to={edge.to_node}"
            )
            return False

        # Build loaders for nodes not already registered
        for node in (from_node, to_node):
            if node.id not in NODE_LOADERS:
                loader = self.try_build_loader(node)
                if loader is None:
                    logger.warning(
                        f"DynamicLoader: cannot build loader for node '{node.id}'"
                    )
                    return False
                NODE_LOADERS[node.id] = loader
                logger.info(f"DynamicLoader: registered loader for node '{node.id}'")

        # Register edge mapping
        EDGE_NODE_MAP[edge.id] = (edge.from_node, edge.to_node)
        logger.info(
            f"DynamicLoader: registered edge '{edge.id}' "
            f"({edge.from_node} -> {edge.to_node})"
        )
        return True

    def auto_populate_from_dag(
        self,
        dag: DAGSpec,
    ) -> dict[str, str]:
        """Scan all edges in the DAG and register missing ones.

        Skips edges already in EDGE_NODE_MAP (backward-compatible).

        Returns:
            {edge_id: "registered" | "skipped" | "failed"}
        """
        from shared.engine.data_assembler import EDGE_NODE_MAP

        results: dict[str, str] = {}

        for edge in dag.edges:
            if edge.id in EDGE_NODE_MAP:
                results[edge.id] = "skipped"
                continue

            ok = self.register_edge_from_spec(edge, dag)
            results[edge.id] = "registered" if ok else "failed"

        return results
