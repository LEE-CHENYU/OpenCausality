"""
DataScout Agent: Auto-download missing data using existing clients.

DataScout checks which DAG nodes lack local data and uses the
registered data clients (FRED, BNS, NBK) to download them,
respecting a configurable byte budget.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result / report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DownloadResult:
    """Outcome of a single node download attempt."""

    node_id: str
    connector: str
    success: bool
    size_bytes: int = 0
    file_path: Path | None = None
    error: str = ""
    row_count: int = 0


@dataclass
class DataCard:
    """Metadata card for a downloaded dataset."""

    node_id: str
    connector: str
    dataset: str
    series: str
    download_time: datetime
    file_path: str
    size_bytes: int
    row_count: int
    date_range: tuple[str, str] | None = None
    quality_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "connector": self.connector,
            "dataset": self.dataset,
            "series": self.series,
            "download_time": self.download_time.isoformat(),
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "row_count": self.row_count,
            "date_range": list(self.date_range) if self.date_range else None,
            "quality_notes": self.quality_notes,
            "provenance": {
                "source": "data_scout",
                "added_at": self.download_time.strftime("%Y-%m-%d"),
                "connector": self.connector,
                "dataset": self.dataset,
            },
        }

    def save(self, output_dir: Path) -> Path:
        """Save data card to YAML."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.node_id}.yaml"
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
        return path


@dataclass
class DataScoutReport:
    """Summary of a DataScout download run."""

    total_nodes: int
    downloaded: int
    skipped: int
    failed: int
    total_bytes: int
    budget_bytes: int
    results: list[DownloadResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Connector → Client registry (static connectors are skipped)
# ---------------------------------------------------------------------------

_STATIC_CONNECTORS = frozenset({
    "baumeister", "kspi_quarterly", "world_bank", "panel",
})


def _is_static(connector: str) -> bool:
    """Return True for connectors that reference on-disk / static data."""
    return connector in _STATIC_CONNECTORS or connector == ""


# ---------------------------------------------------------------------------
# DataScout
# ---------------------------------------------------------------------------


class DataScout:
    """
    Auto-download missing DAG node data using existing data clients.

    Budget tracking: after each download, file size is accumulated against
    ``budget_bytes``. When exhausted, remaining nodes are skipped.
    All downloads are wrapped in try/except — failures are logged, never block.

    When static connectors fail, DataScout can delegate to a
    DataResolverAgent for LLM-powered adaptive data discovery.
    """

    def __init__(
        self,
        budget_mb: int = 100,
        output_dir: Path | None = None,
        llm_client: Any | None = None,
    ):
        self.budget_bytes = budget_mb * 1024 * 1024
        self.output_dir = output_dir or Path("outputs/agentic/cards/data")
        self._bytes_used = 0
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_missing(
        self,
        dag: Any,  # DAGSpec
        missing_nodes: list[str],
    ) -> DataScoutReport:
        """
        Attempt to download data for each missing node.

        Args:
            dag: The DAGSpec (shared.agentic.dag.parser.DAGSpec)
            missing_nodes: Node IDs that need data

        Returns:
            DataScoutReport
        """
        node_map = {n.id: n for n in dag.nodes}
        self._current_dag = dag  # stash for DataResolverAgent fallback
        results: list[DownloadResult] = []
        downloaded = 0
        skipped = 0
        failed = 0

        for node_id in missing_nodes:
            node = node_map.get(node_id)
            if node is None:
                logger.warning(f"DataScout: node '{node_id}' not in DAG, skipping")
                skipped += 1
                continue

            # Budget check
            if self._bytes_used >= self.budget_bytes:
                logger.info(
                    f"DataScout: budget exhausted "
                    f"({self._bytes_used / 1024 / 1024:.1f} MB used), "
                    f"skipping remaining nodes"
                )
                skipped += len(missing_nodes) - (downloaded + skipped + failed)
                break

            result = self._download_node(node)
            results.append(result)

            if result.success:
                downloaded += 1
                self._bytes_used += result.size_bytes
                card = self._generate_data_card(node, result)
                if card:
                    card.save(self.output_dir)
            elif result.error == "static":
                skipped += 1
            else:
                failed += 1

        return DataScoutReport(
            total_nodes=len(missing_nodes),
            downloaded=downloaded,
            skipped=skipped,
            failed=failed,
            total_bytes=self._bytes_used,
            budget_bytes=self.budget_bytes,
            results=results,
        )

    def generate_user_guidance(
        self,
        report: DataScoutReport,
        dag: Any,
    ) -> list[str]:
        """Generate actionable guidance for failed downloads.

        For each node that failed to download, produce a message telling
        the user how to manually provide the data.

        Args:
            report: The DataScoutReport from download_missing().
            dag: The DAGSpec (for node name lookup).

        Returns:
            List of guidance strings, one per failed node.
        """
        node_map = {n.id: n for n in dag.nodes}
        guidance: list[str] = []

        for result in report.results:
            if result.success or result.error == "static":
                continue
            node = node_map.get(result.node_id)
            node_name = getattr(node, "name", result.node_id) if node else result.node_id
            msg = (
                f"Node '{result.node_id}': Auto-download failed ({result.error}).\n"
                f"  -> Drop data file into data/raw/ and run: opencausality data ingest\n"
                f"  -> Expected: time-series with date column and '{node_name}' values\n"
                f"  -> Supported formats: CSV, Excel, Parquet, JSON"
            )
            guidance.append(msg)

        return guidance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_node(self, node: Any) -> DownloadResult:
        """Try to download data for a single DAG node."""
        # Resolve connector + series from source spec
        connector = ""
        dataset = ""
        series = ""
        if node.source and node.source.preferred:
            src = node.source.preferred[0]
            connector = src.connector
            dataset = src.dataset
            series = getattr(src, "series", "")

        if _is_static(connector):
            return DownloadResult(
                node_id=node.id, connector=connector,
                success=False, error="static",
            )

        try:
            if connector == "fred":
                return self._fetch_fred(node.id, series or dataset)
            elif connector == "bns":
                return self._fetch_bns(node.id, dataset)
            elif connector == "nbk":
                return self._fetch_nbk(node.id)
            else:
                # Unknown connector — try DataResolverAgent if available
                return self._try_resolver_fallback(
                    node, f"unknown connector: {connector}",
                )
        except Exception as e:
            logger.warning(f"DataScout: download failed for {node.id}: {e}")
            # Try DataResolverAgent as last resort
            return self._try_resolver_fallback(node, str(e))

    # -- FRED --

    def _fetch_fred(self, node_id: str, series: str) -> DownloadResult:
        from shared.data.fred_client import FREDClient

        client = FREDClient()
        df = client.fetch(series=series)
        save_path = Path(f"data/raw/fred/{series}.parquet")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)

        size = os.path.getsize(save_path)
        logger.info(
            f"DataScout: FRED/{series} -> {save_path} "
            f"({len(df)} rows, {size / 1024:.1f} KB)"
        )
        return DownloadResult(
            node_id=node_id, connector="fred", success=True,
            size_bytes=size, file_path=save_path, row_count=len(df),
        )

    # -- BNS --

    def _fetch_bns(self, node_id: str, dataset: str) -> DownloadResult:
        from shared.data.kazakhstan_bns import KazakhstanBNSClient, BNSDataType

        client = KazakhstanBNSClient()
        data_type = BNSDataType(dataset)
        df = client.fetch(data_type)
        save_path = Path(f"data/raw/kazakhstan_bns/{dataset}.parquet")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)

        size = os.path.getsize(save_path)
        logger.info(
            f"DataScout: BNS/{dataset} -> {save_path} "
            f"({len(df)} rows, {size / 1024:.1f} KB)"
        )
        return DownloadResult(
            node_id=node_id, connector="bns", success=True,
            size_bytes=size, file_path=save_path, row_count=len(df),
        )

    # -- NBK (Exchange Rate) --

    def _fetch_nbk(self, node_id: str) -> DownloadResult:
        from shared.data.exchange_rate import ExchangeRateClient

        client = ExchangeRateClient()
        df = client.fetch(tier=1)
        save_path = Path("data/raw/nbk/usd_kzt.parquet")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)

        size = os.path.getsize(save_path)
        logger.info(
            f"DataScout: NBK/usd_kzt -> {save_path} "
            f"({len(df)} rows, {size / 1024:.1f} KB)"
        )
        return DownloadResult(
            node_id=node_id, connector="nbk", success=True,
            size_bytes=size, file_path=save_path, row_count=len(df),
        )

    # -- DataResolverAgent fallback --

    def _try_resolver_fallback(
        self, node: Any, original_error: str,
    ) -> DownloadResult:
        """Try the LLM-powered DataResolverAgent as a last resort.

        Only available when DataScout is initialized with an llm_client.
        """
        if self.llm_client is None:
            return DownloadResult(
                node_id=node.id, connector="",
                success=False, error=original_error,
            )

        logger.info(
            f"DataScout: static connectors failed for '{node.id}', "
            f"delegating to DataResolverAgent"
        )
        try:
            from shared.agentic.agents.data_resolver import DataResolverAgent

            # We need a dag reference; get it from the node's parent if available
            dag = getattr(self, "_current_dag", None)
            agent = DataResolverAgent(
                llm_client=self.llm_client,
                dag=dag,
                dynamic_loader=None,
            )
            result = agent.resolve_node(node.id)

            if result.success:
                return DownloadResult(
                    node_id=node.id,
                    connector=f"api:{result.provider}",
                    success=True,
                    size_bytes=0,  # size tracked separately
                    file_path=Path(result.file_path) if result.file_path else None,
                    row_count=result.rows,
                )
            else:
                return DownloadResult(
                    node_id=node.id, connector="",
                    success=False,
                    error=f"{original_error}; resolver also failed: {result.error}",
                )
        except Exception as e:
            logger.warning(f"DataScout: DataResolverAgent fallback failed for {node.id}: {e}")
            return DownloadResult(
                node_id=node.id, connector="",
                success=False, error=f"{original_error}; resolver error: {e}",
            )

    # -- Data card generation --

    def _generate_data_card(
        self, node: Any, result: DownloadResult,
    ) -> DataCard | None:
        """Generate a DataCard for a successful download."""
        if not result.success or result.file_path is None:
            return None

        connector = result.connector
        dataset = ""
        series = ""
        if node.source and node.source.preferred:
            src = node.source.preferred[0]
            dataset = src.dataset
            series = getattr(src, "series", "")

        # Try to infer date range from parquet
        date_range = None
        try:
            import pandas as pd

            df = pd.read_parquet(result.file_path)
            if "date" in df.columns:
                date_range = (str(df["date"].min()), str(df["date"].max()))
            elif hasattr(df.index, "min"):
                date_range = (str(df.index.min()), str(df.index.max()))
        except Exception:
            pass

        quality_notes: list[str] = []
        if result.row_count < 30:
            quality_notes.append("small_sample")

        return DataCard(
            node_id=node.id,
            connector=connector,
            dataset=dataset,
            series=series,
            download_time=datetime.now(),
            file_path=str(result.file_path),
            size_bytes=result.size_bytes,
            row_count=result.row_count,
            date_range=date_range,
            quality_notes=quality_notes,
        )
