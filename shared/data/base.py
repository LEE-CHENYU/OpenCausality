"""
Abstract base classes for data sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib
import json
import logging

import pandas as pd
from diskcache import Cache

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class DataSourceMetadata:
    """Metadata about a data source fetch."""

    source_name: str
    fetch_time: datetime
    url: str | None = None
    cache_key: str | None = None
    row_count: int | None = None
    columns: list[str] = field(default_factory=list)
    date_range: tuple[str, str] | None = None
    notes: str = ""


class DataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, cache_dir: Path | None = None):
        settings = get_settings()
        self.cache_dir = cache_dir or settings.project_root / settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(self.cache_dir / self.source_name))
        self._metadata: list[DataSourceMetadata] = []

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier for this data source."""
        pass

    @abstractmethod
    def fetch(self, **kwargs: Any) -> pd.DataFrame:
        """Fetch data from the source."""
        pass

    def _cache_key(self, **kwargs: Any) -> str:
        """Generate cache key from parameters."""
        key_data = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def get_cached(self, cache_key: str) -> pd.DataFrame | None:
        """Retrieve data from cache if available."""
        try:
            data = self._cache.get(cache_key)
            if data is not None:
                logger.debug(f"Cache hit for {self.source_name}: {cache_key}")
                return pd.DataFrame(data)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None

    def set_cached(
        self, cache_key: str, df: pd.DataFrame, ttl_seconds: int | None = None
    ) -> None:
        """Store data in cache."""
        settings = get_settings()
        ttl = ttl_seconds or (settings.cache_ttl_days * 86400)
        try:
            self._cache.set(cache_key, df.to_dict("records"), expire=ttl)
            logger.debug(f"Cached {self.source_name}: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear_cache(self) -> None:
        """Clear all cached data for this source."""
        self._cache.clear()
        logger.info(f"Cleared cache for {self.source_name}")

    def fetch_with_cache(self, **kwargs: Any) -> pd.DataFrame:
        """Fetch data with caching."""
        cache_key = self._cache_key(**kwargs)

        # Try cache first
        cached = self.get_cached(cache_key)
        if cached is not None:
            return cached

        # Fetch fresh data
        logger.info(f"Fetching {self.source_name} with params: {kwargs}")
        df = self.fetch(**kwargs)

        # Cache result
        self.set_cached(cache_key, df)

        return df

    def save_raw(self, df: pd.DataFrame, filename: str) -> Path:
        """Save raw data to disk."""
        settings = get_settings()
        raw_dir = settings.project_root / settings.raw_data_dir / self.source_name
        raw_dir.mkdir(parents=True, exist_ok=True)

        filepath = raw_dir / filename
        if filename.endswith(".parquet"):
            df.to_parquet(filepath, index=False)
        elif filename.endswith(".csv"):
            df.to_csv(filepath, index=False)
        else:
            df.to_parquet(filepath.with_suffix(".parquet"), index=False)

        logger.info(f"Saved raw data to {filepath}")
        return filepath


class HTTPDataSource(DataSource):
    """Base class for HTTP-based data sources."""

    def __init__(self, cache_dir: Path | None = None):
        super().__init__(cache_dir)
        self._client = None

    @property
    def client(self):
        """Lazy-loaded HTTP client."""
        if self._client is None:
            import httpx

            settings = get_settings()
            self._client = httpx.Client(
                timeout=settings.http_timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; KazakhstanResearch/1.0)"
                },
            )
        return self._client

    def __del__(self):
        if self._client is not None:
            self._client.close()
