"""
Pydantic settings for Kazakhstan Welfare Model.
"""

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    fred_api_key: str = Field(default="", description="FRED API key")

    # Directories
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Project root directory",
    )
    cache_dir: Path = Field(default=Path(".cache"), description="Cache directory")
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    output_dir: Path = Field(default=Path("outputs"), description="Output directory")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # Data collection
    http_timeout: float = Field(default=30.0, description="HTTP request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts for HTTP requests")
    cache_ttl_days: int = Field(default=7, description="Cache TTL in days")

    # BNS Settings
    bns_base_url: str = Field(
        default="https://stat.gov.kz",
        description="Kazakhstan Bureau of National Statistics base URL",
    )

    # Model settings
    baseline_start_year: int = Field(
        default=2010, description="Start year for baseline exposure calculation"
    )
    baseline_end_year: int = Field(
        default=2013, description="End year for baseline exposure calculation"
    )

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def crosswalks_dir(self) -> Path:
        return self.data_dir / "crosswalks"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
