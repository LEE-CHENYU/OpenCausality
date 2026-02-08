"""
OpenCausality Platform settings.
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

    # LLM
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    openai_api_key: str = Field(default="", description="OpenAI API key (for litellm provider)")
    llm_provider: str = Field(
        default="codex",
        description="LLM provider: anthropic | litellm | codex | claude_cli",
    )
    llm_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Model ID for chosen LLM provider",
    )
    codex_model: str = Field(
        default="gpt-5.3-codex",
        description="Model for codex CLI provider",
    )

    # API Keys
    fred_api_key: str = Field(default="", description="FRED API key")

    # Literature search
    semantic_scholar_api_key: str = Field(default="", description="Semantic Scholar API key")
    openalex_mailto: str = Field(default="", description="Email for OpenAlex polite pool")
    unpaywall_email: str = Field(default="", description="Email for Unpaywall API")
    core_api_key: str = Field(default="", description="CORE API key")

    # Directories
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Project root directory",
    )
    cache_dir: Path = Field(default=Path(".cache"), description="Cache directory")
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    output_dir: Path = Field(default=Path("outputs"), description="Output directory")

    # Defaults
    default_dag_path: str = Field(
        default="config/agentic/dags/kspi_k2_full.yaml",
        description="Default DAG specification path",
    )
    default_query_mode: str = Field(
        default="REDUCED_FORM",
        description="Default query mode: STRUCTURAL | REDUCED_FORM | DESCRIPTIVE",
    )

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
