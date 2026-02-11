"""
API Tool Implementations for DataResolverAgent.

Stateless tool functions that the LLM calls via the agent loop to
discover, fetch, parse, and register external data sources.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Directory for API-downloaded data
_API_DATA_DIR = Path("data/raw/api_downloads")


# ---------------------------------------------------------------------------
# Tool definitions (Anthropic tool schema format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_providers",
        "description": (
            "List available free data providers with their API documentation. "
            "Call this first to understand what sources are available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "search_catalog",
        "description": (
            "Search a data provider's indicator catalog for matching time series. "
            "Returns a list of matching indicators with metadata."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "enum": ["world_bank", "fred", "imf", "dbnomics"],
                    "description": "Data provider to search",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (e.g. 'imports share GDP')",
                },
                "country": {
                    "type": "string",
                    "description": "ISO2 country code (default: KZ)",
                    "default": "KZ",
                },
            },
            "required": ["provider", "query"],
        },
    },
    {
        "name": "fetch_api",
        "description": (
            "Make an HTTP GET request to any URL. Returns parsed JSON or raw text. "
            "Use this to fetch data from provider APIs after finding the right indicator."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to fetch",
                },
                "params": {
                    "type": "object",
                    "description": "Optional query parameters",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "parse_time_series",
        "description": (
            "Extract a time series from a raw API response. "
            "Handles World Bank JSON, FRED JSON, CSV, and common formats. "
            "Returns a list of {date, value} records."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "raw_data": {
                    "type": "string",
                    "description": "Raw JSON string from fetch_api",
                },
                "format": {
                    "type": "string",
                    "enum": ["world_bank", "fred", "imf", "csv", "auto"],
                    "description": "Response format to parse",
                },
                "value_key": {
                    "type": "string",
                    "description": "Key containing the numeric value (default: 'value')",
                    "default": "value",
                },
            },
            "required": ["raw_data", "format"],
        },
    },
    {
        "name": "save_and_register",
        "description": (
            "Save parsed time series records to parquet and register a data loader. "
            "This makes the data available to the estimation engine."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "records": {
                    "type": "string",
                    "description": "JSON string of [{date, value}, ...] records",
                },
                "node_id": {
                    "type": "string",
                    "description": "DAG node ID to register this data for",
                },
                "frequency": {
                    "type": "string",
                    "enum": ["daily", "monthly", "quarterly", "annual"],
                    "description": "Data frequency",
                },
                "resample": {
                    "type": "string",
                    "enum": ["monthly_ffill", "monthly_interp", "quarterly_mean", "none"],
                    "description": "Resampling strategy if frequency conversion needed",
                    "default": "none",
                },
                "source_provider": {
                    "type": "string",
                    "description": "Provider name for provenance tracking",
                },
                "indicator_id": {
                    "type": "string",
                    "description": "Provider-specific indicator ID for provenance",
                },
            },
            "required": ["records", "node_id", "frequency"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def list_providers() -> str:
    """List available free data providers with API documentation."""
    providers = [
        {
            "name": "world_bank",
            "full_name": "World Bank World Development Indicators (WDI)",
            "access": "Free, no API key required",
            "search_url": "https://api.worldbank.org/v2/indicator?q={query}&format=json&per_page=20",
            "data_url": "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=500&date=2000:2025",
            "coverage": "200+ countries, 1400+ indicators, annual/quarterly",
            "examples": [
                "NE.IMP.GNFS.ZS - Imports of goods and services (% of GDP)",
                "FP.CPI.TOTL.ZG - Inflation, consumer prices (annual %)",
                "FB.AST.NPER.ZS - Bank nonperforming loans (% of total)",
                "NY.GDP.MKTP.KD.ZG - GDP growth (annual %)",
            ],
        },
        {
            "name": "fred",
            "full_name": "Federal Reserve Economic Data (FRED)",
            "access": "Free, API key in environment as FRED_API_KEY",
            "search_url": "https://api.stlouisfed.org/fred/series/search?search_text={query}&api_key={api_key}&file_type=json&limit=10",
            "data_url": "https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={api_key}&file_type=json",
            "coverage": "800,000+ US and intl time series, various frequencies",
            "examples": [
                "DCOILBRENTEU - Crude Oil Prices: Brent (daily)",
                "FEDFUNDS - Federal Funds Effective Rate (monthly)",
                "CPIAUCSL - Consumer Price Index for All Urban Consumers (monthly)",
            ],
        },
        {
            "name": "imf",
            "full_name": "IMF Financial Soundness Indicators (FSI)",
            "access": "Free, no API key required",
            "search_url": "http://dataservices.imf.org/REST/SDMX_JSON.svc/Dataflow",
            "data_url": "http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/FSI/{freq}.{country}.{indicator}",
            "coverage": "Financial soundness indicators for 100+ countries",
            "examples": [
                "FSANL_PT - Non-performing loans to total gross loans",
                "FSKA_PT - Capital to assets",
                "FSRL_PT - Return on equity",
            ],
        },
        {
            "name": "dbnomics",
            "full_name": "DBnomics (aggregator of 70+ providers)",
            "access": "Free, no API key required",
            "search_url": "https://api.db.nomics.world/v22/search?q={query}&limit=10",
            "data_url": "https://api.db.nomics.world/v22/series/{provider_code}/{dataset_code}/{series_code}",
            "coverage": "Aggregates World Bank, IMF, ECB, Eurostat, OECD, BIS and 60+ more",
            "examples": [
                "WB/WDI/A-NE.IMP.GNFS.ZS-KAZ - World Bank imports for Kazakhstan via DBnomics",
            ],
        },
    ]
    return json.dumps(providers, indent=2)


def search_catalog(provider: str, query: str, country: str = "KZ") -> str:
    """Search a provider's indicator catalog."""
    import httpx

    try:
        if provider == "world_bank":
            url = f"https://api.worldbank.org/v2/indicator?q={query}&format=json&per_page=15"
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()
            if len(data) < 2:
                return json.dumps({"results": [], "message": "No results found"})
            indicators = []
            for item in data[1][:10]:
                indicators.append({
                    "id": item.get("id", ""),
                    "name": item.get("name", ""),
                    "source": item.get("source", {}).get("value", ""),
                    "topics": [t.get("value", "") for t in item.get("topics", [])],
                })
            return json.dumps({"results": indicators})

        elif provider == "fred":
            import os
            api_key = os.environ.get("FRED_API_KEY", "")
            if not api_key:
                return json.dumps({"error": "FRED_API_KEY not set in environment"})
            url = (
                f"https://api.stlouisfed.org/fred/series/search?"
                f"search_text={query}&api_key={api_key}&file_type=json&limit=10"
            )
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()
            series = []
            for s in data.get("seriess", [])[:10]:
                series.append({
                    "id": s.get("id", ""),
                    "title": s.get("title", ""),
                    "frequency": s.get("frequency_short", ""),
                    "units": s.get("units", ""),
                    "observation_start": s.get("observation_start", ""),
                    "observation_end": s.get("observation_end", ""),
                })
            return json.dumps({"results": series})

        elif provider == "imf":
            url = "http://dataservices.imf.org/REST/SDMX_JSON.svc/Dataflow"
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            text = resp.text
            # IMF returns XML-like SDMX; search for query terms
            results = []
            query_lower = query.lower()
            for line in text.split("\n"):
                if query_lower in line.lower():
                    results.append(line.strip()[:200])
                    if len(results) >= 10:
                        break
            return json.dumps({"results": results, "note": "IMF SDMX catalog (text search)"})

        elif provider == "dbnomics":
            url = f"https://api.db.nomics.world/v22/search?q={query}&limit=10"
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", [])[:10]:
                results.append({
                    "provider_code": item.get("provider_code", ""),
                    "dataset_code": item.get("dataset_code", ""),
                    "dataset_name": item.get("dataset_name", ""),
                    "nb_series": item.get("nb_series", 0),
                })
            return json.dumps({"results": results})

        else:
            return json.dumps({"error": f"Unknown provider: {provider}"})

    except Exception as e:
        logger.warning(f"search_catalog({provider}, {query}): {e}")
        return json.dumps({"error": str(e)})


def fetch_api(url: str, params: dict[str, str] | None = None) -> str:
    """Make an HTTP GET request. Returns parsed JSON or raw text."""
    import httpx

    try:
        resp = httpx.get(url, params=params, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "json" in content_type:
            data = resp.json()
            # Truncate large responses for LLM context
            text = json.dumps(data, indent=2)
            if len(text) > 8000:
                text = text[:8000] + "\n... (truncated)"
            return text
        else:
            text = resp.text
            if len(text) > 8000:
                text = text[:8000] + "\n... (truncated)"
            return text
    except Exception as e:
        logger.warning(f"fetch_api({url}): {e}")
        return json.dumps({"error": str(e)})


def parse_time_series(
    raw_data: str,
    format: str,
    value_key: str = "value",
) -> str:
    """Extract a time series from a raw API response."""
    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    except json.JSONDecodeError:
        return json.dumps({"error": "Could not parse raw_data as JSON", "hint": "pass raw JSON string"})

    records: list[dict[str, Any]] = []

    try:
        if format == "world_bank":
            # WB format: [metadata, [{"date": "2020", "value": 1.23, ...}, ...]]
            if isinstance(data, list) and len(data) >= 2:
                for item in data[1] or []:
                    val = item.get("value")
                    date_str = item.get("date", "")
                    if val is not None and date_str:
                        records.append({"date": date_str, "value": float(val)})
            elif isinstance(data, dict):
                # Handle truncated response from fetch
                for item in data.get("data", data.get("records", [])):
                    val = item.get(value_key) or item.get("value")
                    date_str = item.get("date", "")
                    if val is not None:
                        records.append({"date": date_str, "value": float(val)})

        elif format == "fred":
            observations = data.get("observations", [])
            for obs in observations:
                val = obs.get("value", ".")
                if val != "." and val is not None:
                    try:
                        records.append({
                            "date": obs.get("date", ""),
                            "value": float(val),
                        })
                    except (ValueError, TypeError):
                        pass

        elif format == "imf":
            # IMF SDMX compact data
            datasets = data.get("CompactData", {}).get("DataSet", {})
            series = datasets.get("Series", {})
            obs_list = series.get("Obs", [])
            if isinstance(obs_list, dict):
                obs_list = [obs_list]
            for obs in obs_list:
                date_str = obs.get("@TIME_PERIOD", "")
                val = obs.get("@OBS_VALUE")
                if val is not None:
                    records.append({"date": date_str, "value": float(val)})

        elif format in ("csv", "auto"):
            # Try to parse as CSV-like list of dicts
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        date_str = item.get("date", item.get("period", ""))
                        val = item.get(value_key, item.get("value"))
                        if val is not None:
                            records.append({"date": str(date_str), "value": float(val)})

        # Sort by date
        records.sort(key=lambda r: r["date"])

        return json.dumps({
            "records": records,
            "count": len(records),
            "date_range": [records[0]["date"], records[-1]["date"]] if records else [],
        })

    except Exception as e:
        return json.dumps({"error": f"Parse failed: {e}", "format": format})


def save_and_register(
    records: str,
    node_id: str,
    frequency: str,
    resample: str = "none",
    source_provider: str = "",
    indicator_id: str = "",
) -> str:
    """Save parsed time series to parquet and register a data loader."""
    try:
        parsed = json.loads(records) if isinstance(records, str) else records
        record_list = parsed if isinstance(parsed, list) else parsed.get("records", [])

        if not record_list:
            return json.dumps({"error": "No records to save"})

        df = pd.DataFrame(record_list)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])

        if df.empty:
            return json.dumps({"error": "All values were NaN after parsing"})

        # Resample if requested
        series = df["value"]
        if resample == "monthly_ffill" and frequency == "annual":
            series = series.resample("MS").ffill()
        elif resample == "monthly_interp" and frequency == "annual":
            series = series.resample("MS").interpolate(method="linear")
        elif resample == "quarterly_mean" and frequency in ("monthly", "daily"):
            series = series.resample("QS").mean()

        series = series.dropna()
        series.name = node_id

        # Save to parquet
        save_dir = _API_DATA_DIR / source_provider if source_provider else _API_DATA_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = node_id.replace("/", "_").replace(" ", "_")
        save_path = save_dir / f"{safe_name}.parquet"
        series.to_frame(name="value").to_parquet(save_path)

        # Register loader in NODE_LOADERS
        from shared.engine.data_assembler import NODE_LOADERS

        def _loader(_path=save_path, _node_id=node_id) -> pd.Series:
            df_loaded = pd.read_parquet(_path)
            s = df_loaded["value"].dropna()
            s.name = _node_id
            return s

        NODE_LOADERS[node_id] = _loader

        result = {
            "success": True,
            "file_path": str(save_path),
            "rows": len(series),
            "date_range": [str(series.index.min().date()), str(series.index.max().date())],
            "node_id": node_id,
            "registered_loader": True,
            "provenance": {
                "provider": source_provider,
                "indicator": indicator_id,
                "download_time": datetime.now().isoformat(),
                "frequency": frequency,
                "resample": resample,
            },
        }
        logger.info(
            f"DataResolver: saved {len(series)} rows for '{node_id}' "
            f"-> {save_path}"
        )
        return json.dumps(result)

    except Exception as e:
        logger.warning(f"save_and_register({node_id}): {e}")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool executor (dispatch function for the agent loop)
# ---------------------------------------------------------------------------

class WorldBankFallback:
    """Minimal fallback client for DynamicLoaderFactory's _try_api_fallback.

    Fetches a World Bank WDI indicator by series code (e.g. "NE.IMP.GNFS.ZS")
    and returns it as a pd.Series.  Used as the ``api_fallback`` for the
    world_bank connector strategy.
    """

    def fetch_series(self, series: str, country: str = "KAZ") -> pd.Series:
        """Fetch a WDI indicator as a pd.Series."""
        import httpx

        url = (
            f"https://api.worldbank.org/v2/country/{country}/indicator/{series}"
            f"?format=json&per_page=500&date=2000:2025"
        )
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list) or len(data) < 2 or data[1] is None:
            raise ValueError(f"No WB data for {series}/{country}")

        records = []
        for item in data[1]:
            val = item.get("value")
            date_str = item.get("date", "")
            if val is not None and date_str:
                records.append({"date": date_str, "value": float(val)})

        if not records:
            raise ValueError(f"Empty WB series for {series}/{country}")

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Save for future runs
        save_dir = _API_DATA_DIR / "world_bank"
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = series.replace(".", "_")
        save_path = save_dir / f"{safe_name}.parquet"
        df.to_parquet(save_path)
        logger.info(f"WorldBankFallback: saved {len(df)} rows -> {save_path}")

        return df["value"]


def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Execute a tool by name. Used as the tool_executor callback."""
    dispatch = {
        "list_providers": lambda args: list_providers(),
        "search_catalog": lambda args: search_catalog(
            provider=args["provider"],
            query=args["query"],
            country=args.get("country", "KZ"),
        ),
        "fetch_api": lambda args: fetch_api(
            url=args["url"],
            params=args.get("params"),
        ),
        "parse_time_series": lambda args: parse_time_series(
            raw_data=args["raw_data"],
            format=args["format"],
            value_key=args.get("value_key", "value"),
        ),
        "save_and_register": lambda args: save_and_register(
            records=args["records"],
            node_id=args["node_id"],
            frequency=args["frequency"],
            resample=args.get("resample", "none"),
            source_provider=args.get("source_provider", ""),
            indicator_id=args.get("indicator_id", ""),
        ),
    }

    handler = dispatch.get(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    return handler(tool_input)
