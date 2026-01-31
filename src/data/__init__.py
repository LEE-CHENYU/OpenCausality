"""
Data collection and processing modules.
"""

from src.data.base import DataSource, HTTPDataSource
from src.data.kazakhstan_bns import KazakhstanBNSClient, BNSDataType
from src.data.fred_client import FREDClient, FREDSeries
from src.data.baumeister_loader import BaumeisterLoader
from src.data.data_pipeline import DataPipeline, run_pipeline

__all__ = [
    "DataSource",
    "HTTPDataSource",
    "KazakhstanBNSClient",
    "BNSDataType",
    "FREDClient",
    "FREDSeries",
    "BaumeisterLoader",
    "DataPipeline",
    "run_pipeline",
]
