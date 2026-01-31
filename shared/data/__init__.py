"""
Shared data infrastructure for Kazakhstan econometric research.

Contains:
- base.py: Abstract DataSource base class
- kazakhstan_bns.py: BNS client for Kazakhstan statistics
- fred_client.py: FRED economic data client
- policy_events.py: Policy calendar for research
- data_pipeline.py: Data orchestration utilities
"""

from shared.data.base import DataSource, HTTPDataSource, DataSourceMetadata
from shared.data.kazakhstan_bns import KazakhstanBNSClient, BNSDataType
from shared.data.fred_client import FREDClient, FREDSeries
from shared.data.policy_events import PolicyEvent, PolicyCalendar, get_kazakhstan_policy_calendar

__all__ = [
    "DataSource",
    "HTTPDataSource",
    "DataSourceMetadata",
    "KazakhstanBNSClient",
    "BNSDataType",
    "FREDClient",
    "FREDSeries",
    "PolicyEvent",
    "PolicyCalendar",
    "get_kazakhstan_policy_calendar",
]
