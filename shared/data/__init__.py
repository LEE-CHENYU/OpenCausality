"""
Shared data infrastructure for Kazakhstan econometric research.

Contains:
- base.py: Abstract DataSource base class
- kazakhstan_bns.py: BNS client for Kazakhstan statistics
- fred_client.py: FRED economic data client
- policy_events.py: Policy calendar for research
- data_pipeline.py: Data orchestration utilities
- nbk_credit.py: NBK credit quality data client
- imf_fsi.py: IMF Financial Soundness Indicators client
"""

from shared.data.base import DataSource, HTTPDataSource, DataSourceMetadata
from shared.data.kazakhstan_bns import KazakhstanBNSClient, BNSDataType
from shared.data.fred_client import FREDClient, FREDSeries
from shared.data.policy_events import PolicyEvent, PolicyCalendar, get_kazakhstan_policy_calendar
from shared.data.nbk_credit import NBKCreditClient, get_nbk_credit_client
from shared.data.imf_fsi import IMFFSIClient, fetch_kazakhstan_npl

__all__ = [
    # Base classes
    "DataSource",
    "HTTPDataSource",
    "DataSourceMetadata",
    # BNS (Kazakhstan statistics)
    "KazakhstanBNSClient",
    "BNSDataType",
    # FRED (US/global economic data)
    "FREDClient",
    "FREDSeries",
    # Policy events
    "PolicyEvent",
    "PolicyCalendar",
    "get_kazakhstan_policy_calendar",
    # NBK credit data
    "NBKCreditClient",
    "get_nbk_credit_client",
    # IMF FSI
    "IMFFSIClient",
    "fetch_kazakhstan_npl",
]
