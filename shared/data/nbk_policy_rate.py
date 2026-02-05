"""
National Bank of Kazakhstan Policy Rate Client.

Fetches NBK base rate decisions and builds time series.

Data source:
- https://nationalbank.kz/en/page/bazovoie-stavka
- Decision archive with historical rate changes

Features:
- Parses decision dates and rate levels
- Builds daily series with forward-fill between decisions
- Aggregates to monthly (end-of-month and average)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from shared.data.base import DataSource

logger = logging.getLogger(__name__)


# NBK base rate decision history (manually curated from NBK announcements)
# Format: (effective_date, rate_pct)
# Source: https://nationalbank.kz/en/page/bazovoie-stavka
NBK_RATE_HISTORY = [
    # 2024
    ("2024-12-02", 15.25),
    ("2024-10-25", 15.25),
    ("2024-09-06", 14.25),
    ("2024-07-26", 14.25),
    ("2024-06-14", 14.50),
    ("2024-04-26", 14.75),
    ("2024-03-01", 14.75),
    ("2024-01-26", 15.25),
    # 2023
    ("2023-12-01", 15.75),
    ("2023-10-27", 15.75),
    ("2023-09-08", 16.00),
    ("2023-07-28", 16.50),
    ("2023-06-09", 16.50),
    ("2023-04-28", 16.75),
    ("2023-03-03", 16.75),
    ("2023-01-27", 16.75),
    # 2022
    ("2022-12-05", 16.75),
    ("2022-10-28", 16.00),
    ("2022-09-05", 14.50),
    ("2022-07-25", 14.50),
    ("2022-06-06", 14.00),
    ("2022-04-25", 14.00),
    ("2022-03-24", 13.50),
    ("2022-02-24", 13.50),  # Emergency hike after Ukraine invasion
    ("2022-01-24", 10.25),
    # 2021
    ("2021-12-06", 9.75),
    ("2021-10-25", 9.75),
    ("2021-09-13", 9.50),
    ("2021-07-26", 9.25),
    ("2021-06-14", 9.25),
    ("2021-04-26", 9.00),
    ("2021-03-01", 9.00),
    ("2021-01-25", 9.00),
    # 2020
    ("2020-12-14", 9.00),
    ("2020-10-26", 9.00),
    ("2020-09-14", 9.00),
    ("2020-07-20", 9.00),
    ("2020-06-08", 9.50),
    ("2020-05-04", 9.50),
    ("2020-04-13", 12.00),  # COVID emergency
    ("2020-03-16", 12.00),  # COVID emergency
    ("2020-01-27", 9.25),
    # 2019
    ("2019-12-09", 9.25),
    ("2019-10-28", 9.25),
    ("2019-09-09", 9.25),
    ("2019-07-15", 9.00),
    ("2019-06-03", 9.00),
    ("2019-04-15", 9.00),
    ("2019-03-04", 9.25),
    ("2019-01-21", 9.25),
    # 2018
    ("2018-12-10", 9.25),
    ("2018-10-29", 9.25),
    ("2018-09-10", 9.00),
    ("2018-07-09", 9.00),
    ("2018-06-04", 9.00),
    ("2018-04-09", 9.25),
    ("2018-03-05", 9.50),
    ("2018-01-15", 9.75),
    # 2017
    ("2017-12-04", 10.25),
    ("2017-10-16", 10.25),
    ("2017-09-04", 10.25),
    ("2017-07-10", 10.50),
    ("2017-06-05", 10.50),
    ("2017-04-10", 11.00),
    ("2017-03-06", 11.00),
    ("2017-01-16", 11.00),
    # 2016
    ("2016-11-28", 12.00),
    ("2016-10-10", 12.50),
    ("2016-09-05", 13.00),
    ("2016-07-11", 13.00),
    ("2016-06-06", 15.00),
    ("2016-05-04", 15.00),
    ("2016-04-04", 15.00),
    ("2016-03-02", 17.00),
    ("2016-02-15", 17.00),
    # 2015 (Tenge float period)
    ("2015-10-02", 16.00),
    ("2015-09-02", 12.00),
    ("2015-08-20", 12.00),  # Tenge float announcement
    ("2015-07-31", 5.50),
    ("2015-06-15", 5.50),
    ("2015-04-20", 5.50),
    ("2015-03-02", 5.50),
    ("2015-02-02", 5.50),
]


@dataclass
class RateDecision:
    """A single NBK rate decision."""

    decision_date: date
    effective_date: date
    new_rate: float
    previous_rate: float | None = None
    change_bps: int | None = None
    decision_type: str = "scheduled"  # scheduled, emergency, technical

    def __post_init__(self):
        if self.previous_rate is not None and self.change_bps is None:
            self.change_bps = int((self.new_rate - self.previous_rate) * 100)


@dataclass
class NBKPolicyRateSeries:
    """Complete policy rate time series."""

    decisions: list[RateDecision]
    daily: pd.DataFrame  # date, rate, days_since_decision
    monthly: pd.DataFrame  # month, eom_rate, avg_rate, decisions_in_month
    metadata: dict = field(default_factory=dict)


class NBKPolicyRateClient(DataSource):
    """
    Client for National Bank of Kazakhstan base rate data.

    Provides:
    - Historical rate decisions with effective dates
    - Daily rate series (forward-filled between decisions)
    - Monthly aggregations (end-of-month, average)

    Example usage:
        client = NBKPolicyRateClient()
        daily = client.build_daily_series()
        monthly = client.aggregate_to_monthly()
    """

    @property
    def source_name(self) -> str:
        return "nbk_policy_rate"

    def __init__(self, cache_dir: Path | None = None):
        """Initialize NBK policy rate client."""
        super().__init__(cache_dir)
        self._decisions: list[RateDecision] | None = None

    def fetch(self, **kwargs: Any) -> pd.DataFrame:
        """
        Fetch base rate series.

        Args:
            frequency: "daily" or "monthly" (default: "monthly")
            start_date: Start date (default: 2015-01-01)
            end_date: End date (default: today)

        Returns:
            DataFrame with rate series
        """
        frequency = kwargs.get("frequency", "monthly")
        start_date = kwargs.get("start_date", "2015-01-01")
        end_date = kwargs.get("end_date", date.today().isoformat())

        if frequency == "daily":
            df = self.build_daily_series(start_date, end_date)
        else:
            df = self.aggregate_to_monthly(start_date, end_date)

        return df

    def fetch_base_rate_decisions(self) -> pd.DataFrame:
        """
        Get all base rate decisions as DataFrame.

        Returns:
            DataFrame with columns: effective_date, rate, previous_rate, change_bps
        """
        decisions = self._get_decisions()

        records = []
        for d in decisions:
            records.append({
                "effective_date": d.effective_date,
                "rate": d.new_rate,
                "previous_rate": d.previous_rate,
                "change_bps": d.change_bps,
                "decision_type": d.decision_type,
            })

        df = pd.DataFrame(records)
        df["effective_date"] = pd.to_datetime(df["effective_date"])
        df = df.sort_values("effective_date").reset_index(drop=True)

        return df

    def build_daily_series(
        self,
        start_date: str | date = "2015-01-01",
        end_date: str | date | None = None,
    ) -> pd.DataFrame:
        """
        Build daily rate series with forward-fill.

        Args:
            start_date: Start date for series
            end_date: End date (default: today)

        Returns:
            DataFrame with columns: date, rate, days_since_decision
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date is None:
            end_date = date.today()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        decisions = self._get_decisions()

        # Build decision lookup
        decision_dates = {d.effective_date: d.new_rate for d in decisions}

        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        records = []
        current_rate = None
        last_decision_date = None

        for dt in date_range:
            dt_date = dt.date()

            # Check for new decision
            if dt_date in decision_dates:
                current_rate = decision_dates[dt_date]
                last_decision_date = dt_date
            elif current_rate is None:
                # Find most recent decision before start
                prior_decisions = [
                    (d.effective_date, d.new_rate)
                    for d in decisions
                    if d.effective_date < dt_date
                ]
                if prior_decisions:
                    prior_decisions.sort(key=lambda x: x[0], reverse=True)
                    last_decision_date, current_rate = prior_decisions[0]

            days_since = (
                (dt_date - last_decision_date).days
                if last_decision_date
                else None
            )

            records.append({
                "date": dt_date,
                "rate": current_rate,
                "days_since_decision": days_since,
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])

        return df

    def aggregate_to_monthly(
        self,
        start_date: str | date = "2015-01-01",
        end_date: str | date | None = None,
    ) -> pd.DataFrame:
        """
        Aggregate daily series to monthly.

        Args:
            start_date: Start date for series
            end_date: End date (default: today)

        Returns:
            DataFrame with columns: month, eom_rate, avg_rate, min_rate, max_rate,
                                     decisions_in_month
        """
        daily = self.build_daily_series(start_date, end_date)

        # Add month column
        daily["month"] = daily["date"].dt.to_period("M")

        # Count decisions per month
        decisions_df = self.fetch_base_rate_decisions()
        decisions_df["month"] = decisions_df["effective_date"].dt.to_period("M")
        decision_counts = decisions_df.groupby("month").size().reset_index(name="decisions_in_month")

        # Aggregate
        monthly = daily.groupby("month").agg(
            eom_rate=("rate", "last"),
            avg_rate=("rate", "mean"),
            min_rate=("rate", "min"),
            max_rate=("rate", "max"),
        ).reset_index()

        # Merge decision counts
        monthly = monthly.merge(decision_counts, on="month", how="left")
        monthly["decisions_in_month"] = monthly["decisions_in_month"].fillna(0).astype(int)

        # Convert period to timestamp
        monthly["month"] = monthly["month"].dt.to_timestamp()

        return monthly

    def get_rate_at_date(self, query_date: date | str) -> float | None:
        """
        Get the base rate effective on a specific date.

        Args:
            query_date: Date to query

        Returns:
            Rate in percent, or None if before first decision
        """
        if isinstance(query_date, str):
            query_date = datetime.strptime(query_date, "%Y-%m-%d").date()

        decisions = self._get_decisions()

        # Find most recent decision on or before query date
        applicable = [
            d for d in decisions
            if d.effective_date <= query_date
        ]

        if not applicable:
            return None

        # Sort by date descending, take first
        applicable.sort(key=lambda x: x.effective_date, reverse=True)
        return applicable[0].new_rate

    def get_regime_periods(self) -> pd.DataFrame:
        """
        Identify monetary policy regime periods.

        Returns:
            DataFrame with regime start/end dates and characteristics
        """
        decisions = self._get_decisions()

        regimes = [
            {
                "regime": "Pre-float",
                "start": date(2015, 1, 1),
                "end": date(2015, 8, 19),
                "description": "Fixed exchange rate regime",
                "avg_rate": 5.5,
            },
            {
                "regime": "Post-float adjustment",
                "start": date(2015, 8, 20),
                "end": date(2016, 12, 31),
                "description": "High rates after tenge float",
                "avg_rate": 14.0,
            },
            {
                "regime": "Normalization",
                "start": date(2017, 1, 1),
                "end": date(2020, 3, 15),
                "description": "Gradual rate reduction",
                "avg_rate": 10.0,
            },
            {
                "regime": "COVID response",
                "start": date(2020, 3, 16),
                "end": date(2020, 7, 19),
                "description": "Emergency rate hikes",
                "avg_rate": 10.5,
            },
            {
                "regime": "Accommodation",
                "start": date(2020, 7, 20),
                "end": date(2022, 2, 23),
                "description": "Low rates for recovery",
                "avg_rate": 9.25,
            },
            {
                "regime": "Inflation fighting",
                "start": date(2022, 2, 24),
                "end": date.today(),
                "description": "Rate hikes for inflation control",
                "avg_rate": 15.0,
            },
        ]

        return pd.DataFrame(regimes)

    def _get_decisions(self) -> list[RateDecision]:
        """Get parsed rate decisions with caching."""
        if self._decisions is None:
            self._decisions = self._parse_rate_history()
        return self._decisions

    def _parse_rate_history(self) -> list[RateDecision]:
        """Parse the hardcoded rate history into RateDecision objects."""
        decisions = []
        previous_rate = None

        # Sort by date ascending
        sorted_history = sorted(NBK_RATE_HISTORY, key=lambda x: x[0])

        for date_str, rate in sorted_history:
            effective_date = datetime.strptime(date_str, "%Y-%m-%d").date()

            # Determine decision type
            decision_type = "scheduled"
            if date_str in ["2020-03-16", "2020-04-13"]:
                decision_type = "emergency"  # COVID
            elif date_str == "2022-02-24":
                decision_type = "emergency"  # Ukraine invasion
            elif date_str == "2015-08-20":
                decision_type = "emergency"  # Tenge float

            decisions.append(RateDecision(
                decision_date=effective_date,  # Approximate
                effective_date=effective_date,
                new_rate=rate,
                previous_rate=previous_rate,
                decision_type=decision_type,
            ))

            previous_rate = rate

        return decisions

    def get_download_instructions(self) -> str:
        """Get instructions for updating rate data."""
        return """
NBK Policy Rate Data Update Instructions
=========================================

The base rate data is curated from NBK announcements.
To update with new decisions:

1. Check NBK base rate page:
   https://nationalbank.kz/en/page/bazovoie-stavka

2. For each new decision, add entry to NBK_RATE_HISTORY in
   shared/data/nbk_policy_rate.py:

   ("YYYY-MM-DD", rate_pct),

3. The date should be the EFFECTIVE date (when rate applies),
   not the announcement date.

4. Test the update:
   from shared.data.nbk_policy_rate import NBKPolicyRateClient
   client = NBKPolicyRateClient()
   print(client.fetch_base_rate_decisions().tail(10))
"""


def get_nbk_policy_rate_client(cache_dir: Path | None = None) -> NBKPolicyRateClient:
    """Get NBK policy rate client instance."""
    return NBKPolicyRateClient(cache_dir)
