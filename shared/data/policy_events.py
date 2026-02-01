"""
Policy events calendar for Kazakhstan econometric research.

Provides a structured repository of policy events relevant to research:
- Minimum wage changes
- Credit policy changes (DSTI, DTI limits)
- Pension eligibility rules
- TSA (targeted social assistance) thresholds
- Regional reforms

This is CRITICAL for avoiding policy confounds in causal inference.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    """Types of policy events."""

    MINIMUM_WAGE = "minimum_wage"
    CREDIT_POLICY = "credit_policy"
    PENSION = "pension"
    SOCIAL_ASSISTANCE = "social_assistance"
    REGIONAL_REFORM = "regional_reform"
    FISCAL = "fiscal"
    MONETARY = "monetary"
    FX_DEPRECIATION = "fx_depreciation"
    OTHER = "other"


@dataclass
class PolicyEvent:
    """A single policy event."""

    name: str
    policy_type: PolicyType
    effective_date: date
    description: str
    source: str
    source_url: str | None = None
    impact_description: str | None = None
    affected_groups: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "policy_type": self.policy_type.value,
            "effective_date": self.effective_date.isoformat(),
            "description": self.description,
            "source": self.source,
            "source_url": self.source_url,
            "impact_description": self.impact_description,
            "affected_groups": self.affected_groups,
            "metadata": self.metadata,
        }


class PolicyCalendar:
    """
    Calendar of policy events for causal inference.

    Use this to:
    1. Check for policy confounds in your analysis window
    2. Define treatment periods around policy changes
    3. Exclude confounded periods from estimation
    """

    def __init__(self):
        self.events: list[PolicyEvent] = []

    def add_event(self, event: PolicyEvent) -> None:
        """Add a policy event to the calendar."""
        self.events.append(event)
        self.events.sort(key=lambda e: e.effective_date)

    def get_events_in_window(
        self,
        start_date: date,
        end_date: date,
        policy_types: list[PolicyType] | None = None,
    ) -> list[PolicyEvent]:
        """
        Get all events in a date window.

        Args:
            start_date: Window start
            end_date: Window end
            policy_types: Filter by policy type (optional)

        Returns:
            List of events in the window
        """
        events = [
            e for e in self.events
            if start_date <= e.effective_date <= end_date
        ]

        if policy_types:
            events = [e for e in events if e.policy_type in policy_types]

        return events

    def check_confounds(
        self,
        analysis_start: date,
        analysis_end: date,
        exclude_types: list[PolicyType] | None = None,
    ) -> list[PolicyEvent]:
        """
        Check for potential policy confounds in analysis window.

        Args:
            analysis_start: Analysis start date
            analysis_end: Analysis end date
            exclude_types: Policy types to consider as confounds

        Returns:
            List of potential confounding events
        """
        if exclude_types is None:
            # By default, check for credit policy and fiscal confounds
            exclude_types = [PolicyType.CREDIT_POLICY, PolicyType.FISCAL]

        confounds = self.get_events_in_window(
            analysis_start, analysis_end, exclude_types
        )

        if confounds:
            logger.warning(
                f"POTENTIAL CONFOUNDS in analysis window "
                f"({analysis_start} to {analysis_end}):"
            )
            for event in confounds:
                logger.warning(f"  - {event.effective_date}: {event.name}")

        return confounds

    def get_minimum_wage_series(self) -> pd.DataFrame:
        """Get minimum wage time series."""
        mw_events = [
            e for e in self.events
            if e.policy_type == PolicyType.MINIMUM_WAGE
        ]

        records = []
        for event in mw_events:
            amount = event.metadata.get("amount_tenge")
            if amount is not None:
                records.append({
                    "effective_date": event.effective_date,
                    "minimum_wage_tenge": amount,
                    "source": event.source,
                })

        return pd.DataFrame(records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all events to DataFrame."""
        return pd.DataFrame([e.to_dict() for e in self.events])


def get_kazakhstan_policy_calendar() -> PolicyCalendar:
    """
    Get the Kazakhstan policy calendar with all known events.

    Returns:
        PolicyCalendar with Kazakhstan-specific policy events
    """
    calendar = PolicyCalendar()

    # =========================================================================
    # MINIMUM WAGE EVENTS
    # =========================================================================

    calendar.add_event(PolicyEvent(
        name="Minimum Wage Increase 2023",
        policy_type=PolicyType.MINIMUM_WAGE,
        effective_date=date(2023, 1, 1),
        description="Minimum wage increased to 70,000 tenge",
        source="stat.gov.kz",
        source_url="https://stat.gov.kz",
        impact_description="Affects formal workers earning near minimum wage",
        affected_groups=["formal_workers", "low_income"],
        metadata={"amount_tenge": 70000, "previous_amount": 60000},
    ))

    calendar.add_event(PolicyEvent(
        name="Minimum Wage Increase 2024",
        policy_type=PolicyType.MINIMUM_WAGE,
        effective_date=date(2024, 1, 1),
        description="Minimum wage increased to 85,000 tenge (+21.4%)",
        source="stat.gov.kz",
        source_url="https://stat.gov.kz",
        impact_description="Large increase affects broader set of workers",
        affected_groups=["formal_workers", "low_income"],
        metadata={"amount_tenge": 85000, "previous_amount": 70000, "pct_change": 21.4},
    ))

    calendar.add_event(PolicyEvent(
        name="Minimum Wage 2025 (Unchanged)",
        policy_type=PolicyType.MINIMUM_WAGE,
        effective_date=date(2025, 1, 1),
        description="Minimum wage remains at 85,000 tenge",
        source="stat.gov.kz",
        source_url="https://stat.gov.kz",
        impact_description="No change from 2024",
        affected_groups=["formal_workers", "low_income"],
        metadata={"amount_tenge": 85000, "previous_amount": 85000, "pct_change": 0},
    ))

    # =========================================================================
    # CREDIT POLICY EVENTS (CRITICAL FOR CREDIT DEFAULT STUDY)
    # =========================================================================

    calendar.add_event(PolicyEvent(
        name="Personal Bankruptcy Law",
        policy_type=PolicyType.CREDIT_POLICY,
        effective_date=date(2023, 3, 1),
        description="Personal bankruptcy law enacted, changes collections dynamics",
        source="FSAP (Financial Sector Assessment Program)",
        impact_description="May affect default/recovery patterns",
        affected_groups=["borrowers", "over_indebted"],
        metadata={"law_type": "bankruptcy"},
    ))

    calendar.add_event(PolicyEvent(
        name="DSTI Limit Tightening",
        policy_type=PolicyType.CREDIT_POLICY,
        effective_date=date(2023, 12, 1),
        description="Debt-service-to-income limit tightened",
        source="IMF Article IV",
        impact_description="Reduces refinancing options for low-income borrowers",
        affected_groups=["borrowers", "low_income"],
        metadata={"policy_type": "macroprudential", "critical_confound": True},
    ))

    calendar.add_event(PolicyEvent(
        name="DTI Limit Introduction",
        policy_type=PolicyType.CREDIT_POLICY,
        effective_date=date(2024, 6, 1),
        description="Debt-to-income limit introduced (major credit constraint)",
        source="IMF Article IV",
        impact_description="Major new constraint on consumer lending",
        affected_groups=["borrowers", "all"],
        metadata={"policy_type": "macroprudential", "critical_confound": True},
    ))

    # =========================================================================
    # PENSION ELIGIBILITY
    # =========================================================================

    calendar.add_event(PolicyEvent(
        name="Pension Eligibility Rules",
        policy_type=PolicyType.PENSION,
        effective_date=date(2023, 1, 1),
        description="Pension eligibility: Men 63 years, Women 61 years",
        source="gov.kz",
        source_url="https://gov.kz",
        impact_description="Women's age stays at 61 from 2023-2028",
        affected_groups=["near_retirement"],
        metadata={
            "men_age": 63,
            "women_age": 61,
            "women_age_stable_until": 2028,
        },
    ))

    # =========================================================================
    # REGIONAL REFORMS
    # =========================================================================

    calendar.add_event(PolicyEvent(
        name="Turkestan/Shymkent Reform",
        policy_type=PolicyType.REGIONAL_REFORM,
        effective_date=date(2018, 6, 19),
        description="Turkestan Region created from South Kazakhstan; Shymkent becomes city of republican significance",
        source="Official gazette",
        impact_description="May affect regional statistics continuity",
        affected_groups=["south_kazakhstan", "turkestan", "shymkent"],
        metadata={"reform_type": "split", "regions_affected": ["South Kazakhstan"]},
    ))

    calendar.add_event(PolicyEvent(
        name="2022 Regional Reform",
        policy_type=PolicyType.REGIONAL_REFORM,
        effective_date=date(2022, 6, 8),
        description="Three new regions created: Abay, Zhetysu, Ulytau",
        source="Official gazette",
        impact_description="Affects regional data continuity, requires harmonization",
        affected_groups=["east_kazakhstan", "almaty_region", "karaganda"],
        metadata={
            "reform_type": "split",
            "new_regions": ["Abay", "Zhetysu", "Ulytau"],
            "parent_regions": ["East Kazakhstan", "Almaty Region", "Karaganda"],
        },
    ))

    # =========================================================================
    # TSA (TARGETED SOCIAL ASSISTANCE)
    # =========================================================================

    calendar.add_event(PolicyEvent(
        name="TSA Eligibility Rules",
        policy_type=PolicyType.SOCIAL_ASSISTANCE,
        effective_date=date(2023, 1, 1),
        description="TSA threshold: 35% of median income, min 70% of regional subsistence minimum",
        source="gov.kz",
        impact_description="Defines eligibility for targeted social assistance",
        affected_groups=["low_income", "vulnerable"],
        metadata={
            "threshold_pct_median": 35,
            "min_pct_subsistence": 70,
            "formula": "max(0.35 * median_income, 0.70 * regional_subsistence_min)",
        },
    ))

    # =========================================================================
    # FX DEPRECIATION EVENTS (FOR BLOCK F BACKTEST)
    # =========================================================================

    # Clean events (minimal confounds - use for primary backtest)
    calendar.add_event(PolicyEvent(
        name="2014 Devaluation",
        policy_type=PolicyType.FX_DEPRECIATION,
        effective_date=date(2014, 2, 11),
        description="First major devaluation of tenge since 2009; ~19% depreciation",
        source="NBK",
        source_url="https://nationalbank.kz",
        impact_description="Immediate imported inflation; reduced purchasing power",
        affected_groups=["all_households", "importers"],
        metadata={
            "magnitude": 0.19,
            "clean_event": True,
            "quarter": "2014Q1",
            "previous_rate": 155,
            "new_rate": 185,
        },
    ))

    calendar.add_event(PolicyEvent(
        name="Float to Flexible Regime",
        policy_type=PolicyType.FX_DEPRECIATION,
        effective_date=date(2015, 8, 20),
        description="NBK moves to flexible exchange rate regime; ~30% depreciation",
        source="NBK",
        source_url="https://nationalbank.kz",
        impact_description="Regime change to inflation targeting; major price adjustment",
        affected_groups=["all_households", "importers", "exporters"],
        metadata={
            "magnitude": 0.30,
            "clean_event": True,
            "quarter": "2015Q3",
            "previous_rate": 188,
            "new_rate": 255,
            "regime_change": True,
        },
    ))

    # Compound events (multiple confounds - use with caution, report separately)
    calendar.add_event(PolicyEvent(
        name="COVID + Oil Price Collapse",
        policy_type=PolicyType.FX_DEPRECIATION,
        effective_date=date(2020, 3, 9),
        description="FX depreciation during COVID pandemic and oil price war",
        source="NBK",
        source_url="https://nationalbank.kz",
        impact_description="Multiple shocks: COVID lockdowns, fiscal transfers, oil collapse",
        affected_groups=["all_households"],
        metadata={
            "magnitude": 0.15,
            "clean_event": False,
            "quarter": "2020Q1",
            "confounds": ["covid_lockdowns", "fiscal_transfers", "oil_price_collapse"],
            "note": "Compound event - interpret with caution",
        },
    ))

    calendar.add_event(PolicyEvent(
        name="Russia-Ukraine War Spillover",
        policy_type=PolicyType.FX_DEPRECIATION,
        effective_date=date(2022, 2, 28),
        description="FX depreciation during Russia-Ukraine war",
        source="NBK",
        source_url="https://nationalbank.kz",
        impact_description="Trade disruption, sanctions spillover, refugee flows",
        affected_groups=["all_households", "trade_sector"],
        metadata={
            "magnitude": 0.20,
            "clean_event": False,
            "quarter": "2022Q1",
            "confounds": ["trade_disruption", "sanctions_spillover", "refugee_flows"],
            "note": "Compound event - interpret with caution",
        },
    ))

    return calendar


# Convenience function for checking confounds
def check_credit_study_confounds(
    outcome_start: date,
    outcome_end: date,
) -> list[PolicyEvent]:
    """
    Check for policy confounds relevant to credit default study.

    Args:
        outcome_start: Start of outcome observation window
        outcome_end: End of outcome observation window

    Returns:
        List of potential confounding events

    Example:
        >>> confounds = check_credit_study_confounds(
        ...     date(2024, 1, 1),
        ...     date(2024, 5, 31)
        ... )
        >>> if confounds:
        ...     print("WARNING: Potential confounds detected")
    """
    calendar = get_kazakhstan_policy_calendar()
    return calendar.check_confounds(
        outcome_start,
        outcome_end,
        exclude_types=[PolicyType.CREDIT_POLICY],
    )


def get_minimum_wage_at_date(target_date: date) -> int:
    """
    Get the minimum wage effective at a given date.

    Args:
        target_date: Date to check

    Returns:
        Minimum wage in tenge
    """
    calendar = get_kazakhstan_policy_calendar()
    mw_events = [
        e for e in calendar.events
        if e.policy_type == PolicyType.MINIMUM_WAGE
        and e.effective_date <= target_date
    ]

    if not mw_events:
        raise ValueError(f"No minimum wage data available for {target_date}")

    # Get most recent
    latest = max(mw_events, key=lambda e: e.effective_date)
    return latest.metadata["amount_tenge"]


def get_pension_age(gender: str, year: int) -> int:
    """
    Get pension eligibility age.

    Args:
        gender: "M" or "F"
        year: Year to check

    Returns:
        Pension eligibility age
    """
    if gender.upper() == "M":
        return 63

    # Women's age is 61 and stable through 2028
    if year <= 2028:
        return 61

    # After 2028, may increase (not yet specified)
    return 61


def get_fx_depreciation_events(clean_only: bool = False) -> list[PolicyEvent]:
    """
    Get FX depreciation events for Block F backtest.

    Args:
        clean_only: If True, return only clean events (no confounds)

    Returns:
        List of FX depreciation events

    Example:
        >>> events = get_fx_depreciation_events(clean_only=True)
        >>> for e in events:
        ...     print(f"{e.effective_date}: {e.name} ({e.metadata['magnitude']:.0%})")
        2014-02-11: 2014 Devaluation (19%)
        2015-08-20: Float to Flexible Regime (30%)
    """
    calendar = get_kazakhstan_policy_calendar()
    events = [
        e for e in calendar.events
        if e.policy_type == PolicyType.FX_DEPRECIATION
    ]

    if clean_only:
        events = [
            e for e in events
            if e.metadata.get("clean_event", False)
        ]

    return events


def get_fx_event_dates(
    clean_only: bool = False,
    as_quarters: bool = False,
) -> list[date | str]:
    """
    Get FX depreciation event dates.

    Args:
        clean_only: If True, return only clean events
        as_quarters: If True, return quarter strings (e.g., "2014Q1")

    Returns:
        List of dates or quarter strings

    Example:
        >>> get_fx_event_dates(clean_only=True, as_quarters=True)
        ['2014Q1', '2015Q3']
    """
    events = get_fx_depreciation_events(clean_only=clean_only)

    if as_quarters:
        return [e.metadata.get("quarter") for e in events]
    return [e.effective_date for e in events]


def check_fx_confounds(
    analysis_start: date,
    analysis_end: date,
) -> list[PolicyEvent]:
    """
    Check for FX depreciation events in analysis window.

    Useful for identifying periods that may be affected by
    FX shocks when running other analyses.

    Args:
        analysis_start: Analysis start date
        analysis_end: Analysis end date

    Returns:
        List of FX events in the window
    """
    calendar = get_kazakhstan_policy_calendar()
    return calendar.get_events_in_window(
        analysis_start,
        analysis_end,
        policy_types=[PolicyType.FX_DEPRECIATION],
    )
