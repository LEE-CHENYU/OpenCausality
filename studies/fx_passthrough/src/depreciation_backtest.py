"""
Block F Validation: Event Study Backtest Around FX Depreciation Episodes

PURPOSE:
    Diagnostic check - does the sign/direction of spending response match
    the LP-IRF results? This is a BACKTEST, NOT standalone causal identification.

CLEAN EVENTS (use for primary backtest):
    - 2014-Q1: First major devaluation (~19%)
    - 2015-Q3: Float to flexible regime (~30%)

COMPOUND EVENTS (use with caution, report separately):
    - 2020-Q1: COVID + oil collapse (confounds: lockdowns, transfers)
    - 2022-Q1: Russia-Ukraine war (confounds: trade disruption, sanctions)

METHODOLOGY:
    Event study with [-4, +4] quarter window around each event.
    Normalize to t=-1 (pre-event quarter).
    Compare direction with LP-IRF results.

FRAMING:
    This is NOT causal identification. The event study validates that our
    LP-IRF estimates are consistent with observable behavior during
    known FX depreciation episodes.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass
class DepreciationEvent:
    """A single FX depreciation event for backtest."""

    name: str
    date: str  # Quarter string, e.g., "2015Q3"
    magnitude: float  # Approximate depreciation magnitude
    clean: bool  # True if minimal confounds
    confounds: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class BacktestResult:
    """Results from event study backtest."""

    event_name: str
    event_date: str
    clean_event: bool

    # Pre/post comparison
    pre_mean: float
    post_mean: float
    diff: float
    diff_se: float
    diff_pvalue: float

    # Dynamic effects by event-time
    event_times: list[int]  # [-4, -3, ..., 0, ..., 4]
    income_effects: np.ndarray
    income_se: np.ndarray
    expenditure_effects: np.ndarray
    expenditure_se: np.ndarray

    # Sample
    n_obs: int

    def summary(self) -> str:
        """Return formatted summary."""
        clean_str = "CLEAN" if self.clean_event else "COMPOUND (interpret with caution)"
        lines = [
            "=" * 70,
            f"Event Study: {self.event_name} ({self.event_date})",
            f"Event type: {clean_str}",
            "=" * 70,
            "",
            f"Pre-event mean (t=-4 to -1): {self.pre_mean:.4f}",
            f"Post-event mean (t=0 to 4):  {self.post_mean:.4f}",
            f"Difference:                  {self.diff:.4f} (SE: {self.diff_se:.4f})",
            f"p-value:                     {self.diff_pvalue:.4f}",
            "",
            "Dynamic Effects (relative to t=-1):",
            f"{'Event-time':>12} {'Income':>12} {'(SE)':>10} {'Expenditure':>12} {'(SE)':>10}",
            "-" * 60,
        ]

        for i, t in enumerate(self.event_times):
            lines.append(
                f"{t:>12} {self.income_effects[i]:>12.4f} ({self.income_se[i]:.4f}) "
                f"{self.expenditure_effects[i]:>12.4f} ({self.expenditure_se[i]:.4f})"
            )

        lines.extend([
            "",
            f"N observations: {self.n_obs}",
            "=" * 70,
        ])

        return "\n".join(lines)


@dataclass
class DepreciationBacktestSpec:
    """Specification for depreciation backtest."""

    # Outcome variables
    income_outcome: str = "real_income_growth"
    expenditure_outcome: str = "real_expenditure_growth"

    # Event window
    window_pre: int = 4  # Quarters before event
    window_post: int = 4  # Quarters after event

    # Which events to include
    use_clean_only: bool = False  # If True, only use clean events

    # Inference
    cluster_by: str | None = None  # Optional clustering


class DepreciationBacktest:
    """
    Event study backtest for Block F validation.

    Compares observed income/expenditure dynamics around FX depreciation
    events with LP-IRF estimates.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with quarterly panel data.

        Args:
            data: DataFrame with quarter, income, and expenditure series
        """
        self.data = data.copy()
        self._prepare_data()
        self.events = self._define_events()
        self.results: dict[str, BacktestResult] = {}

    def _prepare_data(self) -> None:
        """Prepare data for event study."""
        # Ensure quarter column exists
        if "quarter" not in self.data.columns and "date" in self.data.columns:
            self.data["quarter"] = (
                self.data["date"].dt.year.astype(str) + "Q" +
                self.data["date"].dt.quarter.astype(str)
            )

        # Sort by time
        if "quarter" in self.data.columns:
            self.data = self.data.sort_values("quarter").reset_index(drop=True)

        # Create numeric quarter for event-time calculation
        if "quarter_num" not in self.data.columns:
            self.data["quarter_num"] = range(len(self.data))

    def _define_events(self) -> list[DepreciationEvent]:
        """Define FX depreciation events for backtest."""
        events = [
            # Clean events (use for primary backtest)
            DepreciationEvent(
                name="2014 Devaluation",
                date="2014Q1",
                magnitude=0.19,
                clean=True,
                notes="First major devaluation since 2009",
            ),
            DepreciationEvent(
                name="Float to Flexible",
                date="2015Q3",
                magnitude=0.30,
                clean=True,
                notes="Regime change to flexible exchange rate",
            ),
            # Compound events (report separately)
            DepreciationEvent(
                name="COVID + Oil Collapse",
                date="2020Q1",
                magnitude=0.15,
                clean=False,
                confounds=["covid_lockdowns", "fiscal_transfers", "oil_price_collapse"],
                notes="Multiple simultaneous shocks",
            ),
            DepreciationEvent(
                name="Russia-Ukraine War",
                date="2022Q1",
                magnitude=0.20,
                clean=False,
                confounds=["trade_disruption", "sanctions_spillover", "refugee_flows"],
                notes="War spillover effects",
            ),
        ]

        return events

    def define_events(self, events: list[DepreciationEvent]) -> None:
        """
        Override default events with custom list.

        Args:
            events: List of DepreciationEvent objects
        """
        self.events = events

    def _get_event_quarter_num(self, event_quarter: str) -> int | None:
        """Get numeric quarter index for event quarter string."""
        if "quarter" not in self.data.columns:
            return None

        matches = self.data[self.data["quarter"] == event_quarter]
        if len(matches) == 0:
            return None

        return matches["quarter_num"].iloc[0]

    def construct_event_window(
        self,
        event: DepreciationEvent,
        spec: DepreciationBacktestSpec,
    ) -> pd.DataFrame | None:
        """
        Construct event-time window for a single event.

        Returns DataFrame with event_time column relative to event.
        """
        event_q = self._get_event_quarter_num(event.date)
        if event_q is None:
            logger.warning(f"Event {event.name} quarter {event.date} not in data")
            return None

        # Window boundaries
        start_q = event_q - spec.window_pre
        end_q = event_q + spec.window_post

        # Filter to window
        window_data = self.data[
            (self.data["quarter_num"] >= start_q) &
            (self.data["quarter_num"] <= end_q)
        ].copy()

        if len(window_data) == 0:
            return None

        # Create event-time variable
        window_data["event_time"] = window_data["quarter_num"] - event_q

        # Add event metadata
        window_data["event_name"] = event.name
        window_data["event_date"] = event.date
        window_data["event_clean"] = event.clean

        return window_data

    def estimate_dynamic_effects(
        self,
        event: DepreciationEvent,
        spec: DepreciationBacktestSpec,
    ) -> BacktestResult | None:
        """
        Estimate dynamic effects around a single event.

        Uses event-study regression:
            Y_t = Σ_k β_k × 1{event_time = k} + ε_t
        with k ∈ [-window_pre, ..., -2, 0, ..., window_post]
        (omit k=-1 as reference period)
        """
        window_data = self.construct_event_window(event, spec)
        if window_data is None:
            return None

        # Event times (excluding -1 as reference)
        event_times = list(range(-spec.window_pre, spec.window_post + 1))

        results = {"income": {}, "expenditure": {}}

        for outcome_type, outcome in [
            ("income", spec.income_outcome),
            ("expenditure", spec.expenditure_outcome)
        ]:
            if outcome not in window_data.columns:
                logger.warning(f"Outcome {outcome} not in data")
                continue

            # Create event-time dummies (excluding t=-1)
            for t in event_times:
                if t != -1:
                    window_data[f"d_t{t}"] = (window_data["event_time"] == t).astype(int)

            # Regression
            dummy_cols = [f"d_t{t}" for t in event_times if t != -1]
            valid_dummies = [c for c in dummy_cols if c in window_data.columns]

            if len(valid_dummies) == 0:
                continue

            y = window_data[outcome].dropna()
            X = sm.add_constant(window_data.loc[y.index, valid_dummies])

            model = sm.OLS(y, X).fit(cov_type="HC1")

            # Extract coefficients by event-time
            effects = []
            ses = []
            for t in event_times:
                if t == -1:
                    # Reference period
                    effects.append(0.0)
                    ses.append(0.0)
                else:
                    col = f"d_t{t}"
                    if col in model.params.index:
                        effects.append(model.params[col])
                        ses.append(model.bse[col])
                    else:
                        effects.append(np.nan)
                        ses.append(np.nan)

            results[outcome_type]["effects"] = np.array(effects)
            results[outcome_type]["se"] = np.array(ses)

        # Pre/post comparison for expenditure
        pre_data = window_data[window_data["event_time"] < 0][spec.expenditure_outcome]
        post_data = window_data[window_data["event_time"] >= 0][spec.expenditure_outcome]

        pre_mean = pre_data.mean() if len(pre_data) > 0 else np.nan
        post_mean = post_data.mean() if len(post_data) > 0 else np.nan
        diff = post_mean - pre_mean

        # SE for difference (assuming independence)
        pre_se = pre_data.std() / np.sqrt(len(pre_data)) if len(pre_data) > 1 else np.nan
        post_se = post_data.std() / np.sqrt(len(post_data)) if len(post_data) > 1 else np.nan
        diff_se = np.sqrt(pre_se**2 + post_se**2) if not np.isnan(pre_se) else np.nan

        # Two-sample t-test
        from scipy import stats
        if len(pre_data) > 1 and len(post_data) > 1:
            _, diff_pvalue = stats.ttest_ind(pre_data.dropna(), post_data.dropna())
        else:
            diff_pvalue = np.nan

        return BacktestResult(
            event_name=event.name,
            event_date=event.date,
            clean_event=event.clean,
            pre_mean=pre_mean,
            post_mean=post_mean,
            diff=diff,
            diff_se=diff_se,
            diff_pvalue=diff_pvalue,
            event_times=event_times,
            income_effects=results.get("income", {}).get("effects", np.array([])),
            income_se=results.get("income", {}).get("se", np.array([])),
            expenditure_effects=results.get("expenditure", {}).get("effects", np.array([])),
            expenditure_se=results.get("expenditure", {}).get("se", np.array([])),
            n_obs=len(window_data),
        )

    def run_backtest(
        self,
        spec: DepreciationBacktestSpec | None = None,
    ) -> dict[str, BacktestResult]:
        """
        Run backtest for all events.

        Args:
            spec: Backtest specification

        Returns:
            Dict of results keyed by event name
        """
        if spec is None:
            spec = DepreciationBacktestSpec()

        results = {}

        events_to_run = self.events
        if spec.use_clean_only:
            events_to_run = [e for e in self.events if e.clean]

        for event in events_to_run:
            try:
                result = self.estimate_dynamic_effects(event, spec)
                if result is not None:
                    results[event.name] = result
                    self.results[event.name] = result
                    logger.info(f"Backtest for {event.name}: completed")
            except Exception as e:
                logger.warning(f"Backtest for {event.name} failed: {e}")

        return results

    def run_clean_events_only(
        self,
        spec: DepreciationBacktestSpec | None = None,
    ) -> dict[str, BacktestResult]:
        """Run backtest for clean events only."""
        if spec is None:
            spec = DepreciationBacktestSpec()
        spec.use_clean_only = True
        return self.run_backtest(spec)

    def compare_to_lp(
        self,
        lp_results: Any,  # SpendingResponseResult
    ) -> dict[str, Any]:
        """
        Compare event study results to LP-IRF estimates.

        Checks:
        1. Sign consistency: Do event study and LP show same direction?
        2. Magnitude plausibility: Are effects in similar range?

        Args:
            lp_results: SpendingResponseResult from Block F LP estimation

        Returns:
            Comparison dict with consistency checks
        """
        comparison = {
            "events": {},
            "overall_consistent": None,
        }

        # LP-IRF sign at horizon 0
        lp_income_sign = np.sign(lp_results.income_irf[0])
        lp_exp_sign = np.sign(lp_results.expenditure_irf[0])

        consistent_count = 0
        total_count = 0

        for event_name, backtest_result in self.results.items():
            if not backtest_result.clean_event:
                # Skip compound events for consistency check
                continue

            # Event study sign (post-pre difference)
            es_sign = np.sign(backtest_result.diff)

            # Sign consistency for expenditure
            sign_match = (lp_exp_sign == es_sign) or (es_sign == 0)

            comparison["events"][event_name] = {
                "lp_expenditure_sign": lp_exp_sign,
                "event_study_sign": es_sign,
                "sign_consistent": sign_match,
                "lp_effect": lp_results.expenditure_irf[0],
                "es_effect": backtest_result.diff,
            }

            if sign_match:
                consistent_count += 1
            total_count += 1

        comparison["overall_consistent"] = (
            consistent_count == total_count if total_count > 0 else None
        )
        comparison["consistent_fraction"] = (
            consistent_count / total_count if total_count > 0 else None
        )

        return comparison

    def plot_backtest(
        self,
        event_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate plot data for backtest visualization.

        Returns dict with data for plotting event study figure.
        """
        if event_name is None:
            # Use first available clean event
            clean_events = [e for e in self.events if e.clean]
            if not clean_events:
                return {"error": "No clean events available"}
            event_name = clean_events[0].name

        if event_name not in self.results:
            return {"error": f"No results for event {event_name}"}

        result = self.results[event_name]

        return {
            "event_name": result.event_name,
            "event_date": result.event_date,
            "event_times": result.event_times,
            "income": {
                "effects": result.income_effects.tolist(),
                "se": result.income_se.tolist(),
                "ci_lower": (result.income_effects - 1.96 * result.income_se).tolist(),
                "ci_upper": (result.income_effects + 1.96 * result.income_se).tolist(),
            },
            "expenditure": {
                "effects": result.expenditure_effects.tolist(),
                "se": result.expenditure_se.tolist(),
                "ci_lower": (result.expenditure_effects - 1.96 * result.expenditure_se).tolist(),
                "ci_upper": (result.expenditure_effects + 1.96 * result.expenditure_se).tolist(),
            },
        }

    def summary(self) -> str:
        """Return summary of all backtest results."""
        if not self.results:
            return "No backtest results available. Run run_backtest() first."

        lines = [
            "=" * 75,
            "DEPRECIATION BACKTEST SUMMARY",
            "=" * 75,
            "",
            "PURPOSE: Validate LP-IRF direction (NOT causal identification)",
            "",
            f"{'Event':20} {'Date':8} {'Type':10} {'Exp. Diff':>12} {'p-value':>10}",
            "-" * 75,
        ]

        for event_name, result in self.results.items():
            event_type = "Clean" if result.clean_event else "Compound"
            lines.append(
                f"{event_name:20} {result.event_date:8} {event_type:10} "
                f"{result.diff:>12.4f} {result.diff_pvalue:>10.4f}"
            )

        lines.extend([
            "",
            "NOTE: Compound events have multiple confounds and should be",
            "      interpreted with caution. Report separately from clean events.",
            "=" * 75,
        ])

        return "\n".join(lines)
