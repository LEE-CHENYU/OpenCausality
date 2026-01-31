"""
Portfolio stress testing for credit default study.

Applies estimated income-default elasticities to simulate portfolio-level
default rate changes under macroeconomic stress scenarios.

CRITICAL: External validity caveats apply - see documentation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from studies.credit_default.src.elasticity_store import ElasticityStore, Elasticity
from studies.credit_default.src.scenario_simulator import (
    CreditScenarioSimulator,
    CreditScenarioResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSegment:
    """A segment of the loan portfolio."""

    name: str
    exposure: float  # Total exposure (outstanding balance)
    num_loans: int
    baseline_default_rate: float
    weight: float  # Share of total portfolio

    # Segment characteristics (for external validity assessment)
    income_source: str  # "formal_payroll", "pension", "mixed", "unknown"
    avg_income: float | None = None
    avg_age: float | None = None

    def __post_init__(self):
        if self.weight < 0 or self.weight > 1:
            raise ValueError(f"Weight must be in [0, 1], got {self.weight}")


@dataclass
class StressScenario:
    """A macroeconomic stress scenario."""

    name: str
    description: str

    # Income shocks by segment
    income_shocks: dict[str, float]  # segment_name -> income_change_pct

    # Scenario metadata
    severity: str  # "mild", "moderate", "severe"
    probability: float | None = None  # Scenario probability (if applicable)


@dataclass
class PortfolioStressResult:
    """Results from portfolio stress test."""

    scenario: StressScenario

    # Portfolio-level impacts
    baseline_default_rate: float
    stressed_default_rate: float
    default_rate_change: float  # Absolute change
    default_rate_pct_change: float  # Relative change

    # Loss projections
    baseline_expected_loss: float
    stressed_expected_loss: float
    incremental_loss: float

    # By segment
    segment_results: dict[str, dict[str, float]]

    # Confidence interval
    conf_int_loss: tuple[float, float]

    # External validity
    external_validity_score: float  # 0-1, how applicable are estimates
    external_validity_caveats: list[str]
    segments_with_estimates: list[str]
    segments_without_estimates: list[str]

    def summary(self) -> str:
        """Generate summary."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"PORTFOLIO STRESS TEST: {self.scenario.name}")
        lines.append("=" * 70)

        lines.append(f"\nScenario: {self.scenario.description}")
        lines.append(f"Severity: {self.scenario.severity}")

        lines.append(f"\nPortfolio Default Rate Impact:")
        lines.append(f"  Baseline rate: {self.baseline_default_rate*100:.2f}%")
        lines.append(f"  Stressed rate: {self.stressed_default_rate*100:.2f}%")
        lines.append(f"  Change: {self.default_rate_change*100:+.3f} pp")
        lines.append(f"  Relative change: {self.default_rate_pct_change*100:+.1f}%")

        lines.append(f"\nExpected Loss Impact:")
        lines.append(f"  Baseline EL: {self.baseline_expected_loss:,.0f}")
        lines.append(f"  Stressed EL: {self.stressed_expected_loss:,.0f}")
        lines.append(f"  Incremental loss: {self.incremental_loss:,.0f}")
        lines.append(
            f"  95% CI: [{self.conf_int_loss[0]:,.0f}, {self.conf_int_loss[1]:,.0f}]"
        )

        lines.append(f"\nBy Segment:")
        for segment, results in self.segment_results.items():
            lines.append(f"  {segment}:")
            lines.append(f"    Income shock: {results['income_shock']*100:+.1f}%")
            lines.append(f"    Default change: {results['default_change']*100:+.3f} pp")
            lines.append(f"    Loss impact: {results['loss_impact']:,.0f}")

        lines.append(f"\nExternal Validity Assessment:")
        lines.append(f"  Applicability score: {self.external_validity_score:.0%}")
        lines.append(
            f"  Segments with estimates: {', '.join(self.segments_with_estimates) or 'None'}"
        )
        lines.append(
            f"  Segments without estimates: {', '.join(self.segments_without_estimates) or 'None'}"
        )

        lines.append(f"\nCaveats:")
        for caveat in self.external_validity_caveats:
            lines.append(f"  - {caveat}")

        return "\n".join(lines)


class PortfolioStressTester:
    """
    Portfolio-level stress testing using estimated elasticities.

    CRITICAL CAVEATS:
    1. Estimated elasticities are LATEs - apply primarily to similar borrowers
    2. Linear extrapolation may not hold for large shocks
    3. General equilibrium effects not captured
    4. Portfolio composition may differ from study sample
    """

    # Standard stress scenarios
    STANDARD_SCENARIOS = {
        "mild_recession": StressScenario(
            name="Mild Recession",
            description="Mild economic downturn with moderate income decline",
            income_shocks={
                "formal_workers": -0.05,  # 5% income decline
                "near_retirees": -0.02,  # 2% decline (pensions more stable)
                "default": -0.05,  # Default for unknown segments
            },
            severity="mild",
            probability=0.20,
        ),
        "moderate_recession": StressScenario(
            name="Moderate Recession",
            description="Moderate recession with significant income decline",
            income_shocks={
                "formal_workers": -0.10,  # 10% income decline
                "near_retirees": -0.03,  # 3% decline
                "default": -0.10,
            },
            severity="moderate",
            probability=0.10,
        ),
        "severe_recession": StressScenario(
            name="Severe Recession",
            description="Severe economic crisis with major income decline",
            income_shocks={
                "formal_workers": -0.20,  # 20% income decline
                "near_retirees": -0.05,  # 5% decline
                "default": -0.20,
            },
            severity="severe",
            probability=0.02,
        ),
        "oil_shock": StressScenario(
            name="Oil Price Collapse",
            description="Major oil price decline affecting Kazakhstan economy",
            income_shocks={
                "formal_workers": -0.15,  # 15% decline
                "near_retirees": -0.05,  # 5% decline
                "default": -0.15,
            },
            severity="severe",
            probability=0.05,
        ),
    }

    def __init__(
        self,
        elasticity_store: ElasticityStore | None = None,
        lgd: float = 0.50,  # Loss given default
    ):
        """
        Initialize stress tester.

        Args:
            elasticity_store: Store of estimated elasticities
            lgd: Loss given default rate
        """
        self.store = elasticity_store or ElasticityStore()
        self.lgd = lgd
        self.simulator = CreditScenarioSimulator(elasticity_store=self.store)

    def stress_test(
        self,
        portfolio: list[PortfolioSegment],
        scenario: StressScenario | str,
    ) -> PortfolioStressResult:
        """
        Run portfolio stress test.

        Args:
            portfolio: List of portfolio segments
            scenario: Stress scenario or name of standard scenario

        Returns:
            PortfolioStressResult with impacts
        """
        # Get scenario
        if isinstance(scenario, str):
            if scenario not in self.STANDARD_SCENARIOS:
                raise ValueError(
                    f"Unknown scenario: {scenario}. "
                    f"Available: {list(self.STANDARD_SCENARIOS.keys())}"
                )
            scenario = self.STANDARD_SCENARIOS[scenario]

        # Get available elasticities
        elasticities = self._get_elasticities()

        # Compute portfolio totals
        total_exposure = sum(s.exposure for s in portfolio)
        portfolio_baseline_default = sum(
            s.baseline_default_rate * s.weight for s in portfolio
        )

        # Track results by segment
        segment_results = {}
        segments_with_estimates = []
        segments_without_estimates = []

        total_stressed_default = 0.0
        total_se_squared = 0.0
        total_baseline_loss = 0.0
        total_stressed_loss = 0.0

        for segment in portfolio:
            # Get income shock for this segment
            income_shock = scenario.income_shocks.get(
                segment.income_source,
                scenario.income_shocks.get("default", 0),
            )

            # Get elasticity for this segment
            elasticity = elasticities.get(segment.income_source)

            if elasticity is not None:
                # Apply elasticity
                default_change = elasticity * income_shock
                segments_with_estimates.append(segment.name)

                # Get SE for confidence interval
                elast_obj = self.store.get_elasticity(
                    "credit_default", segment.income_source
                )
                if elast_obj:
                    se_change = elast_obj.std_error * abs(income_shock)
                else:
                    se_change = abs(default_change) * 0.3  # Assume 30% uncertainty
            else:
                # No estimate - use conservative assumption or skip
                # Use default elasticity from formal_workers as proxy
                proxy_elasticity = elasticities.get("formal_workers")
                if proxy_elasticity is not None:
                    default_change = proxy_elasticity * income_shock
                    se_change = abs(default_change) * 0.5  # Higher uncertainty
                else:
                    default_change = 0
                    se_change = 0

                segments_without_estimates.append(segment.name)

            # Compute stressed default rate
            stressed_default = segment.baseline_default_rate + default_change
            stressed_default = max(0, min(1, stressed_default))  # Bound to [0,1]

            # Compute losses
            baseline_loss = segment.exposure * segment.baseline_default_rate * self.lgd
            stressed_loss = segment.exposure * stressed_default * self.lgd

            segment_results[segment.name] = {
                "income_shock": income_shock,
                "elasticity_used": elasticity,
                "baseline_default": segment.baseline_default_rate,
                "stressed_default": stressed_default,
                "default_change": default_change,
                "baseline_loss": baseline_loss,
                "stressed_loss": stressed_loss,
                "loss_impact": stressed_loss - baseline_loss,
            }

            # Aggregate
            total_stressed_default += stressed_default * segment.weight
            total_se_squared += (se_change * segment.weight) ** 2
            total_baseline_loss += baseline_loss
            total_stressed_loss += stressed_loss

        # Portfolio-level metrics
        default_rate_change = total_stressed_default - portfolio_baseline_default
        default_rate_pct_change = (
            default_rate_change / portfolio_baseline_default
            if portfolio_baseline_default > 0
            else 0
        )

        incremental_loss = total_stressed_loss - total_baseline_loss
        total_se = np.sqrt(total_se_squared)

        # Loss confidence interval
        loss_ci = (
            incremental_loss - 1.96 * total_se * total_exposure * self.lgd,
            incremental_loss + 1.96 * total_se * total_exposure * self.lgd,
        )

        # External validity assessment
        external_validity_score = self._compute_external_validity_score(
            segments_with_estimates, segments_without_estimates, portfolio
        )

        caveats = self._get_external_validity_caveats(
            segments_with_estimates, segments_without_estimates, scenario
        )

        return PortfolioStressResult(
            scenario=scenario,
            baseline_default_rate=portfolio_baseline_default,
            stressed_default_rate=total_stressed_default,
            default_rate_change=default_rate_change,
            default_rate_pct_change=default_rate_pct_change,
            baseline_expected_loss=total_baseline_loss,
            stressed_expected_loss=total_stressed_loss,
            incremental_loss=incremental_loss,
            segment_results=segment_results,
            conf_int_loss=loss_ci,
            external_validity_score=external_validity_score,
            external_validity_caveats=caveats,
            segments_with_estimates=segments_with_estimates,
            segments_without_estimates=segments_without_estimates,
        )

    def run_all_scenarios(
        self,
        portfolio: list[PortfolioSegment],
    ) -> dict[str, PortfolioStressResult]:
        """
        Run all standard stress scenarios.

        Args:
            portfolio: List of portfolio segments

        Returns:
            Dictionary of scenario name to results
        """
        results = {}
        for name, scenario in self.STANDARD_SCENARIOS.items():
            try:
                results[name] = self.stress_test(portfolio, scenario)
            except Exception as e:
                logger.warning(f"Failed to run scenario {name}: {e}")

        return results

    def _get_elasticities(self) -> dict[str, float]:
        """Get elasticities for simulation."""
        eset = self.store.get("credit_default")
        if eset is None:
            return {}
        return {e.segment: e.coefficient for e in eset.elasticities}

    def _compute_external_validity_score(
        self,
        segments_with_estimates: list[str],
        segments_without_estimates: list[str],
        portfolio: list[PortfolioSegment],
    ) -> float:
        """
        Compute external validity score (0-1).

        Higher score means estimates are more applicable to portfolio.
        """
        if not segments_with_estimates:
            return 0.0

        # Weight by exposure
        total_exposure = sum(s.exposure for s in portfolio)
        covered_exposure = sum(
            s.exposure
            for s in portfolio
            if s.name in segments_with_estimates
        )

        exposure_coverage = covered_exposure / total_exposure if total_exposure > 0 else 0

        # Penalize for segments without estimates
        segment_coverage = len(segments_with_estimates) / (
            len(segments_with_estimates) + len(segments_without_estimates)
        )

        # Combined score
        return 0.6 * exposure_coverage + 0.4 * segment_coverage

    def _get_external_validity_caveats(
        self,
        segments_with_estimates: list[str],
        segments_without_estimates: list[str],
        scenario: StressScenario,
    ) -> list[str]:
        """Get external validity caveats for stress test."""
        caveats = []

        # Base caveats
        caveats.append(
            "Estimated elasticities are local average treatment effects (LATE)"
        )
        caveats.append(
            "Linear extrapolation may not hold for large income shocks"
        )

        # Scenario-specific
        if scenario.severity == "severe":
            caveats.append(
                "Severe scenarios may involve nonlinear effects not captured"
            )
            caveats.append(
                "General equilibrium effects (unemployment, asset prices) not modeled"
            )

        # Coverage
        if segments_without_estimates:
            caveats.append(
                f"No causal estimates for: {', '.join(segments_without_estimates)}. "
                "Using proxy elasticities with higher uncertainty."
            )

        # Segment-specific
        eset = self.store.get("credit_default")
        if eset:
            for e in eset.elasticities:
                caveats.extend(e.external_validity_caveats)

        return list(set(caveats))  # Deduplicate

    def summary_all_scenarios(
        self,
        results: dict[str, PortfolioStressResult],
    ) -> str:
        """Generate summary of all scenario results."""
        lines = []
        lines.append("=" * 70)
        lines.append("PORTFOLIO STRESS TEST SUMMARY")
        lines.append("=" * 70)

        for name, result in results.items():
            lines.append(f"\n{result.scenario.name} ({result.scenario.severity}):")
            lines.append(
                f"  Default rate: {result.baseline_default_rate*100:.2f}% -> "
                f"{result.stressed_default_rate*100:.2f}% "
                f"({result.default_rate_change*100:+.3f} pp)"
            )
            lines.append(
                f"  Expected loss: +{result.incremental_loss:,.0f} "
                f"(95% CI: [{result.conf_int_loss[0]:,.0f}, {result.conf_int_loss[1]:,.0f}])"
            )
            lines.append(
                f"  External validity: {result.external_validity_score:.0%}"
            )

        lines.append("\n" + "=" * 70)
        lines.append("IMPORTANT: These projections are subject to significant")
        lines.append("uncertainty. See individual scenario caveats for details.")

        return "\n".join(lines)


def create_sample_portfolio() -> list[PortfolioSegment]:
    """Create sample portfolio for demonstration."""
    return [
        PortfolioSegment(
            name="Formal Payroll Loans",
            exposure=1_000_000_000,  # 1B tenge
            num_loans=10_000,
            baseline_default_rate=0.05,
            weight=0.50,
            income_source="formal_workers",
            avg_income=150_000,
            avg_age=35,
        ),
        PortfolioSegment(
            name="Pension Borrowers",
            exposure=300_000_000,  # 300M tenge
            num_loans=3_000,
            baseline_default_rate=0.03,
            weight=0.15,
            income_source="near_retirees",
            avg_income=120_000,
            avg_age=62,
        ),
        PortfolioSegment(
            name="Mixed Income",
            exposure=500_000_000,  # 500M tenge
            num_loans=5_000,
            baseline_default_rate=0.06,
            weight=0.25,
            income_source="mixed",
            avg_income=130_000,
            avg_age=40,
        ),
        PortfolioSegment(
            name="Unknown Income Source",
            exposure=200_000_000,  # 200M tenge
            num_loans=2_000,
            baseline_default_rate=0.08,
            weight=0.10,
            income_source="unknown",
            avg_income=None,
            avg_age=None,
        ),
    ]


def run_stress_test(
    scenario: str = "moderate_recession",
    lgd: float = 0.50,
) -> PortfolioStressResult:
    """
    Convenience function to run stress test.

    Args:
        scenario: Scenario name
        lgd: Loss given default

    Returns:
        PortfolioStressResult
    """
    tester = PortfolioStressTester(lgd=lgd)
    portfolio = create_sample_portfolio()
    return tester.stress_test(portfolio, scenario)
