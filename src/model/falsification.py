"""
Falsification tests for shift-share design.

Implements:
1. Pre-trends tests (leads of shock interactions)
2. Placebo exposures (non-oil regions)
3. Region reform placebos (no jumps at boundary dates)
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

logger = logging.getLogger(__name__)


@dataclass
class PreTrendsResult:
    """Results from pre-trends test."""

    leads: list[int]
    coefficients: np.ndarray
    std_errors: np.ndarray
    pvalues: np.ndarray
    joint_fstat: float
    joint_pvalue: float
    passed: bool


@dataclass
class PlaceboResult:
    """Results from placebo test."""

    placebo_name: str
    coefficient: float
    std_error: float
    pvalue: float
    passed: bool
    notes: str = ""


@dataclass
class ReformPlaceboResult:
    """Results from region reform placebo test."""

    reform_date: str
    pre_mean: float
    post_mean: float
    diff: float
    std_error: float
    pvalue: float
    passed: bool


class FalsificationTests:
    """Suite of falsification tests for shift-share design."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with panel data.

        Args:
            data: Panel DataFrame with region-quarter observations
        """
        self.data = data.copy()
        self.results: dict[str, Any] = {}

    def run_all(
        self,
        outcome: str = "log_income_pc",
        exposure: str = "E_oil_r",
        shock: str = "oil_supply_shock",
    ) -> dict[str, Any]:
        """Run all falsification tests."""
        results = {}

        # Pre-trends
        try:
            results["pre_trends"] = self.test_pre_trends(outcome, exposure, shock)
        except Exception as e:
            logger.error(f"Pre-trends test failed: {e}")

        # Placebo exposures
        try:
            results["placebo_exposure"] = self.test_placebo_exposure(outcome, shock)
        except Exception as e:
            logger.error(f"Placebo exposure test failed: {e}")

        # Region reform placebos
        try:
            results["reform_2018"] = self.test_region_reform(outcome, "2018Q1")
            results["reform_2022"] = self.test_region_reform(outcome, "2022Q1")
        except Exception as e:
            logger.error(f"Reform placebo test failed: {e}")

        self.results = results
        return results

    def test_pre_trends(
        self,
        outcome: str,
        exposure: str,
        shock: str,
        leads: list[int] | None = None,
    ) -> PreTrendsResult:
        """
        Test for pre-trends using leads of shock interactions.

        If identification is valid, leads should be insignificant.

        Args:
            outcome: Outcome variable
            exposure: Exposure variable
            shock: Shock variable
            leads: Lead periods to test (default: [1, 2, 3, 4])

        Returns:
            PreTrendsResult with coefficients and joint test
        """
        if leads is None:
            leads = [1, 2, 3, 4]

        data = self.data.copy()

        # Reset index if needed
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()

        # Sort by region and time
        data = data.sort_values(["region", "quarter"])

        # Create lead variables
        for lead in leads:
            lead_name = f"{shock}_lead{lead}"
            data[lead_name] = data.groupby("region")[shock].shift(-lead)

            # Create interaction
            interaction_name = f"{exposure}_x_lead{lead}"
            data[interaction_name] = data[exposure] * data[lead_name]

        # Prepare regression
        interaction_cols = [f"{exposure}_x_lead{lead}" for lead in leads]
        main_interaction = f"{exposure}_x_{shock.replace('_shock', '')}"

        # Create main interaction if not exists
        if main_interaction not in data.columns:
            data[main_interaction] = data[exposure] * data[shock]

        # Drop missing
        all_vars = [outcome, main_interaction] + interaction_cols
        data = data.dropna(subset=all_vars)

        # Set up panel
        data = data.set_index(["region", "quarter"])

        # Fit model with leads
        X = data[[main_interaction] + interaction_cols]
        y = data[outcome]

        model = PanelOLS(y, X, entity_effects=True, time_effects=True)
        result = model.fit(cov_type="kernel")

        # Extract lead coefficients
        coefficients = np.array([result.params[f"{exposure}_x_lead{lead}"] for lead in leads])
        std_errors = np.array([result.std_errors[f"{exposure}_x_lead{lead}"] for lead in leads])
        pvalues = np.array([result.pvalues[f"{exposure}_x_lead{lead}"] for lead in leads])

        # Joint F-test on leads
        # H0: all lead coefficients = 0
        from scipy import stats

        # Wald test
        lead_params = np.array([result.params[f"{exposure}_x_lead{lead}"] for lead in leads])
        lead_vcov = result.cov.loc[
            [f"{exposure}_x_lead{lead}" for lead in leads],
            [f"{exposure}_x_lead{lead}" for lead in leads],
        ].values

        try:
            wald_stat = lead_params @ np.linalg.solve(lead_vcov, lead_params)
            joint_fstat = wald_stat / len(leads)
            joint_pvalue = 1 - stats.f.cdf(joint_fstat, len(leads), result.df_resid)
        except Exception:
            joint_fstat = np.nan
            joint_pvalue = np.nan

        # Test passes if joint p-value > 0.05
        passed = joint_pvalue > 0.05 if not np.isnan(joint_pvalue) else False

        return PreTrendsResult(
            leads=leads,
            coefficients=coefficients,
            std_errors=std_errors,
            pvalues=pvalues,
            joint_fstat=joint_fstat,
            joint_pvalue=joint_pvalue,
            passed=passed,
        )

    def test_placebo_exposure(
        self,
        outcome: str,
        shock: str,
        placebo_exposure: str | None = None,
    ) -> PlaceboResult:
        """
        Test placebo exposure: non-oil regions should not respond to oil shocks.

        Args:
            outcome: Outcome variable
            shock: Shock variable
            placebo_exposure: Placebo exposure variable (default: inverse of oil exposure)

        Returns:
            PlaceboResult
        """
        data = self.data.copy()

        # Reset index if needed
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()

        # Create placebo exposure if not provided
        if placebo_exposure is None:
            # Use inverse of oil exposure (non-oil share)
            if "E_oil_r" in data.columns:
                data["E_nonoil_r"] = 1 - data["E_oil_r"]
                placebo_exposure = "E_nonoil_r"
            else:
                raise ValueError("No oil exposure variable found")

        # Create placebo interaction
        placebo_interaction = f"{placebo_exposure}_x_{shock.replace('_shock', '')}"
        data[placebo_interaction] = data[placebo_exposure] * data[shock]

        # Drop missing
        data = data.dropna(subset=[outcome, placebo_interaction])

        # Set up panel
        data = data.set_index(["region", "quarter"])

        # Fit model
        X = data[[placebo_interaction]]
        y = data[outcome]

        model = PanelOLS(y, X, entity_effects=True, time_effects=True)
        result = model.fit(cov_type="kernel")

        coef = result.params[placebo_interaction]
        se = result.std_errors[placebo_interaction]
        pval = result.pvalues[placebo_interaction]

        # Test passes if placebo coefficient is insignificant
        passed = pval > 0.05

        return PlaceboResult(
            placebo_name=placebo_exposure,
            coefficient=coef,
            std_error=se,
            pvalue=pval,
            passed=passed,
            notes="Non-oil exposure × oil shock should be insignificant",
        )

    def test_region_reform(
        self,
        outcome: str,
        reform_date: str,
        window: int = 4,
    ) -> ReformPlaceboResult:
        """
        Test for jumps at region reform boundaries.

        If harmonization is correct, there should be no discontinuity.

        Args:
            outcome: Outcome variable
            reform_date: Reform date (e.g., "2018Q1")
            window: Window around reform (quarters)

        Returns:
            ReformPlaceboResult
        """
        data = self.data.copy()

        # Reset index if needed
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()

        # Parse reform date
        reform_year = int(reform_date[:4])
        reform_q = int(reform_date[-1])

        # Create period indicator
        def get_period_distance(quarter: str) -> int:
            """Get distance from reform in quarters."""
            year = int(quarter[:4])
            q = int(quarter[-1])
            return (year - reform_year) * 4 + (q - reform_q)

        data["period_dist"] = data["quarter"].apply(get_period_distance)

        # Filter to window
        data = data[abs(data["period_dist"]) <= window]

        # Create post indicator
        data["post"] = (data["period_dist"] >= 0).astype(int)

        # Identify affected regions
        affected_regions = []
        if reform_date == "2018Q1":
            affected_regions = ["South Kazakhstan"]  # After harmonization
        elif reform_date == "2022Q1":
            affected_regions = ["East Kazakhstan", "Almaty Region", "Karaganda"]

        # Create treatment indicator
        data["affected"] = data["region"].isin(affected_regions).astype(int)
        data["affected_post"] = data["affected"] * data["post"]

        # Drop missing
        data = data.dropna(subset=[outcome])

        if len(data) == 0:
            return ReformPlaceboResult(
                reform_date=reform_date,
                pre_mean=np.nan,
                post_mean=np.nan,
                diff=np.nan,
                std_error=np.nan,
                pvalue=np.nan,
                passed=False,
            )

        # Set up panel
        data = data.set_index(["region", "quarter"])

        # DiD regression
        X = data[["affected_post"]]
        y = data[outcome]

        model = PanelOLS(y, X, entity_effects=True, time_effects=True)
        result = model.fit(cov_type="kernel")

        diff = result.params["affected_post"]
        se = result.std_errors["affected_post"]
        pval = result.pvalues["affected_post"]

        # Compute pre/post means for affected regions
        data = data.reset_index()
        affected_data = data[data["affected"] == 1]
        pre_mean = affected_data[affected_data["post"] == 0][outcome].mean()
        post_mean = affected_data[affected_data["post"] == 1][outcome].mean()

        # Test passes if no significant jump
        passed = pval > 0.05

        return ReformPlaceboResult(
            reform_date=reform_date,
            pre_mean=pre_mean,
            post_mean=post_mean,
            diff=diff,
            std_error=se,
            pvalue=pval,
            passed=passed,
        )

    def summary(self) -> str:
        """Print summary of falsification tests."""
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append("FALSIFICATION TEST RESULTS")
        lines.append(f"{'='*70}")

        for name, result in self.results.items():
            lines.append(f"\n{name.upper()}")
            lines.append("-" * 40)

            if isinstance(result, PreTrendsResult):
                status = "PASSED" if result.passed else "FAILED"
                lines.append(f"Status: {status}")
                lines.append(f"Joint F-stat: {result.joint_fstat:.3f}")
                lines.append(f"Joint p-value: {result.joint_pvalue:.4f}")
                lines.append("\nLead coefficients:")
                for i, lead in enumerate(result.leads):
                    stars = "***" if result.pvalues[i] < 0.01 else "**" if result.pvalues[i] < 0.05 else "*" if result.pvalues[i] < 0.1 else ""
                    lines.append(
                        f"  Lead {lead}: {result.coefficients[i]:.4f} "
                        f"(SE: {result.std_errors[i]:.4f}, p: {result.pvalues[i]:.4f}){stars}"
                    )

            elif isinstance(result, PlaceboResult):
                status = "PASSED" if result.passed else "FAILED"
                lines.append(f"Status: {status}")
                lines.append(f"Placebo: {result.placebo_name}")
                lines.append(f"Coefficient: {result.coefficient:.4f}")
                lines.append(f"Std Error: {result.std_error:.4f}")
                lines.append(f"P-value: {result.pvalue:.4f}")
                lines.append(f"Notes: {result.notes}")

            elif isinstance(result, ReformPlaceboResult):
                status = "PASSED" if result.passed else "FAILED"
                lines.append(f"Status: {status}")
                lines.append(f"Reform date: {result.reform_date}")
                lines.append(f"Pre-reform mean: {result.pre_mean:.4f}")
                lines.append(f"Post-reform mean: {result.post_mean:.4f}")
                lines.append(f"DiD estimate: {result.diff:.4f}")
                lines.append(f"Std Error: {result.std_error:.4f}")
                lines.append(f"P-value: {result.pvalue:.4f}")

        return "\n".join(lines)


def run_falsification_suite(
    panel: pd.DataFrame,
    outcome: str = "log_income_pc",
    exposure: str = "E_oil_r",
    shock: str = "oil_supply_shock",
) -> dict[str, Any]:
    """
    Run full falsification test suite.

    Args:
        panel: Panel data
        outcome: Outcome variable
        exposure: Exposure variable
        shock: Shock variable

    Returns:
        Dictionary of test results
    """
    tests = FalsificationTests(panel)
    return tests.run_all(outcome, exposure, shock)
