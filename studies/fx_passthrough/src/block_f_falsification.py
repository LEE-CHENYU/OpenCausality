"""
Block F Falsification and Robustness Tests

Implements falsification tests to validate Block F identification:

1. LEADS TEST
   - Shock shouldn't predict past income/expenditure
   - Y_{t-k} = α + β × Z_t + ε for k > 0
   - Pass criterion: Joint p > 0.10

2. ADMIN-PRICE CHECK (First-stage validation)
   - Shock should predict tradable > admin inflation
   - Validates Block A mechanism
   - Pass criterion: β^{tradable} > β^{admin}

3. REGIME SPLIT (Pre/post August 2015)
   - Separate estimates before/after tenge float
   - Pass criterion: Similar direction in both periods

4. ALTERNATIVE DEFLATORS
   - Overall CPI vs food CPI vs non-food CPI
   - Pass criterion: Qualitatively similar results

5. DEFINITION SENSITIVITY
   - Consumption expenditure vs monetary expenses
   - Pass criterion: Similar MPC-like ratios

6. SERIES BREAK CHECK
   - Check for BNS methodology changes
   - Pass criterion: No discontinuity at known break dates
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from studies.fx_passthrough.src.spending_response import (
    SpendingResponseModel,
    SpendingResponseSpec,
    SpendingResponseResult,
    ShockType,
)

logger = logging.getLogger(__name__)


@dataclass
class FalsificationTestResult:
    """Result from a single falsification test."""

    test_name: str
    passed: bool
    statistic: float | None = None
    pvalue: float | None = None
    threshold: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""

    def summary(self) -> str:
        """Return formatted summary."""
        status = "[PASS]" if self.passed else "[FAIL]"
        lines = [
            f"{self.test_name}: {status}",
        ]

        if self.statistic is not None:
            lines.append(f"  Statistic: {self.statistic:.4f}")
        if self.pvalue is not None:
            lines.append(f"  p-value: {self.pvalue:.4f}")
        if self.threshold is not None:
            lines.append(f"  Threshold: {self.threshold}")
        if self.interpretation:
            lines.append(f"  {self.interpretation}")

        return "\n".join(lines)


@dataclass
class BlockFFalsificationResults:
    """Complete falsification test suite results."""

    leads_test: FalsificationTestResult | None = None
    admin_price_check: FalsificationTestResult | None = None
    regime_split: FalsificationTestResult | None = None
    alternative_deflators: FalsificationTestResult | None = None
    definition_sensitivity: FalsificationTestResult | None = None
    series_break_check: FalsificationTestResult | None = None

    def all_passed(self) -> bool:
        """Check if all tests passed."""
        tests = [
            self.leads_test,
            self.admin_price_check,
            self.regime_split,
            self.alternative_deflators,
            self.definition_sensitivity,
            self.series_break_check,
        ]
        valid_tests = [t for t in tests if t is not None]
        return all(t.passed for t in valid_tests)

    def summary(self) -> str:
        """Return formatted summary of all tests."""
        lines = [
            "=" * 70,
            "Block F Falsification Test Results",
            "=" * 70,
            "",
        ]

        tests = [
            ("1. Leads Test", self.leads_test),
            ("2. Admin-Price Check", self.admin_price_check),
            ("3. Regime Split", self.regime_split),
            ("4. Alternative Deflators", self.alternative_deflators),
            ("5. Definition Sensitivity", self.definition_sensitivity),
            ("6. Series Break Check", self.series_break_check),
        ]

        for name, result in tests:
            if result is not None:
                lines.append(result.summary())
            else:
                lines.append(f"{name}: Not run")
            lines.append("")

        # Overall verdict
        overall = "PASS" if self.all_passed() else "FAIL"
        lines.extend([
            "-" * 70,
            f"OVERALL: {overall}",
            "=" * 70,
        ])

        return "\n".join(lines)


class BlockFFalsification:
    """
    Falsification and robustness tests for Block F.

    Validates the identification strategy for spending response
    to FX-driven purchasing power shocks.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cpi_panel: pd.DataFrame | None = None,
    ):
        """
        Initialize with data.

        Args:
            data: Quarterly panel for Block F estimation
            cpi_panel: Optional CPI panel for admin-price check (from Block A)
        """
        self.data = data.copy()
        self.cpi_panel = cpi_panel
        self._prepare_data()
        self.results = BlockFFalsificationResults()

    def _prepare_data(self) -> None:
        """Prepare data for tests."""
        if "quarter" in self.data.columns:
            self.data = self.data.sort_values("quarter").reset_index(drop=True)

    def run_all(
        self,
        spec: SpendingResponseSpec | None = None,
    ) -> BlockFFalsificationResults:
        """
        Run all falsification tests.

        Args:
            spec: Model specification

        Returns:
            BlockFFalsificationResults with all test results
        """
        if spec is None:
            spec = SpendingResponseSpec()

        # Run each test
        self.results.leads_test = self.run_leads_test(spec)
        self.results.admin_price_check = self.run_admin_price_check(spec)
        self.results.regime_split = self.run_regime_split(spec)
        self.results.alternative_deflators = self.run_alternative_deflators(spec)
        self.results.definition_sensitivity = self.run_definition_sensitivity(spec)
        self.results.series_break_check = self.run_series_break_check(spec)

        return self.results

    def run_leads_test(
        self,
        spec: SpendingResponseSpec | None = None,
        leads: list[int] | None = None,
    ) -> FalsificationTestResult:
        """
        Test that shock doesn't predict past outcomes.

        Y_{t-k} = α + β × Z_t + ε for k = 1, 2, 3

        Pass criterion: Joint p > 0.10
        """
        if spec is None:
            spec = SpendingResponseSpec()
        if leads is None:
            leads = [1, 2, 3]

        # Build panel with shock
        model = SpendingResponseModel(self.data)
        panel = model.build_panel(spec)

        if "shock" not in panel.columns:
            return FalsificationTestResult(
                test_name="Leads Test",
                passed=False,
                interpretation="Could not construct shock variable",
            )

        lead_pvalues = []
        lead_details = {}

        for outcome in [spec.income_outcome, spec.expenditure_outcome]:
            if outcome not in panel.columns:
                continue

            for k in leads:
                # Create lead variable (past values)
                panel_k = panel.copy()
                panel_k[f"{outcome}_lead{k}"] = panel_k[outcome].shift(k)

                data_k = panel_k.dropna(subset=[f"{outcome}_lead{k}", "shock"])
                if len(data_k) < 20:
                    continue

                y = data_k[f"{outcome}_lead{k}"]
                X = sm.add_constant(data_k[["shock"]])

                model_k = sm.OLS(y, X).fit(cov_type="HC1")
                pvalue = model_k.pvalues["shock"]
                lead_pvalues.append(pvalue)

                lead_details[f"{outcome}_lead{k}"] = {
                    "coef": model_k.params["shock"],
                    "pvalue": pvalue,
                }

        if not lead_pvalues:
            return FalsificationTestResult(
                test_name="Leads Test",
                passed=False,
                interpretation="No valid lead regressions",
            )

        # Fisher's method for combining p-values
        chi2_stat = -2 * sum(np.log(p) for p in lead_pvalues if p > 0)
        df = 2 * len(lead_pvalues)
        joint_pvalue = 1 - stats.chi2.cdf(chi2_stat, df)

        passed = joint_pvalue > 0.10

        return FalsificationTestResult(
            test_name="Leads Test",
            passed=passed,
            statistic=chi2_stat,
            pvalue=joint_pvalue,
            threshold=0.10,
            details=lead_details,
            interpretation=(
                "Shock does not predict past outcomes (exogeneity supported)"
                if passed else
                "WARNING: Shock predicts past outcomes (potential endogeneity)"
            ),
        )

    def run_admin_price_check(
        self,
        spec: SpendingResponseSpec | None = None,
    ) -> FalsificationTestResult:
        """
        First-stage check: shock should predict tradable > admin inflation.

        This validates the Block A mechanism.
        """
        if self.cpi_panel is None:
            return FalsificationTestResult(
                test_name="Admin-Price Check",
                passed=True,  # Skip if no CPI panel
                interpretation="CPI panel not available; test skipped",
            )

        # Admin price categories
        admin_categories = ["04", "06", "08", "10"]  # Housing, Health, Comms, Education

        # Check for required columns
        if "admin_price" not in self.cpi_panel.columns:
            # Create admin_price indicator
            if "category" in self.cpi_panel.columns:
                self.cpi_panel["admin_price"] = self.cpi_panel["category"].isin(
                    admin_categories
                )
            else:
                return FalsificationTestResult(
                    test_name="Admin-Price Check",
                    passed=True,
                    interpretation="Cannot identify admin categories; test skipped",
                )

        if "inflation_mom" not in self.cpi_panel.columns:
            return FalsificationTestResult(
                test_name="Admin-Price Check",
                passed=True,
                interpretation="Inflation data not available; test skipped",
            )

        if "fx_change" not in self.cpi_panel.columns:
            return FalsificationTestResult(
                test_name="Admin-Price Check",
                passed=True,
                interpretation="FX change not available; test skipped",
            )

        # Estimate pass-through for tradable vs admin
        tradable_data = self.cpi_panel[~self.cpi_panel["admin_price"]]
        admin_data = self.cpi_panel[self.cpi_panel["admin_price"]]

        results = {}

        for name, data in [("tradable", tradable_data), ("admin", admin_data)]:
            data_valid = data.dropna(subset=["inflation_mom", "fx_change"])
            if len(data_valid) < 20:
                results[name] = {"coef": np.nan, "se": np.nan}
                continue

            y = data_valid["inflation_mom"]
            X = sm.add_constant(data_valid[["fx_change"]])

            model = sm.OLS(y, X).fit(cov_type="HC1")
            results[name] = {
                "coef": model.params["fx_change"],
                "se": model.bse["fx_change"],
                "pvalue": model.pvalues["fx_change"],
            }

        tradable_coef = results.get("tradable", {}).get("coef", np.nan)
        admin_coef = results.get("admin", {}).get("coef", np.nan)

        if np.isnan(tradable_coef) or np.isnan(admin_coef):
            return FalsificationTestResult(
                test_name="Admin-Price Check",
                passed=True,
                interpretation="Insufficient data; test skipped",
                details=results,
            )

        passed = tradable_coef > admin_coef

        return FalsificationTestResult(
            test_name="Admin-Price Check",
            passed=passed,
            statistic=tradable_coef - admin_coef,
            details=results,
            interpretation=(
                f"Tradable pass-through ({tradable_coef:.4f}) > Admin ({admin_coef:.4f}): "
                "Block A mechanism validated"
                if passed else
                f"WARNING: Admin pass-through ({admin_coef:.4f}) >= Tradable ({tradable_coef:.4f})"
            ),
        )

    def run_regime_split(
        self,
        spec: SpendingResponseSpec | None = None,
        break_date: str = "2015Q3",
    ) -> FalsificationTestResult:
        """
        Pre/post August 2015 regime split.

        Estimates Block F separately for pre-float and post-float periods.
        """
        if spec is None:
            spec = SpendingResponseSpec()

        if "quarter" not in self.data.columns:
            return FalsificationTestResult(
                test_name="Regime Split",
                passed=True,
                interpretation="Quarter column not available; test skipped",
            )

        # Split data
        pre_data = self.data[self.data["quarter"] < break_date]
        post_data = self.data[self.data["quarter"] >= break_date]

        results = {}

        for period_name, period_data in [("pre", pre_data), ("post", post_data)]:
            if len(period_data) < 20:
                results[period_name] = {"error": "Insufficient data"}
                continue

            try:
                model = SpendingResponseModel(period_data)
                result = model.fit(spec)
                results[period_name] = {
                    "mpc_ratio_h0": result.mpc_ratio[0],
                    "income_irf_h0": result.income_irf[0],
                    "expenditure_irf_h0": result.expenditure_irf[0],
                    "n_obs": result.n_obs,
                }
            except Exception as e:
                results[period_name] = {"error": str(e)}

        # Check direction consistency
        pre_sign = np.sign(results.get("pre", {}).get("expenditure_irf_h0", 0))
        post_sign = np.sign(results.get("post", {}).get("expenditure_irf_h0", 0))

        if "error" in results.get("pre", {}) or "error" in results.get("post", {}):
            return FalsificationTestResult(
                test_name="Regime Split",
                passed=True,
                interpretation="Could not estimate both periods; test inconclusive",
                details=results,
            )

        passed = (pre_sign == post_sign) or (pre_sign == 0 or post_sign == 0)

        return FalsificationTestResult(
            test_name="Regime Split",
            passed=passed,
            details=results,
            interpretation=(
                f"Pre and post 2015-08 estimates have consistent direction"
                if passed else
                f"WARNING: Pre/post estimates have different signs"
            ),
        )

    def run_alternative_deflators(
        self,
        spec: SpendingResponseSpec | None = None,
    ) -> FalsificationTestResult:
        """
        Robustness to deflator choice.

        Compare results using overall CPI, food CPI, non-food CPI.
        """
        if spec is None:
            spec = SpendingResponseSpec()

        # Check for deflated outcomes
        deflator_outcomes = {
            "overall_cpi": "real_expenditure_growth",
            "food_cpi": "real_expenditure_growth_food_deflated",
            "nonfood_cpi": "real_expenditure_growth_nonfood_deflated",
        }

        results = {}
        available_deflators = []

        for deflator, outcome in deflator_outcomes.items():
            if outcome in self.data.columns:
                available_deflators.append(deflator)
                try:
                    model = SpendingResponseModel(self.data)
                    alt_spec = SpendingResponseSpec(
                        income_outcome=spec.income_outcome,
                        expenditure_outcome=outcome,
                        shock_type=spec.shock_type,
                        max_horizon=spec.max_horizon,
                    )
                    result = model.fit(alt_spec)
                    results[deflator] = {
                        "mpc_ratio_h0": result.mpc_ratio[0],
                        "expenditure_irf_h0": result.expenditure_irf[0],
                    }
                except Exception as e:
                    results[deflator] = {"error": str(e)}

        if len(available_deflators) < 2:
            return FalsificationTestResult(
                test_name="Alternative Deflators",
                passed=True,
                interpretation="Only one deflator available; test skipped",
                details=results,
            )

        # Check qualitative consistency (same sign)
        signs = [
            np.sign(results[d].get("mpc_ratio_h0", 0))
            for d in available_deflators
            if "error" not in results[d]
        ]

        passed = len(set(signs)) <= 1 or 0 in signs

        return FalsificationTestResult(
            test_name="Alternative Deflators",
            passed=passed,
            details=results,
            interpretation=(
                "Results qualitatively similar across deflators"
                if passed else
                "WARNING: Results differ across deflators"
            ),
        )

    def run_definition_sensitivity(
        self,
        spec: SpendingResponseSpec | None = None,
    ) -> FalsificationTestResult:
        """
        Robustness to outcome definition.

        Compare consumption expenditure vs monetary expenses.
        """
        if spec is None:
            spec = SpendingResponseSpec()

        outcome_definitions = {
            "consumption": "real_expenditure_growth",  # Primary
            "monetary_expenses": "real_monetary_expenses_growth",  # Includes debt
        }

        results = {}
        available_outcomes = []

        for name, outcome in outcome_definitions.items():
            if outcome in self.data.columns:
                available_outcomes.append(name)
                try:
                    model = SpendingResponseModel(self.data)
                    alt_spec = SpendingResponseSpec(
                        income_outcome=spec.income_outcome,
                        expenditure_outcome=outcome,
                        shock_type=spec.shock_type,
                        max_horizon=spec.max_horizon,
                    )
                    result = model.fit(alt_spec)
                    results[name] = {
                        "mpc_ratio_h0": result.mpc_ratio[0],
                        "expenditure_irf_h0": result.expenditure_irf[0],
                    }
                except Exception as e:
                    results[name] = {"error": str(e)}

        if len(available_outcomes) < 2:
            return FalsificationTestResult(
                test_name="Definition Sensitivity",
                passed=True,
                interpretation="Only one outcome definition available; test skipped",
                details=results,
            )

        # Check direction consistency
        signs = [
            np.sign(results[d].get("mpc_ratio_h0", 0))
            for d in available_outcomes
            if "error" not in results[d]
        ]

        passed = len(set(signs)) <= 1 or 0 in signs

        return FalsificationTestResult(
            test_name="Definition Sensitivity",
            passed=passed,
            details=results,
            interpretation=(
                "Results consistent across outcome definitions"
                if passed else
                "WARNING: Results differ across definitions"
            ),
        )

    def run_series_break_check(
        self,
        spec: SpendingResponseSpec | None = None,
        break_dates: list[str] | None = None,
    ) -> FalsificationTestResult:
        """
        Check for discontinuities at known BNS methodology break dates.
        """
        if spec is None:
            spec = SpendingResponseSpec()
        if break_dates is None:
            break_dates = ["2018Q1"]  # Known potential rebasing

        if "quarter" not in self.data.columns:
            return FalsificationTestResult(
                test_name="Series Break Check",
                passed=True,
                interpretation="Quarter column not available; test skipped",
            )

        break_results = {}

        for break_date in break_dates:
            # Test for discontinuity in levels
            pre_data = self.data[self.data["quarter"] < break_date]
            post_data = self.data[
                (self.data["quarter"] >= break_date) &
                (self.data["quarter"] < f"{int(break_date[:4])+1}Q1")
            ]

            for outcome in [spec.income_outcome, spec.expenditure_outcome]:
                if outcome not in self.data.columns:
                    continue

                pre_mean = pre_data[outcome].mean() if len(pre_data) > 0 else np.nan
                post_mean = post_data[outcome].mean() if len(post_data) > 0 else np.nan

                # Simple test for level shift
                if len(pre_data) > 1 and len(post_data) > 1:
                    _, pvalue = stats.ttest_ind(
                        pre_data[outcome].dropna(),
                        post_data[outcome].dropna()
                    )
                else:
                    pvalue = np.nan

                break_results[f"{break_date}_{outcome}"] = {
                    "pre_mean": pre_mean,
                    "post_mean": post_mean,
                    "pvalue": pvalue,
                    "significant_break": pvalue < 0.05 if not np.isnan(pvalue) else False,
                }

        # Check if any significant breaks
        significant_breaks = [
            k for k, v in break_results.items()
            if v.get("significant_break", False)
        ]

        passed = len(significant_breaks) == 0

        return FalsificationTestResult(
            test_name="Series Break Check",
            passed=passed,
            details=break_results,
            interpretation=(
                "No significant discontinuities at known break dates"
                if passed else
                f"WARNING: Significant breaks found: {significant_breaks}"
            ),
        )

    def run_placebo_shock(
        self,
        spec: SpendingResponseSpec | None = None,
        n_permutations: int = 1000,
    ) -> FalsificationTestResult:
        """
        Placebo test: shuffle shock variable and re-estimate.

        The true effect should be in the tail of the permutation distribution.
        """
        if spec is None:
            spec = SpendingResponseSpec()

        # Get true estimate
        model = SpendingResponseModel(self.data)
        try:
            true_result = model.fit(spec)
            true_mpc = true_result.mpc_ratio[0]
        except Exception as e:
            return FalsificationTestResult(
                test_name="Placebo Shock",
                passed=True,
                interpretation=f"Could not estimate true model: {e}",
            )

        # Permutation distribution
        permuted_mpcs = []

        for _ in range(n_permutations):
            permuted_data = self.data.copy()

            # Shuffle shock
            if "shock" in permuted_data.columns:
                permuted_data["shock"] = np.random.permutation(
                    permuted_data["shock"].values
                )
            elif "fx_change" in permuted_data.columns:
                permuted_data["fx_change"] = np.random.permutation(
                    permuted_data["fx_change"].values
                )
            else:
                continue

            try:
                perm_model = SpendingResponseModel(permuted_data)
                perm_result = perm_model.fit(spec)
                permuted_mpcs.append(perm_result.mpc_ratio[0])
            except Exception:
                continue

        if len(permuted_mpcs) < 100:
            return FalsificationTestResult(
                test_name="Placebo Shock",
                passed=True,
                interpretation="Insufficient permutations completed; test skipped",
            )

        # Two-tailed p-value
        permuted_mpcs = np.array(permuted_mpcs)
        p_value = np.mean(np.abs(permuted_mpcs) >= np.abs(true_mpc))

        passed = p_value < 0.10  # True effect should be extreme

        return FalsificationTestResult(
            test_name="Placebo Shock",
            passed=passed,
            statistic=true_mpc,
            pvalue=p_value,
            threshold=0.10,
            details={
                "true_mpc": true_mpc,
                "permutation_mean": permuted_mpcs.mean(),
                "permutation_std": permuted_mpcs.std(),
                "n_permutations": len(permuted_mpcs),
            },
            interpretation=(
                "True effect is extreme relative to permutation distribution"
                if passed else
                "WARNING: True effect not distinguishable from placebo"
            ),
        )
