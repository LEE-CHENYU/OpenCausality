"""
Causal Chain Orchestrator for FX Passthrough Study.

Connects all blocks (A through E) to estimate the full causal chain:
    Exchange Rate → Inflation → (Nominal Income & Transfers) → Real Income → Expenditure

Provides:
1. Sequential estimation of all blocks
2. Instrument passing between blocks
3. Comprehensive falsification battery
4. Summary of full chain results
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from studies.fx_passthrough.src.cpi_pass_through import (
    CPIPassThroughModel,
    CPIPassThroughSpec,
    CPIPassThroughResult,
)
from studies.fx_passthrough.src.income_lp_iv import (
    IncomeLPIVModel,
    IncomeLPIVSpec,
    IncomeLPIVResult,
)
from studies.fx_passthrough.src.accounting_identity import (
    AccountingIdentity,
    RealIncomeDecomposition,
)
from studies.fx_passthrough.src.transfer_mechanism import (
    TransferMechanismModel,
    TransferMechanismSpec,
    TransferMechanismResult,
)
from studies.fx_passthrough.src.expenditure_response import (
    ExpenditureLPIVModel,
    ExpenditureLPIVSpec,
    ExpenditureLPIVResult,
)

logger = logging.getLogger(__name__)


@dataclass
class CausalChainConfig:
    """Configuration for full causal chain analysis."""

    # Block A
    block_a_spec: CPIPassThroughSpec = field(default_factory=CPIPassThroughSpec)

    # Block B
    block_b_outcomes: list[str] = field(default_factory=lambda: [
        "nominal_income_growth",
        "wage_income_growth",
        "transfer_income_growth",
    ])
    block_b_max_horizon: int = 12

    # Block D
    run_transfer_tests: bool = True

    # Block E
    run_expenditure: bool = True

    # Falsification
    run_falsification: bool = True

    # Structural break
    structural_break_date: str = "2015-08"
    run_structural_break: bool = True

    # Output
    output_dir: Path = field(default_factory=lambda: Path("outputs"))


@dataclass
class CausalChainSummary:
    """Summary of full causal chain results."""

    # Timestamp
    estimated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Block A summary
    block_a_fx_passthrough: float | None = None
    block_a_pvalue: float | None = None
    block_a_permutation_pvalue: float | None = None

    # Block B summary
    block_b_income_response: float | None = None
    block_b_first_stage_f: float | None = None
    block_b_weak_iv: bool | None = None

    # Block C summary
    block_c_real_income_effect: float | None = None
    block_c_price_channel: float | None = None
    block_c_nominal_channel: float | None = None

    # Block D summary
    block_d_transfer_stabilizer: bool | None = None
    block_d_composition_shift: bool | None = None

    # Block E summary
    block_e_expenditure_response: float | None = None
    block_e_cumulative_effect: float | None = None

    # Falsification summary
    all_falsification_pass: bool | None = None
    pre_trends_pass: bool | None = None
    admin_prices_pass: bool | None = None
    weak_iv_pass: bool | None = None

    # Overall
    causal_chain_valid: bool | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "=" * 70,
            "CAUSAL CHAIN ANALYSIS SUMMARY",
            f"Estimated: {self.estimated_at}",
            "=" * 70,
            "",
            "BLOCK A: FX → CPI Pass-Through",
            f"  Pass-through coefficient: {self.block_a_fx_passthrough:.4f}" if self.block_a_fx_passthrough else "  Not estimated",
            f"  p-value (OLS): {self.block_a_pvalue:.4f}" if self.block_a_pvalue else "",
            f"  p-value (permutation): {self.block_a_permutation_pvalue:.4f}" if self.block_a_permutation_pvalue else "",
            "",
            "BLOCK B: Inflation → Income (LP-IV)",
            f"  Income response: {self.block_b_income_response:.4f}" if self.block_b_income_response else "  Not estimated",
            f"  First-stage F: {self.block_b_first_stage_f:.2f}" if self.block_b_first_stage_f else "",
            f"  Weak IV: {'YES - CAUTION' if self.block_b_weak_iv else 'No'}" if self.block_b_weak_iv is not None else "",
            "",
            "BLOCK C: Real Income Decomposition",
            f"  Real income effect: {self.block_c_real_income_effect:.4f}" if self.block_c_real_income_effect else "  Not estimated",
            f"  Price channel: {self.block_c_price_channel:.4f}" if self.block_c_price_channel else "",
            f"  Nominal channel: {self.block_c_nominal_channel:.4f}" if self.block_c_nominal_channel else "",
            "",
            "BLOCK D: Transfer Mechanism",
            f"  Transfers as stabilizer: {'YES' if self.block_d_transfer_stabilizer else 'No'}" if self.block_d_transfer_stabilizer is not None else "  Not estimated",
            f"  Composition shift: {'YES' if self.block_d_composition_shift else 'No'}" if self.block_d_composition_shift is not None else "",
            "",
            "BLOCK E: Expenditure Response",
            f"  Expenditure response: {self.block_e_expenditure_response:.4f}" if self.block_e_expenditure_response else "  Not estimated",
            f"  Cumulative effect: {self.block_e_cumulative_effect:.4f}" if self.block_e_cumulative_effect else "",
            "",
            "FALSIFICATION",
            f"  All tests pass: {'YES' if self.all_falsification_pass else 'NO'}" if self.all_falsification_pass is not None else "  Not run",
            f"  Pre-trends: {'PASS' if self.pre_trends_pass else 'FAIL'}" if self.pre_trends_pass is not None else "",
            f"  Admin prices: {'PASS' if self.admin_prices_pass else 'FAIL'}" if self.admin_prices_pass is not None else "",
            f"  Weak IV: {'PASS' if self.weak_iv_pass else 'FAIL'}" if self.weak_iv_pass is not None else "",
            "",
            "=" * 70,
            f"CAUSAL CHAIN VALID: {'YES' if self.causal_chain_valid else 'NO / INCONCLUSIVE'}",
            "=" * 70,
        ]

        return "\n".join(lines)


class CausalChainAnalysis:
    """
    Orchestrates full causal chain analysis.

    Estimates all blocks sequentially, passing instruments and results
    between blocks as needed.
    """

    def __init__(
        self,
        cpi_panel: pd.DataFrame | None = None,
        income_data: pd.DataFrame | None = None,
        expenditure_data: pd.DataFrame | None = None,
        config: CausalChainConfig | None = None,
    ):
        """
        Initialize causal chain analysis.

        Args:
            cpi_panel: CPI category panel for Block A
            income_data: National income time series for Blocks B-D
            expenditure_data: Expenditure data for Block E
            config: Analysis configuration
        """
        self.cpi_panel = cpi_panel
        self.income_data = income_data
        self.expenditure_data = expenditure_data
        self.config = config or CausalChainConfig()

        # Results storage
        self.block_a_result: CPIPassThroughResult | None = None
        self.block_b_results: dict[str, IncomeLPIVResult] = {}
        self.block_c_result: RealIncomeDecomposition | None = None
        self.block_d_results: dict[str, TransferMechanismResult] = {}
        self.block_e_result: ExpenditureLPIVResult | None = None

        self.falsification_results: dict[str, Any] = {}
        self.summary: CausalChainSummary | None = None

    def set_data(
        self,
        cpi_panel: pd.DataFrame | None = None,
        income_data: pd.DataFrame | None = None,
        expenditure_data: pd.DataFrame | None = None,
    ) -> None:
        """Set data for analysis."""
        if cpi_panel is not None:
            self.cpi_panel = cpi_panel
        if income_data is not None:
            self.income_data = income_data
        if expenditure_data is not None:
            self.expenditure_data = expenditure_data

    def run_block_a(self) -> CPIPassThroughResult:
        """Run Block A: CPI Pass-Through."""
        if self.cpi_panel is None:
            raise ValueError("CPI panel data required for Block A")

        logger.info("Running Block A: CPI Pass-Through")

        model = CPIPassThroughModel(self.cpi_panel)
        self.block_a_result = model.fit(self.config.block_a_spec)

        return self.block_a_result

    def run_block_b(self) -> dict[str, IncomeLPIVResult]:
        """Run Block B: Income LP-IV."""
        if self.income_data is None:
            raise ValueError("Income data required for Block B")

        if self.block_a_result is None:
            logger.warning("Block A not run. Running now...")
            self.run_block_a()

        logger.info("Running Block B: Income LP-IV")

        # Add imported inflation instrument to income data
        if self.block_a_result.imported_inflation is not None:
            # Merge instrument into income data
            instrument = self.block_a_result.imported_inflation.reset_index()
            instrument.columns = ["time_idx", "imported_inflation"]

            if "time_idx" in self.income_data.columns:
                self.income_data = self.income_data.merge(
                    instrument, on="time_idx", how="left"
                )

        model = IncomeLPIVModel(self.income_data)

        for outcome in self.config.block_b_outcomes:
            if outcome in self.income_data.columns or self._can_create_outcome(outcome):
                try:
                    spec = IncomeLPIVSpec(
                        outcome=outcome,
                        max_horizon=self.config.block_b_max_horizon,
                    )
                    result = model.fit(spec)
                    self.block_b_results[outcome] = result
                except Exception as e:
                    logger.warning(f"Block B {outcome} failed: {e}")

        return self.block_b_results

    def _can_create_outcome(self, outcome: str) -> bool:
        """Check if outcome can be created."""
        base = outcome.replace("_growth", "")
        return base in self.income_data.columns

    def run_block_c(self) -> RealIncomeDecomposition:
        """Run Block C: Real Income Decomposition."""
        if self.block_a_result is None:
            self.run_block_a()
        if not self.block_b_results:
            self.run_block_b()

        logger.info("Running Block C: Real Income Decomposition")

        # Get nominal income result
        nominal_result = self.block_b_results.get("nominal_income_growth")
        if nominal_result is None:
            nominal_result = list(self.block_b_results.values())[0]

        identity = AccountingIdentity(
            block_a_result=self.block_a_result,
            block_b_result=nominal_result,
        )

        self.block_c_result = identity.compute_decomposition()

        return self.block_c_result

    def run_block_d(self) -> dict[str, TransferMechanismResult]:
        """Run Block D: Transfer Mechanism Tests."""
        if self.income_data is None:
            raise ValueError("Income data required for Block D")

        if not self.config.run_transfer_tests:
            logger.info("Block D skipped (disabled in config)")
            return {}

        logger.info("Running Block D: Transfer Mechanism Tests")

        model = TransferMechanismModel(self.income_data)
        self.block_d_results = model.fit_all()

        return self.block_d_results

    def run_block_e(self) -> ExpenditureLPIVResult:
        """Run Block E: Expenditure Response."""
        if not self.config.run_expenditure:
            logger.info("Block E skipped (disabled in config)")
            return None

        data = self.expenditure_data or self.income_data
        if data is None:
            raise ValueError("Expenditure or income data required for Block E")

        logger.info("Running Block E: Expenditure Response")

        model = ExpenditureLPIVModel(data)
        self.block_e_result = model.fit()

        return self.block_e_result

    def run_falsification(self) -> dict[str, Any]:
        """Run full falsification battery."""
        if not self.config.run_falsification:
            logger.info("Falsification skipped (disabled in config)")
            return {}

        logger.info("Running falsification tests")

        results = {}

        # Block A falsification
        if self.cpi_panel is not None:
            model_a = CPIPassThroughModel(self.cpi_panel)
            results["block_a"] = model_a.run_all_falsification(self.config.block_a_spec)

        # Weak IV check (Block B)
        if self.block_b_results:
            first_result = list(self.block_b_results.values())[0]
            results["weak_iv"] = {
                "first_stage_f": first_result.first_stage_f,
                "pass": first_result.first_stage_f >= 10,
            }

        # Transfer composition check (Block D)
        if self.income_data is not None:
            model_d = TransferMechanismModel(self.income_data)
            results["composition_check"] = model_d.check_accounting_identity()

        self.falsification_results = results
        return results

    def run_structural_break(self) -> dict[str, Any]:
        """Run structural break analysis (pre/post August 2015)."""
        if not self.config.run_structural_break:
            return {}

        logger.info("Running structural break analysis")

        break_date = pd.Timestamp(self.config.structural_break_date)
        results = {"break_date": str(break_date)}

        # Split CPI panel
        if self.cpi_panel is not None and "date" in self.cpi_panel.columns:
            pre_panel = self.cpi_panel[self.cpi_panel["date"] < break_date]
            post_panel = self.cpi_panel[self.cpi_panel["date"] >= break_date]

            if len(pre_panel) >= 50 and len(post_panel) >= 50:
                # Pre-period estimate
                try:
                    model_pre = CPIPassThroughModel(pre_panel)
                    result_pre = model_pre.fit(self.config.block_a_spec)
                    results["pre_period"] = {
                        "beta": result_pre.beta,
                        "se": result_pre.beta_se,
                        "pvalue": result_pre.beta_pvalue,
                        "n_obs": result_pre.n_obs,
                    }
                except Exception as e:
                    results["pre_period"] = {"error": str(e)}

                # Post-period estimate
                try:
                    model_post = CPIPassThroughModel(post_panel)
                    result_post = model_post.fit(self.config.block_a_spec)
                    results["post_period"] = {
                        "beta": result_post.beta,
                        "se": result_post.beta_se,
                        "pvalue": result_post.beta_pvalue,
                        "n_obs": result_post.n_obs,
                    }
                except Exception as e:
                    results["post_period"] = {"error": str(e)}

                # Compare
                if "beta" in results.get("pre_period", {}) and "beta" in results.get("post_period", {}):
                    pre_beta = results["pre_period"]["beta"]
                    post_beta = results["post_period"]["beta"]
                    results["beta_change"] = post_beta - pre_beta
                    results["beta_change_pct"] = (post_beta - pre_beta) / abs(pre_beta) * 100 if pre_beta != 0 else np.nan

        return results

    def run_full_chain(self) -> CausalChainSummary:
        """Run complete causal chain analysis."""
        logger.info("=" * 50)
        logger.info("RUNNING FULL CAUSAL CHAIN ANALYSIS")
        logger.info("=" * 50)

        # Block A
        try:
            self.run_block_a()
        except Exception as e:
            logger.error(f"Block A failed: {e}")

        # Block B
        try:
            self.run_block_b()
        except Exception as e:
            logger.error(f"Block B failed: {e}")

        # Block C
        try:
            self.run_block_c()
        except Exception as e:
            logger.error(f"Block C failed: {e}")

        # Block D
        try:
            self.run_block_d()
        except Exception as e:
            logger.error(f"Block D failed: {e}")

        # Block E
        try:
            self.run_block_e()
        except Exception as e:
            logger.error(f"Block E failed: {e}")

        # Falsification
        try:
            self.run_falsification()
        except Exception as e:
            logger.error(f"Falsification failed: {e}")

        # Create summary
        self.summary = self._create_summary()

        return self.summary

    def _create_summary(self) -> CausalChainSummary:
        """Create summary from all results."""
        summary = CausalChainSummary()

        # Block A
        if self.block_a_result:
            summary.block_a_fx_passthrough = self.block_a_result.beta
            summary.block_a_pvalue = self.block_a_result.beta_pvalue
            summary.block_a_permutation_pvalue = self.block_a_result.permutation_pvalue

        # Block B
        if self.block_b_results:
            nominal_result = self.block_b_results.get(
                "nominal_income_growth",
                list(self.block_b_results.values())[0]
            )
            summary.block_b_income_response = nominal_result.coefficients[0]
            summary.block_b_first_stage_f = nominal_result.first_stage_f
            summary.block_b_weak_iv = nominal_result.weak_iv_flag

        # Block C
        if self.block_c_result:
            summary.block_c_real_income_effect = self.block_c_result.real_response[-1]
            summary.block_c_price_channel = self.block_c_result.price_channel[-1]
            summary.block_c_nominal_channel = self.block_c_result.nominal_channel[-1]

        # Block D
        if self.block_d_results:
            d1 = self.block_d_results.get("d1_stabilizer")
            d2 = self.block_d_results.get("d2_composition")
            if d1:
                summary.block_d_transfer_stabilizer = d1.d1_prediction_met
            if d2:
                summary.block_d_composition_shift = (
                    d2.d2_wage_prediction_met or d2.d2_transfer_prediction_met
                )

        # Block E
        if self.block_e_result:
            summary.block_e_expenditure_response = self.block_e_result.coefficients[0]
            summary.block_e_cumulative_effect = self.block_e_result.cumulative_effect

        # Falsification
        if self.falsification_results:
            block_a_fals = self.falsification_results.get("block_a", {})
            summary.pre_trends_pass = block_a_fals.get("pre_trends", {}).get("pre_trends_pass")
            summary.admin_prices_pass = block_a_fals.get("admin_prices", {}).get("admin_test_pass")
            summary.weak_iv_pass = self.falsification_results.get("weak_iv", {}).get("pass")

            summary.all_falsification_pass = all([
                summary.pre_trends_pass or summary.pre_trends_pass is None,
                summary.admin_prices_pass or summary.admin_prices_pass is None,
                summary.weak_iv_pass or summary.weak_iv_pass is None,
            ])

        # Overall validity
        summary.causal_chain_valid = (
            summary.all_falsification_pass is not False and
            summary.block_b_weak_iv is False
        )

        return summary

    def save_results(self, output_dir: Path | None = None) -> dict[str, Path]:
        """Save all results to files."""
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Summary
        if self.summary:
            path = output_dir / "causal_chain_summary.json"
            with open(path, "w") as f:
                json.dump(self.summary.to_dict(), f, indent=2, default=str)
            paths["summary"] = path

        # Block A
        if self.block_a_result:
            path = output_dir / "block_a_results.json"
            with open(path, "w") as f:
                json.dump({
                    "beta": self.block_a_result.beta,
                    "beta_se": self.block_a_result.beta_se,
                    "beta_ci": self.block_a_result.beta_ci,
                    "beta_pvalue": self.block_a_result.beta_pvalue,
                    "permutation_pvalue": self.block_a_result.permutation_pvalue,
                    "n_categories": self.block_a_result.n_categories,
                    "n_months": self.block_a_result.n_months,
                }, f, indent=2)
            paths["block_a"] = path

        # Falsification
        if self.falsification_results:
            path = output_dir / "falsification_tests.json"
            with open(path, "w") as f:
                json.dump(self.falsification_results, f, indent=2, default=str)
            paths["falsification"] = path

        logger.info(f"Results saved to {output_dir}")
        return paths
