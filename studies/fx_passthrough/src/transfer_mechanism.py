"""
Block D: Transfer Mechanism Tests

Tests whether social transfers act as automatic stabilizers during
inflation episodes driven by external FX/import price pressure.

D1: Transfers as Automatic Stabilizer (IV via Imported Inflation)
    ΔT_{t+h} = α_h + β_h × π̂_t + controls + ε_{t+h}
    Prediction: β_h > 0 (transfers rise in response to externally-driven inflation)

D2: Composition Mechanism
    Δ(wage_share)_{t+h} = α_h + β_h × π̂_t + ...
    Δ(transfer_share)_{t+h} = α_h + γ_h × π̂_t + ...
    Prediction: β_h < 0 (wage share falls), γ_h > 0 (transfer share rises)

D3: RDiT Around Indexation (Suggestive only due to January confounds)
    Tests for discontinuities around pension/benefit indexation dates.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass
class TransferMechanismSpec:
    """Specification for transfer mechanism tests."""

    # Test type
    test: Literal["d1_stabilizer", "d2_composition", "d3_rdit"] = "d1_stabilizer"

    # For D1: Transfer response
    outcome: str = "transfer_income_growth"
    endogenous: str = "headline_inflation"
    instrument: str = "imported_inflation"
    max_horizon: int = 12

    # For D2: Composition
    wage_share_col: str = "wage_share"
    transfer_share_col: str = "transfer_share"

    # For D3: RDiT
    indexation_months: list[int] = field(default_factory=lambda: [1, 4])
    bandwidth_months: int = 3

    # Controls
    controls: list[str] = field(default_factory=list)

    # Inference
    cov_type: str = "HAC"


@dataclass
class TransferMechanismResult:
    """Results from transfer mechanism tests."""

    test_name: str
    horizons: list[int] | None = None

    # D1: Transfer response coefficients
    coefficients: np.ndarray | None = None
    std_errors: np.ndarray | None = None
    pvalues: np.ndarray | None = None

    # D2: Composition results
    wage_share_effect: float | None = None
    wage_share_se: float | None = None
    wage_share_pvalue: float | None = None
    transfer_share_effect: float | None = None
    transfer_share_se: float | None = None
    transfer_share_pvalue: float | None = None

    # D3: RDiT results
    rdit_coefficient: float | None = None
    rdit_se: float | None = None
    rdit_pvalue: float | None = None

    # Predictions
    d1_prediction_met: bool | None = None  # β > 0
    d2_wage_prediction_met: bool | None = None  # β < 0
    d2_transfer_prediction_met: bool | None = None  # γ > 0

    # Diagnostics
    first_stage_f: float | None = None
    n_obs: int = 0

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "=" * 70,
            f"Block D: Transfer Mechanism Test - {self.test_name}",
            "=" * 70,
        ]

        if self.test_name == "d1_stabilizer":
            lines.extend([
                "",
                "D1: Transfers as Automatic Stabilizer",
                "Prediction: β > 0 (transfers rise with inflation)",
                "",
                f"{'Horizon':>8} {'Coef':>10} {'SE':>10} {'p-value':>10}",
                "-" * 40,
            ])
            if self.coefficients is not None:
                for i, h in enumerate(self.horizons):
                    stars = "***" if self.pvalues[i] < 0.01 else (
                        "**" if self.pvalues[i] < 0.05 else (
                            "*" if self.pvalues[i] < 0.1 else ""
                        )
                    )
                    lines.append(
                        f"{h:>8} {self.coefficients[i]:>10.4f} "
                        f"{self.std_errors[i]:>10.4f} {self.pvalues[i]:>8.4f}{stars:>2}"
                    )
            lines.append("")
            lines.append(
                f"Prediction met: {'YES' if self.d1_prediction_met else 'NO'}"
            )

        elif self.test_name == "d2_composition":
            lines.extend([
                "",
                "D2: Income Composition Changes",
                "Predictions: wage share falls (β < 0), transfer share rises (γ > 0)",
                "",
                f"Wage share effect: {self.wage_share_effect:.4f} "
                f"(SE: {self.wage_share_se:.4f}, p: {self.wage_share_pvalue:.4f})",
                f"  Prediction met: {'YES' if self.d2_wage_prediction_met else 'NO'}",
                "",
                f"Transfer share effect: {self.transfer_share_effect:.4f} "
                f"(SE: {self.transfer_share_se:.4f}, p: {self.transfer_share_pvalue:.4f})",
                f"  Prediction met: {'YES' if self.d2_transfer_prediction_met else 'NO'}",
            ])

        elif self.test_name == "d3_rdit":
            lines.extend([
                "",
                "D3: RDiT Around Indexation",
                "NOTE: Suggestive only due to January/timing confounds",
                "",
                f"Discontinuity coefficient: {self.rdit_coefficient:.4f}",
                f"Standard error: {self.rdit_se:.4f}",
                f"p-value: {self.rdit_pvalue:.4f}",
            ])

        lines.append("")
        lines.append(f"N observations: {self.n_obs}")

        return "\n".join(lines)


class TransferMechanismModel:
    """
    Transfer mechanism tests for Block D.

    Tests whether transfers act as automatic stabilizers during
    externally-driven inflation episodes.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with income data.

        Args:
            data: DataFrame with income components and inflation
        """
        self.data = data.copy()
        self._prepare_data()
        self.results: dict[str, TransferMechanismResult] = {}

    def _prepare_data(self) -> None:
        """Prepare data for analysis."""
        # Compute growth rates if needed
        for col in ["transfer_income", "wage_income", "nominal_income"]:
            if col in self.data.columns and f"{col}_growth" not in self.data.columns:
                self.data[f"{col}_growth"] = np.log(self.data[col]).diff()

        # Compute shares if needed
        if "nominal_income" in self.data.columns:
            if "wage_income" in self.data.columns and "wage_share" not in self.data.columns:
                self.data["wage_share"] = (
                    self.data["wage_income"] / self.data["nominal_income"]
                )
            if "transfer_income" in self.data.columns and "transfer_share" not in self.data.columns:
                self.data["transfer_share"] = (
                    self.data["transfer_income"] / self.data["nominal_income"]
                )

        # Ensure sorted
        if "date" in self.data.columns:
            self.data = self.data.sort_values("date").reset_index(drop=True)

    def fit_d1_stabilizer(
        self,
        spec: TransferMechanismSpec | None = None,
    ) -> TransferMechanismResult:
        """
        D1: Test if transfers rise in response to inflation (automatic stabilizer).

        ΔT_{t+h} = α_h + β_h × π̂_t + controls + ε_{t+h}
        """
        if spec is None:
            spec = TransferMechanismSpec(test="d1_stabilizer")

        data = self.data.dropna(subset=[spec.outcome, spec.endogenous, spec.instrument])

        horizons = list(range(spec.max_horizon + 1))
        coefficients = []
        std_errors = []
        pvalues = []

        # First stage for F-stat
        y_first = data[spec.endogenous]
        X_first = sm.add_constant(data[spec.instrument])
        first_stage = sm.OLS(y_first, X_first).fit()
        first_stage_f = first_stage.tvalues[spec.instrument] ** 2

        for h in horizons:
            try:
                # Create h-ahead outcome
                data_h = data.copy()
                outcome_h = f"{spec.outcome}_h{h}"
                data_h[outcome_h] = data_h[spec.outcome].shift(-h)
                data_h = data_h.dropna(subset=[outcome_h])

                if len(data_h) < 20:
                    coefficients.append(np.nan)
                    std_errors.append(np.nan)
                    pvalues.append(np.nan)
                    continue

                # 2SLS
                y = data_h[outcome_h]
                endog = data_h[spec.endogenous]
                instr = data_h[spec.instrument]

                # First stage
                X_fs = sm.add_constant(instr)
                fs_model = sm.OLS(endog, X_fs).fit()
                endog_fitted = fs_model.fittedvalues

                # Second stage
                X_ss = sm.add_constant(endog_fitted)
                ss_model = sm.OLS(y, X_ss).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

                coefficients.append(ss_model.params.iloc[1])
                std_errors.append(ss_model.bse.iloc[1])
                pvalues.append(ss_model.pvalues.iloc[1])

            except Exception as e:
                logger.warning(f"D1 horizon {h} failed: {e}")
                coefficients.append(np.nan)
                std_errors.append(np.nan)
                pvalues.append(np.nan)

        # Check prediction: β > 0 (transfers rise with inflation)
        # Use horizon 0-4 average for robustness
        avg_coef = np.nanmean(coefficients[:5])
        avg_pvalue = np.nanmean(pvalues[:5])

        result = TransferMechanismResult(
            test_name="d1_stabilizer",
            horizons=horizons,
            coefficients=np.array(coefficients),
            std_errors=np.array(std_errors),
            pvalues=np.array(pvalues),
            d1_prediction_met=avg_coef > 0 and avg_pvalue < 0.10,
            first_stage_f=first_stage_f,
            n_obs=len(data),
        )

        self.results["d1_stabilizer"] = result
        return result

    def fit_d2_composition(
        self,
        spec: TransferMechanismSpec | None = None,
    ) -> TransferMechanismResult:
        """
        D2: Test income composition changes.

        Wage share should fall, transfer share should rise during inflation.
        """
        if spec is None:
            spec = TransferMechanismSpec(test="d2_composition")

        data = self.data.dropna(subset=[
            spec.wage_share_col, spec.transfer_share_col,
            spec.endogenous, spec.instrument
        ])

        if len(data) < 20:
            raise ValueError("Insufficient observations for D2 test")

        # Compute share changes
        data["wage_share_change"] = data[spec.wage_share_col].diff()
        data["transfer_share_change"] = data[spec.transfer_share_col].diff()
        data = data.dropna()

        # First stage
        y_fs = data[spec.endogenous]
        X_fs = sm.add_constant(data[spec.instrument])
        fs_model = sm.OLS(y_fs, X_fs).fit()
        endog_fitted = fs_model.fittedvalues

        # Wage share IV
        y_wage = data["wage_share_change"]
        X_wage = sm.add_constant(endog_fitted)
        wage_model = sm.OLS(y_wage, X_wage).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

        # Transfer share IV
        y_transfer = data["transfer_share_change"]
        X_transfer = sm.add_constant(endog_fitted)
        transfer_model = sm.OLS(y_transfer, X_transfer).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4}
        )

        result = TransferMechanismResult(
            test_name="d2_composition",
            wage_share_effect=wage_model.params.iloc[1],
            wage_share_se=wage_model.bse.iloc[1],
            wage_share_pvalue=wage_model.pvalues.iloc[1],
            transfer_share_effect=transfer_model.params.iloc[1],
            transfer_share_se=transfer_model.bse.iloc[1],
            transfer_share_pvalue=transfer_model.pvalues.iloc[1],
            d2_wage_prediction_met=(
                wage_model.params.iloc[1] < 0 and wage_model.pvalues.iloc[1] < 0.10
            ),
            d2_transfer_prediction_met=(
                transfer_model.params.iloc[1] > 0 and transfer_model.pvalues.iloc[1] < 0.10
            ),
            first_stage_f=fs_model.tvalues[spec.instrument] ** 2,
            n_obs=len(data),
        )

        self.results["d2_composition"] = result
        return result

    def fit_d3_rdit(
        self,
        spec: TransferMechanismSpec | None = None,
    ) -> TransferMechanismResult:
        """
        D3: RDiT around pension/benefit indexation.

        Tests for discontinuity in transfer growth around indexation dates.
        NOTE: Suggestive only due to January confounds and other seasonal effects.
        """
        if spec is None:
            spec = TransferMechanismSpec(test="d3_rdit")

        data = self.data.copy()

        if "date" not in data.columns:
            raise ValueError("Date column required for RDiT")

        data["month"] = pd.to_datetime(data["date"]).dt.month

        # Create running variable: distance to nearest indexation month
        def distance_to_indexation(month, indexation_months):
            distances = [min(abs(month - im), 12 - abs(month - im))
                        for im in indexation_months]
            return min(distances)

        data["dist_to_indexation"] = data["month"].apply(
            lambda m: distance_to_indexation(m, spec.indexation_months)
        )

        # Post-indexation indicator
        data["post_indexation"] = data["month"].isin(spec.indexation_months).astype(int)

        # Filter to bandwidth
        data = data[data["dist_to_indexation"] <= spec.bandwidth_months]

        if len(data) < 20:
            raise ValueError("Insufficient observations near indexation dates")

        # Simple RD regression
        y = data["transfer_income_growth"] if "transfer_income_growth" in data.columns else (
            np.log(data["transfer_income"]).diff()
        )
        y = y.dropna()

        X = sm.add_constant(data.loc[y.index, ["post_indexation", "dist_to_indexation"]])

        model = sm.OLS(y, X).fit(cov_type="HC1")

        result = TransferMechanismResult(
            test_name="d3_rdit",
            rdit_coefficient=model.params["post_indexation"],
            rdit_se=model.bse["post_indexation"],
            rdit_pvalue=model.pvalues["post_indexation"],
            n_obs=len(y),
        )

        self.results["d3_rdit"] = result
        return result

    def fit_all(
        self,
        spec: TransferMechanismSpec | None = None,
    ) -> dict[str, TransferMechanismResult]:
        """Run all transfer mechanism tests."""
        try:
            self.fit_d1_stabilizer(spec)
        except Exception as e:
            logger.warning(f"D1 test failed: {e}")

        try:
            self.fit_d2_composition(spec)
        except Exception as e:
            logger.warning(f"D2 test failed: {e}")

        try:
            self.fit_d3_rdit(spec)
        except Exception as e:
            logger.warning(f"D3 test failed: {e}")

        return self.results

    def check_accounting_identity(self) -> dict[str, Any]:
        """
        Verify that wage + transfer shares approximately sum to 1.

        This is a consistency check, not a causal test.
        """
        if "wage_share" not in self.data.columns or "transfer_share" not in self.data.columns:
            return {"error": "Share columns not available"}

        data = self.data.dropna(subset=["wage_share", "transfer_share"])

        # Compute implied "other" share
        data["other_share"] = 1 - data["wage_share"] - data["transfer_share"]

        return {
            "mean_wage_share": data["wage_share"].mean(),
            "mean_transfer_share": data["transfer_share"].mean(),
            "mean_other_share": data["other_share"].mean(),
            "shares_sum": data["wage_share"].mean() + data["transfer_share"].mean(),
            "consistency_check": abs(data["other_share"].mean()) < 0.5,  # Other < 50%
        }
