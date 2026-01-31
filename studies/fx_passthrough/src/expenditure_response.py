"""
Block E: Expenditure Response (LP-IV)

Estimates the causal effect of externally-driven inflation on real expenditure.

Estimand:
    ΔE_{t+h} = α_h + β_h × π̂_t + controls + ε_{t+h}

Outcome:
    Real household expenditure / "income used for consumption" (BNS concept)

This block completes the causal chain from FX to household welfare measured
through expenditure (consumption smoothing perspective).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass
class ExpenditureLPIVSpec:
    """LP-IV specification for expenditure response."""

    # Primary outcome
    outcome: str = "real_expenditure_growth"

    # Alternative outcomes for robustness
    alternative_outcomes: list[str] = field(default_factory=lambda: [
        "consumption_expenditure_growth",
        "food_expenditure_share_change",
        "nonfood_expenditure_share_change",
    ])

    # Instrument from Block A
    instrument: str = "imported_inflation"
    endogenous: str = "headline_inflation"

    # Dynamic specification
    max_horizon: int = 12

    # Controls
    controls: list[str] = field(default_factory=list)

    # Inference
    cov_type: str = "HAC"
    hac_maxlags: int | None = None


@dataclass
class ExpenditureLPIVResult:
    """Block E results."""

    outcome: str
    horizons: list[int]
    coefficients: np.ndarray
    std_errors: np.ndarray
    conf_lower: np.ndarray
    conf_upper: np.ndarray
    pvalues: np.ndarray

    # First-stage diagnostics
    first_stage_f: float
    weak_iv_flag: bool

    # Cumulative effect
    cumulative_effect: float
    cumulative_se: float

    # Sample info
    n_obs: int = 0

    # Expenditure constraint check
    expenditure_leq_income: bool | None = None  # E ≤ Y check

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "=" * 70,
            f"Block E: Expenditure LP-IV Results - {self.outcome}",
            "=" * 70,
            "",
            f"First-stage F: {self.first_stage_f:.2f}",
            f"Weak IV: {'YES - CAUTION' if self.weak_iv_flag else 'No'}",
            "",
            "Impulse Response Function:",
            f"{'Horizon':>8} {'Coef':>10} {'SE':>10} {'p-value':>10} {'95% CI':>20}",
            "-" * 60,
        ]

        for i, h in enumerate(self.horizons):
            ci_str = f"[{self.conf_lower[i]:.4f}, {self.conf_upper[i]:.4f}]"
            stars = ""
            if self.pvalues[i] < 0.01:
                stars = "***"
            elif self.pvalues[i] < 0.05:
                stars = "**"
            elif self.pvalues[i] < 0.1:
                stars = "*"
            lines.append(
                f"{h:>8} {self.coefficients[i]:>10.4f} {self.std_errors[i]:>10.4f} "
                f"{self.pvalues[i]:>8.4f}{stars:>2} {ci_str:>20}"
            )

        lines.extend([
            "",
            f"Cumulative effect: {self.cumulative_effect:.4f} (SE: {self.cumulative_se:.4f})",
            f"N observations: {self.n_obs}",
        ])

        if self.expenditure_leq_income is not None:
            lines.append(
                f"Expenditure ≤ Income constraint: "
                f"{'Satisfied' if self.expenditure_leq_income else 'VIOLATED'}"
            )

        return "\n".join(lines)


class ExpenditureLPIVModel:
    """
    Expenditure LP-IV model for Block E.

    Estimates expenditure response to inflation instrumented by
    imported inflation pressure from Block A.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with expenditure data.

        Args:
            data: DataFrame with expenditure, inflation, and instrument
        """
        self.data = data.copy()
        self._prepare_data()
        self.results: dict[str, ExpenditureLPIVResult] = {}

    def _prepare_data(self) -> None:
        """Prepare data for LP-IV estimation."""
        # Compute growth rates if needed
        for col in ["real_expenditure", "consumption_expenditure",
                    "food_expenditure", "nonfood_expenditure"]:
            if col in self.data.columns and f"{col}_growth" not in self.data.columns:
                self.data[f"{col}_growth"] = np.log(self.data[col]).diff()

        # Compute share changes if needed
        if "real_expenditure" in self.data.columns:
            for col in ["food_expenditure", "nonfood_expenditure"]:
                if col in self.data.columns:
                    share_col = f"{col}_share"
                    if share_col not in self.data.columns:
                        self.data[share_col] = self.data[col] / self.data["real_expenditure"]
                    change_col = f"{share_col}_change"
                    if change_col not in self.data.columns:
                        self.data[change_col] = self.data[share_col].diff()

        # Ensure sorted
        if "date" in self.data.columns:
            self.data = self.data.sort_values("date").reset_index(drop=True)

    def fit(
        self,
        spec: ExpenditureLPIVSpec | None = None,
        **kwargs: Any,
    ) -> ExpenditureLPIVResult:
        """
        Fit expenditure LP-IV model.

        Args:
            spec: Model specification
            **kwargs: Override parameters

        Returns:
            ExpenditureLPIVResult with IRF estimates
        """
        if spec is None:
            spec = ExpenditureLPIVSpec()

        for key, value in kwargs.items():
            if hasattr(spec, key):
                setattr(spec, key, value)

        # Check required columns
        required = [spec.outcome, spec.endogenous, spec.instrument]
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        data = self.data.dropna(subset=required)

        # First stage
        y_fs = data[spec.endogenous]
        X_fs = sm.add_constant(data[spec.instrument])
        fs_model = sm.OLS(y_fs, X_fs).fit()
        first_stage_f = fs_model.tvalues[spec.instrument] ** 2

        # LP-IV for each horizon
        horizons = list(range(spec.max_horizon + 1))
        coefficients = []
        std_errors = []
        conf_lower = []
        conf_upper = []
        pvalues = []

        for h in horizons:
            try:
                result_h = self._fit_horizon(data, spec, h)
                coefficients.append(result_h["coef"])
                std_errors.append(result_h["se"])
                conf_lower.append(result_h["ci_lower"])
                conf_upper.append(result_h["ci_upper"])
                pvalues.append(result_h["pvalue"])
            except Exception as e:
                logger.warning(f"Horizon {h} failed: {e}")
                coefficients.append(np.nan)
                std_errors.append(np.nan)
                conf_lower.append(np.nan)
                conf_upper.append(np.nan)
                pvalues.append(np.nan)

        # Cumulative effect (sum of IRF)
        cumulative = np.nansum(coefficients)
        cumulative_se = np.sqrt(np.nansum(np.array(std_errors) ** 2))

        # Check expenditure constraint
        expenditure_check = self._check_expenditure_constraint(data, spec)

        result = ExpenditureLPIVResult(
            outcome=spec.outcome,
            horizons=horizons,
            coefficients=np.array(coefficients),
            std_errors=np.array(std_errors),
            conf_lower=np.array(conf_lower),
            conf_upper=np.array(conf_upper),
            pvalues=np.array(pvalues),
            first_stage_f=first_stage_f,
            weak_iv_flag=first_stage_f < 10,
            cumulative_effect=cumulative,
            cumulative_se=cumulative_se,
            n_obs=len(data),
            expenditure_leq_income=expenditure_check,
        )

        self.results[spec.outcome] = result
        return result

    def _fit_horizon(
        self,
        data: pd.DataFrame,
        spec: ExpenditureLPIVSpec,
        horizon: int,
    ) -> dict[str, float]:
        """Fit IV regression for specific horizon."""
        # Create h-ahead outcome
        outcome_h = f"{spec.outcome}_h{horizon}"
        data = data.copy()
        data[outcome_h] = data[spec.outcome].shift(-horizon)
        data = data.dropna(subset=[outcome_h])

        if len(data) < 20:
            raise ValueError(f"Insufficient observations: {len(data)}")

        y = data[outcome_h]
        endog = data[spec.endogenous]
        instr = data[spec.instrument]

        # First stage
        X_fs = sm.add_constant(instr)
        fs_model = sm.OLS(endog, X_fs).fit()
        endog_fitted = fs_model.fittedvalues

        # Second stage
        X_ss = sm.add_constant(endog_fitted)

        if spec.cov_type == "HAC":
            maxlags = spec.hac_maxlags or int(np.floor(4 * (len(y) / 100) ** (2/9)))
            ss_model = sm.OLS(y, X_ss).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        else:
            ss_model = sm.OLS(y, X_ss).fit(cov_type=spec.cov_type)

        coef = ss_model.params.iloc[1]
        se = ss_model.bse.iloc[1]
        pvalue = ss_model.pvalues.iloc[1]
        ci = ss_model.conf_int().iloc[1]

        return {
            "coef": coef,
            "se": se,
            "pvalue": pvalue,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
        }

    def _check_expenditure_constraint(
        self,
        data: pd.DataFrame,
        spec: ExpenditureLPIVSpec,
    ) -> bool | None:
        """
        Check that expenditure ≤ income (budget constraint).

        This is a falsification/sanity check.
        """
        if "real_expenditure" not in data.columns or "nominal_income" not in data.columns:
            return None

        # Expenditure should not systematically exceed income
        exp_income_ratio = data["real_expenditure"] / data["nominal_income"]
        return exp_income_ratio.mean() <= 1.1  # Allow 10% buffer for measurement error

    def fit_all_outcomes(
        self,
        spec: ExpenditureLPIVSpec | None = None,
    ) -> dict[str, ExpenditureLPIVResult]:
        """Fit LP-IV for all expenditure outcomes."""
        if spec is None:
            spec = ExpenditureLPIVSpec()

        # Main outcome
        try:
            self.fit(spec)
        except Exception as e:
            logger.warning(f"Main outcome failed: {e}")

        # Alternative outcomes
        for outcome in spec.alternative_outcomes:
            if outcome in self.data.columns:
                try:
                    alt_spec = ExpenditureLPIVSpec(
                        outcome=outcome,
                        instrument=spec.instrument,
                        endogenous=spec.endogenous,
                        max_horizon=spec.max_horizon,
                    )
                    self.fit(alt_spec)
                except Exception as e:
                    logger.warning(f"Alternative outcome {outcome} failed: {e}")

        return self.results

    def consumption_smoothing_test(self) -> dict[str, Any]:
        """
        Test for consumption smoothing behavior.

        If households smooth consumption, expenditure response should be
        smaller than income response to transitory shocks.
        """
        if "real_expenditure_growth" not in self.results:
            raise ValueError("Fit expenditure model first")

        exp_result = self.results["real_expenditure_growth"]

        # Compare to income response (would need Block B results)
        return {
            "expenditure_irf": exp_result.coefficients,
            "cumulative_expenditure": exp_result.cumulative_effect,
            "interpretation": (
                "If expenditure response < income response, households smooth consumption. "
                "Compare with Block B income IRF for full test."
            ),
        }

    def food_share_analysis(self) -> dict[str, Any]:
        """
        Analyze food share response (Engel curve perspective).

        During inflation, food share may increase as households
        substitute toward necessities.
        """
        if "food_expenditure_share_change" not in self.results:
            # Try to fit it
            try:
                spec = ExpenditureLPIVSpec(outcome="food_expenditure_share_change")
                self.fit(spec)
            except Exception:
                return {"error": "Food share data not available"}

        food_result = self.results.get("food_expenditure_share_change")
        if food_result is None:
            return {"error": "Food share estimation failed"}

        # Food share should increase during inflation (necessity substitution)
        avg_effect = np.nanmean(food_result.coefficients[:5])

        return {
            "food_share_effect": avg_effect,
            "interpretation": (
                "Positive effect means households shift toward food (necessities) "
                "during inflation. Negative means shift toward non-food."
            ),
            "welfare_implication": (
                "Food share increase suggests welfare loss as households "
                "cut discretionary spending."
            ) if avg_effect > 0 else (
                "Food share decrease or stable suggests households maintain "
                "consumption composition."
            ),
        }
