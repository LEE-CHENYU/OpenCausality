"""
Block F: Spending Response to FX-Driven Purchasing Power Shocks

Estimates LP IRFs for income and expenditure to the same FX shock,
then computes an MPC-like ratio.

Research Question:
    How does household spending respond to FX-driven real purchasing power shocks?

Key Parameter:
    MPC-like ratio = IRF_C(h) / IRF_Y(0)

CRITICAL CAVEAT:
    This is NOT a universal MPC. It captures spending response to externally-driven
    purchasing power shocks (imported inflation via FX), which affect expenditure
    through multiple channels:
    - Real income erosion
    - Relative price changes (tradables vs non-tradables)
    - Credit constraints
    - Consumption timing shifts
    - Uncertainty effects

Estimation:
    Step 1 - Estimate shock IRFs:
        IRF_Y(h): ΔY^R_{t+h} = α_h + β_h Z_t + controls + ε_{t+h}
        IRF_C(h): ΔC^R_{t+h} = α_h + γ_h Z_t + controls + u_{t+h}

    Step 2 - Compute MPC-like ratio:
        MPC(h) ≡ IRF_C(h) / IRF_Y(0)
        or cumulative:
        MPC(h) ≡ Σ_{j=0}^h IRF_C(j) / Σ_{j=0}^h IRF_Y(j)

Shock Construction:
    Option 1 (FX-based, minimal): Z_t = Σ_{k=1}^{3} ΔFX_{t-k}
    Option 2 (Tradable pressure, leverages Block A): Z_t = predicted tradable inflation
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

logger = logging.getLogger(__name__)


class ShockType(Enum):
    """Shock construction method for Block F."""

    FX_LAGGED = "fx_lagged"  # Σ ΔFX_{t-k} for k=1..3
    TRADABLE_PRESSURE = "tradable_pressure"  # Predicted π̂^tradable from Block A


@dataclass
class SpendingResponseSpec:
    """
    Specification for Block F: Spending Response Model.

    Key parameter: MPC-like ratio = IRF_C(h) / IRF_Y(0)
    """

    # Outcome variables
    income_outcome: str = "real_income_growth"
    expenditure_outcome: str = "real_expenditure_growth"

    # Alternative expenditure outcomes for robustness
    alternative_expenditure_outcomes: list[str] = field(default_factory=lambda: [
        "real_food_expenditure_growth",
        "real_nonfood_expenditure_growth",
        "real_services_expenditure_growth",
    ])

    # Shock construction
    shock_type: ShockType = ShockType.FX_LAGGED
    shock_lags: int = 3  # Number of lags for shock construction

    # Dynamic specification
    max_horizon: int = 4  # Quarters

    # Controls (lagged to avoid bad controls)
    controls: list[str] = field(default_factory=lambda: [
        "oil_price_growth_lag",
        "global_gdp_growth_lag",
    ])

    # Inference
    cov_type: Literal["HAC", "HC1", "robust"] = "HAC"
    hac_maxlags: int | None = None  # Auto if None

    # Pre-specification choices
    deflator: str = "cpi"  # Overall CPI
    sample_start: str = "2010-Q2"
    sample_end: str | None = None  # Latest available


@dataclass
class SpendingResponseResult:
    """
    Block F results: IRFs and MPC-like ratio.

    CAVEAT: MPC-like ratio is NOT a universal MPC.
    See module docstring for interpretation guidance.
    """

    # IRF estimates
    income_irf: np.ndarray
    income_se: np.ndarray
    income_pvalues: np.ndarray
    income_ci_lower: np.ndarray
    income_ci_upper: np.ndarray

    expenditure_irf: np.ndarray
    expenditure_se: np.ndarray
    expenditure_pvalues: np.ndarray
    expenditure_ci_lower: np.ndarray
    expenditure_ci_upper: np.ndarray

    horizons: list[int]

    # MPC-like ratio
    mpc_ratio: np.ndarray  # IRF_C(h) / IRF_Y(0)
    mpc_ratio_se: np.ndarray  # Delta method SE
    cumulative_mpc: np.ndarray  # Σ IRF_C / Σ IRF_Y by horizon

    # Shock diagnostics
    shock_type: str
    shock_std: float
    shock_n_obs: int

    # Sample info
    n_obs: int
    sample_period: tuple[str, str]

    # Specification
    income_outcome: str
    expenditure_outcome: str

    def summary(self) -> str:
        """Return formatted summary with interpretation guidance."""
        lines = [
            "=" * 75,
            "Block F: Spending Response to FX-Driven Purchasing Power Shocks",
            "=" * 75,
            "",
            "INTERPRETATION CAVEAT:",
            "  MPC-like ratio is NOT a universal MPC. It captures spending response",
            "  to externally-driven FX/purchasing power shocks via multiple channels.",
            "",
            f"Shock type: {self.shock_type}",
            f"Sample: {self.sample_period[0]} to {self.sample_period[1]} (N={self.n_obs})",
            "",
            "-" * 75,
            f"INCOME IRF ({self.income_outcome})",
            "-" * 75,
            f"{'Horizon':>8} {'Coef':>10} {'SE':>10} {'p-value':>10} {'95% CI':>24}",
        ]

        for i, h in enumerate(self.horizons):
            ci_str = f"[{self.income_ci_lower[i]:.4f}, {self.income_ci_upper[i]:.4f}]"
            stars = self._significance_stars(self.income_pvalues[i])
            lines.append(
                f"{h:>8} {self.income_irf[i]:>10.4f} {self.income_se[i]:>10.4f} "
                f"{self.income_pvalues[i]:>8.4f}{stars:>2} {ci_str:>24}"
            )

        lines.extend([
            "",
            "-" * 75,
            f"EXPENDITURE IRF ({self.expenditure_outcome})",
            "-" * 75,
            f"{'Horizon':>8} {'Coef':>10} {'SE':>10} {'p-value':>10} {'95% CI':>24}",
        ])

        for i, h in enumerate(self.horizons):
            ci_str = f"[{self.expenditure_ci_lower[i]:.4f}, {self.expenditure_ci_upper[i]:.4f}]"
            stars = self._significance_stars(self.expenditure_pvalues[i])
            lines.append(
                f"{h:>8} {self.expenditure_irf[i]:>10.4f} {self.expenditure_se[i]:>10.4f} "
                f"{self.expenditure_pvalues[i]:>8.4f}{stars:>2} {ci_str:>24}"
            )

        lines.extend([
            "",
            "-" * 75,
            "MPC-LIKE RATIO: IRF_C(h) / IRF_Y(0)",
            "-" * 75,
            f"{'Horizon':>8} {'MPC(h)':>10} {'SE':>10} {'Cumul. MPC':>12}",
        ])

        for i, h in enumerate(self.horizons):
            lines.append(
                f"{h:>8} {self.mpc_ratio[i]:>10.4f} {self.mpc_ratio_se[i]:>10.4f} "
                f"{self.cumulative_mpc[i]:>12.4f}"
            )

        lines.extend([
            "",
            "=" * 75,
            "INTERPRETATION EXAMPLE:",
            f"  MPC(0) = {self.mpc_ratio[0]:.2f} means:",
            "  'For an FX-driven shock that reduces real income by 1%,",
            f"   contemporaneous real expenditure falls by {abs(self.mpc_ratio[0]):.1%}.'",
            "",
            "  This is NOT: 'For any income shock, expenditure falls by X%.'",
            "=" * 75,
        ])

        return "\n".join(lines)

    @staticmethod
    def _significance_stars(pvalue: float) -> str:
        """Return significance stars."""
        if pvalue < 0.01:
            return "***"
        elif pvalue < 0.05:
            return "**"
        elif pvalue < 0.1:
            return "*"
        return ""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame({
            "horizon": self.horizons,
            "income_irf": self.income_irf,
            "income_se": self.income_se,
            "income_pvalue": self.income_pvalues,
            "expenditure_irf": self.expenditure_irf,
            "expenditure_se": self.expenditure_se,
            "expenditure_pvalue": self.expenditure_pvalues,
            "mpc_ratio": self.mpc_ratio,
            "mpc_ratio_se": self.mpc_ratio_se,
            "cumulative_mpc": self.cumulative_mpc,
        })


class SpendingResponseModel:
    """
    Block F: Spending Response Model.

    Estimates LP IRFs for income and expenditure to FX shock,
    then computes MPC-like ratio.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with quarterly panel data.

        Args:
            data: DataFrame with columns:
                - date or quarter: Time identifier
                - Real income and expenditure series
                - FX change or imported inflation instrument
                - Optional controls
        """
        self.data = data.copy()
        self._prepare_data()
        self.results: dict[str, SpendingResponseResult] = {}

    def _prepare_data(self) -> None:
        """Prepare data for LP estimation."""
        # Standardize time column
        if "quarter" in self.data.columns and "date" not in self.data.columns:
            self.data["date"] = pd.to_datetime(
                self.data["quarter"].str.replace("Q", "-") + "-01"
            )

        # Sort by time
        if "date" in self.data.columns:
            self.data = self.data.sort_values("date").reset_index(drop=True)

        # Create time index
        if "time_idx" not in self.data.columns:
            self.data["time_idx"] = range(len(self.data))

        # Compute growth rates if not present
        for col in ["real_income", "nominal_income", "real_expenditure",
                    "consumption_expenditure", "food_expenditure",
                    "nonfood_expenditure", "services_expenditure"]:
            if col in self.data.columns:
                growth_col = f"{col}_growth"
                if growth_col not in self.data.columns:
                    self.data[growth_col] = np.log(self.data[col]).diff()

    def build_panel(
        self,
        spec: SpendingResponseSpec | None = None,
    ) -> pd.DataFrame:
        """
        Build panel for LP estimation.

        Constructs shock variable and ensures all required columns exist.
        """
        if spec is None:
            spec = SpendingResponseSpec()

        panel = self.data.copy()

        # Construct shock variable
        panel = self._construct_shock(panel, spec)

        # Filter sample period
        if spec.sample_start and "quarter" in panel.columns:
            panel = panel[panel["quarter"] >= spec.sample_start]
        if spec.sample_end and "quarter" in panel.columns:
            panel = panel[panel["quarter"] <= spec.sample_end]

        return panel

    def _construct_shock(
        self,
        data: pd.DataFrame,
        spec: SpendingResponseSpec,
    ) -> pd.DataFrame:
        """
        Construct shock variable Z_t.

        Options:
        1. FX_LAGGED: Z_t = Σ_{k=1}^{lags} ΔFX_{t-k}
        2. TRADABLE_PRESSURE: Uses Block A estimates (requires imported_inflation)
        """
        data = data.copy()

        if spec.shock_type == ShockType.FX_LAGGED:
            # Sum of lagged FX changes
            if "fx_change" not in data.columns:
                raise ValueError(
                    "fx_change column required for FX_LAGGED shock. "
                    "Run data pipeline to add FX changes."
                )

            # Create lagged sum
            shock_cols = []
            for k in range(1, spec.shock_lags + 1):
                lag_col = f"fx_change_lag{k}"
                data[lag_col] = data["fx_change"].shift(k)
                shock_cols.append(lag_col)

            data["shock"] = data[shock_cols].sum(axis=1)
            logger.info(f"Constructed FX_LAGGED shock with {spec.shock_lags} lags")

        elif spec.shock_type == ShockType.TRADABLE_PRESSURE:
            # Use imported inflation from Block A
            if "imported_inflation" not in data.columns:
                # Try to use tradable_inflation if available
                if "tradable_inflation" in data.columns:
                    shock_cols = []
                    for k in range(1, spec.shock_lags + 1):
                        lag_col = f"tradable_inflation_lag{k}"
                        data[lag_col] = data["tradable_inflation"].shift(k)
                        shock_cols.append(lag_col)
                    data["shock"] = data[shock_cols].sum(axis=1)
                else:
                    raise ValueError(
                        "imported_inflation or tradable_inflation column required "
                        "for TRADABLE_PRESSURE shock. Run Block A first."
                    )
            else:
                # Sum of lagged imported inflation
                shock_cols = []
                for k in range(1, spec.shock_lags + 1):
                    lag_col = f"imported_inflation_lag{k}"
                    data[lag_col] = data["imported_inflation"].shift(k)
                    shock_cols.append(lag_col)
                data["shock"] = data[shock_cols].sum(axis=1)

            logger.info(
                f"Constructed TRADABLE_PRESSURE shock with {spec.shock_lags} lags"
            )

        return data

    def fit(
        self,
        spec: SpendingResponseSpec | None = None,
        **kwargs: Any,
    ) -> SpendingResponseResult:
        """
        Fit Block F model: LP IRFs and MPC-like ratio.

        Args:
            spec: Model specification
            **kwargs: Override specification parameters

        Returns:
            SpendingResponseResult with IRFs and MPC ratio
        """
        if spec is None:
            spec = SpendingResponseSpec()

        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(spec, key):
                setattr(spec, key, value)

        # Build panel with shock
        panel = self.build_panel(spec)

        # Check required columns
        required = [spec.income_outcome, spec.expenditure_outcome, "shock"]
        missing = [c for c in required if c not in panel.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Estimate income IRF
        income_results = self._estimate_irf(
            panel, spec.income_outcome, spec
        )

        # Estimate expenditure IRF
        expenditure_results = self._estimate_irf(
            panel, spec.expenditure_outcome, spec
        )

        # Compute MPC-like ratio
        mpc_ratio, mpc_se = self._compute_mpc_ratio(
            income_results, expenditure_results
        )

        # Compute cumulative MPC
        cumulative_mpc = self._compute_cumulative_mpc(
            income_results, expenditure_results
        )

        # Sample info
        valid_data = panel.dropna(subset=required)
        sample_period = (
            valid_data["quarter"].min() if "quarter" in valid_data.columns else "N/A",
            valid_data["quarter"].max() if "quarter" in valid_data.columns else "N/A",
        )

        result = SpendingResponseResult(
            income_irf=income_results["coefficients"],
            income_se=income_results["std_errors"],
            income_pvalues=income_results["pvalues"],
            income_ci_lower=income_results["ci_lower"],
            income_ci_upper=income_results["ci_upper"],
            expenditure_irf=expenditure_results["coefficients"],
            expenditure_se=expenditure_results["std_errors"],
            expenditure_pvalues=expenditure_results["pvalues"],
            expenditure_ci_lower=expenditure_results["ci_lower"],
            expenditure_ci_upper=expenditure_results["ci_upper"],
            horizons=list(range(spec.max_horizon + 1)),
            mpc_ratio=mpc_ratio,
            mpc_ratio_se=mpc_se,
            cumulative_mpc=cumulative_mpc,
            shock_type=spec.shock_type.value,
            shock_std=panel["shock"].std(),
            shock_n_obs=panel["shock"].notna().sum(),
            n_obs=len(valid_data),
            sample_period=sample_period,
            income_outcome=spec.income_outcome,
            expenditure_outcome=spec.expenditure_outcome,
        )

        self.results[f"{spec.income_outcome}_{spec.expenditure_outcome}"] = result
        return result

    def _estimate_irf(
        self,
        data: pd.DataFrame,
        outcome: str,
        spec: SpendingResponseSpec,
    ) -> dict[str, np.ndarray]:
        """
        Estimate LP-IRF for a single outcome.

        ΔY_{t+h} = α_h + β_h × Z_t + controls + ε_{t+h}
        """
        horizons = list(range(spec.max_horizon + 1))
        coefficients = []
        std_errors = []
        pvalues = []
        ci_lower = []
        ci_upper = []

        for h in horizons:
            try:
                result_h = self._fit_horizon(data, outcome, spec, h)
                coefficients.append(result_h["coef"])
                std_errors.append(result_h["se"])
                pvalues.append(result_h["pvalue"])
                ci_lower.append(result_h["ci_lower"])
                ci_upper.append(result_h["ci_upper"])
            except Exception as e:
                logger.warning(f"Horizon {h} failed for {outcome}: {e}")
                coefficients.append(np.nan)
                std_errors.append(np.nan)
                pvalues.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)

        return {
            "coefficients": np.array(coefficients),
            "std_errors": np.array(std_errors),
            "pvalues": np.array(pvalues),
            "ci_lower": np.array(ci_lower),
            "ci_upper": np.array(ci_upper),
        }

    def _fit_horizon(
        self,
        data: pd.DataFrame,
        outcome: str,
        spec: SpendingResponseSpec,
        horizon: int,
    ) -> dict[str, float]:
        """
        Fit LP regression for specific horizon.

        Y_{t+h} = α + β × Z_t + controls + ε_{t+h}
        """
        data = data.copy()

        # Create h-period ahead outcome
        outcome_h = f"{outcome}_h{horizon}"
        data[outcome_h] = data[outcome].shift(-horizon)

        # Drop missing
        required = [outcome_h, "shock"]
        data = data.dropna(subset=required)

        if len(data) < 20:
            raise ValueError(f"Insufficient observations: {len(data)}")

        y = data[outcome_h]
        X = data[["shock"]]

        # Add controls if available
        for ctrl in spec.controls:
            if ctrl in data.columns:
                X = pd.concat([X, data[[ctrl]]], axis=1)

        X = sm.add_constant(X)

        # HAC standard errors for time series
        if spec.cov_type == "HAC":
            maxlags = spec.hac_maxlags or int(np.floor(4 * (len(y) / 100) ** (2/9)))
            model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        else:
            model = sm.OLS(y, X).fit(cov_type=spec.cov_type)

        coef = model.params["shock"]
        se = model.bse["shock"]
        pvalue = model.pvalues["shock"]
        ci = model.conf_int().loc["shock"]

        return {
            "coef": coef,
            "se": se,
            "pvalue": pvalue,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
        }

    def _compute_mpc_ratio(
        self,
        income_results: dict[str, np.ndarray],
        expenditure_results: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute MPC-like ratio: MPC(h) = IRF_C(h) / IRF_Y(0).

        Uses delta method for standard errors.
        """
        income_irf_0 = income_results["coefficients"][0]
        income_se_0 = income_results["std_errors"][0]

        exp_irf = expenditure_results["coefficients"]
        exp_se = expenditure_results["std_errors"]

        # MPC ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            mpc_ratio = exp_irf / income_irf_0

        # Delta method SE: Var(a/b) ≈ (1/b²)Var(a) + (a²/b⁴)Var(b)
        # Assuming independence (conservative)
        with np.errstate(divide="ignore", invalid="ignore"):
            mpc_se = np.sqrt(
                (1 / income_irf_0**2) * exp_se**2 +
                (exp_irf**2 / income_irf_0**4) * income_se_0**2
            )

        return mpc_ratio, mpc_se

    def _compute_cumulative_mpc(
        self,
        income_results: dict[str, np.ndarray],
        expenditure_results: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute cumulative MPC: Σ IRF_C / Σ IRF_Y by horizon.
        """
        income_irf = income_results["coefficients"]
        exp_irf = expenditure_results["coefficients"]

        cumulative_mpc = []
        for h in range(len(income_irf)):
            sum_income = np.nansum(income_irf[:h+1])
            sum_exp = np.nansum(exp_irf[:h+1])

            if sum_income != 0:
                cumulative_mpc.append(sum_exp / sum_income)
            else:
                cumulative_mpc.append(np.nan)

        return np.array(cumulative_mpc)

    def estimate_income_irf(
        self,
        spec: SpendingResponseSpec | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Estimate income IRF only.

        Useful for diagnostics and comparison.
        """
        if spec is None:
            spec = SpendingResponseSpec()

        panel = self.build_panel(spec)
        return self._estimate_irf(panel, spec.income_outcome, spec)

    def estimate_expenditure_irf(
        self,
        spec: SpendingResponseSpec | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Estimate expenditure IRF only.
        """
        if spec is None:
            spec = SpendingResponseSpec()

        panel = self.build_panel(spec)
        return self._estimate_irf(panel, spec.expenditure_outcome, spec)

    def run_diagnostics(
        self,
        spec: SpendingResponseSpec | None = None,
    ) -> dict[str, Any]:
        """
        Run diagnostic tests for Block F.

        Tests:
        1. Weak shock test: Is shock variable meaningful?
        2. Pre-trends: Does shock predict past outcomes?
        3. Sample characteristics
        """
        if spec is None:
            spec = SpendingResponseSpec()

        panel = self.build_panel(spec)
        diagnostics = {}

        # Shock diagnostics
        shock = panel["shock"].dropna()
        diagnostics["shock_mean"] = shock.mean()
        diagnostics["shock_std"] = shock.std()
        diagnostics["shock_n"] = len(shock)
        diagnostics["shock_nonzero_pct"] = (shock != 0).mean() * 100

        # Pre-trends test (shock shouldn't predict past outcomes)
        pre_trends_results = self._test_pre_trends(panel, spec)
        diagnostics["pre_trends"] = pre_trends_results

        # Sample characteristics
        valid_data = panel.dropna(
            subset=[spec.income_outcome, spec.expenditure_outcome, "shock"]
        )
        diagnostics["n_obs"] = len(valid_data)
        diagnostics["sample_start"] = (
            valid_data["quarter"].min() if "quarter" in valid_data.columns else None
        )
        diagnostics["sample_end"] = (
            valid_data["quarter"].max() if "quarter" in valid_data.columns else None
        )

        return diagnostics

    def _test_pre_trends(
        self,
        data: pd.DataFrame,
        spec: SpendingResponseSpec,
    ) -> dict[str, Any]:
        """
        Test pre-trends: shock shouldn't predict past outcomes.

        Y_{t-k} = α + β × Z_t + ε for k = 1, 2, 3
        """
        results = {"leads": [], "joint_pvalue": None, "pass": None}

        for outcome in [spec.income_outcome, spec.expenditure_outcome]:
            outcome_leads = []

            for k in [1, 2, 3]:
                data_k = data.copy()
                # Lead of outcome (past values)
                data_k[f"{outcome}_lead{k}"] = data_k[outcome].shift(k)
                data_k = data_k.dropna(subset=[f"{outcome}_lead{k}", "shock"])

                if len(data_k) < 20:
                    continue

                y = data_k[f"{outcome}_lead{k}"]
                X = sm.add_constant(data_k[["shock"]])

                model = sm.OLS(y, X).fit(cov_type="HC1")
                outcome_leads.append({
                    "lead": k,
                    "coef": model.params["shock"],
                    "pvalue": model.pvalues["shock"],
                })

            results["leads"].append({
                "outcome": outcome,
                "lead_tests": outcome_leads,
            })

        # Joint test: all lead coefficients should be insignificant
        all_pvalues = [
            test["pvalue"]
            for lead_group in results["leads"]
            for test in lead_group["lead_tests"]
        ]

        if all_pvalues:
            # Fisher's method for combining p-values
            chi2_stat = -2 * sum(np.log(p) for p in all_pvalues if p > 0)
            df = 2 * len(all_pvalues)
            results["joint_pvalue"] = 1 - stats.chi2.cdf(chi2_stat, df)
            results["pass"] = results["joint_pvalue"] > 0.10

        return results

    def fit_alternative_outcomes(
        self,
        spec: SpendingResponseSpec | None = None,
    ) -> dict[str, SpendingResponseResult]:
        """
        Fit model for alternative expenditure outcomes.

        Returns dict of results keyed by outcome name.
        """
        if spec is None:
            spec = SpendingResponseSpec()

        results = {}

        # Main outcome
        try:
            results["main"] = self.fit(spec)
        except Exception as e:
            logger.warning(f"Main outcome failed: {e}")

        # Alternative outcomes
        for alt_outcome in spec.alternative_expenditure_outcomes:
            if alt_outcome in self.data.columns:
                try:
                    alt_spec = SpendingResponseSpec(
                        income_outcome=spec.income_outcome,
                        expenditure_outcome=alt_outcome,
                        shock_type=spec.shock_type,
                        max_horizon=spec.max_horizon,
                    )
                    results[alt_outcome] = self.fit(alt_spec)
                except Exception as e:
                    logger.warning(f"Alternative outcome {alt_outcome} failed: {e}")

        return results

    def compare_shock_types(
        self,
        spec: SpendingResponseSpec | None = None,
    ) -> dict[str, SpendingResponseResult]:
        """
        Compare results across shock construction methods.

        Runs estimation with both FX_LAGGED and TRADABLE_PRESSURE shocks.
        """
        if spec is None:
            spec = SpendingResponseSpec()

        results = {}

        for shock_type in ShockType:
            try:
                shock_spec = SpendingResponseSpec(
                    income_outcome=spec.income_outcome,
                    expenditure_outcome=spec.expenditure_outcome,
                    shock_type=shock_type,
                    max_horizon=spec.max_horizon,
                )
                results[shock_type.value] = self.fit(shock_spec)
            except Exception as e:
                logger.warning(f"Shock type {shock_type.value} failed: {e}")

        return results
