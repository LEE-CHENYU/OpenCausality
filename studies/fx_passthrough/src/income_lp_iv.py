"""
Block B: Income Response to Externally-Driven Inflation (LP-IV)

Estimates the causal effect of inflation (instrumented by imported inflation)
on income components using Local Projections with Instrumental Variables.

First Stage:
    π_t = a + θ × Z_t + Γ(L)controls + u_t

Second Stage (LP-IV):
    ΔY_{t+h} = α_h + β_h × π̂_t + Φ_h(L)controls + ε_{t+h}

Outcomes Y:
- Nominal monetary income
- Wage income component
- Transfer income component

Exogeneity Claim:
    Imported inflation pressure Z_t affects income only through headline
    inflation (exclusion restriction).

Causal Statement:
    "Inflation plausibly driven by external FX/import-price pressure causes
    changes in nominal income components."
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class IncomeLPIVSpec:
    """LP-IV specification for income response to instrumented inflation."""

    # Outcome
    outcome: Literal[
        "nominal_income_growth",
        "wage_income_growth",
        "transfer_income_growth",
    ] = "nominal_income_growth"

    # Instrument from Block A
    instrument: str = "imported_inflation"  # Z_t
    endogenous: str = "headline_inflation"  # π_t

    # Dynamic specification
    max_horizon: int = 12  # Quarters or months

    # Controls (lags only - avoid bad controls)
    controls: list[str] = field(default_factory=lambda: [
        "oil_price_lag",
        "global_activity_lag",
    ])
    n_lags: int = 2

    # Inference
    cov_type: Literal["HC1", "HAC", "robust"] = "HAC"
    hac_maxlags: int | None = None  # Auto if None

    # Diagnostics
    weak_iv_threshold: float = 10.0  # F-statistic threshold


@dataclass
class IncomeLPIVResult:
    """Block B results with IRF."""

    outcome: str
    horizons: list[int]
    coefficients: np.ndarray
    std_errors: np.ndarray
    conf_lower: np.ndarray
    conf_upper: np.ndarray
    pvalues: np.ndarray

    # First-stage diagnostics
    first_stage_f: float
    first_stage_coef: float
    first_stage_se: float
    first_stage_pvalue: float
    weak_iv_flag: bool  # True if F < 10

    # Sample info
    n_obs: int = 0

    # OLS comparison (for checking IV vs OLS)
    ols_coefficients: np.ndarray | None = None
    ols_std_errors: np.ndarray | None = None

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "=" * 70,
            f"Block B: Income LP-IV Results - {self.outcome}",
            "=" * 70,
            "",
            "First Stage (Inflation on Imported Inflation):",
            f"  Coefficient: {self.first_stage_coef:.4f}",
            f"  Standard Error: {self.first_stage_se:.4f}",
            f"  F-statistic: {self.first_stage_f:.2f}",
            f"  Weak IV: {'YES - CAUTION' if self.weak_iv_flag else 'No'}",
            "",
            f"Second Stage IRF ({self.outcome}):",
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

        lines.append("")
        lines.append(f"N observations: {self.n_obs}")

        return "\n".join(lines)


class IncomeLPIVModel:
    """
    Income LP-IV model.

    Implements Block B: income response to inflation instrumented by
    imported inflation pressure from Block A.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with time series data.

        Args:
            data: DataFrame with columns:
                - date or quarter: Time identifier
                - nominal_income, wage_income, transfer_income: Income series
                - headline_inflation: Endogenous regressor
                - imported_inflation: Instrument from Block A
                - Optional controls
        """
        self.data = data.copy()
        self._prepare_data()
        self.results: dict[str, IncomeLPIVResult] = {}

    def _prepare_data(self) -> None:
        """Prepare data for LP-IV estimation."""
        # Standardize column names
        if "quarter" in self.data.columns and "date" not in self.data.columns:
            self.data["date"] = pd.to_datetime(
                self.data["quarter"].str.replace("Q", "-") + "-01"
            )

        # Ensure sorted by time
        if "date" in self.data.columns:
            self.data = self.data.sort_values("date").reset_index(drop=True)

        # Create time index
        self.data["time_idx"] = range(len(self.data))

        # Compute growth rates if not present
        for col in ["nominal_income", "wage_income", "transfer_income"]:
            if col in self.data.columns and f"{col}_growth" not in self.data.columns:
                self.data[f"{col}_growth"] = np.log(self.data[col]).diff()

    def fit(
        self,
        spec: IncomeLPIVSpec | None = None,
        **kwargs: Any,
    ) -> IncomeLPIVResult:
        """
        Fit LP-IV model.

        Args:
            spec: Model specification
            **kwargs: Override specification parameters

        Returns:
            IncomeLPIVResult with IRF estimates
        """
        if spec is None:
            spec = IncomeLPIVSpec()

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(spec, key):
                setattr(spec, key, value)

        # Check required columns
        required = [spec.outcome, spec.endogenous, spec.instrument]
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # First stage estimation
        first_stage_result = self._fit_first_stage(spec)

        # LP-IV for each horizon
        horizons = list(range(spec.max_horizon + 1))
        coefficients = []
        std_errors = []
        conf_lower = []
        conf_upper = []
        pvalues = []
        ols_coefficients = []
        ols_std_errors = []

        for h in horizons:
            try:
                iv_result, ols_result = self._fit_horizon(spec, h)

                coefficients.append(iv_result["coef"])
                std_errors.append(iv_result["se"])
                conf_lower.append(iv_result["ci_lower"])
                conf_upper.append(iv_result["ci_upper"])
                pvalues.append(iv_result["pvalue"])

                if ols_result:
                    ols_coefficients.append(ols_result["coef"])
                    ols_std_errors.append(ols_result["se"])

            except Exception as e:
                logger.warning(f"Horizon {h} estimation failed: {e}")
                coefficients.append(np.nan)
                std_errors.append(np.nan)
                conf_lower.append(np.nan)
                conf_upper.append(np.nan)
                pvalues.append(np.nan)
                ols_coefficients.append(np.nan)
                ols_std_errors.append(np.nan)

        result = IncomeLPIVResult(
            outcome=spec.outcome,
            horizons=horizons,
            coefficients=np.array(coefficients),
            std_errors=np.array(std_errors),
            conf_lower=np.array(conf_lower),
            conf_upper=np.array(conf_upper),
            pvalues=np.array(pvalues),
            first_stage_f=first_stage_result["f_stat"],
            first_stage_coef=first_stage_result["coef"],
            first_stage_se=first_stage_result["se"],
            first_stage_pvalue=first_stage_result["pvalue"],
            weak_iv_flag=first_stage_result["f_stat"] < spec.weak_iv_threshold,
            n_obs=len(self.data),
            ols_coefficients=np.array(ols_coefficients) if ols_coefficients else None,
            ols_std_errors=np.array(ols_std_errors) if ols_std_errors else None,
        )

        self.results[spec.outcome] = result
        return result

    def _fit_first_stage(self, spec: IncomeLPIVSpec) -> dict[str, float]:
        """
        Fit first stage: headline inflation on imported inflation.

        π_t = a + θ × Z_t + controls + u_t
        """
        data = self.data.dropna(subset=[spec.endogenous, spec.instrument])

        y = data[spec.endogenous]
        X = data[[spec.instrument]]

        # Add controls
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

        # Extract instrument coefficient
        coef = model.params[spec.instrument]
        se = model.bse[spec.instrument]
        tstat = model.tvalues[spec.instrument]
        pvalue = model.pvalues[spec.instrument]

        # F-statistic for weak IV test
        f_stat = tstat ** 2

        return {
            "coef": coef,
            "se": se,
            "tstat": tstat,
            "pvalue": pvalue,
            "f_stat": f_stat,
            "r2": model.rsquared,
        }

    def _fit_horizon(
        self,
        spec: IncomeLPIVSpec,
        horizon: int,
    ) -> tuple[dict[str, float], dict[str, float] | None]:
        """
        Fit IV regression for a specific horizon.

        ΔY_{t+h} = α + β × π̂_t + controls + ε_{t+h}
        """
        data = self.data.copy()

        # Create h-period ahead outcome
        outcome_h = f"{spec.outcome}_h{horizon}"
        data[outcome_h] = data[spec.outcome].shift(-horizon)

        # Drop missing
        required = [outcome_h, spec.endogenous, spec.instrument]
        data = data.dropna(subset=required)

        if len(data) < 20:
            raise ValueError(f"Insufficient observations: {len(data)}")

        y = data[outcome_h]
        endog = data[spec.endogenous]
        instrument = data[spec.instrument]

        # Build control matrix
        controls_data = pd.DataFrame()
        for ctrl in spec.controls:
            if ctrl in data.columns:
                controls_data[ctrl] = data[ctrl]

        # 2SLS estimation using statsmodels
        # First stage: get fitted values
        X_first = sm.add_constant(pd.concat([instrument, controls_data], axis=1))
        first_stage = sm.OLS(endog, X_first).fit()
        endog_fitted = first_stage.fittedvalues

        # Second stage
        X_second = sm.add_constant(
            pd.concat([pd.Series(endog_fitted, name=spec.endogenous), controls_data], axis=1)
        )

        # HAC for second stage
        if spec.cov_type == "HAC":
            maxlags = spec.hac_maxlags or int(np.floor(4 * (len(y) / 100) ** (2/9)))
            second_stage = sm.OLS(y, X_second).fit(
                cov_type="HAC", cov_kwds={"maxlags": maxlags}
            )
        else:
            second_stage = sm.OLS(y, X_second).fit(cov_type=spec.cov_type)

        # Note: SEs need adjustment for 2SLS
        # For simplicity, we use the naive SEs; proper 2SLS uses IV2SLS
        # Let's use linearmodels IV2SLS for correct SEs
        try:
            from linearmodels.iv import IV2SLS

            # Reshape for IV2SLS
            y_iv = y.reset_index(drop=True)
            endog_iv = endog.reset_index(drop=True)
            instr_iv = instrument.reset_index(drop=True)

            if len(controls_data) > 0:
                controls_iv = controls_data.reset_index(drop=True)
                exog = sm.add_constant(controls_iv)
            else:
                exog = pd.DataFrame({"const": 1}, index=range(len(y_iv)))

            instr_df = pd.DataFrame({spec.instrument: instr_iv})

            iv_model = IV2SLS(
                y_iv,
                exog,
                pd.DataFrame({spec.endogenous: endog_iv}),
                instr_df,
            )
            iv_result = iv_model.fit(cov_type="robust")

            coef = iv_result.params[spec.endogenous]
            se = iv_result.std_errors[spec.endogenous]
            pvalue = iv_result.pvalues[spec.endogenous]
            ci = iv_result.conf_int().loc[spec.endogenous]

            iv_dict = {
                "coef": coef,
                "se": se,
                "pvalue": pvalue,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            }

        except ImportError:
            # Fallback to naive 2SLS SEs
            coef = second_stage.params[spec.endogenous]
            se = second_stage.bse[spec.endogenous]
            pvalue = second_stage.pvalues[spec.endogenous]
            ci = second_stage.conf_int().loc[spec.endogenous]

            iv_dict = {
                "coef": coef,
                "se": se,
                "pvalue": pvalue,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            }

        # OLS comparison
        X_ols = sm.add_constant(
            pd.concat([endog.reset_index(drop=True), controls_data.reset_index(drop=True)], axis=1)
        )
        ols_model = sm.OLS(y.reset_index(drop=True), X_ols).fit(cov_type="robust")

        ols_dict = {
            "coef": ols_model.params[spec.endogenous],
            "se": ols_model.bse[spec.endogenous],
        }

        return iv_dict, ols_dict

    def fit_all_outcomes(
        self,
        spec: IncomeLPIVSpec | None = None,
    ) -> dict[str, IncomeLPIVResult]:
        """Fit LP-IV for all income outcomes."""
        if spec is None:
            spec = IncomeLPIVSpec()

        outcomes = [
            "nominal_income_growth",
            "wage_income_growth",
            "transfer_income_growth",
        ]

        for outcome in outcomes:
            if outcome in self.data.columns or self._can_create_outcome(outcome):
                try:
                    spec_copy = IncomeLPIVSpec(
                        outcome=outcome,
                        instrument=spec.instrument,
                        endogenous=spec.endogenous,
                        max_horizon=spec.max_horizon,
                        controls=spec.controls,
                    )
                    self.fit(spec_copy)
                except Exception as e:
                    logger.warning(f"Failed to fit {outcome}: {e}")

        return self.results

    def _can_create_outcome(self, outcome: str) -> bool:
        """Check if outcome can be created from available data."""
        base_col = outcome.replace("_growth", "")
        return base_col in self.data.columns

    def compare_ols_iv(self, outcome: str | None = None) -> pd.DataFrame:
        """
        Compare OLS vs IV estimates.

        Useful for assessing endogeneity bias.
        """
        if outcome is None:
            outcome = list(self.results.keys())[0]

        result = self.results.get(outcome)
        if result is None:
            raise ValueError(f"No results for {outcome}")

        if result.ols_coefficients is None:
            raise ValueError("OLS comparison not available")

        comparison = pd.DataFrame({
            "horizon": result.horizons,
            "IV_coef": result.coefficients,
            "IV_se": result.std_errors,
            "OLS_coef": result.ols_coefficients,
            "OLS_se": result.ols_std_errors,
        })

        comparison["IV_OLS_diff"] = comparison["IV_coef"] - comparison["OLS_coef"]
        comparison["diff_pct"] = (
            comparison["IV_OLS_diff"] / np.abs(comparison["OLS_coef"]) * 100
        )

        return comparison

    def hausman_test(self, outcome: str | None = None) -> dict[str, float]:
        """
        Hausman test for endogeneity.

        H0: OLS is consistent (inflation is exogenous)
        H1: OLS is inconsistent (inflation is endogenous)
        """
        if outcome is None:
            outcome = list(self.results.keys())[0]

        result = self.results.get(outcome)
        if result is None or result.ols_coefficients is None:
            raise ValueError("Need both IV and OLS results for Hausman test")

        # Use horizon 0 for Hausman test
        iv_coef = result.coefficients[0]
        iv_se = result.std_errors[0]
        ols_coef = result.ols_coefficients[0]
        ols_se = result.ols_std_errors[0]

        # Hausman statistic
        diff = iv_coef - ols_coef
        var_diff = iv_se**2 - ols_se**2  # Asymptotic variance difference

        if var_diff <= 0:
            # Use absolute value (this can happen in finite samples)
            var_diff = abs(var_diff) + 1e-10

        hausman_stat = diff**2 / var_diff
        pvalue = 1 - stats.chi2.cdf(hausman_stat, df=1)

        return {
            "hausman_statistic": hausman_stat,
            "pvalue": pvalue,
            "reject_exogeneity": pvalue < 0.05,
            "iv_coef": iv_coef,
            "ols_coef": ols_coef,
        }
