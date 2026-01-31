"""
Block A: CPI Category Pass-Through (Quasi-Experimental DiD)

Estimates differential inflation pass-through by import intensity.

Estimand:
    π_{c,t+h} = α_c + δ_t + Σ_{k=0}^K β_{h,k}(s_c × ΔFX_{t-k}) + ε_{c,t+h}

Exogeneity Claim:
    In absence of FX changes, high- vs low-import categories would not
    have systematically different inflation in the same month.

Key Assumption:
    s_c (import intensity) is predetermined (fixed using pre-period data).

Constructed Instrument (for Blocks B, D, E):
    Z_t ≡ Σ_c w_c × s_c × ΔFX_t
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels.panel.results import PanelEffectsResults

from shared.model.small_n_inference import (
    run_small_n_inference,
    SmallNInferenceResult,
    permutation_test,
)

logger = logging.getLogger(__name__)


@dataclass
class CPIPassThroughSpec:
    """Dynamic LP-style CPI pass-through specification."""

    # Outcome and exposure
    outcome: str = "inflation_mom"
    exposure: str = "import_share"  # s_c
    shock: str = "fx_change"  # ΔFX_t

    # Dynamic specification
    max_horizon: int = 12  # Months
    max_lags: int = 3
    include_contemporaneous: bool = True

    # Fixed effects
    category_effects: bool = True
    time_effects: bool = True

    # Exclusions
    exclude_admin_prices: bool = True
    admin_categories: list[str] = field(default_factory=lambda: [
        "04", "06", "08", "10"  # Housing, Health, Communications, Education
    ])

    # Inference
    cov_type: Literal["kernel", "clustered", "robust"] = "kernel"
    run_small_n_inference: bool = True
    bootstrap_iterations: int = 999
    permutation_iterations: int = 1000


@dataclass
class CPIPassThroughResult:
    """Block A results."""

    # Main estimates
    beta: float
    beta_se: float
    beta_ci: tuple[float, float]
    beta_pvalue: float

    # Dynamic estimates (IRF)
    horizons: list[int] | None = None
    irf_coefficients: np.ndarray | None = None
    irf_std_errors: np.ndarray | None = None
    irf_ci_lower: np.ndarray | None = None
    irf_ci_upper: np.ndarray | None = None

    # Sample info
    n_categories: int = 0
    n_months: int = 0
    n_obs: int = 0
    r2_within: float = 0.0

    # Constructed instrument
    imported_inflation: pd.Series | None = None

    # Small-N inference
    permutation_pvalue: float | None = None
    bootstrap_pvalue: float | None = None
    small_n_result: SmallNInferenceResult | None = None

    # Falsification
    pre_trend_pvalue: float | None = None
    admin_price_coefficient: float | None = None
    admin_price_pvalue: float | None = None

    # Raw results
    linearmodels_result: PanelEffectsResults | None = None

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "=" * 70,
            "Block A: CPI Category Pass-Through Results",
            "=" * 70,
            f"Main Coefficient (s_c × ΔFX): {self.beta:.4f}",
            f"Standard Error: {self.beta_se:.4f}",
            f"95% CI: [{self.beta_ci[0]:.4f}, {self.beta_ci[1]:.4f}]",
            f"p-value: {self.beta_pvalue:.4f}",
            "",
            f"N categories: {self.n_categories}",
            f"N months: {self.n_months}",
            f"N observations: {self.n_obs}",
            f"R² within: {self.r2_within:.4f}",
        ]

        if self.permutation_pvalue is not None:
            lines.extend([
                "",
                "Small-N Inference:",
                f"  Permutation p-value: {self.permutation_pvalue:.4f}",
            ])
            if self.bootstrap_pvalue is not None:
                lines.append(f"  Bootstrap p-value: {self.bootstrap_pvalue:.4f}")

        if self.pre_trend_pvalue is not None:
            lines.extend([
                "",
                "Falsification:",
                f"  Pre-trend test p-value: {self.pre_trend_pvalue:.4f}",
            ])
            if self.admin_price_pvalue is not None:
                lines.append(
                    f"  Admin price effect: {self.admin_price_coefficient:.4f} "
                    f"(p={self.admin_price_pvalue:.4f})"
                )

        return "\n".join(lines)


class CPIPassThroughModel:
    """
    CPI category pass-through model.

    Implements Block A: differential inflation response to FX shocks
    by import intensity using panel DiD.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with CPI panel data.

        Args:
            data: Panel DataFrame with columns:
                - category: COICOP category code
                - date or month: Time identifier
                - inflation_mom or cpi_index: Outcome
                - import_share or s_c: Exposure
                - fx_change: FX shock
        """
        self.data = data.copy()
        self._prepare_data()
        self.results: dict[str, CPIPassThroughResult] = {}

    def _prepare_data(self) -> None:
        """Prepare data for panel regression."""
        # Standardize column names
        col_mapping = {
            "s_c": "import_share",
            "delta_fx": "fx_change",
            "month": "date",
        }
        self.data = self.data.rename(columns={
            k: v for k, v in col_mapping.items() if k in self.data.columns
        })

        # Ensure date is datetime
        if "date" in self.data.columns:
            self.data["date"] = pd.to_datetime(self.data["date"])

        # Create time index if needed
        if "time_idx" not in self.data.columns:
            if "date" in self.data.columns:
                self.data["time_idx"] = (
                    self.data["date"].dt.year * 100 + self.data["date"].dt.month
                )

        # Create category index
        if "category_idx" not in self.data.columns:
            self.data["category_idx"] = pd.Categorical(self.data["category"]).codes

    def fit(
        self,
        spec: CPIPassThroughSpec | None = None,
        **kwargs: Any,
    ) -> CPIPassThroughResult:
        """
        Fit CPI pass-through model.

        Args:
            spec: Model specification
            **kwargs: Override specification parameters

        Returns:
            CPIPassThroughResult with estimates and inference
        """
        if spec is None:
            spec = CPIPassThroughSpec()

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(spec, key):
                setattr(spec, key, value)

        # Prepare data
        data = self._filter_data(spec)

        if len(data) == 0:
            raise ValueError("No observations after filtering")

        # Create interaction variable
        interaction_name = f"{spec.exposure}_x_{spec.shock}"
        data[interaction_name] = data[spec.exposure] * data[spec.shock]

        # Set panel index
        data = data.set_index(["category", "time_idx"])

        # Fit main model
        result = self._fit_panel_ols(data, spec, interaction_name)

        # Run small-N inference if requested
        if spec.run_small_n_inference:
            result = self._add_small_n_inference(data, spec, interaction_name, result)

        # Construct imported inflation instrument
        result.imported_inflation = self._construct_instrument(spec)

        # Store result
        self.results["main"] = result

        return result

    def _filter_data(self, spec: CPIPassThroughSpec) -> pd.DataFrame:
        """Filter data according to specification."""
        data = self.data.copy()

        # Exclude admin price categories
        if spec.exclude_admin_prices:
            data = data[~data["category"].isin(spec.admin_categories)]

        # Drop missing
        required_cols = [spec.outcome, spec.exposure, spec.shock]
        available_cols = [c for c in required_cols if c in data.columns]
        data = data.dropna(subset=available_cols)

        return data

    def _fit_panel_ols(
        self,
        data: pd.DataFrame,
        spec: CPIPassThroughSpec,
        interaction_name: str,
    ) -> CPIPassThroughResult:
        """Fit panel OLS with fixed effects."""
        # Prepare regression
        y = data[spec.outcome]
        X = data[[interaction_name]]

        # Fit PanelOLS
        model = PanelOLS(
            y,
            X,
            entity_effects=spec.category_effects,
            time_effects=spec.time_effects,
            drop_absorbed=True,
        )

        # Fit with appropriate covariance
        if spec.cov_type == "kernel":
            result = model.fit(cov_type="kernel")
        elif spec.cov_type == "clustered":
            result = model.fit(cov_type="clustered", cluster_entity=True)
        else:
            result = model.fit(cov_type=spec.cov_type)

        # Extract coefficient
        beta = result.params[interaction_name]
        beta_se = result.std_errors[interaction_name]
        beta_pvalue = result.pvalues[interaction_name]
        ci = result.conf_int().loc[interaction_name]

        return CPIPassThroughResult(
            beta=beta,
            beta_se=beta_se,
            beta_ci=(ci[0], ci[1]),
            beta_pvalue=beta_pvalue,
            n_categories=data.index.get_level_values("category").nunique(),
            n_months=data.index.get_level_values("time_idx").nunique(),
            n_obs=result.nobs,
            r2_within=result.rsquared_within,
            linearmodels_result=result,
        )

    def _add_small_n_inference(
        self,
        data: pd.DataFrame,
        spec: CPIPassThroughSpec,
        interaction_name: str,
        result: CPIPassThroughResult,
    ) -> CPIPassThroughResult:
        """Add small-N inference results."""
        # Reset index for inference functions
        data_reset = data.reset_index()

        # Define model function for inference
        def model_func(d):
            d = d.set_index(["category", "time_idx"])
            y = d[spec.outcome]
            X = d[[interaction_name]]

            model = PanelOLS(
                y, X,
                entity_effects=spec.category_effects,
                time_effects=spec.time_effects,
                drop_absorbed=True,
            )
            res = model.fit(cov_type="robust")
            return res.params[interaction_name], res.std_errors[interaction_name]

        try:
            small_n = run_small_n_inference(
                data=data_reset,
                model_func=model_func,
                exposure_var=spec.exposure,
                cluster_var="category",
                run_bootstrap=True,
                run_permutation=True,
                n_bootstrap=spec.bootstrap_iterations,
                n_permutations=spec.permutation_iterations,
            )

            result.permutation_pvalue = small_n.permutation_pvalue
            result.bootstrap_pvalue = small_n.bootstrap_pvalue
            result.small_n_result = small_n

            if small_n.bootstrap_ci:
                result.beta_ci = small_n.bootstrap_ci

        except Exception as e:
            logger.warning(f"Small-N inference failed: {e}")

        return result

    def _construct_instrument(self, spec: CPIPassThroughSpec) -> pd.Series:
        """
        Construct imported inflation instrument Z_t.

        Z_t = Σ_c w_c × s_c × ΔFX_t

        where w_c is the CPI weight of category c.
        """
        data = self.data.copy()

        # Get CPI weights (use equal weights if not available)
        if "cpi_weight" in data.columns:
            weights = data.groupby("category")["cpi_weight"].first()
        else:
            n_categories = data["category"].nunique()
            weights = pd.Series(
                1 / n_categories,
                index=data["category"].unique()
            )

        # Get import shares by category
        import_shares = data.groupby("category")[spec.exposure].first()

        # Compute weighted import share
        weighted_shares = weights * import_shares

        # Compute Z_t for each time period
        def compute_z(group):
            fx_change = group[spec.shock].iloc[0]
            z = (group[spec.exposure] * fx_change).sum() / len(group)
            return z

        z_t = data.groupby("time_idx").apply(compute_z)
        z_t.name = "imported_inflation"

        return z_t

    def fit_dynamic(
        self,
        spec: CPIPassThroughSpec | None = None,
    ) -> CPIPassThroughResult:
        """
        Fit dynamic LP-style model for impulse response.

        Estimates β_h for h = 0, 1, ..., H horizons.
        """
        if spec is None:
            spec = CPIPassThroughSpec()

        data = self._filter_data(spec)

        horizons = list(range(spec.max_horizon + 1))
        coefficients = []
        std_errors = []
        ci_lower = []
        ci_upper = []

        for h in horizons:
            try:
                # Create h-period ahead outcome
                data_h = data.copy()
                data_h[f"{spec.outcome}_h{h}"] = (
                    data_h.groupby("category")[spec.outcome].shift(-h)
                )

                # Create interaction
                interaction_name = f"{spec.exposure}_x_{spec.shock}"
                data_h[interaction_name] = data_h[spec.exposure] * data_h[spec.shock]

                # Drop missing
                data_h = data_h.dropna(subset=[f"{spec.outcome}_h{h}", interaction_name])

                if len(data_h) < 20:
                    coefficients.append(np.nan)
                    std_errors.append(np.nan)
                    ci_lower.append(np.nan)
                    ci_upper.append(np.nan)
                    continue

                # Set panel index
                data_h = data_h.set_index(["category", "time_idx"])

                # Fit model
                y = data_h[f"{spec.outcome}_h{h}"]
                X = data_h[[interaction_name]]

                model = PanelOLS(
                    y, X,
                    entity_effects=spec.category_effects,
                    time_effects=spec.time_effects,
                    drop_absorbed=True,
                )
                result = model.fit(cov_type=spec.cov_type)

                coefficients.append(result.params[interaction_name])
                std_errors.append(result.std_errors[interaction_name])
                ci = result.conf_int().loc[interaction_name]
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])

            except Exception as e:
                logger.warning(f"Horizon {h} estimation failed: {e}")
                coefficients.append(np.nan)
                std_errors.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)

        # Create result
        result = CPIPassThroughResult(
            beta=coefficients[0] if coefficients else np.nan,
            beta_se=std_errors[0] if std_errors else np.nan,
            beta_ci=(ci_lower[0], ci_upper[0]) if ci_lower else (np.nan, np.nan),
            beta_pvalue=np.nan,  # Would need to compute
            horizons=horizons,
            irf_coefficients=np.array(coefficients),
            irf_std_errors=np.array(std_errors),
            irf_ci_lower=np.array(ci_lower),
            irf_ci_upper=np.array(ci_upper),
            n_categories=data["category"].nunique(),
            n_months=data["time_idx"].nunique() if "time_idx" in data.columns else 0,
            n_obs=len(data),
        )

        result.imported_inflation = self._construct_instrument(spec)
        self.results["dynamic"] = result

        return result

    def test_pre_trends(
        self,
        spec: CPIPassThroughSpec | None = None,
        n_leads: int = 3,
    ) -> dict[str, Any]:
        """
        Test for pre-trends using leads of the shock.

        Tests H0: β_{-k} = 0 for k = 1, ..., n_leads
        """
        if spec is None:
            spec = CPIPassThroughSpec()

        data = self._filter_data(spec)

        lead_coefficients = []
        lead_pvalues = []

        for k in range(1, n_leads + 1):
            try:
                # Create k-period lead of FX shock
                data_k = data.copy()
                data_k[f"{spec.shock}_lead{k}"] = (
                    data_k.groupby("category")[spec.shock].shift(k)
                )

                # Create interaction with lead
                interaction_name = f"{spec.exposure}_x_{spec.shock}_lead{k}"
                data_k[interaction_name] = (
                    data_k[spec.exposure] * data_k[f"{spec.shock}_lead{k}"]
                )

                # Drop missing
                data_k = data_k.dropna(subset=[spec.outcome, interaction_name])
                data_k = data_k.set_index(["category", "time_idx"])

                # Fit model
                y = data_k[spec.outcome]
                X = data_k[[interaction_name]]

                model = PanelOLS(
                    y, X,
                    entity_effects=spec.category_effects,
                    time_effects=spec.time_effects,
                    drop_absorbed=True,
                )
                result = model.fit(cov_type=spec.cov_type)

                lead_coefficients.append(result.params[interaction_name])
                lead_pvalues.append(result.pvalues[interaction_name])

            except Exception as e:
                logger.warning(f"Lead {k} test failed: {e}")
                lead_coefficients.append(np.nan)
                lead_pvalues.append(np.nan)

        # Joint test (chi-square)
        from scipy import stats
        valid_pvalues = [p for p in lead_pvalues if not np.isnan(p)]
        if valid_pvalues:
            # Fisher's method for combining p-values
            chi2_stat = -2 * sum(np.log(p) for p in valid_pvalues)
            joint_pvalue = 1 - stats.chi2.cdf(chi2_stat, df=2 * len(valid_pvalues))
        else:
            joint_pvalue = np.nan

        return {
            "lead_coefficients": lead_coefficients,
            "lead_pvalues": lead_pvalues,
            "joint_pvalue": joint_pvalue,
            "pre_trends_pass": joint_pvalue > 0.05 if not np.isnan(joint_pvalue) else None,
        }

    def test_admin_prices(
        self,
        spec: CPIPassThroughSpec | None = None,
    ) -> dict[str, Any]:
        """
        Falsification test: administered prices should not respond.

        Tests that categories with admin prices (utilities, education, etc.)
        do not show FX pass-through.
        """
        if spec is None:
            spec = CPIPassThroughSpec()

        # Use admin price categories only
        data = self.data.copy()
        data = data[data["category"].isin(spec.admin_categories)]

        if len(data) < 20:
            return {
                "admin_coefficient": np.nan,
                "admin_pvalue": np.nan,
                "admin_test_pass": None,
                "error": "Insufficient admin price observations",
            }

        # Create interaction
        interaction_name = f"{spec.exposure}_x_{spec.shock}"
        data[interaction_name] = data[spec.exposure] * data[spec.shock]

        data = data.dropna(subset=[spec.outcome, interaction_name])
        data = data.set_index(["category", "time_idx"])

        try:
            y = data[spec.outcome]
            X = data[[interaction_name]]

            model = PanelOLS(
                y, X,
                entity_effects=spec.category_effects,
                time_effects=spec.time_effects,
                drop_absorbed=True,
            )
            result = model.fit(cov_type=spec.cov_type)

            return {
                "admin_coefficient": result.params[interaction_name],
                "admin_se": result.std_errors[interaction_name],
                "admin_pvalue": result.pvalues[interaction_name],
                "admin_test_pass": result.pvalues[interaction_name] > 0.05,
            }

        except Exception as e:
            return {
                "admin_coefficient": np.nan,
                "admin_pvalue": np.nan,
                "admin_test_pass": None,
                "error": str(e),
            }

    def run_all_falsification(
        self,
        spec: CPIPassThroughSpec | None = None,
    ) -> dict[str, Any]:
        """Run all falsification tests."""
        if spec is None:
            spec = CPIPassThroughSpec()

        results = {
            "pre_trends": self.test_pre_trends(spec),
            "admin_prices": self.test_admin_prices(spec),
        }

        # Summarize
        results["all_tests_pass"] = (
            results["pre_trends"].get("pre_trends_pass", False) and
            results["admin_prices"].get("admin_test_pass", False)
        )

        return results
