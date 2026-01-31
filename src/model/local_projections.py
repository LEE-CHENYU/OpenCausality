"""
Local projections for dynamic impulse response functions.

Implements Jordà (2005) local projections for multi-horizon effects
with Driscoll-Kraay or HAC inference.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

logger = logging.getLogger(__name__)


@dataclass
class LocalProjectionResult:
    """Results from local projection at a single horizon."""

    horizon: int
    params: pd.Series
    std_errors: pd.Series
    pvalues: pd.Series
    conf_int: pd.DataFrame
    nobs: int


@dataclass
class IRFResult:
    """Impulse response function results across all horizons."""

    variable: str
    horizons: list[int]
    coefficients: np.ndarray
    std_errors: np.ndarray
    conf_lower: np.ndarray
    conf_upper: np.ndarray
    pvalues: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for plotting."""
        return pd.DataFrame(
            {
                "horizon": self.horizons,
                "coefficient": self.coefficients,
                "std_error": self.std_errors,
                "conf_lower": self.conf_lower,
                "conf_upper": self.conf_upper,
                "pvalue": self.pvalues,
            }
        )


@dataclass
class LocalProjectionSpec:
    """Specification for local projections."""

    outcome: str = "log_income_pc"
    interaction: tuple[str, str] = ("E_oil_r", "oil_supply_shock")
    controls: list[str] = field(default_factory=list)
    max_horizon: int = 12
    entity_effects: bool = True
    time_effects: bool = True
    cov_type: str = "kernel"


class LocalProjections:
    """
    Local projections for dynamic effects.

    Estimates:
        y_{i,t+h} - y_{i,t-1} = α_i + γ_t + β_h * (E_i × S_t) + ε_{i,t+h}

    for h = 0, 1, ..., H
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with panel data.

        Args:
            data: Panel DataFrame with region-quarter observations
        """
        self.data = data.copy()
        self.results: dict[int, LocalProjectionResult] = {}
        self.irf_results: dict[str, IRFResult] = {}

    def fit(
        self,
        spec: LocalProjectionSpec | None = None,
        **kwargs: Any,
    ) -> dict[str, IRFResult]:
        """
        Estimate local projections for all horizons.

        Args:
            spec: LP specification
            **kwargs: Override specification parameters

        Returns:
            Dictionary of IRF results by variable
        """
        if spec is None:
            spec = LocalProjectionSpec()

        # Prepare data
        data = self._prepare_data(spec)

        # Estimate for each horizon
        for h in range(spec.max_horizon + 1):
            result = self._estimate_horizon(data, spec, h)
            self.results[h] = result

        # Collect IRF for each variable
        self._collect_irfs(spec)

        return self.irf_results

    def _prepare_data(self, spec: LocalProjectionSpec) -> pd.DataFrame:
        """Prepare data for local projections."""
        data = self.data.copy()

        # Reset index if needed
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()

        # Create interaction variable
        exposure, shock = spec.interaction
        interaction_name = f"{exposure}_x_{shock.replace('_shock', '')}"

        if exposure in data.columns and shock in data.columns:
            data[interaction_name] = data[exposure] * data[shock]

        # Sort by region and time
        data = data.sort_values(["region", "quarter"])

        return data

    def _estimate_horizon(
        self,
        data: pd.DataFrame,
        spec: LocalProjectionSpec,
        horizon: int,
    ) -> LocalProjectionResult:
        """Estimate local projection at a single horizon."""
        exposure, shock = spec.interaction
        interaction_name = f"{exposure}_x_{shock.replace('_shock', '')}"

        # Create forward-looking outcome
        data = data.copy()
        data = data.sort_values(["region", "quarter"])

        # Compute h-period ahead change
        # y_{t+h} - y_{t-1}
        data["y_forward"] = data.groupby("region")[spec.outcome].shift(-horizon)
        data["y_lag"] = data.groupby("region")[spec.outcome].shift(1)
        data["dy"] = data["y_forward"] - data["y_lag"]

        # Drop missing
        required = ["dy", interaction_name] + spec.controls
        available = [c for c in required if c in data.columns]
        data = data.dropna(subset=available)

        if len(data) == 0:
            logger.warning(f"No observations for horizon {horizon}")
            return LocalProjectionResult(
                horizon=horizon,
                params=pd.Series(),
                std_errors=pd.Series(),
                pvalues=pd.Series(),
                conf_int=pd.DataFrame(),
                nobs=0,
            )

        # Set up panel - convert quarter to numeric for linearmodels
        if data["quarter"].dtype == object:
            data["time_idx"] = data["quarter"].apply(
                lambda q: int(q[:4]) * 10 + int(q[-1])
            )
        else:
            data["time_idx"] = data["quarter"]
        data = data.set_index(["region", "time_idx"])

        # Prepare regressors
        X_cols = [interaction_name] + [c for c in spec.controls if c in data.columns]
        X = data[X_cols]
        y = data["dy"]

        # Fit PanelOLS
        model = PanelOLS(
            y,
            X,
            entity_effects=spec.entity_effects,
            time_effects=spec.time_effects,
            drop_absorbed=True,
        )

        # Driscoll-Kraay for serial correlation in LP errors
        result = model.fit(cov_type=spec.cov_type)

        return LocalProjectionResult(
            horizon=horizon,
            params=result.params,
            std_errors=result.std_errors,
            pvalues=result.pvalues,
            conf_int=result.conf_int(),
            nobs=result.nobs,
        )

    def _collect_irfs(self, spec: LocalProjectionSpec) -> None:
        """Collect IRF results from horizon estimates."""
        exposure, shock = spec.interaction
        interaction_name = f"{exposure}_x_{shock.replace('_shock', '')}"

        horizons = []
        coefficients = []
        std_errors = []
        conf_lower = []
        conf_upper = []
        pvalues = []

        for h, result in sorted(self.results.items()):
            if result.nobs == 0:
                continue

            horizons.append(h)

            if interaction_name in result.params.index:
                coefficients.append(result.params[interaction_name])
                std_errors.append(result.std_errors[interaction_name])
                pvalues.append(result.pvalues[interaction_name])

                ci = result.conf_int.loc[interaction_name]
                conf_lower.append(ci["lower"])
                conf_upper.append(ci["upper"])
            else:
                coefficients.append(np.nan)
                std_errors.append(np.nan)
                pvalues.append(np.nan)
                conf_lower.append(np.nan)
                conf_upper.append(np.nan)

        self.irf_results[interaction_name] = IRFResult(
            variable=interaction_name,
            horizons=horizons,
            coefficients=np.array(coefficients),
            std_errors=np.array(std_errors),
            conf_lower=np.array(conf_lower),
            conf_upper=np.array(conf_upper),
            pvalues=np.array(pvalues),
        )

    def summary(self) -> str:
        """Print summary of IRF results."""
        lines = []
        for var, irf in self.irf_results.items():
            lines.append(f"\n{'='*70}")
            lines.append(f"IRF: {var}")
            lines.append(f"{'='*70}")
            lines.append(
                f"{'Horizon':>8} {'Coef':>12} {'Std.Err':>12} {'95% CI':>24} {'p-val':>10}"
            )
            lines.append("-" * 70)

            for i, h in enumerate(irf.horizons):
                coef = irf.coefficients[i]
                se = irf.std_errors[i]
                ci = f"[{irf.conf_lower[i]:.4f}, {irf.conf_upper[i]:.4f}]"
                pval = irf.pvalues[i]
                stars = ""
                if pval < 0.01:
                    stars = "***"
                elif pval < 0.05:
                    stars = "**"
                elif pval < 0.1:
                    stars = "*"
                lines.append(f"{h:>8} {coef:>12.4f} {se:>12.4f} {ci:>24} {pval:>8.4f}{stars}")

        return "\n".join(lines)

    def plot_irf(
        self,
        variable: str | None = None,
        figsize: tuple[int, int] = (10, 6),
    ) -> Any:
        """
        Plot impulse response function.

        Args:
            variable: Variable to plot (default: first variable)
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, cannot plot")
            return None

        if variable is None:
            variable = list(self.irf_results.keys())[0]

        irf = self.irf_results[variable]
        df = irf.to_dataframe()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot point estimates
        ax.plot(df["horizon"], df["coefficient"], "b-", linewidth=2, label="Point estimate")

        # Plot confidence interval
        ax.fill_between(
            df["horizon"],
            df["conf_lower"],
            df["conf_upper"],
            alpha=0.3,
            color="blue",
            label="95% CI",
        )

        # Reference line at zero
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)

        ax.set_xlabel("Horizon (quarters)")
        ax.set_ylabel("Effect on log income")
        ax.set_title(f"Impulse Response: {variable}")
        ax.legend()

        return fig


def estimate_local_projections(
    panel: pd.DataFrame,
    spec: LocalProjectionSpec | None = None,
) -> dict[str, IRFResult]:
    """
    Convenience function to estimate local projections.

    Args:
        panel: Panel data
        spec: LP specification

    Returns:
        Dictionary of IRF results
    """
    lp = LocalProjections(panel)
    return lp.fit(spec)
