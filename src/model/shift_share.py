"""
Shift-share regression model for Kazakhstan household welfare.

Implements panel DiD with exposure × shock interactions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels.panel.results import PanelEffectsResults

logger = logging.getLogger(__name__)


@dataclass
class ShiftShareSpec:
    """Specification for a shift-share regression."""

    name: str
    outcome: str = "log_income_pc"
    interactions: list[tuple[str, str]] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)
    entity_effects: bool = True
    time_effects: bool = True
    cov_type: Literal["kernel", "clustered", "robust"] = "kernel"
    cluster_entity: bool = True
    bandwidth: int | None = None


# Default specifications
BASELINE_SPEC = ShiftShareSpec(
    name="baseline",
    outcome="log_income_pc",
    interactions=[
        ("E_oil_r", "oil_supply_shock"),
        ("E_oil_r", "aggregate_demand_shock"),
        ("E_cyc_r", "global_activity_shock"),
    ],
    controls=[],  # No mediators in baseline
    entity_effects=True,
    time_effects=True,
    cov_type="kernel",  # Driscoll-Kraay
)

AUXILIARY_SPEC = ShiftShareSpec(
    name="auxiliary_mechanisms",
    outcome="log_income_pc",
    interactions=[
        ("E_oil_r", "oil_supply_shock"),
    ],
    controls=["cpi_regional", "exchange_rate", "fiscal_transfers_pc"],
    entity_effects=True,
    time_effects=True,
    cov_type="kernel",
)


@dataclass
class RegressionResult:
    """Results from a shift-share regression."""

    spec_name: str
    params: pd.Series
    std_errors: pd.Series
    pvalues: pd.Series
    conf_int: pd.DataFrame
    nobs: int
    r2: float
    r2_within: float
    cov_type: str
    formula: str
    linearmodels_result: PanelEffectsResults | None = None


class ShiftShareModel:
    """Shift-share regression model with exposure × shock interactions."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with panel data.

        Args:
            data: Panel DataFrame with region-quarter observations
        """
        self.data = data.copy()
        self._prepare_data()
        self.results: dict[str, RegressionResult] = {}

    def _prepare_data(self) -> None:
        """Prepare data for panel regression."""
        # Set multi-index for panel
        if "region" in self.data.columns and "quarter" in self.data.columns:
            # Convert quarter string to numeric time index for linearmodels
            # linearmodels requires numeric or date-like time index
            if self.data["quarter"].dtype == object:
                # Create numeric time index from quarter string (e.g., "2020Q1" -> 20201)
                self.data["time_idx"] = self.data["quarter"].apply(
                    lambda q: int(q[:4]) * 10 + int(q[-1])
                )
            else:
                self.data["time_idx"] = self.data["quarter"]

            # Create entity (region) and time identifiers
            self.data = self.data.set_index(["region", "time_idx"])

    def fit(
        self,
        spec: ShiftShareSpec | None = None,
        **kwargs: Any,
    ) -> RegressionResult:
        """
        Fit shift-share regression.

        Args:
            spec: Regression specification
            **kwargs: Override specification parameters

        Returns:
            RegressionResult with estimates and inference
        """
        if spec is None:
            spec = BASELINE_SPEC

        # Build formula
        formula, interaction_names = self._build_formula(spec)

        # Create interaction variables
        data = self._create_interactions(spec)

        # Drop missing
        required_cols = [spec.outcome] + interaction_names + spec.controls
        available_cols = [c for c in required_cols if c in data.columns]
        data = data.dropna(subset=available_cols)

        if len(data) == 0:
            raise ValueError("No observations after dropping missing values")

        # Fit PanelOLS
        y = data[spec.outcome]
        X = data[interaction_names + [c for c in spec.controls if c in data.columns]]

        # Add constant if no controls (interactions already demeaned by FE)
        if X.empty:
            X = pd.DataFrame({"const": 1}, index=y.index)

        model = PanelOLS(
            y,
            X,
            entity_effects=spec.entity_effects,
            time_effects=spec.time_effects,
        )

        # Fit with Driscoll-Kraay or clustered SEs
        if spec.cov_type == "kernel":
            # Driscoll-Kraay HAC
            result = model.fit(cov_type="kernel", bandwidth=spec.bandwidth)
        elif spec.cov_type == "clustered":
            result = model.fit(
                cov_type="clustered",
                cluster_entity=spec.cluster_entity,
            )
        else:
            result = model.fit(cov_type=spec.cov_type)

        # Package results
        reg_result = RegressionResult(
            spec_name=spec.name,
            params=result.params,
            std_errors=result.std_errors,
            pvalues=result.pvalues,
            conf_int=result.conf_int(),
            nobs=result.nobs,
            r2=result.rsquared,
            r2_within=result.rsquared_within,
            cov_type=spec.cov_type,
            formula=formula,
            linearmodels_result=result,
        )

        self.results[spec.name] = reg_result
        return reg_result

    def _build_formula(self, spec: ShiftShareSpec) -> tuple[str, list[str]]:
        """Build regression formula and interaction names."""
        interaction_names = []

        for exposure, shock in spec.interactions:
            name = f"{exposure}_x_{shock.replace('_shock', '')}"
            interaction_names.append(name)

        # Build formula string (for documentation)
        lhs = spec.outcome
        rhs_parts = interaction_names + spec.controls

        if spec.entity_effects:
            rhs_parts.append("EntityEffects")
        if spec.time_effects:
            rhs_parts.append("TimeEffects")

        formula = f"{lhs} ~ {' + '.join(rhs_parts)}"

        return formula, interaction_names

    def _create_interactions(self, spec: ShiftShareSpec) -> pd.DataFrame:
        """Create interaction variables for regression."""
        data = self.data.copy()

        for exposure, shock in spec.interactions:
            if exposure in data.columns and shock in data.columns:
                name = f"{exposure}_x_{shock.replace('_shock', '')}"
                data[name] = data[exposure] * data[shock]

        return data

    def fit_baseline(self) -> RegressionResult:
        """Fit baseline specification."""
        return self.fit(BASELINE_SPEC)

    def fit_auxiliary(self) -> RegressionResult:
        """Fit auxiliary specification with mediators."""
        return self.fit(AUXILIARY_SPEC)

    def fit_all_specs(self) -> dict[str, RegressionResult]:
        """Fit all defined specifications."""
        self.fit(BASELINE_SPEC)
        self.fit(AUXILIARY_SPEC)
        return self.results

    def summary(self, spec_name: str | None = None) -> str:
        """Print summary of results."""
        if spec_name:
            results = {spec_name: self.results[spec_name]}
        else:
            results = self.results

        lines = []
        for name, res in results.items():
            lines.append(f"\n{'='*70}")
            lines.append(f"Specification: {name}")
            lines.append(f"{'='*70}")
            lines.append(f"Formula: {res.formula}")
            lines.append(f"N obs: {res.nobs:,}")
            lines.append(f"R² within: {res.r2_within:.4f}")
            lines.append(f"Covariance: {res.cov_type}")
            lines.append(f"\n{'Coefficient':<30} {'Estimate':>12} {'Std.Err':>12} {'p-value':>12}")
            lines.append("-" * 70)

            for var in res.params.index:
                coef = res.params[var]
                se = res.std_errors[var]
                pval = res.pvalues[var]
                stars = ""
                if pval < 0.01:
                    stars = "***"
                elif pval < 0.05:
                    stars = "**"
                elif pval < 0.1:
                    stars = "*"
                lines.append(f"{var:<30} {coef:>12.4f} {se:>12.4f} {pval:>10.4f}{stars}")

        return "\n".join(lines)

    def get_multipliers(self, spec_name: str | None = None) -> dict[str, float]:
        """Extract multipliers from results.

        Args:
            spec_name: Specification name to extract from (default: first available)

        Returns:
            Dictionary of parameter name to coefficient value
        """
        if spec_name is None:
            if "baseline" in self.results:
                spec_name = "baseline"
            elif self.results:
                spec_name = next(iter(self.results))
            else:
                raise ValueError("No results available. Fit a specification first.")

        if spec_name not in self.results:
            raise ValueError(f"Specification {spec_name} not found in results")

        result = self.results[spec_name]
        return result.params.to_dict()


def estimate_shift_share(
    panel: pd.DataFrame,
    spec: ShiftShareSpec | None = None,
) -> RegressionResult:
    """
    Convenience function to estimate shift-share model.

    Args:
        panel: Panel data
        spec: Regression specification

    Returns:
        Regression results
    """
    model = ShiftShareModel(panel)
    return model.fit(spec)
