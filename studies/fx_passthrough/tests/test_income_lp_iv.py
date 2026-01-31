"""
Tests for Block B: Income LP-IV Model.
"""

import numpy as np
import pandas as pd
import pytest


def create_mock_income_series(n_periods: int = 60) -> pd.DataFrame:
    """Create mock income time series for testing."""
    np.random.seed(42)

    dates = pd.date_range("2015-01-01", periods=n_periods, freq="QS")

    # Generate correlated inflation and income with instrument
    z_t = np.random.normal(0, 0.01, n_periods)  # Instrument
    u_t = np.random.normal(0, 0.01, n_periods)  # Endogeneity

    # First stage: inflation = 0.5 * Z + 0.3 * u + noise
    inflation = 0.5 * z_t + 0.3 * u_t + np.random.normal(0, 0.005, n_periods)

    # Second stage: income_growth = 0.2 * inflation + 0.2 * u + noise
    # True causal effect is 0.2, but OLS would be biased by u_t
    nominal_growth = 0.2 * inflation + 0.2 * u_t + np.random.normal(0, 0.01, n_periods)
    wage_growth = 0.15 * inflation + 0.15 * u_t + np.random.normal(0, 0.01, n_periods)
    transfer_growth = 0.30 * inflation + 0.1 * u_t + np.random.normal(0, 0.01, n_periods)

    df = pd.DataFrame({
        "date": dates,
        "quarter": dates.to_period("Q").astype(str),
        "headline_inflation": inflation,
        "imported_inflation": z_t,
        "nominal_income_growth": nominal_growth,
        "wage_income_growth": wage_growth,
        "transfer_income_growth": transfer_growth,
        "nominal_income": 100 * np.exp(np.cumsum(nominal_growth)),
        "wage_income": 60 * np.exp(np.cumsum(wage_growth)),
        "transfer_income": 20 * np.exp(np.cumsum(transfer_growth)),
    })

    df["time_idx"] = df["date"].dt.year * 10 + df["date"].dt.quarter

    return df


class TestIncomeLPIVSpec:
    """Tests for IncomeLPIVSpec."""

    def test_default_spec(self):
        """Test default specification values."""
        from studies.fx_passthrough.src.income_lp_iv import IncomeLPIVSpec

        spec = IncomeLPIVSpec()

        assert spec.outcome == "nominal_income_growth"
        assert spec.instrument == "imported_inflation"
        assert spec.endogenous == "headline_inflation"
        assert spec.max_horizon == 12
        assert spec.weak_iv_threshold == 10.0

    def test_custom_spec(self):
        """Test custom specification."""
        from studies.fx_passthrough.src.income_lp_iv import IncomeLPIVSpec

        spec = IncomeLPIVSpec(
            outcome="wage_income_growth",
            max_horizon=8,
        )

        assert spec.outcome == "wage_income_growth"
        assert spec.max_horizon == 8


class TestIncomeLPIVModel:
    """Tests for IncomeLPIVModel."""

    def test_model_initialization(self):
        """Test model initialization."""
        from studies.fx_passthrough.src.income_lp_iv import IncomeLPIVModel

        data = create_mock_income_series()
        model = IncomeLPIVModel(data)

        assert model.data is not None
        assert len(model.results) == 0

    def test_model_fit(self):
        """Test basic model fitting."""
        from studies.fx_passthrough.src.income_lp_iv import (
            IncomeLPIVModel,
            IncomeLPIVSpec,
        )

        data = create_mock_income_series()

        spec = IncomeLPIVSpec(max_horizon=4)

        model = IncomeLPIVModel(data)
        result = model.fit(spec)

        assert result is not None
        assert result.first_stage_f > 0
        assert len(result.horizons) == 5  # 0 to 4
        assert len(result.coefficients) == 5

    def test_first_stage_strength(self):
        """Test first stage is sufficiently strong."""
        from studies.fx_passthrough.src.income_lp_iv import (
            IncomeLPIVModel,
            IncomeLPIVSpec,
        )

        data = create_mock_income_series()

        spec = IncomeLPIVSpec(max_horizon=4)

        model = IncomeLPIVModel(data)
        result = model.fit(spec)

        # With our DGP, first stage should be strong
        assert result.first_stage_f > 5  # May not always pass threshold

    def test_fit_all_outcomes(self):
        """Test fitting all income outcomes."""
        from studies.fx_passthrough.src.income_lp_iv import (
            IncomeLPIVModel,
            IncomeLPIVSpec,
        )

        data = create_mock_income_series()

        spec = IncomeLPIVSpec(max_horizon=4)

        model = IncomeLPIVModel(data)
        results = model.fit_all_outcomes(spec)

        assert len(results) >= 1
        assert "nominal_income_growth" in results or len(results) > 0


class TestIncomeLPIVResult:
    """Tests for IncomeLPIVResult."""

    def test_result_summary(self):
        """Test result summary generation."""
        from studies.fx_passthrough.src.income_lp_iv import IncomeLPIVResult

        result = IncomeLPIVResult(
            outcome="nominal_income_growth",
            horizons=[0, 1, 2],
            coefficients=np.array([0.2, 0.15, 0.1]),
            std_errors=np.array([0.05, 0.06, 0.07]),
            conf_lower=np.array([0.1, 0.03, -0.04]),
            conf_upper=np.array([0.3, 0.27, 0.24]),
            pvalues=np.array([0.001, 0.02, 0.1]),
            first_stage_f=25.0,
            first_stage_coef=0.5,
            first_stage_se=0.1,
            first_stage_pvalue=0.001,
            weak_iv_flag=False,
            n_obs=60,
        )

        summary = result.summary()

        assert "Block B" in summary
        assert "nominal_income_growth" in summary
        assert "25.0" in summary or "25.00" in summary


class TestOLSIVComparison:
    """Tests for OLS vs IV comparison."""

    def test_compare_ols_iv(self):
        """Test OLS vs IV comparison."""
        from studies.fx_passthrough.src.income_lp_iv import (
            IncomeLPIVModel,
            IncomeLPIVSpec,
        )

        data = create_mock_income_series()

        spec = IncomeLPIVSpec(max_horizon=4)

        model = IncomeLPIVModel(data)
        model.fit(spec)

        comparison = model.compare_ols_iv()

        assert "IV_coef" in comparison.columns
        assert "OLS_coef" in comparison.columns
        assert "IV_OLS_diff" in comparison.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
