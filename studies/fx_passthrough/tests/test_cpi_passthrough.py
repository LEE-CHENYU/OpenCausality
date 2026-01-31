"""
Tests for Block A: CPI Pass-Through Model.
"""

import numpy as np
import pandas as pd
import pytest


def create_mock_cpi_panel(n_categories: int = 8, n_months: int = 60) -> pd.DataFrame:
    """Create mock CPI panel data for testing."""
    np.random.seed(42)

    categories = [f"{i:02d}" for i in range(1, n_categories + 1)]
    dates = pd.date_range("2015-01-01", periods=n_months, freq="MS")

    records = []
    for cat in categories:
        # Category-specific import share (predetermined)
        import_share = 0.3 + 0.5 * (int(cat) / n_categories)

        for date in dates:
            # Generate FX change
            fx_change = np.random.normal(0, 0.02)

            # Generate inflation with pass-through
            base_inflation = np.random.normal(0.005, 0.01)
            passthrough_effect = 0.3 * import_share * fx_change
            inflation = base_inflation + passthrough_effect + np.random.normal(0, 0.005)

            records.append({
                "category": cat,
                "date": date,
                "cpi_index": 100 + np.random.normal(0, 5),
                "inflation_mom": inflation,
                "import_share": import_share,
                "fx_change": fx_change,
                "tradable": int(cat) <= 5,
                "admin_price": cat in ["04", "06"],
            })

    df = pd.DataFrame(records)
    df["time_idx"] = df["date"].dt.year * 100 + df["date"].dt.month
    df["category_idx"] = pd.Categorical(df["category"]).codes

    return df


class TestCPIPassThroughSpec:
    """Tests for CPIPassThroughSpec."""

    def test_default_spec(self):
        """Test default specification values."""
        from studies.fx_passthrough.src.cpi_pass_through import CPIPassThroughSpec

        spec = CPIPassThroughSpec()

        assert spec.outcome == "inflation_mom"
        assert spec.exposure == "import_share"
        assert spec.shock == "fx_change"
        assert spec.max_horizon == 12
        assert spec.category_effects is True
        assert spec.time_effects is True
        assert spec.exclude_admin_prices is True
        assert "04" in spec.admin_categories

    def test_custom_spec(self):
        """Test custom specification."""
        from studies.fx_passthrough.src.cpi_pass_through import CPIPassThroughSpec

        spec = CPIPassThroughSpec(
            max_horizon=6,
            exclude_admin_prices=False,
            run_small_n_inference=False,
        )

        assert spec.max_horizon == 6
        assert spec.exclude_admin_prices is False


class TestCPIPassThroughModel:
    """Tests for CPIPassThroughModel."""

    def test_model_initialization(self):
        """Test model initialization with mock data."""
        from studies.fx_passthrough.src.cpi_pass_through import CPIPassThroughModel

        panel = create_mock_cpi_panel()
        model = CPIPassThroughModel(panel)

        assert model.data is not None
        assert len(model.results) == 0

    def test_model_fit(self):
        """Test basic model fitting."""
        from studies.fx_passthrough.src.cpi_pass_through import (
            CPIPassThroughModel,
            CPIPassThroughSpec,
        )

        panel = create_mock_cpi_panel()

        spec = CPIPassThroughSpec(
            run_small_n_inference=False,  # Skip for faster test
        )

        model = CPIPassThroughModel(panel)
        result = model.fit(spec)

        assert result is not None
        assert result.beta is not None
        assert result.beta_se > 0
        assert result.n_categories > 0
        assert result.n_months > 0

    def test_admin_price_exclusion(self):
        """Test that admin prices are properly excluded."""
        from studies.fx_passthrough.src.cpi_pass_through import (
            CPIPassThroughModel,
            CPIPassThroughSpec,
        )

        panel = create_mock_cpi_panel()

        # With exclusion
        spec_exclude = CPIPassThroughSpec(
            exclude_admin_prices=True,
            run_small_n_inference=False,
        )
        model = CPIPassThroughModel(panel)
        result_exclude = model.fit(spec_exclude)

        # Without exclusion
        spec_include = CPIPassThroughSpec(
            exclude_admin_prices=False,
            run_small_n_inference=False,
        )
        model2 = CPIPassThroughModel(panel)
        result_include = model2.fit(spec_include)

        # Should have fewer categories when excluding
        assert result_exclude.n_categories < result_include.n_categories

    def test_construct_instrument(self):
        """Test imported inflation instrument construction."""
        from studies.fx_passthrough.src.cpi_pass_through import (
            CPIPassThroughModel,
            CPIPassThroughSpec,
        )

        panel = create_mock_cpi_panel()

        spec = CPIPassThroughSpec(run_small_n_inference=False)
        model = CPIPassThroughModel(panel)
        result = model.fit(spec)

        assert result.imported_inflation is not None
        assert len(result.imported_inflation) > 0


class TestCPIPassThroughResult:
    """Tests for CPIPassThroughResult."""

    def test_result_summary(self):
        """Test result summary generation."""
        from studies.fx_passthrough.src.cpi_pass_through import CPIPassThroughResult

        result = CPIPassThroughResult(
            beta=0.25,
            beta_se=0.05,
            beta_ci=(0.15, 0.35),
            beta_pvalue=0.001,
            n_categories=8,
            n_months=60,
            n_obs=480,
            r2_within=0.15,
        )

        summary = result.summary()

        assert "Block A" in summary
        assert "0.25" in summary
        assert "8" in summary


class TestFalsificationTests:
    """Tests for falsification battery."""

    def test_pre_trends_test(self):
        """Test pre-trends falsification."""
        from studies.fx_passthrough.src.cpi_pass_through import (
            CPIPassThroughModel,
            CPIPassThroughSpec,
        )

        panel = create_mock_cpi_panel()

        spec = CPIPassThroughSpec(run_small_n_inference=False)
        model = CPIPassThroughModel(panel)
        model.fit(spec)

        pre_trends = model.test_pre_trends(spec, n_leads=2)

        assert "lead_coefficients" in pre_trends
        assert "joint_pvalue" in pre_trends
        assert len(pre_trends["lead_coefficients"]) == 2

    def test_admin_prices_test(self):
        """Test admin prices falsification."""
        from studies.fx_passthrough.src.cpi_pass_through import (
            CPIPassThroughModel,
            CPIPassThroughSpec,
        )

        panel = create_mock_cpi_panel()

        spec = CPIPassThroughSpec(run_small_n_inference=False)
        model = CPIPassThroughModel(panel)
        model.fit(spec)

        admin_test = model.test_admin_prices(spec)

        assert "admin_coefficient" in admin_test
        assert "admin_test_pass" in admin_test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
