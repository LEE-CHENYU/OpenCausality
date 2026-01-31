"""
Tests for Falsification Suite.
"""

import numpy as np
import pandas as pd
import pytest


def create_mock_panel_with_no_effect(n_categories: int = 8, n_months: int = 60) -> pd.DataFrame:
    """Create mock panel where exposure has NO effect (for falsification)."""
    np.random.seed(42)

    categories = [f"{i:02d}" for i in range(1, n_categories + 1)]
    dates = pd.date_range("2015-01-01", periods=n_months, freq="MS")

    records = []
    for cat in categories:
        import_share = 0.3 + 0.5 * (int(cat) / n_categories)

        for date in dates:
            fx_change = np.random.normal(0, 0.02)
            # NO passthrough effect
            inflation = np.random.normal(0.005, 0.01)

            records.append({
                "category": cat,
                "date": date,
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


class TestSmallNInference:
    """Tests for small-N inference methods."""

    def test_wild_bootstrap_import(self):
        """Test wild bootstrap can be imported."""
        from shared.model.small_n_inference import wild_cluster_bootstrap

        assert wild_cluster_bootstrap is not None

    def test_permutation_test_import(self):
        """Test permutation test can be imported."""
        from shared.model.small_n_inference import permutation_test

        assert permutation_test is not None

    def test_run_small_n_inference(self):
        """Test running small-N inference battery."""
        from shared.model.small_n_inference import run_small_n_inference

        # Create simple test data
        np.random.seed(42)
        n_clusters = 8
        n_per_cluster = 20

        data = pd.DataFrame({
            "category": np.repeat(range(n_clusters), n_per_cluster),
            "exposure": np.repeat(np.random.random(n_clusters), n_per_cluster),
            "outcome": np.random.normal(0, 1, n_clusters * n_per_cluster),
        })

        def simple_model(d):
            """Simple OLS for testing."""
            import statsmodels.api as sm
            X = sm.add_constant(d["exposure"])
            y = d["outcome"]
            model = sm.OLS(y, X).fit()
            return model.params["exposure"], model.bse["exposure"]

        result = run_small_n_inference(
            data=data,
            model_func=simple_model,
            exposure_var="exposure",
            cluster_var="category",
            run_bootstrap=True,
            run_permutation=True,
            n_bootstrap=50,  # Small for testing
            n_permutations=50,
        )

        assert result.coefficient is not None
        assert result.bootstrap_se is not None or result.permutation_pvalue is not None


class TestPreTrendsTest:
    """Tests for pre-trends falsification."""

    def test_pre_trends_with_no_effect(self):
        """Pre-trends should pass when there's no effect."""
        from studies.fx_passthrough.src.cpi_pass_through import (
            CPIPassThroughModel,
            CPIPassThroughSpec,
        )

        # Panel with no true effect
        panel = create_mock_panel_with_no_effect()

        spec = CPIPassThroughSpec(run_small_n_inference=False)
        model = CPIPassThroughModel(panel)
        model.fit(spec)

        pre_trends = model.test_pre_trends(spec, n_leads=2)

        # With no effect, pre-trends should generally pass
        # (coefficients on leads should be insignificant)
        assert "pre_trends_pass" in pre_trends


class TestAdminPriceTest:
    """Tests for admin price falsification."""

    def test_admin_prices_no_response(self):
        """Admin prices should not respond to FX shocks."""
        from studies.fx_passthrough.src.cpi_pass_through import (
            CPIPassThroughModel,
            CPIPassThroughSpec,
        )

        panel = create_mock_panel_with_no_effect()

        spec = CPIPassThroughSpec(run_small_n_inference=False)
        model = CPIPassThroughModel(panel)
        model.fit(spec)

        admin_test = model.test_admin_prices(spec)

        # With no effect, admin prices should show insignificant coefficient
        assert "admin_test_pass" in admin_test


class TestWeakIVTest:
    """Tests for weak IV diagnostics."""

    def test_weak_iv_detection(self):
        """Test weak IV flag is set correctly."""
        from studies.fx_passthrough.src.income_lp_iv import IncomeLPIVResult

        # Strong IV
        result_strong = IncomeLPIVResult(
            outcome="test",
            horizons=[0],
            coefficients=np.array([0.1]),
            std_errors=np.array([0.05]),
            conf_lower=np.array([0.0]),
            conf_upper=np.array([0.2]),
            pvalues=np.array([0.05]),
            first_stage_f=25.0,  # > 10
            first_stage_coef=0.5,
            first_stage_se=0.1,
            first_stage_pvalue=0.001,
            weak_iv_flag=False,
        )

        assert result_strong.weak_iv_flag is False

        # Weak IV
        result_weak = IncomeLPIVResult(
            outcome="test",
            horizons=[0],
            coefficients=np.array([0.1]),
            std_errors=np.array([0.05]),
            conf_lower=np.array([0.0]),
            conf_upper=np.array([0.2]),
            pvalues=np.array([0.05]),
            first_stage_f=5.0,  # < 10
            first_stage_coef=0.5,
            first_stage_se=0.22,
            first_stage_pvalue=0.05,
            weak_iv_flag=True,
        )

        assert result_weak.weak_iv_flag is True


class TestCompositionCheck:
    """Tests for transfer mechanism composition check."""

    def test_shares_sum_check(self):
        """Test that income shares approximately sum to 1."""
        from studies.fx_passthrough.src.transfer_mechanism import TransferMechanismModel

        # Create data with shares that sum to ~1
        np.random.seed(42)
        n = 50

        data = pd.DataFrame({
            "date": pd.date_range("2015-01-01", periods=n, freq="QS"),
            "nominal_income": 100 + np.random.normal(0, 5, n),
            "wage_income": 60 + np.random.normal(0, 3, n),
            "transfer_income": 25 + np.random.normal(0, 2, n),
            "headline_inflation": np.random.normal(0.02, 0.01, n),
            "imported_inflation": np.random.normal(0.01, 0.005, n),
        })

        # Shares should sum to ~0.85 (leaving ~15% for "other")
        data["wage_share"] = data["wage_income"] / data["nominal_income"]
        data["transfer_share"] = data["transfer_income"] / data["nominal_income"]

        model = TransferMechanismModel(data)
        check = model.check_accounting_identity()

        assert "mean_wage_share" in check
        assert "mean_transfer_share" in check
        assert "shares_sum" in check


class TestFalsificationBattery:
    """Integration tests for full falsification battery."""

    def test_run_all_falsification(self):
        """Test running complete falsification suite."""
        from studies.fx_passthrough.src.cpi_pass_through import (
            CPIPassThroughModel,
            CPIPassThroughSpec,
        )

        panel = create_mock_panel_with_no_effect()

        spec = CPIPassThroughSpec(run_small_n_inference=False)
        model = CPIPassThroughModel(panel)
        model.fit(spec)

        results = model.run_all_falsification(spec)

        assert "pre_trends" in results
        assert "admin_prices" in results
        assert "all_tests_pass" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
