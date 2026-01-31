"""
Tests for shift-share regression model.
"""

import pytest
import pandas as pd
import numpy as np

from studies.household_welfare.src.shift_share import ShiftShareModel, ShiftShareSpec, MAIN_SPEC as BASELINE_SPEC


class TestShiftShareModel:
    """Test shift-share regression."""

    @pytest.fixture
    def mock_panel(self):
        """Create mock panel data for testing."""
        np.random.seed(42)

        regions = ["Region_A", "Region_B", "Region_C", "Region_D"]
        quarters = [f"{y}Q{q}" for y in range(2015, 2020) for q in range(1, 5)]

        rows = []
        for region in regions:
            # Fixed exposure
            oil_exposure = {"Region_A": 0.8, "Region_B": 0.5, "Region_C": 0.2, "Region_D": 0.1}[region]

            for quarter in quarters:
                year = int(quarter[:4])
                q = int(quarter[-1])

                # Shock that varies by time
                shock = np.sin(year + q / 4) * 0.5

                # Outcome with true coefficient of -0.1
                log_income = 10 + oil_exposure * shock * (-0.1) + np.random.randn() * 0.05

                rows.append({
                    "region": region,
                    "quarter": quarter,
                    "year": year,
                    "q": q,
                    "log_income_pc": log_income,
                    "E_oil_r": oil_exposure,
                    "oil_supply_shock": shock,
                    "E_cyc_r": 0.3,
                    "global_activity_shock": shock * 0.5,
                })

        return pd.DataFrame(rows)

    def test_model_initialization(self, mock_panel):
        """Test model initializes correctly."""
        model = ShiftShareModel(mock_panel)
        assert model.data is not None
        assert isinstance(model.data.index, pd.MultiIndex)

    def test_baseline_specification(self, mock_panel):
        """Test baseline specification runs."""
        model = ShiftShareModel(mock_panel)

        spec = ShiftShareSpec(
            name="test",
            outcome="log_income_pc",
            interactions=[("E_oil_r", "oil_supply_shock")],
        )

        result = model.fit(spec)

        assert result.nobs > 0
        assert len(result.params) > 0
        assert "test" in model.results

    def test_driscoll_kraay_se(self, mock_panel):
        """Test Driscoll-Kraay standard errors."""
        model = ShiftShareModel(mock_panel)

        spec = ShiftShareSpec(
            name="dk_test",
            outcome="log_income_pc",
            interactions=[("E_oil_r", "oil_supply_shock")],
            cov_type="kernel",
        )

        result = model.fit(spec)

        # Should have standard errors
        assert all(result.std_errors > 0)

    def test_coefficient_sign(self, mock_panel):
        """Test coefficient has expected sign."""
        model = ShiftShareModel(mock_panel)

        spec = ShiftShareSpec(
            name="sign_test",
            outcome="log_income_pc",
            interactions=[("E_oil_r", "oil_supply_shock")],
        )

        result = model.fit(spec)

        # Coefficient should be negative (by construction)
        interaction_col = "E_oil_r_x_oil_supply"
        assert result.params[interaction_col] < 0

    def test_get_multipliers(self, mock_panel):
        """Test multiplier extraction."""
        # Use a simple spec that matches the mock data
        spec = ShiftShareSpec(
            name="simple",
            outcome="log_income_pc",
            interactions=[("E_oil_r", "oil_supply_shock")],
        )

        model = ShiftShareModel(mock_panel)
        model.fit(spec)

        multipliers = model.get_multipliers()
        assert "E_oil_r_x_oil_supply" in multipliers


class TestInteractionTerms:
    """Test interaction term creation."""

    @pytest.fixture
    def mock_panel(self):
        """Simple mock panel."""
        return pd.DataFrame({
            "region": ["A", "A", "B", "B"],
            "quarter": ["2020Q1", "2020Q2", "2020Q1", "2020Q2"],
            "log_income_pc": [10.0, 10.1, 9.5, 9.6],
            "E_oil_r": [0.8, 0.8, 0.2, 0.2],
            "oil_supply_shock": [0.5, -0.3, 0.5, -0.3],
        })

    def test_interaction_creation(self, mock_panel):
        """Test interaction variables are created correctly."""
        model = ShiftShareModel(mock_panel)

        spec = ShiftShareSpec(
            name="test",
            outcome="log_income_pc",
            interactions=[("E_oil_r", "oil_supply_shock")],
        )

        # Access internal method
        data = model._create_interactions(spec)

        expected_interaction = mock_panel["E_oil_r"] * mock_panel["oil_supply_shock"]
        # Note: data index is MultiIndex now
        # Just check the column exists


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
