"""Tests for bank stress simulation."""

import pytest

from studies.kaspi_holdco.src.bank_state import BankState, BankScenario
from studies.kaspi_holdco.src.bank_stress import (
    simulate_bank_stress,
    run_bank_stress_grid,
    K1_2_MINIMUM,
    K2_MINIMUM,
)
from studies.kaspi_holdco.src.stress_scenarios import (
    FY2024_BANK_STATE,
    FY2024_ANNUAL_NET_INCOME,
    BANK_STRESS_SCENARIOS,
)


class TestBankState:
    """Tests for BankState dataclass."""

    def test_k1_2_ratio(self) -> None:
        """Test K1-2 ratio calculation."""
        state = BankState(
            rwa=1000,
            tier1_capital=126,
            total_capital=127,
            retail_deposits=0,
            corporate_deposits=0,
        )
        assert abs(state.k1_2_ratio - 0.126) < 0.001

    def test_k2_ratio(self) -> None:
        """Test K2 ratio calculation."""
        state = BankState(
            rwa=1000,
            tier1_capital=126,
            total_capital=127,
            retail_deposits=0,
            corporate_deposits=0,
        )
        assert abs(state.k2_ratio - 0.127) < 0.001

    def test_k2_headroom(self) -> None:
        """Test K2 headroom calculation."""
        state = BankState(
            rwa=1000,
            tier1_capital=126,
            total_capital=127,
            retail_deposits=0,
            corporate_deposits=0,
        )
        # Headroom = 127 - 0.12 * 1000 = 127 - 120 = 7
        assert abs(state.k2_headroom - 7) < 0.001

    def test_fy2024_baseline_ratios(self) -> None:
        """Test FY2024 baseline ratios match expected values."""
        state = FY2024_BANK_STATE
        # K1-2 should be around 12.6%
        assert abs(state.k1_2_ratio - 0.126) < 0.01
        # K2 should be around 12.7%
        assert abs(state.k2_ratio - 0.127) < 0.01
        # K2 headroom should be around 60 bn
        assert 55 < state.k2_headroom < 65


class TestBankScenario:
    """Tests for BankScenario dataclass."""

    def test_stressed_income(self) -> None:
        """Test stressed income calculation."""
        scenario = BankScenario(
            name="test",
            annual_net_income=100,
            profit_multiplier=0.5,
        )
        assert scenario.stressed_income == 50

    def test_retained_earnings(self) -> None:
        """Test retained earnings calculation."""
        scenario = BankScenario(
            name="test",
            annual_net_income=100,
            profit_multiplier=0.5,
            payout_ratio=0.3,
        )
        # Stressed = 50, retained = 50 * 0.7 = 35
        assert scenario.retained_earnings == 35


class TestSimulateBankStress:
    """Tests for simulate_bank_stress function."""

    def test_baseline_no_stress(self) -> None:
        """Baseline with no stress should maintain ratios."""
        state = FY2024_BANK_STATE
        scenario = BankScenario(
            name="no_stress",
            annual_net_income=FY2024_ANNUAL_NET_INCOME,
            profit_multiplier=1.0,
            credit_loss_rate=0.0,
            rwa_multiplier=1.0,
            retail_run_rate=0.0,
        )

        result = simulate_bank_stress(state, scenario)

        # Should remain above minimum
        assert result.k2_after > K2_MINIMUM
        assert result.capital_shortfall == 0
        assert not result.support_needed

    def test_severe_stress_creates_shortfall(self) -> None:
        """Severe stress should create capital shortfall."""
        state = FY2024_BANK_STATE
        scenario = BankScenario(
            name="severe",
            annual_net_income=FY2024_ANNUAL_NET_INCOME,
            profit_multiplier=0.0,
            credit_loss_rate=0.07,  # 7% credit losses
            rwa_multiplier=1.25,  # +25% RWA
            retail_run_rate=0.30,
        )

        result = simulate_bank_stress(state, scenario)

        # Should be below minimum
        assert result.k2_after < K2_MINIMUM
        assert result.capital_shortfall > 0
        assert result.support_needed

    def test_credit_losses_reduce_capital(self) -> None:
        """Credit losses should reduce total capital."""
        state = FY2024_BANK_STATE
        scenario = BankScenario(
            name="credit_loss",
            annual_net_income=FY2024_ANNUAL_NET_INCOME,
            profit_multiplier=1.0,
            credit_loss_rate=0.05,  # 5% of gross loans
            rwa_multiplier=1.0,
        )

        result = simulate_bank_stress(state, scenario)

        # Credit losses should be ~5% of gross loans
        expected_loss = state.gross_loans * 0.05
        assert abs(result.credit_losses - expected_loss) < 1.0

    def test_rwa_inflation_effect(self) -> None:
        """RWA inflation should inflate RWA and affect ratio calculation."""
        state = FY2024_BANK_STATE
        scenario = BankScenario(
            name="rwa_inflation",
            annual_net_income=FY2024_ANNUAL_NET_INCOME,
            profit_multiplier=0.0,  # No profit to isolate RWA effect
            credit_loss_rate=0.0,
            rwa_multiplier=1.20,  # +20% RWA
        )

        result = simulate_bank_stress(state, scenario)

        # RWA should be inflated
        assert result.rwa_after == state.rwa * 1.20
        # With no earnings and higher RWA, K2 should decrease
        assert result.k2_after < result.k2_before

    def test_standard_scenarios(self) -> None:
        """Test all standard scenarios run without error."""
        state = FY2024_BANK_STATE

        for name, scenario in BANK_STRESS_SCENARIOS.items():
            result = simulate_bank_stress(state, scenario)

            # All results should be valid
            assert result.scenario_name == scenario.name
            assert result.k2_after >= 0
            assert result.capital_shortfall >= 0


class TestRunBankStressGrid:
    """Tests for run_bank_stress_grid function."""

    def test_grid_returns_dataframe(self) -> None:
        """Grid should return a DataFrame with expected columns."""
        df = run_bank_stress_grid(
            FY2024_BANK_STATE,
            FY2024_ANNUAL_NET_INCOME,
            profit_multipliers=[1.0, 0.5],
            credit_loss_rates=[0.01, 0.03],
            rwa_multipliers=[1.0, 1.10],
            retail_run_rates=[0.05, 0.15],
        )

        # Should have 2 * 2 * 2 * 2 = 16 rows
        assert len(df) == 16

        # Required columns
        required_cols = [
            "scenario",
            "k2_after",
            "capital_shortfall",
            "support_needed",
        ]
        for col in required_cols:
            assert col in df.columns

    def test_grid_identifies_threshold(self) -> None:
        """Grid should identify scenarios above/below threshold."""
        df = run_bank_stress_grid(
            FY2024_BANK_STATE,
            FY2024_ANNUAL_NET_INCOME,
            profit_multipliers=[0.0],  # No profit to make stress binding
            credit_loss_rates=[0.01, 0.07],  # 7% is severe
            rwa_multipliers=[1.0, 1.25],  # 25% RWA inflation
            retail_run_rates=[0.05],
        )

        # Should have some viable and some not
        assert df["support_needed"].sum() > 0
        assert (~df["support_needed"]).sum() > 0
