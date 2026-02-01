"""Tests for holdco cash flow simulation."""

import pytest

from studies.kaspi_holdco.src.holdco_state import HoldCoState, HoldCoScenario
from studies.kaspi_holdco.src.holdco_simulator import (
    simulate_holdco_12m,
    scenario_grid,
    HoldCoStressTester,
)
from studies.kaspi_holdco.src.bank_integration import (
    PassthroughConfig,
    compute_dividend_payout_fraction,
    compute_capital_call,
    build_holdco_scenario_from_bank_stress,
)
from studies.kaspi_holdco.src.bank_stress import simulate_bank_stress, BankStressResult
from studies.kaspi_holdco.src.stress_scenarios import (
    FY2024_HOLDCO_STATE,
    FY2024_BASELINE_SCENARIO,
    FY2024_BANK_STATE,
    HOLDCO_SCENARIOS,
    BANK_STRESS_SCENARIOS,
)


class TestHoldCoState:
    """Tests for HoldCoState dataclass."""

    def test_annual_fixed_costs(self) -> None:
        """Test annual fixed costs calculation."""
        state = HoldCoState(
            cash=100,
            annual_gna=20,
            annual_interest=5,
            other_fixed=3,
        )
        assert state.annual_fixed_costs == 28

    def test_monthly_fixed_costs(self) -> None:
        """Test monthly fixed costs calculation."""
        state = HoldCoState(cash=100, annual_gna=24)
        assert state.monthly_fixed_costs == 2

    def test_months_of_runway(self) -> None:
        """Test standalone runway calculation."""
        state = HoldCoState(cash=100, annual_gna=12)
        # 100 / (12/12) = 100 months
        assert state.months_of_runway_standalone == 100

    def test_fy2024_baseline(self) -> None:
        """Test FY2024 baseline values."""
        state = FY2024_HOLDCO_STATE
        assert abs(state.cash - 324.993) < 0.01
        assert abs(state.annual_gna - 20.810) < 0.01
        # Runway should be high (low G&A relative to cash)
        assert state.months_of_runway_standalone > 150


class TestHoldCoScenario:
    """Tests for HoldCoScenario dataclass."""

    def test_total_inflows(self) -> None:
        """Test total inflows calculation."""
        scenario = HoldCoScenario(
            name="test",
            dividend_from_bank=100,
            dividend_from_other_subs=200,
            new_equity_raised=50,
            new_debt_raised=25,
        )
        assert scenario.total_inflows == 375

    def test_total_outflows(self) -> None:
        """Test outflow calculations."""
        scenario = HoldCoScenario(
            name="test",
            dividend_from_bank=100,
            dividend_from_other_subs=200,
            dividends_to_shareholders=150,
            buybacks=25,
            capital_injection_to_bank=50,
        )
        assert scenario.total_discretionary_outflows == 175
        assert scenario.total_required_outflows == 50


class TestSimulateHoldco12m:
    """Tests for simulate_holdco_12m function."""

    def test_baseline_positive_ending_cash(self) -> None:
        """Baseline should have positive ending cash."""
        result = simulate_holdco_12m(FY2024_HOLDCO_STATE, FY2024_BASELINE_SCENARIO)

        assert result.ending_cash > 0
        assert not result.went_negative
        assert not result.needs_external_funding

    def test_no_inflows_depletes_cash(self) -> None:
        """No inflows should deplete cash over time."""
        state = HoldCoState(cash=50, annual_gna=24)  # Monthly burn = 2
        scenario = HoldCoScenario(
            name="no_inflows",
            dividend_from_bank=0,
            dividend_from_other_subs=0,
            dividends_to_shareholders=0,
        )

        result = simulate_holdco_12m(state, scenario)

        # 50 - (2 * 12) = 50 - 24 = 26
        assert abs(result.ending_cash - 26) < 0.1

    def test_large_capital_call_goes_negative(self) -> None:
        """Large capital call at start should make cash negative."""
        state = HoldCoState(cash=100, annual_gna=12)
        scenario = HoldCoScenario(
            name="big_call",
            dividend_from_bank=50,
            dividend_from_other_subs=50,
            capital_injection_to_bank=200,  # > starting cash
        )

        result = simulate_holdco_12m(state, scenario)

        assert result.went_negative
        assert result.needs_external_funding
        assert result.min_cash_balance < 0

    def test_monthly_detail(self) -> None:
        """Monthly detail should return DataFrame."""
        result, monthly_df = simulate_holdco_12m(
            FY2024_HOLDCO_STATE,
            FY2024_BASELINE_SCENARIO,
            monthly_detail=True,
        )

        # 13 rows (month 0 through 12)
        assert len(monthly_df) == 13

        # Required columns
        assert "month" in monthly_df.columns
        assert "cash_balance" in monthly_df.columns

    def test_all_standard_scenarios(self) -> None:
        """All standard scenarios should run without error."""
        for name, scenario in HOLDCO_SCENARIOS.items():
            result = simulate_holdco_12m(FY2024_HOLDCO_STATE, scenario)
            assert result.scenario_name == scenario.name


class TestScenarioGrid:
    """Tests for scenario_grid function."""

    def test_grid_returns_dataframe(self) -> None:
        """Grid should return DataFrame with expected dimensions."""
        df = scenario_grid(
            FY2024_HOLDCO_STATE,
            bank_dividends=[0, 100, 200],
            capital_calls=[0, 100],
            shareholder_dividends=[0, 300],
        )

        # 3 * 2 * 2 = 12 rows
        assert len(df) == 12

    def test_grid_identifies_funding_needs(self) -> None:
        """Grid should identify scenarios needing funding."""
        df = scenario_grid(
            FY2024_HOLDCO_STATE,
            bank_dividends=[0, 285],
            capital_calls=[0, 500],
            shareholder_dividends=[0],
        )

        # Large capital call should need funding
        assert df["needs_external_funding"].sum() > 0


class TestPassthroughConfig:
    """Tests for PassthroughConfig and integration functions."""

    def test_dividend_payout_at_baseline(self) -> None:
        """At baseline K2, payout should be 100%."""
        config = PassthroughConfig(k2_baseline=0.127, k2_minimum=0.12)
        payout = compute_dividend_payout_fraction(0.127, config)
        assert abs(payout - 1.0) < 0.01

    def test_dividend_payout_at_minimum(self) -> None:
        """At minimum K2, payout should be 0%."""
        config = PassthroughConfig(k2_baseline=0.127, k2_minimum=0.12)
        payout = compute_dividend_payout_fraction(0.12, config)
        assert abs(payout - 0.0) < 0.01

    def test_dividend_payout_linear(self) -> None:
        """Linear gamma should give linear interpolation."""
        config = PassthroughConfig(k2_baseline=0.127, k2_minimum=0.12, gamma=1.0)
        # Midpoint: k2 = 0.1235 (halfway between 0.12 and 0.127)
        payout = compute_dividend_payout_fraction(0.1235, config)
        assert abs(payout - 0.5) < 0.1

    def test_dividend_payout_convex(self) -> None:
        """Higher gamma should give lower payout (more conservative)."""
        config_linear = PassthroughConfig(gamma=1.0)
        config_convex = PassthroughConfig(gamma=2.0)

        k2_stressed = 0.123  # Below baseline

        payout_linear = compute_dividend_payout_fraction(k2_stressed, config_linear)
        payout_convex = compute_dividend_payout_fraction(k2_stressed, config_convex)

        # Convex should be lower (more conservative)
        assert payout_convex < payout_linear

    def test_capital_call_zero_shortfall(self) -> None:
        """No shortfall should mean no capital call."""
        config = PassthroughConfig(capital_call_multiplier=1.5)
        call = compute_capital_call(0.0, config)
        assert call == 0.0

    def test_capital_call_with_multiplier(self) -> None:
        """Capital call should apply multiplier."""
        config = PassthroughConfig(capital_call_multiplier=1.25)
        call = compute_capital_call(100.0, config)
        assert call == 125.0

    def test_build_holdco_scenario(self) -> None:
        """Build holdco scenario from bank stress result."""
        # Create a mock bank stress result
        bank_result = BankStressResult(
            scenario_name="test",
            k1_2_before=0.126,
            k1_2_after=0.11,
            k2_before=0.127,
            k2_after=0.115,  # Below minimum
            capital_shortfall=50.0,
            liquidity_gap=0.0,
            support_needed=True,
            retained_earnings=100,
            credit_losses=150,
            fire_sale_losses=0,
            mtm_losses=0,
            total_outflows=0,
            repo_used=0,
            securities_sold=0,
            rwa_after=8500,
            total_capital_after=950,
        )

        config = PassthroughConfig(capital_call_multiplier=1.0)
        holdco_sc = build_holdco_scenario_from_bank_stress(
            bank_result, FY2024_BASELINE_SCENARIO, config
        )

        # Dividend should be cut (K2 below baseline)
        assert holdco_sc.dividend_from_bank < FY2024_BASELINE_SCENARIO.dividend_from_bank

        # Capital call should match shortfall
        assert holdco_sc.capital_injection_to_bank == 50.0

        # Shareholder dividend should be cut (capital call needed)
        assert holdco_sc.dividends_to_shareholders == 0.0


class TestHoldCoStressTester:
    """Tests for HoldCoStressTester class."""

    def test_run_named_scenario(self) -> None:
        """Run named scenario should work."""
        tester = HoldCoStressTester(FY2024_HOLDCO_STATE)
        result = tester.run_named_scenario("baseline")
        assert result.scenario_name == "Baseline (FY2024 Actuals)"

    def test_run_all_named_scenarios(self) -> None:
        """Run all scenarios should return dict."""
        tester = HoldCoStressTester(FY2024_HOLDCO_STATE)
        results = tester.run_all_named_scenarios()

        assert len(results) == len(HOLDCO_SCENARIOS)
        for name in HOLDCO_SCENARIOS:
            assert name in results

    def test_find_max_capital_call(self) -> None:
        """Find max capital call should return reasonable value."""
        tester = HoldCoStressTester(FY2024_HOLDCO_STATE)

        # With full dividends and no shareholder payout
        max_call = tester.find_max_capital_call(
            bank_dividend=285,
            shareholder_dividend=0,
        )

        # Should be positive and reasonable
        assert max_call > 0
        assert max_call < 2000  # Sanity check

    def test_capacity_analysis(self) -> None:
        """Capacity analysis should return dict with expected keys."""
        tester = HoldCoStressTester(FY2024_HOLDCO_STATE)
        capacity = tester.capacity_analysis()

        expected_keys = [
            "full_dividends",
            "no_shareholder_dividend",
            "no_bank_dividend",
            "half_bank_dividend",
        ]
        for key in expected_keys:
            assert key in capacity
            assert capacity[key] >= 0


class TestIntegration:
    """Integration tests for full bank → holdco passthrough."""

    def test_full_passthrough_baseline(self) -> None:
        """Full passthrough with baseline bank scenario."""
        # Bank stress
        bank_sc = BANK_STRESS_SCENARIOS["baseline"]
        bank_result = simulate_bank_stress(FY2024_BANK_STATE, bank_sc)

        # Translate to holdco
        config = PassthroughConfig()
        holdco_sc = build_holdco_scenario_from_bank_stress(
            bank_result, FY2024_BASELINE_SCENARIO, config
        )

        # Run holdco simulation
        holdco_result = simulate_holdco_12m(FY2024_HOLDCO_STATE, holdco_sc)

        # Baseline should be viable
        assert not holdco_result.needs_external_funding

    def test_full_passthrough_severe(self) -> None:
        """Full passthrough with severe bank scenario."""
        # Bank stress
        bank_sc = BANK_STRESS_SCENARIOS["severe"]
        bank_result = simulate_bank_stress(FY2024_BANK_STATE, bank_sc)

        # Translate to holdco
        config = PassthroughConfig()
        holdco_sc = build_holdco_scenario_from_bank_stress(
            bank_result, FY2024_BASELINE_SCENARIO, config
        )

        # Run holdco simulation
        holdco_result = simulate_holdco_12m(FY2024_HOLDCO_STATE, holdco_sc)

        # Results should be valid (may or may not need funding)
        assert holdco_result.ending_cash is not None
