"""
Tests for difference-in-discontinuities estimator.
"""

import pytest
import pandas as pd
import numpy as np

from studies.credit_default.src.diff_in_discs import (
    DiffInDiscsEstimator,
    DiffInDiscsResult,
    estimate_mw_effect,
)


class TestDiffInDiscsEstimator:
    """Test diff-in-discs estimation."""

    @pytest.fixture
    def mock_panel(self):
        """Create mock panel data for testing."""
        np.random.seed(42)

        n_borrowers = 1000
        n_periods = 6  # 3 pre + 3 post

        rows = []
        for i in range(n_borrowers):
            # Assign pre-policy payroll near cutoff
            pre_payroll = np.random.uniform(55000, 100000)
            below_cutoff = pre_payroll < 70000

            for period in range(n_periods):
                post = period >= 3

                # Treatment effect for below-cutoff borrowers in post period
                treatment_effect = -0.02 if (below_cutoff and post) else 0

                # Base default probability
                base_prob = 0.05 + np.random.randn() * 0.01

                dpd30 = np.random.binomial(1, max(0, min(1, base_prob + treatment_effect)))

                rows.append({
                    "borrower_id": f"B{i:04d}",
                    "loan_id": f"L{i:04d}",
                    "period": period,
                    "post": int(post),
                    "pre_policy_payroll": pre_payroll,
                    "dpd30": dpd30,
                    "loan_amount": 500000 + np.random.randn() * 100000,
                })

        return pd.DataFrame(rows)

    def test_estimator_initialization(self):
        """Test estimator initializes correctly."""
        estimator = DiffInDiscsEstimator()
        assert estimator.cutoff == 70000
        assert estimator.new_mw == 85000

    def test_estimator_custom_cutoff(self):
        """Test estimator with custom cutoffs."""
        estimator = DiffInDiscsEstimator(
            old_minimum_wage=60000,
            new_minimum_wage=80000,
        )
        assert estimator.cutoff == 60000
        assert estimator.new_mw == 80000

    def test_estimate_returns_result(self, mock_panel):
        """Test estimation returns DiffInDiscsResult."""
        estimator = DiffInDiscsEstimator()
        result = estimator.estimate(mock_panel, outcome="dpd30")

        assert isinstance(result, DiffInDiscsResult)
        assert result.n_obs > 0
        assert result.outcome == "dpd30"

    def test_estimate_coefficient_reasonable(self, mock_panel):
        """Test estimated coefficient is in reasonable range."""
        estimator = DiffInDiscsEstimator()
        result = estimator.estimate(mock_panel, outcome="dpd30")

        # Coefficient should be relatively small (effect on default probability)
        assert abs(result.coefficient) < 0.5

    def test_estimate_has_confidence_interval(self, mock_panel):
        """Test result has valid confidence interval."""
        estimator = DiffInDiscsEstimator()
        result = estimator.estimate(mock_panel, outcome="dpd30")

        assert result.conf_int[0] < result.conf_int[1]
        assert result.conf_int[0] <= result.coefficient <= result.conf_int[1]

    def test_estimate_with_custom_bandwidth(self, mock_panel):
        """Test estimation with custom bandwidth."""
        estimator = DiffInDiscsEstimator()
        result = estimator.estimate(
            mock_panel,
            outcome="dpd30",
            bandwidth=15000,
        )

        assert result.bandwidth == 15000

    def test_summary_generation(self, mock_panel):
        """Test summary string generation."""
        estimator = DiffInDiscsEstimator()
        result = estimator.estimate(mock_panel, outcome="dpd30")

        summary = estimator.summary(result)
        assert "DIFF-IN-DISCONTINUITIES" in summary
        assert "dpd30" in summary
        assert "Coefficient" in summary


class TestConvenienceFunction:
    """Test convenience function."""

    @pytest.fixture
    def mock_panel(self):
        """Minimal mock panel."""
        np.random.seed(42)
        n = 500

        return pd.DataFrame({
            "borrower_id": [f"B{i}" for i in range(n)],
            "loan_id": [f"L{i}" for i in range(n)],
            "post": np.random.binomial(1, 0.5, n),
            "pre_policy_payroll": np.random.uniform(50000, 100000, n),
            "dpd30": np.random.binomial(1, 0.05, n),
        })

    def test_estimate_mw_effect(self, mock_panel):
        """Test convenience function."""
        result = estimate_mw_effect(mock_panel, outcome="dpd30")
        assert isinstance(result, DiffInDiscsResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
