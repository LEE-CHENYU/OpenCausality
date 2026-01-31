"""
Tests for policy confound checks.
"""

import pytest
from datetime import date

from studies.credit_default.src.confound_checks import (
    ConfoundChecker,
    ConfoundCheckResult,
    check_confounds,
)


class TestConfoundChecker:
    """Test confound checking functionality."""

    @pytest.fixture
    def checker(self):
        return ConfoundChecker()

    def test_origination_window_pass(self, checker):
        """Test origination window passes when before DSTI."""
        result = checker.check_origination_window(date(2023, 11, 30))
        assert result.passed
        assert len(result.confounds_found) == 0

    def test_origination_window_fail(self, checker):
        """Test origination window fails when after DSTI."""
        result = checker.check_origination_window(date(2024, 1, 15))
        assert not result.passed
        assert len(result.confounds_found) > 0

    def test_outcome_window_pass(self, checker):
        """Test outcome window passes when before DTI."""
        result = checker.check_outcome_window(
            date(2024, 1, 1),
            date(2024, 5, 31),
        )
        assert result.passed
        assert len(result.confounds_found) == 0

    def test_outcome_window_fail(self, checker):
        """Test outcome window fails when after DTI."""
        result = checker.check_outcome_window(
            date(2024, 1, 1),
            date(2024, 7, 31),
        )
        assert not result.passed
        assert len(result.confounds_found) > 0

    def test_treatment_timing_pass(self, checker):
        """Test treatment timing passes for Jan 2024."""
        result = checker.check_treatment_timing(date(2024, 1, 1))
        assert result.passed

    def test_treatment_timing_fail(self, checker):
        """Test treatment timing fails for wrong date."""
        result = checker.check_treatment_timing(date(2023, 6, 1))
        assert not result.passed

    def test_run_all_checks_pass(self, checker):
        """Test all checks pass with correct dates."""
        results = checker.run_all_checks(
            origination_cutoff=date(2023, 11, 30),
            outcome_start=date(2024, 1, 1),
            outcome_end=date(2024, 5, 31),
            treatment_date=date(2024, 1, 1),
        )
        assert all(r.passed for r in results.values())

    def test_validate_or_fail_success(self, checker):
        """Test validate_or_fail passes with correct dates."""
        result = checker.validate_or_fail(
            origination_cutoff=date(2023, 11, 30),
            outcome_start=date(2024, 1, 1),
            outcome_end=date(2024, 5, 31),
            treatment_date=date(2024, 1, 1),
        )
        assert result is True

    def test_validate_or_fail_raises(self, checker):
        """Test validate_or_fail raises with incorrect dates."""
        with pytest.raises(ValueError):
            checker.validate_or_fail(
                origination_cutoff=date(2024, 2, 1),
                outcome_start=date(2024, 1, 1),
                outcome_end=date(2024, 7, 31),
                treatment_date=date(2024, 1, 1),
            )


class TestConvenienceFunction:
    """Test convenience function."""

    def test_check_confounds_default(self):
        """Test check_confounds with defaults."""
        results = check_confounds()
        assert "origination_window" in results
        assert "outcome_window" in results
        assert "treatment_timing" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
