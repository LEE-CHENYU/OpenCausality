"""
Policy confound validation for credit default study.

CRITICAL: Must check for policy confounds before estimation.

Key confounds:
- Dec 2023: DSTI limit tightening
- Jun 2024: DTI limit introduction
- Mar 2023: Personal bankruptcy law
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from shared.data.policy_events import (
    PolicyCalendar,
    PolicyType,
    get_kazakhstan_policy_calendar,
)

logger = logging.getLogger(__name__)


@dataclass
class ConfoundCheckResult:
    """Result from confound check."""

    check_name: str
    passed: bool
    description: str
    confounds_found: list[str]
    recommendations: list[str]


class ConfoundChecker:
    """
    Validates that analysis window avoids policy confounds.

    CRITICAL: The credit default study is vulnerable to:
    1. DSTI tightening (Dec 2023) - affects low-income refinancing
    2. DTI introduction (Jun 2024) - major credit constraint
    3. Personal bankruptcy law (Mar 2023) - changes collections dynamics
    """

    def __init__(self):
        """Initialize with Kazakhstan policy calendar."""
        self.calendar = get_kazakhstan_policy_calendar()

    def check_origination_window(
        self,
        origination_cutoff: date,
    ) -> ConfoundCheckResult:
        """
        Check that origination window avoids DSTI confound.

        Loans originated after Dec 2023 may be affected by DSTI tightening,
        which could select for different borrower types.

        Args:
            origination_cutoff: Latest origination date in sample

        Returns:
            ConfoundCheckResult
        """
        confounds = []
        recommendations = []

        # DSTI tightening was Dec 2023
        dsti_date = date(2023, 12, 1)

        if origination_cutoff >= dsti_date:
            confounds.append(
                f"Origination cutoff ({origination_cutoff}) includes post-DSTI period"
            )
            recommendations.append(
                "Restrict to loans originated before December 2023"
            )

        passed = len(confounds) == 0

        return ConfoundCheckResult(
            check_name="origination_window",
            passed=passed,
            description=(
                "Check that loan originations avoid DSTI tightening (Dec 2023)"
            ),
            confounds_found=confounds,
            recommendations=recommendations,
        )

    def check_outcome_window(
        self,
        outcome_start: date,
        outcome_end: date,
    ) -> ConfoundCheckResult:
        """
        Check that outcome window avoids DTI confound.

        Outcomes after June 2024 may be affected by DTI introduction,
        which is a major credit constraint affecting default dynamics.

        Args:
            outcome_start: Start of outcome observation
            outcome_end: End of outcome observation

        Returns:
            ConfoundCheckResult
        """
        confounds = []
        recommendations = []

        # DTI introduction was June 2024
        dti_date = date(2024, 6, 1)

        if outcome_end >= dti_date:
            confounds.append(
                f"Outcome window ({outcome_start} to {outcome_end}) "
                "includes post-DTI period"
            )
            recommendations.append(
                "Truncate outcome window at May 2024"
            )

        # Personal bankruptcy (Mar 2023) is in window but may be controlled for
        bankruptcy_date = date(2023, 3, 1)
        if outcome_start <= bankruptcy_date <= outcome_end:
            # This is less critical but should be noted
            recommendations.append(
                "Consider adding bankruptcy law indicator as control "
                "(March 2023)"
            )

        passed = len(confounds) == 0

        return ConfoundCheckResult(
            check_name="outcome_window",
            passed=passed,
            description=(
                "Check that outcome window avoids DTI introduction (Jun 2024)"
            ),
            confounds_found=confounds,
            recommendations=recommendations,
        )

    def check_treatment_timing(
        self,
        treatment_date: date,
    ) -> ConfoundCheckResult:
        """
        Check that treatment timing is clean.

        For MW design: treatment should be Jan 2024
        Must be after old MW period but before DTI confound.

        Args:
            treatment_date: Date of treatment (MW increase)

        Returns:
            ConfoundCheckResult
        """
        confounds = []
        recommendations = []

        # MW increase was Jan 1, 2024
        expected_mw_date = date(2024, 1, 1)

        if treatment_date != expected_mw_date:
            confounds.append(
                f"Treatment date ({treatment_date}) does not match "
                f"MW increase ({expected_mw_date})"
            )

        # Check that treatment is after any recent confounds
        dsti_date = date(2023, 12, 1)
        if treatment_date.month == dsti_date.month and treatment_date.year == dsti_date.year:
            confounds.append(
                "Treatment month overlaps with DSTI tightening month"
            )
            recommendations.append(
                "Ensure treatment is MW shock, not DSTI shock"
            )

        passed = len(confounds) == 0

        return ConfoundCheckResult(
            check_name="treatment_timing",
            passed=passed,
            description=(
                "Check that treatment timing matches MW increase (Jan 2024)"
            ),
            confounds_found=confounds,
            recommendations=recommendations,
        )

    def run_all_checks(
        self,
        origination_cutoff: date,
        outcome_start: date,
        outcome_end: date,
        treatment_date: date,
    ) -> dict[str, ConfoundCheckResult]:
        """
        Run all confound checks.

        Args:
            origination_cutoff: Latest origination date
            outcome_start: Start of outcome window
            outcome_end: End of outcome window
            treatment_date: Treatment date

        Returns:
            Dictionary of check name to result
        """
        results = {}

        results["origination_window"] = self.check_origination_window(
            origination_cutoff
        )
        results["outcome_window"] = self.check_outcome_window(
            outcome_start, outcome_end
        )
        results["treatment_timing"] = self.check_treatment_timing(
            treatment_date
        )

        return results

    def validate_or_fail(
        self,
        origination_cutoff: date,
        outcome_start: date,
        outcome_end: date,
        treatment_date: date,
    ) -> bool:
        """
        Run all checks and raise error if any fail.

        Args:
            origination_cutoff: Latest origination date
            outcome_start: Start of outcome window
            outcome_end: End of outcome window
            treatment_date: Treatment date

        Returns:
            True if all checks pass

        Raises:
            ValueError if any check fails
        """
        results = self.run_all_checks(
            origination_cutoff, outcome_start, outcome_end, treatment_date
        )

        failures = [r for r in results.values() if not r.passed]

        if failures:
            error_msg = "POLICY CONFOUND CHECK FAILED:\n"
            for f in failures:
                error_msg += f"\n{f.check_name}:\n"
                for c in f.confounds_found:
                    error_msg += f"  - {c}\n"
                for r in f.recommendations:
                    error_msg += f"  Recommendation: {r}\n"

            raise ValueError(error_msg)

        logger.info("All policy confound checks passed")
        return True

    def summary(
        self,
        results: dict[str, ConfoundCheckResult],
    ) -> str:
        """Generate summary of confound checks."""
        lines = []
        lines.append("=" * 60)
        lines.append("POLICY CONFOUND CHECK SUMMARY")
        lines.append("=" * 60)

        all_passed = all(r.passed for r in results.values())

        for name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"\n{name}: {status}")
            lines.append(f"  {result.description}")

            if result.confounds_found:
                lines.append("  Confounds found:")
                for c in result.confounds_found:
                    lines.append(f"    - {c}")

            if result.recommendations:
                lines.append("  Recommendations:")
                for r in result.recommendations:
                    lines.append(f"    - {r}")

        lines.append("\n" + "=" * 60)
        overall = "ALL CHECKS PASS" if all_passed else "SOME CHECKS FAIL"
        lines.append(f"Overall: {overall}")

        return "\n".join(lines)


def check_confounds(
    origination_cutoff: date = date(2023, 12, 1),
    outcome_start: date = date(2024, 1, 1),
    outcome_end: date = date(2024, 5, 31),
    treatment_date: date = date(2024, 1, 1),
) -> dict[str, ConfoundCheckResult]:
    """
    Convenience function to run confound checks with defaults.

    Default windows are:
    - Origination: before Dec 2023 (avoids DSTI)
    - Outcome: Jan-May 2024 (avoids DTI)
    - Treatment: Jan 2024 (MW increase)

    Args:
        origination_cutoff: Latest origination date
        outcome_start: Start of outcome window
        outcome_end: End of outcome window
        treatment_date: Treatment date

    Returns:
        Dictionary of check results
    """
    checker = ConfoundChecker()
    return checker.run_all_checks(
        origination_cutoff, outcome_start, outcome_end, treatment_date
    )
