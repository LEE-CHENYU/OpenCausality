"""
Report Consistency Checker.

Validates that markdown estimation reports match the underlying EdgeCard data.
This ensures that report tables, statistics, and text accurately reflect
the actual estimation results.

Key checks:
1. Point estimates in tables match EdgeCard.estimates.point
2. Standard errors match EdgeCard.estimates.se
3. N (sample size) values match EdgeCard sample size fields
4. Confidence intervals match EdgeCard.estimates.ci_95
5. Unit normalization table covers all edges
6. Reaction function warnings present for RF edges
7. Credibility ratings match EdgeCard.credibility_rating
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.agentic.output.edge_card import EdgeCard


@dataclass
class MatchResult:
    """Result of matching a single value."""
    matched: bool
    expected: Any
    found: Any | None
    tolerance: float = 0.01

    def __bool__(self) -> bool:
        return self.matched


@dataclass
class EdgeCheckResult:
    """Result of checking a single edge against the report."""
    edge_id: str
    point_estimate: MatchResult | None = None
    standard_error: MatchResult | None = None
    sample_size: MatchResult | None = None
    ci_lower: MatchResult | None = None
    ci_upper: MatchResult | None = None
    credibility_rating: MatchResult | None = None
    unit_documented: bool = False
    reaction_warning_present: bool | None = None  # None if not a reaction function

    def all_matched(self) -> bool:
        """Check if all non-None checks matched."""
        checks = [
            self.point_estimate,
            self.standard_error,
            self.sample_size,
            self.ci_lower,
            self.ci_upper,
            self.credibility_rating,
        ]
        return all(c.matched for c in checks if c is not None)


@dataclass
class ReportCheckResult:
    """Overall result of report consistency check."""
    passed: bool
    edge_results: dict[str, EdgeCheckResult] = field(default_factory=dict)
    missing_edges: list[str] = field(default_factory=list)
    extra_edges: list[str] = field(default_factory=list)
    unit_table_present: bool = False
    unit_table_complete: bool = False
    missing_from_unit_table: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown summary of check results."""
        lines = ["# Report Consistency Check Results", ""]

        status = "PASSED" if self.passed else "FAILED"
        lines.append(f"**Status:** {status}")
        lines.append(f"**Edges Checked:** {len(self.edge_results)}")
        lines.append(f"**Errors:** {len(self.errors)}")
        lines.append(f"**Warnings:** {len(self.warnings)}")
        lines.append("")

        if self.errors:
            lines.append("## Errors")
            for err in self.errors:
                lines.append(f"- {err}")
            lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            for warn in self.warnings:
                lines.append(f"- {warn}")
            lines.append("")

        if self.missing_edges:
            lines.append("## Missing Edges (in cards but not in report)")
            for edge in self.missing_edges:
                lines.append(f"- `{edge}`")
            lines.append("")

        # Summary table
        lines.append("## Edge Match Summary")
        lines.append("")
        lines.append("| Edge | Point | SE | N | CI | Rating | Units | RF Warning |")
        lines.append("|------|-------|----|----|----|----|-------|------------|")

        for edge_id, result in sorted(self.edge_results.items()):
            point = "✓" if result.point_estimate and result.point_estimate.matched else "✗"
            se = "✓" if result.standard_error and result.standard_error.matched else "✗"
            n = "✓" if result.sample_size and result.sample_size.matched else ("—" if result.sample_size is None else "✗")
            ci = "✓" if (result.ci_lower and result.ci_lower.matched and result.ci_upper and result.ci_upper.matched) else "✗"
            rating = "✓" if result.credibility_rating and result.credibility_rating.matched else "✗"
            units = "✓" if result.unit_documented else "✗"
            rf = "—"
            if result.reaction_warning_present is not None:
                rf = "✓" if result.reaction_warning_present else "✗"

            lines.append(f"| `{edge_id}` | {point} | {se} | {n} | {ci} | {rating} | {units} | {rf} |")

        lines.append("")
        return "\n".join(lines)


class ReportConsistencyChecker:
    """
    Checks that a markdown report matches EdgeCard data.

    Usage:
        checker = ReportConsistencyChecker(report_content, edge_cards)
        result = checker.check()
        print(result.to_markdown())
    """

    def __init__(
        self,
        report_content: str,
        edge_cards: dict[str, EdgeCard],
        reaction_function_edges: list[str] | None = None,
        tolerance: float = 0.01,
    ):
        """
        Initialize checker.

        Args:
            report_content: Full markdown content of the report
            edge_cards: Dict of edge_id -> EdgeCard
            reaction_function_edges: List of edge IDs that are reaction functions
            tolerance: Numeric tolerance for matching (relative, e.g., 0.01 = 1%)
        """
        self.report = report_content
        self.cards = edge_cards
        self.rf_edges = set(reaction_function_edges or [])
        self.tolerance = tolerance

    def check(self) -> ReportCheckResult:
        """Run all consistency checks."""
        result = ReportCheckResult(passed=True)

        # Find edges mentioned in report
        report_edges = self._find_edges_in_report()

        # Check which cards are missing from report
        for edge_id in self.cards:
            if edge_id not in report_edges:
                result.missing_edges.append(edge_id)
                result.warnings.append(f"Edge `{edge_id}` has EdgeCard but not in report")

        # Check each edge
        for edge_id, card in self.cards.items():
            if edge_id in report_edges:
                edge_result = self._check_edge(edge_id, card)
                result.edge_results[edge_id] = edge_result

                # Report mismatches
                if edge_result.point_estimate and not edge_result.point_estimate.matched:
                    result.errors.append(
                        f"`{edge_id}`: Point estimate mismatch. "
                        f"Card={edge_result.point_estimate.expected}, "
                        f"Report={edge_result.point_estimate.found}"
                    )
                    result.passed = False

                if edge_result.standard_error and not edge_result.standard_error.matched:
                    result.errors.append(
                        f"`{edge_id}`: SE mismatch. "
                        f"Card={edge_result.standard_error.expected}, "
                        f"Report={edge_result.standard_error.found}"
                    )
                    result.passed = False

                if edge_result.credibility_rating and not edge_result.credibility_rating.matched:
                    result.errors.append(
                        f"`{edge_id}`: Rating mismatch. "
                        f"Card={edge_result.credibility_rating.expected}, "
                        f"Report={edge_result.credibility_rating.found}"
                    )
                    result.passed = False

                if not edge_result.unit_documented:
                    result.warnings.append(f"`{edge_id}`: Units not documented in report")

                if edge_result.reaction_warning_present is False:
                    result.warnings.append(
                        f"`{edge_id}`: Reaction function edge missing warning in report"
                    )

        # Check unit table
        result.unit_table_present = "Unit Normalization" in self.report
        if result.unit_table_present:
            result.missing_from_unit_table = self._check_unit_table()
            result.unit_table_complete = len(result.missing_from_unit_table) == 0
            if not result.unit_table_complete:
                for edge in result.missing_from_unit_table:
                    result.warnings.append(f"`{edge}`: Missing from Unit Normalization table")
        else:
            result.warnings.append("Report missing Unit Normalization Reference section")

        return result

    def _find_edges_in_report(self) -> set[str]:
        """Find all edge IDs mentioned in the report."""
        # Pattern: backtick-quoted edge names with _to_
        pattern = re.compile(r"`([a-z][a-z0-9_]*_to_[a-z][a-z0-9_]*)`")
        return set(pattern.findall(self.report))

    def _check_edge(self, edge_id: str, card: EdgeCard) -> EdgeCheckResult:
        """Check a single edge against the report."""
        result = EdgeCheckResult(edge_id=edge_id)

        if card.estimates:
            # Find numbers near this edge in the report
            # Look for table rows containing this edge
            edge_context = self._get_edge_context(edge_id)

            # Point estimate
            if card.estimates.point is not None:
                found = self._find_number_in_context(edge_context, card.estimates.point)
                result.point_estimate = MatchResult(
                    matched=found is not None,
                    expected=card.estimates.point,
                    found=found,
                )

            # Standard error
            if card.estimates.se is not None and card.estimates.se > 0:
                found = self._find_number_in_context(edge_context, card.estimates.se)
                result.standard_error = MatchResult(
                    matched=found is not None,
                    expected=card.estimates.se,
                    found=found,
                )

            # Sample size (N)
            n = card.estimates.n_effective_obs_h0 or card.estimates.n_calendar_periods
            if n is not None:
                found = self._find_integer_in_context(edge_context, n)
                result.sample_size = MatchResult(
                    matched=found is not None,
                    expected=n,
                    found=found,
                )

            # CI
            if card.estimates.ci_95:
                ci_lower, ci_upper = card.estimates.ci_95
                found_lower = self._find_number_in_context(edge_context, ci_lower)
                found_upper = self._find_number_in_context(edge_context, ci_upper)
                result.ci_lower = MatchResult(
                    matched=found_lower is not None,
                    expected=ci_lower,
                    found=found_lower,
                )
                result.ci_upper = MatchResult(
                    matched=found_upper is not None,
                    expected=ci_upper,
                    found=found_upper,
                )

        # Credibility rating
        found_rating = self._find_rating_in_context(self._get_edge_context(edge_id), card.credibility_rating)
        result.credibility_rating = MatchResult(
            matched=found_rating == card.credibility_rating,
            expected=card.credibility_rating,
            found=found_rating,
        )

        # Unit documented
        result.unit_documented = self._is_unit_documented(edge_id)

        # Reaction function warning
        if edge_id in self.rf_edges:
            result.reaction_warning_present = self._has_reaction_warning(edge_id)

        return result

    def _get_edge_context(self, edge_id: str, lines_before: int = 2, lines_after: int = 2) -> str:
        """Get text context around an edge mention."""
        lines = self.report.split("\n")
        context_lines = []

        for i, line in enumerate(lines):
            if edge_id in line:
                start = max(0, i - lines_before)
                end = min(len(lines), i + lines_after + 1)
                context_lines.extend(lines[start:end])

        return "\n".join(context_lines)

    def _find_number_in_context(self, context: str, target: float) -> float | None:
        """Find a number close to target in the context."""
        # Handle special cases
        if target == 0:
            # For zero, look for exact "0" or "0.0" etc
            if re.search(r"\b0(\.0+)?\b", context):
                return 0.0
            return None

        # Find all numbers in context
        numbers = re.findall(r"-?[\d,]+\.?\d*", context)

        for num_str in numbers:
            try:
                num = float(num_str.replace(",", ""))
                # Check if within tolerance
                if abs(target) > 0.0001:
                    rel_diff = abs(num - target) / abs(target)
                    if rel_diff < self.tolerance:
                        return num
                else:
                    if abs(num - target) < 0.0001:
                        return num
            except ValueError:
                continue

        return None

    def _find_integer_in_context(self, context: str, target: int) -> int | None:
        """Find an integer in context."""
        # Find all integers
        numbers = re.findall(r"\b(\d+)\b", context)

        for num_str in numbers:
            try:
                num = int(num_str)
                if num == target:
                    return num
            except ValueError:
                continue

        return None

    def _find_rating_in_context(self, context: str, target: str) -> str | None:
        """Find credibility rating in context."""
        # Look for rating pattern: | A | or | B | etc
        pattern = rf"\|\s*{target}\s*\|"
        if re.search(pattern, context):
            return target

        # Also check for unquoted rating at end of line or before |
        pattern2 = rf"[\s|]{target}[\s|$]"
        if re.search(pattern2, context):
            return target

        return None

    def _is_unit_documented(self, edge_id: str) -> bool:
        """Check if edge has units documented in Unit Normalization table."""
        if "Unit Normalization" not in self.report:
            return False

        # Find the Unit Normalization section
        unit_section_match = re.search(
            r"## Unit Normalization.*?(?=##|\Z)",
            self.report,
            re.DOTALL
        )
        if not unit_section_match:
            return False

        unit_section = unit_section_match.group()
        return edge_id in unit_section

    def _has_reaction_warning(self, edge_id: str) -> bool:
        """Check if reaction function edge has warning in report."""
        context = self._get_edge_context(edge_id, lines_before=10, lines_after=5)

        warning_patterns = [
            r"reaction\s+function",
            r"NOT.*causal",
            r"endogenous.*response",
            r"should\s+NOT\s+be\s+used",
            r"\*\*Warning\*\*",
        ]

        for pattern in warning_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return True

        return False

    def _check_unit_table(self) -> list[str]:
        """Check which edges are missing from unit table."""
        missing = []

        # Find Unit Normalization section
        unit_section_match = re.search(
            r"## Unit Normalization.*?(?=##|\Z)",
            self.report,
            re.DOTALL
        )
        if not unit_section_match:
            return list(self.cards.keys())

        unit_section = unit_section_match.group()

        for edge_id in self.cards:
            if edge_id not in unit_section:
                missing.append(edge_id)

        return missing


def check_report_consistency(
    report_path: str | Path,
    edge_cards: dict[str, EdgeCard],
    reaction_function_edges: list[str] | None = None,
) -> ReportCheckResult:
    """
    Convenience function to check report consistency.

    Args:
        report_path: Path to markdown report file
        edge_cards: Dict of edge_id -> EdgeCard
        reaction_function_edges: List of reaction function edge IDs

    Returns:
        ReportCheckResult with detailed findings
    """
    with open(report_path) as f:
        content = f.read()

    checker = ReportConsistencyChecker(
        content,
        edge_cards,
        reaction_function_edges,
    )
    return checker.check()
