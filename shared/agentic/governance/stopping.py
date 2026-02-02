"""
Stopping Criteria.

Defines when to stop iterating to prevent unbounded p-hacking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StoppingDecision:
    """Decision on whether to stop."""

    should_stop: bool
    reason: str
    continue_allowed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_stop": self.should_stop,
            "reason": self.reason,
            "continue_allowed": self.continue_allowed,
        }


@dataclass
class StoppingCriteria:
    """
    Criteria for stopping iteration.

    Prevents unbounded specification search (p-hacking).
    """

    # Iteration limits
    max_iterations: int = 3

    # Improvement thresholds
    improvement_threshold: float = 0.05  # Stop if score improves by < 5%

    # Critical edges threshold
    all_critical_edges_threshold: float = 0.60  # Min credibility for target path

    # NULL ACCEPTANCE (critical for avoiding p-hacking)
    null_acceptance_enabled: bool = True
    equivalence_bound: float = 0.1  # "precisely null" if |β| < bound and SE small

    def should_stop(
        self,
        iteration: int,
        credibility_delta: float,
        critical_edges_scores: list[float],
        mode: str = "EXPLORATION",
    ) -> StoppingDecision:
        """
        Determine whether to stop iteration.

        Args:
            iteration: Current iteration number
            credibility_delta: Change in mean credibility from last iteration
            critical_edges_scores: Credibility scores for critical path edges
            mode: "EXPLORATION" or "CONFIRMATION"

        Returns:
            StoppingDecision with reasoning
        """
        # CONFIRMATION mode never iterates
        if mode == "CONFIRMATION":
            return StoppingDecision(
                should_stop=True,
                reason="CONFIRMATION mode: no iteration allowed",
                continue_allowed=False,
            )

        # Max iterations reached
        if iteration >= self.max_iterations:
            return StoppingDecision(
                should_stop=True,
                reason=f"Maximum iterations ({self.max_iterations}) reached",
                continue_allowed=False,
            )

        # All critical edges meet threshold
        if critical_edges_scores:
            min_score = min(critical_edges_scores)
            if min_score >= self.all_critical_edges_threshold:
                return StoppingDecision(
                    should_stop=True,
                    reason=f"All critical edges meet threshold ({min_score:.2f} >= {self.all_critical_edges_threshold})",
                    continue_allowed=True,
                )

        # No meaningful improvement
        if credibility_delta < self.improvement_threshold and iteration > 0:
            return StoppingDecision(
                should_stop=True,
                reason=f"No meaningful improvement ({credibility_delta:.3f} < {self.improvement_threshold})",
                continue_allowed=True,
            )

        # Continue
        return StoppingDecision(
            should_stop=False,
            reason="Continue: improvement possible",
            continue_allowed=True,
        )

    def check_null_acceptance(
        self,
        estimate: float,
        se: float,
    ) -> tuple[bool, str]:
        """
        Check if an estimate should be accepted as "precisely null".

        This is CRITICAL for avoiding p-hacking: a null result with
        tight standard errors is a valid finding, not a failure.

        Args:
            estimate: Point estimate
            se: Standard error

        Returns:
            Tuple of (is_precisely_null, message)
        """
        if not self.null_acceptance_enabled:
            return False, "Null acceptance disabled"

        bound = self.equivalence_bound

        # Check if estimate is within equivalence bound
        if abs(estimate) < bound:
            # Check if CI is tight enough to be informative
            ci_width = 1.96 * se
            if ci_width < 2 * bound:
                return True, f"Effect precisely null: |β|={abs(estimate):.4f} < {bound}, CI width={ci_width:.4f}"

        return False, f"Effect not null: |β|={abs(estimate):.4f}"

    def should_continue_searching(
        self,
        current_score: float,
        best_score: float,
        iteration: int,
    ) -> bool:
        """
        Check if we should continue searching for better specifications.

        Args:
            current_score: Current credibility score
            best_score: Best score achieved so far
            iteration: Current iteration

        Returns:
            True if should continue, False if should stop
        """
        if iteration >= self.max_iterations:
            return False

        # If current is already good enough, stop
        if current_score >= self.all_critical_edges_threshold:
            return False

        # If we've tried 2+ iterations with no improvement, stop
        if iteration >= 2 and current_score <= best_score:
            return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "improvement_threshold": self.improvement_threshold,
            "all_critical_edges_threshold": self.all_critical_edges_threshold,
            "null_acceptance_enabled": self.null_acceptance_enabled,
            "equivalence_bound": self.equivalence_bound,
        }


def evaluate_stopping(
    iteration: int,
    scores_before: list[float],
    scores_after: list[float],
    critical_edges: list[str] | None = None,
    criteria: StoppingCriteria | None = None,
) -> StoppingDecision:
    """
    Evaluate whether to stop iteration.

    Convenience function that computes deltas and evaluates criteria.

    Args:
        iteration: Current iteration
        scores_before: Credibility scores before this iteration
        scores_after: Credibility scores after this iteration
        critical_edges: List of critical edge IDs (if None, uses all)
        criteria: Stopping criteria (uses defaults if None)

    Returns:
        StoppingDecision
    """
    criteria = criteria or StoppingCriteria()

    # Compute delta
    mean_before = sum(scores_before) / len(scores_before) if scores_before else 0
    mean_after = sum(scores_after) / len(scores_after) if scores_after else 0
    delta = mean_after - mean_before

    # For now, use all scores as critical if not specified
    critical_scores = scores_after

    return criteria.should_stop(
        iteration=iteration,
        credibility_delta=delta,
        critical_edges_scores=critical_scores,
    )
