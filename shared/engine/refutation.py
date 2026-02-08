"""
DoWhy Refutation Engine.

Post-estimation diagnostic that runs DoWhy refuters on any estimation result.
NOT an adapter — called after estimation to stress-test causal estimates.

Refuters:
    1. random_common_cause: Add random confounder — estimate should not change.
    2. placebo_treatment: Permute treatment — estimate should vanish.
    3. data_subset: Re-estimate on subset — estimate should be stable.
    4. add_unobserved_common_cause: Sensitivity to unobserved confounding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RefutationResult:
    """Result from a single DoWhy refutation test."""

    refuter_name: str
    passed: bool
    original_estimate: float
    refuted_estimate: float
    pvalue: float | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "refuter_name": self.refuter_name,
            "passed": self.passed,
            "original_estimate": self.original_estimate,
            "refuted_estimate": self.refuted_estimate,
            "pvalue": self.pvalue,
            "message": self.message,
        }


class RefutationEngine:
    """Run DoWhy refutation tests on a CausalModel + estimate.

    Usage:
        model = CausalModel(data=df, treatment=T, outcome=Y, graph=gml)
        estimand = model.identify_effect()
        estimate = model.estimate_effect(estimand, method_name=...)

        engine = RefutationEngine()
        results = engine.refute(model, estimand, estimate)

        # Convert to DiagnosticResult objects for EdgeCard
        diagnostics = engine.to_diagnostic_results(results)
    """

    DEFAULT_REFUTERS = [
        "random_common_cause",
        "placebo_treatment_refuter",
        "data_subset_refuter",
        "add_unobserved_common_cause",
    ]

    def refute(
        self,
        model: Any,
        identified_estimand: Any,
        estimate: Any,
        refuters: list[str] | None = None,
    ) -> list[RefutationResult]:
        """Run refutation tests and return results.

        Args:
            model: DoWhy CausalModel instance.
            identified_estimand: Result of model.identify_effect().
            estimate: Result of model.estimate_effect().
            refuters: List of refuter names to run. Defaults to all 4.

        Returns:
            List of RefutationResult objects.
        """
        refuters = refuters or self.DEFAULT_REFUTERS
        results: list[RefutationResult] = []

        original_value = float(estimate.value)

        for refuter_name in refuters:
            try:
                result = self._run_refuter(
                    model, identified_estimand, estimate, refuter_name, original_value,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Refuter {refuter_name} failed: {e}")
                results.append(RefutationResult(
                    refuter_name=refuter_name,
                    passed=False,
                    original_estimate=original_value,
                    refuted_estimate=float("nan"),
                    message=f"Refuter failed: {e}",
                ))

        return results

    def _run_refuter(
        self,
        model: Any,
        estimand: Any,
        estimate: Any,
        refuter_name: str,
        original_value: float,
    ) -> RefutationResult:
        """Run a single refuter."""
        refutation = model.refute_estimate(
            estimand,
            estimate,
            method_name=refuter_name,
        )

        refuted_value = float(refutation.new_effect)
        pvalue = None

        if hasattr(refutation, "refutation_result") and refutation.refutation_result is not None:
            pvalue = float(refutation.refutation_result)

        # Determine pass/fail based on refuter type
        if refuter_name == "placebo_treatment_refuter":
            # Placebo treatment: refuted estimate should be near zero
            passed = abs(refuted_value) < abs(original_value) * 0.5
            message = (
                f"Placebo estimate={refuted_value:.4f} vs original={original_value:.4f}. "
                f"{'Near zero as expected.' if passed else 'Still large — potential concern.'}"
            )
        elif refuter_name == "random_common_cause":
            # Random confounder: estimate should not change much
            change = abs(refuted_value - original_value) / max(abs(original_value), 1e-10)
            passed = change < 0.15  # <15% change
            message = (
                f"Estimate changed by {change:.1%} after adding random confounder. "
                f"{'Robust.' if passed else 'Sensitive to confounding.'}"
            )
        elif refuter_name == "data_subset_refuter":
            # Subset stability: estimate should be similar
            change = abs(refuted_value - original_value) / max(abs(original_value), 1e-10)
            passed = change < 0.30  # <30% change on subset
            message = (
                f"Subset estimate={refuted_value:.4f} vs full={original_value:.4f}. "
                f"{'Stable.' if passed else 'Unstable across subsets.'}"
            )
        elif refuter_name == "add_unobserved_common_cause":
            # Sensitivity: check if estimate sign changes
            passed = (refuted_value * original_value) > 0  # same sign
            message = (
                f"With unobserved confounder: {refuted_value:.4f}. "
                f"{'Sign preserved.' if passed else 'Sign flipped — fragile to unobserved confounding.'}"
            )
        else:
            passed = True
            message = f"Refuted estimate: {refuted_value:.4f}"

        return RefutationResult(
            refuter_name=refuter_name,
            passed=passed,
            original_estimate=original_value,
            refuted_estimate=refuted_value,
            pvalue=pvalue,
            message=message,
        )

    @staticmethod
    def to_diagnostic_results(
        refutation_results: list[RefutationResult],
    ) -> list[Any]:
        """Convert RefutationResults to DiagnosticResult objects for EdgeCard.

        Returns a list of ``shared.agentic.output.edge_card.DiagnosticResult`` objects.
        """
        from shared.agentic.output.edge_card import DiagnosticResult

        diagnostics = []
        for r in refutation_results:
            diagnostics.append(DiagnosticResult(
                name=f"refutation_{r.refuter_name}",
                passed=r.passed,
                value=r.refuted_estimate,
                pvalue=r.pvalue,
                message=r.message,
            ))
        return diagnostics
