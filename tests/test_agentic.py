"""
Tests for the agentic causal inference framework.

Covers:
- compute_credibility_score (clamping, rating thresholds)
- validate_edge_card (NaN/Inf checks, structural validation)
- validate_lp_result (vector length, per-element invariants)
- validate_chain_units (unit compatibility across chains)
- EdgeCard dataclass (diagnostics, serialization)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.agentic.output.edge_card import (
    EdgeCard,
    Estimates,
    DiagnosticResult,
    Interpretation,
    FailureFlags,
    compute_credibility_score,
    rating_from_score,
    RATING_THRESHOLDS,
    RATING_DEFAULT,
)
from shared.agentic.output.provenance import SpecDetails
from shared.agentic.validation import (
    ValidationSeverity,
    validate_edge_card,
    validate_chain_units,
    _is_bad_float,
    _extract_unit_tokens,
)
from shared.engine.ts_estimator import LPResult, validate_lp_result


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_edge_card() -> EdgeCard:
    """Construct a valid EdgeCard with realistic values."""
    estimates = Estimates(
        point=0.50,
        se=0.10,
        ci_95=(0.30, 0.70),
        pvalue=0.001,
        horizons=[0, 1, 2],
        irf=[0.50, 0.40, 0.30],
        irf_ci_lower=[0.30, 0.20, 0.10],
        irf_ci_upper=[0.70, 0.60, 0.50],
        treatment_unit="1 SD shock",
        outcome_unit="% change in Y",
    )
    diagnostics = {
        "test_diag": DiagnosticResult(
            name="test_diag", passed=True, value=0.8, threshold=0.5,
        ),
    }
    score, rating = compute_credibility_score(
        diagnostics=diagnostics,
        failure_flags=FailureFlags(),
        design_weight=0.7,
        data_coverage=0.9,
    )
    return EdgeCard(
        edge_id="x_to_y",
        dag_version_hash="abc123",
        spec_details=SpecDetails(design="LOCAL_PROJECTIONS"),
        estimates=estimates,
        diagnostics=diagnostics,
        interpretation=Interpretation(estimand="test"),
        failure_flags=FailureFlags(),
        credibility_rating=rating,
        credibility_score=score,
    )


# ===========================================================================
# TestComputeCredibilityScore
# ===========================================================================


class TestComputeCredibilityScore:
    """Tests for compute_credibility_score and rating_from_score."""

    def test_all_pass_high_score(self):
        """All diagnostics pass, no flags -> high score."""
        diagnostics = {
            "d1": DiagnosticResult(name="d1", passed=True),
            "d2": DiagnosticResult(name="d2", passed=True),
        }
        score, rating = compute_credibility_score(
            diagnostics=diagnostics,
            failure_flags=FailureFlags(),
            design_weight=0.8,
            data_coverage=1.0,
        )
        assert 0 <= score <= 1
        assert rating in ("A", "B")
        assert score >= 0.80  # Should be high enough for A

    def test_clamping_excess_inputs(self):
        """design_weight > 1.0 and data_coverage > 1.0 are clamped."""
        diagnostics = {
            "d1": DiagnosticResult(name="d1", passed=True),
        }
        score, _ = compute_credibility_score(
            diagnostics=diagnostics,
            failure_flags=FailureFlags(),
            design_weight=5.0,  # Should be clamped to 1.0
            data_coverage=3.0,  # Should be clamped to 1.0
        )
        assert score <= 1.0, f"Score {score} exceeds 1.0 despite clamping"

    def test_empty_diagnostics(self):
        """Empty diagnostics use 0.5 pass rate."""
        score, rating = compute_credibility_score(
            diagnostics={},
            failure_flags=FailureFlags(),
            design_weight=0.6,
            data_coverage=0.8,
        )
        assert 0 <= score <= 1
        assert rating in ("A", "B", "C", "D")

    def test_all_flags_low_score(self):
        """Many failure flags -> low score."""
        diagnostics = {
            "d1": DiagnosticResult(name="d1", passed=False),
        }
        flags = FailureFlags(
            weak_identification=True,
            potential_bad_control=True,
            mechanical_identity_risk=True,
            regime_break_detected=True,
            small_sample=True,
            high_missing_rate=True,
        )
        score, rating = compute_credibility_score(
            diagnostics=diagnostics,
            failure_flags=flags,
            design_weight=0.3,
            data_coverage=0.2,
        )
        assert score < 0.40
        assert rating in ("C", "D")

    def test_rating_from_score_thresholds(self):
        """rating_from_score follows RATING_THRESHOLDS exactly."""
        assert rating_from_score(1.0) == "A"
        assert rating_from_score(0.80) == "A"
        assert rating_from_score(0.79) == "B"
        assert rating_from_score(0.60) == "B"
        assert rating_from_score(0.59) == "C"
        assert rating_from_score(0.40) == "C"
        assert rating_from_score(0.39) == "D"
        assert rating_from_score(0.0) == "D"

    def test_negative_inputs_clamped(self):
        """Negative design_weight / data_coverage clamped to 0."""
        diagnostics = {"d1": DiagnosticResult(name="d1", passed=True)}
        score, _ = compute_credibility_score(
            diagnostics=diagnostics,
            failure_flags=FailureFlags(),
            design_weight=-1.0,
            data_coverage=-0.5,
        )
        assert score >= 0.0


# ===========================================================================
# TestValidateEdgeCard
# ===========================================================================


class TestValidateEdgeCard:
    """Tests for validate_edge_card."""

    def test_valid_card_passes(self, minimal_edge_card):
        """A well-formed card has no ERRORs."""
        result = validate_edge_card(minimal_edge_card)
        assert result.passed is True
        errors = [i for i in result.issues if i.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0

    def test_nan_point_estimate(self, minimal_edge_card):
        """NaN point estimate -> ERROR."""
        minimal_edge_card.estimates.point = float("nan")
        result = validate_edge_card(minimal_edge_card)
        assert result.passed is False
        assert any(i.check_id == "nan_inf_point" for i in result.issues)

    def test_nan_se(self, minimal_edge_card):
        """NaN standard error -> ERROR."""
        minimal_edge_card.estimates.se = float("nan")
        result = validate_edge_card(minimal_edge_card)
        assert any(i.check_id == "nan_inf_se" for i in result.issues)

    def test_nan_pvalue(self, minimal_edge_card):
        """NaN p-value -> ERROR."""
        minimal_edge_card.estimates.pvalue = float("nan")
        result = validate_edge_card(minimal_edge_card)
        assert any(i.check_id == "nan_inf_pvalue" for i in result.issues)

    def test_nan_score(self, minimal_edge_card):
        """NaN credibility score -> ERROR."""
        minimal_edge_card.credibility_score = float("nan")
        result = validate_edge_card(minimal_edge_card)
        assert any(i.check_id == "nan_inf_score" for i in result.issues)

    def test_inf_ci(self, minimal_edge_card):
        """Inf CI bound -> ERROR."""
        minimal_edge_card.estimates.ci_95 = (float("-inf"), 0.70)
        result = validate_edge_card(minimal_edge_card)
        assert any(i.check_id == "nan_inf_ci" for i in result.issues)

    def test_negative_se(self, minimal_edge_card):
        """Negative SE -> ERROR."""
        minimal_edge_card.estimates.se = -0.05
        result = validate_edge_card(minimal_edge_card)
        assert any(i.check_id == "negative_se" for i in result.issues)

    def test_ci_order(self, minimal_edge_card):
        """CI lower > upper -> ERROR."""
        minimal_edge_card.estimates.ci_95 = (0.70, 0.30)
        result = validate_edge_card(minimal_edge_card)
        assert any(i.check_id == "ci_order" for i in result.issues)

    def test_pvalue_range(self, minimal_edge_card):
        """p-value > 1 -> ERROR."""
        minimal_edge_card.estimates.pvalue = 1.5
        result = validate_edge_card(minimal_edge_card)
        assert any(i.check_id == "pvalue_range" for i in result.issues)

    def test_irf_length_mismatch(self, minimal_edge_card):
        """IRF length != horizons length -> ERROR."""
        minimal_edge_card.estimates.irf = [0.5, 0.4]  # 2 instead of 3
        result = validate_edge_card(minimal_edge_card)
        assert any(i.check_id == "irf_length_mismatch" for i in result.issues)

    def test_rating_mismatch(self, minimal_edge_card):
        """Rating more generous than score implies -> WARNING."""
        minimal_edge_card.credibility_score = 0.50  # Should be C
        minimal_edge_card.credibility_rating = "A"  # Too generous
        result = validate_edge_card(minimal_edge_card)
        assert any(i.check_id == "rating_too_generous" for i in result.issues)


# ===========================================================================
# TestValidateLPResult
# ===========================================================================


class TestValidateLPResult:
    """Tests for validate_lp_result."""

    @pytest.fixture
    def valid_lp_result(self) -> LPResult:
        """A valid LP result."""
        np.random.seed(42)
        return LPResult(
            edge_id="test_edge",
            horizons=[0, 1, 2],
            coefficients=[0.5, 0.4, 0.3],
            std_errors=[0.1, 0.12, 0.15],
            ci_lower=[0.3, 0.16, 0.0],
            ci_upper=[0.7, 0.64, 0.6],
            pvalues=[0.001, 0.01, 0.05],
            nobs=[100, 95, 90],
        )

    def test_valid_result_no_violations(self, valid_lp_result):
        """Valid LP result has no violations."""
        violations = validate_lp_result(valid_lp_result)
        assert violations == []

    def test_length_mismatch(self, valid_lp_result):
        """Vector length != horizons length -> violation."""
        valid_lp_result.coefficients = [0.5, 0.4]  # 2 instead of 3
        violations = validate_lp_result(valid_lp_result)
        assert len(violations) >= 1
        assert "coefficients length" in violations[0]

    def test_negative_se(self, valid_lp_result):
        """Negative SE -> violation."""
        valid_lp_result.std_errors = [0.1, -0.05, 0.15]
        violations = validate_lp_result(valid_lp_result)
        assert any("Negative SE" in v for v in violations)

    def test_pvalue_range(self, valid_lp_result):
        """p-value > 1 -> violation."""
        valid_lp_result.pvalues = [0.001, 1.5, 0.05]
        violations = validate_lp_result(valid_lp_result)
        assert any("p-value out of" in v for v in violations)

    def test_nan_elements_skipped(self, valid_lp_result):
        """NaN values in vectors are valid (failed horizons) and not flagged."""
        valid_lp_result.std_errors = [0.1, float("nan"), 0.15]
        valid_lp_result.pvalues = [0.001, float("nan"), 0.05]
        violations = validate_lp_result(valid_lp_result)
        assert violations == []


# ===========================================================================
# TestValidateChainUnits
# ===========================================================================


class TestValidateChainUnits:
    """Tests for validate_chain_units."""

    def _make_card(self, edge_id: str, treatment_unit: str, outcome_unit: str) -> EdgeCard:
        """Helper to make a minimal EdgeCard with units."""
        return EdgeCard(
            edge_id=edge_id,
            dag_version_hash="test",
            spec_details=SpecDetails(design="TEST"),
            estimates=Estimates(
                point=0.5, se=0.1, ci_95=(0.3, 0.7),
                treatment_unit=treatment_unit,
                outcome_unit=outcome_unit,
            ),
            interpretation=Interpretation(estimand="test"),
            credibility_score=0.7,
            credibility_rating="B",
        )

    def test_compatible_chain(self):
        """Compatible units (shared tokens) -> no warnings."""
        cards = {
            "a_to_b": self._make_card("a_to_b", "1 SD shock", "pp NBK rate"),
            "b_to_c": self._make_card("b_to_c", "1pp NBK rate increase", "bps NPL"),
        }
        dag_edges = {
            "a_to_b": ("node_a", "node_b"),
            "b_to_c": ("node_b", "node_c"),
        }
        result = validate_chain_units(cards, dag_edges)
        # "nbk" and "rate" should overlap
        assert len([i for i in result.issues if i.check_id == "chain_unit_mismatch"]) == 0

    def test_incompatible_chain(self):
        """Incompatible units (no shared tokens) -> warning."""
        cards = {
            "a_to_b": self._make_card("a_to_b", "1 SD shock", "USD millions"),
            "b_to_c": self._make_card("b_to_c", "1pp KZT rate", "bps NPL"),
        }
        dag_edges = {
            "a_to_b": ("node_a", "node_b"),
            "b_to_c": ("node_b", "node_c"),
        }
        result = validate_chain_units(cards, dag_edges)
        mismatches = [i for i in result.issues if i.check_id == "chain_unit_mismatch"]
        assert len(mismatches) >= 1

    def test_missing_units_skipped(self):
        """Empty unit strings are silently skipped."""
        cards = {
            "a_to_b": self._make_card("a_to_b", "1 SD shock", ""),
            "b_to_c": self._make_card("b_to_c", "", "bps NPL"),
        }
        dag_edges = {
            "a_to_b": ("node_a", "node_b"),
            "b_to_c": ("node_b", "node_c"),
        }
        result = validate_chain_units(cards, dag_edges)
        assert len([i for i in result.issues if i.check_id == "chain_unit_mismatch"]) == 0

    def test_stopword_only_overlap_flagged(self):
        """Units that only share stopwords ('change', 'per') should be flagged."""
        cards = {
            "a_to_b": self._make_card("a_to_b", "shock", "percent change in USD"),
            "b_to_c": self._make_card("b_to_c", "percent change in KZT", "bps"),
        }
        dag_edges = {
            "a_to_b": ("node_a", "node_b"),
            "b_to_c": ("node_b", "node_c"),
        }
        result = validate_chain_units(cards, dag_edges)
        # After stopword removal, "usd" vs "kzt" -> no overlap -> warning
        mismatches = [i for i in result.issues if i.check_id == "chain_unit_mismatch"]
        assert len(mismatches) >= 1


# ===========================================================================
# TestEdgeCardDataclass
# ===========================================================================


class TestEdgeCardDataclass:
    """Tests for EdgeCard dataclass methods."""

    def test_all_diagnostics_pass(self, minimal_edge_card):
        """all_diagnostics_pass returns True when all pass."""
        assert minimal_edge_card.all_diagnostics_pass() is True

        # Add a failing diagnostic
        minimal_edge_card.diagnostics["bad"] = DiagnosticResult(
            name="bad", passed=False,
        )
        assert minimal_edge_card.all_diagnostics_pass() is False

    def test_to_dict_keys(self, minimal_edge_card):
        """to_dict() has expected top-level keys."""
        d = minimal_edge_card.to_dict()
        expected_keys = {
            "edge_id", "dag_version_hash", "created_at",
            "data_provenance", "spec_hash", "spec_details",
            "estimates", "diagnostics", "all_diagnostics_pass",
            "interpretation", "failure_flags", "counterfactual",
            "credibility_rating", "credibility_score",
            "identification", "counterfactual_block",
            "propagation_role", "literature",
            "companion_edge_id", "is_precisely_null",
            "null_equivalence_bound",
        }
        assert expected_keys.issubset(set(d.keys()))


# ===========================================================================
# TestHelpers
# ===========================================================================


class TestHelpers:
    """Tests for helper functions."""

    def test_is_bad_float(self):
        assert _is_bad_float(float("nan")) is True
        assert _is_bad_float(float("inf")) is True
        assert _is_bad_float(float("-inf")) is True
        assert _is_bad_float(0.0) is False
        assert _is_bad_float(1.5) is False
        assert _is_bad_float(None) is False

    def test_extract_unit_tokens(self):
        tokens = _extract_unit_tokens("1pp NBK base rate increase")
        assert "nbk" in tokens
        assert "base" in tokens
        assert "rate" in tokens
        # Stopwords removed
        assert "increase" not in tokens
        assert "1" not in tokens

    def test_extract_unit_tokens_punctuation(self):
        tokens = _extract_unit_tokens("% change in USD/KZT (MoM)")
        assert "usd" in tokens
        assert "kzt" in tokens
        assert "mom" in tokens
        # Stopwords removed
        assert "in" not in tokens
        assert "change" not in tokens
