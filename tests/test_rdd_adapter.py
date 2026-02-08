"""Tests for the RDD adapter using synthetic DGPs."""

from __future__ import annotations

import pytest

from shared.engine.adapters.base import EstimationRequest
from shared.engine.adapters.rdd_adapter import RDDAdapter
from tests.fixtures.synthetic_dgp import make_sharp_rdd_dgp, make_fuzzy_rdd_dgp


@pytest.fixture
def adapter():
    return RDDAdapter()


class TestSharpRDD:
    def test_sharp_rdd_recovers_effect(self, adapter):
        """Sharp RDD should recover tau=3.0 within tolerance."""
        df, truth = make_sharp_rdd_dgp(n=2000, tau=3.0, seed=42)
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="D",
            edge_id="test_sharp_rdd",
            extra={
                "running_variable": "X",
                "cutoff": truth["cutoff"],
            },
        )
        result = adapter.estimate(req)
        assert abs(result.point - truth["tau"]) < 0.5, (
            f"Sharp RDD estimate {result.point:.3f} too far from truth {truth['tau']}"
        )
        assert result.n_obs > 100
        assert result.method_name == "SHARP_RDD"


class TestFuzzyRDD:
    def test_fuzzy_rdd_recovers_late(self, adapter):
        """Fuzzy RDD should recover LATE=2.5 within tolerance."""
        df, truth = make_fuzzy_rdd_dgp(n=2000, late=2.5, seed=42)
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="D",
            edge_id="test_fuzzy_rdd",
            extra={
                "running_variable": "X",
                "cutoff": truth["cutoff"],
            },
        )
        result = adapter.estimate(req)
        assert abs(result.point - truth["late"]) < 0.8, (
            f"Fuzzy RDD estimate {result.point:.3f} too far from truth {truth['late']}"
        )
        assert result.method_name == "FUZZY_RDD"
        assert "first_stage_f" in result.diagnostics


class TestDensityDiagnostic:
    def test_mccrary_passes_valid_data(self, adapter):
        """Density test should pass on well-behaved (uniform) running variable."""
        df, truth = make_sharp_rdd_dgp(n=2000, tau=3.0, seed=42)
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="D",
            edge_id="test_density",
            extra={
                "running_variable": "X",
                "cutoff": truth["cutoff"],
            },
        )
        result = adapter.estimate(req)
        assert result.diagnostics["density_test_pass"] is True
