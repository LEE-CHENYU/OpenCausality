"""
Tests for DoubleML and EconML CATE adapters.

These tests require doubleml and econml to be installed. Skip if not available.
"""

from __future__ import annotations

import pytest

from tests.fixtures.synthetic_dgp import (
    make_dml_dgp,
    make_heterogeneous_dgp,
)

try:
    import doubleml
    HAS_DML = True
except ImportError:
    HAS_DML = False

try:
    import econml
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False


@pytest.mark.skipif(not HAS_DML, reason="doubleml not installed")
class TestDMLAdapter:
    def test_plr_recovery(self):
        from shared.engine.adapters.base import EstimationRequest
        from shared.engine.adapters.dml_adapter import DMLAdapter

        df, truth = make_dml_dgp(n=500, theta=1.5)

        adapter = DMLAdapter()
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="D",
            controls=["X1", "X2"],
            extra={"dml_model": "PLR", "n_folds": 3},
        )

        result = adapter.estimate(req)
        # Should recover theta ≈ 1.5 (with some tolerance for ML estimation)
        assert abs(result.point - truth["theta"]) < 1.0, (
            f"Expected ~{truth['theta']}, got {result.point}"
        )
        assert result.method_name == "DML_PLR"
        assert result.library == "doubleml"
        assert result.diagnostics["n_folds"] == 3

    def test_irm_binary_treatment(self):
        from shared.engine.adapters.base import EstimationRequest
        from shared.engine.adapters.dml_adapter import DMLAdapter

        df, truth = make_heterogeneous_dgp(n=500, base_effect=2.0)

        adapter = DMLAdapter()
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="D",
            controls=["X1", "X2"],
            extra={"dml_model": "IRM", "n_folds": 3},
        )

        result = adapter.estimate(req)
        # Should recover ATE ≈ 2.0
        assert abs(result.point - truth["ate"]) < 1.5, (
            f"Expected ~{truth['ate']}, got {result.point}"
        )
        assert result.method_name == "DML_IRM"

    def test_supported_designs(self):
        from shared.engine.adapters.dml_adapter import DMLAdapter
        designs = DMLAdapter().supported_designs()
        assert "DML_PLR" in designs
        assert "DML_IRM" in designs
        assert "DML_PLIV" in designs


@pytest.mark.skipif(not HAS_ECONML, reason="econml not installed")
class TestEconMLCATEAdapter:
    def test_linear_dml_ate(self):
        from shared.engine.adapters.base import EstimationRequest
        from shared.engine.adapters.econml_cate_adapter import EconMLCATEAdapter

        df, truth = make_heterogeneous_dgp(n=500, base_effect=2.0, heterogeneity=1.0)

        adapter = EconMLCATEAdapter()
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="D",
            controls=["X1", "X2"],
            extra={"cate_model": "linear_dml"},
        )

        result = adapter.estimate(req)
        # ATE should be close to 2.0
        assert abs(result.point - truth["ate"]) < 1.5, (
            f"Expected ~{truth['ate']}, got {result.point}"
        )
        assert result.method_name == "EconML_linear_dml"
        assert "cate_std" in result.diagnostics
        assert "heterogeneity_detected" in result.diagnostics

    def test_heterogeneity_detection(self):
        from shared.engine.adapters.base import EstimationRequest
        from shared.engine.adapters.econml_cate_adapter import EconMLCATEAdapter

        # Large heterogeneity should be detected
        df, truth = make_heterogeneous_dgp(n=1000, base_effect=0.5, heterogeneity=3.0)

        adapter = EconMLCATEAdapter()
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="D",
            controls=["X1", "X2"],
            extra={"cate_model": "linear_dml"},
        )

        result = adapter.estimate(req)
        # With heterogeneity=3.0, spread should be large
        assert result.diagnostics["cate_std"] > 0.5

    def test_validation_no_controls(self):
        from shared.engine.adapters.base import EstimationRequest
        from shared.engine.adapters.econml_cate_adapter import EconMLCATEAdapter

        df, _ = make_heterogeneous_dgp(n=50)
        adapter = EconMLCATEAdapter()
        req = EstimationRequest(df=df, outcome="Y", treatment="D", controls=[])
        errors = adapter.validate_request(req)
        assert any("control" in e.lower() for e in errors)
