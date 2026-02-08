"""
Tests for DoWhy adapters (backdoor, IV, frontdoor) and refutation engine.

These tests require dowhy to be installed. Skip if not available.
"""

from __future__ import annotations

import pytest

from tests.fixtures.synthetic_dgp import (
    make_backdoor_dgp,
    make_frontdoor_dgp,
    make_iv_dgp,
)

try:
    import dowhy
    HAS_DOWHY = True
except ImportError:
    HAS_DOWHY = False

pytestmark = pytest.mark.skipif(not HAS_DOWHY, reason="dowhy not installed")


def _make_gml(nodes: list[str], edges: list[tuple[str, str]]) -> str:
    """Helper to create a GML string."""
    lines = ["graph [", "  directed 1"]
    for node in nodes:
        lines.append(f'  node [ id "{node}" label "{node}" ]')
    for u, v in edges:
        lines.append(f'  edge [ source "{u}" target "{v}" ]')
    lines.append("]")
    return "\n".join(lines)


class TestDoWhyBackdoorAdapter:
    def test_basic_estimation(self):
        from shared.engine.adapters.base import EstimationRequest
        from shared.engine.adapters.dowhy_backdoor_adapter import DoWhyBackdoorAdapter

        df, truth = make_backdoor_dgp(n=500, beta=2.0)

        # Graph: Z -> X, Z -> Y, X -> Y
        gml = _make_gml(
            nodes=["X", "Y", "Z"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )

        adapter = DoWhyBackdoorAdapter()
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="X",
            controls=["Z"],
            extra={"graph_gml": gml},
        )

        result = adapter.estimate(req)
        # Should recover beta ≈ 2.0 (within tolerance)
        assert abs(result.point - truth["beta"]) < 0.5, (
            f"Expected ~{truth['beta']}, got {result.point}"
        )
        assert result.method_name.startswith("DoWhy_")
        assert result.library == "dowhy"

    def test_validation_missing_gml(self):
        from shared.engine.adapters.base import EstimationRequest
        from shared.engine.adapters.dowhy_backdoor_adapter import DoWhyBackdoorAdapter

        df, _ = make_backdoor_dgp(n=50)
        adapter = DoWhyBackdoorAdapter()
        req = EstimationRequest(df=df, outcome="Y", treatment="X")
        errors = adapter.validate_request(req)
        assert any("graph_gml" in e for e in errors)

    def test_supported_designs(self):
        from shared.engine.adapters.dowhy_backdoor_adapter import DoWhyBackdoorAdapter
        assert "DOWHY_BACKDOOR" in DoWhyBackdoorAdapter().supported_designs()


class TestDoWhyIVAdapter:
    def test_basic_iv_estimation(self):
        from shared.engine.adapters.base import EstimationRequest
        from shared.engine.adapters.dowhy_iv_adapter import DoWhyIVAdapter

        df, truth = make_iv_dgp(n=1000, beta=0.5)

        # Graph: Z -> X, X -> Y (with unobserved U -> X, U -> Y)
        gml = _make_gml(
            nodes=["X", "Y", "Z"],
            edges=[("Z", "X"), ("X", "Y")],
        )

        adapter = DoWhyIVAdapter()
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="X",
            instruments=["Z"],
            extra={"graph_gml": gml},
        )

        result = adapter.estimate(req)
        # IV should recover beta ≈ 0.5
        assert abs(result.point - truth["beta"]) < 1.0, (
            f"Expected ~{truth['beta']}, got {result.point}"
        )
        assert result.method_name == "DoWhy_IV"

    def test_validation_missing_instruments(self):
        from shared.engine.adapters.base import EstimationRequest
        from shared.engine.adapters.dowhy_iv_adapter import DoWhyIVAdapter

        df, _ = make_iv_dgp(n=50)
        adapter = DoWhyIVAdapter()
        req = EstimationRequest(
            df=df, outcome="Y", treatment="X",
            extra={"graph_gml": "dummy"},
        )
        errors = adapter.validate_request(req)
        assert any("instrument" in e.lower() for e in errors)


class TestDoWhyFrontdoorAdapter:
    def test_basic_frontdoor_estimation(self):
        from shared.engine.adapters.base import EstimationRequest
        from shared.engine.adapters.dowhy_frontdoor_adapter import DoWhyFrontdoorAdapter

        df, truth = make_frontdoor_dgp(n=1000)

        # Graph: X -> M -> Y (with unobserved U -> X, U -> Y)
        gml = _make_gml(
            nodes=["X", "M", "Y", "U"],
            edges=[("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y")],
        )

        adapter = DoWhyFrontdoorAdapter()
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="X",
            extra={"graph_gml": gml, "mediators": ["M"]},
        )

        result = adapter.estimate(req)
        assert result.method_name == "DoWhy_Frontdoor"
        assert result.n_obs > 0


class TestRefutationEngine:
    def test_basic_refutation(self):
        from shared.engine.refutation import RefutationEngine
        from dowhy import CausalModel

        df, _ = make_backdoor_dgp(n=300)
        gml = _make_gml(
            nodes=["X", "Y", "Z"],
            edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        )

        model = CausalModel(data=df, treatment="X", outcome="Y", graph=gml)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand, method_name="backdoor.linear_regression",
        )

        engine = RefutationEngine()
        results = engine.refute(
            model, estimand, estimate,
            refuters=["random_common_cause", "data_subset_refuter"],
        )

        assert len(results) == 2
        for r in results:
            assert r.refuter_name in ("random_common_cause", "data_subset_refuter")
            assert isinstance(r.passed, bool)

    def test_to_diagnostic_results(self):
        from shared.engine.refutation import RefutationEngine, RefutationResult

        results = [
            RefutationResult(
                refuter_name="random_common_cause",
                passed=True,
                original_estimate=2.0,
                refuted_estimate=1.98,
                message="Robust.",
            ),
        ]
        diagnostics = RefutationEngine.to_diagnostic_results(results)
        assert len(diagnostics) == 1
        assert diagnostics[0].name == "refutation_random_common_cause"
        assert diagnostics[0].passed is True
