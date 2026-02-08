"""Tests for RegressionKinkAdapter."""

import pytest

from shared.engine.adapters.base import EstimationRequest
from shared.engine.adapters.regression_kink_adapter import RegressionKinkAdapter
from tests.fixtures.synthetic_dgp import make_regression_kink_dgp


@pytest.fixture
def adapter():
    return RegressionKinkAdapter()


@pytest.fixture
def kink_data():
    return make_regression_kink_dgp(n=1000, kink_point=0.0, slope_change=1.5, seed=42)


def test_supported_designs(adapter):
    assert "REGRESSION_KINK" in adapter.supported_designs()


def test_validate_requires_running_variable(adapter, kink_data):
    df, _ = kink_data
    req = EstimationRequest(df=df, outcome="Y", treatment="X", extra={})
    errors = adapter.validate_request(req)
    assert any("running_variable" in e for e in errors)


def test_validate_requires_kink_point(adapter, kink_data):
    df, _ = kink_data
    req = EstimationRequest(
        df=df, outcome="Y", treatment="X",
        extra={"running_variable": "X"},
    )
    errors = adapter.validate_request(req)
    assert any("kink_point" in e for e in errors)


def test_kink_recovers_slope_change(adapter, kink_data):
    df, truth = kink_data
    req = EstimationRequest(
        df=df,
        outcome="Y",
        treatment="X",
        extra={"running_variable": "X", "kink_point": truth["kink_point"]},
    )
    result = adapter.estimate(req)
    assert abs(result.point - truth["slope_change"]) < 0.5, (
        f"Estimate {result.point:.3f} too far from true {truth['slope_change']}"
    )
    assert result.se > 0
    assert result.n_obs > 0
    assert result.method_name == "REGRESSION_KINK"
    assert "density_ratio" in result.diagnostics
