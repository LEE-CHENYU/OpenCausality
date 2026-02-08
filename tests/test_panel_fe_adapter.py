"""Tests for PanelFEBackdoorAdapter."""

import pytest

from shared.engine.adapters.base import EstimationRequest
from shared.engine.adapters.panel_fe_adapter import PanelFEBackdoorAdapter
from tests.fixtures.synthetic_dgp import make_panel_fe_dgp


@pytest.fixture
def adapter():
    return PanelFEBackdoorAdapter()


@pytest.fixture
def panel_data():
    return make_panel_fe_dgp(n_units=20, n_periods=30, beta=2.0, seed=42)


def test_supported_designs(adapter):
    assert "PANEL_FE_BACKDOOR" in adapter.supported_designs()


def test_validate_requires_unit_time(adapter, panel_data):
    df, _ = panel_data
    req = EstimationRequest(df=df, outcome="Y", treatment="X")
    errors = adapter.validate_request(req)
    assert any("unit" in e.lower() for e in errors)
    assert any("time" in e.lower() for e in errors)


def test_panel_fe_recovers_effect(adapter, panel_data):
    df, truth = panel_data
    req = EstimationRequest(
        df=df,
        outcome="Y",
        treatment="X",
        unit="unit",
        time="time",
    )
    result = adapter.estimate(req)
    assert abs(result.point - truth["beta"]) < 0.5, (
        f"Estimate {result.point:.3f} too far from true {truth['beta']}"
    )
    assert result.se > 0
    assert result.n_obs > 0
    assert result.method_name == "PANEL_FE_BACKDOOR"
    assert result.diagnostics.get("r2_within", 0) > 0
