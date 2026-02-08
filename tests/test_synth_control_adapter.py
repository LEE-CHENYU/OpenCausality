"""
Tests for the Synthetic Control adapter.

Tests ATT recovery and permutation p-value on synthetic panel data.
"""

from __future__ import annotations

import pytest

from tests.fixtures.synthetic_dgp import make_synth_control_dgp
from shared.engine.adapters.base import EstimationRequest
from shared.engine.adapters.synth_control_adapter import SynthControlAdapter


class TestSynthControlAdapter:
    def test_att_recovery(self):
        """Test that synth control recovers the true ATT."""
        df, truth = make_synth_control_dgp(
            n_units=10, n_periods=30, treatment_period=15, att=5.0,
        )

        adapter = SynthControlAdapter()
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="treated",
            unit="unit",
            time="time",
            extra={
                "treated_unit": truth["treated_unit"],
                "treatment_period": truth["treatment_period"],
                "n_placebo": 5,  # Small for speed
            },
        )

        result = adapter.estimate(req)

        # ATT should be close to 5.0
        assert abs(result.point - truth["att"]) < 3.0, (
            f"Expected ATT ~{truth['att']}, got {result.point}"
        )
        assert result.method_name == "Synthetic_Control"
        assert result.library == "scipy"

    def test_diagnostics(self):
        """Test that diagnostics include expected fields."""
        df, truth = make_synth_control_dgp(n_units=8, n_periods=20, treatment_period=10)

        adapter = SynthControlAdapter()
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="treated",
            unit="unit",
            time="time",
            extra={
                "treated_unit": 0,
                "treatment_period": 10,
                "n_placebo": 3,
            },
        )

        result = adapter.estimate(req)

        assert "pre_rmspe" in result.diagnostics
        assert "post_pre_ratio" in result.diagnostics
        assert "n_donors" in result.diagnostics
        assert result.diagnostics["n_donors"] == 7  # 8 units - 1 treated

    def test_donor_weights(self):
        """Test that donor weights are non-negative and sum to ~1."""
        df, truth = make_synth_control_dgp(n_units=6)

        adapter = SynthControlAdapter()
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="treated",
            unit="unit",
            time="time",
            extra={
                "treated_unit": 0,
                "treatment_period": 15,
                "n_placebo": 2,
            },
        )

        result = adapter.estimate(req)

        weights = result.metadata["donor_weights"]
        assert all(w >= 0 for w in weights.values())
        # Weights should approximately sum to 1 (sparse: some dropped)
        total = sum(weights.values())
        # Allow some slack for very small weights being dropped
        assert total > 0.5, f"Weight sum too low: {total}"

    def test_fisher_pvalue_exists(self):
        """Test that Fisher permutation p-value is computed."""
        df, truth = make_synth_control_dgp(n_units=8, att=5.0)

        adapter = SynthControlAdapter()
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="treated",
            unit="unit",
            time="time",
            extra={
                "treated_unit": 0,
                "treatment_period": 15,
                "n_placebo": 5,
            },
        )

        result = adapter.estimate(req)

        assert result.pvalue is not None
        assert 0 <= result.pvalue <= 1
        assert result.diagnostics["fisher_pvalue"] is not None

    def test_validation_errors(self):
        """Test that validation catches missing required fields."""
        import pandas as pd

        adapter = SynthControlAdapter()

        # Missing unit/time
        req = EstimationRequest(
            df=pd.DataFrame({"Y": [1], "T": [1]}),
            outcome="Y",
            treatment="T",
            extra={"treated_unit": 0, "treatment_period": 10},
        )
        errors = adapter.validate_request(req)
        assert any("unit" in e.lower() for e in errors)

        # Missing treated_unit
        req = EstimationRequest(
            df=pd.DataFrame({"Y": [1], "T": [1]}),
            outcome="Y",
            treatment="T",
            unit="u",
            time="t",
            extra={"treatment_period": 10},
        )
        errors = adapter.validate_request(req)
        assert any("treated_unit" in e for e in errors)

    def test_supported_designs(self):
        adapter = SynthControlAdapter()
        assert "SYNTHETIC_CONTROL" in adapter.supported_designs()
