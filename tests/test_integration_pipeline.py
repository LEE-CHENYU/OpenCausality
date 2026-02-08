"""
Integration Tests for the OpenCausality Pipeline.

Tests the full pipeline from DGP generation through estimation,
validation, and issue detection. Uses fixed seeds for reproducibility.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from tests.fixtures.synthetic_dgp import (
    make_clean_ts,
    make_confounded_ts,
    make_minimal_dag_yaml,
    make_nonstationary_ts,
    make_regime_break_ts,
    make_reverse_causal_ts,
    make_small_sample_ts,
)


class TestDGPSanity:
    """Verify DGP functions produce expected data shapes."""

    def test_clean_ts_shape(self):
        df = make_clean_ts(n=100)
        assert len(df) == 100
        assert set(df.columns) >= {"date", "x", "y"}

    def test_confounded_ts_has_confounder(self):
        df = make_confounded_ts(n=100)
        assert "z" in df.columns
        assert len(df) == 100

    def test_nonstationary_ts_is_random_walk(self):
        df = make_nonstationary_ts(n=200, seed=42)
        # Non-trivial random walk
        assert df["x"].iloc[-1] != df["x"].iloc[0]

    def test_regime_break_sign_flip(self):
        df = make_regime_break_ts(n=100, beta_pre=0.5, beta_post=-0.3, break_at=50)
        # Pre-period: positive relationship
        pre = df[df["regime"] == "pre"]
        post = df[df["regime"] == "post"]
        corr_pre = np.corrcoef(pre["x"], pre["y"])[0, 1]
        corr_post = np.corrcoef(post["x"], post["y"])[0, 1]
        assert corr_pre > 0, "Pre-break should have positive correlation"
        assert corr_post < 0, "Post-break should have negative correlation"

    def test_reverse_causal_ts_lag_structure(self):
        df = make_reverse_causal_ts(n=200, seed=42)
        # X should be correlated with lagged Y
        corr = np.corrcoef(df["x"].iloc[1:].values, df["y"].iloc[:-1].values)[0, 1]
        assert abs(corr) > 0.2, "X should correlate with lagged Y"

    def test_small_sample(self):
        df = make_small_sample_ts(n=20)
        assert len(df) == 20


class TestMinimalDAG:
    """Test DAG YAML generation and parsing."""

    def test_create_and_parse_dag(self, tmp_path):
        dag_path = make_minimal_dag_yaml(
            edges=[{"id": "x_to_y", "from": "x", "to": "y"}],
            nodes=[{"id": "x", "name": "X"}, {"id": "y", "name": "Y"}],
            tmp_path=tmp_path,
        )
        assert dag_path.exists()

        from shared.agentic.dag.parser import parse_dag

        dag = parse_dag(dag_path)
        assert dag.metadata.name == "test_dag"
        assert len(dag.nodes) == 2
        assert len(dag.edges) == 1
        assert dag.edges[0].from_node == "x"
        assert dag.edges[0].to_node == "y"

    def test_dag_with_latents(self, tmp_path):
        dag_path = make_minimal_dag_yaml(
            edges=[{"id": "x_to_y", "from": "x", "to": "y"}],
            nodes=[
                {"id": "x", "name": "X"},
                {"id": "y", "name": "Y"},
            ],
            tmp_path=tmp_path,
            latents=[{"id": "U", "description": "Unobserved confounder", "affects": ["x", "y"]}],
        )

        from shared.agentic.dag.parser import parse_dag

        dag = parse_dag(dag_path)
        assert len(dag.latents) == 1
        assert dag.latents[0].id == "U"
        assert set(dag.latents[0].affects) == {"x", "y"}


class TestCleanDGPEstimation:
    """Test that clean DGP produces sensible estimates."""

    def test_clean_dgp_lp_estimate(self):
        """Clean DGP with known beta should produce close estimate."""
        df = make_clean_ts(n=200, beta=0.5, seed=42)

        from shared.engine.ts_estimator import estimate_lp

        result = estimate_lp(
            y=df["y"],
            x=df["x"],
            max_horizon=3,
            n_lags=2,
            edge_id="test_clean",
        )
        # Impact estimate should be close to true beta
        assert abs(result.impact_estimate - 0.5) < 0.15, (
            f"Expected ~0.5, got {result.impact_estimate:.4f}"
        )


class TestAdapterBase:
    """Test adapter dataclasses and base validation."""

    def test_estimation_request_creation(self):
        import pandas as pd

        from shared.engine.adapters.base import EstimationRequest

        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        req = EstimationRequest(df=df, outcome="y", treatment="x")
        assert req.outcome == "y"
        assert req.treatment == "x"
        assert req.controls == []

    def test_estimation_result_creation(self):
        from shared.engine.adapters.base import EstimationResult

        result = EstimationResult(
            point=0.5,
            se=0.1,
            ci_lower=0.3,
            ci_upper=0.7,
            pvalue=0.001,
            n_obs=100,
            method_name="OLS",
            library="statsmodels",
            library_version="0.14.0",
        )
        assert result.point == 0.5
        assert result.n_obs == 100

    def test_adapter_validate_missing_column(self):
        import pandas as pd

        from shared.engine.adapters.base import EstimationRequest
        from shared.engine.adapters.lp_adapter import LPAdapter

        adapter = LPAdapter()
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        req = EstimationRequest(df=df, outcome="y", treatment="missing_col")
        errors = adapter.validate_request(req)
        assert any("missing_col" in e for e in errors)


class TestAdapterRegistry:
    """Test the adapter registry."""

    def test_get_lp_adapter(self):
        from shared.engine.adapters.registry import get_adapter

        adapter = get_adapter("LOCAL_PROJECTIONS")
        assert "LOCAL_PROJECTIONS" in adapter.supported_designs()

    def test_get_panel_lp_adapter(self):
        from shared.engine.adapters.registry import get_adapter

        adapter = get_adapter("PANEL_LP_EXPOSURE_FE")
        assert "PANEL_LP_EXPOSURE_FE" in adapter.supported_designs()

    def test_unknown_design_raises(self):
        from shared.engine.adapters.registry import get_adapter

        with pytest.raises(ValueError, match="No adapter registered"):
            get_adapter("NONEXISTENT_DESIGN_12345")

    def test_list_adapters(self):
        from shared.engine.adapters.registry import list_adapters

        adapters = list_adapters()
        assert "LOCAL_PROJECTIONS" in adapters
        assert "PANEL_LP_EXPOSURE_FE" in adapters


class TestDAGVizBuilder:
    """Test DAG visualization builder."""

    def test_build_from_dag_yaml(self, tmp_path):
        dag_path = make_minimal_dag_yaml(
            edges=[
                {"id": "x_to_y", "from": "x", "to": "y"},
                {"id": "y_to_z", "from": "y", "to": "z"},
            ],
            nodes=[
                {"id": "x", "name": "Treatment"},
                {"id": "y", "name": "Mediator"},
                {"id": "z", "name": "Outcome"},
            ],
            tmp_path=tmp_path,
        )
        output_path = tmp_path / "viz.html"

        from scripts.build_dag_viz import build

        result = build(dag_path, output_path=output_path)
        assert result.exists()

        content = result.read_text()
        assert "Treatment" in content
        assert "Mediator" in content
        assert "Outcome" in content
        assert "d3.v7" in content


class TestHITLPanelBuilder:
    """Test HITL panel builds without error from minimal state."""

    def test_panel_builds_from_minimal_state(self, tmp_path):
        # Create minimal state.json
        state = {
            "issues": {
                "TEST_RULE:test_edge": {
                    "rule_id": "TEST_RULE",
                    "severity": "HIGH",
                    "status": "OPEN",
                    "message": "Test issue for integration testing",
                }
            }
        }
        state_path = tmp_path / "state.json"
        with open(state_path, "w") as f:
            json.dump(state, f)

        # Create minimal edge card
        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        card = {
            "edge_id": "test_edge",
            "estimates": {"point": 0.5, "se": 0.1, "pvalue": 0.01},
            "diagnostics": {
                "test_diag": {"name": "test_diag", "passed": True, "value": 1.0}
            },
            "credibility_rating": "B",
            "credibility_score": 0.65,
            "spec_details": {"design": "LOCAL_PROJECTIONS"},
            "identification": {"claim_level": "REDUCED_FORM"},
            "failure_flags": {},
        }
        with open(cards_dir / "test_edge.yaml", "w") as f:
            yaml.dump(card, f)

        # Create minimal actions
        actions_path = tmp_path / "actions.yaml"
        with open(actions_path, "w") as f:
            yaml.dump({
                "actions": {
                    "TEST_RULE": {
                        "action_type": "resolve",
                        "options": [
                            {"value": "accept", "label": "Accept", "tooltip": "Accept the result"},
                        ],
                    },
                    "_default": {
                        "action_type": "resolve",
                        "options": [
                            {"value": "accept", "label": "Accept"},
                        ],
                    },
                }
            }, f)

        # Create minimal registry
        registry_path = tmp_path / "registry.yaml"
        with open(registry_path, "w") as f:
            yaml.dump({
                "rules": [
                    {
                        "rule_id": "TEST_RULE",
                        "description": "Test rule",
                        "explanation": "Test explanation",
                        "guidance": "Test guidance",
                    }
                ]
            }, f)

        # Create minimal DAG
        dag_path = make_minimal_dag_yaml(
            edges=[{"id": "test_edge", "from": "x", "to": "y"}],
            nodes=[{"id": "x", "name": "X"}, {"id": "y", "name": "Y"}],
            tmp_path=tmp_path / "dag",
        )

        from scripts.build_hitl_panel import build

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result_path = build(
            state_path=state_path,
            cards_dir=cards_dir,
            actions_path=actions_path,
            registry_path=registry_path,
            output_dir=output_dir,
            dag_path=dag_path,
        )

        assert result_path.exists()
        content = result_path.read_text()
        # Check key content is present
        assert "TEST_RULE" in content
        assert "test_edge" in content
        assert "Decision Guide" in content or "decision-guide" in content
