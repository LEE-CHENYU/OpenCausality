"""
Pipeline Smoke Test for KSPI K2 DAG.

Validates that all edges in EDGE_NODE_MAP dispatch correctly through
get_edge_group() and that the corresponding estimator functions accept
the assembled data without errors. Uses synthetic fixtures where real
data is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.engine.data_assembler import (
    ACCOUNTING_BRIDGE_EDGES,
    EDGE_NODE_MAP,
    get_edge_group,
)


class TestEdgeDispatch:
    """Verify every edge in EDGE_NODE_MAP maps to a known group."""

    KNOWN_GROUPS = {
        "MONTHLY_LP",
        "IMMUTABLE",
        "QUARTERLY_LP",
        "ANNUAL_LP",
        "PANEL_LP",
        "KSPI_ONLY",
        "ACCOUNTING_BRIDGE",
        "IDENTITY",
        "DYNAMIC_LP",
    }

    def test_all_edges_dispatch_to_known_group(self):
        """Every edge must map to a recognized group, never UNKNOWN."""
        unknown_edges = []
        for edge_id in EDGE_NODE_MAP:
            group = get_edge_group(edge_id)
            if group == "UNKNOWN":
                unknown_edges.append(edge_id)
            assert group in self.KNOWN_GROUPS, (
                f"Edge '{edge_id}' dispatched to unrecognized group '{group}'"
            )
        assert not unknown_edges, f"Edges dispatched to UNKNOWN: {unknown_edges}"

    def test_no_none_groups(self):
        """get_edge_group must never return None for a registered edge."""
        for edge_id in EDGE_NODE_MAP:
            group = get_edge_group(edge_id)
            assert group is not None, f"Edge '{edge_id}' returned None group"

    def test_edge_group_counts(self):
        """Verify expected group distribution."""
        groups: dict[str, list[str]] = {}
        for edge_id in EDGE_NODE_MAP:
            g = get_edge_group(edge_id)
            groups.setdefault(g, []).append(edge_id)

        assert len(groups.get("MONTHLY_LP", [])) == 6
        assert len(groups.get("IMMUTABLE", [])) == 4
        assert len(groups.get("QUARTERLY_LP", [])) == 4
        assert len(groups.get("ACCOUNTING_BRIDGE", [])) == 2
        assert len(groups.get("IDENTITY", [])) == 2


class TestImmutableEvidence:
    """Test that immutable evidence edges return valid results."""

    IMMUTABLE_EDGES = [
        "fx_to_cpi_tradable",
        "fx_to_cpi_nontradable",
        "cpi_to_nominal_income",
        "fx_to_real_expenditure",
    ]

    def test_immutable_results_available(self):
        from shared.engine.ts_estimator import get_immutable_result

        for edge_id in self.IMMUTABLE_EDGES:
            result = get_immutable_result(edge_id)
            assert result is not None
            assert result.edge_id == edge_id
            assert isinstance(result.point_estimate, float)
            assert isinstance(result.se, float)
            assert result.se >= 0

    def test_immutable_group_classification(self):
        for edge_id in self.IMMUTABLE_EDGES:
            assert get_edge_group(edge_id) == "IMMUTABLE"


class TestIdentitySensitivity:
    """Test that identity edges compute sensitivities."""

    def test_identity_computation(self):
        from shared.engine.ts_estimator import compute_identity_sensitivity

        results = compute_identity_sensitivity(capital=500.0, rwa=3000.0)
        assert "capital_to_k2" in results
        assert "rwa_to_k2" in results

        cap_result = results["capital_to_k2"]
        assert abs(cap_result.sensitivity - 100.0 / 3000.0) < 1e-10

        rwa_result = results["rwa_to_k2"]
        expected = -100.0 * 500.0 / (3000.0**2)
        assert abs(rwa_result.sensitivity - expected) < 1e-10

    def test_identity_group_classification(self):
        for edge_id in ["capital_to_k2", "rwa_to_k2"]:
            assert get_edge_group(edge_id) == "IDENTITY"


class TestAccountingBridge:
    """Test accounting bridge edges."""

    def test_accounting_bridge_computation(self):
        from shared.engine.ts_estimator import compute_accounting_bridge

        result = compute_accounting_bridge(
            edge_id="loan_portfolio_to_rwa",
            loans=2000.0,
            rwa=3000.0,
            cor=2.0,
            capital=500.0,
        )
        assert result.is_deterministic
        assert abs(result.sensitivity - 1.5) < 1e-10  # 3000/2000

        result2 = compute_accounting_bridge(
            edge_id="cor_to_capital",
            loans=2000.0,
            rwa=3000.0,
            cor=2.0,
            capital=500.0,
        )
        assert result2.is_deterministic
        expected = -2000.0 * 0.80 / 100  # -loans*(1-tax)/100
        assert abs(result2.sensitivity - expected) < 1e-10

    def test_accounting_bridge_group_classification(self):
        for edge_id in ACCOUNTING_BRIDGE_EDGES:
            assert get_edge_group(edge_id) == "ACCOUNTING_BRIDGE"


class TestLPEstimation:
    """Test LP estimation on synthetic data."""

    def test_lp_accepts_synthetic_data(self):
        from shared.engine.ts_estimator import estimate_lp
        from tests.fixtures.synthetic_dgp import make_clean_ts

        df = make_clean_ts(n=100, beta=0.5, seed=42)
        result = estimate_lp(
            y=df["y"],
            x=df["x"],
            max_horizon=3,
            n_lags=2,
            edge_id="test_synthetic",
        )
        assert len(result.horizons) == 4  # 0,1,2,3
        assert result.impact_estimate != 0.0
        assert abs(result.impact_estimate - 0.5) < 0.2

    def test_monthly_lp_edges_classified_correctly(self):
        monthly_lp_edges = [
            "oil_supply_to_brent",
            "oil_supply_to_fx",
            "oil_demand_to_fx",
            "vix_to_fx",
            "cpi_to_nbk_rate",
            "fx_to_nbk_rate",
        ]
        for edge_id in monthly_lp_edges:
            assert get_edge_group(edge_id) == "MONTHLY_LP"

    def test_quarterly_lp_edges_classified_correctly(self):
        quarterly_lp_edges = [
            "shock_to_npl_kspi",
            "shock_to_cor_kspi",
            "nbk_rate_to_deposit_cost",
            "nbk_rate_to_cor",
        ]
        for edge_id in quarterly_lp_edges:
            assert get_edge_group(edge_id) == "QUARTERLY_LP"


class TestEdgeNodeMapCompleteness:
    """Verify EDGE_NODE_MAP is internally consistent."""

    def test_all_edges_have_two_nodes(self):
        for edge_id, (treatment, outcome) in EDGE_NODE_MAP.items():
            assert isinstance(treatment, str) and treatment, (
                f"Edge '{edge_id}' has invalid treatment node"
            )
            assert isinstance(outcome, str) and outcome, (
                f"Edge '{edge_id}' has invalid outcome node"
            )

    def test_no_self_loops(self):
        for edge_id, (treatment, outcome) in EDGE_NODE_MAP.items():
            assert treatment != outcome, (
                f"Edge '{edge_id}' is a self-loop: {treatment} -> {outcome}"
            )

    def test_edge_count(self):
        """Verify expected number of edges."""
        assert len(EDGE_NODE_MAP) >= 18, (
            f"Expected at least 18 edges, found {len(EDGE_NODE_MAP)}"
        )
