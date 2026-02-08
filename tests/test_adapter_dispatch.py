"""
Adapter Dispatch Tests.

Verifies that all registered designs have working adapters and that
the adapter-based dispatch produces valid EstimationResult objects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.engine.adapters.base import EstimationRequest, EstimationResult
from shared.engine.adapters.registry import get_adapter, list_adapters


class TestAdapterRegistration:
    """Verify all adapters are registered and loadable."""

    EXPECTED_DESIGNS = [
        "LOCAL_PROJECTIONS",
        "PANEL_LP_EXPOSURE_FE",
        "IMMUTABLE_EVIDENCE",
        "IDENTITY",
        "ACCOUNTING_BRIDGE",
    ]

    def test_all_expected_designs_registered(self):
        adapters = list_adapters()
        for design in self.EXPECTED_DESIGNS:
            assert design in adapters, f"Design '{design}' not in registry"

    def test_all_adapters_instantiate(self):
        for design in self.EXPECTED_DESIGNS:
            adapter = get_adapter(design)
            assert adapter is not None
            assert hasattr(adapter, "estimate")
            assert hasattr(adapter, "supported_designs")


class TestImmutableAdapterDispatch:
    """Test ImmutableAdapter produces valid EstimationResult."""

    IMMUTABLE_EDGES = [
        "fx_to_cpi_tradable",
        "fx_to_cpi_nontradable",
        "cpi_to_nominal_income",
        "fx_to_real_expenditure",
    ]

    def test_immutable_edges_produce_valid_results(self):
        adapter = get_adapter("IMMUTABLE_EVIDENCE")
        for edge_id in self.IMMUTABLE_EDGES:
            req = EstimationRequest(
                df=pd.DataFrame(),
                outcome="",
                treatment="",
                edge_id=edge_id,
            )
            result = adapter.estimate(req)
            assert isinstance(result, EstimationResult)
            assert isinstance(result.point, float)
            assert result.se >= 0
            assert result.n_obs >= 0
            assert result.method_name == "IMMUTABLE_EVIDENCE"


class TestIdentityAdapterDispatch:
    """Test IdentityAdapter produces valid EstimationResult."""

    def test_capital_to_k2(self):
        adapter = get_adapter("IDENTITY")
        req = EstimationRequest(
            df=pd.DataFrame({"x": [1]}),
            outcome="k2",
            treatment="capital",
            edge_id="capital_to_k2",
            extra={"capital": 500.0, "rwa": 3000.0},
        )
        result = adapter.estimate(req)
        assert isinstance(result, EstimationResult)
        assert result.method_name == "IDENTITY"
        assert abs(result.point - 100.0 / 3000.0) < 1e-10
        assert result.se == 0.0

    def test_rwa_to_k2(self):
        adapter = get_adapter("IDENTITY")
        req = EstimationRequest(
            df=pd.DataFrame({"x": [1]}),
            outcome="k2",
            treatment="rwa",
            edge_id="rwa_to_k2",
            extra={"capital": 500.0, "rwa": 3000.0},
        )
        result = adapter.estimate(req)
        assert isinstance(result, EstimationResult)
        expected = -100.0 * 500.0 / (3000.0**2)
        assert abs(result.point - expected) < 1e-10


class TestAccountingBridgeAdapterDispatch:
    """Test AccountingBridgeAdapter produces valid EstimationResult."""

    def test_loan_portfolio_to_rwa(self):
        adapter = get_adapter("ACCOUNTING_BRIDGE")
        req = EstimationRequest(
            df=pd.DataFrame({"x": [1]}),
            outcome="rwa",
            treatment="loans",
            edge_id="loan_portfolio_to_rwa",
            extra={"loans": 2000.0, "rwa": 3000.0, "cor": 2.0, "capital": 500.0},
        )
        result = adapter.estimate(req)
        assert isinstance(result, EstimationResult)
        assert result.method_name == "ACCOUNTING_BRIDGE"
        assert abs(result.point - 1.5) < 1e-10  # 3000/2000
        assert result.se == 0.0

    def test_cor_to_capital(self):
        adapter = get_adapter("ACCOUNTING_BRIDGE")
        req = EstimationRequest(
            df=pd.DataFrame({"x": [1]}),
            outcome="capital",
            treatment="cor",
            edge_id="cor_to_capital",
            extra={"loans": 2000.0, "rwa": 3000.0, "cor": 2.0, "capital": 500.0},
        )
        result = adapter.estimate(req)
        assert isinstance(result, EstimationResult)
        expected = -2000.0 * 0.80 / 100
        assert abs(result.point - expected) < 1e-10


class TestLPAdapterDispatch:
    """Test LPAdapter produces valid EstimationResult on synthetic data."""

    def test_lp_adapter_on_clean_dgp(self):
        from tests.fixtures.synthetic_dgp import make_clean_ts

        adapter = get_adapter("LOCAL_PROJECTIONS")
        df = make_clean_ts(n=100, beta=0.5, seed=42)
        # Rename for adapter: outcome and treatment columns must match
        data = df.rename(columns={"y": "outcome", "x": "treatment"}).set_index("date")
        req = EstimationRequest(
            df=data,
            outcome="outcome",
            treatment="treatment",
            edge_id="test_lp_clean",
        )
        result = adapter.estimate(req)
        assert isinstance(result, EstimationResult)
        assert result.method_name == "LOCAL_PROJECTIONS"
        assert result.n_obs > 0
        assert abs(result.point - 0.5) < 0.2


class TestGroupToDesignMapping:
    """Test that the agent_loop GROUP_TO_DESIGN mapping is complete."""

    def test_all_groups_mapped(self):
        from shared.agentic.agent_loop import AgentLoop

        expected_groups = {
            "IMMUTABLE", "IDENTITY", "MONTHLY_LP", "QUARTERLY_LP",
            "DYNAMIC_LP", "KSPI_ONLY", "PANEL_LP", "ACCOUNTING_BRIDGE",
        }
        for group in expected_groups:
            assert group in AgentLoop._GROUP_TO_DESIGN, (
                f"Group '{group}' missing from _GROUP_TO_DESIGN mapping"
            )
