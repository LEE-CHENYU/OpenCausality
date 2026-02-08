"""Tests for AgentLoop core functionality.

Uses synthetic DAGs and mocked components to test sanitization,
DataScout integration, blocked edge marking, auto-ingest wiring,
and DAG auto-repair loop cap.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from shared.agentic.agent_loop import AgentLoop, AgentLoopConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_dag():
    """Create a minimal mock DAGSpec for testing."""
    node_a = MagicMock()
    node_a.id = "a"
    node_a.name = "Node A"
    node_a.observed = True
    node_a.latent = False
    node_a.source = None

    node_b = MagicMock()
    node_b.id = "b"
    node_b.name = "Node B"
    node_b.observed = True
    node_b.latent = False
    node_b.source = None

    edge = MagicMock()
    edge.id = "a_to_b"
    edge.from_node = "a"
    edge.to_node = "b"

    dag = MagicMock()
    dag.nodes = [node_a, node_b]
    dag.edges = [edge]
    dag.metadata.name = "test_dag"
    dag.metadata.target_node = "b"
    dag.get_node = lambda nid: {"a": node_a, "b": node_b}.get(nid)
    dag.get_edge = lambda eid: edge if eid == "a_to_b" else None
    dag.compute_hash.return_value = "abc123"

    return dag


def _setup_data_scout_loop(dag, report):
    """Set up an AgentLoop with mocked internals for DataScout testing."""
    with patch.object(AgentLoop, "__init__", lambda self, *a, **kw: None):
        loop = AgentLoop.__new__(AgentLoop)
        loop.dag = dag
        loop.config = AgentLoopConfig()
        loop.catalog = MagicMock()
        loop.catalog.is_available.return_value = False
        loop.dynamic_loader = MagicMock()
        loop.dynamic_loader.auto_populate_from_dag.return_value = {}
        loop.data_scout = MagicMock()
        loop.data_scout.download_missing.return_value = report
        loop.data_scout.generate_user_guidance.return_value = []
        loop.queue = MagicMock()
        loop.audit_log = MagicMock()
        loop._data_guidance = []
        loop._data_hash = ""
        return loop


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSanitizeEdgeIds:
    """Test edge ID sanitization."""

    def test_sanitize_edge_id_arrow(self):
        result = AgentLoop._sanitize_edge_id("oil->cpi")
        assert result == "oil_to_cpi"

    def test_sanitize_edge_id_spaces(self):
        result = AgentLoop._sanitize_edge_id("oil price->cpi index")
        assert result == "oil_price_to_cpi_index"

    def test_sanitize_edge_id_clean(self):
        result = AgentLoop._sanitize_edge_id("oil_to_cpi")
        assert result == "oil_to_cpi"


class TestDataScoutRegistersLoaders:
    """Test that DataScout downloads register loaders correctly (bug fix 5A)."""

    def test_data_scout_registers_from_results(self):
        """Verify register_from_download is called with result.node_id and result.file_path."""
        from shared.agentic.agents.data_scout import DownloadResult, DataScoutReport

        dag = _make_mock_dag()

        result = DownloadResult(
            node_id="a",
            connector="fred",
            success=True,
            size_bytes=1024,
            file_path=Path("/tmp/a.parquet"),
            row_count=100,
        )
        report = DataScoutReport(
            total_nodes=1, downloaded=1, skipped=0, failed=0,
            total_bytes=1024, budget_bytes=100 * 1024 * 1024,
            results=[result],
        )

        loop = _setup_data_scout_loop(dag, report)

        # After first registration, catalog should report node as available
        # so the second scout pass finds no missing nodes and exits early.
        def _is_available_after_register(node_id):
            return loop.dynamic_loader.register_from_download.called

        loop.catalog.is_available.side_effect = _is_available_after_register

        with patch("shared.engine.data_assembler.EDGE_NODE_MAP", {}):
            loop._run_data_scout()

        loop.dynamic_loader.register_from_download.assert_called_once_with(
            edge_id="",
            node_id="a",
            file_path=Path("/tmp/a.parquet"),
        )


class TestBlockedEdgesAfterScoutExhaustion:
    """Test that edges are marked BLOCKED when DataScout exhausts all retries."""

    def test_blocked_after_exhaustion(self):
        from shared.agentic.agents.data_scout import DownloadResult, DataScoutReport

        dag = _make_mock_dag()

        fail_result = DownloadResult(
            node_id="a", connector="fred", success=False, error="404 Not Found",
        )
        report = DataScoutReport(
            total_nodes=1, downloaded=0, skipped=0, failed=1,
            total_bytes=0, budget_bytes=100 * 1024 * 1024,
            results=[fail_result],
        )

        loop = _setup_data_scout_loop(dag, report)
        loop.data_scout.generate_user_guidance.return_value = ["guidance msg"]

        with patch("shared.engine.data_assembler.EDGE_NODE_MAP", {}):
            loop._run_data_scout()

        # Verify mark_failed was called (BLOCKED)
        loop.queue.mark_failed.assert_called()
        call_args = loop.queue.mark_failed.call_args
        assert "a_to_b" in call_args[0][0]


class TestAutoIngestCalledOnRun:
    """Test that auto_ingest is called during run() startup."""

    def test_auto_ingest_called(self):
        """Verify auto_ingest() is called at the start of run()."""
        with patch.object(AgentLoop, "__init__", lambda self, *a, **kw: None):
            loop = AgentLoop.__new__(AgentLoop)
            loop.dag = _make_mock_dag()
            loop.config = AgentLoopConfig(max_iterations=0)
            loop.mode = "EXPLORATION"
            loop.run_id = "test123"
            loop.iteration = 0
            loop._dag_hash = "abc"
            loop._data_hash = ""
            loop._data_guidance = []
            loop.audit_log = MagicMock()
            loop.validator = MagicMock()
            loop.queue = MagicMock()
            loop.queue.is_complete.return_value = True
            loop.artifact_store = MagicMock()
            loop.artifact_store.get_all_edge_cards.return_value = []
            loop.issue_ledger = MagicMock()
            loop.issue_ledger.issues = []
            loop.hitl_gate = MagicMock()
            hitl_result = MagicMock()
            hitl_result.pending_count = 0
            loop.hitl_gate.evaluate.return_value = hitl_result
            loop.cross_run_reducer = MagicMock()
            loop.notifier = MagicMock()
            loop.query_mode = MagicMock()
            loop.query_mode.value = "REDUCED_FORM"
            loop.ts_guard_results = {}

            # Make validator pass
            mock_report = MagicMock()
            mock_report.is_valid = True
            loop.validator.validate.return_value = mock_report

            # Patch internal phases + report creation
            loop._sanitize_edge_ids = MagicMock()
            loop._run_data_scout = MagicMock()
            loop._run_paper_scout = MagicMock()
            loop._create_system_report = MagicMock(return_value=MagicMock())

            with patch("shared.engine.ingest.auto_ingest") as mock_ingest:
                loop.run()
                mock_ingest.assert_called_once()


class TestDagAutoRepairLoopCapsAt3:
    """Test that DAG auto-repair stops after max_attempts=3."""

    def test_repair_caps_at_3(self):
        with patch.object(AgentLoop, "__init__", lambda self, *a, **kw: None):
            loop = AgentLoop.__new__(AgentLoop)
            loop.patch_policy = MagicMock()
            loop.patch_policy.is_llm_repair_allowed.return_value = True
            loop.patch_bot = MagicMock()
            result = MagicMock()
            result.applied = True
            loop.patch_bot._apply_single_fix.return_value = result
            loop.mode = "EXPLORATION"
            loop.run_id = "test"

            # Validator always returns invalid
            mock_report = MagicMock()
            mock_report.is_valid = False
            error = MagicMock()
            error.error_type = "invalid_edge_id"
            error.edge_id = "test"
            error.context = {}
            mock_report.errors.return_value = [error]

            loop.validator = MagicMock()
            loop.validator.validate.return_value = mock_report

            final = loop._attempt_dag_auto_repair(mock_report, max_attempts=3)

            # Should have attempted validation 3 times (once per attempt after repair)
            assert loop.validator.validate.call_count == 3
            assert not final.is_valid


class TestDagAutoRepairFixesAndRevalidates:
    """Test that DAG auto-repair applies fix once when second validation passes."""

    def test_repair_fixes_and_stops(self):
        with patch.object(AgentLoop, "__init__", lambda self, *a, **kw: None):
            loop = AgentLoop.__new__(AgentLoop)
            loop.patch_policy = MagicMock()
            loop.patch_policy.is_llm_repair_allowed.return_value = True
            loop.patch_bot = MagicMock()
            fix_result = MagicMock()
            fix_result.applied = True
            loop.patch_bot._apply_single_fix.return_value = fix_result
            loop.mode = "EXPLORATION"
            loop.run_id = "test"

            # First call: invalid; second call: valid
            invalid_report = MagicMock()
            invalid_report.is_valid = False
            error = MagicMock()
            error.error_type = "invalid_edge_id"
            error.edge_id = "test"
            error.context = {}
            invalid_report.errors.return_value = [error]

            valid_report = MagicMock()
            valid_report.is_valid = True

            loop.validator = MagicMock()
            loop.validator.validate.return_value = valid_report

            final = loop._attempt_dag_auto_repair(invalid_report, max_attempts=3)

            assert final.is_valid
            assert loop.patch_bot._apply_single_fix.call_count == 1
