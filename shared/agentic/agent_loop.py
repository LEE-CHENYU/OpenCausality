"""
Agent Loop Orchestrator.

Orchestrates the DataScout → ModelSmith → Estimator → Judge cycle
with exploration/confirmation mode split.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from shared.agentic.dag.parser import DAGSpec, parse_dag
from shared.agentic.dag.validator import DAGValidator, ValidationReport
from shared.agentic.queue.queue import TaskQueue, create_queue_from_dag
from shared.agentic.queue.task import LinkageTask, TaskStatus
from shared.agentic.design.registry import DesignRegistry
from shared.agentic.design.selector import DesignSelector
from shared.agentic.output.edge_card import EdgeCard, compute_credibility_score
from shared.agentic.output.system_report import SystemReport, CriticalPathSummary
from shared.agentic.governance.audit_log import AuditLog, Hashes, ResultDelta
from shared.agentic.governance.whitelist import RefinementWhitelist, Refinement
from shared.agentic.governance.stopping import StoppingCriteria
from shared.agentic.artifact_store import ArtifactStore

# New imports for comprehensive agentic system
from shared.agentic.issues.issue_ledger import IssueLedger
from shared.agentic.issues.issue_registry import IssueRegistry
from shared.agentic.issues.issue_gates import IssueGates
from shared.agentic.issues.cross_run_reducer import CrossRunReducer
from shared.agentic.identification.screen import IdentifiabilityScreen
from shared.agentic.ts_guard import TSGuard
from shared.agentic.governance.hitl_gate import HITLGate
from shared.agentic.governance.patch_policy import PatchPolicy
from shared.agentic.agents.patch_bot import PatchBot, PatchResult
from shared.agentic.agents.data_scout import DataScout
from shared.agentic.agents.paper_scout import PaperScout
from shared.engine.dynamic_loader import DynamicLoaderFactory
from shared.agentic.output.edge_card import (
    IdentificationBlock,
    CounterfactualBlock,
    PropagationRole,
)

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    """Result of a single iteration of the agent loop."""

    iteration: int
    edges_estimated: int
    patches_applied: int
    edges_requeued: list[str]
    issues_open: int
    issues_critical: int
    hitl_required: bool
    should_stop: bool
    stop_reason: str


@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop."""

    # Mode
    mode: Literal["EXPLORATION", "CONFIRMATION"] = "EXPLORATION"

    # Query mode (STRUCTURAL / REDUCED_FORM / DESCRIPTIVE)
    query_mode: str = "REDUCED_FORM"

    # Iteration limits
    max_iterations: int = 3

    # Data budget
    download_budget_mb: int = 100
    catalog_only_first: bool = True

    # Refinement whitelist
    allowed_refinements: RefinementWhitelist | None = None

    # Stopping criteria
    stopping_criteria: StoppingCriteria | None = None

    # Output
    output_dir: Path = field(default_factory=lambda: Path("outputs/agentic"))

    # Holdout period (for confirmation mode)
    holdout_period: tuple[str, str] | None = None

    # Frozen spec (for confirmation mode)
    frozen_spec_path: Path | None = None

    # Notifications
    auto_open_browser: bool = False


@dataclass
class DataCatalog:
    """Catalog of available data assets."""

    assets: dict[str, 'DataAsset'] = field(default_factory=dict)

    def add(self, asset: 'DataAsset') -> None:
        self.assets[asset.node_id] = asset

    def get(self, node_id: str) -> 'DataAsset | None':
        return self.assets.get(node_id)

    def is_available(self, node_id: str) -> bool:
        asset = self.assets.get(node_id)
        return asset is not None and asset.is_fetched


@dataclass
class DataAsset:
    """Metadata about a data asset."""

    node_id: str
    connector: str
    dataset: str
    is_available: bool = False  # Exists in source
    is_fetched: bool = False    # Downloaded locally
    row_count: int | None = None
    date_range: tuple[str, str] | None = None
    data_ref: str | None = None  # Path or cache key


class AgentLoop:
    """
    Orchestrates the full DAG estimation process.

    Phases:
    1. DataScout: Catalog data availability, fetch on demand
    2. ModelSmith: Select designs for each edge
    3. Estimator: Run models, produce EdgeCards
    4. Judge: Evaluate credibility, propose refinements
    5. Iterate (in EXPLORATION mode only)
    """

    def __init__(
        self,
        dag: DAGSpec,
        config: AgentLoopConfig | None = None,
    ):
        """
        Initialize agent loop.

        Args:
            dag: The DAG specification
            config: Loop configuration
        """
        self.dag = dag
        self.config = config or AgentLoopConfig()

        # Run identification
        self.run_id = str(uuid.uuid4())[:8]
        self.mode = self.config.mode
        self.iteration = 0

        # Query mode
        from shared.agentic.query_mode import QueryModeConfig, QueryMode
        self.query_mode_config = QueryModeConfig.load()
        self.query_mode = QueryMode(self.config.query_mode)
        self.query_mode_spec = self.query_mode_config.get_spec(self.query_mode)

        # Core components
        self.queue = create_queue_from_dag(dag)
        self.validator = DAGValidator(dag)
        self.design_registry = DesignRegistry()
        self.design_selector = DesignSelector(self.design_registry)

        # Governance
        self.whitelist = self.config.allowed_refinements or RefinementWhitelist()
        self.stopping_criteria = self.config.stopping_criteria or StoppingCriteria()
        self.audit_log = AuditLog(
            whitelist=self.whitelist,
            log_path=self.config.output_dir / "ledger" / f"{self.run_id}.jsonl",
        )

        # Storage
        self.artifact_store = ArtifactStore(self.config.output_dir / "cards")

        # Issue tracking system
        self.issue_ledger = IssueLedger(
            run_id=self.run_id,
            output_dir=self.config.output_dir / "issues",
        )
        self.issue_registry = IssueRegistry.load()
        self.issue_gates = IssueGates()

        # Identification screen
        self.id_screen = IdentifiabilityScreen()

        # TSGuard
        self.ts_guard = TSGuard()

        # HITL gate
        self.hitl_gate = HITLGate()

        # PatchPolicy and PatchBot
        self.patch_policy = PatchPolicy.load()
        self.patch_bot = PatchBot(self.patch_policy, artifact_store=self.artifact_store)

        # Dynamic loader factory
        self.dynamic_loader = DynamicLoaderFactory()

        # DataScout (real data download)
        self.data_scout = DataScout(
            budget_mb=self.config.download_budget_mb,
            output_dir=self.config.output_dir / "cards" / "data",
        )

        # PaperScout (literature search — multi-source)
        from config.settings import get_settings
        _settings = get_settings()
        self.paper_scout = PaperScout(
            output_dir=self.config.output_dir / "citations",
            s2_api_key=_settings.semantic_scholar_api_key or None,
            openalex_mailto=_settings.openalex_mailto or None,
            unpaywall_email=_settings.unpaywall_email or None,
            core_api_key=_settings.core_api_key or None,
        )
        self.citation_bundles: dict[str, Any] = {}

        # Notifier
        from shared.agentic.governance.notifier import Notifier
        self.notifier = Notifier(
            output_dir=self.config.output_dir,
            auto_open=self.config.auto_open_browser,
        )

        # Cross-run state
        self.cross_run_reducer = CrossRunReducer(
            issues_dir=self.config.output_dir / "issues",
        )

        # State
        self.catalog = DataCatalog()
        self.validation_report: ValidationReport | None = None
        self.system_report: SystemReport | None = None
        self.id_results: dict[str, Any] = {}  # edge_id -> IdentifiabilityResult
        self.ts_guard_results: dict[str, Any] = {}  # edge_id -> TSGuardResult

        # Computed hashes
        self._dag_hash = dag.compute_hash()
        self._data_hash = ""

    def run(self) -> SystemReport:
        """
        Run the complete agent loop.

        In EXPLORATION mode, iterates: Estimate → Detect Issues → Fix →
        Re-queue (only when re-estimation required) → Re-estimate →
        Check convergence → Repeat (up to max_iterations).

        In CONFIRMATION mode, executes a single pass (no iteration).

        Returns:
            SystemReport with all results
        """
        logger.info("=" * 60)
        logger.info(f"AGENT LOOP: {self.dag.metadata.name}")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Query Mode: {self.query_mode.value}")
        logger.info(f"Max iterations: {self.config.max_iterations}")
        logger.info("=" * 60)

        # Set audit context
        self.audit_log.set_run_context(
            run_id=self.run_id,
            dag_hash=self._dag_hash,
            data_hash="",  # Will be updated after data fetch
            mode=self.mode,
        )

        # Phase 0: Validate DAG
        self.validation_report = self.validator.validate()
        if not self.validation_report.is_valid:
            logger.error("DAG validation failed")
            logger.error(self.validation_report.summary())
            return self._create_error_report("DAG validation failed")

        # Phase 1: DataScout
        self._run_data_scout()

        # Phase 1.5: PaperScout (literature search, non-blocking)
        self._run_paper_scout()

        # Iteration loop
        for self.iteration in range(self.config.max_iterations):
            result = self._run_iteration()
            logger.info(
                f"Iteration {self.iteration}: {result.edges_estimated} estimated, "
                f"{result.patches_applied} patches, "
                f"{len(result.edges_requeued)} re-queued"
            )
            if result.should_stop:
                logger.info(f"Stopping: {result.stop_reason}")
                break

        # Post-loop: HITL gate (produce checklist if needed)
        edge_cards = self.artifact_store.get_all_edge_cards()
        edge_card_dict = {c.edge_id: c for c in edge_cards}
        hitl_checklist = self.hitl_gate.evaluate(
            edge_cards=edge_card_dict,
            ledger=self.issue_ledger,
            ts_guard_results=self.ts_guard_results,
            run_id=self.run_id,
        )
        if hitl_checklist.pending_count > 0:
            logger.warning(f"HITL checklist has {hitl_checklist.pending_count} pending items")
            checklist_path = self.config.output_dir / "hitl_checklist.md"
            checklist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checklist_path, "w") as f:
                f.write(hitl_checklist.to_markdown())
            logger.info(f"HITL checklist written to {checklist_path}")

            # Build HITL panel and notify
            panel_path = None
            try:
                from scripts.build_hitl_panel import build
                panel_path = build(
                    state_path=self.config.output_dir / "issues" / "state.json",
                    cards_dir=self.config.output_dir / "cards" / "edge_cards",
                    actions_path=Path("config/agentic/hitl_actions.yaml"),
                    registry_path=Path("config/agentic/issue_registry.yaml"),
                    output_dir=self.config.output_dir,
                )
            except Exception as e:
                logger.warning(f"HITL panel build failed (non-blocking): {e}")

            self.notifier.notify_hitl_required(
                hitl_checklist,
                panel_path=panel_path,
                run_id=self.run_id,
            )

        # Flush issues and update cross-run state
        self.issue_ledger.flush()
        self.cross_run_reducer.reduce_incremental(
            self.run_id, self.issue_ledger.issues,
        )

        # Create system report
        self.system_report = self._create_system_report()

        # Notify run completion
        self.notifier.notify_run_complete(self.system_report, run_id=self.run_id)

        logger.info("=" * 60)
        logger.info("AGENT LOOP COMPLETE")
        logger.info(f"Total iterations: {self.iteration + 1}")
        logger.info(self.issue_ledger.summary())
        logger.info(self.system_report.summary())
        logger.info("=" * 60)

        return self.system_report

    def _run_iteration(self) -> IterationResult:
        """
        Execute a single iteration of the agent loop.

        Phases per iteration:
        2. Process ready tasks (estimate/re-estimate)
        3. Post-run issue detection
        4. Gate evaluation
        5. HITL check
        6. PatchBot applies auto-fixes, re-queues affected edges
        7. Convergence check

        Returns:
            IterationResult with iteration outcome
        """
        logger.info(f"--- Iteration {self.iteration} ---")

        # Phase 2: Process ready tasks (estimate / re-estimate)
        edges_estimated = 0
        while not self.queue.is_complete():
            ready_before = len(self.queue.get_ready())
            if ready_before == 0:
                break
            self._process_ready_tasks()
            edges_estimated += ready_before

        # Phase 3: Post-run issue detection
        edge_cards = self.artifact_store.get_all_edge_cards()
        edge_card_dict = {c.edge_id: c for c in edge_cards}
        self.issue_registry.detect_post_run_issues(edge_card_dict, self.issue_ledger)

        # Phase 4: Gate evaluation
        gate_eval = self.issue_gates.evaluate(
            self.issue_ledger, self.mode,
            query_mode=self.query_mode.value,
        )
        logger.info(gate_eval.summary())

        open_issues = self.issue_ledger.get_open_issues()
        critical_issues = self.issue_ledger.get_critical_open()

        # Phase 5: HITL check — if requires_human issues exist, stop iterating
        hitl_required = self.issue_ledger.has_human_required()

        # CONFIRMATION mode: always single-pass
        if self.mode == "CONFIRMATION":
            return IterationResult(
                iteration=self.iteration,
                edges_estimated=edges_estimated,
                patches_applied=0,
                edges_requeued=[],
                issues_open=len(open_issues),
                issues_critical=len(critical_issues),
                hitl_required=hitl_required,
                should_stop=True,
                stop_reason="CONFIRMATION mode: single pass only",
            )

        # Phase 6: PatchBot applies auto-fixes
        patches_applied = 0
        requeued: list[str] = []
        if gate_eval.auto_fixable_count > 0:
            patch_results = self.patch_bot.apply_fixes(self.issue_ledger, self.mode)
            for pr in patch_results:
                logger.info(
                    f"PatchBot: {pr.action} -> "
                    f"{'applied' if pr.applied else 'rejected'} "
                    f"reest={pr.requires_reestimation}: {pr.message}"
                )
                if pr.applied:
                    patches_applied += 1
                    if pr.requires_reestimation:
                        for edge_id in pr.affected_edges:
                            if self.queue.reset_task(edge_id, reason=f"patch:{pr.action}"):
                                requeued.append(edge_id)
                                logger.info(f"Re-queued {edge_id} for re-estimation (patch:{pr.action})")

            # Increment iteration counter in audit log when patches applied
            if patches_applied > 0:
                self.audit_log.increment_iteration()

        # Phase 7: Convergence check
        # STOP if HITL issues require human input
        if hitl_required:
            return IterationResult(
                iteration=self.iteration,
                edges_estimated=edges_estimated,
                patches_applied=patches_applied,
                edges_requeued=requeued,
                issues_open=len(open_issues),
                issues_critical=len(critical_issues),
                hitl_required=True,
                should_stop=True,
                stop_reason="HITL required: issues need human input",
            )

        # STOP if no tasks were re-queued AND no auto-fixable issues remain
        remaining_auto_fixable = len(self.issue_ledger.get_auto_fixable())
        if not requeued and remaining_auto_fixable == 0:
            return IterationResult(
                iteration=self.iteration,
                edges_estimated=edges_estimated,
                patches_applied=patches_applied,
                edges_requeued=requeued,
                issues_open=len(open_issues),
                issues_critical=len(critical_issues),
                hitl_required=False,
                should_stop=True,
                stop_reason="Converged: no re-queued tasks and no auto-fixable issues",
            )

        # Log credibility delta for observability (NOT used as optimization target)
        prev_scores = self.queue.get_prev_credibility_scores()
        if prev_scores:
            delta_info = {
                eid: f"{prev:.2f} -> re-queued"
                for eid, prev in prev_scores.items()
                if eid in requeued
            }
            logger.info(f"Credibility delta (informational): {delta_info}")

        # Continue to next iteration
        return IterationResult(
            iteration=self.iteration,
            edges_estimated=edges_estimated,
            patches_applied=patches_applied,
            edges_requeued=requeued,
            issues_open=len(open_issues),
            issues_critical=len(critical_issues),
            hitl_required=False,
            should_stop=False,
            stop_reason="",
        )

    def run_confirmation(
        self,
        frozen_spec_path: Path,
        holdout_period: tuple[str, str],
    ) -> SystemReport:
        """
        Run confirmation mode with frozen spec on holdout data.

        Args:
            frozen_spec_path: Path to frozen specification
            holdout_period: Start and end of holdout period

        Returns:
            SystemReport
        """
        # Check if CONFIRMATION is allowed (no CRITICAL open issues)
        ready, msg = self.issue_gates.check_confirmation_ready(self.issue_ledger)
        if not ready:
            logger.error(f"Cannot enter CONFIRMATION mode: {msg}")
            return self._create_error_report(f"CONFIRMATION blocked: {msg}")

        # Freeze validation: verify state matches exploration manifest
        from shared.agentic.governance.freeze_validator import FreezeValidator, FreezeManifest

        freeze_validator = FreezeValidator()
        try:
            manifest = FreezeManifest.load(frozen_spec_path)
            violations = freeze_validator.validate_before_confirmation(self, manifest)
            if violations:
                for v in violations:
                    logger.error(f"Freeze violation: {v}")
                return self._create_error_report(
                    f"CONFIRMATION blocked: {len(violations)} freeze violation(s). "
                    f"First: {violations[0]}"
                )
            logger.info("Freeze validation passed: state matches exploration manifest")
        except FileNotFoundError:
            logger.error(f"Freeze manifest not found: {frozen_spec_path}")
            return self._create_error_report(
                f"CONFIRMATION blocked: freeze manifest not found at {frozen_spec_path}"
            )

        self.mode = "CONFIRMATION"
        self.config.holdout_period = holdout_period
        self.config.frozen_spec_path = frozen_spec_path

        logger.info(f"CONFIRMATION RUN: holdout {holdout_period[0]} to {holdout_period[1]}")

        return self.run()

    def _run_data_scout(self) -> None:
        """Run DataScout phase: catalog and fetch data."""
        logger.info("Phase 1: DataScout")

        # Pass 1: Catalog only (check availability without downloading)
        if self.config.catalog_only_first:
            self._catalog_data()

        # Auto-populate missing loaders from DAG source specs
        populate_results = self.dynamic_loader.auto_populate_from_dag(self.dag)
        registered_count = sum(1 for v in populate_results.values() if v == "registered")
        if registered_count > 0:
            logger.info(f"DynamicLoader: auto-registered {registered_count} edges")
            self._catalog_data()  # Re-catalog with new loaders

        # DataScout: download missing node data
        missing_nodes = [
            node.id for node in self.dag.nodes
            if not self.catalog.is_available(node.id)
            and getattr(node, "observed", True)
            and not getattr(node, "latent", False)
        ]
        if missing_nodes:
            logger.info(f"DataScout: {len(missing_nodes)} nodes need data")
            scout_report = self.data_scout.download_missing(self.dag, missing_nodes)
            logger.info(
                f"DataScout: downloaded {scout_report.downloaded}, "
                f"skipped {scout_report.skipped}, failed {scout_report.failed}"
            )
            if scout_report.downloaded > 0:
                self._catalog_data()  # Re-catalog with newly downloaded data

        # Mark data available in queue
        for node in self.dag.nodes:
            asset = self.catalog.get(node.id)
            if asset and asset.is_fetched:
                self.queue.mark_data_available(node.id, asset.data_ref or "")

        # Compute data hash
        self._data_hash = self._compute_data_hash()
        self.audit_log.data_hash = self._data_hash

    def _run_paper_scout(self) -> None:
        """Run PaperScout: search literature for all edges (non-blocking)."""
        logger.info("Phase 1.5: PaperScout")
        try:
            self.citation_bundles = self.paper_scout.search_all_edges(self.dag)
            self.paper_scout.save_bundles()
            logger.info(f"PaperScout: citations for {len(self.citation_bundles)} edges")
        except Exception as e:
            logger.warning(f"PaperScout failed (non-blocking): {e}")

    def _catalog_data(self) -> None:
        """Catalog data availability by probing actual data sources."""
        from shared.engine.data_assembler import NODE_LOADERS

        for node in self.dag.nodes:
            connector = ""
            dataset = ""
            if node.source and node.source.preferred:
                connector = node.source.preferred[0].connector
                dataset = node.source.preferred[0].dataset

            # Check if we have a real loader for this node
            has_loader = node.id in NODE_LOADERS
            is_fetched = False
            row_count = None
            date_range = None
            data_ref = None

            if has_loader:
                try:
                    s = NODE_LOADERS[node.id]()
                    is_fetched = True
                    row_count = len(s)
                    if hasattr(s, 'index') and len(s) > 0:
                        date_range = (str(s.index.min()), str(s.index.max()))
                    data_ref = f"loader:{node.id}"
                except Exception as e:
                    logger.warning(f"Data probe failed for {node.id}: {e}")
                    is_fetched = False
            else:
                # No loader; mark as available but not fetched if latent
                is_fetched = getattr(node, 'observed', True)

            asset = DataAsset(
                node_id=node.id,
                connector=connector,
                dataset=dataset,
                is_available=has_loader or is_fetched,
                is_fetched=is_fetched,
                row_count=row_count,
                date_range=date_range,
                data_ref=data_ref,
            )
            self.catalog.add(asset)

    def _compute_data_hash(self) -> str:
        """Compute hash of data catalog."""
        content = json.dumps({
            node_id: {
                "connector": asset.connector,
                "dataset": asset.dataset,
                "is_fetched": asset.is_fetched,
            }
            for node_id, asset in self.catalog.assets.items()
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _process_ready_tasks(self) -> None:
        """Process all ready tasks."""
        ready_tasks = self.queue.get_ready()

        if not ready_tasks:
            # No ready tasks - check if blocked
            stats = self.queue.stats()
            if stats.blocked > 0 or stats.running > 0:
                logger.info(f"Waiting: {stats.blocked} blocked, {stats.running} running")
            return

        for task in ready_tasks:
            self._process_task(task)

    def _process_task(self, task: LinkageTask) -> None:
        """Process a single task using real estimators."""
        from shared.engine.data_assembler import get_edge_group, EDGE_NODE_MAP

        logger.info(f"Processing task: {task.edge_id}")
        task.start()
        self.queue.update_status(task.edge_id, TaskStatus.RUNNING)

        try:
            # Get edge spec
            edge = self.dag.get_edge(task.edge_id)
            if not edge:
                raise ValueError(f"Edge not found: {task.edge_id}")

            # Determine edge group and dispatch to appropriate estimator
            group = get_edge_group(task.edge_id)

            if group == "IMMUTABLE":
                edge_card = self._create_immutable_card(task)
            elif group == "IDENTITY":
                edge_card = self._create_identity_card(task)
            elif group in ("MONTHLY_LP", "QUARTERLY_LP"):
                edge_card = self._create_lp_card(task, is_quarterly=(group == "QUARTERLY_LP"))
            elif group == "DYNAMIC_LP":
                edge_card = self._create_lp_card(task, is_quarterly=False)
            else:
                # Fallback: try LP estimation if edge is in EDGE_NODE_MAP
                if task.edge_id in EDGE_NODE_MAP:
                    edge_card = self._create_lp_card(task, is_quarterly=False)
                else:
                    self.queue.mark_failed(
                        task.edge_id,
                        f"No data loader registered for edge '{task.edge_id}'. "
                        f"Add to EDGE_NODE_MAP or register a NODE_LOADER."
                    )
                    logger.warning(
                        f"Edge {task.edge_id}: BLOCKED — no data loader. "
                        f"Skipping estimation to avoid placeholder results."
                    )
                    return

            # Run identifiability screen (post-estimation)
            design_name = edge_card.spec_details.design if edge_card.spec_details else ""
            id_result = self.id_screen.screen_post_estimation(
                edge_id=task.edge_id,
                design=design_name,
                diagnostics=edge_card.diagnostics,
                ts_guard_result=self.ts_guard_results.get(task.edge_id),
            )
            self.id_results[task.edge_id] = id_result

            # Attach identification block to card
            edge_card.identification = IdentificationBlock(
                claim_level=id_result.claim_level,
                risks=id_result.risks,
                untestable_assumptions=id_result.untestable_assumptions,
                testable_threats_passed=id_result.testable_threats_passed,
                testable_threats_failed=id_result.testable_threats_failed,
            )

            # Split counterfactual block (mode-aware)
            from shared.agentic.query_mode import (
                derive_propagation_role,
                is_edge_allowed_for_propagation,
                is_shock_cf_allowed,
                is_policy_cf_allowed,
            )

            shock_ok, shock_reason = is_shock_cf_allowed(
                id_result.claim_level, self.query_mode_spec,
            )
            policy_ok, policy_reason = is_policy_cf_allowed(
                id_result.claim_level, self.query_mode_spec,
            )
            edge_card.counterfactual_block = CounterfactualBlock(
                shock_scenario_allowed=shock_ok and id_result.shock_scenario_allowed,
                policy_intervention_allowed=policy_ok and id_result.policy_intervention_allowed,
                reason_shock_blocked=shock_reason or id_result.reason_shock_blocked,
                reason_policy_blocked=policy_reason or id_result.reason_policy_blocked,
            )

            # Populate propagation role with per-mode permissions
            edge_type = edge.get_edge_type() if hasattr(edge, 'get_edge_type') else "causal"
            role = derive_propagation_role(edge_type, id_result.claim_level)

            mode_prop, mode_shock, mode_policy = {}, {}, {}
            for mn, ms in self.query_mode_config.modes.items():
                mode_prop[mn] = is_edge_allowed_for_propagation(role, ms)
                s_ok, _ = is_shock_cf_allowed(id_result.claim_level, ms)
                p_ok, _ = is_policy_cf_allowed(id_result.claim_level, ms)
                mode_shock[mn] = s_ok
                mode_policy[mn] = p_ok

            edge_card.propagation_role = PropagationRole(
                role=role,
                mode_propagation_allowed=mode_prop,
                mode_shock_cf_allowed=mode_shock,
                mode_policy_cf_allowed=mode_policy,
                selected_for_counterfactual=mode_shock.get(self.query_mode.value, False),
            )

            # Attach literature citations if available
            bundle = self.citation_bundles.get(task.edge_id)
            if bundle and bundle.citations:
                from shared.agentic.output.edge_card import LiteratureBlock
                supporting = [c.to_dict() for c in bundle.citations if c.relevance == "supporting"]
                challenging = [c.to_dict() for c in bundle.citations if c.relevance == "challenging"]
                methodological = [c.to_dict() for c in bundle.citations if c.relevance == "methodological"]
                edge_card.literature = LiteratureBlock(
                    supporting=supporting,
                    challenging=challenging,
                    methodological=methodological,
                    search_status=self.paper_scout.search_status.get(task.edge_id, "PENDING"),
                    search_timestamp=datetime.now().isoformat(),
                    search_query=bundle.citations[0].search_query if bundle.citations else "",
                    total_results=len(bundle.citations),
                )

            # Validate edge card (log-only, non-blocking)
            from shared.agentic.validation import validate_edge_card, ValidationSeverity
            card_validation = validate_edge_card(edge_card)
            for issue in card_validation.issues:
                if issue.severity == ValidationSeverity.ERROR:
                    logger.error(f"EdgeCard validation [{edge_card.edge_id}]: {issue.check_id}: {issue.message}")
                else:
                    logger.warning(f"EdgeCard validation [{edge_card.edge_id}]: {issue.check_id}: {issue.message}")

            # Log initial specification
            self.audit_log.log_initial(task.edge_id, edge_card.spec_hash)

            # Save artifact
            results_ref = str(self.artifact_store.save_edge_card(edge_card))

            # Update queue
            self.queue.mark_complete(
                task.edge_id,
                results_ref,
                edge_card.credibility_score,
                edge_card.credibility_rating,
            )

            # Print per-edge risk block
            self._print_edge_risk_block(task.edge_id, edge_card, id_result)

        except Exception as e:
            logger.error(f"Task {task.edge_id} failed: {e}")
            self.queue.mark_failed(task.edge_id, str(e))

    def _create_immutable_card(self, task: LinkageTask) -> EdgeCard:
        """Create EdgeCard from validated evidence (immutable blocks)."""
        from shared.engine.ts_estimator import get_immutable_result
        from shared.agentic.output.edge_card import (
            Estimates, DiagnosticResult, Interpretation,
            FailureFlags, CounterfactualApplicability,
        )
        from shared.agentic.output.provenance import SpecDetails

        ir = get_immutable_result(task.edge_id)

        estimates = Estimates(
            point=ir.point_estimate,
            se=ir.se,
            ci_95=(ir.ci_lower, ir.ci_upper),
            pvalue=ir.pvalue,
        )
        diagnostics = {
            "validated_evidence": DiagnosticResult(
                name="validated_evidence", passed=True, value=1.0,
                message=f"Validated from {ir.source_block}",
            ),
        }
        spec_details = SpecDetails(
            design="IMMUTABLE_EVIDENCE",
            se_method="from_source_block",
        )
        score, rating = compute_credibility_score(
            diagnostics=diagnostics,
            failure_flags=FailureFlags(),
            design_weight=0.9,
            data_coverage=1.0,
        )
        return EdgeCard(
            edge_id=task.edge_id,
            dag_version_hash=self._dag_hash,
            spec_hash=spec_details.compute_hash(),
            spec_details=spec_details,
            estimates=estimates,
            diagnostics=diagnostics,
            interpretation=Interpretation(
                estimand=ir.source_description,
                is_not="Re-estimated result; locked validated evidence",
            ),
            failure_flags=FailureFlags(),
            counterfactual=CounterfactualApplicability(
                supports_shock_path=True,
                supports_policy_intervention=False,
                intervention_note=f"Validated evidence from {ir.source_block}",
            ),
            credibility_rating=rating,
            credibility_score=score,
            is_precisely_null=(ir.point_estimate == 0.0),
            null_equivalence_bound=float(2 * ir.se) if ir.point_estimate == 0.0 else None,
        )

    def _create_identity_card(self, task: LinkageTask) -> EdgeCard:
        """Create EdgeCard from identity (mechanical) sensitivity."""
        from shared.engine.ts_estimator import compute_identity_sensitivity
        from shared.engine.data_assembler import _load_kspi_quarterly
        from shared.agentic.output.edge_card import (
            Estimates, DiagnosticResult, Interpretation,
            FailureFlags, CounterfactualApplicability,
        )
        from shared.agentic.output.provenance import SpecDetails

        kspi_df = _load_kspi_quarterly()
        latest = kspi_df.iloc[-1]
        capital = float(latest["total_capital"])
        rwa = float(latest["rwa"])

        id_results = compute_identity_sensitivity(capital, rwa)
        ir = id_results[task.edge_id]

        estimates = Estimates(
            point=ir.sensitivity,
            se=0.0,
            ci_95=(ir.sensitivity, ir.sensitivity),
            pvalue=None,
        )
        diagnostics = {
            "identity_check": DiagnosticResult(
                name="identity_check", passed=True,
                value=ir.sensitivity,
                message=f"Deterministic: {ir.formula}",
            ),
        }
        failure_flags = FailureFlags(mechanical_identity_risk=True)
        spec_details = SpecDetails(design="IDENTITY", se_method="deterministic")

        score, rating = compute_credibility_score(
            diagnostics=diagnostics,
            failure_flags=failure_flags,
            design_weight=1.0,
            data_coverage=1.0,
        )
        return EdgeCard(
            edge_id=task.edge_id,
            dag_version_hash=self._dag_hash,
            spec_hash=spec_details.compute_hash(),
            spec_details=spec_details,
            estimates=estimates,
            diagnostics=diagnostics,
            interpretation=Interpretation(
                estimand=f"Sensitivity: {ir.formula}",
                is_not="Causal effect; this is a mechanical identity",
            ),
            failure_flags=failure_flags,
            counterfactual=CounterfactualApplicability(
                supports_shock_path=True,
                supports_policy_intervention=True,
                intervention_note="Mechanical identity; always holds by definition",
            ),
            credibility_rating=rating,
            credibility_score=score,
        )

    def _create_lp_card(
        self,
        task: LinkageTask,
        is_quarterly: bool = False,
    ) -> EdgeCard:
        """Create EdgeCard from Local Projections estimation."""
        import numpy as np_
        from shared.engine.data_assembler import assemble_edge_data, EDGE_NODE_MAP
        from shared.engine.ts_estimator import estimate_lp, check_sign_consistency
        from shared.agentic.output.edge_card import (
            Estimates, DiagnosticResult, Interpretation,
            FailureFlags, CounterfactualApplicability,
        )
        from shared.agentic.output.provenance import SpecDetails, DataProvenance, SourceProvenance

        data = assemble_edge_data(task.edge_id)
        max_h = 2 if is_quarterly else 6
        n_lags = 1 if is_quarterly else 2

        lp = estimate_lp(
            y=data["outcome"],
            x=data["treatment"],
            max_horizon=max_h,
            n_lags=n_lags,
            edge_id=task.edge_id,
        )

        # Run TSGuard validation on the LP result
        try:
            # Get break_dates from edge spec if available
            edge = self.dag.get_edge(task.edge_id)
            break_dates = None
            if edge and edge.acceptance_criteria.stability.regime_split_date:
                break_dates = [edge.acceptance_criteria.stability.regime_split_date]

            ts_result = self.ts_guard.validate(
                edge_id=task.edge_id,
                y=data["outcome"],
                x=data["treatment"],
                lp_result=lp,
                break_dates=break_dates,
            )
            self.ts_guard_results[task.edge_id] = ts_result
        except Exception as e:
            logger.warning(f"TSGuard for {task.edge_id}: {e}")

        point = lp.impact_estimate
        se = lp.impact_se
        n_obs = lp.nobs[0] if lp.nobs else 0
        ci_lo = point - 1.96 * se if not np_.isnan(se) else 0.0
        ci_hi = point + 1.96 * se if not np_.isnan(se) else 0.0
        pval = lp.pvalues[0] if lp.pvalues and not np_.isnan(lp.pvalues[0]) else None

        estimates = Estimates(
            point=float(point) if not np_.isnan(point) else 0.0,
            se=float(se) if not np_.isnan(se) else 0.0,
            ci_95=(float(ci_lo), float(ci_hi)),
            pvalue=float(pval) if pval is not None else None,
            horizons=lp.horizons,
            irf=lp.coefficients,
            irf_ci_lower=lp.ci_lower,
            irf_ci_upper=lp.ci_upper,
        )

        diagnostics: dict[str, DiagnosticResult] = {}
        if lp.hac_bandwidth:
            diagnostics["hac_bandwidth"] = DiagnosticResult(
                name="hac_bandwidth", passed=True,
                value=float(lp.hac_bandwidth[0]),
                message=f"Newey-West bandwidth: {lp.hac_bandwidth[0]}",
            )
        diagnostics["effective_obs"] = DiagnosticResult(
            name="effective_obs", passed=n_obs >= 20,
            value=float(n_obs), threshold=20.0,
        )
        sign_ok, sign_msg = check_sign_consistency(lp)
        diagnostics["sign_consistency"] = DiagnosticResult(
            name="sign_consistency", passed=sign_ok, message=sign_msg,
        )

        # Merge TSGuard diagnostics into the edge card diagnostics
        ts_result = self.ts_guard_results.get(task.edge_id)
        regime_break = False
        if ts_result:
            for diag_name, diag_status in ts_result.diagnostics_results.items():
                if diag_status != "not_run":
                    diagnostics[f"ts_{diag_name}"] = DiagnosticResult(
                        name=f"ts_{diag_name}",
                        passed=(diag_status == "pass"),
                        message=f"TSGuard: {diag_status}",
                    )
            regime_break = ts_result.diagnostics_results.get("regime_stability") == "fail"

        failure_flags = FailureFlags(
            small_sample=is_quarterly or n_obs < 30,
            regime_break_detected=regime_break,
        )
        treatment_node, outcome_node = EDGE_NODE_MAP[task.edge_id]
        spec_details = SpecDetails(
            design="LOCAL_PROJECTIONS",
            controls=[f"y_lag{i}" for i in range(1, n_lags + 1)] + [f"x_lag{i}" for i in range(1, n_lags + 1)],
            se_method="HAC_newey_west",
            horizon=lp.horizons,
        )
        design_weight = 0.5 if is_quarterly else 0.7
        data_coverage = min(1.0, n_obs / 17) if is_quarterly else min(1.0, n_obs / 100)
        score, rating = compute_credibility_score(
            diagnostics=diagnostics, failure_flags=failure_flags,
            design_weight=design_weight, data_coverage=data_coverage,
        )
        if is_quarterly and rating == "A":
            rating = "B"
            score = min(score, 0.79)

        # Null detection
        is_null = False
        null_bound = None
        if pval is not None and pval > 0.10 and abs(point) < 2 * se:
            is_null = True
            null_bound = float(2 * se)

        date_range = (str(data.index.min().date()), str(data.index.max().date())) if len(data) > 0 else None
        provenance = DataProvenance(
            treatment_source=SourceProvenance(connector="parquet_file", dataset=treatment_node, row_count=len(data), date_range=date_range),
            outcome_source=SourceProvenance(connector="parquet_file", dataset=outcome_node, row_count=len(data), date_range=date_range),
            combined_row_count=len(data),
            combined_date_range=date_range,
        )

        return EdgeCard(
            edge_id=task.edge_id,
            dag_version_hash=self._dag_hash,
            spec_hash=spec_details.compute_hash(),
            spec_details=spec_details,
            data_provenance=provenance,
            estimates=estimates,
            diagnostics=diagnostics,
            interpretation=Interpretation(
                estimand=f"IRF of {outcome_node} to a unit shock in {treatment_node}",
                is_not="Structural causal effect under all interventions",
                channels=["direct", "indirect"],
            ),
            failure_flags=failure_flags,
            counterfactual=CounterfactualApplicability(
                supports_shock_path=True,
                supports_policy_intervention=False,
                intervention_note="Reduced-form LP estimate",
            ),
            credibility_rating=rating,
            credibility_score=score,
            is_precisely_null=is_null,
            null_equivalence_bound=null_bound,
        )

    def _create_system_report(self) -> SystemReport:
        """Create system report from current state."""
        edge_cards = self.artifact_store.get_all_edge_cards()

        report = SystemReport(
            dag_name=self.dag.metadata.name,
            dag_version_hash=self._dag_hash,
            mode=self.mode,
            iteration=self.iteration,
        )

        # Add edge cards
        for card in edge_cards:
            task = self.queue.get(card.edge_id)
            status = task.status.value if task else "UNKNOWN"
            report.add_edge_card(card, status)

        # Add blocked edges
        for edge_id, reason in self.queue.get_blocking_issues().items():
            report.add_blocked_edge(edge_id, reason)

        # Compute critical path
        critical_tasks = self.queue.get_critical_tasks()
        critical_ids = [t.edge_id for t in critical_tasks]
        critical_scores = [
            report.edge_summaries[i].credibility_score
            for i, s in enumerate(report.edge_summaries)
            if s.edge_id in critical_ids
        ]

        report.critical_path = CriticalPathSummary(
            target_node=self.dag.metadata.target_node,
            path_edges=critical_ids,
            path_complete=all(t.is_done() for t in critical_tasks),
            min_credibility=min(critical_scores) if critical_scores else 0.0,
            blocking_edges=[t.edge_id for t in critical_tasks if t.is_blocked()],
        )

        report.total_iterations = self.iteration
        report.refinements_applied = self.audit_log.count_refinements()

        return report

    def _print_edge_risk_block(
        self,
        edge_id: str,
        card: EdgeCard,
        id_result: Any,
    ) -> None:
        """Print per-edge risk block after estimation (CLI reminder)."""
        lines = [
            "",
            "\u2501" * 55,
            f"Edge: {edge_id}",
            "\u2501" * 55,
            f"Claim Level: {id_result.claim_level}",
        ]

        # Identification risks
        if id_result.risks:
            lines.append("Identification Risks:")
            for risk, level in id_result.risks.items():
                if level in ("medium", "high"):
                    lines.append(f"  - {risk}: {level.upper()}")

        # TS dynamics risks
        ts_result = self.ts_guard_results.get(edge_id)
        if ts_result:
            lines.append("TS Dynamics Risks:")
            for risk, level in ts_result.dynamics_risk.items():
                if level in ("medium", "high"):
                    lines.append(f"  - {risk.replace('_risk', '')}: {level.upper()}")

        # Diagnostics summary
        if card.diagnostics:
            lines.append("Diagnostics:")
            for name, diag in card.diagnostics.items():
                status = "PASS" if diag.passed else "FAIL"
                msg = f" ({diag.message})" if diag.message and not diag.passed else ""
                lines.append(f"  - {name}: {status}{msg}")

        # Counterfactual status
        cf_allowed = id_result.counterfactual_allowed
        lines.append(f"Counterfactual Use: {'ALLOWED' if cf_allowed else 'BLOCKED'}")
        if not cf_allowed and id_result.counterfactual_reason_blocked:
            lines.append(f"Reason: {id_result.counterfactual_reason_blocked}")

        lines.append("")
        lines.append("Even if p<0.05, this does not establish a causal effect.")
        lines.append("\u2501" * 55)

        for line in lines:
            logger.info(line)

    def generate_id_dashboard(self) -> str:
        """Generate the identifiability risk dashboard."""
        return self.id_screen.generate_dashboard(self.id_results)

    def _create_error_report(self, error: str) -> SystemReport:
        """Create error system report."""
        report = SystemReport(
            dag_name=self.dag.metadata.name,
            dag_version_hash=self._dag_hash,
            mode=self.mode,
            iteration=0,
        )
        report.add_blocked_edge("all", error)
        return report


def run_dag(
    dag_path: Path | str,
    output_dir: Path | str | None = None,
    mode: str = "EXPLORATION",
    query_mode: str = "REDUCED_FORM",
) -> SystemReport:
    """
    Run a DAG specification.

    Convenience function for CLI usage.

    Args:
        dag_path: Path to DAG YAML file
        output_dir: Output directory (optional)
        mode: "EXPLORATION" or "CONFIRMATION"
        query_mode: "STRUCTURAL", "REDUCED_FORM", or "DESCRIPTIVE"

    Returns:
        SystemReport
    """
    dag = parse_dag(Path(dag_path))

    config = AgentLoopConfig(
        mode=mode,
        query_mode=query_mode,
        output_dir=Path(output_dir) if output_dir else Path("outputs/agentic"),
    )

    loop = AgentLoop(dag, config)
    return loop.run()
