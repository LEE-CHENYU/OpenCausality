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
from shared.agentic.design.selector import DesignSelector, SelectedDesign, NotIdentified
from shared.agentic.design.feasibility import DataReport
from shared.agentic.output.edge_card import EdgeCard, compute_credibility_score
from shared.agentic.output.system_report import SystemReport, CriticalPathSummary
from shared.agentic.governance.audit_log import AuditLog, Hashes, ResultDelta
from shared.agentic.governance.whitelist import RefinementWhitelist, Refinement
from shared.agentic.governance.stopping import StoppingCriteria, StoppingDecision
from shared.agentic.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)


@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop."""

    # Mode
    mode: Literal["EXPLORATION", "CONFIRMATION"] = "EXPLORATION"

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

        # State
        self.catalog = DataCatalog()
        self.validation_report: ValidationReport | None = None
        self.system_report: SystemReport | None = None

        # Computed hashes
        self._dag_hash = dag.compute_hash()
        self._data_hash = ""

    def run(self) -> SystemReport:
        """
        Run the complete agent loop.

        Returns:
            SystemReport with all results
        """
        logger.info("=" * 60)
        logger.info(f"AGENT LOOP: {self.dag.metadata.name}")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Mode: {self.mode}")
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

        # Phase 2-4: Edge estimation loop
        while not self.queue.is_complete():
            self._process_ready_tasks()

        # Create system report
        self.system_report = self._create_system_report()

        # Phase 5: Judge evaluation (in EXPLORATION mode)
        if self.mode == "EXPLORATION":
            self._run_judge_evaluation()

        logger.info("=" * 60)
        logger.info("AGENT LOOP COMPLETE")
        logger.info(self.system_report.summary())
        logger.info("=" * 60)

        return self.system_report

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

        # Mark data available in queue
        for node in self.dag.nodes:
            asset = self.catalog.get(node.id)
            if asset and asset.is_fetched:
                self.queue.mark_data_available(node.id, asset.data_ref or "")

        # Compute data hash
        self._data_hash = self._compute_data_hash()
        self.audit_log.data_hash = self._data_hash

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
            else:
                # Fallback: try LP estimation if edge is in EDGE_NODE_MAP
                if task.edge_id in EDGE_NODE_MAP:
                    edge_card = self._create_lp_card(task, is_quarterly=False)
                else:
                    # Get forbidden controls from validation
                    forbidden_controls = set()
                    if self.validation_report and task.edge_id in self.validation_report.forbidden_controls:
                        forbidden_controls = self.validation_report.forbidden_controls[task.edge_id].total_forbidden

                    data_report = DataReport(
                        is_panel=True, is_time_series=True,
                        n_obs=100, n_periods=40, n_units=10,
                        treatment_type="continuous", outcome_type="continuous",
                        n_instruments=len(edge.instruments),
                    )
                    result = self.design_selector.select(
                        edge=edge, data_report=data_report,
                        forbidden_controls=forbidden_controls,
                    )
                    if isinstance(result, NotIdentified):
                        self.queue.mark_blocked(task.edge_id, result.reason, TaskStatus.BLOCKED_ID)
                        logger.warning(f"Edge {task.edge_id} not identified: {result.reason}")
                        return
                    edge_card = self._create_edge_card(task, result, edge)

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

        failure_flags = FailureFlags(small_sample=is_quarterly or n_obs < 30)
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

    def _create_edge_card(
        self,
        task: LinkageTask,
        design: SelectedDesign,
        edge: Any,
    ) -> EdgeCard:
        """Create EdgeCard from estimation results."""
        from shared.agentic.output.edge_card import (
            Estimates,
            DiagnosticResult,
            Interpretation,
            FailureFlags,
            CounterfactualApplicability,
        )
        from shared.agentic.output.provenance import SpecDetails

        # In production, actual estimation would happen here
        # For now, create a placeholder card

        spec_details = SpecDetails(
            design=design.design.id,
            controls=design.controls,
            instruments=design.instruments,
            fixed_effects=design.fixed_effects,
            se_method=design.se_method,
        )

        # Placeholder estimates (would come from actual model)
        estimates = Estimates(
            point=-0.15,
            se=0.05,
            ci_95=(-0.25, -0.05),
            pvalue=0.003,
        )

        # Placeholder diagnostics
        diagnostics = {
            "residual_checks": DiagnosticResult(
                name="residual_checks",
                passed=True,
                value=0.95,
            ),
        }

        # Interpretation
        interpretation = Interpretation(
            estimand=f"Effect of {task.treatment} on {task.outcome}",
            is_not="Universal causal effect",
            channels=["direct", "indirect"],
        )

        # Failure flags
        failure_flags = FailureFlags()

        # Counterfactual
        counterfactual = CounterfactualApplicability(
            supports_shock_path=True,
            supports_policy_intervention=False,
            intervention_note="Reduced-form estimate",
        )

        # Compute credibility
        score, rating = compute_credibility_score(
            diagnostics=diagnostics,
            failure_flags=failure_flags,
            design_weight=design.credibility_weight,
        )

        return EdgeCard(
            edge_id=task.edge_id,
            dag_version_hash=self._dag_hash,
            spec_hash=design.spec_hash,
            spec_details=spec_details,
            estimates=estimates,
            diagnostics=diagnostics,
            interpretation=interpretation,
            failure_flags=failure_flags,
            counterfactual=counterfactual,
            credibility_rating=rating,
            credibility_score=score,
        )

    def _run_judge_evaluation(self) -> None:
        """Run Judge evaluation and potentially iterate."""
        logger.info("Phase 5: Judge evaluation")

        edge_cards = self.artifact_store.get_all_edge_cards()
        scores = [c.credibility_score for c in edge_cards]

        # Check stopping criteria
        decision = self.stopping_criteria.should_stop(
            iteration=self.iteration,
            credibility_delta=0.0,  # First iteration
            critical_edges_scores=scores,
            mode=self.mode,
        )

        if decision.should_stop:
            logger.info(f"Stopping: {decision.reason}")
            return

        # Propose refinements (simplified)
        # In production, Judge agent would analyze weak edges and propose fixes

        # Check if we should continue
        self.iteration += 1
        self.audit_log.increment_iteration()

        if self.iteration < self.config.max_iterations:
            logger.info(f"Iteration {self.iteration}: checking for improvements")
            # Could re-run with refinements here

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
) -> SystemReport:
    """
    Run a DAG specification.

    Convenience function for CLI usage.

    Args:
        dag_path: Path to DAG YAML file
        output_dir: Output directory (optional)
        mode: "EXPLORATION" or "CONFIRMATION"

    Returns:
        SystemReport
    """
    dag = parse_dag(Path(dag_path))

    config = AgentLoopConfig(
        mode=mode,
        output_dir=Path(output_dir) if output_dir else Path("outputs/agentic"),
    )

    loop = AgentLoop(dag, config)
    return loop.run()
