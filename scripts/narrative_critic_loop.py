#!/usr/bin/env python3
"""
Narrative Critic Loop — Principle-Based DAG Quality Improvement.

Orchestrates an iterative loop:
  Iteration 0: generate_narrative_dag()  [existing single-pass]
  Iteration 1..N:
    - AgentLoop.run()  [estimate, detect, patch]
    - collect_feedback()  [structured from run artifacts]
    - DAGCritic.critique()  [LLM proposes revisions]
    - CriticRevisionValidator.validate()  [code-verify]
    - apply_revisions()  [update DAG YAML]
    - repair_narrative_dag()  [existing structural repair]
    - quality_score() → convergence?

Usage:
    python scripts/narrative_critic_loop.py examples/kaspi_narrative.txt
    python scripts/narrative_critic_loop.py examples/kaspi_narrative.txt --iterations 3
    python scripts/narrative_critic_loop.py examples/kaspi_narrative.txt --critic-backend api
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.agentic.agents.critic_schema import (
    CriticReport,
    CriticRevision,
    QualityScore,
    load_principles,
)
from shared.agentic.agents.dag_critic import DAGCritic

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────

@dataclass
class CriticLoopConfig:
    """Configuration for the narrative critic loop."""

    # Number of critic iterations (0 = generate only, no critic)
    max_iterations: int = 3

    # LLM backend for critique: claude-code, codex, api
    critic_backend: str = "claude-code"

    # Whether to run AgentLoop estimation between critic iterations
    run_estimation: bool = True

    # AgentLoop mode for estimation passes
    estimation_mode: str = "EXPLORATION"
    estimation_max_iterations: int = 2

    # Force continue despite DAG validation errors
    force: bool = False

    # Output directory (derived from DAG path if not specified)
    output_dir: Path | None = None

    # Convergence from principles YAML (overridable)
    min_delta: float = 0.02
    stable_iterations: int = 2


# ──────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────

class NarrativeCriticLoop:
    """Outer loop orchestrating narrative DAG generation + critic iterations."""

    def __init__(
        self,
        narrative_path: Path,
        base_dag_path: Path,
        output_dag_path: Path,
        config: CriticLoopConfig | None = None,
    ):
        self.narrative_path = narrative_path
        self.base_dag_path = base_dag_path
        self.output_dag_path = output_dag_path
        self.config = config or CriticLoopConfig()

        # Load convergence criteria from principles
        principles = load_principles()
        conv = principles.get("convergence", {})
        if self.config.min_delta == 0.02:  # default not overridden
            self.config.min_delta = conv.get("min_delta", 0.02)
        if self.config.stable_iterations == 2:
            self.config.stable_iterations = conv.get("stable_iterations", 2)

        # Derive output dir from DAG path
        if self.config.output_dir is None:
            slug = output_dag_path.stem
            self.config.output_dir = Path("outputs/agentic") / slug

    def run(self) -> CriticReport:
        """Run the full generation + critic loop.

        Returns:
            CriticReport with iteration history and final quality score.
        """
        from rich.console import Console
        console = Console()

        # ── Iteration 0: Generate narrative DAG ──────────────────
        console.print("\n[bold cyan]Iteration 0: Generating narrative DAG[/bold cyan]")
        self._generate_initial_dag()

        # Create critic
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        critic = DAGCritic(
            dag_path=self.output_dag_path,
            output_dir=output_dir,
            backend=self.config.critic_backend,
        )

        # Initial quality score
        initial_score = critic.compute_quality_score()
        score_history = [initial_score.total]
        all_applied: list[CriticRevision] = []
        all_rejected: list[CriticRevision] = []
        stable_count = 0
        convergence_reason = ""
        actual_iterations = 0

        console.print(
            f"  Initial quality score: [bold]{initial_score.total:.3f}[/bold]"
        )

        if self.config.max_iterations == 0:
            return CriticReport(
                iterations=0,
                initial_score=initial_score,
                final_score=initial_score,
                convergence_reason="max_iterations=0 (generate only)",
                score_history=score_history,
            )

        # ── Iterations 1..N ──────────────────────────────────────
        for iteration in range(1, self.config.max_iterations + 1):
            actual_iterations = iteration
            console.print(
                f"\n[bold cyan]Iteration {iteration}/{self.config.max_iterations}"
                f"[/bold cyan]"
            )

            # Run estimation if configured
            if self.config.run_estimation:
                console.print("  [dim]Running AgentLoop estimation...[/dim]")
                self._run_estimation(iteration)
                critic.reload_dag()

            # Collect feedback
            console.print("  [dim]Collecting feedback...[/dim]")
            feedback = critic.collect_feedback(iteration)
            if feedback.causal_logic_probes:
                n_warn = sum(1 for p in feedback.causal_logic_probes if p.get("severity") == "warning")
                n_info = len(feedback.causal_logic_probes) - n_warn
                console.print(f"  Causal logic probes: {n_warn} warning(s), {n_info} info")

            # Get critique from LLM
            console.print(
                f"  [dim]Getting critique (backend={self.config.critic_backend})...[/dim]"
            )
            revisions = critic.critique(feedback, iteration)
            console.print(f"  Proposed {len(revisions)} revision(s)")

            # Check for zero-revision convergence
            if not revisions:
                convergence_reason = "Zero revisions proposed"
                console.print(f"  [green]Converged: {convergence_reason}[/green]")
                score_history.append(feedback.quality_score.total)
                break

            # Validate and apply
            applied, rejected = critic.apply_revisions(revisions)
            all_applied.extend(applied)
            all_rejected.extend(rejected)
            console.print(
                f"  Applied {len(applied)}, rejected {len(rejected)} revision(s)"
            )

            # Run structural repair after applying revisions
            if applied:
                console.print("  [dim]Running structural repair...[/dim]")
                self._run_repair()
                critic.reload_dag()

            # Compute new quality score
            new_score = critic.compute_quality_score()
            score_history.append(new_score.total)
            delta = new_score.total - score_history[-2] if len(score_history) >= 2 else 0

            console.print(
                f"  Quality: {new_score.total:.3f} "
                f"(delta={delta:+.3f})"
            )

            # Check convergence
            if abs(delta) < self.config.min_delta:
                stable_count += 1
                if stable_count >= self.config.stable_iterations:
                    convergence_reason = (
                        f"Quality delta < {self.config.min_delta} "
                        f"for {self.config.stable_iterations} consecutive iterations"
                    )
                    console.print(f"  [green]Converged: {convergence_reason}[/green]")
                    break
            else:
                stable_count = 0

        if not convergence_reason:
            convergence_reason = f"Max iterations ({self.config.max_iterations}) reached"

        # Final score
        final_score = critic.compute_quality_score()

        report = CriticReport(
            iterations=actual_iterations,
            initial_score=initial_score,
            final_score=final_score,
            revisions_applied=all_applied,
            revisions_rejected=all_rejected,
            convergence_reason=convergence_reason,
            score_history=score_history,
        )

        # Write report
        self._write_report(report)

        # Print summary
        console.print("\n" + "=" * 60)
        console.print("[bold green]CRITIC LOOP COMPLETE[/bold green]")
        console.print("=" * 60)
        console.print(f"  Iterations: {report.iterations}")
        console.print(f"  Initial score: {report.initial_score.total:.3f}")
        console.print(f"  Final score: {report.final_score.total:.3f}")
        improvement = report.final_score.total - report.initial_score.total
        console.print(f"  Improvement: {improvement:+.3f}")
        console.print(f"  Revisions applied: {len(report.revisions_applied)}")
        console.print(f"  Revisions rejected: {len(report.revisions_rejected)}")
        console.print(f"  Convergence: {report.convergence_reason}")
        console.print(f"  Report: {self.config.output_dir / 'critic_report.json'}")

        return report

    # ── Internal methods ──────────────────────────────────────────

    def _generate_initial_dag(self) -> None:
        """Run initial narrative DAG generation (iteration 0)."""
        from scripts.generate_narrative_dag import generate

        generate(
            narrative_path=self.narrative_path,
            base_dag_path=self.base_dag_path,
            output_path=self.output_dag_path,
        )

    def _run_estimation(self, iteration: int) -> None:
        """Run AgentLoop estimation on the current DAG."""
        try:
            from shared.agentic.dag.parser import parse_dag
            from shared.agentic.agent_loop import AgentLoop, AgentLoopConfig

            dag = parse_dag(self.output_dag_path)
            loop_config = AgentLoopConfig(
                mode=self.config.estimation_mode,
                max_iterations=self.config.estimation_max_iterations,
                output_dir=self.config.output_dir,
                force_run=self.config.force,
            )
            loop = AgentLoop(dag, loop_config)
            loop.run()
        except Exception as e:
            logger.warning(f"Estimation failed at iteration {iteration}: {e}")

    def _run_repair(self) -> None:
        """Run structural repair on the current DAG."""
        from scripts.generate_narrative_dag import repair_narrative_dag

        with open(self.output_dag_path) as f:
            raw = yaml.safe_load(f)
        repaired = repair_narrative_dag(raw, self.base_dag_path)
        with open(self.output_dag_path, "w", encoding="utf-8") as f:
            yaml.dump(
                repaired, f,
                sort_keys=False, allow_unicode=True, default_flow_style=False,
            )

    def _write_report(self, report: CriticReport) -> None:
        """Write critic report to output directory."""
        report_path = self.config.output_dir / "critic_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)


# ──────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run narrative DAG generation with principle-based critic loop.",
    )
    parser.add_argument(
        "narrative", type=Path,
        help="Path to narrative text file",
    )
    parser.add_argument(
        "--base-dag", type=Path,
        default=PROJECT_ROOT / "config" / "agentic" / "dags" / "kspi_k2_full.yaml",
        help="Path to base DAG YAML for node matching",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        default=PROJECT_ROOT / "config" / "agentic" / "dags" / "kspi_k2_narrative.yaml",
        help="Output path for generated DAG YAML",
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=3,
        help="Number of critic iterations (0=generate only)",
    )
    parser.add_argument(
        "--critic-backend", type=str, default="claude-code",
        choices=["claude-code", "codex", "api"],
        help="LLM backend for critique",
    )
    parser.add_argument(
        "--no-estimation", action="store_true",
        help="Skip AgentLoop estimation between iterations",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Continue despite DAG validation errors",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = CriticLoopConfig(
        max_iterations=args.iterations,
        critic_backend=args.critic_backend,
        run_estimation=not args.no_estimation,
        force=args.force,
    )

    loop = NarrativeCriticLoop(
        narrative_path=args.narrative,
        base_dag_path=args.base_dag,
        output_dag_path=args.output,
        config=config,
    )

    try:
        report = loop.run()
        if report.final_score.total < report.initial_score.total:
            logger.warning("Final score decreased — review revisions")
    except Exception as e:
        logger.error(f"Critic loop failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
