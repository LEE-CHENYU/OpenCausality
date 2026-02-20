#!/usr/bin/env python3
"""
Full-Scope Estimation-Aware Repair Loop.

Orchestrates the DAGCritic with all four auto-fix stages:
  1. Identity claim fixes
  2. Unit chain fixes
  3. Identity coefficient LP overrides
  4. LP re-estimation of broken causal edges

Then optionally runs LLM-driven structural critique iterations.

Usage:
    python scripts/full_scope_repair.py
    python scripts/full_scope_repair.py --dag config/agentic/dags/kspi_k2_narrative.yaml
    python scripts/full_scope_repair.py --max-iterations 5 --no-llm
    python scripts/full_scope_repair.py --critic-backend api
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.agentic.agents.dag_critic import DAGCritic

logger = logging.getLogger(__name__)


def run_full_scope_repair(
    dag_path: Path,
    output_dir: Path,
    max_iterations: int = 5,
    min_delta: float = 0.01,
    critic_backend: str = "claude-code",
    run_llm: bool = True,
) -> dict:
    """Run the full-scope repair loop.

    Args:
        dag_path: Path to DAG YAML file.
        output_dir: Path to output artifacts directory.
        max_iterations: Maximum number of repair iterations.
        min_delta: Convergence threshold for quality delta.
        critic_backend: LLM backend for structural critique.
        run_llm: Whether to run LLM structural critique after auto-fixes.

    Returns:
        Summary dict with iteration history and final quality.
    """
    critic = DAGCritic(
        dag_path=dag_path,
        output_dir=output_dir,
        backend=critic_backend,
    )

    history: list[dict] = []
    prev_quality = 0.0

    print(f"\n{'='*60}")
    print("Full-Scope Estimation-Aware Repair Loop")
    print(f"{'='*60}")
    print(f"DAG:        {dag_path}")
    print(f"Output:     {output_dir}")
    print(f"Backend:    {critic_backend}")
    print(f"Max iter:   {max_iterations}")
    print(f"LLM critic: {'enabled' if run_llm else 'disabled'}")
    print(f"{'='*60}\n")

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        # Phase A: Deterministic auto-fixes (all 4 steps)
        print("  [A] Running deterministic auto-fixes...")
        n_auto_fixes = critic.apply_auto_fixes()
        print(f"      Applied {n_auto_fixes} auto-fixes")

        # Phase B: Compute quality score
        quality = critic.compute_quality_score()
        delta = quality.total - prev_quality

        print(f"  [Q] Quality: {quality.total:.3f} (delta={delta:+.3f})")
        print(f"      edge_type_diversity:     {quality.edge_type_diversity:.3f}")
        print(f"      identification_coverage: {quality.identification_coverage:.3f}")
        print(f"      metadata_completeness:   {quality.metadata_completeness:.3f}")
        print(f"      structural_soundness:    {quality.structural_soundness:.3f}")
        print(f"      unit_completeness:       {quality.unit_completeness:.3f}")
        print(f"      formula_correctness:     {quality.formula_correctness:.3f}")
        print(f"      propagation_health:      {quality.propagation_health:.3f}")

        iteration_record = {
            "iteration": iteration + 1,
            "auto_fixes": n_auto_fixes,
            "quality": quality.total,
            "delta": delta,
            "components": {
                "edge_type_diversity": quality.edge_type_diversity,
                "identification_coverage": quality.identification_coverage,
                "metadata_completeness": quality.metadata_completeness,
                "structural_soundness": quality.structural_soundness,
                "unit_completeness": quality.unit_completeness,
                "formula_correctness": quality.formula_correctness,
                "propagation_health": quality.propagation_health,
            },
        }

        # Phase C: LLM structural critique (optional)
        n_llm_applied = 0
        if run_llm and iteration < max_iterations - 1:
            print("  [C] Running LLM structural critique...")
            try:
                feedback = critic.collect_feedback(iteration=iteration)
                revisions = critic.critique(feedback, iteration=iteration)
                if revisions:
                    applied, rejected = critic.apply_revisions(revisions)
                    n_llm_applied = len(applied)
                    print(
                        f"      LLM revisions: {n_llm_applied} applied, "
                        f"{len(rejected)} rejected"
                    )
                else:
                    print("      LLM proposed no revisions")
            except Exception as e:
                logger.warning(f"LLM critique failed: {e}")
                print(f"      LLM critique failed: {e}")

        iteration_record["llm_revisions"] = n_llm_applied
        history.append(iteration_record)
        prev_quality = quality.total

        # Convergence check
        if abs(delta) < min_delta and n_auto_fixes == 0 and n_llm_applied == 0:
            print(f"\n  Converged (delta={delta:+.3f} < {min_delta})")
            break

    # Final summary
    final_quality = critic.compute_quality_score()
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Iterations:          {len(history)}")
    print(f"  Final quality:       {final_quality.total:.3f}")
    print(f"  Propagation health:  {final_quality.propagation_health:.3f}")

    # Report remaining issues
    print(f"\n--- Remaining Issues ---")
    feedback = critic.collect_feedback()

    if feedback.edges_missing_identification:
        print(f"  Missing identification: {', '.join(feedback.edges_missing_identification)}")
    if feedback.edges_missing_units:
        print(f"  Missing units: {', '.join(feedback.edges_missing_units)}")
    if feedback.structural_gaps:
        for gap in feedback.structural_gaps:
            print(f"  Gap [{gap['rule']}]: {gap['node_id']} — {gap['context']}")

    # Report estimation problems that couldn't be fixed
    placeholder_remaining = critic._detect_placeholder_edges()
    if placeholder_remaining:
        print(f"  Placeholder edges: {', '.join(p['edge_id'] for p in placeholder_remaining)}")

    wrong_sign_remaining = critic._detect_wrong_signs()
    if wrong_sign_remaining:
        print(f"  Wrong-sign edges: {', '.join(p['edge_id'] for p in wrong_sign_remaining)}")

    print(f"{'='*60}\n")

    return {
        "dag_path": str(dag_path),
        "output_dir": str(output_dir),
        "iterations": len(history),
        "history": history,
        "final_quality": final_quality.total,
        "final_propagation_health": final_quality.propagation_health,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full-scope estimation-aware DAG repair loop",
    )
    parser.add_argument(
        "--dag",
        type=Path,
        default=Path("config/agentic/dags/kspi_k2_narrative.yaml"),
        help="Path to DAG YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: derived from DAG name)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of repair iterations",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.01,
        help="Quality delta convergence threshold",
    )
    parser.add_argument(
        "--critic-backend",
        choices=["claude-code", "codex", "api"],
        default="claude-code",
        help="LLM backend for structural critique",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM structural critique (auto-fixes only)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Resolve output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        dag_stem = args.dag.stem
        output_dir = Path("outputs/agentic") / dag_stem

    if not args.dag.exists():
        print(f"Error: DAG file not found: {args.dag}", file=sys.stderr)
        sys.exit(1)

    result = run_full_scope_repair(
        dag_path=args.dag,
        output_dir=output_dir,
        max_iterations=args.max_iterations,
        min_delta=args.min_delta,
        critic_backend=args.critic_backend,
        run_llm=not args.no_llm,
    )

    # Save result
    result_path = output_dir / "full_scope_repair_result.yaml"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        yaml.dump(result, f, sort_keys=False, default_flow_style=False)
    print(f"Result saved to {result_path}")


if __name__ == "__main__":
    main()
