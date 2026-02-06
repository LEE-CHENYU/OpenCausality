#!/usr/bin/env python3
"""
Test Run Wrapper for Agentic Loop Evaluation.

Wraps run_real_estimation.main() with:
  - Timestamped output directory
  - Run metadata collection (git, python version, timestamp)
  - Post-run issue detection via IssueRegistry + IssueGates
  - Supplementary artifact export (issue summary, identification results, etc.)

Usage:
    python scripts/run_test_estimation.py
    python scripts/run_test_estimation.py --output-dir outputs/test_run/my_run
    python scripts/run_test_estimation.py --tag "agentic-v3-validation"
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import platform
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_real_estimation import main as run_main
from shared.agentic.issues.issue_ledger import IssueLedger
from shared.agentic.issues.issue_registry import IssueRegistry
from shared.agentic.issues.issue_gates import IssueGates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _git_info() -> dict[str, str | bool]:
    """Collect current git metadata (branch, commit, dirty status)."""
    info: dict[str, str | bool] = {}
    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT, text=True,
        ).strip()
    except Exception:
        info["branch"] = "unknown"
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, text=True,
        ).strip()
    except Exception:
        info["commit"] = "unknown"
    try:
        diff_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, text=True,
        ).strip()
        info["dirty"] = len(diff_out) > 0
    except Exception:
        info["dirty"] = True
    return info


def _collect_run_metadata(tag: str | None = None) -> dict:
    """Build run metadata dict."""
    git = _git_info()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_branch": git.get("branch", "unknown"),
        "git_commit": git.get("commit", "unknown"),
        "git_dirty": git.get("dirty", True),
        "tag": tag or "",
    }


def _extract_identification_results(cards: dict) -> dict:
    """Extract identification blocks from edge cards."""
    results = {}
    for edge_id, card in cards.items():
        id_block = getattr(card, "identification", None)
        if id_block and id_block.claim_level:
            results[edge_id] = {
                "claim_level": id_block.claim_level,
                "risks": dict(id_block.risks) if id_block.risks else {},
                "untestable_assumptions": list(id_block.untestable_assumptions),
                "testable_threats_passed": list(id_block.testable_threats_passed),
                "testable_threats_failed": list(id_block.testable_threats_failed),
            }
    return results


def _extract_estimation_summary(cards: dict) -> dict:
    """Extract summary statistics from edge cards."""
    rating_counts: dict[str, int] = {}
    edges_by_rating: dict[str, list[str]] = {}
    null_edges: list[str] = []

    for edge_id, card in cards.items():
        r = card.credibility_rating
        rating_counts[r] = rating_counts.get(r, 0) + 1
        edges_by_rating.setdefault(r, []).append(edge_id)
        if card.is_precisely_null:
            null_edges.append(edge_id)

    return {
        "total_edges": len(cards),
        "rating_distribution": rating_counts,
        "edges_by_rating": edges_by_rating,
        "precisely_null_count": len(null_edges),
        "precisely_null_edges": null_edges,
    }


def run_test(output_dir: Path, tag: str | None = None) -> None:
    """Execute a full test estimation run with artifact collection."""

    logger.info("=" * 70)
    logger.info("TEST ESTIMATION RUN")
    logger.info(f"  Output: {output_dir}")
    if tag:
        logger.info(f"  Tag: {tag}")
    logger.info("=" * 70)

    # 1. Collect run metadata
    metadata = _collect_run_metadata(tag)
    run_id = f"test_{metadata['git_commit']}_{datetime.now().strftime('%H%M%S')}"
    metadata["run_id"] = run_id

    # 2. Run the full estimation pipeline
    logger.info("Starting full estimation pipeline...")
    start_time = datetime.now()
    cards = run_main(output_dir=output_dir)
    elapsed = (datetime.now() - start_time).total_seconds()
    metadata["elapsed_seconds"] = elapsed
    metadata["edge_count"] = len(cards)
    logger.info(f"Pipeline complete: {len(cards)} edge cards in {elapsed:.1f}s")

    # 3. Post-run issue detection
    logger.info("Running post-run issue detection...")
    issues_dir = output_dir / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    ledger = IssueLedger(run_id=run_id, output_dir=issues_dir)

    try:
        registry = IssueRegistry.load()
        detected = registry.detect_post_run_issues(cards, ledger)
        logger.info(f"  Detected {len(detected)} issues")
    except Exception as e:
        logger.warning(f"  Issue detection failed: {e}")
        detected = []

    # 4. Evaluate gates
    logger.info("Evaluating issue gates...")
    try:
        gates = IssueGates()
        gate_eval = gates.evaluate(ledger)
        logger.info(f"  {gate_eval.summary()}")
    except Exception as e:
        logger.warning(f"  Gate evaluation failed: {e}")
        gate_eval = None

    # 5. Flush issue ledger
    try:
        jsonl_path = ledger.flush()
        logger.info(f"  Issue ledger written to {jsonl_path}")
    except Exception as e:
        logger.warning(f"  Issue ledger flush failed: {e}")

    # 6. Save supplementary artifacts
    logger.info("Saving supplementary artifacts...")

    # run_metadata.json
    meta_path = output_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"  {meta_path}")

    # identification_results.json
    id_results = _extract_identification_results(cards)
    id_path = output_dir / "identification_results.json"
    with open(id_path, "w") as f:
        json.dump(id_results, f, indent=2, default=str)
    logger.info(f"  {id_path} ({len(id_results)} edges with identification)")

    # estimation_summary.json
    est_summary = _extract_estimation_summary(cards)
    est_path = output_dir / "estimation_summary.json"
    with open(est_path, "w") as f:
        json.dump(est_summary, f, indent=2, default=str)
    logger.info(f"  {est_path}")

    # issue_summary.json
    issue_summary = {
        "run_id": run_id,
        "total_issues": len(ledger.issues),
        "open_issues": len(ledger.get_open_issues()),
        "critical_open": len(ledger.get_critical_open()),
        "human_required": len(ledger.get_issues_requiring_human()),
        "auto_fixable": len(ledger.get_auto_fixable()),
        "by_severity": {},
        "by_rule": {},
    }
    for issue in ledger.issues:
        issue_summary["by_severity"][issue.severity] = (
            issue_summary["by_severity"].get(issue.severity, 0) + 1
        )
        issue_summary["by_rule"][issue.rule_id] = (
            issue_summary["by_rule"].get(issue.rule_id, 0) + 1
        )
    if gate_eval:
        issue_summary["gate_evaluation"] = {
            "can_proceed": gate_eval.can_proceed,
            "can_promote_to_confirmation": gate_eval.can_promote_to_confirmation,
            "requires_hitl": gate_eval.requires_hitl,
            "auto_fixable_count": gate_eval.auto_fixable_count,
            "gates_triggered": [
                r.gate_name for r in gate_eval.results if r.triggered
            ],
        }
    iss_path = output_dir / "issue_summary.json"
    with open(iss_path, "w") as f:
        json.dump(issue_summary, f, indent=2, default=str)
    logger.info(f"  {iss_path}")

    # 7. Print final summary
    print("\n" + "=" * 70)
    print("TEST RUN SUMMARY")
    print("=" * 70)
    print(f"  Run ID:     {run_id}")
    print(f"  Tag:        {tag or '(none)'}")
    print(f"  Output:     {output_dir}")
    print(f"  Duration:   {elapsed:.1f}s")
    print(f"  Edge cards: {len(cards)}")
    print(f"  ID blocks:  {len(id_results)}")
    print()
    print(f"  Rating distribution:")
    for r in sorted(est_summary["rating_distribution"]):
        print(f"    {r}: {est_summary['rating_distribution'][r]}")
    print()
    print(f"  Issues: {len(ledger.issues)} total, "
          f"{len(ledger.get_open_issues())} open, "
          f"{len(ledger.get_critical_open())} critical")
    if gate_eval:
        print(f"  Gates: {'PASS' if gate_eval.can_proceed else 'BLOCKED'}")
    print()
    print(f"  Artifacts saved to {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run test estimation with agentic loop evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/test_run/<timestamp>)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag for the run (e.g., 'agentic-v3-validation')",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs/test_run") / ts
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    run_test(output_dir, tag=args.tag)


if __name__ == "__main__":
    main()
