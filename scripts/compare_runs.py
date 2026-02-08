#!/usr/bin/env python3
"""
Compare Edge Cards Across Two Estimation Runs.

Loads edge card YAMLs from two directories (Run A = baseline, Run B = new)
and produces a structured markdown diff report covering estimates,
credibility, diagnostics, identification, and counterfactual status.

Usage:
    python scripts/compare_runs.py outputs/agentic outputs/test_run/<timestamp>
    python scripts/compare_runs.py --run-a outputs/agentic --run-b outputs/test_run/<timestamp> -o report.md
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import yaml


def _load_cards(directory: Path) -> dict[str, dict]:
    """Load all edge card YAMLs from a directory."""
    cards_dir = directory / "cards" / "edge_cards"
    if not cards_dir.exists():
        print(f"Warning: {cards_dir} does not exist", file=sys.stderr)
        return {}

    cards = {}
    for yaml_path in sorted(cards_dir.glob("*.yaml")):
        with open(yaml_path) as f:
            card = yaml.safe_load(f)
        if card and "edge_id" in card:
            cards[card["edge_id"]] = card
    return cards


def _safe_get(d: dict | None, *keys, default=None):
    """Safely traverse nested dicts."""
    val = d
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val


def _float_eq(a, b, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    """Compare two floats with tolerance, handling None."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def _ci_eq(a, b) -> bool:
    """Compare CI tuples/lists."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return all(_float_eq(x, y) for x, y in zip(a, b))


def _compare_estimates(card_a: dict, card_b: dict) -> dict:
    """Compare estimate fields. Returns dict of divergences."""
    divergences = {}
    est_a = card_a.get("estimates") or {}
    est_b = card_b.get("estimates") or {}

    for field in ["point", "se", "pvalue"]:
        va = est_a.get(field)
        vb = est_b.get(field)
        if not _float_eq(va, vb):
            divergences[field] = {"run_a": va, "run_b": vb}

    ci_a = est_a.get("ci_95")
    ci_b = est_b.get("ci_95")
    if not _ci_eq(ci_a, ci_b):
        divergences["ci_95"] = {"run_a": ci_a, "run_b": ci_b}

    return divergences


def _compare_credibility(card_a: dict, card_b: dict) -> dict | None:
    """Compare credibility rating/score. Returns change dict or None."""
    ra = card_a.get("credibility_rating")
    rb = card_b.get("credibility_rating")
    sa = card_a.get("credibility_score")
    sb = card_b.get("credibility_score")

    if ra != rb or not _float_eq(sa, sb):
        return {
            "rating_a": ra, "rating_b": rb,
            "score_a": sa, "score_b": sb,
        }
    return None


def _compare_diagnostics(card_a: dict, card_b: dict) -> dict:
    """Compare diagnostic pass/fail. Returns dict of changes."""
    diag_a = card_a.get("diagnostics") or {}
    diag_b = card_b.get("diagnostics") or {}
    changes = {}

    all_keys = set(diag_a.keys()) | set(diag_b.keys())
    for key in sorted(all_keys):
        da = diag_a.get(key)
        db = diag_b.get(key)
        pa = _safe_get(da, "passed") if da else None
        pb = _safe_get(db, "passed") if db else None

        if da is None and db is not None:
            changes[key] = {"status": "NEW", "passed": pb}
        elif da is not None and db is None:
            changes[key] = {"status": "REMOVED", "was_passed": pa}
        elif pa != pb:
            changes[key] = {"status": "FLIPPED", "was": pa, "now": pb}

    return changes


def _extract_identification(card: dict) -> dict | None:
    """Extract identification block from a card, if present."""
    id_block = card.get("identification")
    if not id_block:
        return None
    claim = id_block.get("claim_level", "")
    if not claim:
        return None
    return id_block


def _extract_counterfactual(card: dict) -> dict | None:
    """Extract counterfactual_block from a card, if present."""
    cf = card.get("counterfactual_block")
    if not cf:
        return None
    if "allowed" not in cf:
        return None
    return cf


def generate_comparison_report(
    cards_a: dict[str, dict],
    cards_b: dict[str, dict],
    dir_a: Path,
    dir_b: Path,
) -> str:
    """Generate a structured markdown comparison report."""

    lines: list[str] = []

    def h1(text: str):
        lines.append(f"\n# {text}\n")

    def h2(text: str):
        lines.append(f"\n## {text}\n")

    def h3(text: str):
        lines.append(f"\n### {text}\n")

    # Header
    lines.append("# Edge Card Comparison Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Run A (baseline):** `{dir_a}`")
    lines.append(f"**Run B (new):** `{dir_b}`")
    lines.append(f"**Run A edges:** {len(cards_a)}")
    lines.append(f"**Run B edges:** {len(cards_b)}")
    lines.append("")

    # Align by edge_id
    all_edges = sorted(set(cards_a.keys()) | set(cards_b.keys()))
    common = sorted(set(cards_a.keys()) & set(cards_b.keys()))
    only_a = sorted(set(cards_a.keys()) - set(cards_b.keys()))
    only_b = sorted(set(cards_b.keys()) - set(cards_a.keys()))

    # ---------------------------------------------------------------
    # Section 1: Estimate Validation
    # ---------------------------------------------------------------
    h2("1. Estimate Validation")
    lines.append("Estimates should be **identical** across runs (same data + code).")
    lines.append("Any divergence indicates a potential bug.\n")

    estimate_divergences: dict[str, dict] = {}
    for edge_id in common:
        div = _compare_estimates(cards_a[edge_id], cards_b[edge_id])
        if div:
            estimate_divergences[edge_id] = div

    if not estimate_divergences:
        lines.append("**PASS: All estimates match across runs.**\n")
    else:
        lines.append(f"**WARNING: {len(estimate_divergences)} edge(s) have estimate divergences.**\n")
        lines.append("| Edge | Field | Run A | Run B |")
        lines.append("|------|-------|-------|-------|")
        for edge_id, divs in sorted(estimate_divergences.items()):
            for field, vals in divs.items():
                va = vals["run_a"]
                vb = vals["run_b"]
                lines.append(f"| {edge_id} | {field} | {va} | {vb} |")
        lines.append("")

    # ---------------------------------------------------------------
    # Section 2: Credibility Rating Changes
    # ---------------------------------------------------------------
    h2("2. Credibility Rating Changes")

    cred_changes: dict[str, dict] = {}
    for edge_id in common:
        change = _compare_credibility(cards_a[edge_id], cards_b[edge_id])
        if change:
            cred_changes[edge_id] = change

    if not cred_changes:
        lines.append("No credibility rating changes.\n")
    else:
        lines.append(f"{len(cred_changes)} edge(s) with changed credibility:\n")
        lines.append("| Edge | Rating A | Rating B | Score A | Score B |")
        lines.append("|------|----------|----------|---------|---------|")
        for edge_id, ch in sorted(cred_changes.items()):
            lines.append(
                f"| {edge_id} | {ch['rating_a']} | {ch['rating_b']} "
                f"| {ch['score_a']:.3f} | {ch['score_b']:.3f} |"
            )
        lines.append("")

    # ---------------------------------------------------------------
    # Section 3: Diagnostic Changes
    # ---------------------------------------------------------------
    h2("3. Diagnostic Changes")

    diag_changes_all: dict[str, dict] = {}
    for edge_id in common:
        changes = _compare_diagnostics(cards_a[edge_id], cards_b[edge_id])
        if changes:
            diag_changes_all[edge_id] = changes

    if not diag_changes_all:
        lines.append("No diagnostic pass/fail changes.\n")
    else:
        lines.append(f"{len(diag_changes_all)} edge(s) with diagnostic changes:\n")
        lines.append("| Edge | Diagnostic | Status | Details |")
        lines.append("|------|-----------|--------|---------|")
        for edge_id, changes in sorted(diag_changes_all.items()):
            for diag_name, info in sorted(changes.items()):
                status = info["status"]
                if status == "FLIPPED":
                    detail = f"was={info['was']}, now={info['now']}"
                elif status == "NEW":
                    detail = f"passed={info['passed']}"
                else:
                    detail = f"was_passed={info['was_passed']}"
                lines.append(f"| {edge_id} | {diag_name} | {status} | {detail} |")
        lines.append("")

    # ---------------------------------------------------------------
    # Section 4: New Agentic Fields
    # ---------------------------------------------------------------
    h2("4. New Agentic Fields")
    lines.append("Fields only present in Run B (identification, counterfactual_block, propagation_role).\n")

    # Identification blocks
    h3("4.1 Identification Blocks")

    id_blocks: dict[str, dict] = {}
    claim_counts: dict[str, int] = {}
    for edge_id in sorted(cards_b.keys()):
        ib = _extract_identification(cards_b[edge_id])
        if ib:
            id_blocks[edge_id] = ib
            cl = ib.get("claim_level", "UNKNOWN")
            claim_counts[cl] = claim_counts.get(cl, 0) + 1

    if not id_blocks:
        lines.append("No identification blocks found in Run B.\n")
    else:
        lines.append(f"**{len(id_blocks)} edges** with identification blocks.\n")
        lines.append("**Claim level distribution:**\n")
        for cl, count in sorted(claim_counts.items()):
            lines.append(f"- {cl}: {count}")
        lines.append("")

        lines.append("| Edge | Claim Level | High Risks | Threats Failed |")
        lines.append("|------|------------|------------|----------------|")
        for edge_id, ib in sorted(id_blocks.items()):
            cl = ib.get("claim_level", "")
            risks = ib.get("risks", {})
            high_risks = [k for k, v in risks.items() if v == "high"]
            threats_failed = ib.get("testable_threats_failed", [])
            hr_str = ", ".join(high_risks) if high_risks else "-"
            tf_str = ", ".join(threats_failed) if threats_failed else "-"
            lines.append(f"| {edge_id} | {cl} | {hr_str} | {tf_str} |")
        lines.append("")

    # Counterfactual blocks
    h3("4.2 Counterfactual Status")

    cf_blocks: dict[str, dict] = {}
    blocked_edges: list[str] = []
    for edge_id in sorted(cards_b.keys()):
        cf = _extract_counterfactual(cards_b[edge_id])
        if cf:
            cf_blocks[edge_id] = cf
            if not cf.get("allowed", False):
                blocked_edges.append(edge_id)

    if not cf_blocks:
        lines.append("No counterfactual blocks found in Run B.\n")
    else:
        allowed_count = len(cf_blocks) - len(blocked_edges)
        lines.append(
            f"**{len(cf_blocks)} edges** with counterfactual assessment: "
            f"{allowed_count} allowed, {len(blocked_edges)} blocked.\n"
        )
        if blocked_edges:
            lines.append("**Blocked edges:**\n")
            lines.append("| Edge | Reason Blocked |")
            lines.append("|------|---------------|")
            for edge_id in blocked_edges:
                reason = cf_blocks[edge_id].get("reason_blocked", "-")
                lines.append(f"| {edge_id} | {reason} |")
            lines.append("")

    # Propagation roles
    h3("4.3 Propagation Roles")

    prop_roles: dict[str, str] = {}
    for edge_id in sorted(cards_b.keys()):
        pr = cards_b[edge_id].get("propagation_role")
        if pr and pr.get("role"):
            prop_roles[edge_id] = pr.get("role", "")

    if not prop_roles:
        lines.append("No propagation roles found in Run B.\n")
    else:
        role_counts: dict[str, int] = {}
        for role in prop_roles.values():
            role_counts[role] = role_counts.get(role, 0) + 1
        lines.append(f"**{len(prop_roles)} edges** with propagation roles.\n")
        lines.append("**Role distribution:**\n")
        for role, count in sorted(role_counts.items()):
            lines.append(f"- {role}: {count}")
        lines.append("")

    # ---------------------------------------------------------------
    # Section 5: Edge Coverage Differences
    # ---------------------------------------------------------------
    h2("5. Edge Coverage Differences")

    if only_a:
        lines.append(f"**{len(only_a)} edge(s) only in Run A:**\n")
        for edge_id in only_a:
            lines.append(f"- `{edge_id}`")
        lines.append("")
    if only_b:
        lines.append(f"**{len(only_b)} edge(s) only in Run B:**\n")
        for edge_id in only_b:
            lines.append(f"- `{edge_id}`")
        lines.append("")
    if not only_a and not only_b:
        lines.append("Both runs have identical edge coverage.\n")

    # ---------------------------------------------------------------
    # Section 6: Summary Assessment
    # ---------------------------------------------------------------
    h2("6. Summary Assessment")

    # Correctness verdict
    if not estimate_divergences:
        lines.append("### Correctness")
        lines.append("**VALIDATED**: All estimates are identical across runs. "
                      "The agentic modules do not alter core estimation results.\n")
    else:
        lines.append("### Correctness")
        lines.append(f"**INVESTIGATE**: {len(estimate_divergences)} edge(s) show "
                      "estimate divergences. This may indicate a bug in the "
                      "estimation pipeline or a non-deterministic component.\n")

    # Agentic value assessment
    lines.append("### Agentic Value Added")
    value_items = []
    if id_blocks:
        value_items.append(
            f"- **Identification screening**: {len(id_blocks)} edges assessed, "
            f"claim levels assigned ({', '.join(f'{k}={v}' for k, v in sorted(claim_counts.items()))})"
        )
    if cf_blocks:
        value_items.append(
            f"- **Counterfactual gating**: {len(blocked_edges)} edges blocked "
            f"from counterfactual use"
        )
    if cred_changes:
        value_items.append(
            f"- **Credibility recalibration**: {len(cred_changes)} edges "
            f"with adjusted ratings"
        )
    if diag_changes_all:
        new_diags = sum(
            1 for changes in diag_changes_all.values()
            for info in changes.values()
            if info["status"] == "NEW"
        )
        if new_diags:
            value_items.append(f"- **New diagnostics**: {new_diags} additional checks")

    if value_items:
        for item in value_items:
            lines.append(item)
    else:
        lines.append("No new agentic fields detected.")
    lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by `compare_runs.py` at {datetime.now().isoformat()}*")

    return "\n".join(lines)


def compare(
    baseline_dir: Path,
    new_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Compare edge cards from two estimation runs.

    Args:
        baseline_dir: Path to baseline (Run A) output directory.
        new_dir: Path to new (Run B) output directory.
        output_path: Output path for report (default: <new_dir>/COMPARISON_REPORT.md).

    Returns:
        Path to the written report.
    """
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory does not exist: {baseline_dir}")
    if not new_dir.exists():
        raise FileNotFoundError(f"New directory does not exist: {new_dir}")

    print(f"Loading Run A cards from {baseline_dir}...")
    cards_a = _load_cards(baseline_dir)
    print(f"  Loaded {len(cards_a)} edge cards")

    print(f"Loading Run B cards from {new_dir}...")
    cards_b = _load_cards(new_dir)
    print(f"  Loaded {len(cards_b)} edge cards")

    report = generate_comparison_report(cards_a, cards_b, baseline_dir, new_dir)

    out = output_path or (new_dir / "COMPARISON_REPORT.md")
    with open(out, "w") as f:
        f.write(report)

    print(f"\nReport written to {out}")
    print(f"  Common edges: {len(set(cards_a) & set(cards_b))}")
    print(f"  Only in A: {len(set(cards_a) - set(cards_b))}")
    print(f"  Only in B: {len(set(cards_b) - set(cards_a))}")

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compare edge cards from two estimation runs"
    )
    parser.add_argument(
        "run_a",
        nargs="?",
        type=Path,
        default=None,
        help="Path to Run A (baseline) output directory",
    )
    parser.add_argument(
        "run_b",
        nargs="?",
        type=Path,
        default=None,
        help="Path to Run B (new) output directory",
    )
    parser.add_argument(
        "--run-a",
        type=Path,
        default=None,
        dest="run_a_flag",
        help="Path to Run A (baseline) output directory (flag form)",
    )
    parser.add_argument(
        "--run-b",
        type=Path,
        default=None,
        dest="run_b_flag",
        help="Path to Run B (new) output directory (flag form)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for report (default: <run_b>/COMPARISON_REPORT.md)",
    )
    args = parser.parse_args()

    dir_a = args.run_a_flag or args.run_a
    dir_b = args.run_b_flag or args.run_b

    if dir_a is None or dir_b is None:
        parser.error("Both run-a and run-b directories are required")

    try:
        compare(dir_a, dir_b, args.output)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
