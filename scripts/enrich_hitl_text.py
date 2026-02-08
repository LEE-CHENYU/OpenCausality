#!/usr/bin/env python3
"""
Enrich HITL YAML files with LLM-generated descriptions.

Reads issue_registry.yaml and hitl_actions.yaml, generates missing
explanation/guidance/tooltip fields via LLM, and writes enriched YAML.

This script is idempotent: it only generates text for fields that are
missing or empty. Run it whenever new rules are added to regenerate
descriptions for the new entries.

Usage:
    python scripts/enrich_hitl_text.py
    python scripts/enrich_hitl_text.py --dry-run
    python scripts/enrich_hitl_text.py --output-suffix .enriched.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REGISTRY_PATH = PROJECT_ROOT / "config" / "agentic" / "issue_registry.yaml"
ACTIONS_PATH = PROJECT_ROOT / "config" / "agentic" / "hitl_actions.yaml"

ENRICHMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "guidance": {"type": "string"},
    },
    "required": ["explanation", "guidance"],
}

TOOLTIP_SCHEMA = {
    "type": "object",
    "properties": {
        "tooltip": {"type": "string"},
    },
    "required": ["tooltip"],
}


def enrich_registry(dry_run: bool = False, suffix: str = "") -> int:
    """Enrich issue_registry.yaml with explanation/guidance fields."""
    from shared.llm.client import get_llm_client

    with open(REGISTRY_PATH) as f:
        data = yaml.safe_load(f)

    llm = get_llm_client()
    count = 0

    for rule in data.get("rules", []):
        rid = rule.get("rule_id", "")
        desc = rule.get("description", "")

        if rule.get("explanation") and rule.get("guidance"):
            continue  # Already enriched

        print(f"  Generating explanation/guidance for {rid}...")
        result = llm.complete_structured(
            system=(
                "You are an econometrics methodology expert. For the given issue rule, "
                "write (a) a 2-4 sentence 'explanation' of why this issue matters for causal "
                "inference, what methodological loophole it catches, and what happens if ignored; "
                "and (b) a 1-2 sentence 'guidance' for the analyst deciding how to resolve it."
            ),
            user=f"Rule: {rid}\nDescription: {desc}\nSeverity: {rule.get('severity', '')}\n",
            schema=ENRICHMENT_SCHEMA,
        )
        rule["explanation"] = result.get("explanation", "")
        rule["guidance"] = result.get("guidance", "")
        count += 1

    if not dry_run:
        out_path = REGISTRY_PATH.with_suffix(suffix + ".yaml") if suffix else REGISTRY_PATH
        with open(out_path, "w") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        print(f"  Wrote {out_path}")

    return count


def enrich_actions(dry_run: bool = False, suffix: str = "") -> int:
    """Enrich hitl_actions.yaml with tooltip fields on each option."""
    from shared.llm.client import get_llm_client

    with open(ACTIONS_PATH) as f:
        data = yaml.safe_load(f)

    llm = get_llm_client()
    count = 0

    for rule_id, action_cfg in data.get("actions", {}).items():
        for opt in action_cfg.get("options", []):
            if opt.get("tooltip"):
                continue  # Already has tooltip

            label = opt.get("label", "")
            value = opt.get("value", "")
            action_type = action_cfg.get("action_type", "")

            print(f"  Generating tooltip for {rule_id}/{value}...")
            result = llm.complete_structured(
                system=(
                    "You are an econometrics methodology expert. Write a concise tooltip "
                    "(1-2 sentences) explaining what happens when the analyst selects this "
                    "action option. Include downstream consequences for estimation, propagation, "
                    "or reporting."
                ),
                user=f"Rule: {rule_id}\nAction type: {action_type}\nOption: {label} ({value})",
                schema=TOOLTIP_SCHEMA,
            )
            opt["tooltip"] = result.get("tooltip", "")
            count += 1

    if not dry_run:
        out_path = ACTIONS_PATH.with_suffix(suffix + ".yaml") if suffix else ACTIONS_PATH
        with open(out_path, "w") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        print(f"  Wrote {out_path}")

    return count


def enrich(
    dry_run: bool = False,
    registry_only: bool = False,
    actions_only: bool = False,
) -> int:
    """Enrich HITL YAML files with LLM-generated text.

    Args:
        dry_run: Don't write files, just show what would be generated.
        registry_only: Only enrich issue_registry.yaml.
        actions_only: Only enrich hitl_actions.yaml.

    Returns:
        Total number of LLM calls made.
    """
    print("Enriching HITL text...")
    total = 0

    if not actions_only:
        print("\n[1/2] issue_registry.yaml")
        reg_count = enrich_registry(dry_run=dry_run)
        print(f"  Generated {reg_count} explanation/guidance pairs")
        total += reg_count

    if not registry_only:
        print("\n[2/2] hitl_actions.yaml")
        act_count = enrich_actions(dry_run=dry_run)
        print(f"  Generated {act_count} tooltips")
        total += act_count

    print(f"\nDone. Total LLM calls: {total}")
    if dry_run:
        print("(Dry run -- no files written)")

    return total


def main():
    parser = argparse.ArgumentParser(description="Enrich HITL YAML files with LLM-generated text.")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files, just show what would be generated")
    parser.add_argument("--registry-only", action="store_true", help="Only enrich issue_registry.yaml")
    parser.add_argument("--actions-only", action="store_true", help="Only enrich hitl_actions.yaml")
    args = parser.parse_args()

    enrich(dry_run=args.dry_run, registry_only=args.registry_only, actions_only=args.actions_only)


if __name__ == "__main__":
    main()
