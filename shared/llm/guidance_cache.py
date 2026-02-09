"""
Shared LLM Guidance Cache.

Generates LLM guidance (per-issue decision guidance + per-edge annotations)
once and saves to disk so both DAG viz and HITL panel can load it without
duplicate API calls.

Cache dir: outputs/agentic/llm_cache/
├── issue_guidance.json    # {issue_key: text}
├── edge_annotations.json  # {edge_id: text}
└── metadata.json          # {generated_at, model, dag_path, state_hash}
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "outputs"
    / "agentic"
    / "llm_cache"
)


def _load_edge_card(card_path: Path) -> dict | None:
    """Load edge card YAML and extract LLM-relevant fields."""
    if not card_path.exists():
        return None
    with open(card_path) as f:
        card = yaml.safe_load(f)
    if not card:
        return None

    estimates = card.get("estimates", {})
    diagnostics_raw = card.get("diagnostics", {})
    identification = card.get("identification", {})
    spec = card.get("spec_details", {})

    diagnostics = []
    for diag in diagnostics_raw.values():
        if isinstance(diag, dict):
            diagnostics.append({
                "name": diag.get("name", ""),
                "passed": bool(diag.get("passed", False)),
            })

    n_obs = None
    eff_obs = diagnostics_raw.get("effective_obs", {})
    if isinstance(eff_obs, dict) and eff_obs.get("value") is not None:
        n_obs = int(eff_obs["value"])

    return {
        "point": estimates.get("point"),
        "se": estimates.get("se"),
        "pvalue": estimates.get("pvalue"),
        "design": spec.get("design"),
        "rating": card.get("credibility_rating"),
        "claim_level": identification.get("claim_level"),
        "diagnostics": diagnostics,
        "n_obs": n_obs,
    }


def _state_hash(state_path: Path) -> str:
    """Compute a short hash of the state file for cache invalidation."""
    if not state_path.exists():
        return "no_state"
    content = state_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:12]


def _generate_issue_guidance(
    open_issues: dict[str, dict],
    edge_data: dict[str, dict],
) -> dict[str, str]:
    """Generate LLM per-issue decision guidance.

    Parameters
    ----------
    open_issues : dict
        Keyed by issue_key (``RULE_ID:edge_id``), values are issue dicts
        with ``rule_id``, ``message``, ``severity``.
    edge_data : dict
        Keyed by edge_id, values are edge card summaries.

    Returns
    -------
    dict
        ``{issue_key: guidance_text}``
    """
    from shared.llm.client import get_llm_client
    from shared.llm.prompts import DECISION_GUIDANCE_SYSTEM, DECISION_GUIDANCE_USER

    client = get_llm_client()
    guidance: dict[str, str] = {}

    for key, issue in open_issues.items():
        parts = key.split(":", 1)
        edge_id = parts[1] if len(parts) > 1 else key
        edge = edge_data.get(edge_id, {})

        failed_diags = [
            d["name"] for d in edge.get("diagnostics", []) if not d.get("passed", True)
        ]

        user_msg = DECISION_GUIDANCE_USER.format(
            rule_id=issue.get("rule_id", ""),
            edge_id=edge_id,
            message=issue.get("message", ""),
            severity=issue.get("severity", ""),
            point=edge.get("point", "N/A"),
            se=edge.get("se", "N/A"),
            pvalue=edge.get("pvalue", "N/A"),
            n_obs=edge.get("n_obs", "N/A"),
            design=edge.get("design", "N/A"),
            claim_level=edge.get("claim_level", "N/A"),
            rating=edge.get("rating", "N/A"),
            failed_diagnostics=", ".join(failed_diags) if failed_diags else "none",
        )

        try:
            result = client.complete(
                system=DECISION_GUIDANCE_SYSTEM,
                user=user_msg,
                max_tokens=300,
            )
            guidance[key] = result.strip()
            print(f"    {key}")
        except Exception as e:
            logger.warning(f"LLM guidance failed for {key}: {e}")

    return guidance


def _generate_edge_annotations(
    dag: dict,
    edge_data: dict[str, dict],
) -> dict[str, str]:
    """Generate LLM substantive annotations for each edge.

    Parameters
    ----------
    dag : dict
        Parsed DAG YAML.
    edge_data : dict
        Keyed by edge_id, values are edge card summaries.

    Returns
    -------
    dict
        ``{edge_id: annotation_text}``
    """
    from shared.llm.client import get_llm_client
    from shared.llm.prompts import EDGE_ANNOTATION_SYSTEM

    client = get_llm_client()
    annotations: dict[str, str] = {}

    for edge in dag.get("edges", []):
        eid = edge.get("id", "")
        from_node = edge.get("from", "")
        to_node = edge.get("to", "")
        edge_type = edge.get("edge_type", "causal")

        card = edge_data.get(eid, {})
        card_context = ""
        if card.get("point") is not None:
            card_context = (
                f"Estimate: {card.get('point')}, SE: {card.get('se')}, "
                f"p-value: {card.get('pvalue')}, "
                f"Claim level: {card.get('claim_level')}, "
                f"Rating: {card.get('rating')}"
            )

        user_msg = (
            f"Edge: {from_node} -> {to_node} (id: {eid})\n"
            f"Type: {edge_type}\n"
            f"Notes: {edge.get('notes', '')}\n"
            f"{card_context}"
        )

        try:
            result = client.complete(
                system=EDGE_ANNOTATION_SYSTEM,
                user=user_msg,
                max_tokens=200,
            )
            annotations[eid] = result.strip()
            print(f"  Annotated: {eid}")
        except Exception as e:
            logger.warning(f"Edge annotation failed for {eid}: {e}")

    return annotations


def _find_orphan_nodes(dag: dict) -> list[dict]:
    """Return list of nodes with no edges connecting them."""
    nodes = {n["id"]: n for n in dag.get("nodes", [])}
    edge_nodes: set[str] = set()
    for e in dag.get("edges", []):
        edge_nodes.add(e.get("from", ""))
        edge_nodes.add(e.get("to", ""))
    for lat in dag.get("latents", []):
        for a in lat.get("affects", []):
            edge_nodes.add(a)
    orphans = []
    for nid, node in nodes.items():
        if nid not in edge_nodes:
            orphans.append(node)
    return orphans


def _generate_orphan_explanations(
    dag: dict,
) -> dict[str, str]:
    """Generate LLM explanations for why orphan nodes are unwired.

    Returns ``{node_id: explanation_text}``.
    """
    from shared.llm.client import get_llm_client
    from shared.llm.prompts import ORPHAN_NODE_SYSTEM, ORPHAN_NODE_USER

    orphans = _find_orphan_nodes(dag)
    if not orphans:
        return {}

    # Build context strings
    nodes = {n["id"]: n for n in dag.get("nodes", [])}
    edge_nodes: set[str] = set()
    for e in dag.get("edges", []):
        edge_nodes.add(e.get("from", ""))
        edge_nodes.add(e.get("to", ""))

    connected_summary = "\n".join(
        f"  {nid}: {nodes[nid].get('name', nid)}"
        for nid in sorted(edge_nodes) if nid in nodes
    )
    edges_summary = "\n".join(
        f"  {e.get('id', '?')}: {e.get('from', '?')} -> {e.get('to', '?')} ({e.get('edge_type', 'causal')})"
        for e in dag.get("edges", [])
    )

    client = get_llm_client()
    explanations: dict[str, str] = {}

    for node in orphans:
        nid = node.get("id", "")
        user_msg = ORPHAN_NODE_USER.format(
            node_id=nid,
            node_name=node.get("name", nid),
            node_description=node.get("description", ""),
            node_unit=node.get("unit", ""),
            connected_nodes=connected_summary,
            existing_edges=edges_summary,
        )

        try:
            result = client.complete(
                system=ORPHAN_NODE_SYSTEM,
                user=user_msg,
                max_tokens=400,
            )
            text = result.strip()
            if text:
                explanations[nid] = text
                print(f"  Orphan explained: {nid}")
            else:
                print(f"  Warning: empty response for {nid}, retrying...")
                result2 = client.complete(
                    system=ORPHAN_NODE_SYSTEM,
                    user=user_msg + "\n\nPlease provide the Role, Blocker, and Path forward sections.",
                    max_tokens=400,
                )
                text2 = result2.strip()
                explanations[nid] = text2
                print(f"  Orphan explained (retry): {nid}" if text2 else f"  Warning: still empty for {nid}")
        except Exception as e:
            logger.warning(f"Orphan explanation failed for {nid}: {e}")

    return explanations


def generate_and_cache(
    state_path: Path,
    cards_dir: Path,
    dag_path: Path,
    cache_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Generate both issue guidance and edge annotations, save to *cache_dir*.

    Returns ``{"issue_guidance": {...}, "edge_annotations": {...}}``.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load DAG
    with open(dag_path) as f:
        dag = yaml.safe_load(f) or {}

    # Load state (open issues)
    open_issues: dict[str, dict] = {}
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        for key, issue in state.get("issues", {}).items():
            if issue.get("status") == "OPEN":
                open_issues[key] = issue

    # Collect all edge IDs (from issues + dag edges)
    edge_ids: set[str] = set()
    for key in open_issues:
        parts = key.split(":", 1)
        edge_ids.add(parts[1] if len(parts) > 1 else key)
    for edge in dag.get("edges", []):
        eid = edge.get("id", "")
        if eid:
            edge_ids.add(eid)

    # Load edge card data for all relevant edges
    edge_data: dict[str, dict] = {}
    for eid in sorted(edge_ids):
        card = _load_edge_card(cards_dir / f"{eid}.yaml")
        if card:
            edge_data[eid] = card

    # Generate
    print("  Generating per-issue decision guidance...")
    issue_guidance = _generate_issue_guidance(open_issues, edge_data)
    print(f"  {len(issue_guidance)} issue guidance entries generated")

    print("  Generating per-edge annotations...")
    edge_annotations = _generate_edge_annotations(dag, edge_data)
    print(f"  {len(edge_annotations)} edge annotations generated")

    print("  Generating orphan node explanations...")
    orphan_explanations = _generate_orphan_explanations(dag)
    print(f"  {len(orphan_explanations)} orphan explanations generated")

    # Save to cache
    with open(cache_dir / "issue_guidance.json", "w") as f:
        json.dump(issue_guidance, f, indent=2)

    with open(cache_dir / "edge_annotations.json", "w") as f:
        json.dump(edge_annotations, f, indent=2)

    with open(cache_dir / "orphan_explanations.json", "w") as f:
        json.dump(orphan_explanations, f, indent=2)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dag_path": str(dag_path),
        "state_hash": _state_hash(state_path),
    }
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    result = {
        "issue_guidance": issue_guidance,
        "edge_annotations": edge_annotations,
        "orphan_explanations": orphan_explanations,
    }
    print(f"  Cache written to {cache_dir}")
    return result


def load_cache(cache_dir: Path | None = None) -> dict[str, dict[str, str]] | None:
    """Load cached guidance from *cache_dir*.

    Returns ``{"issue_guidance": {...}, "edge_annotations": {...}, "orphan_explanations": {...}}``
    or ``None`` if the cache is missing or incomplete.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    ig_path = cache_dir / "issue_guidance.json"
    ea_path = cache_dir / "edge_annotations.json"
    oe_path = cache_dir / "orphan_explanations.json"

    if not ig_path.exists() or not ea_path.exists():
        return None

    with open(ig_path) as f:
        issue_guidance = json.load(f)
    with open(ea_path) as f:
        edge_annotations = json.load(f)

    orphan_explanations: dict[str, str] = {}
    if oe_path.exists():
        with open(oe_path) as f:
            orphan_explanations = json.load(f)

    return {
        "issue_guidance": issue_guidance,
        "edge_annotations": edge_annotations,
        "orphan_explanations": orphan_explanations,
    }
