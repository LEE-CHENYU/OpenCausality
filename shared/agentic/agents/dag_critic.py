"""
DAG Critic Agent.

Principle-based quality improvement loop for narrative DAGs.
Collects structured feedback, sends to LLM for critique, validates
proposed revisions, and applies them to the DAG YAML.

Backends:
- claude-code: shell out to `claude` CLI (default, uses Claude Code credits)
- codex: shell out to `codex` CLI
- api: direct API via shared/llm/client.py
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from shared.agentic.agents.critic_schema import (
    CriticFeedback,
    CriticReport,
    CriticRevision,
    CriticRevisionValidator,
    QualityScore,
    RevisionType,
    VALID_EDGE_TYPES,
    load_principles,
)
from shared.llm.prompts import DAG_CRITIC_SYSTEM, DAG_CRITIC_USER

logger = logging.getLogger(__name__)


class DAGCritic:
    """Principle-based DAG quality critic.

    Operates at a different governance tier than PatchBot:
    - PatchBot fixes EdgeCards within a fixed DAG structure
    - Critic modifies DAG structure (edge types, formulas, nodes/edges)
    - change_edge_type is forbidden for PatchBot — Critic owns this
    """

    def __init__(
        self,
        dag_path: Path,
        output_dir: Path,
        backend: str = "claude-code",
        principles_path: Path | None = None,
    ):
        self.dag_path = dag_path
        self.output_dir = output_dir
        self.backend = backend
        self.principles = load_principles(principles_path)

        # Load current DAG
        with open(dag_path) as f:
            self.dag: dict[str, Any] = yaml.safe_load(f)

        self.nodes = {n["id"]: n for n in self.dag.get("nodes", [])}
        self.edges = {e["id"]: e for e in self.dag.get("edges", [])}

        # Quality weights from config
        self.weights = self.principles.get("quality_weights", {})
        self.confidence_threshold = self.principles.get("confidence_threshold", 0.7)

    def reload_dag(self) -> None:
        """Reload DAG from disk (after revisions applied externally)."""
        with open(self.dag_path) as f:
            self.dag = yaml.safe_load(f)
        self.nodes = {n["id"]: n for n in self.dag.get("nodes", [])}
        self.edges = {e["id"]: e for e in self.dag.get("edges", [])}

    # ──────────────────────────────────────────────────────────────
    # Quality Score
    # ──────────────────────────────────────────────────────────────

    def compute_quality_score(self) -> QualityScore:
        """Compute composite quality score from current DAG state."""
        edges_list = list(self.edges.values())
        n_total = len(edges_list)
        if n_total == 0:
            return QualityScore(total=0.0)

        # 1. Edge type diversity: 1 - (n_causal / n_total)
        n_causal = sum(1 for e in edges_list if e.get("edge_type") == "causal")
        edge_type_diversity = 1.0 - (n_causal / n_total) if n_total > 0 else 0.0

        # 2. Identification coverage: fraction of causal edges with strategy
        causal_edges = [e for e in edges_list if e.get("edge_type") == "causal"]
        if causal_edges:
            n_with_strategy = sum(
                1 for e in causal_edges
                if _has_identification_strategy(e)
            )
            identification_coverage = n_with_strategy / len(causal_edges)
        else:
            identification_coverage = 1.0  # No causal edges = fully covered

        # 3. Metadata completeness: fraction with expected_sign + timing + forbidden_controls
        n_metadata_complete = 0
        for e in edges_list:
            has_sign = bool(
                e.get("acceptance_criteria", {})
                .get("plausibility", {})
                .get("expected_sign")
            )
            has_timing = bool(e.get("timing", {}).get("lag") is not None)
            has_controls = "forbidden_controls" in e
            if has_sign and has_timing and has_controls:
                n_metadata_complete += 1
        metadata_completeness = n_metadata_complete / n_total

        # 4. Structural soundness: run pre-estimation checks
        structural_soundness = self._compute_structural_soundness()

        # 5. Unit completeness: fraction with both treatment + outcome units
        n_units_complete = 0
        for e in edges_list:
            unit_spec = e.get("unit_specification", {})
            if unit_spec.get("treatment_unit") and unit_spec.get("outcome_unit"):
                n_units_complete += 1
        unit_completeness = n_units_complete / n_total

        # 6. Formula correctness: fraction of identity edges with valid formulas
        identity_edges = [
            e for e in edges_list if e.get("edge_type") == "identity"
        ]
        if identity_edges:
            n_formula_ok = 0
            for e in identity_edges:
                to_node = self.nodes.get(e.get("to", ""))
                if to_node and not _has_double_log(to_node, self.nodes):
                    n_formula_ok += 1
            formula_correctness = n_formula_ok / len(identity_edges)
        else:
            formula_correctness = 1.0

        # Weighted total
        w = self.weights
        total = (
            w.get("edge_type_diversity", 0.15) * edge_type_diversity
            + w.get("identification_coverage", 0.25) * identification_coverage
            + w.get("metadata_completeness", 0.20) * metadata_completeness
            + w.get("structural_soundness", 0.15) * structural_soundness
            + w.get("unit_completeness", 0.15) * unit_completeness
            + w.get("formula_correctness", 0.10) * formula_correctness
        )

        return QualityScore(
            edge_type_diversity=edge_type_diversity,
            identification_coverage=identification_coverage,
            metadata_completeness=metadata_completeness,
            structural_soundness=structural_soundness,
            unit_completeness=unit_completeness,
            formula_correctness=formula_correctness,
            total=total,
        )

    def _compute_structural_soundness(self) -> float:
        """Run DAGValidator pre-estimation checks and score.

        Applies a gap penalty: each structural gap reduces soundness
        by 5%, capped at 50% total reduction.
        """
        from shared.agentic.validation import DAGValidator

        validator = DAGValidator(self.dag)
        result = validator.validate_pre_estimation()
        n_errors = result.error_count()
        n_checks = len(result.checks_run)
        if n_checks == 0:
            base = 1.0
        else:
            # Penalty per error: lose 1/n_checks per error
            penalty = min(n_errors / n_checks, 1.0)
            base = 1.0 - penalty

        # Gap penalty: each gap reduces soundness by 5%, capped at 50%
        n_gaps = len(self._detect_structural_gaps())
        gap_factor = max(0.5, 1.0 - n_gaps * 0.05)
        return base * gap_factor

    # ──────────────────────────────────────────────────────────────
    # Structural Gap Detection
    # ──────────────────────────────────────────────────────────────

    def _detect_structural_gaps(self) -> list[dict[str, Any]]:
        """Detect structural gaps using domain-agnostic graph rules.

        Returns a list of gap dicts with keys: rule, node_id, context.
        """
        gaps: list[dict[str, Any]] = []
        target = self.dag.get("target_node", "")
        exogenous = set(self.dag.get("exogenous_nodes", []))
        latents = {
            a.get("latent", "")
            for a in self.dag.get("assumptions", [])
            if a.get("latent")
        }

        # Build in/out degree maps
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        out_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for e in self.edges.values():
            fn, tn = e.get("from", ""), e.get("to", "")
            if tn in in_degree:
                in_degree[tn] += 1
            if fn in out_degree:
                out_degree[fn] += 1

        gap_rules = self.principles.get("structural_gap_rules", [])
        rule_ids = {r["id"] for r in gap_rules}

        # Rule 1: undriven_identity_dependency
        if "undriven_identity_dependency" in rule_ids:
            seen: set[str] = set()
            for nid, node in self.nodes.items():
                deps = (
                    node.get("identity", {}).get("depends_on")
                    or node.get("depends_on", [])
                )
                for dep in deps:
                    if (
                        dep in self.nodes
                        and dep not in seen
                        and in_degree.get(dep, 0) == 0
                        and dep not in exogenous
                        and dep not in latents
                    ):
                        gaps.append({
                            "rule": "undriven_identity_dependency",
                            "node_id": dep,
                            "context": (
                                f"Required by {nid}'s identity formula "
                                f"but has no incoming edges"
                            ),
                        })
                        seen.add(dep)

        # Rule 2: dead_end_leaf
        if "dead_end_leaf" in rule_ids:
            for nid in self.nodes:
                if (
                    nid != target
                    and in_degree.get(nid, 0) > 0
                    and out_degree.get(nid, 0) == 0
                ):
                    gaps.append({
                        "rule": "dead_end_leaf",
                        "node_id": nid,
                        "context": (
                            f"Has {in_degree[nid]} incoming but "
                            f"0 outgoing edges (dead-end)"
                        ),
                    })

        # Rule 3: endogenous_root
        if "endogenous_root" in rule_ids:
            for nid in self.nodes:
                if (
                    out_degree.get(nid, 0) > 0
                    and in_degree.get(nid, 0) == 0
                    and nid not in exogenous
                    and nid not in latents
                    and not self.nodes[nid].get("exogenous")
                ):
                    gaps.append({
                        "rule": "endogenous_root",
                        "node_id": nid,
                        "context": (
                            "Root node (no incoming edges) "
                            "but not marked exogenous"
                        ),
                    })

        # Rule 4: paper_mentioned_variable
        if "paper_mentioned_variable" in rule_ids:
            gaps.extend(self._extract_paper_mentioned_variables())

        return gaps

    # Common statistical/method abbreviations to exclude from paper variable extraction
    _METHOD_STOPWORDS = frozenset({
        "ols", "iv", "2sls", "gmm", "ml", "mle", "var", "svar", "vecm",
        "arima", "garch", "did", "rdd", "rct", "fe", "re", "lp", "irf",
        "capm", "apt", "dsge", "nber", "imf", "bis", "ecb", "fed",
        "gdp", "eme", "sdp", "usa", "usd", "eur", "cee", "see",
        "log", "ln", "pp", "bp", "bps", "pct", "std", "se",
        "trumps", "covid", "pandemic", "crisis",
    })

    def _extract_paper_mentioned_variables(self) -> list[dict[str, Any]]:
        """Extract variable mentions from literature that don't match DAG nodes."""
        cards_dir = self.output_dir / "cards" / "edge_cards"
        if not cards_dir.exists():
            return []

        # Collect all node IDs and names (lowercased) for matching
        known = set()
        for nid, node in self.nodes.items():
            known.add(nid.lower())
            name = node.get("name", "")
            if name:
                known.add(name.lower())
                # Also add individual words for fuzzy matching
                for word in name.lower().split():
                    if len(word) > 3:
                        known.add(word)

        # Scan excerpts for quoted terms or parenthetical abbreviations
        mentioned: dict[str, str] = {}  # variable -> first edge context
        for card_path in sorted(cards_dir.glob("*.yaml")):
            try:
                with open(card_path) as f:
                    card_data = yaml.safe_load(f)
                lit = card_data.get("literature", {})
                if not lit:
                    continue
                edge_id = card_data.get("edge_id", card_path.stem)
                for category in ("supporting", "challenging", "methodological"):
                    for paper in lit.get(category, []):
                        excerpt = paper.get("excerpt", "")
                        if not excerpt:
                            continue
                        # Find quoted terms: "term" or 'term'
                        for match in re.findall(r'["\']([a-zA-Z][a-zA-Z_ ]{2,30})["\']', excerpt):
                            term = match.strip().lower()
                            if (
                                term not in known
                                and term not in mentioned
                                and term not in self._METHOD_STOPWORDS
                            ):
                                mentioned[term] = edge_id
                        # Find parenthetical abbreviations: (ABC)
                        for match in re.findall(r'\(([A-Z]{2,8})\)', excerpt):
                            term = match.lower()
                            if (
                                term not in known
                                and term not in mentioned
                                and term not in self._METHOD_STOPWORDS
                            ):
                                mentioned[term] = edge_id
            except (yaml.YAMLError, OSError):
                continue

        return [
            {
                "rule": "paper_mentioned_variable",
                "node_id": var,
                "context": f"Mentioned in literature for {ctx} but not in DAG",
            }
            for var, ctx in list(mentioned.items())[:10]  # cap at 10
        ]

    # ──────────────────────────────────────────────────────────────
    # Feedback Collection
    # ──────────────────────────────────────────────────────────────

    def collect_feedback(self, iteration: int = 0) -> CriticFeedback:
        """Collect structured feedback from current DAG state.

        This is also exposed via CLI as `dag critic-feedback`.
        """
        quality = self.compute_quality_score()

        # Edge diagnostics
        edge_diagnostics = []
        for eid, edge in self.edges.items():
            diag = {
                "edge_id": eid,
                "from": edge.get("from", ""),
                "to": edge.get("to", ""),
                "edge_type": edge.get("edge_type", ""),
                "has_identification": _has_identification_strategy(edge),
                "has_units": bool(
                    edge.get("unit_specification", {}).get("treatment_unit")
                    and edge.get("unit_specification", {}).get("outcome_unit")
                ),
                "has_expected_sign": bool(
                    edge.get("acceptance_criteria", {})
                    .get("plausibility", {})
                    .get("expected_sign")
                ),
                "has_timing": edge.get("timing", {}).get("lag") is not None,
                "has_forbidden_controls": "forbidden_controls" in edge,
            }
            edge_diagnostics.append(diag)

        # Open issues (from issue ledger if available)
        open_issues = self._load_open_issues()

        # Edges missing identification
        edges_missing_id = [
            eid for eid, edge in self.edges.items()
            if edge.get("edge_type") == "causal"
            and not _has_identification_strategy(edge)
        ]

        # Edges missing units
        edges_missing_units = [
            eid for eid, edge in self.edges.items()
            if not (
                edge.get("unit_specification", {}).get("treatment_unit")
                and edge.get("unit_specification", {}).get("outcome_unit")
            )
        ]

        # Formula violations
        formula_violations = []
        for nid, node in self.nodes.items():
            if _has_double_log(node, self.nodes):
                formula_violations.append({
                    "node_id": nid,
                    "issue": "double_log",
                    "formula": node.get("identity", {}).get("formula", ""),
                })

        # Structural gaps
        structural_gaps = self._detect_structural_gaps()

        # Causal logic probes
        causal_logic_probes = self._compute_causal_logic_probes()

        return CriticFeedback(
            quality_score=quality,
            edge_diagnostics=edge_diagnostics,
            open_issues=open_issues,
            edges_missing_identification=edges_missing_id,
            edges_missing_units=edges_missing_units,
            formula_violations=formula_violations,
            structural_gaps=structural_gaps,
            causal_logic_probes=causal_logic_probes,
        )

    def _load_open_issues(self) -> list[dict[str, Any]]:
        """Load open issues from the issue ledger state file."""
        state_path = self.output_dir / "issues" / "state.json"
        if not state_path.exists():
            return []
        try:
            with open(state_path) as f:
                state = json.load(f)
            return [
                {"rule_id": iss.get("rule_id", ""), "edge_id": iss.get("edge_id", ""),
                 "severity": iss.get("severity", ""), "message": iss.get("message", "")}
                for iss in state.get("open_issues", [])
            ]
        except (json.JSONDecodeError, KeyError):
            return []

    def _load_edge_card_estimates(self) -> dict[str, dict[str, Any]]:
        """Load point estimates from edge card YAML files.

        Returns:
            {edge_id: {point, se, pvalue, treatment_unit, outcome_unit}}
        """
        cards_dir = self.output_dir / "cards" / "edge_cards"
        if not cards_dir.exists():
            return {}

        estimates: dict[str, dict[str, Any]] = {}
        for card_path in sorted(cards_dir.glob("*.yaml")):
            try:
                with open(card_path) as f:
                    card_data = yaml.safe_load(f)
                edge_id = card_data.get("edge_id", card_path.stem)
                est = card_data.get("estimates", {})
                if not est:
                    continue
                estimates[edge_id] = {
                    "point": est.get("point"),
                    "se": est.get("se"),
                    "pvalue": est.get("pvalue"),
                    "treatment_unit": est.get("treatment_unit", ""),
                    "outcome_unit": est.get("outcome_unit", ""),
                }
            except (yaml.YAMLError, OSError):
                continue
        return estimates

    def _compute_causal_logic_probes(self) -> list[dict[str, Any]]:
        """Run chain-level causal logic probes against current DAG state.

        Returns list of probe findings:
            [{probe, finding, edges_involved, severity}]
        """
        from shared.agentic.propagation import UnitSpec

        findings: list[dict[str, Any]] = []
        estimates = self._load_edge_card_estimates()

        # ── Probe 1: sign_coherence ──
        # For each node with 2+ incoming edges having expected_sign,
        # flag if signs conflict (positive vs negative to same node).
        incoming: dict[str, list[tuple[str, str]]] = {}  # node -> [(edge_id, sign)]
        for eid, edge in self.edges.items():
            to_node = edge.get("to", "")
            sign = (
                edge.get("acceptance_criteria", {})
                .get("plausibility", {})
                .get("expected_sign", "")
            )
            if sign and to_node:
                incoming.setdefault(to_node, []).append((eid, sign))

        for node_id, edge_signs in incoming.items():
            if len(edge_signs) < 2:
                continue
            signs_set = {s for _, s in edge_signs if s in ("positive", "negative")}
            if len(signs_set) > 1:
                involved = [eid for eid, _ in edge_signs]
                findings.append({
                    "probe": "sign_coherence",
                    "finding": (
                        f"Node '{node_id}' has conflicting incoming signs: "
                        + ", ".join(f"{eid}={s}" for eid, s in edge_signs)
                    ),
                    "edges_involved": involved,
                    "severity": "warning",
                })

        # ── Probe 2: scale_consistency ──
        # Flag non-unity scales and unit-kind mismatches on consecutive edges.
        for eid, edge in self.edges.items():
            unit_spec = edge.get("unit_specification", {})
            tu = unit_spec.get("treatment_unit", "")
            ou = unit_spec.get("outcome_unit", "")
            for label, text in [("treatment_unit", tu), ("outcome_unit", ou)]:
                if not text:
                    continue
                parsed = UnitSpec.parse(text)
                if parsed.scale != 1.0:
                    findings.append({
                        "probe": "scale_note",
                        "finding": (
                            f"Edge '{eid}' {label}='{text}' has "
                            f"non-unity scale={parsed.scale}"
                        ),
                        "edges_involved": [eid],
                        "severity": "info",
                    })

        # Check unit-kind mismatch on consecutive edges (A->B, B->C)
        outgoing: dict[str, list[str]] = {}  # node -> [edge_id]
        for eid, edge in self.edges.items():
            fn = edge.get("from", "")
            if fn:
                outgoing.setdefault(fn, []).append(eid)

        for eid_up, edge_up in self.edges.items():
            mid_node = edge_up.get("to", "")
            ou_text = edge_up.get("unit_specification", {}).get("outcome_unit", "")
            if not mid_node or not ou_text:
                continue
            ou_kind = UnitSpec.parse(ou_text).kind
            for eid_down in outgoing.get(mid_node, []):
                tu_text = self.edges[eid_down].get("unit_specification", {}).get(
                    "treatment_unit", ""
                )
                if not tu_text:
                    continue
                tu_kind = UnitSpec.parse(tu_text).kind
                if ou_kind != "unknown" and tu_kind != "unknown" and ou_kind != tu_kind:
                    findings.append({
                        "probe": "scale_consistency",
                        "finding": (
                            f"Unit-kind mismatch: {eid_up} outcome_unit "
                            f"kind='{ou_kind}' vs {eid_down} treatment_unit "
                            f"kind='{tu_kind}'"
                        ),
                        "edges_involved": [eid_up, eid_down],
                        "severity": "warning",
                    })

        # ── Probe 3: magnitude_plausibility ──
        # Only check edges that exist in the current DAG (skip orphan cards)
        for eid, est in estimates.items():
            if eid not in self.edges:
                continue
            point = est.get("point")
            pvalue = est.get("pvalue")
            if point is None:
                continue
            abs_point = abs(point)
            if abs_point > 10:
                findings.append({
                    "probe": "magnitude_flag",
                    "finding": (
                        f"Edge '{eid}' has |coefficient|={abs_point:.3f} "
                        f"(> 10, possibly mis-scaled)"
                    ),
                    "edges_involved": [eid],
                    "severity": "warning",
                })
            elif abs_point < 0.001 and pvalue is not None and pvalue < 0.05:
                findings.append({
                    "probe": "magnitude_flag",
                    "finding": (
                        f"Edge '{eid}' has |coefficient|={abs_point:.6f} "
                        f"with p={pvalue:.4f} (negligible but significant)"
                    ),
                    "edges_involved": [eid],
                    "severity": "info",
                })

        # ── Probe 4: sign_vs_estimate ──
        for eid, edge in self.edges.items():
            expected = (
                edge.get("acceptance_criteria", {})
                .get("plausibility", {})
                .get("expected_sign", "")
            )
            if not expected or expected == "any":
                continue
            est = estimates.get(eid)
            if not est or est.get("point") is None:
                continue
            point = est["point"]
            actual_sign = "positive" if point > 0 else "negative" if point < 0 else "zero"
            if (
                (expected == "positive" and point < 0)
                or (expected == "negative" and point > 0)
            ):
                findings.append({
                    "probe": "sign_vs_estimate",
                    "finding": (
                        f"Edge '{eid}' expected_sign='{expected}' but "
                        f"estimated coefficient={point:.4f} ({actual_sign})"
                    ),
                    "edges_involved": [eid],
                    "severity": "warning",
                })

        # ── Probe 5: derived_frequency_mismatch ──
        # For derived nodes with identity formulas, check if all dependency
        # nodes share the same declared frequency.
        _FREQ_RANK = {"daily": 1, "monthly": 2, "quarterly": 3, "annual": 4}
        for nid, node in self.nodes.items():
            if not node.get("derived"):
                continue
            identity = node.get("identity", {})
            deps = identity.get("depends_on") or node.get("depends_on", [])
            if len(deps) < 2:
                continue
            dep_freqs: dict[str, str] = {}
            for dep in deps:
                dep_node = self.nodes.get(dep)
                if dep_node:
                    freq = dep_node.get("frequency", "")
                    if freq:
                        dep_freqs[dep] = freq
            if len(dep_freqs) < 2:
                continue
            unique_freqs = set(dep_freqs.values())
            if len(unique_freqs) > 1:
                detail = ", ".join(f"{d}={f}" for d, f in dep_freqs.items())
                findings.append({
                    "probe": "derived_frequency_mismatch",
                    "finding": (
                        f"Derived node '{nid}' formula mixes frequencies: "
                        f"{detail}. Arithmetic on mismatched frequencies "
                        f"produces nonsensical results."
                    ),
                    "edges_involved": [
                        eid for eid, e in self.edges.items()
                        if e.get("to") == nid
                    ],
                    "severity": "error",
                })

        # ── Probe 6: treatment_outcome_scale_gap ──
        # Detect edges where unit kinds suggest different data magnitudes
        # (e.g., decimal growth rate treatment with percentage-form outcome).
        _DECIMAL_KINDS = {"ratio", "log_point"}
        _PERCENT_KINDS = {"pct", "pp", "bps"}
        for eid, edge in self.edges.items():
            unit_spec = edge.get("unit_specification", {})
            tu = unit_spec.get("treatment_unit", "")
            ou = unit_spec.get("outcome_unit", "")
            if not tu or not ou:
                continue
            tu_parsed = UnitSpec.parse(tu)
            ou_parsed = UnitSpec.parse(ou)
            # Flag when one side is decimal-scale and the other is pct-scale
            tu_is_decimal = tu_parsed.kind in _DECIMAL_KINDS
            ou_is_decimal = ou_parsed.kind in _DECIMAL_KINDS
            tu_is_pct = tu_parsed.kind in _PERCENT_KINDS
            ou_is_pct = ou_parsed.kind in _PERCENT_KINDS
            if (tu_is_decimal and ou_is_pct) or (tu_is_pct and ou_is_decimal):
                findings.append({
                    "probe": "treatment_outcome_scale_gap",
                    "finding": (
                        f"Edge '{eid}' has scale gap: "
                        f"treatment_unit='{tu}' (kind={tu_parsed.kind}) vs "
                        f"outcome_unit='{ou}' (kind={ou_parsed.kind}). "
                        f"Decimal vs percentage scale difference (~100x) "
                        f"inflates coefficient magnitudes."
                    ),
                    "edges_involved": [eid],
                    "severity": "warning",
                })

        return findings

    # ──────────────────────────────────────────────────────────────
    # Critique (LLM call)
    # ──────────────────────────────────────────────────────────────

    def critique(
        self,
        feedback: CriticFeedback,
        iteration: int = 0,
    ) -> list[CriticRevision]:
        """Send feedback to LLM and get proposed revisions.

        Dispatches to backend: claude-code, codex, or api.
        All backends receive the same prompt and return JSON.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(feedback, iteration)

        if self.backend == "claude-code":
            raw = self._call_claude_code(system_prompt, user_prompt)
        elif self.backend == "codex":
            raw = self._call_codex(system_prompt, user_prompt)
        elif self.backend == "api":
            raw = self._call_api(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown critic backend: {self.backend}")

        return self._parse_revisions(raw)

    def _build_system_prompt(self) -> str:
        """Format system prompt from principles YAML."""
        rules = self.principles.get("structural_rules", {})
        rules_text = _format_rules(rules)

        questions = self.principles.get("domain_extraction_questions", {})
        questions_text = "\n".join(
            f"- **{key}**: {val}" for key, val in questions.items()
        )

        gap_rules = self.principles.get("structural_gap_rules", [])
        gap_rules_text = _format_gap_rules(gap_rules)

        probes = self.principles.get("causal_logic_probes", {})
        probes_text = "\n".join(
            f"- **{key}**: {val}" for key, val in probes.items()
        ) if probes else "No causal logic probes configured."

        return DAG_CRITIC_SYSTEM.format(
            structural_rules=rules_text,
            domain_questions=questions_text,
            structural_gap_rules=gap_rules_text,
            causal_logic_probes_questions=probes_text,
            confidence_threshold=self.confidence_threshold,
        )

    def _build_user_prompt(self, feedback: CriticFeedback, iteration: int) -> str:
        """Format user prompt from feedback data."""
        qs = feedback.quality_score

        # Edge diagnostics table
        diag_lines = []
        for d in feedback.edge_diagnostics:
            flags = []
            if not d["has_identification"]:
                flags.append("NO_ID")
            if not d["has_units"]:
                flags.append("NO_UNITS")
            if not d["has_expected_sign"]:
                flags.append("NO_SIGN")
            flag_str = ", ".join(flags) if flags else "OK"
            diag_lines.append(
                f"- {d['edge_id']}: {d['from']} -> {d['to']} "
                f"[{d['edge_type']}] {flag_str}"
            )
        edge_diagnostics_str = "\n".join(diag_lines) or "None"

        # Open issues
        issue_lines = [
            f"- [{i['severity']}] {i['edge_id']}: {i['rule_id']} — {i['message']}"
            for i in feedback.open_issues
        ]
        open_issues_str = "\n".join(issue_lines) or "None"

        # Missing ID
        missing_id_str = ", ".join(feedback.edges_missing_identification) or "None"

        # Missing units
        missing_units_str = ", ".join(feedback.edges_missing_units) or "None"

        # Formula violations
        formula_lines = [
            f"- {v['node_id']}: {v['issue']} (formula: {v['formula']})"
            for v in feedback.formula_violations
        ]
        formula_str = "\n".join(formula_lines) or "None"

        # Node summary
        node_lines = [
            f"- {nid}: {n.get('name', '')} [{n.get('unit', '')}] "
            f"{'(derived)' if n.get('derived') else '(observed)'}"
            for nid, n in self.nodes.items()
        ]
        node_summary_str = "\n".join(node_lines)

        # Literature summary (from edge cards)
        literature_str = self._collect_literature_summary()

        # Structural gaps
        gap_lines = [
            f"- [{g['rule']}] {g['node_id']}: {g['context']}"
            for g in feedback.structural_gaps
        ]
        structural_gaps_str = "\n".join(gap_lines) or "None"

        # Causal logic probes
        probe_lines = [
            f"- [{p['severity'].upper()}] **{p['probe']}**: {p['finding']} "
            f"(edges: {', '.join(p['edges_involved'])})"
            for p in feedback.causal_logic_probes
        ]
        causal_logic_probes_str = "\n".join(probe_lines) or "None"

        return DAG_CRITIC_USER.format(
            iteration=iteration,
            edge_type_diversity=qs.edge_type_diversity,
            identification_coverage=qs.identification_coverage,
            metadata_completeness=qs.metadata_completeness,
            structural_soundness=qs.structural_soundness,
            unit_completeness=qs.unit_completeness,
            formula_correctness=qs.formula_correctness,
            total=qs.total,
            edge_diagnostics=edge_diagnostics_str,
            open_issues=open_issues_str,
            edges_missing_identification=missing_id_str,
            edges_missing_units=missing_units_str,
            formula_violations=formula_str,
            structural_gaps=structural_gaps_str,
            causal_logic_probes=causal_logic_probes_str,
            placebo_results="None",  # populated when placebo data available
            node_summary=node_summary_str,
            literature_summary=literature_str,
        )

    def _collect_literature_summary(self) -> str:
        """Collect literature evidence from edge cards for Layer 2 context."""
        cards_dir = self.output_dir / "cards" / "edge_cards"
        if not cards_dir.exists():
            return "No edge cards available yet."

        summaries = []
        for card_path in sorted(cards_dir.glob("*.yaml")):
            try:
                with open(card_path) as f:
                    card_data = yaml.safe_load(f)
                lit = card_data.get("literature", {})
                if not lit or lit.get("search_status") != "SEARCHED":
                    continue
                edge_id = card_data.get("edge_id", card_path.stem)
                supporting = lit.get("supporting", [])
                challenging = lit.get("challenging", [])
                if supporting or challenging:
                    lines = [f"### {edge_id}"]
                    for paper in supporting[:3]:
                        title = paper.get("title", "Unknown")
                        excerpt = paper.get("excerpt", "")[:200]
                        lines.append(f"  [+] {title}: {excerpt}")
                    for paper in challenging[:2]:
                        title = paper.get("title", "Unknown")
                        excerpt = paper.get("excerpt", "")[:200]
                        lines.append(f"  [-] {title}: {excerpt}")
                    summaries.append("\n".join(lines))
            except (yaml.YAMLError, OSError):
                continue

        return "\n\n".join(summaries) if summaries else "No literature evidence available yet."

    # ── Backend dispatch ──────────────────────────────────────────

    def _call_claude_code(self, system: str, user: str) -> str:
        """Shell out to claude CLI."""
        prompt = f"System:\n{system}\n\nUser:\n{user}"
        try:
            result = subprocess.run(
                ["claude", "-p", prompt, "--output-format", "text"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                logger.warning(f"claude CLI failed: {result.stderr[:200]}")
            return result.stdout.strip()
        except FileNotFoundError:
            logger.warning("claude CLI not found, falling back to api backend")
            return self._call_api(system, user)
        except subprocess.TimeoutExpired:
            logger.warning("claude CLI timed out")
            return "[]"

    def _call_codex(self, system: str, user: str) -> str:
        """Shell out to codex CLI."""
        prompt = f"System:\n{system}\n\nUser:\n{user}"
        try:
            result = subprocess.run(
                ["codex", "exec", "--full-auto", prompt],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                logger.warning(f"codex CLI failed: {result.stderr[:200]}")
            return result.stdout.strip()
        except FileNotFoundError:
            logger.warning("codex CLI not found, falling back to api backend")
            return self._call_api(system, user)
        except subprocess.TimeoutExpired:
            logger.warning("codex CLI timed out")
            return "[]"

    def _call_api(self, system: str, user: str) -> str:
        """Direct API call via shared/llm/client.py."""
        from shared.llm.client import get_llm_client

        client = get_llm_client()
        return client.complete(system, user, max_tokens=4096)

    def _parse_revisions(self, raw_response: str) -> list[CriticRevision]:
        """Parse LLM response into CriticRevision objects."""
        # Extract JSON array from response
        json_str = _extract_json_array(raw_response)
        if not json_str:
            logger.warning("No JSON array found in critic response")
            return []

        try:
            items = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse critic JSON: {e}")
            return []

        if not isinstance(items, list):
            logger.warning("Critic response is not a JSON array")
            return []

        revisions = []
        for item in items:
            try:
                rev = CriticRevision.from_dict(item)
                revisions.append(rev)
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping malformed revision: {e}")
                continue

        return revisions

    # ──────────────────────────────────────────────────────────────
    # Apply Revisions
    # ──────────────────────────────────────────────────────────────

    def apply_revisions(
        self,
        revisions: list[CriticRevision],
    ) -> tuple[list[CriticRevision], list[CriticRevision]]:
        """Validate and apply revisions to DAG YAML.

        Returns:
            (applied, rejected) tuple of revision lists.
        """
        validator = CriticRevisionValidator(
            self.dag, confidence_threshold=self.confidence_threshold
        )
        validated = validator.validate_batch(revisions)

        applied: list[CriticRevision] = []
        rejected: list[CriticRevision] = []

        for rev in validated:
            if not rev.validated:
                rejected.append(rev)
                logger.info(
                    f"Rejected revision {rev.revision_type.value} on "
                    f"{rev.target_edge_id}: {rev.rejection_reason}"
                )
                continue

            try:
                self._apply_single(rev)
                applied.append(rev)
                logger.info(
                    f"Applied revision {rev.revision_type.value} on "
                    f"{rev.target_edge_id}"
                )
            except Exception as e:
                rev.validated = False
                rev.rejection_reason = f"Application error: {e}"
                rejected.append(rev)
                logger.warning(
                    f"Failed to apply revision {rev.revision_type.value} on "
                    f"{rev.target_edge_id}: {e}"
                )

        # Write updated DAG
        if applied:
            self._write_dag()
            self.reload_dag()

        return applied, rejected

    def _apply_single(self, rev: CriticRevision) -> None:
        """Apply a single validated revision to the in-memory DAG."""
        handler = getattr(self, f"_apply_{rev.revision_type.value}", None)
        if handler is None:
            raise ValueError(f"No handler for {rev.revision_type.value}")
        handler(rev)

    def _apply_upgrade_edge_type(self, rev: CriticRevision) -> None:
        """Change edge_type."""
        for edge in self.dag["edges"]:
            if edge["id"] == rev.target_edge_id:
                edge["edge_type"] = rev.details["new_edge_type"]
                break

    def _apply_add_identification_strategy(self, rev: CriticRevision) -> None:
        """Add identification_strategy to edge."""
        for edge in self.dag["edges"]:
            if edge["id"] == rev.target_edge_id:
                edge["identification_strategy"] = rev.details["identification_strategy"]
                break

    def _apply_fix_formula(self, rev: CriticRevision) -> None:
        """Fix node formula."""
        node_id = rev.details["node_id"]
        for node in self.dag["nodes"]:
            if node["id"] == node_id:
                if "identity" not in node:
                    node["identity"] = {}
                node["identity"]["formula"] = rev.details["new_formula"]
                break

    def _apply_add_missing_node(self, rev: CriticRevision) -> None:
        """Add a new node to the DAG."""
        node_def = rev.details["node_definition"]
        self.dag.setdefault("nodes", []).append({
            "id": node_def["id"],
            "name": node_def.get("name", node_def["id"]),
            "unit": node_def.get("unit", "level"),
            "description": node_def.get("description", ""),
            "type": node_def.get("type", "continuous"),
            "observed": True,
            "frequency": node_def.get("frequency", "M"),
            "tags": ["critic_generated"],
        })

    def _apply_add_missing_edge(self, rev: CriticRevision) -> None:
        """Add a new edge to the DAG."""
        d = rev.details
        from_node = d["from_node"]
        to_node = d["to_node"]
        edge_id = f"{from_node}_to_{to_node}"
        self.dag.setdefault("edges", []).append({
            "id": edge_id,
            "from": from_node,
            "to": to_node,
            "edge_type": d.get("edge_type", "causal"),
            "notes": f"Critic-added: {d.get('mechanism', '')}",
        })

    def _apply_add_structural_metadata(self, rev: CriticRevision) -> None:
        """Add metadata to edge."""
        meta = rev.details.get("metadata", {})
        for edge in self.dag["edges"]:
            if edge["id"] == rev.target_edge_id:
                # expected_sign
                if "expected_sign" in meta:
                    edge.setdefault("acceptance_criteria", {}).setdefault(
                        "plausibility", {}
                    )["expected_sign"] = meta["expected_sign"]
                # timing
                if "timing" in meta:
                    edge.setdefault("timing", {}).update(meta["timing"])
                # forbidden_controls
                if "forbidden_controls" in meta:
                    edge["forbidden_controls"] = meta["forbidden_controls"]
                break

    def _apply_add_unit_specification(self, rev: CriticRevision) -> None:
        """Add unit_specification to edge."""
        units = rev.details.get("unit_specification", {})
        for edge in self.dag["edges"]:
            if edge["id"] == rev.target_edge_id:
                spec = edge.setdefault("unit_specification", {})
                if units.get("treatment_unit"):
                    spec["treatment_unit"] = units["treatment_unit"]
                if units.get("outcome_unit"):
                    spec["outcome_unit"] = units["outcome_unit"]
                break

    def _apply_invert_edge_direction(self, rev: CriticRevision) -> None:
        """Swap from and to on an edge."""
        for edge in self.dag["edges"]:
            if edge["id"] == rev.target_edge_id:
                edge["from"], edge["to"] = edge["to"], edge["from"]
                # Update edge ID to reflect new direction
                edge["id"] = f"{edge['from']}_to_{edge['to']}"
                break

    def _write_dag(self) -> None:
        """Write current DAG state back to YAML file."""
        with open(self.dag_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.dag, f,
                sort_keys=False, allow_unicode=True, default_flow_style=False,
            )
        logger.info(f"Wrote updated DAG to {self.dag_path}")


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _has_identification_strategy(edge: dict) -> bool:
    """Check if edge has an identification strategy."""
    strategy = edge.get("identification_strategy", {})
    if isinstance(strategy, dict) and strategy.get("type"):
        return strategy["type"] != "none"
    return False


def _has_double_log(node: dict, all_nodes: dict[str, dict]) -> bool:
    """Check if a node's identity formula double-logs any dependency."""
    identity = node.get("identity")
    if not identity:
        return False
    formula = identity.get("formula", "")
    if not formula:
        return False
    deps = identity.get("depends_on") or node.get("depends_on", [])
    for dep in deps:
        dep_node = all_nodes.get(dep)
        if not dep_node:
            continue
        if "log" in dep_node.get("transforms", []) and f"log({dep})" in formula:
            return True
    return False


def _format_rules(rules: dict) -> str:
    """Format structural rules from YAML into numbered text."""
    lines = []
    for category, rule_list in rules.items():
        lines.append(f"\n### {category.replace('_', ' ').title()}")
        for rule in rule_list:
            rid = rule.get("id", "")
            pred = rule.get("predicate", "")
            req = rule.get("required", "")
            sev = rule.get("severity", "warning")
            lines.append(f"- [{sev.upper()}] **{rid}**: IF {pred} THEN {req}")
    return "\n".join(lines)


def _format_gap_rules(gap_rules: list[dict]) -> str:
    """Format structural gap rules from YAML into numbered text."""
    if not gap_rules:
        return "No structural gap rules configured."
    lines = []
    for rule in gap_rules:
        rid = rule.get("id", "")
        pred = rule.get("predicate", "")
        signal = rule.get("signal", "")
        sev = rule.get("severity", "info")
        lines.append(f"- [{sev.upper()}] **{rid}**: IF {pred} THEN {signal}")
    return "\n".join(lines)


def _extract_json_array(text: str) -> str | None:
    """Extract a JSON array from text, handling markdown fences."""
    # Try direct parse
    text = text.strip()
    if text.startswith("["):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

    # Try markdown fence
    match = re.search(r"```(?:json)?\s*\n?(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(1))
            return match.group(1)
        except json.JSONDecodeError:
            pass

    # Try finding first [ ... ] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(0))
            return match.group(0)
        except json.JSONDecodeError:
            pass

    return None
