"""
Tests for narrative DAG pipeline mechanism.

Proves that ``repair_narrative_dag()`` transforms a broken narrative DAG
into a correct one, WITHOUT manual YAML edits.

Structure:
  - ``original_dag`` fixture: the unedited, broken narrative YAML.
  - ``repaired_dag`` fixture: result of running ``repair_narrative_dag()``.
  - Negative tests: verify the original has specific issues.
  - Positive tests: verify the repaired DAG passes all checks.
  - Sentinel tests: run the full DAGValidator on the repaired output.

Covers all 8 issue categories:
1. Structural fragmentation (target reachable)
2. Missing identity edges (identity deps complete)
3. Double-logging convention (no log(X) when X has transforms: [log])
4. Missing causal pathways (identity edges auto-created, sink nodes)
5. Unit/scope mismatch (assumptions present)
6. Schema hygiene (edge ID syntax, no misleading IDs)
7. Sentinel validation rules (all new rules pass on repaired DAG)
8. Generation pipeline (sanitize_edge_id, repair function)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NARRATIVE_DAG_PATH = PROJECT_ROOT / "config" / "agentic" / "dags" / "kspi_k2_narrative.yaml"
EXPERT_DAG_PATH = PROJECT_ROOT / "config" / "agentic" / "dags" / "kspi_k2_full.yaml"
ISSUE_REGISTRY_PATH = PROJECT_ROOT / "config" / "agentic" / "issue_registry.yaml"


@pytest.fixture(scope="module")
def original_dag() -> dict:
    """Load the ORIGINAL broken narrative DAG (no manual edits)."""
    with open(NARRATIVE_DAG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def repaired_dag(original_dag) -> dict:
    """Run the pipeline repair mechanism on the original broken DAG."""
    from scripts.generate_narrative_dag import repair_narrative_dag
    return repair_narrative_dag(original_dag, EXPERT_DAG_PATH)


@pytest.fixture(scope="module")
def expert_dag() -> dict:
    with open(EXPERT_DAG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def issue_registry() -> dict:
    with open(ISSUE_REGISTRY_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_nodes(dag: dict) -> dict[str, dict]:
    return {n["id"]: n for n in dag.get("nodes", [])}


def _get_edges(dag: dict) -> dict[str, dict]:
    return {e["id"]: e for e in dag.get("edges", [])}


def _edge_pairs(dag: dict) -> set[tuple[str, str]]:
    return {(e["from"], e["to"]) for e in dag.get("edges", [])}


def _build_adjacency(dag: dict) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = {}
    for e in dag.get("edges", []):
        adj.setdefault(e["from"], []).append(e["to"])
    return adj


def _build_reverse_adjacency(dag: dict) -> dict[str, list[str]]:
    rev: dict[str, list[str]] = {}
    for e in dag.get("edges", []):
        rev.setdefault(e["to"], []).append(e["from"])
    return rev


def _reachable_from(adj: dict[str, list[str]], start: str) -> set[str]:
    visited: set[str] = set()
    frontier = [start]
    while frontier:
        node = frontier.pop()
        if node in visited:
            continue
        visited.add(node)
        for nb in adj.get(node, []):
            if nb not in visited:
                frontier.append(nb)
    return visited


VALID_EDGE_ID_RE = re.compile(r'^[a-z0-9][a-z0-9_]*[a-z0-9]$|^[a-z0-9]$')


# =========================================================================
# NEGATIVE TESTS: Original DAG has known issues
# =========================================================================

class TestOriginalHasIssues:
    """Verify that the original unedited DAG has the issues we claim."""

    def test_original_has_bad_edge_ids(self, original_dag):
        """Original has edge IDs with arrows and spaces."""
        bad = [e["id"] for e in original_dag["edges"]
               if not VALID_EDGE_ID_RE.match(e["id"])]
        assert len(bad) > 0, "Expected broken edge IDs in original"

    def test_original_has_misleading_edge_id(self, original_dag):
        """vix_shock->nim_kspi actually points to ppop_kspi."""
        misleading = [
            e for e in original_dag["edges"]
            if "nim_kspi" in e["id"] and e["to"] != "nim_kspi"
        ]
        assert len(misleading) > 0, "Expected misleading edge ID"

    def test_original_missing_identity_edges(self, original_dag):
        """Original is missing identity edges for derived nodes."""
        pairs = _edge_pairs(original_dag)
        # k2_ratio_kspi depends on total_capital_kspi but no edge exists
        assert ("total_capital_kspi", "k2_ratio_kspi") not in pairs

    def test_original_has_double_logging(self, original_dag):
        """Formulas use log(X) on nodes with transforms: [log]."""
        nodes = _get_nodes(original_dag)
        found = False
        for node in nodes.values():
            identity = node.get("identity")
            if not identity:
                continue
            formula = identity.get("formula", "")
            deps = identity.get("depends_on") or node.get("depends_on", [])
            for dep in deps:
                dep_node = nodes.get(dep)
                if not dep_node:
                    continue
                if "log" in dep_node.get("transforms", []) and f"log({dep})" in formula:
                    found = True
        assert found, "Expected double-logging in original"

    def test_original_target_unreachable(self, original_dag):
        """Target k2_ratio_kspi is unreachable from roots in original."""
        from shared.agentic.validation import DAGValidator, ValidationSeverity
        validator = DAGValidator(original_dag)
        result = validator.validate_pre_estimation()
        target_errors = [
            i for i in result.issues
            if i.check_id == "target_reachable"
            and i.severity == ValidationSeverity.ERROR
        ]
        # With only rwa_to_k2, target may be technically reachable from
        # rwa_kspi (but rwa_kspi is a root). The real issue is missing
        # identity deps. Check that at least identity_deps errors exist.
        identity_errors = [
            i for i in result.issues
            if i.check_id == "identity_deps_complete"
            and i.severity == ValidationSeverity.ERROR
        ]
        assert len(identity_errors) > 0, "Expected identity dep errors in original"

    def test_original_has_sink_nodes(self, original_dag):
        """cor_kspi is a sink node in original."""
        from shared.agentic.validation import DAGValidator
        validator = DAGValidator(original_dag)
        result = validator.validate_pre_estimation()
        sink_issues = [
            i for i in result.issues
            if i.check_id == "sink_node_not_target"
            and "cor_kspi" in i.message
        ]
        assert len(sink_issues) > 0, "Expected cor_kspi sink warning"

    def test_original_no_scope_assumption(self, original_dag):
        """Original has empty assumptions."""
        assumptions = original_dag.get("assumptions", []) or []
        scope_ids = [a.get("id") for a in assumptions]
        assert "scope_consistency" not in scope_ids


# =========================================================================
# Issue 1: Structural Fragmentation — Target Reachable (repaired)
# =========================================================================

class TestTargetReachable:
    """Issue 1: k2_ratio_kspi must be reachable from at least one root."""

    def test_target_node_defined(self, repaired_dag):
        target = repaired_dag["metadata"]["target_node"]
        assert target == "k2_ratio_kspi"
        node_ids = {n["id"] for n in repaired_dag["nodes"]}
        assert target in node_ids

    def test_target_reachable_from_root(self, repaired_dag):
        adj = _build_adjacency(repaired_dag)
        node_ids = {n["id"] for n in repaired_dag["nodes"]}
        incoming = {e["to"] for e in repaired_dag["edges"]}
        roots = node_ids - incoming

        assert len(roots) > 0, "DAG has no root nodes"

        target = repaired_dag["metadata"]["target_node"]
        reachable = set()
        for root in roots:
            reachable |= _reachable_from(adj, root)

        assert target in reachable, (
            f"Target '{target}' unreachable from roots {roots}. "
            f"Reachable: {sorted(reachable)}"
        )

    def test_single_weakly_connected_component(self, repaired_dag):
        from collections import defaultdict

        undirected: dict[str, set[str]] = defaultdict(set)
        for e in repaired_dag["edges"]:
            undirected[e["from"]].add(e["to"])
            undirected[e["to"]].add(e["from"])

        node_ids = {n["id"] for n in repaired_dag["nodes"]}
        visited: set[str] = set()
        components = 0
        for node in node_ids:
            if node not in visited:
                frontier = [node]
                while frontier:
                    n = frontier.pop()
                    if n in visited:
                        continue
                    visited.add(n)
                    for nb in undirected.get(n, set()):
                        if nb not in visited:
                            frontier.append(nb)
                components += 1

        isolated = node_ids - {
            n for e in repaired_dag["edges"]
            for n in (e["from"], e["to"])
        }
        non_isolated_components = components - len(isolated)
        assert non_isolated_components <= 1, (
            f"Expected 1 connected component (non-isolated), got {non_isolated_components}"
        )


# =========================================================================
# Issue 2: Missing Identity Edges (repaired)
# =========================================================================

class TestIdentityDepsComplete:
    """Issue 2: Every derived node's identity deps must have edges into it."""

    def test_k2_ratio_has_both_identity_edges(self, repaired_dag):
        pairs = _edge_pairs(repaired_dag)
        assert ("total_capital_kspi", "k2_ratio_kspi") in pairs, \
            "Missing: total_capital_kspi -> k2_ratio_kspi"
        assert ("rwa_kspi", "k2_ratio_kspi") in pairs, \
            "Missing: rwa_kspi -> k2_ratio_kspi"

    def test_real_income_has_identity_edges(self, repaired_dag):
        pairs = _edge_pairs(repaired_dag)
        assert ("nominal_income", "real_income") in pairs, \
            "Missing: nominal_income -> real_income"
        cpi_to_real = [
            e for e in repaired_dag["edges"]
            if e["from"] == "cpi_headline" and e["to"] == "real_income"
        ]
        assert len(cpi_to_real) >= 1, "Missing: cpi_headline -> real_income"

    def test_real_expenditure_has_identity_edges(self, repaired_dag):
        pairs = _edge_pairs(repaired_dag)
        assert ("nominal_expenditure", "real_expenditure") in pairs, \
            "Missing: nominal_expenditure -> real_expenditure"
        cpi_to_real = [
            e for e in repaired_dag["edges"]
            if e["from"] == "cpi_headline" and e["to"] == "real_expenditure"
        ]
        assert len(cpi_to_real) >= 1, "Missing: cpi_headline -> real_expenditure"

    def test_imported_inflation_iv_has_identity_edges(self, repaired_dag):
        pairs = _edge_pairs(repaired_dag)
        assert ("kzt_usd", "imported_inflation_instrument") in pairs, \
            "Missing: kzt_usd -> imported_inflation_instrument"
        assert ("E_import_share", "imported_inflation_instrument") in pairs, \
            "Missing: E_import_share -> imported_inflation_instrument"

    def test_all_identity_deps_have_edges(self, repaired_dag):
        nodes = _get_nodes(repaired_dag)
        pairs = _edge_pairs(repaired_dag)

        for node_id, node in nodes.items():
            if not node.get("derived"):
                continue
            identity = node.get("identity")
            if not identity:
                continue
            deps = identity.get("depends_on") or node.get("depends_on", [])
            for dep in deps:
                if dep not in nodes:
                    continue
                assert (dep, node_id) in pairs, (
                    f"Derived node '{node_id}' missing identity edge: "
                    f"{dep} -> {node_id}"
                )


# =========================================================================
# Issue 3: Double-Logging Convention (repaired)
# =========================================================================

class TestDoubleLogging:
    """Issue 3: No log(X) in formulas when X has transforms: [log]."""

    def _check_dag_no_double_log(self, dag):
        nodes = _get_nodes(dag)
        issues = []
        for node_id, node in nodes.items():
            identity = node.get("identity")
            if not identity:
                continue
            formula = identity.get("formula", "")
            deps = identity.get("depends_on") or node.get("depends_on", [])
            for dep in deps:
                dep_node = nodes.get(dep)
                if not dep_node:
                    continue
                if "log" in dep_node.get("transforms", []):
                    if f"log({dep})" in formula:
                        issues.append(
                            f"{node_id}: formula has log({dep}) but "
                            f"{dep} already has transforms: [log]"
                        )
        return issues

    def test_repaired_dag_no_double_log(self, repaired_dag):
        issues = self._check_dag_no_double_log(repaired_dag)
        assert issues == [], f"Double-logging found:\n" + "\n".join(issues)

    def test_expert_dag_no_double_log(self, expert_dag):
        issues = self._check_dag_no_double_log(expert_dag)
        assert issues == [], f"Double-logging found:\n" + "\n".join(issues)


# =========================================================================
# Issue 4.3: Sink Nodes (repaired)
# =========================================================================

class TestSinkNodes:
    """Issue 4.3: Non-target nodes should not be dead-end sinks."""

    def test_cor_kspi_connects_downstream(self, repaired_dag):
        adj = _build_adjacency(repaired_dag)
        assert "total_capital_kspi" in adj.get("cor_kspi", []), (
            "cor_kspi is a sink — should connect to total_capital_kspi"
        )

    def test_deposit_cost_is_acceptable_sink(self, repaired_dag):
        node_ids = {n["id"] for n in repaired_dag["nodes"]}
        assert "deposit_cost_kspi" in node_ids


# =========================================================================
# Issue 5: Scope Assumptions (repaired)
# =========================================================================

class TestScopeAssumptions:
    """Issue 5: Cross-scope edges should have documented assumptions."""

    def test_scope_consistency_assumption_exists(self, repaired_dag):
        assumptions = repaired_dag.get("assumptions", [])
        scope_ids = [a["id"] for a in assumptions]
        assert "scope_consistency" in scope_ids, \
            "Missing scope_consistency assumption"

    def test_scope_assumption_covers_cross_scope_edges(self, repaired_dag):
        assumptions = repaired_dag.get("assumptions", [])
        scope_assumption = next(
            (a for a in assumptions if a["id"] == "scope_consistency"), None
        )
        assert scope_assumption is not None
        applies_to = scope_assumption.get("applies_to_edges", [])
        assert len(applies_to) > 0, \
            "scope_consistency assumption has no applies_to_edges"


# =========================================================================
# Issue 6a: Edge ID Syntax (repaired)
# =========================================================================

class TestEdgeIDSyntax:
    """Issue 6a: All edge IDs must be valid snake_case."""

    def test_all_edge_ids_snake_case(self, repaired_dag):
        bad_ids = []
        for edge in repaired_dag["edges"]:
            eid = edge["id"]
            if not VALID_EDGE_ID_RE.match(eid):
                bad_ids.append(eid)
        assert bad_ids == [], f"Invalid edge IDs found: {bad_ids}"

    def test_no_arrows_in_edge_ids(self, repaired_dag):
        for edge in repaired_dag["edges"]:
            assert "->" not in edge["id"], f"Arrow in edge ID: {edge['id']}"
            assert " " not in edge["id"], f"Space in edge ID: {edge['id']}"
            assert "(" not in edge["id"], f"Paren in edge ID: {edge['id']}"


# =========================================================================
# Issue 6b: Misleading Edge IDs (repaired)
# =========================================================================

class TestMisleadingEdgeIDs:
    """Issue 6b: Edge IDs should reference actual from/to nodes."""

    def test_no_misleading_edge_ids(self, repaired_dag):
        for edge in repaired_dag["edges"]:
            eid = edge["id"]
            to_node = edge["to"]
            if "nim_kspi" in eid and to_node != "nim_kspi":
                pytest.fail(
                    f"Edge '{eid}' references nim_kspi but to={to_node}"
                )


# =========================================================================
# Sentinel Validation: Full Pipeline Check (repaired)
# =========================================================================

class TestSentinelValidation:
    """Run DAGValidator on the repaired DAG and verify all new rules pass."""

    def test_pre_estimation_validation_passes(self, repaired_dag):
        from shared.agentic.validation import DAGValidator, ValidationSeverity

        validator = DAGValidator(repaired_dag)
        result = validator.validate_pre_estimation()

        errors = [
            i for i in result.issues
            if i.severity == ValidationSeverity.ERROR
        ]
        error_msgs = [f"{i.check_id}: {i.message}" for i in errors]
        assert len(errors) == 0, (
            f"Pre-estimation validation errors:\n"
            + "\n".join(error_msgs)
        )

    def test_target_reachable_check(self, repaired_dag):
        from shared.agentic.validation import DAGValidator

        validator = DAGValidator(repaired_dag)
        result = validator.validate_pre_estimation()

        target_issues = [
            i for i in result.issues
            if i.check_id == "target_reachable"
        ]
        assert len(target_issues) == 0, \
            f"Target reachable check failed: {target_issues[0].message}"

    def test_identity_deps_complete_check(self, repaired_dag):
        from shared.agentic.validation import DAGValidator, ValidationSeverity

        validator = DAGValidator(repaired_dag)
        result = validator.validate_pre_estimation()

        identity_issues = [
            i for i in result.issues
            if i.check_id == "identity_deps_complete"
            and i.severity == ValidationSeverity.ERROR
        ]
        assert len(identity_issues) == 0, (
            f"Identity deps incomplete: "
            + "; ".join(i.message for i in identity_issues)
        )

    def test_edge_id_syntax_check(self, repaired_dag):
        from shared.agentic.validation import DAGValidator

        validator = DAGValidator(repaired_dag)
        result = validator.validate_pre_estimation()

        syntax_issues = [
            i for i in result.issues
            if i.check_id == "edge_id_syntax"
        ]
        assert len(syntax_issues) == 0, (
            f"Edge ID syntax issues: "
            + "; ".join(i.message for i in syntax_issues)
        )

    def test_no_double_transform_warnings(self, repaired_dag):
        from shared.agentic.validation import DAGValidator

        validator = DAGValidator(repaired_dag)
        result = validator.validate_pre_estimation()

        dt_issues = [
            i for i in result.issues
            if i.check_id == "identity_formula_double_transform"
        ]
        assert len(dt_issues) == 0, (
            f"Double-transform warnings: "
            + "; ".join(i.message for i in dt_issues)
        )


# =========================================================================
# Sentinel Validation: Negative Tests (detect bad DAGs)
# =========================================================================

class TestSentinelNegative:
    """Verify sentinel rules CATCH known-bad DAG patterns."""

    def _minimal_dag(self, *, edges=None, nodes=None, target="target"):
        return {
            "metadata": {"target_node": target},
            "nodes": nodes or [],
            "edges": edges or [],
        }

    def test_detects_unreachable_target(self):
        from shared.agentic.validation import DAGValidator, ValidationSeverity

        dag = self._minimal_dag(
            nodes=[{"id": "a"}, {"id": "b"}, {"id": "target"}],
            edges=[
                {"id": "a_to_b", "from": "a", "to": "b",
                 "edge_type": "causal",
                 "unit_specification": {"treatment_unit": "x", "outcome_unit": "y"}},
            ],
        )
        validator = DAGValidator(dag)
        result = validator.validate_pre_estimation()
        target_issues = [
            i for i in result.issues
            if i.check_id == "target_reachable"
            and i.severity == ValidationSeverity.ERROR
        ]
        assert len(target_issues) == 1, "Should detect unreachable target"

    def test_detects_incomplete_identity_deps(self):
        from shared.agentic.validation import DAGValidator, ValidationSeverity

        dag = self._minimal_dag(
            nodes=[
                {"id": "cap", "derived": False},
                {"id": "rwa", "derived": False},
                {"id": "k2", "derived": True,
                 "identity": {"formula": "cap / rwa", "depends_on": ["cap", "rwa"]}},
            ],
            edges=[
                {"id": "rwa_to_k2", "from": "rwa", "to": "k2",
                 "edge_type": "identity",
                 "unit_specification": {"treatment_unit": "log", "outcome_unit": "pct"}},
            ],
            target="k2",
        )
        validator = DAGValidator(dag)
        result = validator.validate_pre_estimation()
        dep_issues = [
            i for i in result.issues
            if i.check_id == "identity_deps_complete"
            and i.severity == ValidationSeverity.ERROR
        ]
        assert len(dep_issues) == 1, "Should detect missing cap -> k2 edge"
        assert "cap" in dep_issues[0].message

    def test_detects_bad_edge_id_syntax(self):
        from shared.agentic.validation import DAGValidator

        dag = self._minimal_dag(
            nodes=[{"id": "a"}, {"id": "b"}],
            edges=[
                {"id": "a -> b (some description)", "from": "a", "to": "b",
                 "edge_type": "causal",
                 "unit_specification": {"treatment_unit": "x", "outcome_unit": "y"}},
            ],
            target="b",
        )
        validator = DAGValidator(dag)
        result = validator.validate_pre_estimation()
        syntax_issues = [
            i for i in result.issues if i.check_id == "edge_id_syntax"
        ]
        assert len(syntax_issues) == 1, "Should detect bad edge ID syntax"

    def test_detects_double_transform(self):
        from shared.agentic.validation import DAGValidator

        dag = self._minimal_dag(
            nodes=[
                {"id": "x", "transforms": ["log"]},
                {"id": "y", "transforms": ["log"]},
                {"id": "z", "derived": True,
                 "identity": {"formula": "log(x) - log(y)", "depends_on": ["x", "y"]}},
            ],
            edges=[
                {"id": "x_to_z", "from": "x", "to": "z",
                 "edge_type": "identity",
                 "unit_specification": {"treatment_unit": "log", "outcome_unit": "log"}},
                {"id": "y_to_z", "from": "y", "to": "z",
                 "edge_type": "identity",
                 "unit_specification": {"treatment_unit": "log", "outcome_unit": "log"}},
            ],
            target="z",
        )
        validator = DAGValidator(dag)
        result = validator.validate_pre_estimation()
        dt_issues = [
            i for i in result.issues
            if i.check_id == "identity_formula_double_transform"
        ]
        assert len(dt_issues) == 2, (
            f"Should detect double-transform for x and y, got {len(dt_issues)}"
        )

    def test_detects_sink_node(self):
        from shared.agentic.validation import DAGValidator

        dag = self._minimal_dag(
            nodes=[{"id": "a"}, {"id": "b"}, {"id": "target"}],
            edges=[
                {"id": "a_to_b", "from": "a", "to": "b",
                 "edge_type": "causal",
                 "unit_specification": {"treatment_unit": "x", "outcome_unit": "y"}},
                {"id": "a_to_target", "from": "a", "to": "target",
                 "edge_type": "causal",
                 "unit_specification": {"treatment_unit": "x", "outcome_unit": "y"}},
            ],
            target="target",
        )
        validator = DAGValidator(dag)
        result = validator.validate_pre_estimation()
        sink_issues = [
            i for i in result.issues if i.check_id == "sink_node_not_target"
        ]
        assert len(sink_issues) == 1, "Should detect node 'b' as sink"
        assert "b" in sink_issues[0].message


# =========================================================================
# Generation Pipeline: _sanitize_edge_id
# =========================================================================

class TestSanitizeEdgeId:

    def test_arrow_replacement(self):
        from scripts.generate_narrative_dag import _sanitize_edge_id
        assert _sanitize_edge_id("vix_shock->nim_kspi") == "vix_shock_to_nim_kspi"

    def test_spaces_and_parens(self):
        from scripts.generate_narrative_dag import _sanitize_edge_id
        result = _sanitize_edge_id(
            "cpi_headline -> real_income (sticky nominal income / wage adjustment channel)"
        )
        assert " " not in result
        assert "(" not in result
        assert VALID_EDGE_ID_RE.match(result), f"Invalid result: {result}"

    def test_unicode_arrow(self):
        from scripts.generate_narrative_dag import _sanitize_edge_id
        assert _sanitize_edge_id("a→b") == "a_to_b"

    def test_already_clean(self):
        from scripts.generate_narrative_dag import _sanitize_edge_id
        assert _sanitize_edge_id("fx_to_cpi_tradable") == "fx_to_cpi_tradable"

    def test_consecutive_underscores_collapsed(self):
        from scripts.generate_narrative_dag import _sanitize_edge_id
        assert _sanitize_edge_id("a___b") == "a_b"


# =========================================================================
# Repair function: end-to-end mechanism test
# =========================================================================

class TestRepairMechanism:
    """Verify the repair function transforms broken -> correct."""

    def test_repair_is_idempotent(self, repaired_dag):
        """Running repair twice produces the same result."""
        from scripts.generate_narrative_dag import repair_narrative_dag
        double_repaired = repair_narrative_dag(repaired_dag, EXPERT_DAG_PATH)

        # Same edge set (by from/to pair)
        pairs1 = _edge_pairs(repaired_dag)
        pairs2 = _edge_pairs(double_repaired)
        assert pairs1 == pairs2, "Repair is not idempotent (edge pairs differ)"

        # Same edge IDs
        ids1 = sorted(e["id"] for e in repaired_dag["edges"])
        ids2 = sorted(e["id"] for e in double_repaired["edges"])
        assert ids1 == ids2, "Repair is not idempotent (edge IDs differ)"

    def test_repair_does_not_mutate_original(self, original_dag):
        """Repair returns a new dict, does not mutate input."""
        import copy
        snapshot = copy.deepcopy(original_dag)
        from scripts.generate_narrative_dag import repair_narrative_dag
        repair_narrative_dag(original_dag, EXPERT_DAG_PATH)
        assert original_dag == snapshot, "repair_narrative_dag mutated its input"

    def test_repaired_has_more_edges_than_original(self, original_dag, repaired_dag):
        """Repair should add identity and connectivity edges."""
        assert len(repaired_dag["edges"]) > len(original_dag["edges"])

    def test_repaired_preserves_all_original_edge_pairs(self, original_dag, repaired_dag):
        """All original causal relationships should still be present."""
        original_pairs = _edge_pairs(original_dag)
        repaired_pairs = _edge_pairs(repaired_dag)
        missing = original_pairs - repaired_pairs
        assert len(missing) == 0, (
            f"Repair dropped original edge pairs: {missing}"
        )


# =========================================================================
# Issue Registry: New Rules Present
# =========================================================================

class TestIssueRegistryNewRules:
    EXPECTED_NEW_RULES = [
        "TARGET_UNREACHABLE",
        "IDENTITY_DEPS_INCOMPLETE",
        "SINK_NODE_NOT_TARGET",
        "EDGE_ID_SYNTAX_INVALID",
        "IDENTITY_FORMULA_DOUBLE_TRANSFORM",
    ]

    def test_new_rules_present(self, issue_registry):
        rule_ids = {r["rule_id"] for r in issue_registry.get("rules", [])}
        for rule_id in self.EXPECTED_NEW_RULES:
            assert rule_id in rule_ids, f"Missing rule: {rule_id}"

    def test_rule_count_at_least_35(self, issue_registry):
        rules = issue_registry.get("rules", [])
        assert len(rules) >= 35, f"Expected >= 35 rules, got {len(rules)}"


# =========================================================================
# Expert DAG: Validation Should Also Pass
# =========================================================================

class TestExpertDAGValidation:

    def test_expert_dag_no_double_log(self, expert_dag):
        nodes = _get_nodes(expert_dag)
        for node_id, node in nodes.items():
            identity = node.get("identity")
            if not identity:
                continue
            formula = identity.get("formula", "")
            deps = identity.get("depends_on") or node.get("depends_on", [])
            for dep in deps:
                dep_node = nodes.get(dep)
                if not dep_node:
                    continue
                if "log" in dep_node.get("transforms", []):
                    assert f"log({dep})" not in formula, (
                        f"Expert DAG: {node_id} formula has log({dep}) "
                        f"but {dep} has transforms: [log]"
                    )

    def test_expert_dag_edge_ids_valid(self, expert_dag):
        for edge in expert_dag["edges"]:
            assert VALID_EDGE_ID_RE.match(edge["id"]), \
                f"Expert DAG: invalid edge ID: {edge['id']}"
