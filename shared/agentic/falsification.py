"""
Placebo Falsification for Null Links.

The DAG declares which edges exist, but every node pair *without* an edge
is an implicit claim: "no direct effect."  PlaceboFalsifier tests these
null-link claims by running OLS regressions on pairs where the DAG says
no link should exist.  A significant coefficient is a red flag for
mis-specification.

Agentic flow:
1. Enumerate candidate null links from the DAG.
2. Prioritise by structural interestingness (DAG distance, shared neighbors).
3. Data gate — check availability, optionally dispatch DataScout.
4. Run HAC-robust OLS placebo regressions.
5. Write failures to the issue ledger.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlaceboResult:
    """Result of a single placebo falsification test."""

    from_node: str
    to_node: str
    coefficient: float
    se: float
    pvalue: float
    n_obs: int
    passed: bool               # pvalue >= alpha  →  null confirmed
    dag_distance: int          # shortest undirected path, -1 if disconnected
    shared_neighbors: list[str] = field(default_factory=list)
    data_status: str = "available"  # "available" | "fetched" | "missing"
    message: str = ""


# ---------------------------------------------------------------------------
# Falsifier
# ---------------------------------------------------------------------------

class PlaceboFalsifier:
    """Agentic placebo falsification for DAG null links.

    Parameters
    ----------
    dag : DAGSpec
        The DAG under test.
    max_tests : int
        Maximum number of null-link pairs to test.
    alpha : float
        Significance threshold for declaring a failure.
    data_scout : DataScout | None
        If provided, attempt to fetch missing data before skipping.
    issue_ledger : IssueLedger | None
        If provided, significant null links are written as issues.
    """

    def __init__(
        self,
        dag: Any,
        max_tests: int = 20,
        alpha: float = 0.05,
        data_scout: Any | None = None,
        issue_ledger: Any | None = None,
    ):
        self.dag = dag
        self.max_tests = max_tests
        self.alpha = alpha
        self.data_scout = data_scout
        self.issue_ledger = issue_ledger

        # Pre-compute adjacency
        self._adj = self._build_adjacency()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> list[PlaceboResult]:
        """Execute placebo falsification and return results sorted by pvalue."""
        candidates = self._enumerate_null_links()
        prioritised = self._prioritise(candidates)[:self.max_tests]

        results: list[PlaceboResult] = []
        for from_node, to_node in prioritised:
            result = self._test_pair(from_node, to_node)
            results.append(result)

        # Write issues for failures
        if self.issue_ledger is not None:
            for r in results:
                if not r.passed and r.data_status != "missing":
                    self._write_issue(r)

        # Sort by pvalue ascending (most significant first)
        results.sort(key=lambda r: (r.data_status == "missing", r.pvalue))
        return results

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def _enumerate_null_links(self) -> list[tuple[str, str]]:
        """Return all ordered (A, B) pairs where no edge A->B exists."""
        edge_set: set[tuple[str, str]] = set()
        for edge in self.dag.edges:
            edge_set.add((edge.from_node, edge.to_node))

        node_ids = [n.id for n in self.dag.nodes]
        null_links: list[tuple[str, str]] = []
        for a in node_ids:
            for b in node_ids:
                if a != b and (a, b) not in edge_set:
                    null_links.append((a, b))
        return null_links

    def _prioritise(
        self, candidates: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Score and sort candidates by structural interestingness."""
        scored: list[tuple[float, tuple[str, str]]] = []
        for a, b in candidates:
            dist = self._dag_distance(a, b)
            shared = self._shared_neighbors(a, b)
            # Nodes at distance 2 (sharing a neighbor) are most interesting.
            # Disconnected pairs are least interesting.
            dist_score = 10.0 / max(dist, 1) if dist > 0 else 0.5
            shared_score = 2.0 * len(shared)

            # Prefer same frequency (less alignment noise)
            node_a = self.dag.get_node(a)
            node_b = self.dag.get_node(b)
            freq_bonus = 1.0 if (
                node_a and node_b
                and getattr(node_a, "frequency", None)
                and node_a.frequency == node_b.frequency
            ) else 0.0

            score = dist_score + shared_score + freq_bonus
            scored.append((score, (a, b)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [pair for _, pair in scored]

    # ------------------------------------------------------------------
    # Data gate + regression
    # ------------------------------------------------------------------

    def _test_pair(self, from_node: str, to_node: str) -> PlaceboResult:
        """Check data availability and run placebo OLS for one pair."""
        dist = self._dag_distance(from_node, to_node)
        shared = self._shared_neighbors(from_node, to_node)

        from_ok = self._check_data(from_node)
        to_ok = self._check_data(to_node)

        # Attempt DataScout fetch for missing nodes
        data_status = "available"
        if not (from_ok and to_ok) and self.data_scout is not None:
            missing = []
            if not from_ok:
                missing.append(from_node)
            if not to_ok:
                missing.append(to_node)
            try:
                report = self.data_scout.download_missing(self.dag, missing)
                if report.downloaded > 0:
                    data_status = "fetched"
                    # Re-check
                    from_ok = self._check_data(from_node)
                    to_ok = self._check_data(to_node)
            except Exception as e:
                logger.debug(f"DataScout fetch failed for {missing}: {e}")

        if not (from_ok and to_ok):
            return PlaceboResult(
                from_node=from_node,
                to_node=to_node,
                coefficient=float("nan"),
                se=float("nan"),
                pvalue=1.0,
                n_obs=0,
                passed=True,  # can't test → don't flag
                dag_distance=dist,
                shared_neighbors=shared,
                data_status="missing",
                message=f"Data unavailable for {'both' if not from_ok and not to_ok else from_node if not from_ok else to_node}",
            )

        # Run placebo regression
        return self._run_placebo(from_node, to_node, dist, shared, data_status)

    def _run_placebo(
        self,
        from_node: str,
        to_node: str,
        dist: int,
        shared: list[str],
        data_status: str,
    ) -> PlaceboResult:
        """Run OLS placebo regression for a single null-link pair."""
        from shared.engine.data_assembler import load_node_series, _align_frequencies

        try:
            treatment = load_node_series(from_node)
            outcome = load_node_series(to_node)
        except Exception as e:
            return PlaceboResult(
                from_node=from_node,
                to_node=to_node,
                coefficient=float("nan"),
                se=float("nan"),
                pvalue=1.0,
                n_obs=0,
                passed=True,
                dag_distance=dist,
                shared_neighbors=shared,
                data_status="missing",
                message=f"Load failed: {e}",
            )

        # Align frequencies
        treatment, outcome = _align_frequencies(treatment, outcome)

        # Build aligned DataFrame and drop NaN
        combined = pd.DataFrame({
            "treatment": treatment,
            "outcome": outcome,
        }).dropna()

        n_obs = len(combined)
        if n_obs < 12:
            return PlaceboResult(
                from_node=from_node,
                to_node=to_node,
                coefficient=float("nan"),
                se=float("nan"),
                pvalue=1.0,
                n_obs=n_obs,
                passed=True,
                dag_distance=dist,
                shared_neighbors=shared,
                data_status=data_status,
                message=f"Insufficient observations ({n_obs} < 12)",
            )

        # OLS with HAC standard errors
        import statsmodels.api as sm

        X = sm.add_constant(combined["treatment"].values)
        y = combined["outcome"].values

        try:
            model = sm.OLS(y, X).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": min(4, n_obs // 4)},
            )
            coeff = float(model.params[1])
            se = float(model.bse[1])
            pvalue = float(model.pvalues[1])
        except Exception as e:
            return PlaceboResult(
                from_node=from_node,
                to_node=to_node,
                coefficient=float("nan"),
                se=float("nan"),
                pvalue=1.0,
                n_obs=n_obs,
                passed=True,
                dag_distance=dist,
                shared_neighbors=shared,
                data_status=data_status,
                message=f"OLS failed: {e}",
            )

        passed = pvalue >= self.alpha
        if passed:
            msg = f"Null confirmed (p={pvalue:.4f})"
        else:
            msg = (
                f"Null link {from_node}->{to_node} shows significant association "
                f"(beta={coeff:.4f}, p={pvalue:.4f}, N={n_obs}). "
                f"DAG may need this edge."
            )

        return PlaceboResult(
            from_node=from_node,
            to_node=to_node,
            coefficient=coeff,
            se=se,
            pvalue=pvalue,
            n_obs=n_obs,
            passed=passed,
            dag_distance=dist,
            shared_neighbors=shared,
            data_status=data_status,
            message=msg,
        )

    # ------------------------------------------------------------------
    # Issue writing
    # ------------------------------------------------------------------

    def _write_issue(self, r: PlaceboResult) -> None:
        """Write a NULL_LINK_SIGNIFICANT issue to the ledger."""
        severity = "HIGH" if r.pvalue < 0.01 else "MEDIUM"
        self.issue_ledger.add_from_rule(
            rule_id="NULL_LINK_SIGNIFICANT",
            severity=severity,
            scope="dag",
            message=(
                f"Null link {r.from_node}->{r.to_node} shows significant association "
                f"(beta={r.coefficient:.4f}, p={r.pvalue:.4f}, N={r.n_obs}). "
                f"DAG may need this edge."
            ),
            evidence={
                "from_node": r.from_node,
                "to_node": r.to_node,
                "coefficient": r.coefficient,
                "se": r.se,
                "pvalue": r.pvalue,
                "n_obs": r.n_obs,
                "dag_distance": r.dag_distance,
                "shared_neighbors": r.shared_neighbors,
            },
            auto_fixable=False,
            requires_human=True,
        )

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _build_adjacency(self) -> dict[str, set[str]]:
        """Build undirected adjacency dict from DAG edges."""
        adj: dict[str, set[str]] = {}
        for node in self.dag.nodes:
            adj.setdefault(node.id, set())
        for edge in self.dag.edges:
            adj.setdefault(edge.from_node, set()).add(edge.to_node)
            adj.setdefault(edge.to_node, set()).add(edge.from_node)
        return adj

    def _dag_distance(self, a: str, b: str) -> int:
        """BFS shortest path on undirected graph. Returns -1 if disconnected."""
        if a == b:
            return 0
        visited: set[str] = {a}
        queue: deque[tuple[str, int]] = deque([(a, 0)])
        while queue:
            node, depth = queue.popleft()
            for neighbor in self._adj.get(node, set()):
                if neighbor == b:
                    return depth + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return -1

    def _shared_neighbors(self, a: str, b: str) -> list[str]:
        """Return nodes connected to both a and b."""
        neighbors_a = self._adj.get(a, set())
        neighbors_b = self._adj.get(b, set())
        return sorted(neighbors_a & neighbors_b)

    def _check_data(self, node_id: str) -> bool:
        """Check if data is loadable for a node."""
        from shared.engine.data_assembler import NODE_LOADERS
        if node_id not in NODE_LOADERS:
            return False
        try:
            NODE_LOADERS[node_id]()
            return True
        except Exception:
            return False
