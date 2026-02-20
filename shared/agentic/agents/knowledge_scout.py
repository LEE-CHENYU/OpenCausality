"""
KnowledgeScout: Web-backed research agent for DAG formula and fact lookup.

Provides a clean API for other framework components (critic, auto-fixer)
to query external knowledge. Returns structured, YAML-ready responses
that plug directly into DAG identity formulas and edge metadata.

Usage:
    scout = KnowledgeScout(llm_client)

    # Look up an identity formula
    result = scout.lookup_formula(
        node_id="cpi_headline",
        node_description="Kazakhstan headline CPI index",
        depends_on=["cpi_tradable", "cpi_nontradable"],
        context="CPI composition weights for Kazakhstan",
    )
    if result.found:
        print(result.formula)       # "0.43 * cpi_tradable + 0.57 * cpi_nontradable"
        print(result.to_identity())  # dict ready for DAG YAML

    # Look up a specific fact
    result = scout.lookup_fact(
        question="What is the FX pass-through coefficient to tradable CPI in Kazakhstan?",
        expected_type="coefficient",
    )
    if result.found:
        print(result.value)  # 0.113
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses (the clean output API)
# ---------------------------------------------------------------------------


@dataclass
class FormulaResult:
    """Result of a formula lookup. Directly convertible to DAG YAML."""

    found: bool
    formula: str = ""
    depends_on: list[str] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    source: str = ""
    confidence: str = ""  # "high", "medium", "low"
    reasoning: str = ""

    def to_identity(self, name: str = "") -> dict[str, Any]:
        """Convert to a DAG YAML identity block."""
        if not self.found:
            return {}
        return {
            "name": name or "auto_researched",
            "formula": self.formula,
            "depends_on": self.depends_on,
        }


@dataclass
class FactResult:
    """Result of a fact lookup."""

    found: bool
    value: Any = None  # float, str, dict — depends on expected_type
    unit: str = ""
    source: str = ""
    confidence: str = ""
    reasoning: str = ""


# ---------------------------------------------------------------------------
# LLM schemas (for structured output)
# ---------------------------------------------------------------------------

_FORMULA_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {
            "type": "boolean",
            "description": "Whether a reliable formula was found",
        },
        "formula": {
            "type": "string",
            "description": (
                "The formula using variable names from depends_on. "
                "Use Python-compatible syntax: +, -, *, /. "
                "Example: '0.43 * cpi_tradable + 0.57 * cpi_nontradable'"
            ),
        },
        "depends_on": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Variable names used in the formula",
        },
        "coefficients": {
            "type": "object",
            "additionalProperties": {"type": "number"},
            "description": "Mapping of variable name to its coefficient",
        },
        "source": {
            "type": "string",
            "description": "Citation or source URL for the formula",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "Confidence level in the formula",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of how the formula was derived",
        },
    },
    "required": ["found", "formula", "depends_on", "confidence", "reasoning"],
}

_FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {
            "type": "boolean",
            "description": "Whether a reliable answer was found",
        },
        "value": {
            "description": "The answer value (number, string, or object)",
        },
        "unit": {
            "type": "string",
            "description": "Unit of the value if applicable",
        },
        "source": {
            "type": "string",
            "description": "Citation or source URL",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation",
        },
    },
    "required": ["found", "confidence", "reasoning"],
}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_FORMULA_SYSTEM = """\
You are an econometric research assistant. Your task is to find or derive
identity formulas for economic variables.

RULES:
- Only return formulas you are confident about from established economic theory.
- Use EXACT variable names provided in the depends_on list.
- Formulas must be Python-evaluable: use +, -, *, / operators.
- Coefficients should be approximate weights (e.g., CPI basket shares).
- If the formula is well-known (e.g., CPI composition, Fisher identity),
  confidence should be "high".
- If you cannot find a reliable formula, set found=false.
- Do NOT guess or hallucinate formulas.

CONTEXT: This is for a causal DAG in the OpenCausality framework.
The formula will be used to compute identity coefficients via
numerical perturbation of loaded time-series data."""

_FACT_SYSTEM = """\
You are an econometric research assistant. Your task is to find specific
economic facts, coefficients, or parameters from reliable sources.

RULES:
- Only return values you are confident about from published research,
  central bank reports, or standard economic references.
- Include the source/citation.
- If the value is a coefficient, specify the unit (pct, pp, elasticity, etc.).
- If you cannot find a reliable answer, set found=false.
- Do NOT guess or hallucinate values."""


# ---------------------------------------------------------------------------
# KnowledgeScout
# ---------------------------------------------------------------------------


class KnowledgeScout:
    """Web-backed research agent for formula and fact lookup.

    Uses LLM structured output to return framework-compatible responses.
    Optionally enriches prompts with web search results.
    """

    def __init__(self, llm_client: Any, web_search_fn: Any | None = None):
        """
        Args:
            llm_client: An LLMClient instance (AnthropicClient or LiteLLMClient).
            web_search_fn: Optional callable(query: str) -> str that returns
                web search results as text. If None, relies on LLM knowledge only.
        """
        self.llm = llm_client
        self._web_search = web_search_fn

    def _enrich_with_search(self, query: str) -> str:
        """Run web search and return results as context string."""
        if self._web_search is None:
            return ""
        try:
            results = self._web_search(query)
            return f"\n\nWEB SEARCH RESULTS:\n{results}\n"
        except Exception as e:
            logger.warning(f"KnowledgeScout web search failed: {e}")
            return ""

    def lookup_formula(
        self,
        node_id: str,
        node_description: str,
        depends_on: list[str],
        context: str = "",
    ) -> FormulaResult:
        """Look up an identity formula for a DAG node.

        Args:
            node_id: The target node ID (e.g., "cpi_headline").
            node_description: Human-readable description of the node.
            depends_on: List of dependency variable names to use in formula.
            context: Additional context (country, time period, etc.).

        Returns:
            FormulaResult with found=True/False and the formula if found.
        """
        search_query = (
            f"{node_description} composition formula weights "
            f"{context}"
        ).strip()
        web_context = self._enrich_with_search(search_query)

        user_prompt = (
            f"Find the identity formula for '{node_id}'.\n\n"
            f"Description: {node_description}\n"
            f"Variables available: {depends_on}\n"
            f"Context: {context}\n"
            f"{web_context}\n"
            f"Return the formula using ONLY these variable names: {depends_on}"
        )

        try:
            raw = self.llm.complete_structured(
                _FORMULA_SYSTEM, user_prompt, _FORMULA_SCHEMA,
            )
            return FormulaResult(
                found=raw.get("found", False),
                formula=raw.get("formula", ""),
                depends_on=raw.get("depends_on", []),
                coefficients=raw.get("coefficients", {}),
                source=raw.get("source", ""),
                confidence=raw.get("confidence", "low"),
                reasoning=raw.get("reasoning", ""),
            )
        except Exception as e:
            logger.error(f"KnowledgeScout formula lookup failed: {e}")
            return FormulaResult(
                found=False,
                reasoning=f"LLM call failed: {e}",
            )

    def lookup_fact(
        self,
        question: str,
        expected_type: str = "coefficient",
        context: str = "",
    ) -> FactResult:
        """Look up a specific economic fact or coefficient.

        Args:
            question: The specific question to answer.
            expected_type: What kind of answer ("coefficient", "weight",
                "formula", "boolean", "description").
            context: Additional context.

        Returns:
            FactResult with found=True/False and the value if found.
        """
        web_context = self._enrich_with_search(question)

        user_prompt = (
            f"Question: {question}\n"
            f"Expected answer type: {expected_type}\n"
            f"Context: {context}\n"
            f"{web_context}"
        )

        try:
            raw = self.llm.complete_structured(
                _FACT_SYSTEM, user_prompt, _FACT_SCHEMA,
            )
            return FactResult(
                found=raw.get("found", False),
                value=raw.get("value"),
                unit=raw.get("unit", ""),
                source=raw.get("source", ""),
                confidence=raw.get("confidence", "low"),
                reasoning=raw.get("reasoning", ""),
            )
        except Exception as e:
            logger.error(f"KnowledgeScout fact lookup failed: {e}")
            return FactResult(
                found=False,
                reasoning=f"LLM call failed: {e}",
            )


# ---------------------------------------------------------------------------
# Convenience function: batch formula lookup for DAG nodes
# ---------------------------------------------------------------------------


def research_missing_formulas(
    llm_client: Any,
    dag: dict,
    web_search_fn: Any | None = None,
) -> dict[str, FormulaResult]:
    """Find formulas for all identity edges whose target nodes lack formulas.

    Scans the DAG for identity/mechanical edges where the target node has
    no identity block, then uses KnowledgeScout to look up each formula.

    Args:
        llm_client: LLMClient instance.
        dag: The full DAG dict (from YAML).
        web_search_fn: Optional web search callable.

    Returns:
        Dict mapping node_id -> FormulaResult for each node researched.
    """
    scout = KnowledgeScout(llm_client, web_search_fn)
    nodes = {n["id"]: n for n in dag.get("nodes", [])}
    edges = dag.get("edges", [])

    # Find identity/immutable edges where target has no formula
    targets_needing_formula: dict[str, list[str]] = {}
    for edge in edges:
        if edge.get("edge_type") not in ("identity", "immutable"):
            continue
        to_node = edge.get("to", "")
        to_def = nodes.get(to_node, {})
        if to_def.get("identity", {}).get("formula"):
            continue  # Already has formula
        from_node = edge.get("from", "")
        targets_needing_formula.setdefault(to_node, []).append(from_node)

    results: dict[str, FormulaResult] = {}
    for node_id, deps in targets_needing_formula.items():
        node_def = nodes.get(node_id, {})
        result = scout.lookup_formula(
            node_id=node_id,
            node_description=node_def.get("description", node_def.get("name", node_id)),
            depends_on=deps,
            context=dag.get("metadata", {}).get("description", ""),
        )
        results[node_id] = result
        status = "FOUND" if result.found else "NOT FOUND"
        logger.info(
            f"KnowledgeScout: {node_id} formula {status} "
            f"(confidence={result.confidence})"
        )

    return results
