"""
DataResolverAgent: LLM-powered adaptive data discovery.

Uses multi-turn tool-use to dynamically discover, fetch, and register
data from external APIs. The LLM reasons about which provider to try,
constructs API queries, parses responses, and saves results.

This is the adaptive component — it can discover new data sources at
runtime without hardcoded provider mappings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ResolveResult:
    """Outcome of a data resolution attempt."""

    node_id: str
    success: bool
    file_path: str = ""
    rows: int = 0
    provider: str = ""
    indicator: str = ""
    error: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    llm_reasoning: str = ""


class DataResolverAgent:
    """LLM-powered agent that discovers and downloads data for DAG nodes.

    Given a node that failed to load from static sources, the agent uses
    tool-use to search data catalogs, fetch APIs, parse responses, and
    register loaders — all guided by LLM reasoning.

    Args:
        llm_client: An LLMClient that supports run_agent_loop().
        dag: The DAGSpec (for node metadata).
        dynamic_loader: DynamicLoaderFactory (for post-download registration).
    """

    def __init__(
        self,
        llm_client: Any,
        dag: Any,
        dynamic_loader: Any | None = None,
    ):
        self.llm_client = llm_client
        self.dag = dag
        self.dynamic_loader = dynamic_loader

    def resolve_node(
        self,
        node_id: str,
        failed_sources: list[str] | None = None,
        max_turns: int = 10,
    ) -> ResolveResult:
        """Use LLM + tools to find and download data for a failed node.

        Args:
            node_id: The DAG node that needs data.
            failed_sources: List of sources already tried and failed.
            max_turns: Maximum tool-use turns for the agent loop.

        Returns:
            ResolveResult with success/failure and provenance.
        """
        from shared.llm.prompts import DATA_RESOLVER_SYSTEM
        from shared.data.api_tools import TOOL_DEFINITIONS, execute_tool

        node = self.dag.get_node(node_id)
        if node is None:
            return ResolveResult(
                node_id=node_id,
                success=False,
                error=f"Node '{node_id}' not found in DAG",
            )

        # Build context for the LLM
        node_name = getattr(node, "name", node_id)
        node_desc = getattr(node, "description", "")
        node_unit = getattr(node, "unit", "")
        node_freq = ""
        if node.source and node.source.preferred:
            node_freq = getattr(node.source.preferred[0], "frequency", "")

        country = self._infer_country()
        failed_str = ", ".join(failed_sources) if failed_sources else "none"

        user_prompt = (
            f"Find and download time series data for this DAG node:\n\n"
            f"Node ID: {node_id}\n"
            f"Name: {node_name}\n"
            f"Description: {node_desc}\n"
            f"Unit: {node_unit}\n"
            f"Expected frequency: {node_freq or 'unknown'}\n"
            f"Country context: {country}\n"
            f"Previously failed sources: {failed_str}\n\n"
            f"Use the tools to search data catalogs, fetch the data, "
            f"parse it, and save it. Start by listing available providers."
        )

        # Run the agent loop
        try:
            result = self.llm_client.run_agent_loop(
                system=DATA_RESOLVER_SYSTEM,
                user=user_prompt,
                tools=TOOL_DEFINITIONS,
                tool_executor=execute_tool,
                max_turns=max_turns,
            )
        except NotImplementedError:
            return ResolveResult(
                node_id=node_id,
                success=False,
                error="LLM client does not support run_agent_loop (tool-use required)",
            )
        except Exception as e:
            logger.warning(f"DataResolverAgent: agent loop error for {node_id}: {e}")
            return ResolveResult(
                node_id=node_id,
                success=False,
                error=str(e),
            )

        # Inspect tool call history for save_and_register success
        file_path = ""
        rows = 0
        provider = ""
        indicator = ""
        for call in result.tool_calls:
            if call.get("tool") == "save_and_register" and call.get("success"):
                try:
                    output = json.loads(call.get("output", "{}"))
                    if output.get("success"):
                        file_path = output.get("file_path", "")
                        rows = output.get("rows", 0)
                        prov = output.get("provenance", {})
                        provider = prov.get("provider", "")
                        indicator = prov.get("indicator", "")
                except (json.JSONDecodeError, TypeError):
                    pass

        success = bool(file_path and rows > 0 and result.success)

        if success:
            logger.info(
                f"DataResolverAgent: resolved '{node_id}' via {provider}/{indicator} "
                f"({rows} rows -> {file_path})"
            )
        else:
            logger.warning(
                f"DataResolverAgent: failed to resolve '{node_id}': "
                f"{result.error or 'no data saved'}"
            )

        return ResolveResult(
            node_id=node_id,
            success=success,
            file_path=file_path,
            rows=rows,
            provider=provider,
            indicator=indicator,
            error=result.error if not success else "",
            tool_calls=result.tool_calls,
            llm_reasoning=result.text,
        )

    def _infer_country(self) -> str:
        """Infer country context from DAG metadata."""
        meta = getattr(self.dag, "metadata", None)
        if meta:
            name = getattr(meta, "name", "").lower()
            desc = getattr(meta, "description", "").lower()
            for text in (name, desc):
                if "kazakh" in text:
                    return "Kazakhstan (KZ)"
                if "nigeria" in text:
                    return "Nigeria (NG)"
                if "turkey" in text or "türkiye" in text:
                    return "Turkey (TR)"
        return "Kazakhstan (KZ)"  # default for this project
