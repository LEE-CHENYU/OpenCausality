"""
OpenCausality Query REPL — Interactive causal query interface.

Supports:
- NL queries classified by LLM (with regex fallback)
- Slash commands for direct dispatch
- Rich output with tables, panels, trees
- Propagation with full guardrail integration
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

logger = logging.getLogger(__name__)
console = Console()

# ──────────────────────────────────────────────────────────────────────
# Intent Classification
# ──────────────────────────────────────────────────────────────────────

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "shock_scenario", "policy_scenario", "path_query",
                "edge_inspect", "identification_query", "mode_compare",
                "mode_switch", "node_list", "edge_list",
                "paper_search", "propose_edges",
                "help", "unknown",
            ],
        },
        "source_node": {"type": "string"},
        "target_node": {"type": "string"},
        "magnitude": {"type": "number"},
        "unit": {"type": "string"},
        "edge_id": {"type": "string"},
        "mode": {"type": "string"},
        "filter": {"type": "string"},
    },
    "required": ["intent"],
}


def _classify_intent_regex(text: str, node_ids: list[str]) -> dict:
    """Fallback intent classification using regex patterns."""
    text_lower = text.lower().strip()

    # Shock scenario patterns
    shock_match = re.search(
        r"what\s+(?:if|happens|would)\s+.*?(drop|fall|increase|rise|shock|decline)",
        text_lower,
    )
    if shock_match or "impact" in text_lower or "shock" in text_lower:
        # Try to extract magnitude
        mag_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", text_lower)
        magnitude = float(mag_match.group(1)) / 100 if mag_match else None

        # Try to find nodes
        source = _fuzzy_match_node(text_lower, node_ids)
        return {
            "intent": "shock_scenario",
            "source_node": source or "",
            "magnitude": magnitude,
            "unit": "pct" if magnitude else "",
        }

    # Mode switch
    mode_match = re.search(r"(?:switch|set|change)\s+.*?mode\s+(?:to\s+)?(\w+)", text_lower)
    if mode_match:
        return {"intent": "mode_switch", "mode": mode_match.group(1).upper()}

    # Identification query
    if "identif" in text_lower or "weak" in text_lower or "claim" in text_lower:
        return {"intent": "identification_query"}

    # Mode compare
    if "compare" in text_lower and "mode" in text_lower:
        return {"intent": "mode_compare"}

    # Path query
    if "path" in text_lower or "route" in text_lower:
        return {"intent": "path_query"}

    # Edge inspect
    if "card" in text_lower or "inspect" in text_lower or "detail" in text_lower:
        edge = _fuzzy_match_edge(text_lower, node_ids)
        return {"intent": "edge_inspect", "edge_id": edge or ""}

    # Paper search
    if "search" in text_lower and "paper" in text_lower:
        return {"intent": "paper_search"}
    if "literature" in text_lower or "citation" in text_lower:
        return {"intent": "paper_search"}

    # Propose edges from literature
    if "propose" in text_lower and ("edge" in text_lower or "literature" in text_lower):
        return {"intent": "propose_edges"}

    return {"intent": "unknown"}


def _fuzzy_match_node(text: str, node_ids: list[str]) -> str | None:
    """Find the best matching node ID in the text."""
    text_lower = text.lower()
    # Direct match
    for nid in node_ids:
        if nid.lower() in text_lower:
            return nid
    # Partial match (underscore-separated parts)
    for nid in node_ids:
        parts = nid.lower().replace("_", " ").split()
        if any(p in text_lower for p in parts if len(p) > 3):
            return nid
    return None


def _fuzzy_match_edge(text: str, node_ids: list[str]) -> str | None:
    """Try to extract an edge_id from text."""
    # Look for edge_id patterns (word_to_word)
    match = re.search(r"\b(\w+_to_\w+)\b", text.lower())
    if match:
        return match.group(1)
    return None


# ──────────────────────────────────────────────────────────────────────
# LLM Narration (constrained)
# ──────────────────────────────────────────────────────────────────────

NARRATION_SYSTEM = """You are a causal inference assistant narrating propagation results.

HARD RULES:
1. NEVER say "causes" or "causal effect" unless the weakest edge in the path has claim_level == "IDENTIFIED_CAUSAL".
2. ALWAYS state the query mode and the weakest claim_level in the path.
3. ALWAYS include "SE assumes independence" disclaimer when path has >1 estimated edge.
4. Use hedged language for REDUCED_FORM ("is associated with", "predicts") and DESCRIPTIVE ("co-moves with", "accounts for").
5. If any edge was blocked, narrate WHY it was blocked (don't silently omit).
6. Be concise (2-4 sentences).
"""


def _narrate_result(
    llm: Any,
    result: Any,
    mode: str,
    query: str,
) -> str | None:
    """Ask LLM to narrate a PropagationResult (constrained)."""
    if llm is None:
        return None
    try:
        import json
        # Serialize result minimally
        data = {
            "mode": mode,
            "query": query,
            "paths": [],
            "blocked_count": len(result.blocked_edges),
            "warnings": result.warnings,
        }
        for p in result.paths:
            pdata = {
                "total_effect": p.total_effect,
                "total_se": p.total_se,
                "ci_95": list(p.ci_95),
                "se_method": p.se_method,
                "blocked": p.is_blocked,
                "blocked_reasons": p.blocked_reasons,
                "warnings": p.warnings,
                "edges": [
                    {
                        "edge_id": e.edge_id,
                        "role": e.role,
                        "claim_level": e.claim_level,
                        "coefficient": e.coefficient,
                    }
                    for e in p.edges
                ],
            }
            data["paths"].append(pdata)
        return llm.complete(NARRATION_SYSTEM, json.dumps(data), max_tokens=300)
    except Exception as e:
        logger.warning(f"LLM narration failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────
# Query REPL Class
# ──────────────────────────────────────────────────────────────────────


class QueryREPL:
    """Interactive query REPL for causal DAG exploration."""

    def __init__(
        self,
        dag_path: Path | None = None,
        cards_dir: Path | None = None,
        mode: str = "REDUCED_FORM",
    ):
        from config.settings import get_settings
        settings = get_settings()

        # Resolve DAG path
        if dag_path is None:
            dag_path = Path(settings.default_dag_path)
        if not dag_path.exists():
            console.print(f"[red]DAG not found: {dag_path}[/red]")
            raise SystemExit(1)

        # Parse DAG
        from shared.agentic.dag.parser import parse_dag
        self.dag = parse_dag(dag_path)
        self.dag_path = dag_path

        # Load edge cards
        if cards_dir is None:
            cards_dir = Path(settings.output_dir) / "agentic" / "edge_cards"

        self.edge_cards: dict[str, Any] = {}
        if cards_dir.exists():
            from shared.agentic.artifact_store import ArtifactStore
            store = ArtifactStore(cards_dir.parent)
            for card in store.get_all_edge_cards():
                self.edge_cards[card.edge_id] = card
        self.cards_dir = cards_dir

        # Load TSGuard results
        tsguard_dir = Path(settings.output_dir) / "agentic" / "tsguard"
        self.tsguard_results: dict[str, Any] = {}
        if tsguard_dir.exists():
            self._load_tsguard_results(tsguard_dir)

        # Load issue ledger
        self.issue_ledger = None
        issues_dir = Path(settings.output_dir) / "agentic" / "issues"
        if issues_dir.exists():
            self._load_latest_issues(issues_dir)

        # Build propagation engine
        from shared.agentic.propagation import PropagationEngine
        self.engine = PropagationEngine(
            dag=self.dag,
            edge_cards=self.edge_cards,
            tsguard_results=self.tsguard_results,
            issue_ledger=self.issue_ledger,
        )

        self.mode = mode

        # Try to get LLM client
        self.llm = None
        try:
            from shared.llm.client import get_llm_client
            self.llm = get_llm_client(settings)
        except Exception:
            pass  # Regex fallback

    def _load_tsguard_results(self, tsguard_dir: Path) -> None:
        """Load stored TSGuard results from YAML/JSON files."""
        import yaml
        import json as json_mod
        for f in tsguard_dir.glob("*.yaml"):
            try:
                with open(f) as fh:
                    data = yaml.safe_load(fh)
                if data and "edge_id" in data:
                    from shared.agentic.ts_guard import TSGuardResult
                    result = TSGuardResult(
                        edge_id=data["edge_id"],
                        dynamics_risk=data.get("dynamics_risk", {}),
                        diagnostics_results=data.get("diagnostics_results", {}),
                        claim_level_cap=data.get("claim_level_cap"),
                        counterfactual_blocked=data.get("counterfactual_blocked", False),
                    )
                    self.tsguard_results[data["edge_id"]] = result
            except Exception:
                pass

    def _load_latest_issues(self, issues_dir: Path) -> None:
        """Load the latest issue ledger."""
        from shared.agentic.issues.issue_ledger import IssueLedger
        jsonl_files = sorted(issues_dir.glob("*.jsonl"))
        if jsonl_files:
            ledger = IssueLedger()
            issues = ledger.load_from_file(jsonl_files[-1])
            ledger.issues = issues
            self.issue_ledger = ledger

    # ──────────────────────────────────────────────────────────────────
    # REPL loop
    # ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the interactive REPL."""
        self._print_welcome()

        while True:
            try:
                raw = console.input("\n[bold green]query>[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if not raw:
                continue

            # Slash commands
            if raw.startswith("/"):
                self._handle_slash(raw)
                continue

            # NL classification
            intent = self._classify(raw)
            self._dispatch(intent, raw)

    def _print_welcome(self) -> None:
        n_nodes = len(self.dag.nodes)
        n_edges = len(self.dag.edges)
        n_cards = len(self.edge_cards)
        llm_status = "LLM" if self.llm else "regex-only"

        console.print(Panel(
            f"[bold cyan]OpenCausality Query REPL[/bold cyan]\n\n"
            f"DAG: [cyan]{self.dag.metadata.name}[/cyan] "
            f"({n_nodes} nodes, {n_edges} edges)\n"
            f"Cards: {n_cards} | Mode: [yellow]{self.mode}[/yellow] | "
            f"Backend: {llm_status}\n\n"
            f"Type a question or /help for commands.",
            title="Welcome",
        ))

    # ──────────────────────────────────────────────────────────────────
    # Classification
    # ──────────────────────────────────────────────────────────────────

    def _classify(self, text: str) -> dict:
        """Classify user intent using LLM or regex fallback."""
        node_ids = self.engine.get_all_node_ids()

        if self.llm is not None:
            try:
                node_list = ", ".join(node_ids)
                system_prompt = (
                    "You classify causal inference queries. "
                    f"Available nodes: {node_list}. "
                    "Return structured intent matching the schema."
                )
                result = self.llm.complete_structured(
                    system_prompt, text, INTENT_SCHEMA,
                )
                if result and result.get("intent"):
                    return result
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")

        return _classify_intent_regex(text, node_ids)

    # ──────────────────────────────────────────────────────────────────
    # Dispatch
    # ──────────────────────────────────────────────────────────────────

    def _dispatch(self, intent: dict, raw_text: str) -> None:
        intent_type = intent.get("intent", "unknown")
        handler = {
            "shock_scenario": self._handle_shock,
            "policy_scenario": self._handle_policy,
            "path_query": self._handle_path,
            "edge_inspect": self._handle_edge_inspect,
            "identification_query": self._handle_identification,
            "mode_compare": self._handle_mode_compare,
            "mode_switch": self._handle_mode_switch,
            "node_list": self._handle_nodes,
            "edge_list": self._handle_edges,
            "paper_search": self._handle_paper_search,
            "propose_edges": self._handle_propose_edges,
            "help": self._handle_help,
        }.get(intent_type)

        if handler:
            handler(intent, raw_text)
        else:
            console.print("[dim]I didn't understand that. Try /help for commands.[/dim]")

    # ──────────────────────────────────────────────────────────────────
    # Slash command dispatcher
    # ──────────────────────────────────────────────────────────────────

    def _handle_slash(self, raw: str) -> None:
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == "/help":
            self._handle_help({}, raw)
        elif cmd == "/nodes":
            self._handle_nodes({}, raw)
        elif cmd == "/edges":
            mode = parts[1].upper() if len(parts) > 1 else self.mode
            self._handle_edges({"mode": mode}, raw)
        elif cmd == "/mode" and len(parts) > 1:
            self._handle_mode_switch({"mode": parts[1].upper()}, raw)
        elif cmd == "/card" and len(parts) > 1:
            self._handle_edge_inspect({"edge_id": parts[1]}, raw)
        elif cmd == "/impact" and len(parts) >= 4:
            # /impact source target magnitude [unit]
            intent = {
                "intent": "shock_scenario",
                "source_node": parts[1],
                "target_node": parts[2],
                "magnitude": float(parts[3]),
                "unit": parts[4] if len(parts) > 4 else "sd",
            }
            self._handle_shock(intent, raw)
        elif cmd == "/path" and len(parts) >= 3:
            self._handle_path({
                "source_node": parts[1],
                "target_node": parts[2],
            }, raw)
        elif cmd == "/papers":
            self._handle_paper_search({}, raw)
        elif cmd == "/propose":
            self._handle_propose_edges({}, raw)
        elif cmd == "/doctor":
            self._handle_doctor()
        elif cmd in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye.[/dim]")
            raise SystemExit(0)
        else:
            console.print(f"[dim]Unknown command: {cmd}. Try /help.[/dim]")

    # ──────────────────────────────────────────────────────────────────
    # Handlers
    # ──────────────────────────────────────────────────────────────────

    def _handle_help(self, intent: dict, raw: str) -> None:
        console.print(Panel(
            "[bold]Slash Commands[/bold]\n"
            "  /help                            Show this help\n"
            "  /nodes                           List all nodes\n"
            "  /edges [MODE]                    List edges with roles\n"
            "  /mode <MODE>                     Switch query mode\n"
            "  /card <edge_id>                  Inspect edge card\n"
            "  /impact <src> <tgt> <mag> [unit] Explicit shock propagation\n"
            "  /path <source> <target>          Show all paths\n"
            "  /papers                          Search literature\n"
            "  /propose                         Propose edges from papers\n"
            "  /doctor                          DAG health check\n"
            "  /quit                            Exit REPL\n\n"
            "[bold]Natural Language Examples[/bold]\n"
            '  "What if oil drops 30%?"\n'
            '  "Which edges are weakly identified?"\n'
            '  "Compare modes for this DAG"\n'
            '  "Search papers for oil price to inflation"\n'
            '  "Propose new edges from literature"',
            title="Help",
        ))

    def _handle_nodes(self, intent: dict, raw: str) -> None:
        table = Table(title="DAG Nodes")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Frequency", style="yellow")
        table.add_column("Type", style="dim")
        for node in self.dag.nodes:
            table.add_row(node.id, node.name, node.frequency, node.type)
        console.print(table)

    def _handle_edges(self, intent: dict, raw: str) -> None:
        mode = intent.get("mode", self.mode)
        summaries = self.engine.get_edges_summary(mode)
        table = Table(title=f"Edges (mode={mode})")
        table.add_column("Edge ID", style="cyan")
        table.add_column("From -> To", style="white")
        table.add_column("Type", style="dim")
        table.add_column("Role", style="yellow")
        table.add_column("Claim", style="white")
        table.add_column("Allowed", style="green")
        table.add_column("Coeff", style="white")
        for s in summaries:
            allowed_str = "[green]Y[/green]" if s["allowed"] else "[red]N[/red]"
            coeff_str = f"{s['coefficient']:.4f}" if s["coefficient"] is not None else "-"
            table.add_row(
                s["edge_id"],
                f"{s['from']} -> {s['to']}",
                s["edge_type"],
                s["role"],
                s["claim_level"] or "-",
                allowed_str,
                coeff_str,
            )
        console.print(table)

    def _handle_mode_switch(self, intent: dict, raw: str) -> None:
        new_mode = intent.get("mode", "").upper()
        if new_mode in ("STRUCTURAL", "REDUCED_FORM", "DESCRIPTIVE"):
            self.mode = new_mode
            console.print(f"[green]Mode switched to {new_mode}[/green]")
            self._handle_edges({"mode": new_mode}, raw)
        else:
            console.print(
                f"[red]Invalid mode: {new_mode}. "
                "Use STRUCTURAL, REDUCED_FORM, or DESCRIPTIVE.[/red]"
            )

    def _handle_shock(self, intent: dict, raw: str) -> None:
        source = intent.get("source_node", "")
        target_node = intent.get("target_node", "")
        magnitude = intent.get("magnitude")

        # Resolve target from DAG if not specified
        if not target_node:
            target_node = self.dag.metadata.target_node or ""

        if not source:
            console.print("[yellow]Could not determine source node. Try /impact <src> <tgt> <mag>[/yellow]")
            return

        if not target_node:
            console.print("[yellow]Could not determine target node.[/yellow]")
            return

        result = self.engine.find_all_paths(
            source=source,
            target=target_node,
            mode=self.mode,
            scenario_type="shock",
        )
        self._display_propagation_result(result, magnitude, raw)

    def _handle_policy(self, intent: dict, raw: str) -> None:
        source = intent.get("source_node", "")
        target_node = intent.get("target_node", self.dag.metadata.target_node or "")

        if not source or not target_node:
            console.print("[yellow]Specify source and target nodes.[/yellow]")
            return

        result = self.engine.find_all_paths(
            source=source,
            target=target_node,
            mode=self.mode,
            scenario_type="policy",
        )
        self._display_propagation_result(result, intent.get("magnitude"), raw)

    def _handle_path(self, intent: dict, raw: str) -> None:
        source = intent.get("source_node", "")
        target = intent.get("target_node", "")

        if not source or not target:
            # Try to extract from raw text
            node_ids = self.engine.get_all_node_ids()
            found = []
            for nid in node_ids:
                if nid.lower() in raw.lower():
                    found.append(nid)
            if len(found) >= 2:
                source, target = found[0], found[1]
            elif not source or not target:
                console.print("[yellow]Specify /path <source> <target>[/yellow]")
                return

        result = self.engine.find_all_paths(
            source=source, target=target, mode=self.mode,
        )

        if not result.paths:
            console.print(f"[dim]No paths from {source} to {target}[/dim]")
            for w in result.warnings:
                console.print(f"  [yellow]{w}[/yellow]")
            return

        # Display as tree
        for i, path in enumerate(result.paths):
            tree = Tree(f"[bold]Path {i + 1}[/bold]")
            for e in path.edges:
                style = "red" if path.is_blocked else "white"
                tree.add(
                    f"[{style}]{e.from_node} --({e.edge_id})--> {e.to_node}  "
                    f"[dim]coeff={e.coefficient:.4f} SE={e.se:.4f} "
                    f"role={e.role} claim={e.claim_level}[/dim]"
                )
            console.print(tree)

            if path.is_blocked:
                for reason in path.blocked_reasons:
                    console.print(f"  [red]BLOCKED: {reason}[/red]")
            else:
                console.print(
                    f"  Effect={path.total_effect:.4f}  "
                    f"SE={path.total_se:.4f}  "
                    f"95%CI=[{path.ci_95[0]:.4f}, {path.ci_95[1]:.4f}]  "
                    f"[dim]SE: {path.se_method}[/dim]"
                )
            for w in path.warnings:
                console.print(f"  [yellow]WARNING: {w}[/yellow]")

    def _handle_edge_inspect(self, intent: dict, raw: str) -> None:
        edge_id = intent.get("edge_id", "")
        if not edge_id:
            console.print("[yellow]Specify edge ID: /card <edge_id>[/yellow]")
            return

        card = self.edge_cards.get(edge_id)
        if card is None:
            console.print(f"[yellow]No edge card found for '{edge_id}'[/yellow]")
            # List available edge IDs
            available = list(self.edge_cards.keys())[:10]
            if available:
                console.print(f"[dim]Available: {', '.join(available)}[/dim]")
            return

        # Display card details
        console.print(Panel(
            f"[bold]Edge: {edge_id}[/bold]\n"
            f"Rating: {card.credibility_rating} ({card.credibility_score:.2f})",
            title="Edge Card",
        ))

        # Estimates
        if card.estimates:
            est = card.estimates
            console.print(
                f"\n[bold]Estimates:[/bold]  "
                f"point={est.point:.4f}  SE={est.se:.4f}  "
                f"95%CI=[{est.ci_95[0]:.4f}, {est.ci_95[1]:.4f}]"
            )
            if est.treatment_unit:
                console.print(f"  Treatment unit: {est.treatment_unit}")
            if est.outcome_unit:
                console.print(f"  Outcome unit: {est.outcome_unit}")

        # Identification
        if card.identification.claim_level:
            console.print(f"\n[bold]Identification:[/bold]  {card.identification.claim_level}")
            if card.identification.risks:
                for risk, level in card.identification.risks.items():
                    console.print(f"  {risk}: {level}")

        # Counterfactual
        cf = card.counterfactual_block
        shock_ok = cf.shock_scenario_allowed
        policy_ok = cf.policy_intervention_allowed
        console.print(
            f"\n[bold]Counterfactual:[/bold]  "
            f"Shock={'[green]Y[/green]' if shock_ok else '[red]N[/red]'}  "
            f"Policy={'[green]Y[/green]' if policy_ok else '[red]N[/red]'}"
        )

        # TSGuard
        ts = self.tsguard_results.get(edge_id)
        if ts:
            console.print(f"\n[bold]TSGuard:[/bold]  high_risk={ts.has_any_high_risk}")
            if ts.dynamics_risk:
                for k, v in ts.dynamics_risk.items():
                    style = "red" if v == "high" else "yellow" if v == "medium" else "dim"
                    console.print(f"  [{style}]{k}: {v}[/{style}]")

        # Open issues
        if self.issue_ledger:
            edge_issues = [
                i for i in self.issue_ledger.issues
                if i.edge_id == edge_id and i.is_open
            ]
            if edge_issues:
                console.print(f"\n[bold]Open Issues ({len(edge_issues)}):[/bold]")
                for issue in edge_issues:
                    style = "red" if issue.is_critical else "yellow"
                    console.print(f"  [{style}][{issue.severity}] {issue.rule_id}: {issue.message}[/{style}]")

        # Literature
        lit = card.literature
        if lit.total_results > 0:
            console.print(f"\n[bold]Literature:[/bold]  {lit.total_results} papers ({lit.search_status})")
            for c in lit.supporting[:3]:
                console.print(f"  [green]+[/green] {c.get('title', '?')} ({c.get('year', '?')})")
            for c in lit.challenging[:3]:
                console.print(f"  [red]-[/red] {c.get('title', '?')} ({c.get('year', '?')})")

    def _handle_identification(self, intent: dict, raw: str) -> None:
        table = Table(title="Identification Summary")
        table.add_column("Edge", style="cyan")
        table.add_column("Claim Level", style="white")
        table.add_column("Shock CF", style="white")
        table.add_column("Policy CF", style="white")
        table.add_column("Risks", style="yellow")

        for edge in self.dag.edges:
            card = self.edge_cards.get(edge.id)
            if card is None:
                table.add_row(edge.id, "[dim]no card[/dim]", "-", "-", "-")
                continue
            claim = card.identification.claim_level or "-"
            shock = "[green]Y[/green]" if card.counterfactual_block.shock_scenario_allowed else "[red]N[/red]"
            policy = "[green]Y[/green]" if card.counterfactual_block.policy_intervention_allowed else "[red]N[/red]"
            risks = ", ".join(
                f"{k}={v}" for k, v in card.identification.risks.items()
                if v in ("medium", "high")
            ) or "-"
            table.add_row(edge.id, claim, shock, policy, risks)

        console.print(table)

    def _handle_mode_compare(self, intent: dict, raw: str) -> None:
        modes = ["STRUCTURAL", "REDUCED_FORM", "DESCRIPTIVE"]
        table = Table(title="Edge Permissions by Mode")
        table.add_column("Edge", style="cyan")
        for m in modes:
            table.add_column(m, style="white")

        for edge in self.dag.edges:
            row = [edge.id]
            for m in modes:
                summaries = self.engine.get_edges_summary(m)
                s = next((x for x in summaries if x["edge_id"] == edge.id), None)
                if s:
                    val = f"[green]Y[/green] ({s['role']})" if s["allowed"] else f"[red]N[/red]"
                else:
                    val = "-"
                row.append(val)
            table.add_row(*row)

        console.print(table)

    def _handle_paper_search(self, intent: dict, raw: str) -> None:
        """Search papers for causal claims related to the DAG."""
        if self.llm is None:
            console.print("[yellow]Paper search requires LLM. Set ANTHROPIC_API_KEY.[/yellow]")
            return

        console.print("[dim]Searching academic literature...[/dim]")
        try:
            from config.settings import get_settings
            from shared.agentic.agents.paper_scout import PaperScout

            settings = get_settings()
            scout = PaperScout(
                s2_api_key=settings.semantic_scholar_api_key or None,
                openalex_mailto=settings.openalex_mailto or None,
                unpaywall_email=settings.unpaywall_email or None,
                core_api_key=settings.core_api_key or None,
            )

            proposed = scout.search_and_extract(
                dag=self.dag, llm=self.llm,
            )
            self._display_proposed_edges(proposed)

        except Exception as e:
            console.print(f"[red]Paper search failed: {e}[/red]")

    def _handle_propose_edges(self, intent: dict, raw: str) -> None:
        """Full pipeline: search papers, extract claims, propose edges."""
        if self.llm is None:
            console.print("[yellow]Edge proposal requires LLM. Set ANTHROPIC_API_KEY.[/yellow]")
            return

        console.print("[dim]Searching papers and extracting causal claims...[/dim]")
        try:
            from config.settings import get_settings
            from shared.agentic.agents.paper_scout import PaperScout

            settings = get_settings()
            scout = PaperScout(
                s2_api_key=settings.semantic_scholar_api_key or None,
                openalex_mailto=settings.openalex_mailto or None,
                unpaywall_email=settings.unpaywall_email or None,
                core_api_key=settings.core_api_key or None,
            )

            proposed = scout.search_and_extract(
                dag=self.dag, llm=self.llm,
            )
            self._display_proposed_edges(proposed)

            if not proposed:
                return

            # Interactive review
            console.print("\n[bold]Review proposed edges:[/bold]")
            for i, edge in enumerate(proposed):
                accept = console.input(
                    f"  [{i+1}] {edge.from_node} -> {edge.to_node} "
                    f"(conf={edge.match_confidence:.2f}, "
                    f"evidence={len(edge.evidence)}) "
                    f"[bold][a]ccept / [r]eject?[/bold] "
                ).strip().lower()
                if accept.startswith("a"):
                    console.print(f"    [green]Accepted: {edge.edge_id}[/green]")
                else:
                    console.print(f"    [dim]Rejected[/dim]")

        except Exception as e:
            console.print(f"[red]Edge proposal failed: {e}[/red]")

    def _display_proposed_edges(self, proposed: list) -> None:
        """Display proposed edges in a Rich table."""
        if not proposed:
            console.print("[dim]No edges proposed.[/dim]")
            return

        table = Table(title="Proposed Edges from Literature")
        table.add_column("#", style="dim")
        table.add_column("From -> To", style="cyan")
        table.add_column("Edge ID", style="white")
        table.add_column("Type", style="yellow")
        table.add_column("Confidence", style="white")
        table.add_column("Evidence", style="white")
        table.add_column("New Nodes?", style="dim")

        for i, edge in enumerate(proposed):
            table.add_row(
                str(i + 1),
                f"{edge.from_node} -> {edge.to_node}",
                edge.edge_id,
                edge.edge_type,
                f"{edge.match_confidence:.2f}",
                str(len(edge.evidence)),
                ", ".join(edge.requires_new_nodes) if edge.requires_new_nodes else "-",
            )

        console.print(table)

    def _handle_doctor(self) -> None:
        """DAG health check."""
        n_nodes = len(self.dag.nodes)
        n_edges = len(self.dag.edges)
        n_cards = len(self.edge_cards)
        n_ts = len(self.tsguard_results)
        edges_without_cards = [e.id for e in self.dag.edges if e.id not in self.edge_cards]
        n_issues = len(self.issue_ledger.issues) if self.issue_ledger else 0
        n_critical = len(self.issue_ledger.get_critical_open()) if self.issue_ledger else 0

        console.print(Panel(
            f"Nodes: {n_nodes}\n"
            f"Edges: {n_edges}\n"
            f"Edge cards: {n_cards} / {n_edges}\n"
            f"TSGuard results: {n_ts}\n"
            f"Issues: {n_issues} (CRITICAL open: {n_critical})\n"
            f"LLM: {'connected' if self.llm else 'not available (regex fallback)'}\n\n"
            + (f"[yellow]Edges without cards: {', '.join(edges_without_cards[:5])}[/yellow]"
               if edges_without_cards else "[green]All edges have cards[/green]"),
            title="DAG Health",
        ))

    # ──────────────────────────────────────────────────────────────────
    # Display helpers
    # ──────────────────────────────────────────────────────────────────

    def _display_propagation_result(
        self,
        result: Any,
        magnitude: float | None,
        raw_query: str,
    ) -> None:
        if not result.paths:
            console.print("[dim]No paths found.[/dim]")
            for w in result.warnings:
                console.print(f"  [yellow]{w}[/yellow]")
            return

        for i, path in enumerate(result.paths):
            edge_str = " -> ".join(e.edge_id for e in path.edges)

            if path.is_blocked:
                console.print(f"\n[red]Path {i + 1} (BLOCKED): {edge_str}[/red]")
                for reason in path.blocked_reasons:
                    console.print(f"  [red]{reason}[/red]")
                continue

            effect = path.total_effect
            scaled_effect = effect * magnitude if magnitude else effect
            se = path.total_se
            scaled_se = se * abs(magnitude) if magnitude else se

            table = Table(title=f"Path {i + 1}: {edge_str}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Effect (per unit)", f"{effect:.6f}")
            if magnitude:
                table.add_row(f"Effect (x {magnitude})", f"{scaled_effect:.6f}")
            table.add_row("SE", f"{se:.6f}")
            if magnitude:
                ci_lo = scaled_effect - 1.96 * scaled_se
                ci_hi = scaled_effect + 1.96 * scaled_se
                table.add_row("95% CI (scaled)", f"[{ci_lo:.6f}, {ci_hi:.6f}]")
            else:
                table.add_row("95% CI", f"[{path.ci_95[0]:.6f}, {path.ci_95[1]:.6f}]")
            table.add_row("SE method", f"[dim]{path.se_method}[/dim]")
            table.add_row("Edges", str(len(path.edges)))

            console.print(table)

            for w in path.warnings:
                console.print(f"  [yellow]WARNING: {w}[/yellow]")

        # Blocked edges summary
        if result.blocked_edges:
            console.print(f"\n[red]Blocked edges ({len(result.blocked_edges)}):[/red]")
            for be in result.blocked_edges:
                console.print(f"  [red]{be.edge_id}: {be.reason}[/red]")

        # LLM narration
        narration = _narrate_result(self.llm, result, self.mode, raw_query)
        if narration:
            console.print(Panel(narration, title="Interpretation", style="dim"))


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────


def start_repl(
    dag_path: Path | None = None,
    cards_dir: Path | None = None,
    mode: str = "REDUCED_FORM",
) -> None:
    """Start the query REPL. Called from CLI."""
    repl = QueryREPL(dag_path=dag_path, cards_dir=cards_dir, mode=mode)
    repl.run()
