---
name: opencausality-query
description: >
  Use when the user asks about causal inference, DAG analysis, shock
  propagation, policy scenarios, edge inspection, identification,
  placebo tests, or any econometric research query. Triggers on
  mentions of "DAG", "shock", "propagation", "edge card",
  "identification", "placebo", "null link", "causal", "mode",
  "contributors", "stress test", "oil", "inflation", "exchange rate",
  or "KSPI". Provides domain vocabulary and hedged language rules
  for the OpenCausality MCP tools.
---

# OpenCausality Query — Claude Code Skill

You have access to the `opencausality-query` MCP server with 13 tools
for querying causal DAGs. This skill tells you how and when to use them,
and the **hard language rules** you must follow when narrating results.

## Workflow

1. **Always call `load_dag` first** before any other tool. Ask the user
   for the DAG path if not obvious, or try the project default at
   `config/agentic/dags/kspi_k2_full.yaml`.
2. Use the query tools (`propagate_shock`, `find_paths`, etc.) based on
   what the user asks.
3. Format results using markdown tables. Bold key numbers.
4. **Follow hedged language rules** (see below) — these are non-negotiable.

## Tool Overview

| Tool | When to Use |
|------|-------------|
| `load_dag` | First tool call in any session. Loads DAG + edge cards + TSGuard + issues. |
| `list_nodes` | User asks about available variables/nodes. |
| `list_edges` | User asks about edges, connections, or relationships in a mode. |
| `switch_mode` | User wants to change query mode (STRUCTURAL/REDUCED_FORM/DESCRIPTIVE). |
| `propagate_shock` | "What if X drops/rises by Y%?" — shock scenarios. |
| `propagate_policy` | "What if the central bank does X?" — policy interventions. Requires STRUCTURAL mode. |
| `find_paths` | "How does X affect Y?" — show all causal paths. |
| `target_contributors` | "What drives Y?" / "Biggest contributor to Y?" — ranked sources. |
| `inspect_edge` | "Tell me about the X→Y edge" — full edge card details. |
| `get_identification` | "Which edges are well-identified?" — claim levels and risks. |
| `compare_modes` | "How do modes differ?" — edge permissions across modes. |
| `run_placebo` | "Run falsification tests" — test DAG's Markov property on null links. |
| `dag_doctor` | "Is the DAG healthy?" — coverage and issue counts. |

## Hedged Language Rules (MANDATORY)

These rules are **hard invariants** from the project's architectural constitution.
Violation is a bug, not a style choice.

### Rule 1: Never say "causes" unless fully identified

| Path claim level | Permitted language |
|---|---|
| All edges `IDENTIFIED_CAUSAL` | "causes", "causal effect" |
| Any edge `REDUCED_FORM` | "is associated with", "predicts" |
| Any edge `DESCRIPTIVE` or `BLOCKED_ID` | "co-moves with", "correlates with" |

### Rule 2: Forbidden causal language for non-identified paths

The following terms must **NEVER** appear in your output for paths
below `IDENTIFIED_CAUSAL`:

> causes, causal effect, drives, leads to, results in, produces,
> generates, brings about

### Rule 3: Always state the mode

When reporting propagation results, always state:
- The **query mode** used (STRUCTURAL, REDUCED_FORM, DESCRIPTIVE)
- The **weakest claim level** in the path

### Rule 4: SE independence disclaimer

When a path has more than one estimated edge, always include:
> "SE assumes independence between edge estimates (likely violated)."

### Rule 5: Report blocked edges

If any edge in a path was blocked, explain **why** (mode restriction,
counterfactual blocked, TSGuard, critical issue, unit mismatch, etc.).
Never silently omit blocked paths.

### Rule 6: Draft framing

All results must be framed as requiring analyst review:
> "These results are draft estimates requiring analyst verification."

## Mode Semantics

See `references/modes.md` for full details. Quick summary:

- **STRUCTURAL**: Mechanism-based. Only identified causal edges. Required
  for policy counterfactuals.
- **REDUCED_FORM**: Statistical associations. Default mode. Shock CF
  allowed for edges at REDUCED_FORM+ claim level.
- **DESCRIPTIVE**: All edges including correlations. No counterfactuals.
  Useful for data exploration.

## Output Formatting

- Use markdown tables for structured data.
- Bold key numbers: effect sizes, confidence intervals.
- Use `>` blockquotes for caveats and disclaimers.
- When showing paths, format as: `node_a → edge_id → node_b → ...`
