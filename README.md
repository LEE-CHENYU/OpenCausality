```
    ___                    ____                        _ _ _
   / _ \ _ __   ___ _ __  / ___|__ _ _   _ ___  __ _ | (_) |_ _   _
  | | | | '_ \ / _ \ '_ \| |   / _` | | | / __|/ _` || | | __| | | |
  | |_| | |_) |  __/ | | | |__| (_| | |_| \__ \ (_| || | | |_| |_| |
   \___/| .__/ \___|_| |_|\____\__,_|\__,_|___/\__,_||_|_|\__|\__, |
        |_|                                                    |___/

  ~  sketch your causal story  ~  we find the papers, collect the
  data, pick the estimator, run the regressions, and tell you
  what's wrong before you publish something you'll regret  ~
```

# OpenCausality

You describe a causal story. OpenCausality estimates every edge, flags what's wrong, and blocks any claim the research design can't support — all in a hash-chained audit trail.

> **Paper:** [OpenCausality: Auditable Agentic Causal Inference](paper/main.pdf) (Li, 2026)
> **Full reference:** [REFERENCE.md](REFERENCE.md) — complete architecture, CLI reference, adapter details, and design philosophy.

---

## Quick Start

```bash
pip install -e .
opencausality init                    # creates .env with API keys
opencausality dag list                # see available DAGs
opencausality dag run my_dag.yaml     # estimate everything
opencausality query --dag my_dag.yaml # ask questions in plain English
```

No API key? Use `LLM_PROVIDER=claude_cli` or `LLM_PROVIDER=codex` — they shell out to local CLIs.

---

## What It Does

| Step | What happens |
|------|-------------|
| **You write a DAG** | Nodes + directed edges in YAML. Expected signs, identification strategies, unit specs. |
| **Agents run** | DataScout finds data. ModelSmith picks estimators. Estimator runs 19 adapters. Judge scores credibility. |
| **Guardrails fire** | 38 issue rules, 7 propagation guardrails, TSGuard time-series checks. Claims that the design can't support are blocked. |
| **You review** | Interactive HTML panels open automatically. Accept, reject, revise, or escalate each flagged issue. |
| **Audit trail** | Every decision — human and machine — is logged in a hash-chained JSONL ledger. |

The central design rule: **claim levels are a one-way ratchet.** The research design sets a ceiling. Diagnostics and human review can only push it down, never up. Good p-values don't make an OLS estimate causal.

---

## Case Study: Kazakhstan Bank Stress Testing

Starting from a single paragraph describing how FX shocks propagate to bank capital:

| Metric | NL-Extracted | After Critic (4 iters) | Full Pipeline |
|--------|-------------|----------------------|---------------|
| Nodes / Edges | 18 / 22 | 20 / 22 | 20 / 26 |
| Quality score | 0.349 | 0.668 | **0.894** |
| Open causal paths | 0/9 | 9/9 | **33/33** |
| Contributor paths to target | — | — | **91** |
| Identification coverage | 0% | 100% | 100% |

The NL pipeline also discovered **9 edges absent from the expert-built DAG**, including VIX → bank funding costs (risk-taking channel) and household income → cost of risk (bank lending channel).

---

## Claude Code Plugin

Open the project in Claude Code and just ask:

```
You:   "What if oil drops 30%?"
Claude: → propagate_shock(brent_price, -0.30, pct)
        → narrates results with hedged language, cites guardrail blocks

You:   "Which edges are weakly identified?"
Claude: → get_identification() → formats table of claim levels and risks

You:   "Run placebo tests"
Claude: → run_placebo() → inspect_edge() for failures → synthesizes findings
```

13 MCP tools. Full guardrail enforcement. Read-only — AI exploration can't alter governed artifacts.

```bash
pip install "mcp>=1.0.0"  # only prerequisite
# verify: /mcp in Claude Code → opencausality-query with 13 tools
```

---

## Key Features

**Estimation** — 19 adapters: Local Projections, Panel FE, IV-2SLS, DiD, RDD, Regression Kink, Synthetic Control, DoWhy (Backdoor/IV/Frontdoor), DoubleML, EconML CATE, CausalML Uplift, plus Identity/Bridge/Immutable.

**Propagation** — 7 guardrails gate every causal path: mode gating, counterfactual eligibility, TSGuard diagnostics, issue severity, reaction-function blocking, unit compatibility, frequency alignment. Delta-method SEs with independence caveat.

**Issue Detection** — 38 rules catch overclaiming, control shopping, null dropping, specification drift, timing failures. PatchBot auto-fixes safe issues (missing units, edge ID syntax). PatchPolicy explicitly prohibits control shopping, sample trimming, lag searching, outcome switching.

**DAG Critic** — Iterative LLM-based refinement with 10 revision types, 4 causal logic probes, 7-component quality score. KnowledgeScout auto-researches missing formulas. DAGHealthSnapshot tracks per-edge health to catch regressions. Data-availability oracle rejects edges with no data.

**Sentinel Loop** — Always-on background monitor. Auto-starts with the pipeline, re-validates the DAG after every change, auto-heals schema issues, rebuilds and opens HTML panels on completion.

**Governance** — Hash-chained JSONL audit log (no database). HITL panel with accept/reject/revise/escalate. Claim levels determined by design, not results. Everything is git-committable.

---

## DAG in 30 Seconds

```yaml
nodes:
  - id: fx_shock
    label: "Exchange Rate Shock"
    variable: "usdkzt_pct_change"
    frequency: monthly

edges:
  - source: fx_shock
    target: cpi_inflation
    expected_sign: positive
    identification: back_door
    controls: [oil_price, policy_rate]
    unit_spec: "1pp FX depreciation -> X pp inflation"
```

Each edge is a testable hypothesis. The system estimates it, checks identification, flags issues, and produces an EdgeCard (YAML) with estimates, diagnostics, credibility rating, and literature references.

---

## Architecture

```
shared/
├── agentic/          # DAG parser, agents, governance, issues, propagation
│   ├── agents/       #   DataScout, PaperScout, DAGCritic, KnowledgeScout,
│   │                 #   PatchBot, ModelSmithCritic, PaperDAGExtractor
│   ├── propagation.py #  PropagationEngine (7 guardrails)
│   └── ts_guard.py   #  Time-series validator (7 diagnostics)
├── engine/           # 19 estimation adapters + data assembly
├── llm/              # LLM abstraction (Anthropic, LiteLLM, Claude CLI, Codex CLI)
└── data/             # Data clients

config/agentic/dags/  # Your DAG YAML files
outputs/agentic/      # EdgeCards, issues, audit logs, HTML panels
```

---

## Agent Pipeline

| Agent | What it does |
|-------|-------------|
| **DataScout** | Finds and fetches data. Oracle functions reject edges with no data. |
| **ModelSmith** | Picks estimation design from 19 adapters based on DAG structure. |
| **Estimator** | Runs estimation, TSGuard diagnostics, produces EdgeCards. |
| **Judge** | Scores credibility, flags weak links, triggers PatchBot for safe fixes. |
| **DAGCritic** | LLM-based iterative refinement with eval framework. |
| **KnowledgeScout** | Researches missing identity formulas via Claude CLI. |

---

## CLI

```bash
opencausality dag run <path>              # full estimation pipeline
opencausality dag validate <path>         # check structure only
opencausality dag generate <narrative>    # NL text → DAG
opencausality dag viz                     # interactive D3.js graph
opencausality query                       # natural language REPL
opencausality config doctor               # diagnose setup issues
opencausality benchmark run --suite all   # run DGP + ACIC benchmarks
```

Full CLI reference: [REFERENCE.md](REFERENCE.md#cli-reference)

---

## Limitations

- **NL-to-DAG is lossy.** Single paragraphs can't encode full causal chains. Treat extracted DAGs as hypotheses.
- **LLMs hallucinate edges.** All LLM-extracted edges must pass HITL review before use.
- **Independence assumption.** Delta-method SEs assume edge independence. Shared confounders cause underestimation.
- **Single case study.** The platform is domain-agnostic, but bundled data clients are Kazakhstan-specific.

---

## Contributing

Fork, branch from `main`, PR. Write tests (`pytest`). Follow existing patterns (dataclasses, YAML, JSONL). Run `ruff check shared/ scripts/`. See [REFERENCE.md](REFERENCE.md#contributing) for details.

## License

MIT. See [LICENSE](LICENSE).
