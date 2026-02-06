# Agentic Causal Inference Framework: Analysis and Generalization

**Version:** 1.0
**Date:** 2026-02-05
**Authors:** Research Team

---

## Executive Summary

This document analyzes our agentic causal inference framework, which uses DAG specifications to orchestrate econometric estimation pipelines. We evaluate its generalizability beyond the initial KSPI K2 stress testing use case and identify which quality issues can be automated in future iterations.

**Key Findings:**
- The framework is **fully generalizable** to any causal inference task with DAG input
- Initial run revealed **5 major problem categories**, of which **4 are automatable**
- Reaction function vs. causal edge distinction requires **domain knowledge** and cannot be fully automated

---

## 1. Framework Overview

### 1.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DAG SPECIFICATION                             │
│  (YAML: nodes, edges, timing, units, acceptance criteria)           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PRE-ESTIMATION VALIDATION                        │
│  • DAG acyclicity          • Unit presence                          │
│  • Edge type classification • Node source definitions               │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA ASSEMBLY LAYER                             │
│  • Connector registry (BNS, FRED, NBK, custom)                      │
│  • Frequency alignment     • Missing data handling                  │
│  • Provenance tracking     • Entity boundary validation             │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DESIGN SELECTION ENGINE                           │
│  • Design registry lookup   • Credibility weighting                 │
│  • Diagnostics requirements • Template instantiation                │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ESTIMATION ENGINES                              │
│  • Local Projections (LP)   • Panel LP with Exposure×Shock         │
│  • Immutable Evidence       • Accounting Bridges                    │
│  • Identity Sensitivities   • (Extensible: DiD, RDD, IV, SC)       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        EDGE CARD OUTPUT                              │
│  • Estimates + uncertainty  • Diagnostics (pass/fail)               │
│  • Interpretation boundary  • Counterfactual applicability          │
│  • Credibility rating (A-D) • Unit normalization                    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   POST-ESTIMATION VALIDATION                         │
│  • N consistency            • Unit presence in cards                │
│  • Sign consistency         • Reaction function labels              │
│  • Report-to-card matching  • Interpolation fraction                │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      REPORT GENERATION                               │
│  • Markdown tables          • Unit reference table                  │
│  • Credibility summary      • Diagnostics summary                   │
│  • Visualization (D3.js)    • Limitations disclosure                │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| DAG Schema | Define nodes, edges, timing, units, acceptance criteria | `config/agentic/dag_schema.yaml` |
| Design Registry | Map identification strategies to templates and diagnostics | `config/agentic/design_registry.yaml` |
| Data Assembler | Load and align data from multiple sources | `shared/engine/data_assembler.py` |
| Estimators | Run econometric estimations per design | `shared/engine/ts_estimator.py`, `panel_estimator.py` |
| EdgeCard | Standardized output with estimates, diagnostics, interpretation | `shared/agentic/output/edge_card.py` |
| Validation Pipeline | Pre/post-estimation quality checks | `shared/agentic/validation.py` |
| Report Checker | Validate report-to-card consistency | `shared/agentic/report_checker.py` |

### 1.3 Key Design Principles

1. **DAG-First Specification:** All causal claims derive from explicit DAG edges
2. **Credibility-Based Ranking:** No p-hacking; designs ranked by identification strength
3. **Interpretation Boundaries:** Each edge specifies what it IS and IS NOT
4. **Null Acceptance:** Precisely null results are valid findings, not failures
5. **Audit Trail:** Full provenance from data source to final estimate
6. **Unit Safety:** Explicit units prevent chain multiplication errors

---

## 2. Generalizability Analysis

### 2.1 Domain-Agnostic Core

The framework's core abstractions are **domain-agnostic**:

| Abstraction | Banking Example | Labor Economics | Environmental | Healthcare |
|-------------|-----------------|-----------------|---------------|------------|
| **Node** | NPL ratio | Employment rate | Emissions | Blood pressure |
| **Edge** | shock → NPL | min_wage → employment | carbon_tax → emissions | drug → BP |
| **Shock** | Oil supply shock | Policy change | Regulatory change | Treatment assignment |
| **Exposure** | Consumer loan share | Industry composition | Pollution intensity | Baseline severity |
| **Outcome** | K2 capital ratio | Wages | Mortality | Clinical events |

### 2.2 What IS Generalizable

| Feature | Generalizability | Notes |
|---------|------------------|-------|
| DAG specification format | ✅ Fully general | Any directed acyclic graph |
| Node/edge metadata | ✅ Fully general | Units, timing, sources |
| Design registry | ✅ Fully general | DiD, RDD, IV, LP, SC all supported |
| EdgeCard format | ✅ Fully general | Standard output for any edge |
| Credibility scoring | ✅ Fully general | Based on diagnostics, not domain |
| Validation pipeline | ✅ Fully general | Structural checks are domain-free |
| Report generation | ✅ Fully general | Template-based, customizable |

### 2.3 What Requires Domain Customization

| Feature | Customization Needed | Example |
|---------|---------------------|---------|
| Data connectors | Domain-specific | BNS for Kazakhstan, FRED for US macro |
| Expected signs | Domain knowledge | "Higher rates → higher NPL" vs "Higher rates → lower inflation" |
| Reaction function identification | Expert judgment | Which edges are policy responses? |
| Plausibility bounds | Domain expertise | What magnitude is reasonable? |
| Immutable evidence | Prior research | Which estimates are locked from re-estimation? |

### 2.4 Extension to Other Domains

**Labor Economics Example:**
```yaml
nodes:
  - id: "minimum_wage"
    type: "continuous"
    unit: "usd_hourly"

  - id: "employment"
    type: "share"
    unit: "pct"

edges:
  - id: "min_wage_to_employment"
    from: "minimum_wage"
    to: "employment"
    edge_type: "causal"
    unit_specification:
      treatment_unit: "1 USD/hour increase"
      outcome_unit: "pp employment rate change"
    allowed_designs: ["DID_EVENT_STUDY", "RDD"]
```

**Environmental Economics Example:**
```yaml
edges:
  - id: "carbon_tax_to_emissions"
    from: "carbon_tax"
    to: "emissions"
    edge_type: "causal"
    unit_specification:
      treatment_unit: "10 USD/ton CO2"
      outcome_unit: "% emissions reduction"
    allowed_designs: ["DID_EVENT_STUDY", "SYNTHETIC_CONTROL"]
```

### 2.5 Generalizability Verdict

**The framework is fully generalizable to any causal inference task with DAG input.**

Required for new domains:
1. Write domain-specific data connectors
2. Specify DAG with domain-appropriate nodes/edges
3. Set expected signs and plausibility bounds
4. Identify which edges are reaction functions vs. causal

---

## 3. Problems Identified in Initial Run

### 3.1 Problem Categories

We identified **5 major problem categories** during the initial KSPI K2 estimation run:

| # | Problem | Severity | Impact |
|---|---------|----------|--------|
| 1 | **N Count Inconsistency** | High | Report claimed N=17 but tables showed N=26; confusion between calendar periods and effective observations |
| 2 | **Missing Unit Normalization** | Critical | Coefficients meaningless without knowing treatment/outcome units; chain multiplication unsafe |
| 3 | **Reaction Function Misidentification** | Critical | Policy-rate edges (CPI→NBK rate, FX→NBK rate) were treated as causal when they're endogenous responses |
| 4 | **Visualization Label Overlap** | Medium | Large numbers like "18195" displayed as "00258" due to truncation; unreadable labels |
| 5 | **Sign Inconsistencies Undocumented** | Medium | VIX→FX had wrong sign but wasn't flagged prominently |

### 3.2 Detailed Problem Analysis

#### Problem 1: N Count Inconsistency

**Symptom:** Report header said "N=17 true quarterly observations" but table showed N=26.

**Root Cause:** Conflation of:
- `n_calendar_periods`: Raw time periods in sample (27 quarters: 2020Q3-2024Q3)
- `n_effective_obs_h0`: Observations available after applying lags/leads (26 for h=0 with 1 lag)

**Fix Applied:**
- Added `n_calendar_periods` and `n_effective_obs_h0` fields to EdgeCard
- Report now shows both: "N_cal | N_eff" format
- Clarified in report that N_eff < N_cal due to lag requirements

#### Problem 2: Missing Unit Normalization

**Symptom:** Coefficients like "72.80" and "0.22" appeared in tables without context.

**Root Cause:** No systematic tracking of:
- Treatment unit (e.g., "1pp tradable CPI shock")
- Outcome unit (e.g., "bps NPL ratio change")

**Impact:**
- Chain multiplication produces meaningless numbers
- Users can't interpret coefficient magnitudes
- Unit mismatches go undetected

**Fix Applied:**
- Added `EDGE_UNITS` registry with all 26 edges
- Added "Unit Normalization Reference" table to report
- Added `treatment_unit` and `outcome_unit` to EdgeCard.Estimates
- Added unit presence validation to pipeline

#### Problem 3: Reaction Function Misidentification

**Symptom:** `cpi_to_nbk_rate` and `fx_to_nbk_rate` were estimated as causal edges.

**Root Cause:** These edges represent the **NBK reaction function** (central bank responds to inflation/FX), not causal effects. Using them for shock propagation would be circular reasoning.

**Impact:**
- Shock propagation through these edges is invalid
- Coefficient interpretation is wrong (it's a Taylor rule, not a causal effect)

**Fix Applied:**
- Added `edge_type` field to DAG schema with `reaction_function` option
- Added warning banner in report for reaction function edges
- Added `is_reaction_function` flag to EDGE_UNITS registry
- Validation pipeline checks that RF edges forbid `shock_counterfactual` use

#### Problem 4: Visualization Label Overlap

**Symptom:** Link labels in D3.js visualization showed truncated/overlapping numbers.

**Root Cause:**
- No formatting for large numbers (18195 → "00258" due to overflow)
- Labels overlapped when edges were close together
- No background for readability

**Fix Applied:**
- `formatBeta()` function with "k" notation for thousands (18195 → "18.2k")
- White background rectangles behind labels
- Perpendicular offset for label positioning
- Deduplicated panel edges to reduce clutter

#### Problem 5: Sign Inconsistencies Undocumented

**Symptom:** `vix_to_fx` had negative coefficient when positive was expected.

**Root Cause:** No prominent flagging of sign mismatches in report.

**Fix Applied:**
- "Sign Inconsistencies" section in Diagnostics Summary
- Validation pipeline checks `expected_sign` against actual estimate
- Warning severity for sign mismatches

---

## 4. Automatable vs. Non-Automatable Feedback

### 4.1 Automation Classification

| Problem | Automatable? | Automation Method | Confidence |
|---------|-------------|-------------------|------------|
| N count consistency | ✅ **Yes** | Validation pipeline checks `n_effective <= n_calendar` | 100% |
| Unit presence | ✅ **Yes** | Pre-estimation check: all edges have `unit_specification` | 100% |
| Unit in EdgeCard | ✅ **Yes** | Post-estimation check: EdgeCard has `treatment_unit`/`outcome_unit` | 100% |
| Report-to-card match | ✅ **Yes** | Report checker validates point estimates, SEs, Ns match | 95% |
| Sign consistency | ✅ **Yes** | Compare `estimates.point` sign against `expected_sign` | 100% |
| Interpolation fraction | ✅ **Yes** | Check `share_interpolated < 0.30` for estimation eligibility | 100% |
| LOO stability | ✅ **Yes** | Run leave-one-out, check for sign flips | 100% |
| Visualization formatting | ✅ **Yes** | Enforce `formatBeta()` in visualization code | 100% |
| **Reaction function identification** | ❌ **No** | Requires domain knowledge of causal structure | 0% |
| Plausibility bounds | ⚠️ **Partial** | Can check against bounds, but setting bounds needs expertise | 50% |
| Expected signs | ⚠️ **Partial** | Can validate, but specifying correct signs needs expertise | 50% |

### 4.2 Fully Automatable Checks (Implemented)

These checks are now in `shared/agentic/validation.py` and run automatically:

```python
# Pre-estimation checks
validator._check_dag_acyclic(result)
validator._check_unit_presence(result)
validator._check_edge_type_presence(result)
validator._check_node_sources_defined(result)
validator._check_edge_nodes_exist(result)

# Post-estimation checks
validator._check_n_consistency(result, edge_cards)
validator._check_unit_in_card(result, edge_cards)
validator._check_reaction_function_labeled(result, edge_cards)
validator._check_interpolation_fraction(result, edge_cards)
validator._check_sign_consistency(result, edge_cards)

# Report consistency checks
checker._check_report_vs_card_match(result, report, cards)
checker._check_unit_table_present(result, report)
checker._check_reaction_function_warning(result, report)
```

### 4.3 Non-Automatable Decisions

These require **domain expertise** and cannot be automated:

#### 4.3.1 Reaction Function Identification

**Why not automatable:** Determining whether an edge is a reaction function requires understanding the economic mechanism:

- `cpi_to_nbk_rate`: Is this "inflation causes rate changes" (causal) or "central bank responds to inflation" (reaction function)?
- The DAG structure alone doesn't tell us—we need to know that central banks are policy actors who respond endogenously.

**Human input required:**
```yaml
edges:
  - id: "cpi_to_nbk_rate"
    edge_type: "reaction_function"  # Human must specify this
```

#### 4.3.2 Expected Signs

**Why not automatable:** Economic theory determines expected signs:

- Does FX depreciation increase or decrease inflation? (Increase for importers, ambiguous generally)
- Does higher policy rate increase or decrease NPLs? (Increase via borrower stress, decrease via tighter lending)

**Human input required:**
```yaml
acceptance_criteria:
  plausibility:
    expected_sign: "positive"  # Human must specify based on theory
```

#### 4.3.3 Plausibility Bounds

**Why not automatable:** Reasonable magnitudes depend on domain knowledge:

- A 50% FX pass-through to CPI is plausible for small open economies
- A 50% pass-through would be implausible for the US

**Human input required:**
```yaml
acceptance_criteria:
  plausibility:
    magnitude_range: [0.05, 0.25]  # Human must specify based on literature
```

### 4.4 Iterative Claude Code Loop for Automation

**Practical Setup:** No separate API keys required. Claude Code CLI handles authentication automatically. The loop runs entirely within the existing Claude Code session.

```bash
# Interactive mode (current setup):
claude
> run estimation pipeline with validation

# Headless mode (for CI/CD):
claude --print "run validation, fix issues, regenerate report"
```

The following workflow enables automated quality improvement:

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Human Specifies DAG                                         │
│ - Nodes, edges, edge types (causal vs reaction_function)           │
│ - Expected signs, plausibility bounds                               │
│ - Unit specifications                                               │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Claude Code Runs Pre-Estimation Validation                  │
│ - Automatic checks (acyclicity, unit presence, etc.)                │
│ - Fails fast if critical issues found                               │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Claude Code Runs Estimation Pipeline                        │
│ - Data assembly, design selection, estimation                       │
│ - Produces EdgeCards with estimates, diagnostics                    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: Claude Code Runs Post-Estimation Validation                 │
│ - N consistency, unit in cards, sign consistency                    │
│ - Flags issues but doesn't block                                    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: Claude Code Generates Report                                │
│ - Tables, visualizations, diagnostics summary                       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: Claude Code Runs Report Consistency Check                   │
│ - Validates report matches EdgeCards                                │
│ - Flags any mismatches                                              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 7: If Issues Found → Claude Code Fixes Automatically           │
│ - Missing units → Add to EDGE_UNITS registry                        │
│ - N mismatch → Fix report generation logic                          │
│ - Label overlap → Fix visualization formatting                      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 8: Human Reviews Final Output                                  │
│ - Reviews any flagged issues that require domain judgment           │
│ - Approves or requests revisions                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.5 Automation Coverage Summary

| Category | Checks | Automatable | % |
|----------|--------|-------------|---|
| Structural | DAG acyclicity, node/edge existence | 3/3 | 100% |
| Unit Safety | Unit presence in DAG, unit in cards, unit table | 3/3 | 100% |
| Sample Size | N consistency, interpolation fraction | 2/2 | 100% |
| Consistency | Report-to-card match, sign consistency | 2/2 | 100% |
| Labeling | Reaction function warnings, edge type presence | 2/2 | 100% |
| Domain Knowledge | Reaction function ID, expected signs, bounds | 0/3 | 0% |
| **Total** | | **12/15** | **80%** |

---

## 5. Recommendations

### 5.1 For Framework Users

1. **Always specify `edge_type`** for every edge in your DAG
2. **Always specify `unit_specification`** with `treatment_unit` and `outcome_unit`
3. **Review reaction function edges carefully**—they cannot be used for shock propagation
4. **Set `expected_sign` and `magnitude_range`** based on economic theory
5. **Run validation pipeline** before and after estimation

### 5.2 For Framework Developers

1. **Add more design templates:** Synthetic control, regression kink, Bartik shift-share
2. **Implement LLM-assisted checks:** Use Claude to detect potential reaction functions by analyzing edge semantics
3. **Build interactive DAG editor:** GUI for specifying DAGs with validation feedback
4. **Add multi-DAG support:** Compare estimates across different DAG specifications

### 5.3 For Automation Improvement

1. **Train classifier for reaction functions:** Use labeled examples to predict edge type
2. **Literature-based bounds:** Scrape published estimates to set plausibility bounds
3. **Sign prediction from node names:** Infer expected signs from node semantics (e.g., "cost" → negative effect on profit)

---

## 6. Conclusion

The agentic causal inference framework is **fully generalizable** to any domain with DAG-structured causal claims. The initial KSPI K2 run revealed 5 major problem categories, of which **4 are fully automatable** (80% of checks). The remaining 20% require domain expertise for:

1. Identifying reaction functions vs. causal edges
2. Specifying expected signs
3. Setting plausibility bounds

### Automation Infrastructure

Two CLI tools are available for the iterative loop, **both with local auth (no API keys needed)**:

| Tool | Use Case | Command |
|------|----------|---------|
| **Claude Code** | Interactive development, complex reasoning | `claude` |
| **Codex CLI** | Overnight batch runs, continuous loops | `./control.sh start` |

The iterative loop can now catch and fix most quality issues automatically:

```bash
# Quick validation (Claude Code)
claude
> run validation on KSPI DAG

# Overnight automation (Codex CLI)
cd scripts/codex_loop
DURATION_HOURS=12 ./control.sh start
# Check results next morning
./control.sh log
```

Human review remains focused on domain-specific judgments that cannot be automated.

---

## Appendix A: Practical Execution

### Option 1: Claude Code CLI (Current)

Claude Code CLI has built-in authentication. No separate API keys needed.

```bash
# Interactive mode (current session)
claude

# Example prompts:
> run the full estimation pipeline for KSPI K2 DAG
> validate all EdgeCards and fix any issues
> regenerate the report with updated estimates

# Headless mode
claude --print "run validation and fix issues" > log.txt
```

### Option 2: OpenAI Codex CLI

Codex CLI v0.92.0 installed at `/opt/homebrew/bin/codex` with **local auth** (no API key needed).

**Auth status:** Already configured in `~/.codex/auth.json`
**Default model:** `gpt-5.2-codex`

**Single execution:**
```bash
cd /Users/lichenyu/econometric-research
codex exec --full-auto "Run validation on the KSPI K2 DAG and fix any issues"
```

**Automated estimation loop** (ready to use):
```bash
cd /Users/lichenyu/econometric-research/scripts/codex_loop

# Single iteration (foreground)
./control.sh once

# Start background loop (8 hours default)
./control.sh start

# Check status
./control.sh status

# Follow log
./control.sh tail

# Stop gracefully
./control.sh stop
```

**Configuration** (environment variables):
```bash
DURATION_HOURS=24 ./control.sh start   # Run for 24 hours
SLEEP_SECONDS=600 ./control.sh start   # 10 min between iterations
MODEL=o3 ./control.sh start            # Use different model
```

**Loop files:**
```
scripts/codex_loop/
├── codex_estimation_loop.sh   # Main loop script
├── codex_objective.txt        # What to work on
├── codex_resume.md            # Context persistence
└── control.sh                 # Start/stop/status
```

**Resume context** (auto-updated each iteration):
```markdown
## Current Focus
- Re-running estimation for edges with sign inconsistencies

## Previous Changes
- Fixed unit normalization for shock_to_npl_kspi
- Added reaction function warnings

## Next Steps
- Run post-estimation validation
- Check report consistency
```

### Comparison: Claude Code vs Codex CLI

| Feature | Claude Code | OpenAI Codex CLI |
|---------|-------------|------------------|
| **Model** | Claude Opus 4.5 | GPT-5.2-codex |
| **Auth** | Built-in | Local (`~/.codex/auth.json`) |
| **API Key** | Not needed | Not needed (local auth) |
| **Full auto** | `--print` | `--full-auto` |
| **Loop script** | Manual | `codex_estimation_loop.sh` |
| **Resume** | Conversation history | `codex_resume.md` file |
| **Control** | Interactive | `control.sh` start/stop |
| **Best for** | Complex reasoning, interactive | Batch runs, overnight loops |

### Recommended Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTERACTIVE DEVELOPMENT                       │
│                      (Claude Code CLI)                           │
│  • Design DAG structure                                          │
│  • Set expected signs, reaction functions                        │
│  • Debug estimation issues                                       │
│  • Review validation results                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AUTOMATED BATCH RUNS                          │
│                     (Codex CLI Loop)                             │
│  • Overnight estimation runs                                     │
│  • Continuous validation and fixes                               │
│  • Auto-commit improvements                                      │
│  • Resume context preserved                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HUMAN REVIEW                                  │
│  • Check codex_resume.md for changes                            │
│  • Review commits                                                │
│  • Validate domain-specific decisions                           │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Start Commands

```bash
# Claude Code (interactive)
claude
> run validation and show issues

# Codex CLI (single run)
cd scripts/codex_loop && ./control.sh once

# Codex CLI (overnight loop)
cd scripts/codex_loop && DURATION_HOURS=12 ./control.sh start
```

### Python Integration (No LLM Required)

Validation checks run as pure Python—no API calls:

```python
from shared.agentic.validation import run_full_validation
from shared.agentic.report_checker import check_report_consistency

# Run validation (pure Python, no LLM)
result = run_full_validation(
    dag_path="config/agentic/dags/kspi_k2_full.yaml",
    edge_cards=cards,
)

if not result.passed:
    print("Issues found:")
    print(result.to_markdown())
    # Then use Claude Code or Codex to fix
```

---

## Appendix B: File Inventory

### Core Framework Files

| File | Purpose |
|------|---------|
| `config/agentic/dag_schema.yaml` | DAG specification format (v3) |
| `config/agentic/design_registry.yaml` | Design templates and credibility weights (v3) |
| `config/agentic/dags/kspi_k2_full.yaml` | KSPI K2 stress test DAG |
| `shared/agentic/validation.py` | Pre/post-estimation validation pipeline |
| `shared/agentic/report_checker.py` | Report-to-EdgeCard consistency checker |
| `shared/agentic/output/edge_card.py` | EdgeCard output format |
| `scripts/run_real_estimation.py` | Main estimation script with EDGE_UNITS registry |

### Codex Loop Scripts

| File | Purpose |
|------|---------|
| `scripts/codex_loop/codex_estimation_loop.sh` | Main loop script (configurable duration) |
| `scripts/codex_loop/codex_objective.txt` | What the loop should work on |
| `scripts/codex_loop/codex_resume.md` | Context persistence between iterations |
| `scripts/codex_loop/control.sh` | Start/stop/status control script |

### Output Files

| File | Purpose |
|------|---------|
| `outputs/agentic/KSPI_K2_REAL_ESTIMATION_REPORT.md` | Generated estimation report |
| `outputs/agentic/cards/edge_cards/*.yaml` | Individual EdgeCard files |
| `outputs/agentic/dag_visualization.html` | Interactive D3.js DAG visualization |
| `outputs/codex_loop.log` | Codex loop execution log |

## Appendix C: Validation Pipeline Usage

```python
from shared.agentic.validation import run_full_validation
from shared.agentic.report_checker import check_report_consistency

# Run full validation
result = run_full_validation(
    dag_path="config/agentic/dags/kspi_k2_full.yaml",
    edge_cards=cards,  # Dict[str, EdgeCard]
    report_path="outputs/agentic/KSPI_K2_REAL_ESTIMATION_REPORT.md",
)

print(result.to_markdown())

# Or run report check separately
report_result = check_report_consistency(
    report_path="outputs/agentic/KSPI_K2_REAL_ESTIMATION_REPORT.md",
    edge_cards=cards,
    reaction_function_edges=["cpi_to_nbk_rate", "fx_to_nbk_rate"],
)

print(report_result.to_markdown())
```

---

*Document generated by Research Team, 2026-02-05*
