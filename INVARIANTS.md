# OpenCausality: Architectural Invariants and Design Principles

**This document is the architectural constitution of OpenCausality.** Every invariant
listed here is version-independent and must hold across all future changes. AI coding
agents, contributors, and reviewers must treat violations of these invariants as bugs,
regardless of what other code or comments suggest.

If a proposed change conflicts with any invariant in this document, the change is wrong
— not the invariant. Invariants may only be amended by explicit, documented decision
with justification recorded in the audit trail.

---

## 1. The Foundational Principle

**Identification comes from the research design, not from the data.**

This single sentence governs every architectural decision in OpenCausality. A good
t-statistic does not make an OLS estimate causal. A strong first-stage F-statistic
does not make a local projection into an IV estimate. The system exists to enforce
this principle computationally.

---

## 2. Claim Level Hierarchy

### 2.1 The Four Levels (Immutable Ordering)

```
IDENTIFIED_CAUSAL  >  REDUCED_FORM  >  DESCRIPTIVE  >  BLOCKED_ID
     (index 0)          (index 1)       (index 2)       (index 3)
```

This ordering is the foundation of the entire system. It must never be reordered,
extended with intermediate levels, or collapsed. All comparison logic depends on
index position: lower index = stronger claim.

### 2.2 The One-Way Ratchet

**Claim levels can only be maintained or downgraded — never upgraded — after the
design is chosen.**

The mechanism has three stages:

1. **Design sets the ceiling.** The `DESIGN_CLAIM_MAP` determines the maximum claim
   level an edge can ever achieve. This mapping is deterministic and fixed:

   | Design | Maximum Claim |
   |--------|--------------|
   | IV_2SLS, RCT, DID_EVENT_STUDY, RDD, PANEL_LP_EXPOSURE_FE | `IDENTIFIED_CAUSAL` |
   | DOWHY_BACKDOOR, DOWHY_IV, DOWHY_FRONTDOOR | `IDENTIFIED_CAUSAL` |
   | SYNTHETIC_CONTROL, REGRESSION_KINK | `IDENTIFIED_CAUSAL` |
   | LOCAL_PROJECTIONS, VAR, TS_LOCAL_PROJECTION | `REDUCED_FORM` |
   | DML_PLR, DML_IRM, DML_PLIV | `REDUCED_FORM` (ML nuisance does not confer identification) |
   | ECONML_CATE, CAUSALML_UPLIFT | `REDUCED_FORM` (HTE estimation, not identification) |
   | PANEL_FE_BACKDOOR | `REDUCED_FORM` |
   | OLS | `DESCRIPTIVE` |
   | IDENTITY, ACCOUNTING_BRIDGE, IMMUTABLE_EVIDENCE | `IDENTIFIED_CAUSAL` (mechanical) |

2. **Diagnostics can only cap downward.** The `cap_to()` function is monotone:
   ```
   if current_idx < max_idx:    # current is stronger than cap
       claim_level = max_level  # downgrade
   # else: no-op (silently ignore upward cap)
   ```
   Specific diagnostic failures trigger specific caps:
   - Leads test failure → `BLOCKED_ID`
   - Exposure variation failure → `BLOCKED_ID`
   - Weak first-stage F → `REDUCED_FORM`
   - Leave-one-out instability → `REDUCED_FORM`
   - Nonstationarity (non-ECM) → `DESCRIPTIVE`

3. **Human review can only confirm or downgrade.** The HITL panel actions all maintain
   or reduce the claim. There is no "approve as causal" action. Each action has fixed
   pipeline semantics:

   | Action | Pipeline Effect |
   |--------|----------------|
   | **accept** | Acknowledges issue; downgrades credibility rating; edge enters report with caveat. Does **not** restore or elevate claim level. |
   | **reject** | Suppresses the edge entirely from the final report. |
   | **revise** | Triggers re-specification and re-estimation from scratch. The new design determines the new ceiling — starts the ratchet fresh. |
   | **escalate** | Halts the pipeline until a senior reviewer intervenes. |

**Invariant:** No function, handler, agent, or patch may ever increase a claim level
after `screen_post_design()` has been called. Any code path that would do so is a bug.

### 2.3 Edge Type Pre-empts Design Claims

Structural edge types have fixed claim levels that override the design mapping:

| Edge Type | Forced Claim | Rationale |
|-----------|-------------|-----------|
| `reaction_function` | `BLOCKED_ID` (always) | Endogenous policy response |
| `identity` | `IDENTIFIED_CAUSAL` (always) | Mechanical/deterministic |
| `bridge` | `IDENTIFIED_CAUSAL` (always) | Accounting identity |
| `immutable` | `IDENTIFIED_CAUSAL` (always) | Validated prior evidence |

### 2.4 Counterfactual Eligibility Follows Claim Level

When `cap_to()` downgrades to `DESCRIPTIVE` or `BLOCKED_ID`, all counterfactual
permissions are cleared:
- `shock_scenario_allowed = False`
- `policy_intervention_allowed = False`

Policy counterfactuals always require `IDENTIFIED_CAUSAL`. Shock counterfactuals
require at least `REDUCED_FORM`. These thresholds are set in the query mode spec
and are not negotiable.

---

## 3. Three Screening Points (Strict Ordering)

Identification screening happens at exactly three lifecycle points, in this order:

```
screen_pre_design()  →  screen_post_design()  →  screen_post_estimation()
```

1. **Pre-design:** Can this edge ever be identified? (DAG structure check)
2. **Post-design:** Does the chosen design achieve identification? (Design → claim map)
3. **Post-estimation:** Given diagnostics, what is the final claim? (Downgrade only)

No screening function may be called out of sequence. Post-estimation screening
cannot override post-design results upward. Each stage narrows the claim; none widens it.

### 3.1 Five Risk Assessment Dimensions (Fixed Set)

Every `IdentifiabilityResult` reports five risk dimensions:

```
unmeasured_confounding | simultaneity | weak_variation | measurement_error | selection
```

Each dimension is rated `low`, `medium`, or `high`. This set is fixed — adding, removing,
or renaming risk dimensions requires explicit justification. Credibility scoring and
issue detection depend on this closed set.

### 3.2 Testable vs. Untestable Assumptions

Each screening result separates:
- **`testable_threats_passed`**: Diagnostics that verified assumptions
- **`testable_threats_failed`**: Diagnostics that violated assumptions
- **`untestable_assumptions`**: Assumptions that cannot be tested with available data

**Invariant:** Claims cannot be upgraded based on untestable assumptions. Only testable
assumption failures trigger downgrades. Untestable assumptions are documented for
researcher awareness — they are never silently dropped.

---

## 4. Seven Propagation Guardrails (Immutable Sequence)

The `PropagationEngine` applies exactly seven guardrails to every path, in this order:

| # | Guardrail | Blocks When | Severity |
|---|-----------|-------------|----------|
| 1 | **Mode gating** | Edge role not in mode's `propagation_requires` | Fatal |
| 2 | **Counterfactual gating** | `shock_scenario_allowed` or `policy_intervention_allowed` is False | Fatal |
| 3 | **TSGuard gating** | `counterfactual_blocked` or high-risk flag | Fatal |
| 4 | **IssueLedger gating** | Open `CRITICAL`-severity issue on edge | Fatal |
| 5 | **Reaction-function blocking** | `edge_type == "reaction_function"` | Fatal (always) |
| 6 | **Unit compatibility** | Outcome unit of edge _i_ ≠ treatment unit of edge _i+1_ | Fatal |
| 7 | **Frequency alignment** | Mixed frequencies without explicit `frequency_bridge` | Fatal |

**Invariants:**
- Blocking a single edge blocks the entire path (no partial propagation).
- Reaction-function edges are blocked unconditionally, regardless of mode, claim, or diagnostics.
- All guardrails default to blocking (conservative). Propagation only proceeds if ALL seven pass.
- The guardrail sequence must not be reordered, as mode gating is the outer gate.
- Unknown units block in STRUCTURAL/REDUCED_FORM modes; warn in DESCRIPTIVE.

---

## 5. Three Query Modes (Fixed Permissions)

| Mode | Propagation Roles Allowed | Policy CF | Shock CF |
|------|--------------------------|-----------|----------|
| `STRUCTURAL` | structural, bridge, identity | Requires `IDENTIFIED_CAUSAL` | Requires `IDENTIFIED_CAUSAL` |
| `REDUCED_FORM` | structural, reduced_form, bridge, identity | Never | Requires `REDUCED_FORM`+ |
| `DESCRIPTIVE` | all (including diagnostic_only) | Never | Never |

**Invariants:**
- Mode can only restrict, never expand. Mode is the outer gate on all queries.
- Policy counterfactuals are never allowed outside `STRUCTURAL` mode.
- Shock counterfactuals are never allowed in `DESCRIPTIVE` mode.
- The default mode is `REDUCED_FORM`.
- `STRUCTURAL` ⊂ `REDUCED_FORM` ⊂ `DESCRIPTIVE` (subset relation on permitted roles).

### 5.1 Role Derivation (Single Source of Truth)

`derive_propagation_role(edge_type, claim_level)` is the sole function that maps
edge metadata to propagation roles. The mapping is deterministic:

```
identity              → "identity"
mechanical / bridge   → "bridge"
reaction_function     → "diagnostic_only"
immutable             → "structural"
causal + IDENTIFIED   → "structural"
causal + REDUCED_FORM → "reduced_form"
causal + DESCRIPTIVE  → "diagnostic_only"
causal + BLOCKED_ID   → "diagnostic_only"
```

Roles are derived at runtime, never stored. This ensures they always reflect the
current claim level.

---

## 6. EXPLORATION vs. CONFIRMATION Mode

The default pipeline mode is `EXPLORATION`. The `--mode CONFIRMATION` flag must be
explicitly requested.

### 6.1 CONFIRMATION Mode Is Frozen

When `mode == "CONFIRMATION"`:
- **Zero iterations.** The pipeline runs exactly once and stops.
- **PatchBot is disabled.** No auto-fixes of any kind.
- **No re-estimation.** All edges processed once; no re-queueing.
- **No specification changes.** The DAG, designs, and data must match the frozen manifest.

**Invariant:** Any code that modifies specifications, applies patches, or re-queues
edges in CONFIRMATION mode is a bug. This is the system's pre-registration enforcement.

### 6.2 EXPLORATION Mode Constraints

Even in EXPLORATION mode, iteration is bounded:
- **Maximum 3 iterations** (configurable, but must have a finite bound).
- **Improvement threshold:** Iteration stops if credibility improvement < 5%.
- **Credibility threshold:** Iteration stops when all critical-path edges reach ≥ 0.60 credibility.
- **Convergence:** Stops when no re-queued tasks AND no auto-fixable issues remain.
- **HITL pause:** Iteration stops when any issue with `requires_human = True` is open.

### 6.3 Freeze Validation

Before transitioning from EXPLORATION to CONFIRMATION, the `FreezeValidator` checks:

1. **DAG hash** matches frozen manifest (no structural changes).
2. **Design assignments hash** matches (no design swaps).
3. **Data hash** matches (no new data added).
4. **Sample window overlap** is forbidden (exploration and confirmation windows must not
   overlap). "Window" is defined as the date range [start_date, end_date] of the estimation
   sample. Any observation-level overlap (shared dates in panel or time-series data)
   constitutes a violation, not just matching start/end dates.
5. **Frozen timestamp** is in the past (no future-dated freezes).

Any violation blocks the transition. These checks are non-negotiable.

---

## 7. Anti-P-Hacking Constraints

### 7.1 Refinement Whitelist

All specification refinements must pass through `RefinementWhitelist`. Refinements
not on the whitelist raise `GovernanceViolation` — a hard exception, not a warning.

Whitelisted refinements have explicit bounds:
- **Controls:** Only from a predefined list; max 2 additions per iteration.
- **Lags:** Confined to range [1, 6].
- **Instruments:** Must be from a predefined list.
- **Regime splits:** Only at predefined dates.
- **Bandwidth (RDD):** Confined to range [0.5, 2.0].

### 7.2 PatchBot Prohibitions

PatchBot has a hardcoded dispatch table of 14 allowed handlers. Actions not in this
table return `applied=False`. The following are explicitly and permanently prohibited:

- **Control shopping** (unbounded control variable search)
- **Lag searching** (systematic lag length optimization)
- **Sample trimming** (data-driven outlier or observation removal)
- **Instrument shopping** (IV selection based on results)
- **Regime splitting on test data** (data-driven structural break selection)
- **Any modification in CONFIRMATION mode**

The blacklist always wins over the whitelist: if an action is disallowed, no
whitelist entry can override it.

### 7.3 Null Acceptance

A precisely null effect is a valid finding, not a failure. The system must never treat
null results as errors that trigger specification searching. This prevents the
file-drawer problem at the pipeline level.

The null acceptance criterion requires **both** conditions:
1. `|β| < equivalence_bound` (default: 0.1 in natural units)
2. `CI_width < 2 × equivalence_bound` (confidence interval must be tight)

When both are met, the result is "precisely null" — a valid, publishable finding.
The equivalence bound is configurable but must always have a finite, positive value.

### 7.4 Iteration Counter

The iteration counter only increments; it never resets within a run. This is part
of the audit trail and prevents manipulation of the iteration budget.

---

## 8. Hash-Chained Audit Log

### 8.1 What Gets Hashed

Every audit entry records four hashes:
- `dag`: Current DAG structure hash
- `data`: Combined dataset hash
- `spec`: Current specification hash
- `result`: Estimation result hash (when applicable)

### 8.2 Chain Integrity

Each JSONL entry includes the SHA-256 hash of the previous entry. The first entry
has `prev_hash = null`. Modifying any earlier entry invalidates all subsequent hashes.

**Invariants:**
- The audit log is append-only. Entries are never modified or deleted.
- LLM-assisted repairs additionally log: model identifier, prompt SHA-256 hash.
- All HITL decisions are appended as hash-chained entries with: issue ID, action,
  justification, analyst identifier, timestamp.

---

## 9. TSGuard Diagnostics

### 9.1 Seven Required Checks

| Diagnostic | Failure Consequence |
|-----------|-------------------|
| Leads test | `BLOCKED_ID` + block counterfactuals |
| Residual autocorrelation | Set `autocorr_risk = high` |
| HAC sensitivity (NW lags 1,4,8) | Set `autocorr_risk = high` on sign flip |
| Lag sensitivity (L=1,2,4) | Cap rating to B |
| Regime stability (split-sample) | Cap to `REDUCED_FORM` + block CF |
| Placebo time shift | Set `timing_risk = high` |
| Shock support (episodes > 1σ) | Block CF if < 3 episodes |

### 9.2 Governance Rules

1. Both variables in levels + nonstationarity HIGH → cap at `DESCRIPTIVE` (unless ECM).
2. Lead test failure → `BLOCKED_ID` + block all counterfactuals.
3. Regime instability → cap at `REDUCED_FORM`, restrict CF scope.
4. Shock support < 3 → block propagation.

TSGuard results feed into `screen_post_estimation()` via `cap_to()`. They can only
downgrade, never upgrade.

---

## 10. Agent Loop Phase Ordering

The agentic pipeline executes in strict phase order:

```
Phase 0a: Auto-ingest raw data
Phase 0b: DAG validation + LLM auto-repair (max 3 attempts)
Phase 1:  DataScout (data availability, download)
Phase 1.5: PaperScout (literature search)
─── ITERATION LOOP (max_iterations) ───
  Phase 2: Estimate/re-estimate ready tasks
  Phase 3: Post-run issue detection (30 rules)
  Phase 4: Gate evaluation (issue gates)
  Phase 5: HITL check → stop if human required
  Phase 6: PatchBot auto-fixes → re-queue affected edges
  Phase 7: Convergence check → stop if converged
─── END LOOP ───
Post-loop: Final HITL gate evaluation + checklist
Output:    System report, EdgeCards, panels
```

### 10.1 Agent-to-Phase Binding

| Agent | Pipeline Phase | Contract |
|-------|---------------|----------|
| **DataScout** | Phase 1 | Catalog data availability; download within budget; 2-pass cycle (catalog-first, then download) |
| **PaperScout** | Phase 1.5 | Literature search; informational only; does not modify DAG or claims |
| **ModelSmith** | Pre-Phase 2 (design selection) | Select designs from registry; write ModelSpecs with identification assumptions; must complete **before** estimation |
| **Estimator** | Phase 2 | Execute ModelSpecs; run diagnostics; produce EdgeCards |
| **Judge** | Phase 3–4 | Score credibility; flag weak links; propose refinements within RefinementWhitelist bounds |
| **PatchBot** | Phase 6 | Apply auto-fixes within PatchPolicy; re-queue affected edges |

**Invariants:**
- Phases must not be reordered.
- Design selection (ModelSmith) must complete before estimation (Phase 2). Selecting
  designs after seeing results would violate the one-way ratchet.
- Data must be available before estimation (Phase 1 before Phase 2).
- Issue detection must follow estimation (Phase 3 after Phase 2).
- HITL check must precede PatchBot (Phase 5 before Phase 6).
- Re-estimation only occurs via explicit re-queue from PatchBot (Phase 6 → Phase 2).

---

## 11. Adapter Interface Contract

All estimation flows through a single interface:

```python
class EstimatorAdapter(ABC):
    def estimate(self, request: EstimationRequest) -> EstimationResult: ...
    def supported_designs(self) -> list[str]: ...
```

### 11.1 EstimationResult Required Fields

Every adapter must populate: `point`, `se`, `ci_lower`, `ci_upper`, `pvalue`,
`n_obs`, `method_name`, `library`, `library_version`, `diagnostics`.

CI is always 95%, derived from `point ± 1.96 * se`.

### 11.2 Registry Lookup Order

1. Config-driven: `design_registry.yaml` → adapter class path.
2. Built-in fallback: `_BUILTIN_ADAPTERS` hardcoded mapping.

Adapters are cached after first instantiation. Same design ID always returns the
same adapter instance.

---

## 12. EdgeCard Schema Invariants

### 12.1 Credibility Scoring (Independent of Significance)

```
score >= 0.80 → "A"
score >= 0.60 → "B"
score >= 0.40 → "C"
score <  0.40 → "D"
```

Scoring inputs: 40% diagnostic pass rate + 10% design strength + 30% stability +
20% data coverage. **Statistical significance is never a factor in credibility
scoring.** There is no `min_tstat` criterion.

### 12.2 Counterfactual Block (Split Design)

Counterfactuals are split into two independent permissions:
- `shock_scenario_allowed`: Can this edge participate in "what if X shocks?" queries?
- `policy_intervention_allowed`: Can this edge support policy recommendations?

Policy is always stricter than shock. Both must be explicitly allowed; there is no
implicit permission.

### 12.3 Immutable Evidence

If `EdgeSpec.validated_evidence.immutable = True`:
- The edge cannot be re-estimated by the agentic loop.
- The `result_hash` (SHA-256) must be present for audit integrity.
- The agent loop marks the edge as `IMMUTABLE` and skips it.

---

## 13. DAG Specification Invariants

### 13.1 Pre-Estimation Validation (Must Pass Before Any Estimation)

1. **Acyclicity:** DFS-based, including temporal expansion for contemporaneous edges.
2. **Unit presence:** Every edge must have treatment and outcome units.
3. **Edge type labeling:** Every edge must declare its type from the closed set:
   `{causal, reaction_function, bridge, identity}`. Unknown edge types are rejected.
4. **Node-source bindings:** Observed nodes must have data source specifications.
5. **Endpoint existence:** All edge endpoints must exist as nodes.

### 13.2 Forbidden Controls

Forbidden controls = descendants of treatment node + explicitly forbidden list.
Descendant computation is transitive (BFS). This prevents posttreatment bias.
The forbidden set is computed once during validation and is read-only thereafter.

### 13.3 Identity Dependencies

For identity nodes with formula specifications, `depends_on` is auto-derived from
the formula via regex. Built-in mathematical functions (`log`, `exp`, `sqrt`, `abs`,
`diff`, `lag`, etc.) are excluded from dependency extraction. The set of built-in
functions must not change silently.

---

## 14. Discovery and NL-to-DAG Outputs

### 14.1 Discovery Outputs Are Proposals Only

Causal discovery results (PC, GES, FCI, NOTEARS) and LLM-extracted edges are
**never auto-merged** into the DAG. All discovery outputs must pass through HITL
review as `ProposedEdge` objects requiring explicit analyst approval.

### 14.2 Discovery Hyperparameters in Audit Trail

Discovery algorithm runs must log their hyperparameters (significance level α for PC,
score function for GES, independence test for FCI, λ for NOTEARS) in the audit trail.
Without this, an analyst could re-run discovery with different parameters until the
desired graph emerges — a form of p-hacking at the graph structure level. The anti-
p-hacking constraints (§7) extend to discovery, not just estimation.

### 14.3 LLM Confidence Is Not Identification

LLM extraction confidence reflects the strength of the textual claim in the source
material, not the validity of the underlying research design. A high-confidence
LLM extraction paired with an OLS design still receives at most a `DESCRIPTIVE`
ceiling. The system must never treat LLM confidence as a proxy for identification
strength.

### 14.4 Narratives Must Precede Data

The narrative-to-DAG pipeline is valid only when the input narrative encodes prior
domain knowledge — the researcher's causal model before seeing estimation results.
If the narrative is written after seeing results, the DAG structure becomes
data-dependent and the one-way ratchet provides no protection.

---

## 15. SE Propagation Assumptions

The delta method formula for propagated standard errors assumes independence across
edges:

```
Var(τ̂) = Σᵢ (∏ⱼ≠ᵢ βⱼ)² · SEᵢ²
```

**Invariants:**
- The SE method is always labeled `"delta_independence_naive"`.
- Bridge and identity edges have SE = 0 (non-estimated) and are skipped.
- When > 1 estimated edge exists in a path, a warning about the independence
  assumption is emitted.
- Multi-path results are shown separately by default; never auto-summed without
  edge-disjointness verification. When paths share edges, the system **must refuse
  to sum** and present paths individually with a warning about double-counting.
  The aggregation policy `"sum_disjoint"` only activates when no edges overlap.

---

## 16. Issue Severity → Pipeline Behavior

| Severity | Blocks CONFIRMATION? | Can Trigger HITL Pause? | PatchBot Eligible? |
|----------|---------------------|------------------------|-------------------|
| `CRITICAL` | **Yes, always** | If `requires_human` | If `auto_fixable` |
| `HIGH` | No | If `requires_human` | If `auto_fixable` |
| `MEDIUM` | No | If `requires_human` | If `auto_fixable` |
| `LOW` | No | No | If `auto_fixable` |

**Invariant:** No open `CRITICAL` issue may exist when entering CONFIRMATION mode.
The `block_confirmation` gate is non-negotiable.

### 16.1 Critical Issue Rules (Must Always Be Present)

The issue registry (currently 30 rules) is the governance layer's detection capability.
The following seven `CRITICAL`-severity rules are architecturally essential — removing
any of them silently degrades the system's correctness guarantees:

| Rule ID | What It Guards |
|---------|---------------|
| `FREQUENCY_ALIGNMENT_ERROR` | Blocks estimation on mixed frequencies without explicit aggregation |
| `REACTION_FUNCTION_EDGE` | Prevents policy reaction edges from entering shock propagation |
| `TIME_FE_ABSORBS_SHOCK` | Detects when time FE absorbs the treatment; requires exposure interaction |
| `EXPOSURE_NOT_PREDETERMINED` | Ensures exposure variable is measured pre-treatment |
| `SIGNIFICANT_BUT_NOT_IDENTIFIED` | Prevents overclaiming: significance without valid identification |
| `LEADS_SIGNIFICANT_TIMING_FAIL` | Catches reversed temporal ordering in causal claims |
| `UNIT_MISSING_IN_EDGECARD` | Blocks uninterpretable coefficients lacking unit specs |

The total rule count may grow but must never shrink below these seven critical rules.
Adding rules is allowed; removing or disabling any rule listed above is a bug.

**Note:** The anti-p-hacking concerns (null dropping, control shopping, specification
drift) are enforced through dedicated mechanisms — `StoppingCriteria.check_null_acceptance`
(§7.3), `RefinementWhitelist` (§7.1), and the hash-chained audit log (§8) — rather than
through issue registry rules.

### 16.2 Auto-Fixability and HITL Requirement Are Orthogonal

Each rule declares `auto_fixable` and `requires_human` independently. An issue can be
both auto-fixable AND require human review — PatchBot applies the fix, but the issue
still triggers HITL pause. No implicit assumption that "fixable" means "doesn't need
review."

---

## 17. Issue Lifecycle Invariants

### 17.1 Issue Identity

Each issue has a unique key: `"{rule_id}:{edge_id or node_id or 'global'}"`. This key
is used for deduplication across runs. Only one open issue per key is permitted.

### 17.2 Status Transitions

```
OPEN  →  CLOSED (with reason + closer identity)
OPEN  →  WONT_FIX
```

Issues start as `OPEN`. There is no reopen path — if the same problem resurfaces, a new
issue is created. This keeps the ledger append-only and the audit trail linear.

### 17.3 Trigger Timing

| Trigger | When |
|---------|------|
| `pre_run` | Before estimation (DAG-level structural issues) |
| `post_run` | After each edge estimation completes (EdgeCard-level) |
| `cross_run` | After full run, comparing against previous runs via `state.json` |

---

## 18. PatchPolicy LLM Repair Levels

### 18.1 Two Repair Categories

| Category | Actions | Approval |
|----------|---------|----------|
| **Metadata repairs** | `fix_edge_id_syntax`, `fix_missing_source_spec` | Always allowed, audit-logged |
| **DAG repairs** | `fix_dag_identity_deps`, `fix_dag_missing_reaction`, `fix_orphan_identity_edge` | Auto-approved in EXPLORATION; require HITL in CONFIRMATION |

### 18.2 Repair Scope Constraint

LLM repairs target **schema validity** only. The following are permanently out of scope
for automated repair:
- Identification strategies or design selection
- Expected signs
- Causal edge direction
- Control set composition (beyond syntax fixes)

**Invariant:** `is_llm_repair_allowed(action, mode)` returns True only when the action
is a known metadata repair, OR is a DAG repair in EXPLORATION mode (or explicitly
HITL-approved in CONFIRMATION mode). Unknown actions always return False.

---

## 19. Task Queue State Machine

### 19.1 Task Status Transitions

```
NEW → WAITING_DATA | WAITING_ARTIFACT
WAITING_DATA → READY (when data becomes available)
WAITING_ARTIFACT → READY (when upstream task completes)
READY → RUNNING → DONE | DONE_SUGGESTIVE | BLOCKED_ID | FAILED
```

There is no backward transition from terminal states (`DONE`, `BLOCKED_ID`, `FAILED`).
A failed task is not retried unless explicitly re-queued by PatchBot (which creates a
new task, preserving the original's audit trail).

### 19.2 Task Priority Ordering

| Priority | Level | Use |
|----------|-------|-----|
| 1 | CRITICAL | On critical path to analysis target |
| 2 | HIGH | Provides instrument or required artifact |
| 3 | MEDIUM | Reusable intermediate artifact |
| 4 | LOW | Supporting/supplementary |
| 5 | BACKGROUND | Opportunistic (e.g., "download everything") |

Scheduling processes tasks in priority order. Within the same priority, FIFO ordering
applies. Priority is set at task creation and does not change.

---

## 20. Sentinel Loop Invariants

### 20.1 Always-On Background Monitor

The sentinel loop runs continuously alongside estimation. It is **not** a batch job
triggered at the end — it polls on a fixed interval (default: 5 minutes) and catches
regressions between any two pipeline steps.

### 20.2 Validation → Heal → Build → Open

The sentinel's cycle is strictly ordered:

```
1. Re-run full DAG validation suite (40 rules)
2. Auto-heal fixable issues (PatchBot under PatchPolicy)
3. If estimation complete: build DAG visualization + HITL panel HTML
4. Auto-open panels in default browser
```

**Invariants:**
- Healing is governed by `PatchPolicy` — the sentinel cannot bypass patch constraints.
- All sentinel repairs are logged in the audit trail with `source: "sentinel"`.
- Panel build only triggers after estimation completes (never mid-run with partial results).
- If a sentinel instance is already running (PID detected), the pipeline skips re-launch.
  There is never more than one active sentinel per workspace.

### 20.3 Sentinel Is Non-Destructive

The sentinel may fix schema issues (missing unit specs, malformed edge IDs, orphan node
references). It must **never**:
- Modify estimation results
- Change identification strategies or designs
- Alter claim levels
- Remove or suppress issues from the ledger

---

## 21. Data Pipeline Invariants

### 21.1 Auto-Ingest Convention

Any file dropped in `data/raw/` is automatically profiled, standardized to Parquet, and
registered as a node loader before estimation begins. The convention is:

```
data/raw/*.csv   →  auto_ingest()  →  data/processed/*.parquet  →  node_loader registered
data/raw/*.parquet                →  (copy + profile)           →  node_loader registered
```

**Invariants:**
- Raw files are never modified in-place. Standardization produces new files in `data/processed/`.
- Every ingested dataset gets a profile (row count, column types, date range, frequency detection).
- Node loader registration is idempotent — re-ingesting the same file updates metadata but does not duplicate.

### 21.2 DataScout Actionable Guidance

When DataScout cannot auto-fetch data for a DAG node, it must log actionable guidance:
expected file format, required columns, and the exact CLI command (`opencausality data ingest`)
the user should run after dropping files. Silent failure is a bug.

### 21.3 Data Must Precede Estimation

The agent loop phase ordering (§10) requires Phase 0a (auto-ingest) and Phase 1 (DataScout)
to complete before Phase 2 (estimation). Estimation must never run against unregistered or
unprofiled data sources.

---

## 22. LLM Abstraction Layer Invariants

### 22.1 Provider Agnosticism

All LLM calls flow through the `LLMClient` ABC. The system must never contain direct API
calls to any specific LLM provider outside the client implementations. Switching providers
requires changing one environment variable (`LLM_PROVIDER`), not code changes.

### 22.2 Four Provider Implementations

| Provider | Backend | Key Required |
|----------|---------|-------------|
| `anthropic` | Direct Anthropic API | Yes (`ANTHROPIC_API_KEY`) |
| `litellm` | LiteLLM (OpenAI, Cohere, etc.) | Yes (provider-specific) |
| `codex` | Shells out to `codex exec` | No |
| `claude_cli` | Shells out to `claude -p` | No |

### 22.3 Auto-Fallback Hierarchy

If the configured provider fails to initialize (e.g., `anthropic` selected but no API key):

```
anthropic (no key) → auto-fallback to codex CLI → warning emitted
```

**Invariants:**
- Fallback always emits a visible warning. Silent fallback is a bug.
- The fallback target is the `codex` CLI provider (no-key-required).
- `get_llm_client()` is the single factory function. No code should instantiate clients directly.

---

## 23. Query REPL Invariants

### 23.1 Hedged Language Enforcement

The Query REPL must **never** say "causes", "causal effect", or other causal language
unless the entire path is `IDENTIFIED_CAUSAL`. For mixed or non-identified paths:

| Path Status | Language |
|-------------|----------|
| All edges `IDENTIFIED_CAUSAL` | "causes", "causal effect" permitted |
| Any edge `REDUCED_FORM` | "is associated with", "predicts" |
| Any edge `DESCRIPTIVE` or `BLOCKED_ID` | "co-moves with", "correlates with" |

This is a hard rule, not a suggestion. The LLM prompt and regex fallback both enforce it.

**Forbidden causal language for non-identified paths:** The following terms (and their
inflections) must **never** appear in REPL output for paths below `IDENTIFIED_CAUSAL`:
`causes`, `causal effect`, `drives`, `leads to`, `results in`, `produces`, `generates`,
`brings about`. These carry causal connotations and must be reserved exclusively for
fully identified paths. This is a closed forbidden set, not an open suggestion.

### 23.2 Mode-Aware Query Gating

The REPL respects the active query mode (§5). If the user asks a policy counterfactual
question in `REDUCED_FORM` mode, the REPL must refuse with an explanation, not silently
downgrade the query.

### 23.3 Dual Parse Path

```
User query → LLM parser → structured intent
                ↓ (on LLM failure)
             Regex fallback → structured intent
```

Both paths produce the same structured intent format. Results must be identical regardless
of which parser succeeded. The regex fallback exists for resilience, not as a different
code path.

---

## 24. Interactive Panel Contracts

### 24.1 "DRAFT PROPOSAL" Framing

The DAG visualization panel must display a prominent "DRAFT PROPOSAL" banner. All
estimation outputs are framed as requiring analyst sign-off. This framing is non-optional
and must not be removed or softened (e.g., to "Results" or "Report").

### 24.2 Self-Contained HTML

Both panels (DAG visualization and HITL resolution) are single-file HTML documents with
no external dependencies except D3.js CDN for visualization. They must remain openable
by double-clicking the file — no build step, no server, no additional assets required.

### 24.3 JSON Export Format

Both panels export decisions as JSON with:
- `issue_id`: Links to the issue ledger
- `action`: One of accept / reject / revise / escalate
- `justification`: Free-text analyst reasoning
- `source`: Either `"dag_visualization"` or `"hitl_panel"` (for audit trail distinction)
- `timestamp`: ISO 8601

Exported JSON is ingestible by the pipeline for automated re-processing.

### 24.4 Decision Actions Are Downgrade-Compatible

The panel action set (accept, reject, revise, escalate) must never include an "upgrade"
or "approve as causal" action. This mirrors the one-way ratchet (§2.2) at the UI level.

---

## 25. DoWhy Refutation Engine

### 25.1 Four Robustness Checks (Mandatory and Complete)

After DoWhy adapter estimation, the refutation engine runs **all four** tests:

| Refutation | What It Tests |
|-----------|--------------|
| Random Common Cause | Adding a random confounder should not change the estimate |
| Placebo Treatment | Shuffling the treatment should destroy the effect |
| Data Subset | Estimate should be stable across random subsets |
| Unobserved Common Cause | Estimate should survive simulated unobserved confounding |

**Invariant:** All four refutations must run for every DoWhy estimation. No DoWhy result
is valid without the complete battery. Skipping any refutation (e.g., for performance)
is a bug — it removes the stress test that validates the `IDENTIFIED_CAUSAL` ceiling.

### 25.2 Refutation Failure → Downgrade

Refutation failures feed into `screen_post_estimation()` via `cap_to()`. A failed
placebo test or a sensitive-to-confounding result caps the claim downward. Refutation
success does not upgrade claims — it only provides additional confidence that the current
level is warranted.

---

## 26. Graph Format Interoperability

### 26.1 Canonical Internal Format

Internally, DAGs are represented as `networkx.DiGraph` objects with node/edge attribute
dictionaries following the YAML schema. This is the single canonical representation.

### 26.2 Conversion Layer

| Target Format | Converter | Use Case |
|--------------|-----------|----------|
| NetworkX DiGraph | Native | Internal computation |
| DoWhy GML | `to_dowhy_gml()` | DoWhy estimation adapters |
| pywhy-graphs ADMG | `to_admg()` | PyWhy ecosystem interop |

**Invariants:**
- Round-trip conversion (export → import) must preserve all structural information.
- Attribute loss during conversion must be logged as a warning.
- The internal NetworkX format is the source of truth; external formats are projections.

---

## 27. Serialization Conventions

### 27.1 Dataclasses Over Pydantic

All domain objects use Python `dataclasses`. Pydantic is not a dependency. Serialization
is explicit via dedicated reader/writer functions, not implicit magic methods.

**Rationale:** Minimal dependency footprint; explicit serialization boundaries.

### 27.2 Format Assignments (Fixed)

| Artifact | Format | Rationale |
|----------|--------|-----------|
| DAG specifications | YAML | Human-authored, version-controlled |
| EdgeCards | YAML | Dual human-readable + machine-parseable |
| Audit logs | JSONL | Append-only, hash-chained |
| Issue ledger state | JSON | Snapshot of current state |
| Panel outputs | HTML | Self-contained, no-server viewing |
| Benchmark results | JSON/Markdown | Machine + human readable |

These format assignments are fixed. Changing the format of any artifact type requires
explicit justification in the audit trail.

### 27.3 No Silent Schema Migration

If the schema of any serialized artifact changes, the system must either:
1. Support reading the old format with a documented migration path, or
2. Increment the DAG/artifact schema version and reject old formats with a clear error.

Silent schema drift (old files parse incorrectly without error) is a bug.

---

## 28. Dependency Philosophy

### 28.1 Lazy Imports for ML Backends

Heavy ML dependencies (DoWhy, EconML, CausalML, DoubleML, causal-learn, gCastle) are
imported lazily — only when the corresponding adapter is actually invoked. A user who
only runs Local Projections should never need to install TensorFlow or PyTorch.

### 28.2 GPL Isolation

GPL-licensed tools (Tetrad, Tigramite, DAGitty) are supported as **optional external
executors** — they are never bundled as dependencies, never imported, and never linked.
Interaction is via subprocess calls or file-based interchange.

### 28.3 Core Dependencies Are Minimal

The core installation (`pip install -e .`) requires only:
- Standard scientific Python (numpy, pandas, scipy, statsmodels)
- CLI framework (typer, rich)
- YAML/JSON processing (pyyaml)
- Networking (requests, for PaperScout/DataScout)

Everything else is optional extras.

---

## 29. Output Artifact Structure

### 29.1 Fixed Output Directory Layout

```
outputs/agentic/
├── cards/edge_cards/      # One YAML file per edge
├── issues/state.json      # Current issue ledger snapshot
├── ledger/                # Hash-chained JSONL audit log files
├── citations/             # Literature search results (per-edge)
├── dag_visualization.html # Interactive DAG panel
├── hitl_panel.html        # HITL resolution panel
├── hitl_checklist.md      # Markdown summary checklist
└── .notification.json     # Monitor sentinel file (triggers alerts)
```

**Invariants:**
- EdgeCards are the **primary** output artifact — all other outputs derive from or reference them.
- The `.notification.json` sentinel file is the inter-process signal for `opencausality monitor`.
  It must only be written when HITL attention is genuinely required.
- All output paths are relative to the configured output directory. No output may be written
  outside this directory tree.

---

## 30. Benchmark Framework Contracts

### 30.1 Ground Truth Requirement

Every benchmark must have a known true treatment effect. The framework computes RMSE and
CI coverage against this ground truth. Benchmarks without known effects are not benchmarks
— they are case studies.

### 30.2 DGP Contract

```python
class DGPBenchmark:
    def generate(self, seed: int) -> Tuple[pd.DataFrame, float]:
        """Returns (data, true_effect)."""
```

DGP benchmarks are deterministic given a seed. Results must be reproducible across machines
with the same numpy/scipy versions.

### 30.3 Multi-Seed Evaluation

Benchmarks run across multiple seeds (default: 10). Single-seed results are insufficient
for adapter validation. Reports include mean RMSE and CI coverage across all seeds.

### 30.4 Minimum CI Coverage Threshold

An adapter benchmark is considered **passing** when 90% CI coverage ≥ 80% across seeds.
Below this threshold, the adapter's SE computation or model specification is suspect.
Benchmark failures do not automatically block adapter registration but must be documented
as a known limitation in the adapter's output.

---

## 31. Domain Agnosticism

### 31.1 The Platform vs. The Case Study

OpenCausality is domain-agnostic. The Kazakhstan bank stress testing example is a **case
study**, not a platform constraint. All domain-specific components are isolated to:

- `shared/data/` — Data clients (e.g., `kz_bank_panel`)
- `config/agentic/dags/` — Domain-specific DAG specifications
- `examples/` — Narrative inputs and tutorial data

The estimation adapters, issue detection engine, governance layer, propagation engine,
query REPL, and sentinel loop work with **any** domain where causal relationships can be
expressed as a directed graph.

### 31.2 Extension Point: Data Clients

Adding a new domain requires:
1. A data client in `shared/data/` (or files in `data/raw/` for auto-ingest)
2. A DAG YAML specification
3. (Optional) A variable catalog for NL-to-DAG matching

No estimation, governance, or identification code should contain domain-specific logic.
If domain assumptions leak into the core, that is a bug.

---

## 32. Agent Contracts (PaperScout, DataScout, DAG Auto-Repair)

### 32.1 PaperScout

PaperScout searches four APIs (Semantic Scholar, OpenAlex, CORE, arXiv) for literature
supporting or challenging each edge.

**Invariants:**
- Citations are deduplicated across APIs before attachment to EdgeCards.
- Each citation includes a relevance score; low-relevance citations are filtered.
- When local PDFs are available, full-text extraction (pymupdf → pypdf → plain text
  3-tier fallback) feeds richer causal claims to the LLM.
- PaperScout results are **informational** — they do not modify claim levels, designs,
  or identification status.

### 32.2 DataScout

DataScout catalogs data availability for each DAG node and attempts automated download
from configured connectors.

**Invariants:**
- Downloads are budget-bounded (configurable max per run).
- When auto-download fails, actionable guidance is logged (§21.2).
- DataScout never modifies the DAG structure. It populates data availability metadata only.

### 32.3 DAG Auto-Repair

LLM-assisted validation detects and fixes DAG schema errors (invalid edge IDs, missing
node specs, dependency mismatches).

**Invariants:**
- Maximum 3 repair attempts per validation cycle. After 3 failures, fall back to human review.
- All repairs are governed by `PatchPolicy` — the LLM cannot bypass patch constraints.
- Repairs are logged in the audit trail with: model identifier, prompt SHA-256 hash,
  before/after diff.
- Auto-repair targets **schema** errors only (structural validity). It must never modify
  identification strategies, designs, expected signs, or causal semantics.

---

## 33. Unit-Aware Propagation

### 33.1 Eight Named Unit Types Plus Unknown Fallback

```
pp, pct, log_point, bn_kzt, ratio, index, sd, bps   (8 named types)
unknown                                                (fallback for unparseable units)
```

The `UnitSpec` regex parser matches free-text unit specifications against 8 named
patterns. Anything unparseable falls back to `unknown`. The `UNIT_KINDS` type is the
closed set of valid values.

### 33.2 Dimensional Analysis on Multi-Edge Paths

When propagating effects across a path, the PropagationEngine performs dimensional analysis:
the outcome unit of edge _i_ must equal the treatment unit of edge _i+1_. Mismatches
block propagation in `STRUCTURAL` and `REDUCED_FORM` modes; they emit warnings in
`DESCRIPTIVE` mode (see §4, guardrail 6).

### 33.3 UnitSpec Regex Rules

- The `pp` pattern uses `pp\b` (suffix boundary only). A `\b` prefix (`\bpp`) must
  **never** be added — it would fail on "1pp" since "1" is a word character.
- The `%` literal in the `pct` pattern has no `\b` wrapper — `%` is not a word character
  so `\b` would fail to match it.
- Unknown or unparseable units fall back to `"unknown"` and block propagation in
  `STRUCTURAL` and `REDUCED_FORM` modes (conservative default).

---

## 34. Cross-Cutting Principles

### 34.1 Conservative by Default

Every gate, permission, and allowance defaults to the restrictive option:
- Unknown edge types → `BLOCKED_ID`
- Missing units → block propagation
- Missing counterfactual flags → CF not allowed
- Unknown PatchBot actions → `applied = False`

### 34.2 Explicit Over Implicit

- No implicit permissions. Every allowance must be explicitly declared.
- No silent upgrades. Every claim level change must be traceable.
- No hidden state. All decisions flow through the audit log.

### 34.3 Separation of Concerns

- **Claim level** (what can be asserted) is separate from **credibility score** (how reliable the estimate is).
- **Propagation role** (can the edge participate?) is separate from **counterfactual eligibility** (can we do "what if?").
- **Estimation** (fitting models) is separate from **identification** (whether the design supports causal claims).
- **Issue detection** (finding problems) is separate from **issue resolution** (deciding what to do).

### 34.4 Domain Agnosticism (see §31)

OpenCausality is a platform, not a Kazakhstan-specific tool. Domain logic must be
confined to data clients, DAG specs, and example directories. Core modules must never
contain hardcoded domain assumptions.

### 34.5 Auditability as First-Class Output

The audit trail is not a byproduct — it is a primary output artifact. Every
specification change, estimation decision, diagnostic result, issue flag, patch
application, HITL decision, and LLM repair is logged with cryptographic integrity.
The entire history can be committed to version control and independently verified.

---

## Summary: The Invariant Checklist

Before merging any change, verify:

**Identification & Claims (§§1-5)**
- [ ] Claim levels never move upward after design selection
- [ ] `cap_to()` remains monotone (downgrade only)
- [ ] DESIGN_CLAIM_MAP covers all registered adapters
- [ ] Five risk assessment dimensions unchanged (§3.1)
- [ ] Testable/untestable assumption separation preserved (§3.2)
- [ ] Reaction functions remain unconditionally blocked from propagation
- [ ] All seven propagation guardrails are applied in order
- [ ] Mode can only restrict, never expand permissions

**Governance & Anti-P-Hacking (§§6-9)**
- [ ] CONFIRMATION mode remains frozen (zero iterations, no patches)
- [ ] Sample window non-overlap check covers observation-level overlap (§6.3)
- [ ] Default pipeline mode is EXPLORATION
- [ ] PatchBot prohibited actions list is unchanged
- [ ] Refinement whitelist bounds are respected
- [ ] Hash chain append-only property is preserved
- [ ] Null acceptance uses both |β| and CI_width criteria (§7.3)

**Pipeline & Agents (§§10-19)**
- [ ] Agent loop phase ordering is maintained; design selection before estimation (§10.1)
- [ ] Adapter interface contract (EstimationRequest → EstimationResult) is preserved
- [ ] Credibility scoring excludes statistical significance
- [ ] No open CRITICAL issues in CONFIRMATION mode
- [ ] Critical issue rules (§16.1) all present in registry — none removed
- [ ] Auto-fixable ≠ doesn't-need-review (§16.2 orthogonality)
- [ ] Discovery/NL outputs remain proposals requiring HITL approval
- [ ] Discovery hyperparameters logged in audit trail (§14.2)
- [ ] Issue lifecycle is append-only (no reopen, only new issue)
- [ ] LLM repairs limited to schema validity — never causal semantics
- [ ] Task queue transitions are forward-only (no backward from terminal states)
- [ ] Edge types restricted to closed set: {causal, reaction_function, bridge, identity}

**Sentinel & Data (§§20-21)**
- [ ] Sentinel repairs are logged with `source: "sentinel"` and governed by PatchPolicy
- [ ] Sentinel never modifies estimation results, designs, or claim levels
- [ ] Raw files in `data/raw/` are never modified in-place
- [ ] Data availability precedes estimation (Phase 0a/1 before Phase 2)

**LLM & Query (§§22-23)**
- [ ] All LLM calls go through `LLMClient` ABC — no direct provider API calls in core
- [ ] Provider fallback emits a visible warning, never silent
- [ ] Forbidden causal language list enforced for non-identified paths (§23.1)
- [ ] Query mode gating cannot be bypassed by the LLM parser

**Panels & Output (§§24, 29)**
- [ ] "DRAFT PROPOSAL" banner is present on DAG visualization
- [ ] HITL actions match fixed semantics (accept/reject/revise/escalate — see §2.2)
- [ ] Panel actions never include an "upgrade" or "approve as causal" option
- [ ] Both panels remain self-contained single-file HTML
- [ ] EdgeCards remain the primary output artifact; all others derive from them

**Refutation & Propagation (§§15, 25)**
- [ ] All four DoWhy refutations run for every DoWhy estimation — no subset (§25.1)
- [ ] Multi-path aggregation refused when paths share edges (§15)

**Serialization & Dependencies (§§27-28)**
- [ ] No new heavy dependencies without explicit justification
- [ ] ML backends remain lazily imported
- [ ] GPL tools remain external executors (never bundled)
- [ ] Serialization format assignments unchanged (YAML DAGs, JSONL logs, etc.)

**Domain & Scope (§§31-33)**
- [ ] No domain-specific logic in core modules (estimation, governance, propagation)
- [ ] Unit type set changes are deliberate and documented
- [ ] DAG auto-repair targets schema only, never causal semantics
