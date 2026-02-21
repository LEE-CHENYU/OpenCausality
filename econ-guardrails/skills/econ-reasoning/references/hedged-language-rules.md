# Hedged Language Rules — Complete Reference

Source: INVARIANTS.md §23.1 (Hedged Language Enforcement)

## Core Principle

The query system must **never** say "causes", "causal effect", or other causal language unless the entire path is `IDENTIFIED_CAUSAL`. This is a hard rule, not a suggestion.

## Rule 1: No Causal Language Without Full Identification

**Rule**: Never use causal verbs unless ALL edges on ALL open paths have `claim_level = IDENTIFIED_CAUSAL`.

**Forbidden causal language** (closed set — these and their inflections are banned for non-identified paths):
- causes
- causal effect
- drives
- leads to
- results in
- produces
- generates
- brings about

### Examples

**FAIL**: "A 10% oil price shock *causes* a 3% depreciation of the exchange rate."
- Why: Unless every edge from oil price to exchange rate is IDENTIFIED_CAUSAL, "causes" is forbidden.

**PASS**: "A 10% oil price shock *is associated with* a 3% depreciation of the exchange rate (REDUCED_FORM mode)."

**FAIL**: "Higher interest rates *lead to* lower inflation through the credit channel."
- Why: "leads to" is a forbidden causal verb for non-identified paths.

**PASS**: "The model *predicts* that higher interest rates are associated with lower inflation through the credit channel."

---

## Rule 2: Mode-Appropriate Verb Choice

**Rule**: The strongest language allowed depends on the weakest claim level on any open path.

### Complete Verb Table by Claim Level

| Weakest claim on path | Allowed verb phrases | Not allowed |
|---|---|---|
| `IDENTIFIED_CAUSAL` (all edges) | "causes", "causal effect of", "the structural estimate implies", "produces", "generates" | — |
| `REDUCED_FORM` (any edge) | "is associated with", "predicts", "the model predicts", "the reduced-form estimate suggests" | "causes", "drives", "leads to", "results in", "produces", "generates", "brings about" |
| `DESCRIPTIVE` | "correlates with", "co-moves with", "is correlated with" | All causal AND associative language ("is associated with", "predicts") |
| `BLOCKED_ID` | "correlates with", "co-moves with" (same as DESCRIPTIVE) | All causal AND associative language |
| `diagnostic_only` role | Should not appear on open paths — if it does, this is an engine bug | — |

### Examples

**FAIL** (REDUCED_FORM mode): "Oil price shocks *drive* exchange rate movements."
- Why: "drives" is causal language; REDUCED_FORM only allows "is associated with" / "predicts".

**PASS** (REDUCED_FORM mode): "Oil price shocks *are associated with* exchange rate movements."

**FAIL** (DESCRIPTIVE mode): "The data suggests that oil prices *predict* GDP growth."
- Why: "predicts" implies directional association, which exceeds what DESCRIPTIVE mode supports.

**PASS** (DESCRIPTIVE mode): "Oil prices *co-move with* GDP growth in the observed sample."

**PASS** (STRUCTURAL mode, all IDENTIFIED_CAUSAL): "The structural estimate implies that a 1 sd oil supply shock *causes* a 2.3% exchange rate depreciation."

---

## Rule 3: Always State the Query Mode

**Rule**: Every interpretation of propagation results must explicitly name the active query mode.

### Examples

**FAIL**: "A shock to oil prices propagates with an effect of -0.23 on the exchange rate."
- Why: No query mode stated. The reader cannot assess identification strength.

**PASS**: "In REDUCED_FORM mode, a shock to oil prices is associated with an effect of -0.23 on the exchange rate."

---

## Rule 4: SE Independence Disclaimer

**Rule**: For any path with 2 or more edges, include a disclaimer that standard errors assume independence.

### Examples

**FAIL**: "The effect is -0.23 (SE: 0.05, 95% CI: [-0.33, -0.13]) via a 3-edge path."
- Why: Missing SE independence disclaimer for multi-edge path.

**PASS**: "The effect is -0.23 (SE: 0.05, 95% CI: [-0.33, -0.13]) via a 3-edge path. Standard errors assume independence between edge estimates and may understate true uncertainty."

**OK** (single-edge path, no disclaimer needed): "The direct effect is -0.23 (SE: 0.05, 95% CI: [-0.33, -0.13])."

---

## Rule 5: Report All Blocked Paths

**Rule**: Never silently omit blocked edges. Always report how many paths are blocked and why.

### Examples

**FAIL**: "There are 2 open paths from oil to GDP." (when there are also 3 blocked paths)
- Why: Blocked paths are omitted, giving a misleadingly complete picture.

**PASS**: "There are 2 open paths from oil to GDP. 3 additional paths are blocked: 2 due to mode gating (edges lack REDUCED_FORM eligibility) and 1 due to a CRITICAL TSGuard timing failure."

---

## Rule 6: Draft Framing

**Rule**: All propagation results must be framed as draft estimates requiring analyst verification.

### Examples

**FAIL**: "The analysis shows that oil price changes affect GDP by -0.15 per standard deviation."
- Why: Presented as a final conclusion without draft framing.

**PASS**: "The analysis shows that oil price changes are associated with GDP changes of -0.15 per standard deviation. These are draft estimates requiring analyst verification."

---

## Full Interpretation Template

> In **[MODE]** mode, a **[magnitude] [unit]** shock to **[source]** is **[associated with / predicted to produce / structurally estimated to cause]** an effect of **[scaled_effect] [unit]** on **[target]** (95% CI: **[scaled_ci_lower, scaled_ci_upper]**) via **[N]** open path(s). **[K]** path(s) are blocked due to **[reasons]**. Standard errors assume independence between edge estimates and may understate true uncertainty. These are draft estimates requiring analyst verification.

### Filled Example (REDUCED_FORM)

> In **REDUCED_FORM** mode, a **-30% (pct)** shock to **brent_oil** is **associated with** an effect of **+4.2% (pct)** on **usdkzt** (95% CI: [+1.8%, +6.6%]) via **2** open path(s). **1** path is blocked due to mode gating (edge `nbk_rate_to_cpi` lacks REDUCED_FORM eligibility). Standard errors assume independence between edge estimates and may understate true uncertainty. These are draft estimates requiring analyst verification.
