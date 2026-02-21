---
name: econ-reasoning
description: >
  Background knowledge for economics reasoning. Provides error taxonomy,
  sign convention rules, unit conversion tables, and standard elasticity
  ranges to prevent common analysis mistakes.
user-invocable: false
---

# Economics Reasoning Guardrails

You are operating with economics guardrails active. Before presenting any economics analysis, verify your output against these 7 error types.

## Error Type 1: Sign Convention Inconsistency

**Rule**: Every coefficient's sign must be consistent with its verbal description throughout the entire response.

- If you describe an effect as "offsetting" or "reducing", the coefficient must be negative
- If you describe an effect as "amplifying" or "increasing", the coefficient must be positive
- A ratio described as 0.3-0.7 (positive fraction) cannot have a coefficient of -0.5

**Self-check**: For each coefficient mentioned, find every sentence that describes its direction. Do they all agree?

## Error Type 2: Mechanism Confusion

**Rule**: Correctly attribute economic mechanisms to supply-side or demand-side channels.

Supply-side mechanisms:
- OPEC production responses (spare capacity, quota adjustments)
- Technology-driven cost changes (fracking, renewables)
- Input supply shocks (labor, materials, infrastructure)
- Regulatory supply constraints (sanctions, environmental rules)

Demand-side mechanisms:
- Consumer substitution (fuel switching, efficiency)
- Income effects on consumption
- Inventory demand / speculation
- Derived demand from industrial activity

**Self-check**: For each mechanism cited, verify it belongs to the correct side. OPEC responses are supply, not demand.

## Error Type 3: Missing Horizon Differentiation

**Rule**: Always specify whether an elasticity or effect is short-run or long-run. Never present a single number without a time horizon.

Standard patterns:
- Short-run oil price elasticity of demand: -0.02 to -0.10
- Long-run oil price elasticity of demand: -0.20 to -0.60
- Short-run oil price elasticity of supply: 0.01 to 0.05
- Long-run oil price elasticity of supply: 0.10 to 0.40

**Self-check**: Does every elasticity or multiplier have an explicit horizon qualifier?

## Error Type 4: Causal Direction Errors

**Rule**: Distinguish between:
- **Structural equations** (mechanisms, support counterfactuals)
- **Reaction functions** (behavioral responses, do NOT support counterfactuals about the policy instrument)
- **Equilibrium correlations** (reduced-form associations, no causal claims)

A central bank reaction function tells you what the bank *does* react to, not what *would happen* if you changed its instrument exogenously.

**Self-check**: For each causal claim, what is the identification strategy? Can you name it (IV, RDD, diff-in-diff, structural model)?

## Error Type 5: Unit Mixing

**Rule**: Never combine quantities with different units without explicit conversion.

Common traps:
- mb/d (million barrels per day) vs barrels vs annual production
- Percentage change vs percentage points vs basis points
- Nominal vs real values
- Per-capita vs aggregate

**Self-check**: For each calculation, write out the units on both sides. Do they match?

See `references/unit-conversions.md` for conversion tables.

## Error Type 6: Stock vs Flow Confusion

**Rule**: Never conflate stock variables with flow variables.

| Stock (level at a point in time) | Flow (quantity over a period) |
|---|---|
| Reserves (barrels) | Production (barrels/day) |
| Debt outstanding ($) | Budget deficit ($/year) |
| Capital stock ($) | Investment ($/year) |
| Money supply ($) | Money creation ($/year) |
| Foreign reserves ($) | Current account ($/year) |
| Population (persons) | Migration (persons/year) |

**Self-check**: Is each variable classified correctly as stock or flow? Are rates (flows) being compared to levels (stocks)?

## Error Type 7: Real vs Nominal Confusion

**Rule**: When comparing monetary values across time periods, always specify whether values are real or nominal.

- "Oil was $30 in 2003 and $80 in 2024" — nominal comparison, misleading without deflation
- Must specify: base year, deflator (CPI, GDP deflator, PPI)
- Real vs nominal interest rates: Fisher equation (r ≈ i - π)

**Self-check**: Are any dollar values compared across years? If so, is the deflation method specified?

## Hedged Language Rules for Causal Propagation Results

When interpreting OpenCausality propagation results (`propagate_shock`, `propagate_policy`, `find_paths`, `target_contributors`), you must use language that matches the identification strength of the path. This is a **hard rule**, not a suggestion.

### Word Substitution Table

| Path claim level | Allowed verbs | Forbidden verbs |
|---|---|---|
| All edges `IDENTIFIED_CAUSAL` | "causes", "causal effect", "the structural estimate implies" | — |
| Any edge `REDUCED_FORM` | "is associated with", "predicts", "the model predicts" | "causes", "drives", "leads to", "results in", "produces", "generates", "brings about" |
| Any edge `DESCRIPTIVE` or `BLOCKED_ID` | "co-moves with", "correlates with" | All of the above + "is associated with", "predicts" |

### Mode-Appropriate Framing

| Query mode | Opening framing |
|---|---|
| STRUCTURAL | "In STRUCTURAL mode, the structural estimate implies..." |
| REDUCED_FORM | "In REDUCED_FORM mode, the model predicts..." |
| DESCRIPTIVE | "In DESCRIPTIVE mode, the correlation suggests..." |

### Interpretation Template

When presenting propagation results, follow this structure:

> In [MODE] mode, a [magnitude] [unit] shock to [source] is [associated with / predicted to produce / structurally estimated to cause] an effect of [scaled_effect] [unit] on [target] (95% CI: [scaled_ci_lower, scaled_ci_upper]) via [N] open path(s). [K] path(s) are blocked due to [reasons]. Standard errors assume independence between edge estimates and may understate true uncertainty. These are draft estimates requiring analyst verification.

### Mandatory Disclosures

1. **Always state the query mode** — readers must know the identification strength
2. **Report blocked paths** — never silently omit blocked edges; list reasons
3. **SE independence disclaimer** — required for any path with 2+ edges
4. **Time horizon** — specify short-run vs long-run for all elasticities
5. **Draft framing** — all results are draft estimates, not final conclusions

**Self-check**: Does your interpretation use only verbs from the "Allowed" column for the weakest claim level on any open path?

See `references/hedged-language-rules.md` for detailed PASS/FAIL examples.

## Reference Data

For detailed reference tables, see:
- `references/common-errors.md` — Worked examples of each error type
- `references/elasticities-reference.md` — Standard ranges with academic sources
- `references/unit-conversions.md` — Conversion tables for energy, finance, and macro
- `references/hedged-language-rules.md` — Hedged language rules with PASS/FAIL examples
