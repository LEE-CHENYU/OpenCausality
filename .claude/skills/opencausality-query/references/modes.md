# OpenCausality Query Modes

The three query modes form a strict hierarchy of permitted evidence.
Mode can only restrict, never expand. It is the outer gate on all queries.

## Hierarchy

`STRUCTURAL` ⊂ `REDUCED_FORM` ⊂ `DESCRIPTIVE`

## Mode Specifications

### STRUCTURAL (strictest)

**Purpose:** Mechanism-based causal analysis with identified effects.

**Propagation roles allowed:** structural, bridge, identity

**Counterfactuals:**
- Policy interventions: Yes, requires `IDENTIFIED_CAUSAL` claim level
- Shock scenarios: Yes, requires `IDENTIFIED_CAUSAL` claim level

**When to use:** When the user asks about policy interventions,
structural mechanisms, or wants maximum rigor.

**Hedged language:** "causes", "causal effect" permitted only for fully
identified edges.

### REDUCED_FORM (default)

**Purpose:** Statistical association analysis, including well-estimated
reduced-form relationships.

**Propagation roles allowed:** structural, reduced_form, bridge, identity

**Counterfactuals:**
- Policy interventions: Never (requires STRUCTURAL)
- Shock scenarios: Yes, requires `REDUCED_FORM` or higher claim level

**When to use:** Default mode. Suitable for shock scenarios, "what if"
questions based on statistical associations.

**Hedged language:** "is associated with", "predicts", "a one-unit
change in X is associated with..."

### DESCRIPTIVE (most permissive)

**Purpose:** Data exploration. All edges are visible regardless of
identification quality.

**Propagation roles allowed:** all (including diagnostic_only)

**Counterfactuals:**
- Policy interventions: Never
- Shock scenarios: Never

**When to use:** Exploration, understanding the DAG structure, checking
what's available.

**Hedged language:** "co-moves with", "correlates with", "is observed
alongside..."

## Role Derivation

Roles are derived at runtime from `(edge_type, claim_level)`:

| Edge type | Claim level | Role |
|-----------|------------|------|
| identity | any | identity |
| mechanical / bridge | any | bridge |
| reaction_function | any | diagnostic_only |
| immutable | any | structural |
| causal | IDENTIFIED_CAUSAL | structural |
| causal | REDUCED_FORM | reduced_form |
| causal | DESCRIPTIVE | diagnostic_only |
| causal | BLOCKED_ID | diagnostic_only |

## Mode Switching

Users can switch modes at any time. The mode affects only which edges
participate in propagation — it does not change the underlying DAG or
edge cards.

Switching from a less restrictive to a more restrictive mode may cause
previously visible paths to become blocked.
