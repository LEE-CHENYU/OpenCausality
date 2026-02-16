# Hedged Language Rules (INVARIANTS Section 23.1)

These rules are hard invariants. Violations are bugs, not style choices.

## Claim-Level Language Map

| Path Status | Permitted Language | Examples |
|---|---|---|
| All edges `IDENTIFIED_CAUSAL` | "causes", "causal effect", "drives" | "Oil price shocks cause inflation..." |
| Any edge `REDUCED_FORM` | "is associated with", "predicts", "co-varies with" | "Oil price is associated with a 0.3pp increase..." |
| Any edge `DESCRIPTIVE` or `BLOCKED_ID` | "co-moves with", "correlates with", "is observed alongside" | "Oil price co-moves with inflation..." |

The weakest claim level in a multi-edge path determines the language
for the entire path.

## Forbidden Language (Non-Identified Paths)

The following terms and their inflections must **never** appear in output
for paths below `IDENTIFIED_CAUSAL`:

- causes
- causal effect
- drives
- leads to
- results in
- produces
- generates
- brings about

This is a closed forbidden set. All terms carry causal connotations and
are reserved exclusively for fully identified paths.

## Required Disclosures

### 1. Always state the mode

Every propagation result must include:
- "Query mode: REDUCED_FORM" (or whichever is active)

### 2. Always state the weakest claim level

- "Weakest claim in this path: REDUCED_FORM"

### 3. SE independence disclaimer (multi-edge paths)

When a path traverses more than one estimated edge:
> "Standard errors assume independence between edge estimates.
> This assumption is likely violated due to shared data or common shocks.
> Confidence intervals may be too tight."

### 4. Blocked path explanation

If any path was blocked, explain the reason:
- Mode restriction: edge role not permitted in current mode
- Counterfactual blocked: claim level insufficient for shock/policy CF
- TSGuard blocked: time-series diagnostics indicate instability
- Issue ledger blocked: critical open issue on the edge
- Unit mismatch: incompatible units between consecutive edges
- Frequency mismatch: mixed frequencies without frequency bridge

### 5. Draft framing

All estimation outputs must be framed as:
> "Draft — requires analyst review"

This framing is non-optional and must not be softened.

## Examples

### Good (REDUCED_FORM path)

> In **REDUCED_FORM** mode, a 1 SD shock to `brent` is **associated with**
> a 0.0042 change in `kspi_k2` (SE = 0.0018, 95% CI [0.0007, 0.0077]).
> Weakest claim: REDUCED_FORM.
>
> SE assumes independence between edge estimates (likely violated).
>
> *Draft — requires analyst review.*

### Bad (REDUCED_FORM path)

> Oil price **causes** a 0.0042 change in KSPI K2.

This violates Rule 1 (uses "causes" for a REDUCED_FORM path) and omits
the mode, weakest claim, SE disclaimer, and draft framing.
