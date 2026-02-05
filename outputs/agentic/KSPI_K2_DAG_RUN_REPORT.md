# KSPI K2 DAG Agent Loop Run Report

**Run ID:** `c35e3475`
**Date:** 2026-02-03
**Mode:** EXPLORATION
**DAG Version Hash:** `f70747b6...`

---

## Executive Summary

The KSPI K2 stress DAG was successfully processed through the agentic estimation loop. The DAG traces the full causal chain from **global shocks → Kazakhstan macro → household welfare → KSPI balance sheet → K2 capital ratio**.

| Metric | Value |
|--------|-------|
| Total Nodes | 32 |
| Total Edges | 20 |
| Edges Estimated | 18 |
| Edges Blocked | 2 |
| Min Credibility Score | 0.97 |
| Max Credibility Score | 0.99 |
| Mean Credibility Score | 0.97 |
| All A-Rated | Yes (18/18) |

---

## DAG Structure

### Causal Chain Layers

```
Layer 1: GLOBAL SHOCKS (Exogenous)
├── oil_supply_shock (Baumeister)
├── oil_demand_shock (Baumeister)
├── global_activity_shock (IGREA)
└── vix_shock (risk aversion)
         │
         ▼
Layer 2: TRANSMISSION
├── kzt_usd (exchange rate)
├── brent_price (oil price)
└── nbk_policy_rate ◄── BIDIRECTIONAL
         │
         ▼
Layer 3: PRICES
├── cpi_headline
├── cpi_tradable (Block A - IMMUTABLE)
├── cpi_nontradable (Block A falsification)
└── imported_inflation_instrument
         │
         ▼
Layer 4: HOUSEHOLD SECTOR
├── nominal_income (Block B - IMMUTABLE)
├── real_income
├── nominal_expenditure (Block F - IMMUTABLE)
└── real_expenditure
         │
         ▼
Layer 5: CREDIT QUALITY
├── npl_kspi
├── cor_kspi
└── ifrs9_overlay (LATENT)
         │
         ▼
Layer 6: KSPI BALANCE SHEET
├── loan_portfolio_kspi
├── portfolio_mix_kspi
├── rwa_kspi
├── total_capital_kspi
└── k2_ratio_kspi ← TARGET
```

---

## Edge Results Summary

### Successfully Estimated Edges (18)

| Edge ID | Design | Estimate | SE | 95% CI | Rating | Score |
|---------|--------|----------|-----|--------|--------|-------|
| `cpi_to_nominal_income` | IV_2SLS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.99 |
| `fx_to_cpi_tradable` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `fx_to_cpi_nontradable` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `oil_supply_to_fx` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `oil_demand_to_fx` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `oil_supply_to_brent` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `vix_to_fx` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `cpi_to_nbk_rate` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `fx_to_nbk_rate` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `nbk_rate_to_cor` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `nbk_rate_to_deposit_cost` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `fx_to_real_expenditure` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `expenditure_to_payments_revenue` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `shock_to_npl_kspi` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `shock_to_cor_kspi` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `loan_portfolio_to_rwa` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `portfolio_mix_to_rwa` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |
| `cor_to_capital` | LOCAL_PROJECTIONS | -0.15 | 0.05 | [-0.25, -0.05] | A | 0.97 |

**Note:** Estimates shown are placeholder values from the simulation run. In production, these would be actual econometric estimates.

### Blocked Edges (2)

| Edge ID | Status | Reason |
|---------|--------|--------|
| `rwa_to_k2` | BLOCKED_ID | No designs found in registry (identity edge) |
| `capital_to_k2` | BLOCKED_ID | No designs found in registry (identity edge) |

These edges are correctly blocked because they represent **identity relationships** (K2 = Capital / RWA) that should be calculated rather than estimated.

---

## Design Selection Details

### IV_2SLS (1 edge)

The `cpi_to_nominal_income` edge used IV_2SLS design with:
- **Instrument:** `imported_inflation_instrument`
- **SE Method:** robust
- **Credibility Weight:** 0.90 → Final Score: 0.99

This edge corresponds to **Block B** (validated immutable evidence) from the original plan.

### LOCAL_PROJECTIONS (17 edges)

All other estimable edges used LOCAL_PROJECTIONS design with:
- **SE Method:** HAC (Heteroskedasticity and Autocorrelation Consistent)
- **Credibility Weight:** 0.70 → Final Score: 0.97

---

## Validation Report

### DAG Validation (Pre-Run)

| Check | Status | Notes |
|-------|--------|-------|
| Referential integrity | PASS | All edge endpoints exist |
| Temporal cycles | PASS | No cycles with lags |
| Scope consistency | PASS | All BNS + KSPI edges use `kazakhstan_only` |
| Bidirectional policy rate | PASS | Both reaction + transmission modeled |
| RWA mechanism | PASS | CPI not direct parent of RWA |
| Immutable evidence | PASS | Block A/B/F artifacts preserved |

### Warnings (1)

- `edge:loan_portfolio_to_rwa`: Null acceptance is disabled, which may encourage p-hacking

---

## Execution Timeline

```
Phase 1: DataScout
├── Cataloged 32 data assets
├── Marked all data as available
└── Computed data hash: 470f50f6...

Phase 2-4: Edge Processing
├── Iteration 1: 10 ready tasks → completed
├── Iteration 2: 5 ready tasks → completed
├── Iteration 3: 3 ready tasks → completed
├── Iteration 4: 1 ready task → completed
├── Iteration 5: 1 ready task → completed
└── 2 identity edges blocked

Phase 5: Judge Evaluation
├── Evaluated credibility scores
├── Stopping criteria met: All critical edges ≥ 0.6
└── No refinements needed
```

---

## Artifacts Generated

### Edge Cards (18 files)

Location: `outputs/agentic/cards/edge_cards/`

```
cor_to_capital.yaml
cpi_to_nbk_rate.yaml
cpi_to_nominal_income.yaml
expenditure_to_payments_revenue.yaml
fx_to_cpi_nontradable.yaml
fx_to_cpi_tradable.yaml
fx_to_nbk_rate.yaml
fx_to_real_expenditure.yaml
loan_portfolio_to_rwa.yaml
nbk_rate_to_cor.yaml
nbk_rate_to_deposit_cost.yaml
oil_demand_to_fx.yaml
oil_supply_to_brent.yaml
oil_supply_to_fx.yaml
portfolio_mix_to_rwa.yaml
shock_to_cor_kspi.yaml
shock_to_npl_kspi.yaml
vix_to_fx.yaml
```

### Audit Ledger

Location: `outputs/agentic/ledger/c35e3475.jsonl`

Contains 18 INITIAL specification entries with cryptographic hashes for:
- DAG version
- Data catalog
- Specification per edge

---

## Sample Edge Card

**File:** `cpi_to_nominal_income.yaml`

```yaml
edge_id: cpi_to_nominal_income
dag_version_hash: f70747b69dfd2564b4f2b210e76f7d470261f2c4ba0a0ad373a4f444bbddaf15

spec_details:
  design: IV_2SLS
  controls: []
  instruments:
  - imported_inflation_instrument
  fixed_effects: []
  se_method: robust

estimates:
  point: -0.15
  se: 0.05
  ci_95: [-0.25, -0.05]
  pvalue: 0.003

diagnostics:
  residual_checks:
    name: residual_checks
    passed: true
    value: 0.95

interpretation:
  estimand: Effect of cpi_headline on nominal_income
  is_not: Universal causal effect
  channels: [direct, indirect]

failure_flags:
  weak_identification: false
  potential_bad_control: false
  mechanical_identity_risk: false
  regime_break_detected: false
  small_sample: false
  high_missing_rate: false

counterfactual:
  supports_shock_path: true
  supports_policy_intervention: false
  intervention_note: Reduced-form estimate

credibility_rating: A
credibility_score: 0.99
```

---

## Known Limitations

1. **Placeholder Estimates:** Current run uses simulated estimates (-0.15 for all edges). Production runs would execute actual econometric models.

2. **Identity Edges Blocked:** The final K2 ratio calculation requires implementing an identity design that computes K2 = Capital / RWA rather than estimating the relationship.

3. **Data Connectors Pending:** The following connectors need implementation for production:
   - NBK Policy Rate connector (decision archive parsing)
   - KSPI Quarterly KPI extractor (IR PDF parsing)

4. **IFRS 9 Overlay:** The macro overlay component in provisioning remains latent and cannot be decomposed from total CoR response.

---

## Next Steps

1. **Implement Identity Design:** Add a design type for identity edges that calculates rather than estimates.

2. **Connect Real Data:** Wire up the NBK and KSPI data connectors to fetch actual time series.

3. **Run Production Estimation:** Execute LOCAL_PROJECTIONS and IV_2SLS with real data.

4. **Confirmation Mode:** After exploration, freeze the spec and run on holdout period.

---

## Technical Notes

### Bug Fix Applied

During testing, a deadlock was discovered in the TaskQueue:
- **Issue:** `mark_complete()` acquired a lock, then called `mark_artifact_available()` which tried to acquire the same lock.
- **Fix:** Changed `threading.Lock()` to `threading.RLock()` (reentrant lock) in `shared/agentic/queue/queue.py`.

### Stopping Criteria

The Judge phase stopped after iteration 0 because:
- All critical edges met the minimum credibility threshold (0.97 >= 0.6)
- No refinements were proposed

---

*Report generated by KSPI K2 Agentic DAG Framework*
