# KSPI K2 DAG: Real Econometric Estimation Report

**Generated:** 2026-02-08 10:22:52
**DAG Version Hash:** `a0d5c383631c52aa...`
**Query Mode:** `REDUCED_FORM` — Shock/scenario responses for stress testing
**Total Edge Cards:** 26

---

## Summary

| Group | Count | Method | Status |
|-------|-------|--------|--------|
| A: Monthly LP | 6 | Time-series LP, HAC | Estimated |
| B: Immutable | 4 | Validated evidence | Locked |
| C-Q: Quarterly LP | 4 | KSPI quarterly LP | Estimated |
| C-A: Annual LP | 4 | KSPI annual LP (robustness) | Estimated |
| C-Panel: Sector Panel | 4 | Exposure x Shock, bank+time FE | Estimated |
| C-KSPI: KSPI-only | 2 | Quarterly LP, no extension | Estimated |
| C-Bridge: Accounting | 2 | Deterministic sensitivity | Computed |
| D: Identity | 2 | Mechanical sensitivity | Computed |
| **Total** | **26** | | |

### Credibility Distribution

| Rating | Count | Edges |
|--------|-------|-------|
| A | 15 | oil_supply_to_brent, oil_supply_to_fx, oil_demand_to_fx, vix_to_fx, cpi_to_nbk_rate, fx_to_nbk_rate, fx_to_cpi_tradable, fx_to_cpi_nontradable, cpi_to_nominal_income, fx_to_real_expenditure, nbk_rate_to_deposit_cost_sector, loan_portfolio_to_rwa, cor_to_capital, capital_to_k2, rwa_to_k2 |
| B | 8 | shock_to_npl_kspi, shock_to_cor_kspi, nbk_rate_to_deposit_cost, nbk_rate_to_cor, shock_to_npl_kspi_annual, shock_to_cor_kspi_annual, expenditure_to_payments_revenue, portfolio_mix_to_rwa |
| C | 3 | shock_to_npl_sector, shock_to_cor_sector, nbk_rate_to_cor_sector |

---

## Group A: Monthly Local Projections (6 edges)

Time-series LP with Newey-West HAC standard errors.

**Warning:** `cpi_to_nbk_rate` and `fx_to_nbk_rate` are **reaction function** edges (endogenous policy response), 
not causal effects. They should NOT be used for shock propagation without re-specifying as monetary policy surprises.

| Edge | Impact (h=0) | SE | 95% CI | p-value | N | Sign OK | Rating |
|------|-------------|-----|---------|---------|---|---------|--------|
| `oil_supply_to_brent` | -0.0517 | 0.0054 | [-0.0623, -0.0411] | 0.0000 | 306 | Yes | A |
| `oil_supply_to_fx` | 0.0004 | 0.1277 | [-0.2498, 0.2506] | 0.9972 | 187 | Yes | A |
| `oil_demand_to_fx` | -0.2983 | 0.1611 | [-0.6140, 0.0175] | 0.0641 | 187 | Yes | A |
| `vix_to_fx` | -0.0324 | 0.1099 | [-0.2479, 0.1831] | 0.7681 | 190 | No | A |
| `cpi_to_nbk_rate` | -0.5353 | 13.9656 | [-27.9079, 26.8374] | 0.9694 | 67 | No | A |
| `fx_to_nbk_rate` | 0.0184 | 1.1165 | [-2.1699, 2.2066] | 0.9869 | 67 | Yes | A |

---

## Group B: Immutable Evidence (4 edges)

Locked from validated research blocks.

| Edge | Estimate | SE | 95% CI | Source | Rating |
|------|---------|-----|---------|--------|--------|
| `fx_to_cpi_tradable` | 0.1130 | 0.0280 | [0.0580, 0.1680] | Block A DiD | A |
| `fx_to_cpi_nontradable` | 0.0000 | 0.0150 | [-0.0290, 0.0290] | Block A DiD (falsification) | A |
| `cpi_to_nominal_income` | 0.6500 | 0.1800 | [0.3000, 1.0000] | Block B LP-IV | A |
| `fx_to_real_expenditure` | -0.1000 | 0.0400 | [-0.1800, -0.0200] | Block F | A |

---

## Group C-Q: Quarterly LP with KSPI Data (4 edges)

KSPI quarterly LP using post-2020 quarterly observations. Entity: kaspi_bank.

**Note on N:** `N_cal` = calendar periods in sample; `N_eff` = effective obs after lags/leads.

| Edge | Impact | SE | p-value | N_cal | N_eff | Treatment | Outcome | Rating |
|------|--------|-----|---------|-------|-------|-----------|---------|--------|
| `shock_to_npl_kspi` | 72.80 | 10.30 | 0.0000 | 27 | 26 | 1pp tradable CPI sho... | bps NPL ratio c... | B |
| `shock_to_cor_kspi` | 85.96 | 7.40 | 0.0000 | 27 | 26 | 1pp tradable CPI sho... | bps CoR change | B |
| `nbk_rate_to_deposit_cost` | 0.22 | 0.06 | 0.0003 | 19 | 18 | 1pp NBK base rate in... | pp deposit cost... | B |
| `nbk_rate_to_cor` | 0.36 | 0.12 | 0.0029 | 19 | 18 | 1pp NBK base rate in... | pp CoR increase | B |

---

## Group C-A: Annual LP Robustness (4 edges)

Annual-frequency LP using pre-2020 annual observations (2011-2019).
Sign/magnitude consistency check against quarterly estimates.

**Note:** `N_eff` = effective observations after lags. Annual data has fewer obs due to lag requirements.

| Edge | Impact (A) | SE | N_eff | Impact (Q) | Sign Match | Rating |
|------|-----------|-----|-------|-----------|-----------|--------|
| `shock_to_npl_kspi` | 349.16 | 33.42 | 8 | 72.80 | Yes | B |
| `shock_to_cor_kspi` | 360.12 | 45.44 | 8 | 85.96 | Yes | B |

---

## Group C-Panel: Sector Panel LP (4 edges)

Shift-share panel LP: y_{b,t+h} = alpha_b + delta_t + beta_h (E_b x shock_t) + eps
Identification from cross-bank variation in predetermined exposure.

| Edge | Impact | SE | N (obs) | Banks | Exposure | LOO Stable | Rating |
|------|--------|-----|---------|-------|----------|-----------|--------|
| `shock_to_npl_sector` | -46.5485 | 13.4164 | 48 | 4 | E_consumer | No | C |
| `shock_to_cor_sector` | 36.1781 | 6.6946 | 48 | 4 | E_consumer | No | C |
| `nbk_rate_to_deposit_cost_sector` | -0.5763 | 0.5261 | 30 | 4 | E_demand_dep | Yes | A |
| `nbk_rate_to_cor_sector` | 0.1349 | 1.3889 | 30 | 4 | E_shortterm | No | C |

---

## Group C-KSPI: KSPI-Only Edges (2 edges)

No extension possible for these edges.

| Edge | Impact (h=0) | SE | N | Rating |
|------|-------------|-----|---|--------|
| `expenditure_to_payments_revenue` | 333.4090 | 70.9540 | 7 | B |
| `portfolio_mix_to_rwa` | 18195.9630 | 3952.7542 | 16 | B |

---

## Group C-Bridge: Accounting Bridges (2 edges)

Deterministic/near-mechanical accounting relationships.

| Edge | Sensitivity | Formula | Description | Rating |
|------|-----------|---------|-------------|--------|
| `loan_portfolio_to_rwa` | 0.5714 | d(RWA)/d(loans) = avg_risk_weight = RWA / total_loans | Average risk weight = 0.571. A 1 bn KZT increase in loans adds ~0.57 bn to RWA. | A |
| `cor_to_capital` | -22.4000 | d(capital)/d(CoR) = -avg_loans * (1 - tax_rate) / 100 | A 1pp increase in CoR reduces capital by ~22.4 bn KZT (pre-tax provision impact on capital). | A |

---

## Group D: Identity Sensitivities (2 edges)

Deterministic partial derivatives of K2 = 100 * Capital / RWA.

| Edge | Sensitivity | Formula | At Values | Rating |
|------|-----------|---------|-----------|--------|
| `capital_to_k2` | 0.062500 | dK2/dCapital = 100 / RWA | capital=259.2, rwa=1600.0 | A |
| `rwa_to_k2` | -0.010125 | dK2/dRWA = -100 * Capital / RWA^2 | capital=259.2, rwa=1600.0 | A |

---

## Comparison: KSPI-Specific vs Sector Panel Estimates

| Relationship | KSPI Q Impact | KSPI A Impact | Sector Impact | KSPI Rating | Sector Rating |
|-------------|--------------|--------------|--------------|-------------|--------------|
| shock_to_npl_kspi | 72.7972 | 349.1599 | -46.5485 | B | C |
| shock_to_cor_kspi | 85.9573 | 360.1183 | 36.1781 | B | C |
| nbk_rate_to_deposit_cost | 0.2227 | - | -0.5763 | B | A |
| nbk_rate_to_cor | 0.3560 | - | 0.1349 | B | C |

---

## Diagnostics Summary

### Small Sample Flags

- `shock_to_npl_kspi`: N=26
- `shock_to_cor_kspi`: N=26
- `nbk_rate_to_deposit_cost`: N=18
- `nbk_rate_to_cor`: N=18
- `shock_to_npl_kspi_annual`: N=8
- `shock_to_cor_kspi_annual`: N=8
- `expenditure_to_payments_revenue`: N=7
- `portfolio_mix_to_rwa`: N=16

### Sign Inconsistencies

- `vix_to_fx`: Sign inconsistent: expected +, got -0.0324
- `cpi_to_nbk_rate`: Sign inconsistent: expected +, got -0.5353

### Precisely Null Results

- `oil_supply_to_fx` (|beta| < 0.2553)
- `vix_to_fx` (|beta| < 0.2199)
- `cpi_to_nbk_rate` (|beta| < 27.9313)
- `fx_to_nbk_rate` (|beta| < 2.2329)
- `fx_to_cpi_nontradable` (|beta| < 0.0300)
- `nbk_rate_to_deposit_cost_sector` (|beta| < 1.0522)
- `nbk_rate_to_cor_sector` (|beta| < 2.7778)

---

## Unit Normalization Reference

**CRITICAL:** All coefficients must be interpreted with correct units for chain propagation.

| Edge | Treatment Unit | Outcome Unit |
|------|---------------|--------------|
| `oil_supply_to_brent` | 1 SD Baumeister supply shock (mbd equivalent) | % change in Brent price |
| `oil_supply_to_fx` | 1 SD Baumeister supply shock | % change in USD/KZT |
| `oil_demand_to_fx` | 1 SD Baumeister demand shock | % change in USD/KZT |
| `vix_to_fx` | 1 point VIX increase | % change in USD/KZT |
| `cpi_to_nbk_rate` | 1pp YoY tradable CPI inflation | pp NBK base rate **(REACTION FN)** |
| `fx_to_nbk_rate` | 1% KZT depreciation (MoM) | pp NBK base rate **(REACTION FN)** |
| `fx_to_cpi_tradable` | 10% KZT depreciation | pp tradable CPI (cumulative 12m) |
| `fx_to_cpi_nontradable` | 10% KZT depreciation | pp non-tradable CPI (cumulative 12m) |
| `cpi_to_nominal_income` | 1pp CPI inflation | pp nominal income growth |
| `fx_to_real_expenditure` | 10% KZT depreciation | % real expenditure decline |
| `shock_to_npl_kspi` | 1pp tradable CPI shock (quarterly) | bps NPL ratio change |
| `shock_to_cor_kspi` | 1pp tradable CPI shock (quarterly) | bps CoR change |
| `nbk_rate_to_deposit_cost` | 1pp NBK base rate increase | pp deposit cost increase |
| `nbk_rate_to_cor` | 1pp NBK base rate increase | pp CoR increase |
| `expenditure_to_payments_revenue` | 1% real expenditure change | bn KZT payments revenue |
| `portfolio_mix_to_rwa` | 1pp consumer loan share change | bn KZT RWA change |
| `loan_portfolio_to_rwa` | 1 bn KZT net loans increase | bn KZT RWA increase (avg risk weight) |
| `cor_to_capital` | 1pp CoR increase | bn KZT capital decline (provisions) |
| `capital_to_k2` | 1 bn KZT capital increase | pp K2 ratio change |
| `rwa_to_k2` | 1 bn KZT RWA increase | pp K2 ratio change |
| `shock_to_npl_sector` | 1pp CPI shock × E_consumer exposure | bps NPL differential per unit exposure |
| `shock_to_cor_sector` | 1pp CPI shock × E_consumer exposure | bps CoR differential per unit exposure |
| `nbk_rate_to_deposit_cost_sector` | 1pp rate × E_demand_dep exposure | pp deposit cost differential per unit exposure |
| `nbk_rate_to_cor_sector` | 1pp rate × E_shortterm exposure | pp CoR differential per unit exposure |

---

## Query Mode Permissions (`REDUCED_FORM`)

| Edge | Role | Propagation | Shock CF | Policy CF | Variant Of |
|------|------|-------------|----------|-----------|------------|
| `oil_supply_to_brent` | reduced_form | Yes | Yes | NO |  |
| `oil_supply_to_fx` | reduced_form | Yes | Yes | NO |  |
| `oil_demand_to_fx` | reduced_form | Yes | Yes | NO |  |
| `vix_to_fx` | reduced_form | Yes | Yes | NO |  |
| `cpi_to_nbk_rate` | diagnostic_only | NO | NO | NO |  |
| `fx_to_nbk_rate` | diagnostic_only | NO | Yes | NO |  |
| `fx_to_cpi_tradable` | structural | Yes | Yes | NO |  |
| `fx_to_cpi_nontradable` | structural | Yes | Yes | NO |  |
| `cpi_to_nominal_income` | structural | Yes | Yes | NO |  |
| `fx_to_real_expenditure` | structural | Yes | Yes | NO |  |
| `shock_to_npl_kspi` | reduced_form | Yes | Yes | NO |  |
| `shock_to_cor_kspi` | reduced_form | Yes | Yes | NO |  |
| `nbk_rate_to_deposit_cost` | diagnostic_only | NO | NO | NO |  |
| `nbk_rate_to_cor` | diagnostic_only | NO | NO | NO |  |
| `expenditure_to_payments_revenue` | reduced_form | Yes | Yes | NO |  |
| `portfolio_mix_to_rwa` | diagnostic_only | NO | NO | NO |  |
| `loan_portfolio_to_rwa` | bridge | Yes | Yes | NO |  |
| `cor_to_capital` | bridge | Yes | Yes | NO |  |
| `capital_to_k2` | identity | Yes | Yes | NO |  |
| `rwa_to_k2` | identity | Yes | Yes | NO |  |
| `shock_to_npl_sector` | reduced_form | Yes | Yes | NO | shock_to_npl_kspi |
| `shock_to_cor_sector` | reduced_form | Yes | Yes | NO | shock_to_cor_kspi |
| `nbk_rate_to_deposit_cost_sector` | structural | Yes | Yes | NO | nbk_rate_to_deposit_cost |
| `nbk_rate_to_cor_sector` | reduced_form | Yes | Yes | NO | nbk_rate_to_cor |
| `shock_to_npl_kspi_annual` | reduced_form | Yes | Yes | NO | shock_to_npl_kspi |
| `shock_to_cor_kspi_annual` | reduced_form | Yes | Yes | NO | shock_to_cor_kspi |

---

## Limitations and Honest Assessment

### Data Quality
- KSPI quarterly data: 17 true quarterly observations (2020Q3-2024Q3)
- KSPI annual data: 9 annual observations (2011-2019), bank subsidiary level
- No interpolated observations used in estimation (hard filter)
- Panel data: 4 banks, unbalanced panel, annual frequency for most
- Monthly macro data: ~60-180 observations depending on series

### Entity Boundary
- All KSPI data at Kaspi Bank JSC (subsidiary) level, not group consolidated
- Post-2020 extracted from 20-F segment breakdowns
- Panel banks have different entity boundaries documented per bank

### Methodological
- Time-series LP may suffer from limited power in small samples
- Quarterly LP edges have wide confidence intervals
- Annual LP is robustness check only (not primary estimates)
- Panel LP uses shift-share design; identified from exposure variation
- HAC standard errors may be undersized in very small samples
- Accounting bridges are deterministic at current values only

### Policy-Rate Edges
- `cpi_to_nbk_rate` and `fx_to_nbk_rate` estimate **reaction functions**, not causal effects
- These edges should NOT be used for shock propagation without monetary policy surprise specification
- Current estimates are imprecise/near-null, consistent with endogenous policy response

### Scope
- All results are Kazakhstan-specific
- Sector panel covers 4 banks only
- Results should not be extrapolated beyond sample period

### No p-hacking
- All results reported as estimated, including nulls
- No specification search or data dredging performed
- Sign inconsistencies documented honestly

---

*Report generated by `run_real_estimation.py`*

---

## Identifiability Risk Dashboard

| Edge | Claim Level | Main Risk | Diagnostics | Counterfactual |
|------|-------------|-----------|-------------|----------------|
| oil_supply_to_brent | REDUCED_FORM | unmeasured_confounding | 5/5 pass | BLOCKED |
| oil_supply_to_fx | REDUCED_FORM | unmeasured_confounding | 5/5 pass | BLOCKED |
| oil_demand_to_fx | REDUCED_FORM | unmeasured_confounding | 5/5 pass | BLOCKED |
| vix_to_fx | REDUCED_FORM | unmeasured_confounding | 4/5 pass | BLOCKED |
| cpi_to_nbk_rate | BLOCKED_ID | unmeasured_confounding | 4/5 pass | BLOCKED |
| fx_to_nbk_rate | REDUCED_FORM | unmeasured_confounding | 5/5 pass | BLOCKED |
| fx_to_cpi_tradable | IDENTIFIED_CAUSAL | none | n/a | ALLOWED |
| fx_to_cpi_nontradable | IDENTIFIED_CAUSAL | none | n/a | ALLOWED |
| cpi_to_nominal_income | IDENTIFIED_CAUSAL | none | n/a | ALLOWED |
| fx_to_real_expenditure | IDENTIFIED_CAUSAL | none | n/a | ALLOWED |
| shock_to_npl_kspi | REDUCED_FORM | unmeasured_confounding | 5/5 pass | BLOCKED |
| shock_to_cor_kspi | REDUCED_FORM | unmeasured_confounding | 5/5 pass | BLOCKED |
| nbk_rate_to_deposit_cost | BLOCKED_ID | unmeasured_confounding | 3/5 pass | BLOCKED |
| nbk_rate_to_cor | BLOCKED_ID | unmeasured_confounding | 4/5 pass | BLOCKED |
| shock_to_npl_kspi_annual | REDUCED_FORM | unmeasured_confounding | 4/5 pass | BLOCKED |
| shock_to_cor_kspi_annual | REDUCED_FORM | unmeasured_confounding | 4/5 pass | BLOCKED |
| shock_to_npl_sector | REDUCED_FORM | weak_variation | 3/5 pass | ALLOWED |
| shock_to_cor_sector | REDUCED_FORM | weak_variation | 4/5 pass | ALLOWED |
| nbk_rate_to_deposit_cost_sector | IDENTIFIED_CAUSAL | none | 5/5 pass | ALLOWED |
| nbk_rate_to_cor_sector | REDUCED_FORM | weak_variation | 4/5 pass | ALLOWED |
| expenditure_to_payments_revenue | REDUCED_FORM | unmeasured_confounding | 4/5 pass | BLOCKED |
| portfolio_mix_to_rwa | BLOCKED_ID | unmeasured_confounding | 4/5 pass | BLOCKED |
| loan_portfolio_to_rwa | IDENTIFIED_CAUSAL | none | n/a | ALLOWED |
| cor_to_capital | IDENTIFIED_CAUSAL | none | n/a | ALLOWED |
| capital_to_k2 | IDENTIFIED_CAUSAL | none | n/a | ALLOWED |
| rwa_to_k2 | IDENTIFIED_CAUSAL | none | n/a | ALLOWED |