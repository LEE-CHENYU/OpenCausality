# Human-in-the-Loop Checklist
Run ID: 1760dc54

> This checklist contains issues that require your expert judgment before
> the pipeline can proceed. For each item, review the context, understand
> why the issue matters, and record your decision. Your choices are logged
> in the audit trail for reproducibility.

## Pending Decisions

### 1. Regime Instability Decision
- **Edge:** `e_import_share_to_imported_inflation_instrument`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 2. Regime Instability Decision
- **Edge:** `vix_shock_to_deposit_cost_kspi`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 3. Regime Instability Decision
- **Edge:** `imported_inflation_instrument_to_real_income`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 4. Regime Instability Decision
- **Edge:** `imported_inflation_instrument_to_real_expenditure_negative`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 5. Regime Instability Decision
- **Edge:** `cpi_headline_to_real_expenditure`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 6. Regime Instability Decision
- **Edge:** `total_capital_kspi_to_k2_ratio_kspi`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 7. Null link cpi_headline->cpi_nontradable | kzt_usd: significant partial association (beta=0.5442, p=0.0000, N=178). DAG may need this edge.
- from_node: cpi_headline
- to_node: cpi_nontradable
- coefficient: 0.5441943447785492
- se: 0.04175275830914033
- pvalue: 7.866185492520244e-39
- n_obs: 178
- dag_distance: 1
- conditioning_set: ['kzt_usd']
- shared_neighbors: []
- [ ] Decision: _________________

### 8. Null link cpi_headline->cpi_tradable | kzt_usd: significant partial association (beta=0.9622, p=0.0000, N=178). DAG may need this edge.
- from_node: cpi_headline
- to_node: cpi_tradable
- coefficient: 0.9621923746483524
- se: 0.18024449404585535
- pvalue: 9.384166800463749e-08
- n_obs: 178
- dag_distance: 1
- conditioning_set: ['kzt_usd']
- shared_neighbors: []
- [ ] Decision: _________________

### 9. Null link k2_ratio_kspi->total_capital_kspi | cor_kspi, ppop_kspi: significant partial association (beta=8.6636, p=0.0000, N=17). DAG may need this edge.
- from_node: k2_ratio_kspi
- to_node: total_capital_kspi
- coefficient: 8.663613460661505
- se: 0.8798292657481263
- pvalue: 7.067261062020315e-23
- n_obs: 17
- dag_distance: 1
- conditioning_set: ['cor_kspi', 'ppop_kspi']
- shared_neighbors: []
- [ ] Decision: _________________

### 10. p=0.0003 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `imported_inflation_instrument_to_real_expenditure_negative`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.00027174315221028324
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 11. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `kzt_usd_to_imported_inflation_instrument`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.0
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 12. p=0.0031 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `rwa_kspi_to_k2_ratio_kspi`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.003091185818188935
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 13. p=0.0015 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `cpi_nontradable_to_cpi_headline`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.0014589663178228855
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 14. p=0.0231 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `kzt_usd_to_cpi_nontradable`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.023070493914662118
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 15. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `real_expenditure_to_ppop_kspi`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.0
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 16. p=0.0293 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `real_income_to_real_expenditure`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.029338345743874693
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 17. p=0.0000 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `cor_kspi_to_total_capital_kspi`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 4.229850937660592e-19
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 18. p=0.0000 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `ppop_kspi_to_total_capital_kspi`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 2.2243652090044065e-16
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 19. p=0.0492 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `e_import_share_to_imported_inflation_instrument`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.04922340067486691
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 20. p=0.0000 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `cpi_tradable_to_cpi_headline`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 3.334696254595197e-69
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 21. p=0.0000 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `nominal_income_to_real_income`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.1680567298440232e-248
- claim_level: BLOCKED_ID
- [ ] Decision: _________________
