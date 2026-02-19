# Human-in-the-Loop Checklist
Run ID: 26d536df

> This checklist contains issues that require your expert judgment before
> the pipeline can proceed. For each item, review the context, understand
> why the issue matters, and record your decision. Your choices are logged
> in the audit trail for reproducibility.

## Pending Decisions

### 1. Regime Instability Decision
- **Edge:** `vix_to_deposit_cost_kspi`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 2. Regime Instability Decision
- **Edge:** `oil_supply_to_fx`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 3. Regime Instability Decision
- **Edge:** `oil_demand_to_fx`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 4. Regime Instability Decision
- **Edge:** `vix_to_fx`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 5. Regime Instability Decision
- **Edge:** `fx_to_nbk_rate`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 6. Regime Instability Decision
- **Edge:** `cpi_to_nbk_rate`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 7. Null link cpi_headline->cpi_tradable | kzt_usd: significant partial association (beta=0.9622, p=0.0000, N=178). DAG may need this edge.
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

### 8. p=0.0262 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `cpi_nontradable_to_cpi_headline`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.02617600661995132
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 9. p=0.0029 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `shock_to_npl_sector`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.0029314539629232694
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 10. Leave-one-out shows sign flip or >50% magnitude change.
- **Edge:** `shock_to_npl_sector`
- **Why this matters:** Leave-one-out (LOO) analysis drops each unit in turn and re-estimates. If the coefficient flips sign or changes magnitude by more than 50%, the result is driven by a single influential unit rather than a systematic pattern. Such fragility undermines confidence in the estimate as a general causal effect.

- **Decision guidance:** Identify the influential unit and investigate whether it is an outlier or represents a genuine subgroup effect. Consider winsorizing, trimming, or reporting results with and without the influential unit.

- loo_message: Sign flipped when excluding: kaspi
- [ ] Decision: _________________

### 11. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `ppop_to_capital`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 7.906714079184397e-10
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 12. p=0.0018 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `portfolio_mix_to_rwa`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.0018120846584064887
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 13. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `expenditure_to_payments_revenue`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.501773185983678e-49
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 14. p=0.0029 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `nbk_rate_to_cor`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.002858237880200499
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 15. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `real_expenditure_to_ppop_kspi`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.0
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 16. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `oil_supply_to_brent`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.2226952161375292e-21
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 17. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `shock_to_cor_sector`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 4.747533899274181e-05
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 18. Leave-one-out shows sign flip or >50% magnitude change.
- **Edge:** `shock_to_cor_sector`
- **Why this matters:** Leave-one-out (LOO) analysis drops each unit in turn and re-estimates. If the coefficient flips sign or changes magnitude by more than 50%, the result is driven by a single influential unit rather than a systematic pattern. Such fragility undermines confidence in the estimate as a general causal effect.

- **Decision guidance:** Identify the influential unit and investigate whether it is an outlier or represents a genuine subgroup effect. Consider winsorizing, trimming, or reporting results with and without the influential unit.

- loo_message: Sign flipped when excluding: halyk
- [ ] Decision: _________________

### 19. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `deposit_cost_to_ppop`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 2.0440107097137955e-06
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 20. Leave-one-out shows sign flip or >50% magnitude change.
- **Edge:** `nbk_rate_to_cor_sector`
- **Why this matters:** Leave-one-out (LOO) analysis drops each unit in turn and re-estimates. If the coefficient flips sign or changes magnitude by more than 50%, the result is driven by a single influential unit rather than a systematic pattern. Such fragility undermines confidence in the estimate as a general causal effect.

- **Decision guidance:** Identify the influential unit and investigate whether it is an outlier or represents a genuine subgroup effect. Consider winsorizing, trimming, or reporting results with and without the influential unit.

- loo_message: Sign flipped when excluding: bcc, halyk
- [ ] Decision: _________________

### 21. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `payments_revenue_to_ppop`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.597615974506649e-06
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 22. p=0.0098 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `oil_demand_to_fx`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.009817705422092728
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 23. p=0.0003 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `nbk_rate_to_deposit_cost`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.00031350811161777947
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 24. p=0.0000 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `nominal_income_to_real_income`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 2.260243547263154e-117
- claim_level: BLOCKED_ID
- [ ] Decision: _________________
