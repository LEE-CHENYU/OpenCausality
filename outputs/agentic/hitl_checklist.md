# Human-in-the-Loop Checklist
Run ID: f92ae9db

> This checklist contains issues that require your expert judgment before
> the pipeline can proceed. For each item, review the context, understand
> why the issue matters, and record your decision. Your choices are logged
> in the audit trail for reproducibility.

## Pending Decisions

### 1. Regime Instability Decision
- **Edge:** `vix_shock_to_deposit_cost_kspi`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 2. Regime Instability Decision
- **Edge:** `cpi_headline_to_real_expenditure`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 3. p=0.0001 but claim_level=. Significance does not establish causation.
- **Edge:** `fx_to_cpi_tradable`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.0001
- claim_level: 
- [ ] Decision: _________________

### 4. p=0.0031 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `rwa_kspi_to_k2_ratio_kspi`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.003091185818188935
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 5. p=0.0015 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `cpi_nontradable_to_cpi_headline`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.0014589663178228855
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 6. p=0.0000 but claim_level=. Significance does not establish causation.
- **Edge:** `shock_to_npl_kspi_annual`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.510133236009147e-25
- claim_level: 
- [ ] Decision: _________________

### 7. p=0.0029 but claim_level=. Significance does not establish causation.
- **Edge:** `shock_to_npl_sector`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.0029314539629232694
- claim_level: 
- [ ] Decision: _________________

### 8. Leave-one-out shows sign flip or >50% magnitude change.
- **Edge:** `shock_to_npl_sector`
- **Why this matters:** Leave-one-out (LOO) analysis drops each unit in turn and re-estimates. If the coefficient flips sign or changes magnitude by more than 50%, the result is driven by a single influential unit rather than a systematic pattern. Such fragility undermines confidence in the estimate as a general causal effect.

- **Decision guidance:** Identify the influential unit and investigate whether it is an outlier or represents a genuine subgroup effect. Consider winsorizing, trimming, or reporting results with and without the influential unit.

- loo_message: Sign flipped when excluding: kaspi
- [ ] Decision: _________________

### 9. p=0.0000 but claim_level=. Significance does not establish causation.
- **Edge:** `shock_to_cor_kspi_annual`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 2.2841954119670294e-15
- claim_level: 
- [ ] Decision: _________________

### 10. p=0.0003 but claim_level=. Significance does not establish causation.
- **Edge:** `cpi_to_nominal_income`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.0003
- claim_level: 
- [ ] Decision: _________________

### 11. p=0.0231 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `kzt_usd_to_cpi_nontradable`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.023070493914662118
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 12. p=0.0000 but claim_level=. Significance does not establish causation.
- **Edge:** `portfolio_mix_to_rwa`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 4.157223824460209e-06
- claim_level: 
- [ ] Decision: _________________

### 13. p=0.0000 but claim_level=. Significance does not establish causation.
- **Edge:** `expenditure_to_payments_revenue`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 2.615043739933785e-06
- claim_level: 
- [ ] Decision: _________________

### 14. p=0.0029 but claim_level=. Significance does not establish causation.
- **Edge:** `nbk_rate_to_cor`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.002858237880200499
- claim_level: 
- [ ] Decision: _________________

### 15. p=0.0000 but claim_level=. Significance does not establish causation.
- **Edge:** `shock_to_cor_kspi`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 3.565189467439174e-31
- claim_level: 
- [ ] Decision: _________________

### 16. p=0.0000 but claim_level=. Significance does not establish causation.
- **Edge:** `shock_to_npl_kspi`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.5548712171304386e-12
- claim_level: 
- [ ] Decision: _________________

### 17. p=0.0000 but claim_level=. Significance does not establish causation.
- **Edge:** `oil_supply_to_brent`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.2226952161375292e-21
- claim_level: 
- [ ] Decision: _________________

### 18. p=0.0000 but claim_level=. Significance does not establish causation.
- **Edge:** `shock_to_cor_sector`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 4.747533899274181e-05
- claim_level: 
- [ ] Decision: _________________

### 19. Leave-one-out shows sign flip or >50% magnitude change.
- **Edge:** `shock_to_cor_sector`
- **Why this matters:** Leave-one-out (LOO) analysis drops each unit in turn and re-estimates. If the coefficient flips sign or changes magnitude by more than 50%, the result is driven by a single influential unit rather than a systematic pattern. Such fragility undermines confidence in the estimate as a general causal effect.

- **Decision guidance:** Identify the influential unit and investigate whether it is an outlier or represents a genuine subgroup effect. Consider winsorizing, trimming, or reporting results with and without the influential unit.

- loo_message: Sign flipped when excluding: halyk
- [ ] Decision: _________________

### 20. p=0.0120 but claim_level=. Significance does not establish causation.
- **Edge:** `fx_to_real_expenditure`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.012
- claim_level: 
- [ ] Decision: _________________

### 21. Leave-one-out shows sign flip or >50% magnitude change.
- **Edge:** `nbk_rate_to_cor_sector`
- **Why this matters:** Leave-one-out (LOO) analysis drops each unit in turn and re-estimates. If the coefficient flips sign or changes magnitude by more than 50%, the result is driven by a single influential unit rather than a systematic pattern. Such fragility undermines confidence in the estimate as a general causal effect.

- **Decision guidance:** Identify the influential unit and investigate whether it is an outlier or represents a genuine subgroup effect. Consider winsorizing, trimming, or reporting results with and without the influential unit.

- loo_message: Sign flipped when excluding: bcc, halyk
- [ ] Decision: _________________

### 22. p=0.0003 but claim_level=. Significance does not establish causation.
- **Edge:** `nbk_rate_to_deposit_cost`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.00031350811161777947
- claim_level: 
- [ ] Decision: _________________

### 23. p=0.0000 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `cpi_tradable_to_cpi_headline`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 3.334696254595197e-69
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 24. p=0.0000 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `nominal_income_to_real_income`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.1680567298440232e-248
- claim_level: BLOCKED_ID
- [ ] Decision: _________________
