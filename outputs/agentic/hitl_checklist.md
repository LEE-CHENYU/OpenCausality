# Human-in-the-Loop Checklist
Run ID: 7679191c

> This checklist contains issues that require your expert judgment before
> the pipeline can proceed. For each item, review the context, understand
> why the issue matters, and record your decision. Your choices are logged
> in the audit trail for reproducibility.

## Pending Decisions

### 1. Regime Instability Decision
- **Edge:** `oil_supply_to_fx`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 2. Regime Instability Decision
- **Edge:** `oil_demand_to_fx`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 3. Regime Instability Decision
- **Edge:** `fx_to_nbk_rate`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 4. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `loan_portfolio_to_rwa`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 8.05409884280449e-12
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 5. p=0.0000 but claim_level=. Significance does not establish causation.
- **Edge:** `shock_to_npl_kspi_annual`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.5101332360091475e-25
- claim_level: 
- [ ] Decision: _________________

### 6. p=0.0029 but claim_level=. Significance does not establish causation.
- **Edge:** `shock_to_npl_sector`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.0029314539629232694
- claim_level: 
- [ ] Decision: _________________

### 7. Leave-one-out shows sign flip or >50% magnitude change.
- **Edge:** `shock_to_npl_sector`
- **Why this matters:** Leave-one-out (LOO) analysis drops each unit in turn and re-estimates. If the coefficient flips sign or changes magnitude by more than 50%, the result is driven by a single influential unit rather than a systematic pattern. Such fragility undermines confidence in the estimate as a general causal effect.

- **Decision guidance:** Identify the influential unit and investigate whether it is an outlier or represents a genuine subgroup effect. Consider winsorizing, trimming, or reporting results with and without the influential unit.

- loo_message: Sign flipped when excluding: kaspi
- [ ] Decision: _________________

### 8. p=0.0000 but claim_level=. Significance does not establish causation.
- **Edge:** `shock_to_cor_kspi_annual`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 2.2841954119676474e-15
- claim_level: 
- [ ] Decision: _________________

### 9. p=0.0030 but claim_level=. Significance does not establish causation.
- **Edge:** `oil_supply_to_income`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.003
- claim_level: 
- [ ] Decision: _________________

### 10. p=0.0000 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `cor_to_capital`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 4.229850937660592e-19
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 11. p=0.0030 but claim_level=. Significance does not establish causation.
- **Edge:** `global_activity_to_income`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.003
- claim_level: 
- [ ] Decision: _________________

### 12. p=0.0001 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `portfolio_mix_to_rwa`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.00013991257392659715
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 13. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `expenditure_to_payments_revenue`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.0
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
- **Edge:** `shock_to_cor_kspi`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 3.565189467439174e-31
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 16. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `shock_to_npl_kspi`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.5548712171304386e-12
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 17. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `oil_supply_to_brent`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.2226952161375292e-21
- claim_level: REDUCED_FORM
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

### 20. Leave-one-out shows sign flip or >50% magnitude change.
- **Edge:** `nbk_rate_to_cor_sector`
- **Why this matters:** Leave-one-out (LOO) analysis drops each unit in turn and re-estimates. If the coefficient flips sign or changes magnitude by more than 50%, the result is driven by a single influential unit rather than a systematic pattern. Such fragility undermines confidence in the estimate as a general causal effect.

- **Decision guidance:** Identify the influential unit and investigate whether it is an outlier or represents a genuine subgroup effect. Consider winsorizing, trimming, or reporting results with and without the influential unit.

- loo_message: Sign flipped when excluding: bcc, halyk
- [ ] Decision: _________________

### 21. p=0.0003 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `nbk_rate_to_deposit_cost`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.00031350811161777947
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 22. p=0.0030 but claim_level=. Significance does not establish causation.
- **Edge:** `oil_demand_to_income`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.003
- claim_level: 
- [ ] Decision: _________________
