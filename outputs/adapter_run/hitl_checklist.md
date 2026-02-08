# Human-in-the-Loop Checklist
Run ID: da7532bc

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

### 4. p=0.0000 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `portfolio_mix_to_rwa`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 4.157223824460209e-06
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 5. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `expenditure_to_payments_revenue`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 2.615043739933785e-06
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 6. p=0.0029 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `nbk_rate_to_cor`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.002858237880200499
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 7. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `shock_to_cor_kspi`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 3.565189467439174e-31
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 8. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `shock_to_npl_kspi`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.5548712171304386e-12
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 9. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `oil_supply_to_brent`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.2226952161375292e-21
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 10. p=0.0003 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `nbk_rate_to_deposit_cost`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.00031350811161777947
- claim_level: BLOCKED_ID
- [ ] Decision: _________________
