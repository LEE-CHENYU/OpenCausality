# Human-in-the-Loop Checklist
Run ID: b10d97a0

> This checklist contains issues that require your expert judgment before
> the pipeline can proceed. For each item, review the context, understand
> why the issue matters, and record your decision. Your choices are logged
> in the audit trail for reproducibility.

## Pending Decisions

### 1. Regime Instability Decision
- **Edge:** `oil_to_vix`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 2. Regime Instability Decision
- **Edge:** `oil_to_cpi_headline`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 3. p=0.0075 but claim_level=DESCRIPTIVE. Significance does not establish causation.
- **Edge:** `fed_funds_to_unemployment`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.007520998990305605
- claim_level: DESCRIPTIVE
- [ ] Decision: _________________

### 4. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `oil_shock_to_oil_price`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 9.177374660649726e-21
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 5. p=0.0007 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `oil_to_vix`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.0007005190637644158
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 6. p=0.0098 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `oil_to_cpi_core`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.009844530986099033
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 7. p=0.0009 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `vix_to_nfci`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.000906864878427209
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 8. p=0.0235 but claim_level=DESCRIPTIVE. Significance does not establish causation.
- **Edge:** `industrial_to_unemployment`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.02346412802845881
- claim_level: DESCRIPTIVE
- [ ] Decision: _________________
