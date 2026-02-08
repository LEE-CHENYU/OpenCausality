# Human-in-the-Loop Checklist
Run ID: 002a5e7a

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
- **Edge:** `vix_shock__to__cor_kspi`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 3. Regime Instability Decision
- **Edge:** `cpi_tradable->real_expenditure`
- **Suggested:** Split estimand by regime or restrict counterfactual scope
- [ ] Decision: _________________

### 4. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `expenditure_to_payments_revenue`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 2.615043739933785e-06
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 5. p=0.0000 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `ppop_kspi_and_provisions_expense_kspi_to_total_capital_kspi`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 2.2243652090044065e-16
- claim_level: BLOCKED_ID
- [ ] Decision: _________________
