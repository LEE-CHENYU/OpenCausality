# Human-in-the-Loop Checklist
Run ID: c8d49111

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

### 4. Null link rwa_kspi->loan_portfolio_kspi: significant partial association (beta=2.0225, p=0.0000, N=27). DAG may need this edge.
- from_node: rwa_kspi
- to_node: loan_portfolio_kspi
- coefficient: 2.0224836469627254
- se: 0.11632127004166029
- pvalue: 1.0342361089550774e-67
- n_obs: 27
- dag_distance: 1
- conditioning_set: []
- shared_neighbors: []
- [ ] Decision: _________________

### 5. Null link rwa_kspi->portfolio_mix_kspi: significant partial association (beta=0.0000, p=0.0000, N=17). DAG may need this edge.
- from_node: rwa_kspi
- to_node: portfolio_mix_kspi
- coefficient: 2.392473982683388e-05
- se: 1.2454046642916173e-06
- pvalue: 3.028384729634209e-82
- n_obs: 17
- dag_distance: 1
- conditioning_set: []
- shared_neighbors: []
- [ ] Decision: _________________

### 6. Null link k2_ratio_kspi->rwa_kspi | loan_portfolio_kspi, portfolio_mix_kspi: significant partial association (beta=-48.2588, p=0.0000, N=17). DAG may need this edge.
- from_node: k2_ratio_kspi
- to_node: rwa_kspi
- coefficient: -48.25878605070019
- se: 5.195803640362877
- pvalue: 1.571685981726206e-20
- n_obs: 17
- dag_distance: 1
- conditioning_set: ['loan_portfolio_kspi', 'portfolio_mix_kspi']
- shared_neighbors: []
- [ ] Decision: _________________

### 7. p=0.0000 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `portfolio_mix_to_rwa`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 4.157223824460209e-06
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 8. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `expenditure_to_payments_revenue`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 2.615043739933785e-06
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 9. p=0.0029 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `nbk_rate_to_cor`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.002858237880200499
- claim_level: BLOCKED_ID
- [ ] Decision: _________________

### 10. p=0.0000 but claim_level=REDUCED_FORM. Significance does not establish causation.
- **Edge:** `oil_supply_to_brent`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 1.2226952161375292e-21
- claim_level: REDUCED_FORM
- [ ] Decision: _________________

### 11. p=0.0003 but claim_level=BLOCKED_ID. Significance does not establish causation.
- **Edge:** `nbk_rate_to_deposit_cost`
- **Why this matters:** This edge shows a statistically significant result (low p-value), but the identification strategy has not been validated as causal. A significant correlation is not the same as a causal effect — without proper identification (e.g., IV, RDD, DiD), the estimate may reflect reverse causation, omitted variable bias, or spurious correlation. Accepting this as causal without acknowledgement constitutes overclaiming.

- **Decision guidance:** If you have a credible identification strategy, upgrade to IDENTIFIED_CAUSAL and document it. Otherwise, accept as REDUCED_FORM — informative but not usable for counterfactual predictions without caveats.

- pvalue: 0.00031350811161777947
- claim_level: BLOCKED_ID
- [ ] Decision: _________________
