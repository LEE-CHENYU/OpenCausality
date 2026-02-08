# Edge Card Comparison Report

**Generated:** 2026-02-07T23:13:03.664686
**Run A (baseline):** `outputs/manual_baseline`
**Run B (new):** `outputs/adapter_run`
**Run A edges:** 26
**Run B edges:** 20


## 1. Estimate Validation

Estimates should be **identical** across runs (same data + code).
Any divergence indicates a potential bug.

**PASS: All estimates match across runs.**


## 2. Credibility Rating Changes

8 edge(s) with changed credibility:

| Edge | Rating A | Rating B | Score A | Score B |
|------|----------|----------|---------|---------|
| cpi_to_nbk_rate | A | B | 0.824 | 0.771 |
| expenditure_to_payments_revenue | B | B | 0.692 | 0.622 |
| fx_to_nbk_rate | A | B | 0.904 | 0.755 |
| nbk_rate_to_deposit_cost | B | B | 0.730 | 0.757 |
| oil_demand_to_fx | A | A | 0.970 | 0.866 |
| oil_supply_to_fx | A | A | 0.970 | 0.821 |
| portfolio_mix_to_rwa | B | B | 0.798 | 0.789 |
| vix_to_fx | A | A | 0.890 | 0.881 |


## 3. Diagnostic Changes

16 edge(s) with diagnostic changes:

| Edge | Diagnostic | Status | Details |
|------|-----------|--------|---------|
| cpi_to_nbk_rate | r_squared_h0 | REMOVED | was_passed=True |
| cpi_to_nbk_rate | residual_ac1 | REMOVED | was_passed=True |
| cpi_to_nbk_rate | ts_hac_sensitivity | NEW | passed=True |
| cpi_to_nbk_rate | ts_lag_sensitivity | NEW | passed=False |
| cpi_to_nbk_rate | ts_leads_test | NEW | passed=False |
| cpi_to_nbk_rate | ts_regime_stability | NEW | passed=True |
| cpi_to_nbk_rate | ts_residual_autocorr | NEW | passed=True |
| cpi_to_nbk_rate | ts_shock_support | NEW | passed=True |
| cpi_to_nominal_income | effective_obs | REMOVED | was_passed=True |
| expenditure_to_payments_revenue | r_squared_h0 | REMOVED | was_passed=True |
| expenditure_to_payments_revenue | residual_ac1 | REMOVED | was_passed=True |
| expenditure_to_payments_revenue | ts_hac_sensitivity | NEW | passed=True |
| expenditure_to_payments_revenue | ts_lag_sensitivity | NEW | passed=False |
| expenditure_to_payments_revenue | ts_regime_stability | NEW | passed=True |
| expenditure_to_payments_revenue | ts_residual_autocorr | NEW | passed=True |
| expenditure_to_payments_revenue | ts_shock_support | NEW | passed=False |
| fx_to_cpi_nontradable | effective_obs | REMOVED | was_passed=True |
| fx_to_cpi_tradable | effective_obs | REMOVED | was_passed=True |
| fx_to_nbk_rate | r_squared_h0 | REMOVED | was_passed=True |
| fx_to_nbk_rate | residual_ac1 | REMOVED | was_passed=True |
| fx_to_nbk_rate | ts_hac_sensitivity | NEW | passed=True |
| fx_to_nbk_rate | ts_lag_sensitivity | NEW | passed=False |
| fx_to_nbk_rate | ts_leads_test | NEW | passed=True |
| fx_to_nbk_rate | ts_regime_stability | NEW | passed=False |
| fx_to_nbk_rate | ts_residual_autocorr | NEW | passed=True |
| fx_to_nbk_rate | ts_shock_support | NEW | passed=True |
| fx_to_real_expenditure | effective_obs | REMOVED | was_passed=True |
| nbk_rate_to_cor | r_squared_h0 | REMOVED | was_passed=True |
| nbk_rate_to_cor | residual_ac1 | REMOVED | was_passed=True |
| nbk_rate_to_cor | ts_hac_sensitivity | NEW | passed=True |
| nbk_rate_to_cor | ts_lag_sensitivity | NEW | passed=True |
| nbk_rate_to_cor | ts_leads_test | NEW | passed=False |
| nbk_rate_to_cor | ts_regime_stability | NEW | passed=True |
| nbk_rate_to_cor | ts_residual_autocorr | NEW | passed=True |
| nbk_rate_to_cor | ts_shock_support | NEW | passed=True |
| nbk_rate_to_deposit_cost | r_squared_h0 | REMOVED | was_passed=True |
| nbk_rate_to_deposit_cost | residual_ac1 | REMOVED | was_passed=False |
| nbk_rate_to_deposit_cost | ts_hac_sensitivity | NEW | passed=True |
| nbk_rate_to_deposit_cost | ts_lag_sensitivity | NEW | passed=True |
| nbk_rate_to_deposit_cost | ts_leads_test | NEW | passed=False |
| nbk_rate_to_deposit_cost | ts_regime_stability | NEW | passed=True |
| nbk_rate_to_deposit_cost | ts_residual_autocorr | NEW | passed=False |
| nbk_rate_to_deposit_cost | ts_shock_support | NEW | passed=True |
| oil_demand_to_fx | r_squared_h0 | REMOVED | was_passed=True |
| oil_demand_to_fx | residual_ac1 | REMOVED | was_passed=True |
| oil_demand_to_fx | ts_hac_sensitivity | NEW | passed=True |
| oil_demand_to_fx | ts_lag_sensitivity | NEW | passed=True |
| oil_demand_to_fx | ts_leads_test | NEW | passed=True |
| oil_demand_to_fx | ts_regime_stability | NEW | passed=False |
| oil_demand_to_fx | ts_residual_autocorr | NEW | passed=True |
| oil_demand_to_fx | ts_shock_support | NEW | passed=True |
| oil_supply_to_brent | r_squared_h0 | REMOVED | was_passed=True |
| oil_supply_to_brent | residual_ac1 | REMOVED | was_passed=True |
| oil_supply_to_brent | ts_hac_sensitivity | NEW | passed=True |
| oil_supply_to_brent | ts_lag_sensitivity | NEW | passed=True |
| oil_supply_to_brent | ts_leads_test | NEW | passed=True |
| oil_supply_to_brent | ts_regime_stability | NEW | passed=True |
| oil_supply_to_brent | ts_residual_autocorr | NEW | passed=True |
| oil_supply_to_brent | ts_shock_support | NEW | passed=True |
| oil_supply_to_fx | r_squared_h0 | REMOVED | was_passed=True |
| oil_supply_to_fx | residual_ac1 | REMOVED | was_passed=True |
| oil_supply_to_fx | ts_hac_sensitivity | NEW | passed=True |
| oil_supply_to_fx | ts_lag_sensitivity | NEW | passed=False |
| oil_supply_to_fx | ts_leads_test | NEW | passed=True |
| oil_supply_to_fx | ts_regime_stability | NEW | passed=False |
| oil_supply_to_fx | ts_residual_autocorr | NEW | passed=True |
| oil_supply_to_fx | ts_shock_support | NEW | passed=True |
| portfolio_mix_to_rwa | r_squared_h0 | REMOVED | was_passed=True |
| portfolio_mix_to_rwa | residual_ac1 | REMOVED | was_passed=True |
| portfolio_mix_to_rwa | ts_hac_sensitivity | NEW | passed=True |
| portfolio_mix_to_rwa | ts_lag_sensitivity | NEW | passed=True |
| portfolio_mix_to_rwa | ts_leads_test | NEW | passed=False |
| portfolio_mix_to_rwa | ts_regime_stability | NEW | passed=True |
| portfolio_mix_to_rwa | ts_residual_autocorr | NEW | passed=True |
| portfolio_mix_to_rwa | ts_shock_support | NEW | passed=True |
| shock_to_cor_kspi | r_squared_h0 | REMOVED | was_passed=True |
| shock_to_cor_kspi | residual_ac1 | REMOVED | was_passed=True |
| shock_to_cor_kspi | ts_hac_sensitivity | NEW | passed=True |
| shock_to_cor_kspi | ts_lag_sensitivity | NEW | passed=True |
| shock_to_cor_kspi | ts_leads_test | NEW | passed=True |
| shock_to_cor_kspi | ts_regime_stability | NEW | passed=True |
| shock_to_cor_kspi | ts_residual_autocorr | NEW | passed=True |
| shock_to_cor_kspi | ts_shock_support | NEW | passed=True |
| shock_to_npl_kspi | r_squared_h0 | REMOVED | was_passed=True |
| shock_to_npl_kspi | residual_ac1 | REMOVED | was_passed=True |
| shock_to_npl_kspi | ts_hac_sensitivity | NEW | passed=True |
| shock_to_npl_kspi | ts_lag_sensitivity | NEW | passed=True |
| shock_to_npl_kspi | ts_leads_test | NEW | passed=True |
| shock_to_npl_kspi | ts_regime_stability | NEW | passed=True |
| shock_to_npl_kspi | ts_residual_autocorr | NEW | passed=True |
| shock_to_npl_kspi | ts_shock_support | NEW | passed=True |
| vix_to_fx | r_squared_h0 | REMOVED | was_passed=True |
| vix_to_fx | residual_ac1 | REMOVED | was_passed=True |
| vix_to_fx | ts_hac_sensitivity | NEW | passed=True |
| vix_to_fx | ts_lag_sensitivity | NEW | passed=False |
| vix_to_fx | ts_leads_test | NEW | passed=True |
| vix_to_fx | ts_regime_stability | NEW | passed=True |
| vix_to_fx | ts_residual_autocorr | NEW | passed=True |
| vix_to_fx | ts_shock_support | NEW | passed=True |


## 4. New Agentic Fields

Fields only present in Run B (identification, counterfactual_block, propagation_role).


### 4.1 Identification Blocks

**20 edges** with identification blocks.

**Claim level distribution:**

- BLOCKED_ID: 4
- IDENTIFIED_CAUSAL: 8
- REDUCED_FORM: 8

| Edge | Claim Level | High Risks | Threats Failed |
|------|------------|------------|----------------|
| capital_to_k2 | IDENTIFIED_CAUSAL | - | - |
| cor_to_capital | IDENTIFIED_CAUSAL | - | - |
| cpi_to_nbk_rate | BLOCKED_ID | - | sign_consistency, ts_leads_test, ts_lag_sensitivity |
| cpi_to_nominal_income | IDENTIFIED_CAUSAL | - | - |
| expenditure_to_payments_revenue | REDUCED_FORM | - | effective_obs, ts_lag_sensitivity, ts_shock_support |
| fx_to_cpi_nontradable | IDENTIFIED_CAUSAL | - | - |
| fx_to_cpi_tradable | IDENTIFIED_CAUSAL | - | - |
| fx_to_nbk_rate | REDUCED_FORM | - | ts_lag_sensitivity, ts_regime_stability |
| fx_to_real_expenditure | IDENTIFIED_CAUSAL | - | - |
| loan_portfolio_to_rwa | IDENTIFIED_CAUSAL | - | - |
| nbk_rate_to_cor | BLOCKED_ID | - | effective_obs, ts_leads_test |
| nbk_rate_to_deposit_cost | BLOCKED_ID | - | effective_obs, ts_leads_test, ts_residual_autocorr |
| oil_demand_to_fx | REDUCED_FORM | - | ts_regime_stability |
| oil_supply_to_brent | REDUCED_FORM | - | - |
| oil_supply_to_fx | REDUCED_FORM | - | ts_lag_sensitivity, ts_regime_stability |
| portfolio_mix_to_rwa | BLOCKED_ID | - | effective_obs, ts_leads_test |
| rwa_to_k2 | IDENTIFIED_CAUSAL | - | - |
| shock_to_cor_kspi | REDUCED_FORM | - | - |
| shock_to_npl_kspi | REDUCED_FORM | - | - |
| vix_to_fx | REDUCED_FORM | - | sign_consistency, ts_lag_sensitivity |


### 4.2 Counterfactual Status

**20 edges** with counterfactual assessment: 8 allowed, 12 blocked.

**Blocked edges:**

| Edge | Reason Blocked |
|------|---------------|
| cpi_to_nbk_rate | Shock CF requires REDUCED_FORM+, edge has BLOCKED_ID |
| expenditure_to_payments_revenue | Counterfactual Use: BLOCKED
Reason: insufficient shock episodes

Even if p<0.05, this does not establish a causal effect. |
| fx_to_nbk_rate | Counterfactual Use: BLOCKED
Reason: regime break risk

Even if p<0.05, this does not establish a causal effect. |
| nbk_rate_to_cor | Shock CF requires REDUCED_FORM+, edge has BLOCKED_ID |
| nbk_rate_to_deposit_cost | Shock CF requires REDUCED_FORM+, edge has BLOCKED_ID |
| oil_demand_to_fx | Counterfactual Use: BLOCKED
Reason: regime break risk

Even if p<0.05, this does not establish a causal effect. |
| oil_supply_to_brent | Design LOCAL_PROJECTIONS provides REDUCED_FORM level only |
| oil_supply_to_fx | Counterfactual Use: BLOCKED
Reason: regime break risk

Even if p<0.05, this does not establish a causal effect. |
| portfolio_mix_to_rwa | Shock CF requires REDUCED_FORM+, edge has BLOCKED_ID |
| shock_to_cor_kspi | Design LOCAL_PROJECTIONS provides REDUCED_FORM level only |
| shock_to_npl_kspi | Design LOCAL_PROJECTIONS provides REDUCED_FORM level only |
| vix_to_fx | Design LOCAL_PROJECTIONS provides REDUCED_FORM level only |


### 4.3 Propagation Roles

**20 edges** with propagation roles.

**Role distribution:**

- bridge: 2
- diagnostic_only: 5
- identity: 2
- reduced_form: 7
- structural: 4


## 5. Edge Coverage Differences

**6 edge(s) only in Run A:**

- `nbk_rate_to_cor_sector`
- `nbk_rate_to_deposit_cost_sector`
- `shock_to_cor_kspi_annual`
- `shock_to_cor_sector`
- `shock_to_npl_kspi_annual`
- `shock_to_npl_sector`


## 6. Summary Assessment

### Correctness
**VALIDATED**: All estimates are identical across runs. The agentic modules do not alter core estimation results.

### Agentic Value Added
- **Identification screening**: 20 edges assessed, claim levels assigned (BLOCKED_ID=4, IDENTIFIED_CAUSAL=8, REDUCED_FORM=8)
- **Counterfactual gating**: 12 edges blocked from counterfactual use
- **Credibility recalibration**: 8 edges with adjusted ratings
- **New diagnostics**: 71 additional checks

---
*Report generated by `compare_runs.py` at 2026-02-07T23:13:03.664955*