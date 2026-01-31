# Kazakhstan FX-Inflation Pass-Through Study

## Research Question

**How do exchange rate shocks affect household welfare through inflation, income, and expenditure channels?**

## Identification Strategy

This study uses a **CPI category DiD design** to construct an exogenous "imported inflation" instrument:

- **Exposure (s_c)**: Import intensity of CPI category c
- **Shock (ΔFX_t)**: Exchange rate change (USD/KZT)
- **Instrument (Z_t)**: Σ w_c × s_c × ΔFX_t

### Core Causal Chain

```
Exchange Rate → Inflation → (Nominal Income & Transfers) → Real Income → Expenditure
```

### Block Structure

| Block | Name | Method | Estimand |
|-------|------|--------|----------|
| A | CPI Pass-Through | Category DiD | Differential inflation by import intensity |
| B | Income Response | LP-IV | Income response to instrumented inflation |
| C | Real Income | Accounting | Decomposition: nominal - price level |
| D | Transfer Tests | IV | Transfer mechanism verification |
| E | Expenditure | LP-IV | Expenditure response to instrumented inflation |

## Identification Details

### Block A: CPI Category DiD

```
π_{c,t+h} = α_c + δ_t + Σ_{k=0}^K β_{h,k}(s_c × ΔFX_{t-k}) + ε_{c,t+h}
```

**Exogeneity Claim:** In absence of FX changes, high- vs low-import categories would not have systematically different inflation in the same month.

**Key Assumption:** s_c (import intensity) is predetermined (fixed using pre-period IO tables or trade mapping).

### Constructed Instrument

```
Z_t ≡ Σ_c w_c × s_c × ΔFX_t
```

This "imported inflation pressure" instruments headline inflation in downstream blocks.

### Blocks B, D, E: LP-IV

First Stage:
```
π_t = a + θ × Z_t + Γ(L)controls + u_t
```

Second Stage (LP):
```
ΔY_{t+h} = α_h + β_h × π̂_t + Φ_h(L)controls + ε_{t+h}
```

**Exogeneity Claim:** Imported inflation pressure Z_t affects income/expenditure only through headline inflation.

## Data Sources

| Source | Description | Coverage | Status |
|--------|-------------|----------|--------|
| NBK | USD/KZT exchange rate | 2000-present | Primary (Tier 1) |
| IMF EER | NEER/REER | Monthly | Requires auth (Tier 2) |
| World Bank | REER | Annual | Robustness (Tier 3) |
| Kazakhstan BNS | CPI by COICOP | 2010-present | Required |
| Kazakhstan BNS | National income, expenditure | 2010-present | Required |
| FRED | Global activity (IGREA) | 2000-present | Controls |

## Usage

```bash
# Fetch data
kzresearch passthrough fetch-data all

# Build panels
kzresearch passthrough build-cpi-panel
kzresearch passthrough build-income-series

# Estimate blocks
kzresearch passthrough estimate block-a
kzresearch passthrough estimate block-b
kzresearch passthrough estimate all

# Run falsification tests
kzresearch passthrough falsification

# Structural break analysis
kzresearch passthrough structural-break

# Full causal chain analysis
kzresearch passthrough run-full-chain
```

## Structural Break: August 2015

Kazakhstan's FX regime changed in August 2015 (tenge float). We report:

1. **Pre-2015** (2010Q1 - 2015Q2): Managed exchange rate regime
2. **Post-2015** (2015Q3 - present): Floating exchange rate
3. **Full sample pooled**: With regime indicator

## Falsification Tests

| Test | Block | Description | Pass Criterion |
|------|-------|-------------|----------------|
| Pre-trends | A | s_c × ΔFX_{t+k} for k < 0 | Joint p > 0.05 |
| Admin prices | A | Administered prices respond? | Insignificant |
| Placebo exposure | A | Non-tradables respond? | Insignificant |
| Permutation | A | Shuffle s_c across categories | β extreme vs dist |
| Weak IV | B | First-stage F-stat | F > 10 |
| Composition | D | Wage + transfer shares | Sum = 1 |
| Backtest | All | 2014-15, 2020 episodes | Predictions match |

## Key Assumptions

1. **Parallel trends (Block A):** Categories with different import intensity would have parallel inflation trends absent FX shocks
2. **Predetermined exposure:** Import shares s_c fixed before analysis period
3. **Exclusion restriction (Blocks B, D, E):** Imported inflation affects outcomes only through headline inflation
4. **No admin price confounds:** Excluding administered price categories removes policy-driven price changes

## Dependencies

### Shared Infrastructure

- `shared/data/exchange_rate.py` - Three-tier FX fetcher
- `shared/data/bns_cpi_categories.py` - CPI by COICOP
- `shared/data/bns_national_income.py` - Income aggregates
- `shared/data/import_intensity.py` - Category exposure
- `shared/model/small_n_inference.py` - Wild bootstrap, permutation tests

### External Packages

- `linearmodels` - Panel regression
- `statsmodels` - HAC inference
- `scipy` - Statistical tests

## Output Files

```
studies/fx_passthrough/outputs/
├── block_a_results.json       # CPI pass-through estimates
├── block_b_results.json       # Income LP-IV results
├── block_c_results.json       # Real income decomposition
├── block_d_results.json       # Transfer mechanism tests
├── block_e_results.json       # Expenditure LP-IV results
├── causal_chain_summary.json  # Full chain results
├── falsification_tests.json   # Falsification test results
└── figures/
    ├── irf_inflation.png      # Inflation impulse responses
    ├── irf_income.png         # Income impulse responses
    └── structural_break.png   # Pre/post comparison
```

## Citation

If using this study design:

```bibtex
@misc{kz_fx_passthrough,
  title={FX-to-Expenditure Causal Chain Study: Kazakhstan},
  author={Kazakhstan Econometric Research Platform},
  year={2026},
  note={Exchange rate pass-through via CPI category DiD}
}
```

## Contact

For questions about this study, please open an issue in the project repository.
