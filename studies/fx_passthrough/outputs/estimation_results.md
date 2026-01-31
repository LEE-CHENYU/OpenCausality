# Kazakhstan FX Passthrough Study - Estimation Results

**Generated:** January 31, 2026
**Study Version:** 2.1
**Status:** PARTIAL ANALYSIS - Headline CPI Only

---

## Executive Summary

This document presents results from the FX passthrough study using available data. **The CPI category data required for the main causal chain analysis (Block A) is currently unavailable** due to a BNS API error.

### Key Findings

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **3-Month Pass-Through** | 9.8% | 10% FX depreciation → ~1pp extra inflation |
| **Peak Correlation Lag** | 1 month | Inflation responds within 1-2 months |
| **Post-Float Volatility** | 4.38% | Nearly double pre-float (2.28%) |
| **Model R²** | 0.186 | FX explains ~19% of inflation variation |

---

## Data Availability

| Data Source | Status | Observations | Date Range |
|-------------|--------|--------------|------------|
| NBK USD/KZT | **Available** | 192 monthly | 2010-01 to 2025-12 |
| BNS National Income | **Available** | 61 quarters | 2010-Q2 to 2025-Q3 |
| BNS Headline CPI | **Available** | 178 monthly | 2011-01 to 2025-10 |
| **BNS CPI Categories** | **API Error** | - | - |

### BNS API Issue

The Kazakhstan Bureau of National Statistics API is not serving COICOP category-level CPI data:
- Only aggregate "Goods and services" CPI available
- Individual category indices (Food, Clothing, Housing, etc.) not accessible
- **Manual download required** from: https://stat.gov.kz/en/industries/prices/stat-official-ind-prices/

---

## Exchange Rate Analysis

### Regime Change: August 2015 Tenge Float

The National Bank of Kazakhstan abandoned the managed exchange rate regime on **August 20, 2015**, allowing the tenge to float freely.

| Metric | Pre-Float (2010-2015.07) | Post-Float (2015.08-2025) | Change |
|--------|--------------------------|---------------------------|--------|
| Mean Rate (KZT/USD) | 158.2 | 409.2 | +159% |
| Std Deviation | 15.6 | 67.3 | +332% |
| Monthly Volatility | 2.28% | 4.38% | **+92%** |

**Key Finding:** Exchange rate volatility nearly doubled after the float, creating the variation needed to identify pass-through effects.

### Major Depreciation Events

| Date | Monthly Change | Rate After |
|------|----------------|------------|
| 2015-09 | +26.8% | 237.7 KZT/USD |
| 2014-03 | +18.4% | 184.1 KZT/USD |
| 2020-04 | +17.5% | 447.7 KZT/USD |
| 2022-03 | +14.2% | 495.0 KZT/USD |
| 2015-10 | +14.0% | 270.9 KZT/USD |
| 2015-12 | +10.1% | 307.4 KZT/USD |
| 2016-01 | +10.6% | 340.0 KZT/USD |
| 2022-07 | +13.4% | 470.3 KZT/USD |

These large depreciation events (8 episodes >10% monthly) provide sharp identification for pass-through estimation.

---

## Reduced-Form Pass-Through Estimation

### FX-Inflation Correlation Structure

| Lag (months) | Correlation | Interpretation |
|--------------|-------------|----------------|
| 0 (contemp.) | 0.289 | Immediate partial response |
| 1 | **0.386** | Peak response |
| 2 | 0.195 | Declining |
| 3 | 0.154 | Still positive |
| 6 | 0.015 | Near zero |
| 12 | -0.089 | Reversed |

**Finding:** FX pass-through to CPI peaks at 1-month lag and fades within 3-6 months.

### OLS Pass-Through Regression

**Model:** π_t = α + β₁·ΔFX_{t-1} + β₂·ΔFX_{t-2} + β₃·ΔFX_{t-3} + ε_t

| Variable | Coefficient | Std Error | t-stat | p-value |
|----------|-------------|-----------|--------|---------|
| Intercept | 0.0061 | 0.0005 | 13.50 | 0.000*** |
| FX_lag1 | 0.0587 | 0.0111 | 5.28 | 0.000*** |
| FX_lag2 | 0.0188 | 0.0113 | 1.67 | 0.096* |
| FX_lag3 | 0.0208 | 0.0112 | 1.87 | 0.064* |

- **Observations:** 175
- **R-squared:** 0.186
- **Cumulative 3-month pass-through:** 0.098

### Interpretation

- A **10% tenge depreciation** leads to approximately **1 percentage point** additional monthly inflation over the following 3 months
- Most of the pass-through occurs in the first month (β₁ = 0.059)
- The pass-through is **incomplete** (only ~10% of FX shock passes to CPI)
- This is consistent with incomplete exchange rate pass-through literature for emerging markets

---

## Income Dynamics

### National Income Time Series

- **Period:** 2010-Q2 to 2025-Q3 (61 quarters)
- **Mean per capita income:** ~100,000 KZT/quarter
- **Log-income range:** 10.55 to 12.43

### FX-Income Correlation

| Relationship | Correlation |
|--------------|-------------|
| FX depreciation vs Income growth (contemp.) | -0.078 |
| FX depreciation(t-1) vs Income growth(t) | +0.198 |

**Interpretation:**
- Weak contemporaneous negative correlation (depreciation slightly hurts nominal income growth)
- Positive lagged correlation suggests delayed adjustment (nominal incomes catch up after depreciation)

---

## Causal Chain Status

### Block A: CPI Pass-Through by Category
- **Status:** NOT RUN
- **Reason:** Requires CPI data by COICOP division (12 categories)
- **Blocked Analysis:** Cannot estimate differential pass-through by import intensity

### Block B: Income Response (LP-IV)
- **Status:** REDUCED-FORM ONLY
- **Available:** Correlation analysis with headline CPI
- **Blocked:** IV estimation requires "imported inflation" from Block A

### Block C-E: Downstream Analysis
- **Status:** NOT RUN
- **Reason:** Depends on Block A/B outputs

### Falsification Tests
- **Admin Prices Test:** NOT RUN (requires COICOP categories)
- **Pre-Trends Test:** NOT RUN (requires COICOP categories)

---

## Robustness Notes

### Limitations of Reduced-Form Analysis

1. **Endogeneity:** OLS estimates may be biased if FX shocks are correlated with other inflation drivers
2. **No Identification Strategy:** Without COICOP categories, we cannot use the high-import-intensity vs low-import-intensity comparison
3. **Missing Falsification:** Cannot verify that admin prices (utilities, education) show zero pass-through

### What Category Data Would Enable

With 12 COICOP categories, we could:
1. **Identify import-intensive categories** (food, clothing, transport) vs domestic (housing services, education)
2. **Run difference-in-differences:** Compare inflation response to FX shocks across category types
3. **Falsification test:** Verify zero pass-through for regulated prices
4. **Construct "imported inflation" instrument** for LP-IV in Block B

---

## Alternative CPI Data Sources

If BNS API remains unavailable:

| Source | Data | Issue |
|--------|------|-------|
| [BNS Open Data](https://stat.gov.kz/en/industries/prices/stat-official-ind-prices/) | Manual download | Need to navigate site |
| IMF IFS | Monthly CPI | API timeout in testing |
| FRED | Headline only | No category breakdown |
| World Bank | Annual only | Insufficient frequency |

**Recommended:** Manual download from BNS website for category-level indices.

---

## Replication Commands

```bash
# Check data status
PYTHONPATH=. python scripts/download_all_data.py --status

# Run reduced-form analysis
PYTHONPATH=. python -c "
import pandas as pd
import numpy as np

fx = pd.read_parquet('data/raw/nbk/usd_kzt.parquet')
cpi = pd.read_parquet('data/processed/fx_passthrough/headline_cpi.parquet')
income = pd.read_parquet('data/processed/fx_passthrough/income_series.parquet')

print(f'FX: {len(fx)} obs')
print(f'CPI: {len(cpi)} obs')
print(f'Income: {len(income)} obs')
"

# Once CPI categories available, run full chain
kzresearch passthrough run-full-chain
```

---

## Summary Statistics

### Exchange Rate (USD/KZT)

| Statistic | Full Sample | Pre-Float | Post-Float |
|-----------|-------------|-----------|------------|
| Mean | 299.9 | 158.2 | 409.2 |
| Std Dev | 120.2 | 15.6 | 67.3 |
| Min | 134.3 | 134.3 | 185.1 |
| Max | 524.3 | 188.4 | 524.3 |

### Monthly Inflation (CPI MoM)

| Statistic | Value |
|-----------|-------|
| Mean | 0.61% |
| Std Dev | 0.72% |
| Min | -0.30% |
| Max | 3.15% |

---

*Last Updated: January 31, 2026*
*Status: Partial analysis complete; awaiting CPI category data for full causal chain*
*Kazakhstan Econometric Research Platform*
