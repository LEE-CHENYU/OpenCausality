# Credit Quality Study: Estimation Results

## Executive Summary

This document presents **preliminary reduced-form estimates** of the relationship between external shocks and Kazakhstan's banking sector credit quality.

**Key Finding:** Both annual (N=16) and monthly (N=12) samples are too small for reliable inference. Results are exploratory and should not be used for policy conclusions without longer historical data.

**Design:** Reduced-form local projections: External Shocks → NPL Ratio

**NOT estimated:** Causal income → default elasticity (requires micro loan-level data)

---

## Data Sources

### Annual Data (2008-2023)

| Variable | Source | Frequency | Coverage | N |
|----------|--------|-----------|----------|---|
| NPL Ratio | World Bank (FB.AST.NPER.ZS) | Annual | 2008-2023 | 16 |
| Oil Supply Shock | Brent price returns (standardized) | Annual (from daily) | 2008-2023 | 16 |
| VIX Innovation | CBOE VIX (AR residuals) | Annual (from daily) | 2008-2023 | 16 |
| Global Activity | Kilian IGREA (AR residuals) | Annual (from monthly) | 2008-2023 | 16 |

### Monthly Data (2024)

| Variable | Source | Frequency | Coverage | N |
|----------|--------|-----------|----------|---|
| NPL Ratio (90+ days) | NBK Loan Portfolio Quality | Monthly | Jan-Dec 2024 | 12 |
| NPL Ratio (30+ days) | NBK Loan Portfolio Quality | Monthly | Jan-Dec 2024 | 12 |
| Oil Shock | Brent price returns (standardized) | Monthly | Jan-Dec 2024 | 12 |
| VIX Innovation | CBOE VIX changes | Monthly | Jan-Dec 2024 | 12 |

### NPL Ratio Time Series

```
Year    NPL Ratio (%)
2008    7.09
2009    21.17  ← Global Financial Crisis peak
2010    20.96
2011    20.67
2012    29.78  ← Peak (post-crisis restructuring)
2013    19.46
2014    12.37  ← Oil price collapse begins
2015    8.03
2016    6.72
2017    9.31
2018    7.39
2019    8.14
2020    8.35   ← COVID-19
2021    6.59
2022    3.36   ← Major regulatory cleanup
2023    2.89
```

**Notable patterns:**
- NPL ratio peaked at ~30% in 2012 (post-GFC legacy issues)
- Dramatic decline from 2012-2016
- Recent improvement to <3% (regulatory cleanup, write-offs)

---

## Estimation Results

### Contemporaneous Regressions (h=0)

$$NPL_t = \alpha + \beta \cdot Shock_t + \varepsilon_t$$

| Shock Variable | β | SE | t-stat | p-value | R² |
|----------------|---|----|----|---------|-----|
| Oil Supply Shock | 3.10 | 3.05 | 1.02 | 0.31 | 0.04 |
| VIX Innovation | -0.50 | 0.41 | -1.23 | 0.22 | 0.05 |
| Global Activity | 0.04 | 0.13 | 0.32 | 0.75 | 0.01 |

### Local Projection h=1 (1-year ahead)

$$NPL_{t+1} = \alpha + \beta \cdot Shock_t + \varepsilon_t$$

| Shock Variable | β | SE | t-stat | p-value |
|----------------|---|----|----|---------|
| Oil Supply Shock | 1.20 | 4.69 | 0.26 | 0.80 |
| VIX Innovation | 0.51 | 0.50 | 1.03 | 0.30 |
| Global Activity | -0.02 | 0.13 | -0.12 | 0.91 |

### Local Projection h=2 (2-years ahead)

$$NPL_{t+2} = \alpha + \beta \cdot Shock_t + \varepsilon_t$$

| Shock Variable | β | SE | t-stat | p-value |
|----------------|---|----|----|---------|
| Oil Supply Shock | 1.17 | 4.55 | 0.26 | 0.80 |
| VIX Innovation | 0.60 | 0.44 | 1.39 | 0.17 |
| Global Activity | -0.02 | 0.13 | -0.14 | 0.89 |

### Interpretation (Annual)

- **No statistically significant effects** at conventional levels
- **Signs are economically sensible** (oil shock → higher NPL at h=0)
- **Standard errors are large** relative to coefficients
- **R² values are near zero** - shocks explain little variation

---

## Monthly Analysis (2024 NBK Data)

### NPL Ratio Time Series (Monthly 2024)

```
Month       NPL 90+ (%)   NPL 30+ (%)   Brent ($)   VIX
2024-01     2.89          3.66          80.12       13.39
2024-02     3.02          3.85          83.48       13.98
2024-03     3.00          4.23          85.41       13.79
2024-04     3.07          4.11          89.94       16.14
2024-05     3.12          4.13          81.75       13.06
2024-06     3.20          4.28          82.25       12.67
2024-07     3.09          4.19          85.15       14.37
2024-08     3.14          4.30          80.36       19.31
2024-09     3.15          4.29          74.02       17.66
2024-10     3.22          4.29          75.63       19.96
2024-11     3.25          4.41          74.35       16.02
2024-12     3.22          4.32          73.86       15.87
```

**Notable patterns:**
- NPL ratio (90+ days) very stable: 2.89% to 3.25%
- Gradual upward drift through 2024
- Oil prices declined ~8% over the year
- 2024 was a relatively calm period for credit quality

### Monthly Regressions (h=0)

$$NPL_t = \alpha + \beta \cdot Shock_t + \varepsilon_t$$

**NPL Ratio (90+ days overdue):**

| Shock Variable | β | SE | t-stat | p-value | R² |
|----------------|---|----|----|---------|-----|
| Oil Shock | -0.040 | 0.024 | -1.68 | 0.09 | 0.14 |
| VIX Innovation | 0.006 | 0.009 | 0.69 | 0.49 | 0.02 |

**NPL Ratio (30+ days overdue):**

| Shock Variable | β | SE | t-stat | p-value | R² |
|----------------|---|----|----|---------|-----|
| Oil Shock | -0.084 | 0.059 | -1.42 | 0.16 | 0.15 |
| VIX Innovation | 0.018 | 0.016 | 1.09 | 0.28 | 0.03 |

### Interpretation (Monthly)

- **Oil shock shows marginal significance** (p=0.09) with negative coefficient
  - This suggests higher oil prices → lower NPL (counterintuitive at first)
  - However, oil exports benefit Kazakhstan's economy, so this makes sense
- **Effect size is small**: 1 SD oil shock → 0.04 pp change in NPL
- **VIX innovation has no effect** on Kazakhstan NPL
- **R² still low** but higher than annual regressions (14% vs 4%)

**Caution:** N=12 is still too small for reliable inference. The marginal significance of oil shock may be spurious.

---

## Critical Limitations

### 1. Small Sample (N=16)

With only 16 annual observations:
- Statistical power is very low
- Standard errors are inflated
- Cannot reliably distinguish true effects from noise
- Pre-trends testing is infeasible

**Power calculation:** To detect a "medium" effect (β=3, SE≈3), we would need ~50-100 observations for 80% power at α=0.10.

### 2. Annual Frequency

Annual NPL ratios are:
- Slow-moving stock variables
- Affected by write-offs, restructuring, regulatory changes
- Not ideal for capturing shock responses

### 3. NPL ≠ New Defaults

Bank-system NPL ratio is:
- A **stock** (cumulative), not a **flow** (new defaults)
- Affected by numerator changes (new defaults, cures, write-offs)
- Affected by denominator changes (credit growth)

### 4. Identification

This is **reduced-form** analysis:
- Oil shocks may affect NPL through multiple channels (income, FX, inflation, fiscal)
- Cannot isolate "income → default" pathway
- Exclusion restriction is unlikely to hold

---

## What This Study CAN and CANNOT Conclude

### Can Conclude (with caveats)

- Descriptive correlation between shocks and aggregate credit quality
- Direction of relationships appears economically sensible
- Kazakhstan's NPL ratio has declined dramatically (2012-2023)

### Cannot Conclude

- Causal effect of income on default (requires micro data)
- Magnitude of shock effects (imprecisely estimated)
- Lag structure (insufficient power)
- Out-of-sample predictions

---

## Recommendations for Improvement

### Priority 1: Historical Monthly Data (CRITICAL)

**Current status:** We have monthly 2024 data (N=12) but need historical data.

**NBK archives contain monthly data back to 2005:**
- URL: https://nationalbank.kz/en/news/banks-performance/rubrics/2186
- Each year has separate Excel files that need to be downloaded
- Combined historical series would provide ~200+ monthly observations

**Action required:** Download Excel files from NBK archives for years 2010-2023 and combine with 2024 data.

### Priority 2: Flow Variables

Instead of NPL ratio (stock), use:
- New overdue loans (flow)
- Transition rates into delinquency
- Monthly change in NPL
- The NBK data includes this breakdown by overdue days (1-30, 31-60, 61-90, 90+)

### Priority 3: Segment Breakdown

NBK data includes breakdown by:
- Consumer vs. corporate loans (individuals vs. legal entities)
- Loan type (mortgage, consumer, SME)
- This would allow testing heterogeneous effects by segment

---

## Technical Details

### Software

- Python 3.10+
- pandas, numpy, statsmodels
- Heteroskedasticity-robust standard errors (HC1)

### Data Files

- `data/processed/credit_panel_annual.parquet` - Annual panel (2008-2023)
- `data/processed/credit_panel_monthly_2024.csv` - Monthly panel (2024)
- `data/processed/nbk_credit_monthly_2024.csv` - Raw NBK monthly data
- `data/processed/credit_panel.parquet` - Quarterly panel (shocks only)
- `data/raw/imf_fsi/npl_ratio_KZ.csv` - Raw NPL data (World Bank)

### Replication

```python
from studies.credit_default.src.credit_data_pipeline import CreditDataPipeline
from studies.credit_default.src.credit_lp import CreditLocalProjections

pipeline = CreditDataPipeline()
panel = pipeline.build_panel(frequency='quarterly')

lp = CreditLocalProjections(panel)
results = lp.estimate(outcome='npl_ratio', shocks=['oil_supply_shock'])
print(results.summary())
```

---

## Conclusion

This analysis establishes the infrastructure for studying credit quality responses to external shocks in Kazakhstan. We now have:

1. **Annual data (2008-2023):** 16 observations from World Bank
2. **Monthly data (2024):** 12 observations from NBK

**Key findings:**
- No statistically significant effects in annual data
- Marginal evidence (p=0.09) of oil shock effect in monthly data
- Both samples too small for reliable inference
- NPL ratio very stable in 2024 (~3%)

**Next steps:**
1. Download historical monthly NBK data (2010-2023) from archives
2. Combine into ~170+ monthly observations
3. Re-estimate with full historical monthly series
4. Add event study visualization around known episodes (2014 oil collapse, 2020 COVID)

**This document will be updated** when historical monthly data becomes available.

---

*Generated: January 2026*
*Study: credit_default (reduced-form macro design)*
*Version: 2.1 - Added monthly NBK 2024 data*
