# Model Interpretation: Kazakhstan Household Welfare Study (v4)

## CRITICAL DISCLAIMER

**This document provides a brutally honest assessment of the model results. The findings are disappointing but must be reported accurately.**

**Update (v4):** Revised study design - main specification uses oil exposure only (E_oil_r). Cyclical exposure dropped from core model; GRP-based proxy available for robustness checks only.

---

## Executive Summary: Results Are NOT Statistically Significant

The primary finding of this study is **null**. The shift-share regression shows:

### Main Specification (Oil Only)
| Interaction Term | Coefficient | Std. Error | p-value | Interpretation |
|------------------|-------------|------------|---------|----------------|
| Oil Exposure × Supply Shock | 0.035 | 0.040 | **0.38** | NOT significant |
| Oil Exposure × Demand Shock | -0.002 | 0.108 | **0.98** | NOT significant |

### Robustness Specification (Add Cyclical Proxy)
| Interaction Term | Coefficient | Std. Error | p-value | Interpretation |
|------------------|-------------|------------|---------|----------------|
| Oil Exposure × Supply Shock | 0.051 | 0.044 | **0.24** | NOT significant |
| Oil Exposure × Demand Shock | -0.013 | 0.109 | **0.91** | NOT significant |
| Cyclical Proxy × Global Activity | -0.003 | 0.002 | **0.09** | Marginal* |

**Sample:** 746 observations (14 regions × ~53 quarters with non-missing income)

**None of the oil coefficients are statistically significant.** The estimates are noisy and we cannot reject the null hypothesis that oil shocks have zero differential effect on high-exposure vs. low-exposure regions.

---

---

## Study Design (v4)

### Main Specification: Oil Exposure Only

```
y_{r,t} = α_r + δ_t + β(E_oil_r × Shock_oil_t) + u_{r,t}
```

| Component | Meaning |
|-----------|---------|
| `α_r` | Region fixed effects (permanent regional differences) |
| `δ_t` | Time fixed effects (absorbs national cycle, policy, common shocks) |
| `β` | Differential effect of oil shock on high-oil vs low-oil regions |
| `E_oil_r` | Pre-period (2010-2013) oil/mining share by region |
| `Shock_oil_t` | Baumeister structural oil shock |

**Why oil-only is the best default:**
- Clean identification: cross-sectional heterogeneity × common shock
- Avoids constructing noisy cyclical proxies
- Honest about scope: regional distribution of oil shocks, not general cyclical sensitivity

### Robustness Specification: Add Cyclical Proxy

```
y_{r,t} = α_r + δ_t + β(E_oil_r × Shock_oil_t) + θ(E_cyc_proxy_r × Shock_cyc_t) + u_{r,t}
```

**Purpose:** Check if β is stable when adding (noisy) cyclical control
- `E_cyc_proxy_r` is GRP-based, NOT true employment data
- If β stable → oil result isn't secretly "cycle in disguise"

---

## Data Summary (Actual)

| Data | Source | Observations | Notes |
|------|--------|--------------|-------|
| Per-capita income | BNS | 846 rows, 14 regions | 2010Q1-2025Q3 |
| Mining shares | USGS/EITI/stat.gov.kz | 16 regions | E_oil_r range: [0.00, 0.32] |
| Cyclical proxy | GRP-based | 16 regions | E_cyc_proxy range: [0.25, 0.85] |
| Oil shocks | Baumeister | 608 months | 1975-2025 |
| Global activity | FRED IGREA | 311 months | AR(1) innovation |

**Analysis sample:** 746 observations (14 regions × ~53 quarters after dropping missing)

**Missing regions in income data:** West Kazakhstan, North Kazakhstan

---

## The Multiple Comparisons Problem

### What We Tested

This study involves **at least 18 hypothesis tests**:

1. **Baseline regression**: 3 interaction terms x 1 test each = 3 tests
2. **Local projections**: 1 interaction x 6 horizons = 6 tests
3. **Pre-trends leads**: 4 leads = 4 tests
4. **Placebo tests**: 2-3 tests
5. **Robustness variations**: Additional tests

### The Problem

When testing 18+ hypotheses at alpha = 0.05, we expect to find ~1 "significant" result by chance alone, even if the true effect is zero everywhere.

### Bonferroni-Corrected Threshold

For 18 tests at family-wise error rate 0.05:
- Corrected threshold: 0.05 / 18 = **0.0028**
- Horizon 3 result (p = 0.001) barely survives this correction
- But this assumes pre-registration; we selected this horizon post-hoc

---

## Local Projections: The Horizon 3 Problem

### Raw Results

| Horizon | Coefficient | Std. Error | p-value | Significant? |
|---------|-------------|------------|---------|--------------|
| 0 | -0.012 | 0.009 | 0.19 | No |
| 1 | 0.002 | 0.015 | 0.90 | No |
| 2 | -0.012 | 0.017 | 0.47 | No |
| **3** | **-0.039** | **0.012** | **0.001** | Yes*** |
| 4 | -0.018 | 0.018 | 0.31 | No |
| 5 | -0.009 | 0.012 | 0.46 | No |

### Why Horizon 3 May Be Spurious

1. **Selection bias**: We tested 6 horizons and only highlighted the one that was significant.

2. **No pre-registration**: We did not specify horizon 3 as our primary outcome before looking at the data.

3. **Implausible dynamics**: The coefficient at horizon 3 (-0.039) is 3x larger than surrounding horizons. Economic mechanisms would produce smoother IRFs.

4. **Pattern inconsistency**: If effects truly peaked at 3 quarters, we would expect gradually increasing then decreasing effects. Instead, we see erratic patterns more consistent with noise.

### Honest Interpretation

The horizon 3 result is likely a **Type I error** (false positive) arising from testing multiple hypotheses without pre-registration or correction.

---

## Pre-Trends Test: Borderline Concerning

### Results

| Lead | Coefficient | Std. Error | p-value |
|------|-------------|------------|---------|
| 1 quarter | 0.028 | 0.014 | **0.05*** |
| 2 quarters | 0.018 | 0.014 | 0.20 |
| 3 quarters | 0.024 | 0.015 | 0.10* |
| 4 quarters | 0.029 | 0.013 | **0.02*** |

- **Joint F-test p-value**: 0.126 (cannot reject at 10%)
- **But**: 2 of 4 individual leads are significant at 5%

### Interpretation

The joint test technically passes, but the pattern is concerning. If parallel trends truly held, we wouldn't expect 2 of 4 leads to be individually significant at p < 0.05.

**Honest assessment**: Pre-trends are borderline. The identification strategy is not definitively invalidated, but it is also not strongly validated.

---

## What Would Convince Us of a Real Effect?

For credible causal claims, we would want:

1. **Baseline significance**: At least one contemporaneous effect with p < 0.05
2. **Smooth impulse response**: Gradually building and decaying effects, not isolated spikes
3. **Pre-registered horizons**: Specifying the expected lag structure before estimation
4. **Clean pre-trends**: All leads clearly insignificant (p > 0.20)
5. **Multiple comparison correction**: Results surviving Bonferroni or Benjamini-Hochberg correction
6. **Out-of-sample validation**: Predictions validated on held-out episodes
7. **Real exposure data**: Measured from actual statistics, not hardcoded

Our current results satisfy **none** of these criteria.

---

## The R-squared Problem

From the estimation output:
```
Main specification:       R² within = -0.0008
Robustness specification: R² within = -0.0102
```

**A negative within-R-squared means the model explains less variation than a simple intercept.** The interaction terms are adding noise, not signal.

This strongly indicates that:
- The effect sizes are genuinely small (close to zero)
- The model specification is not capturing the true relationship
- Or both

---

## Beta Stability Check: FAILED

Comparing the oil supply coefficient across specifications:

| Specification | β (Oil Supply) | Std. Error | p-value |
|---------------|----------------|------------|---------|
| Main (oil only) | 0.0351 | 0.0396 | 0.375 |
| Robustness (+ cyclical) | 0.0513 | 0.0439 | 0.244 |
| **Change** | +0.0161 | - | **46%** |

**The oil coefficient changes by 46% when adding the cyclical proxy.** This exceeds the 20% stability threshold.

### Interpretation

The instability of β suggests:
1. **Omitted variable bias**: The cyclical proxy is correlated with both oil exposure and the outcome
2. **Multicollinearity**: Oil-exposed regions may also be cyclically sensitive (or the opposite)
3. **Specification sensitivity**: Results are fragile to model specification

However, since **neither coefficient is statistically significant**, this instability is less concerning for policy - both specifications agree that the effect is indistinguishable from zero.

---

## Coefficient Interpretation (Hypothetical)

For context, if the baseline coefficients *were* significant, here's how to interpret them:

**Oil Exposure x Supply Shock coefficient = 0.017**

- A 1 SD negative oil supply shock would reduce log income by 0.017 * E_oil_r
- For a region with 80% oil exposure: Effect = 0.017 * 0.80 = 0.014 (1.4% reduction)
- For a region with 5% oil exposure: Effect = 0.017 * 0.05 = 0.001 (0.1% reduction)
- Differential: 1.3 percentage points

But since p = 0.37, this estimate is indistinguishable from zero.

---

## Why the Null Result?

Several possible explanations:

### 1. True Null Effect
The causal effect is genuinely zero or negligible. Global oil shocks may not differentially affect high-oil regions after controlling for time fixed effects.

### 2. Offsetting Channels
Positive and negative channels may cancel:
- Negative: Direct income loss from lower oil activity
- Positive: Tenge depreciation increases competitiveness
- Positive: Government stabilization transfers target affected regions

### 3. Data Quality Issues (Now Addressed)
- **Previously**: Oil exposure was hardcoded from unknown sources
- **Now**: Pipeline requires real measured exposures (will fail without them)

### 4. Insufficient Variation
- Only 16 regions over ~60 quarters = 960 observations
- Cross-sectional variation in oil exposure is limited
- May lack statistical power to detect moderate effects

### 5. Wrong Specification
- Linear interactions may miss nonlinear effects
- Contemporaneous effects may require different lag structure

---

## Data Integrity (v3 Update)

### Previous Problem (v1)
The original implementation silently substituted hardcoded values when real data was unavailable:
- Oil exposures: `{"Atyrau": 0.8, "Mangystau": 0.7, ...}` with NO source citation
- Cyclical exposures: Similar hardcoded values
- Synthetic fallbacks: `seed=42` random data if downloads failed

### Current Behavior (v3)
**Documented alternative sources now available.** The code now:
- Tries BNS API first for all data
- Falls back to documented alternative sources (USGS, EITI, stat.gov.kz GRP) when BNS fails
- Records data lineage showing which source was used
- Raises `ValueError` only when NO sources (primary or alternative) are available

### Alternative Sources for Mining Shares
| Source | Coverage | Key Data |
|--------|----------|----------|
| USGS Mineral Industry Report | 2016-2022 | Atyrau: 32%, Mangystau: 29%, Aktobe: 15% |
| EITI Kazakhstan | 2005-2021 | Regional GVA from extractive industries |
| stat.gov.kz GRP Publications | 2010-2023 | GRP by region |

**File:** `data/raw/alternative_sources/mining_shares.csv`

---

## Recommendations

### For This Study

1. **Do not claim causal effects exist**. The evidence does not support the claim.

2. **Report the null result honestly**. Null results are scientifically valuable.

3. **Remaining data needs**:
   - ~~Mining sector shares by region (for E_oil_r)~~ **RESOLVED** via USGS/EITI
   - Employment by sector (for E_cyc_r) - still needed
   - Verify debt share data availability

4. **Consider pre-registration** for future analyses.

### For Policy

1. **Do not use these estimates for policy**. The uncertainty bounds include zero.

2. **The scenario engine is not validated**. Simulations based on these multipliers are unreliable.

3. **Further research needed** with better data.

---

## Conclusion

This study has a valid causal design (shift-share with exogenous shocks and pre-determined exposures) but finds **no statistically significant effects**. The lone significant result (horizon 3 in local projections) is likely a false positive from multiple testing.

**v4 Update**: Study design revised to focus on oil exposure only. Main spec uses E_oil_r × oil shocks with clean identification. Cyclical exposure available as GRP-based proxy for robustness checks only.

Honest science requires reporting null results. The hypothesis that global oil shocks differentially affect high-oil-exposure regions in Kazakhstan is not supported by this analysis.

---

*Document version: 4.0*
*Generated: January 2026*
*Assessment: Brutally honest*
*Update: Revised study design - oil exposure only in main spec*
