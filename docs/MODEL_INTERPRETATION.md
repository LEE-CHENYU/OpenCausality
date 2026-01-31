# Model Interpretation: Kazakhstan Household Welfare Study

## CRITICAL DISCLAIMER

**This document provides a brutally honest assessment of the model results. The findings are disappointing but must be reported accurately.**

---

## Executive Summary: Results Are NOT Statistically Significant

The primary finding of this study is **null**. The baseline shift-share regression shows:

| Interaction Term | Coefficient | Std. Error | p-value | Interpretation |
|------------------|-------------|------------|---------|----------------|
| Oil Exposure x Supply Shock | 0.017 | 0.019 | **0.37** | NOT significant |
| Oil Exposure x Demand Shock | -0.006 | 0.055 | **0.92** | NOT significant |
| Cyclical Exposure x Global Activity | -0.001 | 0.001 | **0.38** | NOT significant |

**None of these p-values are below conventional thresholds (0.05 or 0.10).** The estimates are noisy and we cannot reject the null hypothesis that oil shocks have zero differential effect on high-exposure vs. low-exposure regions.

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

1. **Selection bias**: We tested 6 horizons and only highlighted the one that was significant. This is exactly what the multiple comparisons literature warns against.

2. **No pre-registration**: We did not specify horizon 3 as our primary outcome before looking at the data.

3. **Implausible dynamics**: The coefficient at horizon 3 (-0.039) is 3x larger than surrounding horizons. Economic mechanisms (labor adjustment, fiscal transfers) would produce smoother IRFs.

4. **Pattern inconsistency**: If effects truly peaked at 3 quarters, we would expect:
   - Gradually increasing effects from horizon 0 to 3
   - Gradually decreasing effects from horizon 3 to 5

   Instead, we see: 0 → 1 increases (wrong sign), 1 → 2 decreases, 2 → 3 large jump, 3 → 4 falls by half. This pattern is more consistent with noise than signal.

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

### Interpretation

The joint test passes, but individual leads at 1Q and 4Q are significant. This is concerning because:

1. If parallel trends truly held, we wouldn't expect 2 of 4 leads to be individually significant at p < 0.05.

2. The pattern suggests **anticipation effects** or **differential pre-trends** that could bias our estimates.

3. We cannot dismiss this as "just one test" since 2/4 leads are significant.

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

Our current results satisfy **none** of these criteria.

---

## The R-squared Problem

From the estimation output:
```
R² within: -0.0048
```

**A negative within-R² means the model explains less variation than a simple intercept.** The interaction terms are adding noise, not signal.

This is a strong indication that:
- The effect sizes are genuinely small (close to zero)
- The model specification is not capturing the true relationship
- Or both

---

## Coefficient Interpretation (Assuming They Were Significant)

For context, if the baseline coefficients *were* significant, here's how to interpret them:

**Oil Exposure x Supply Shock coefficient = 0.017**

- A 1 SD negative oil supply shock would reduce log income by 0.017 * E_oil_r in high-exposure regions
- For Atyrau (E_oil = 0.80): Effect = 0.017 * 0.80 = 0.014 (1.4% reduction)
- For Almaty City (E_oil = 0.05): Effect = 0.017 * 0.05 = 0.001 (0.1% reduction)
- Differential: 1.3 percentage points

But since p = 0.37, this estimate is indistinguishable from zero.

---

## Why the Null Result?

Several possible explanations:

### 1. True Null Effect
The causal effect is genuinely zero or negligible. Global oil shocks may not differentially affect high-oil regions after controlling for time fixed effects (which absorb common shocks to all regions).

### 2. Offsetting Channels
Positive and negative channels may cancel:
- Negative: Direct income loss from lower oil activity
- Positive: Tenge depreciation increases competitiveness of other exports
- Positive: Government stabilization transfers target affected regions

### 3. Measurement Error
- Oil exposure is hardcoded from "stylized" values, not measured from actual data
- BNS income data has missing observations
- Baumeister shocks may have been synthetic in some runs

### 4. Insufficient Variation
- Only 16 regions over ~60 quarters = 960 observations
- Cross-sectional variation in oil exposure is limited (5 clearly oil-exposed regions)
- May lack statistical power to detect moderate effects

### 5. Wrong Specification
- Linear interactions may miss nonlinear effects
- Contemporaneous effects may require different lag structure
- Regional labor market institutions may modify transmission

---

## What the Documentation (STUDY_DOCUMENTATION.md) Claims vs. Reality

### Claim 1: "Oil demand shocks have statistically significant negative effects"

**Reality**: p = 0.92 for oil demand interaction. This is the opposite of significant.

### Claim 2: "The impact peaks at 3 quarters after the shock"

**Reality**: Horizon 3 shows p = 0.001, but this is 1 of 6 tests without multiple comparison correction. Likely spurious.

### Claim 3: "Falsification tests pass"

**Reality**: Pre-trends joint p = 0.126, but 2 of 4 individual leads are significant at p < 0.05. This is borderline, not a clear pass.

### Claim 4: "Regional heterogeneity is pronounced"

**Reality**: The differential effects are not statistically distinguishable from zero. We cannot claim heterogeneity when the base effects are insignificant.

---

## Recommendations

### For This Study

1. **Do not claim causal effects exist**. The evidence does not support the claim.

2. **Report the null result honestly**. Null results are scientifically valuable and should be published.

3. **Investigate data quality**. Fix hardcoded exposures, verify Baumeister data source, address missing BNS data.

4. **Consider pre-registration**. For future analyses, specify primary hypotheses and horizons before estimation.

### For Policy

1. **Do not use these estimates for policy**. The uncertainty bounds include zero.

2. **The scenario engine is not validated**. Simulations based on these multipliers are unreliable.

3. **Further research needed**. Better data and/or different methodological approaches may yield more reliable estimates.

---

## Conclusion

This study has a valid causal design (shift-share with exogenous shocks and pre-determined exposures) but finds **no statistically significant effects**. The lone significant result (horizon 3 in local projections) is likely a false positive from multiple testing.

Honest science requires reporting null results. The hypothesis that global oil shocks differentially affect high-oil-exposure regions in Kazakhstan is not supported by this analysis.

---

*Document generated: January 2026*
*Assessment: Brutally honest*
