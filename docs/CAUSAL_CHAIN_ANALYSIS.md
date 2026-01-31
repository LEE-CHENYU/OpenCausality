# Causal Chain Analysis: Kazakhstan Household Welfare Study

## Overview

This document traces the causal mechanism from global oil shocks to household welfare in Kazakhstan. We assess where the causal chain is theoretically valid and where it may be empirically broken.

---

## The Theoretical Causal Chain

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GLOBAL OIL MARKET                                 │
│                                                                      │
│   Supply Shock ────► Oil Price ◄──── Demand Shock                   │
│   (OPEC, geopolitics)         (Global business cycle)               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    KAZAKHSTAN TRANSMISSION                           │
│                                                                      │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │   DIRECT     │    │   FISCAL     │    │  EXCHANGE    │         │
│   │   CHANNEL    │    │   CHANNEL    │    │    RATE      │         │
│   ├──────────────┤    ├──────────────┤    ├──────────────┤         │
│   │ Oil sector   │    │ Government   │    │    Tenge     │         │
│   │ employment   │    │ oil revenues │    │ depreciation │         │
│   │ & wages      │    │ & transfers  │    │ & import     │         │
│   │              │    │              │    │   prices     │         │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘         │
│          │                   │                   │                  │
│          └───────────────────┴───────────────────┘                  │
│                              │                                       │
│                              ▼                                       │
│          ┌───────────────────────────────────────┐                  │
│          │      REGIONAL HETEROGENEITY           │                  │
│          │   (Oil Exposure E_oil_r)              │                  │
│          │   Atyrau: 80% │ Almaty City: 5%       │                  │
│          └───────────────────────────────────────┘                  │
│                              │                                       │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    HOUSEHOLD WELFARE                                 │
│                                                                      │
│   Per-capita monetary income (log_income_pc)                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Causal Chain Assessment: Link by Link

### Link 1: Global Shocks are Exogenous to Kazakhstan

**Status: VALID**

| Criterion | Assessment |
|-----------|------------|
| Price-taker assumption | Kazakhstan produces ~2% of global oil. Cannot influence global prices. |
| Shock identification | Baumeister-Hamilton SVAR uses sign restrictions with informative priors. Well-established methodology. |
| No reverse causality | Implausible that Kazakh household income affects OPEC production decisions or global demand. |
| Orthogonality | Shocks driven by geopolitics (Iraq, Libya, Russia-Ukraine), OPEC strategy, and global business cycles - all external to Kazakhstan. |

**Conclusion**: The exogeneity assumption is highly credible. Global oil shocks are plausibly random from Kazakhstan's perspective.

---

### Link 2: Exposures are Pre-Determined

**Status: VALID IN PRINCIPLE, PROBLEMATIC IN IMPLEMENTATION**

| Criterion | Assessment |
|-----------|------------|
| Exposure timing | Exposures computed from 2010-2013 (baseline period before main sample). |
| No endogenous response | Regions cannot adjust their 2010-2013 mining share in response to post-2014 shocks. |
| Time-invariant | Exposure E_oil_r is constant for each region across all time periods. |

**BUT - Implementation Problems**:

| Issue | Severity | Details |
|-------|----------|---------|
| Hardcoded values | HIGH | Oil exposures are not computed from BNS data. They are hardcoded: `{"Atyrau": 0.8, "Mangystau": 0.7, ...}` in `panel_data.py:322-328` |
| No documentation | HIGH | Source of these values is not cited. Are they from academic literature, government reports, or guesswork? |
| No mining data | HIGH | BNS mining endpoint returns 500 errors. Real data unavailable. |

**Conclusion**: The design is valid, but we cannot verify that the hardcoded exposures reflect true economic structure.

---

### Link 3: Parallel Trends Conditional on Fixed Effects

**Status: BORDERLINE**

The key identification assumption:
$$E[\varepsilon_{rt} | E_{kr}, S_{kt}, \alpha_r, \gamma_t] = 0$$

**Pre-Trends Test Results**:

| Lead | p-value | Significant at 5%? |
|------|---------|---------------------|
| 1Q   | 0.05    | YES (marginal) |
| 2Q   | 0.20    | No |
| 3Q   | 0.10    | Marginal at 10% |
| 4Q   | 0.02    | YES |

- **Joint test**: F-stat = 1.81, p = 0.126 (cannot reject at 10%)
- **But**: 2 of 4 individual leads are significant at 5%

**Interpretation**:

The joint test technically passes, but the pattern is concerning:
1. If parallel trends held, we would not expect individual leads to be significant.
2. The significant leads (1Q and 4Q) suggest either:
   - Anticipation effects (markets respond before shocks materialize)
   - Differential pre-trends (high-oil regions were already diverging)
   - Specification error (leads picking up omitted dynamics)

**Conclusion**: Parallel trends are not definitively violated, but not confidently validated either. Borderline.

---

### Link 4: Shock Transmission to Regional Income

**Status: EMPIRICALLY BROKEN**

This is where the causal chain fails. The data do not show differential responses:

| Interaction | Coefficient | p-value | Economic Significance |
|-------------|-------------|---------|----------------------|
| E_oil x Supply | 0.017 | 0.37 | Indistinguishable from zero |
| E_oil x Demand | -0.006 | 0.92 | Indistinguishable from zero |
| E_cyc x Activity | -0.001 | 0.38 | Indistinguishable from zero |

**Why Might Transmission Be Zero?**

1. **Fiscal stabilization**: Government transfers may offset income losses in oil regions
2. **Labor mobility**: Workers may relocate to non-oil regions during downturns
3. **Diversification**: Even "oil" regions may have diversified economies
4. **Measurement error**: Hardcoded exposures may not reflect true heterogeneity
5. **True null**: The effect is genuinely zero

---

## Where the Chain is Broken

### Schematic: Observed vs. Expected

```
EXPECTED CAUSAL CHAIN:
Shock (+) ──► Exposure (×) ──► Differential Effect (≠0) ──► Validated Estimates

OBSERVED CAUSAL CHAIN:
Shock (+) ──► Exposure (?) ──► Differential Effect (≈0) ──► Null Result
        │                   │
        │                   └── Problem: No significant interaction
        └── Problem: Exposures are hardcoded, not measured
```

### Root Cause Analysis

| Potential Problem | Evidence | Severity |
|-------------------|----------|----------|
| Wrong exposure measure | Hardcoded without documentation | HIGH |
| Missing income data | BNS has 22% missing region-quarters | MEDIUM |
| Synthetic shock data | Baumeister loader has seed=42 fallback | MEDIUM (verified real) |
| Wrong lag structure | Contemporaneous may be wrong horizon | MEDIUM |
| Insufficient power | Only 16 regions, 60 quarters | MEDIUM |
| True null effect | Cannot rule out | UNKNOWN |

---

## What Would Validate the Causal Chain?

### Minimum Requirements

1. **Real exposure data**: Mining sector shares from actual BNS statistics
2. **Complete outcome data**: No missing region-quarters
3. **Verified shock data**: Confirmed download from Baumeister website
4. **Significant effects**: At least one interaction with p < 0.05
5. **Smooth IRF**: Plausible impulse response dynamics

### Stronger Evidence

1. **Pass all pre-trends**: No individual leads significant at 10%
2. **Multiple significant horizons**: Cluster of significant effects, not isolated spikes
3. **Out-of-sample validation**: Predictions match realized outcomes for 2014-15, 2020
4. **Mechanism tests**: Mediators (employment, transfers) respond as expected

---

## Honest Assessment

### What We Can Claim

1. The **design** is valid:
   - Shocks are exogenous
   - Exposures are pre-determined
   - Shift-share is appropriate for this setting

2. The **data** are partially available:
   - FRED series are real and high-quality
   - Baumeister shocks appear to be real (not seed=42)
   - BNS income data exists with some gaps

### What We Cannot Claim

1. **Causal effects exist**: The null cannot be rejected
2. **Effects peak at 3 quarters**: Likely spurious from multiple testing
3. **Regional heterogeneity is pronounced**: Cannot be claimed when base effects are zero
4. **The scenario engine is validated**: Predictions are unreliable without significant multipliers

---

## Recommendations

### For This Study

1. **Obtain real mining data**: Contact BNS directly, use OECD regional statistics, or find academic sources for regional oil dependence
2. **Document exposure sources**: Any hardcoded values must cite their origin
3. **Report the null honestly**: The current evidence does not support causal claims

### For Future Research

1. **Alternative designs**:
   - Synthetic control comparing Atyrau to similar non-oil regions
   - Event study around specific large shocks (2014-15 collapse)
   - Firm-level or household survey data for more variation

2. **Longer sample**: More quarters may increase power

3. **Different outcomes**: Employment, expenditure, or subjective wellbeing may respond when income does not

---

## Conclusion

The causal chain from global oil shocks to Kazakhstan household welfare has:

- **Valid identification strategy** (exogenous shocks, pre-determined exposures)
- **Questionable data inputs** (hardcoded exposures, no mining data)
- **Null empirical findings** (no significant differential effects)

The chain is **theoretically coherent** but **empirically unvalidated**. Until we can demonstrate significant differential responses to shocks, the causal story remains a hypothesis, not a finding.

---

*Document generated: January 2026*
*Assessment: Honest evaluation of causal chain validity*
