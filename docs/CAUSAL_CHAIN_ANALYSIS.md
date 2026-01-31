# Causal Chain Analysis: Kazakhstan Household Welfare Study (v4)

## Overview

This document traces the causal mechanism from global oil shocks to household welfare in Kazakhstan. We assess where the causal chain is theoretically valid and where it may be empirically broken.

**v4 Update**: Study design revised. Main spec focuses on oil exposure only (E_oil_r × oil shock). Cyclical exposure dropped from core identification; GRP-based proxy available for robustness only.

---

## The Theoretical Causal Chain

```
+---------------------------------------------------------------------+
|                    GLOBAL OIL MARKET                                |
|                                                                     |
|   Supply Shock -----> Oil Price <----- Demand Shock                 |
|   (OPEC, geopolitics)         (Global business cycle)               |
+----------------------------------+----------------------------------+
                                   |
                                   v
+---------------------------------------------------------------------+
|                    KAZAKHSTAN TRANSMISSION                          |
|                                                                     |
|   +--------------+    +--------------+    +--------------+          |
|   |   DIRECT     |    |   FISCAL     |    |  EXCHANGE    |          |
|   |   CHANNEL    |    |   CHANNEL    |    |    RATE      |          |
|   +--------------+    +--------------+    +--------------+          |
|   | Oil sector   |    | Government   |    |    Tenge     |          |
|   | employment   |    | oil revenues |    | depreciation |          |
|   | & wages      |    | & transfers  |    | & import     |          |
|   |              |    |              |    |   prices     |          |
|   +------+-------+    +------+-------+    +------+-------+          |
|          |                   |                   |                  |
|          +-------------------+-------------------+                  |
|                              |                                      |
|                              v                                      |
|          +-------------------------------------------+              |
|          |      REGIONAL HETEROGENEITY               |              |
|          |   (Oil Exposure E_oil_r)                  |              |
|          |   High-oil vs. Low-oil regions            |              |
|          +-------------------------------------------+              |
|                              |                                      |
+------------------------------+--------------------------------------+
                               |
                               v
+---------------------------------------------------------------------+
|                    HOUSEHOLD WELFARE                                |
|                                                                     |
|   Per-capita monetary income (log_income_pc)                        |
|                                                                     |
+---------------------------------------------------------------------+
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

**Status: VALID IN PRINCIPLE**

| Criterion | Assessment |
|-----------|------------|
| Exposure timing | Exposures computed from 2010-2013 (baseline period before main sample). |
| No endogenous response | Regions cannot adjust their 2010-2013 mining share in response to post-2014 shocks. |
| Time-invariant | Exposure E_oil_r is constant for each region across all time periods. |

**v3 Implementation Status:**

| Issue | Previous (v1) | Current (v3) |
|-------|---------------|--------------|
| Hardcoded values | Code used `{"Atyrau": 0.8, ...}` with no source | Uses USGS/EITI documented values |
| No documentation | Source unknown | Full citation for all sources |
| No mining data | Silent fallback | Falls back to alternative sources with lineage tracking |

**Alternative Sources Now Available:**
- USGS Mineral Industry Report 2022: https://pubs.usgs.gov/myb/vol3/2022/myb3-2022-kazakhstan.pdf
- EITI Kazakhstan: https://eiti.org/countries/kazakhstan
- stat.gov.kz GRP Publications: https://stat.gov.kz/en/industries/economy/national-accounts/publications/

**Conclusion**: The design is valid. Oil exposure (E_oil_r) can now be computed from documented sources.

---

### Link 3: Parallel Trends Conditional on Fixed Effects

**Status: BORDERLINE**

The key identification assumption:

```
E[error | Exposure, Shock, RegionFE, TimeFE] = 0
```

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
2. The significant leads (1Q and 4Q) suggest either anticipation effects or differential pre-trends.

**Conclusion**: Parallel trends are not definitively violated, but not confidently validated either.

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

1. **Fiscal stabilization**: Government transfers may offset income losses
2. **Labor mobility**: Workers may relocate during downturns
3. **Diversification**: Even "oil" regions may have diversified economies
4. **True null**: The effect is genuinely zero

---

## Where the Chain is Broken

### Schematic: Observed vs. Expected

```
EXPECTED CAUSAL CHAIN:
Shock (+) --> Exposure (x) --> Differential Effect (!=0) --> Validated Estimates

OBSERVED CAUSAL CHAIN:
Shock (+) --> Exposure (?) --> Differential Effect (~=0) --> Null Result
        |                   |
        |                   +-- Problem: No significant interaction
        +-- Now requires real data (v2)
```

### Root Cause Analysis

| Potential Problem | Status (v3) | Notes |
|-------------------|-------------|-------|
| Wrong exposure measure | **RESOLVED** | USGS/EITI data now available |
| Missing income data | Active issue | BNS has 2 missing regions |
| Synthetic shock data | **RESOLVED** | Real Baumeister data verified |
| Wrong lag structure | Possible | Contemporaneous may be wrong horizon |
| Insufficient power | Possible | Only 16 regions |
| True null effect | Cannot rule out | Effect may genuinely be zero |

---

## Data Requirements (v4)

### Main Specification (Oil Only)

| Data | Variable | Source | Status |
|------|----------|--------|--------|
| Mining sector shares | E_oil_r | USGS/EITI/stat.gov.kz | **AVAILABLE** |
| Per-capita income | log_income_pc | BNS | Partial (14/16 regions) |
| Oil supply shock | oil_supply_shock | Baumeister | Available |
| Demand shock | aggregate_demand_shock | Baumeister | Available |

### Robustness Specification (Add Cyclical Proxy)

| Data | Variable | Source | Status |
|------|----------|--------|--------|
| Cyclical proxy | E_cyc_proxy_r | GRP-based | **AVAILABLE** (not employment) |
| Global activity | global_activity_shock | FRED IGREA | Available |

### Code Behavior (v4)

```python
# Main spec: Oil exposure only
MAIN_SPEC = ShiftShareSpec(
    name="main_oil_only",
    interactions=[
        ("E_oil_r", "oil_supply_shock"),
        ("E_oil_r", "aggregate_demand_shock"),
    ],
)

# Robustness: Add cyclical proxy
ROBUSTNESS_SPEC = ShiftShareSpec(
    name="robustness_with_cyclical_proxy",
    interactions=[
        ("E_oil_r", "oil_supply_shock"),
        ("E_oil_r", "aggregate_demand_shock"),
        ("E_cyc_proxy_r", "global_activity_shock"),
    ],
)
```

The pipeline is now fully operational for the main specification.

---

## What Would Validate the Causal Chain?

### Minimum Requirements

1. **Real exposure data**: Mining sector shares from actual BNS statistics or documented alternative source
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
   - Exposures are pre-determined (in principle)
   - Shift-share is appropriate for this setting

2. The **code** now enforces correct behavior:
   - No silent fallbacks to hardcoded data
   - Fails loudly when real data unavailable
   - Data lineage tracking available

### What We Cannot Claim

1. **Causal effects exist**: The null cannot be rejected
2. **Effects peak at 3 quarters**: Likely spurious from multiple testing
3. **Regional heterogeneity is pronounced**: Cannot claim when base effects are zero
4. **The scenario engine is validated**: Predictions are unreliable without significant multipliers

---

## Recommendations

### For This Study

1. ~~**Obtain real mining data**~~: **RESOLVED** - USGS/EITI data now in `data/raw/alternative_sources/`
2. **Obtain employment data**: Still needed for E_cyc_r (try ILO, OECD)
3. **Document all data sources**: All values now cite their origin
4. **Report the null honestly**: The current evidence does not support causal claims

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
- **Strict data requirements** (v2 - no hardcoded fallbacks)
- **Null empirical findings** (no significant differential effects)

The chain is **theoretically coherent** but **empirically unvalidated**. The pipeline is now fully operational with oil exposure from documented sources. Cyclical exposure dropped from core model; GRP-based proxy available for robustness checks.

Until we can demonstrate significant differential responses to shocks with real measured exposures, the causal story remains a hypothesis, not a finding.

---

*Document version: 4.0*
*Generated: January 2026*
*Assessment: Honest evaluation of causal chain validity*
*Update: Main spec is oil-only; cyclical proxy for robustness*
