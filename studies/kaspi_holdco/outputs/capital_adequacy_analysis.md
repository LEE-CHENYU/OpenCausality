# Kaspi.kz Holding Company Capital Adequacy Analysis

**Date:** February 2026
**Data Source:** FY2024 20-F Filing
**Framework:** NBK Regulatory Capital (K1-2, K2 ratios)

---

## Executive Summary

Kaspi.kz's holding company exhibits **strong stand-alone solvency** under baseline conditions, with 187 months of runway on cash reserves alone. However, the bank subsidiary operates with a **tight K2 buffer** (70 basis points above the 12% minimum), creating potential vulnerability in stress scenarios.

**Key Finding:** The holding company can absorb capital calls up to ~324 KZT bn (if shareholder dividends are suspended), but severe stress scenarios (5%+ credit losses with RWA inflation) would require external funding support.

---

## 1. Baseline Position (FY2024)

### 1.1 Bank Regulatory Position

| Metric | Value | Minimum | Buffer |
|--------|-------|---------|--------|
| K1-2 Ratio | 12.6% | 10.5% | +210 bps |
| K2 Ratio | 12.7% | 12.0% | +70 bps |
| K2 Headroom | 59.9 KZT bn | - | - |

**Interpretation:** The K1-2 ratio has comfortable headroom (210 bps), but the K2 ratio buffer is notably thin at only 70 basis points. This tightness is partially explained by NBK's conservative RWA methodology, which applies 150%+ risk weights to unsecured consumer lending—Kaspi's core business.

The NBK RWA (8,059 KZT bn) is approximately 45% higher than equivalent Basel III RWA (~5,577 KZT bn), reflecting Kazakhstan's prudent regulatory stance on consumer credit.

### 1.2 Holding Company Position

| Metric | Value |
|--------|-------|
| Parent Cash | 324.993 KZT bn |
| Annual G&A | 20.810 KZT bn |
| Parent Liabilities | 0.155 KZT bn |
| Monthly Burn Rate | 1.73 KZT bn |
| Standalone Runway | 187 months (~15.6 years) |

**Interpretation:** The holding company maintains an exceptionally strong liquidity position. With minimal liabilities (155 million KZT) and low fixed costs, the parent could survive over 15 years on cash reserves alone without any dividend income. This provides substantial buffer for stress scenarios.

### 1.3 Dividend Flows (FY2024 Actuals)

| Flow | Amount (KZT bn) |
|------|-----------------|
| Dividends from Bank | 285.2 |
| Dividends from Other Subs | 467.5 |
| **Total Inflows** | **752.7** |
| Dividends to Shareholders | 646.1 |
| **Net Cash Generation** | **+85.8** |

**Interpretation:** Under normal operations, the holding company generates ~86 KZT bn in annual net cash flow. The diversified dividend stream (62% from non-bank subsidiaries) provides resilience—even if bank dividends are cut, other subsidiaries (payments, marketplace) can continue distributions.

---

## 2. Stress Test Results

### 2.1 Bank-Level Stress Scenarios

| Scenario | Profit | Credit Loss | RWA Mult | K2 After | Shortfall | Dividend Cut |
|----------|--------|-------------|----------|----------|-----------|--------------|
| Baseline | 100% | 1% | 1.00x | 17.6% | 0 | 0% |
| Mild | 80% | 2% | 1.05x | 15.0% | 0 | 0% |
| Moderate | 50% | 3% | 1.10x | 11.7% | 24 bn | 100% |
| Severe | 0% | 5% | 1.15x | 6.8% | 484 bn | 100% |
| Oil Crisis | 0% | 7% | 1.25x | 2.5% | 957 bn | 100% |

**Key Observations:**

1. **Baseline and Mild scenarios** are fully viable. Retained earnings offset credit losses, and K2 remains above minimum. Full dividend capacity preserved.

2. **Moderate scenario** is the critical threshold. At 3% credit losses with 10% RWA inflation, K2 drops below 12%, triggering:
   - Dividend restriction (no payments to parent)
   - Capital shortfall of 24 KZT bn requiring injection

3. **Severe and Oil Crisis scenarios** represent tail risks where the bank would require substantial recapitalization (484-957 KZT bn).

### 2.2 Bank → HoldCo Passthrough Results

| Scenario | Bank Dividend | Capital Call | HoldCo Ending Cash | Status |
|----------|---------------|--------------|-------------------|--------|
| Baseline | 285 bn | 0 | 411 bn | Viable |
| Mild | 285 bn | 0 | 411 bn | Viable |
| Moderate | 0 | 24 bn | 747 bn | Viable |
| Severe | 0 | 484 bn | 288 bn | **NEEDS FUNDING** |
| Oil Crisis | 0 | 957 bn | -185 bn | **NEEDS FUNDING** |

**Interpretation:**

- **Moderate stress is absorbable:** Even with complete loss of bank dividends and a 24 bn capital call, the holdco ends with 747 bn cash (higher than baseline due to suspended shareholder dividends).

- **Severe stress exceeds capacity:** A 484 bn capital call depletes most reserves. While ending cash is still positive (288 bn), this leaves no buffer for continued stress or other contingencies.

- **Oil crisis is not self-funded:** A 957 bn capital call would require external funding (equity raise or debt), as it exceeds the holdco's total available resources.

---

## 3. Capital Call Capacity Analysis

### 3.1 Maximum Absorbable Capital Calls

| Dividend Scenario | Max Capital Call |
|-------------------|------------------|
| Full dividends + full shareholder payout | ~324 bn |
| Full dividends + no shareholder payout | ~324 bn |
| No bank dividend + no shareholder payout | ~324 bn |
| Half bank dividend + no shareholder payout | ~324 bn |

**Note:** The convergence to ~324 bn across scenarios reflects the starting cash position as the binding constraint. The 12-month simulation shows that dividend suspension creates substantial headroom, but the initial cash stock sets the ceiling for immediate capital calls.

### 3.2 Capacity Under Extended Stress

For multi-year stress scenarios, the analysis changes:

| Year | Cumulative Capacity (no bank div, no sh. div) |
|------|----------------------------------------------|
| Year 1 | 324 + 467 - 21 = 770 bn |
| Year 2 | 770 + 467 - 21 = 1,216 bn |
| Year 3 | 1,216 + 467 - 21 = 1,662 bn |

**Interpretation:** If other subsidiaries continue paying dividends (~467 bn/year) while shareholder dividends and bank dividends are suspended, the holdco can fund progressively larger capital calls over time. This provides significant flexibility for phased recapitalization.

---

## 4. Risk Assessment

### 4.1 Key Vulnerabilities

1. **Tight K2 Buffer:** The 70 bps buffer above minimum means even modest stress triggers dividend restrictions. This is a structural feature of Kaspi's high-RWA consumer lending focus.

2. **Concentrated Dividend Dependence:** While diversified, 38% of dividend income comes from the bank. A prolonged banking stress would impact group cash generation.

3. **Correlation Risk:** An oil crisis scenario would likely stress both the bank (credit losses) and non-bank subsidiaries (reduced consumer spending on marketplace/payments), potentially reducing the assumed 467 bn from other subs.

### 4.2 Mitigating Factors

1. **Strong Starting Position:** 325 bn cash with 187 months runway provides substantial cushion.

2. **Low Fixed Costs:** Only 21 bn annual G&A means the holding company has very low cash burn.

3. **Dividend Flexibility:** Ability to suspend shareholder dividends immediately frees 646 bn annually.

4. **Diversified Subsidiaries:** Non-bank businesses (payments, marketplace) have different risk profiles than banking.

5. **No Parent Debt:** Zero interest expense and minimal liabilities means no debt service pressure.

---

## 5. Scenario Probability Assessment

| Scenario | Probability | Trigger Events |
|----------|-------------|----------------|
| Baseline | 70% | Normal economic conditions |
| Mild | 20% | Moderate slowdown, tenge depreciation |
| Moderate | 7% | Recession, commodity price drop |
| Severe | 2.5% | Major crisis (2015-16 magnitude) |
| Oil Crisis | 0.5% | Prolonged $30 oil, banking crisis |

**Expected Loss Analysis:**

Using probability-weighted outcomes:
- Expected capital call = 0.70(0) + 0.20(0) + 0.07(24) + 0.025(484) + 0.005(957) = **18.6 KZT bn**
- This is well within the holdco's capacity

---

## 6. Recommendations

### 6.1 Capital Management

1. **Maintain Current Cash Position:** The 325 bn cash buffer is appropriate given the bank's tight K2 ratio. Do not reduce through extraordinary dividends or buybacks.

2. **Consider Contingent Capital:** Establish committed credit facilities or contingent equity arrangements that could be drawn in stress scenarios.

3. **Monitor K2 Buffer:** Track the K2 ratio monthly. If it approaches 12.5%, consider preemptive dividend retention at the bank level.

### 6.2 Stress Testing

1. **Quarterly Updates:** Re-run this analysis quarterly with updated financials.

2. **Reverse Stress Test:** Identify the specific combination of credit losses, RWA inflation, and deposit runs that would exhaust holdco resources.

3. **Correlation Scenarios:** Model scenarios where non-bank subsidiary dividends are also stressed (e.g., 50% reduction in payments/marketplace income).

### 6.3 Disclosure

1. **Investor Communication:** The tight K2 buffer should be contextualized with NBK's conservative RWA methodology. The equivalent Basel III ratio would be approximately 18.5%.

2. **Rating Agency Engagement:** Proactively share this analysis to demonstrate robust holding company liquidity.

---

## 7. Conclusion

Kaspi.kz's holding company structure is well-positioned to absorb moderate stress scenarios without external funding. The combination of substantial cash reserves (325 bn), diversified dividend income (753 bn annually), and low fixed costs (21 bn annually) creates a resilient capital structure.

However, the bank's tight K2 buffer (70 bps) means that dividend restrictions would be triggered earlier than the headline ratio suggests. In severe stress scenarios (5%+ credit losses with 15%+ RWA inflation), the holdco would require external funding support.

**Bottom Line:** The holding company can self-fund capital calls up to approximately **770 KZT bn over 12 months** (if all discretionary outflows are suspended), covering all but the most extreme tail scenarios. For oil crisis-level stress (7% credit losses, 25% RWA inflation), external capital would be required.

---

## Appendix: Model Parameters

### Passthrough Configuration

```
Dividend Payout Formula:
  payout = ((k2_stressed - 0.12) / (0.127 - 0.12)) ^ 1.0

  At K2 = 12.7%: payout = 100%
  At K2 = 12.35%: payout = 50%
  At K2 = 12.0%: payout = 0%

Capital Call Formula:
  call = shortfall × 1.0 (no buffer multiplier)
```

### Data Sources

- Bank regulatory ratios: FY2024 20-F, NBK filings
- Parent cash and G&A: FY2024 20-F, parent-only statements
- Dividend flows: FY2024 20-F, cash flow statement
- RWA methodology: NBK prudential regulations
