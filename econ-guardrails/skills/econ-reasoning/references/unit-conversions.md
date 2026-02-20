# Unit Conversion Tables for Economics Analysis

## Volume & Production (Oil & Gas)

| From | To | Factor |
|------|----|--------|
| 1 barrel (bbl) | 42 US gallons | × 42 |
| 1 barrel (bbl) | 159 liters | × 159 |
| 1 mb/d (million barrels per day) | barrels per year | × 365,000,000 |
| 1 mb/d | million barrels per year | × 365 |
| 1 boe (barrel of oil equivalent) | ~5,800 cubic feet natural gas | — |
| 1 boe | ~1,700 kWh | — |

### Critical Reminder
- **mb/d is a flow rate**. Revenue = price × mb/d × 365 for annual.
- Do NOT multiply reserves (stock) by price to get revenue (flow).

## Rate Conversions

| From | To | Formula |
|------|----|---------|
| Percentage change (%) | Percentage points (pp) | Not interchangeable! |
| Basis points (bps) | Percentage points (pp) | ÷ 100 |
| Percentage points (pp) | Basis points (bps) | × 100 |
| Log points | Approximate percentage (%) | ≈ same for small changes (<10%) |
| Annual rate | Quarterly rate | (1 + r_annual)^(1/4) - 1 |
| Annual rate | Monthly rate | (1 + r_annual)^(1/12) - 1 |

### Percentage vs Percentage Points
- "Inflation rose from 3% to 5%": that is a **2 percentage point** increase
- "Inflation rose by 67%": that means from 3% to 5.01% (3% × 1.67)
- These are NOT the same. Confusing them is a common and serious error.

## Energy Equivalents

| Unit | Energy Content |
|------|---------------|
| 1 barrel crude oil | ~5.8 million BTU |
| 1 cubic foot natural gas | ~1,032 BTU |
| 1 metric ton oil equivalent (toe) | ~7.33 barrels |
| 1 million BTU | ~0.172 barrels oil equivalent |

## Price Deflation

| Deflator | Use Case | Source |
|----------|----------|--------|
| CPI (Consumer Price Index) | Consumer purchasing power, wage comparisons | BLS / national statistics |
| GDP deflator | Broad economy-wide price level | National accounts |
| PPI (Producer Price Index) | Industrial/production cost comparisons | BLS / national statistics |
| Oil-specific deflator | Real oil price analysis | Typically CPI or GDP deflator |

### Real Price Formula
```
Real Price (base year) = Nominal Price × (Deflator_base / Deflator_current)
```

### Common Benchmarks
- $1 in 2000 ≈ $1.82 in 2024 (US CPI)
- $1 in 2010 ≈ $1.41 in 2024 (US CPI)
- Always state the base year when quoting real prices

## Dimensional Analysis Rules

1. **Both sides of an equation must have the same units**
   - $/day × days = $ ✓
   - $/day × barrels = undefined ✗

2. **Elasticities are dimensionless** (% change / % change)
   - If your elasticity has units, something is wrong

3. **Multipliers have units of (output / input)**
   - Fiscal multiplier: $GDP / $spending (dimensionless if same currency)
   - Pass-through: %CPI / %exchange rate (dimensionless)

4. **Rates vs levels must not be mixed in calculations**
   - Cannot add a flow ($/year) to a stock ($)
   - Cannot compare GDP growth rate (%) to GDP level ($)

5. **Cross-unit multiplication requires conversion factors**
   - mb/d × $/barrel = M$/day (must convert to annual if comparing to yearly GDP)
