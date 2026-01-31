# Kazakhstan FX Passthrough Study - Estimation Results

**Generated:** January 31, 2026
**Study Version:** 2.0
**Status:** PENDING - Awaiting BNS CPI Category Data

---

## Current Status

### Data Availability

| Data Source | Status | Observations | Date Range |
|-------------|--------|--------------|------------|
| NBK USD/KZT Exchange Rate | **Downloaded** | 192 monthly | 2010-01 to 2025-12 |
| BNS National Income | **Downloaded** | 1,126 | 2010-Q1 to 2025-Q3 |
| FRED (IGREA, VIX, Brent) | **Downloaded** | 300-6800 | Various |
| Baumeister Oil Shocks | **Downloaded** | 608 | 1975-02 to 2025-09 |
| **BNS CPI Categories** | **API ERROR** | - | - |
| World Bank REER | Failed (optional) | - | - |

### API Issue

The Kazakhstan Bureau of National Statistics (BNS) API is currently returning **500 Internal Server Error** for CPI category data requests. This is a temporary server-side issue.

- **Endpoint:** `https://stat.gov.kz/api/iblock/element/49140/csv/file/en/`
- **Fallback page:** `https://stat.gov.kz/en/industries/prices/stat-official-ind-prices/` (also returning 500)

### Impact on Analysis

The CPI category data is **required** for:
- Block A: CPI Pass-Through estimation (main identification)
- Falsification tests: Admin price exclusion, pre-trends

Without this data, the full causal chain analysis cannot be completed.

---

## Previous Results Invalidated

The results previously shown in this document were based on **synthetic CPI data** that was inadvertently generated when the BNS API failed in an earlier session.

### Why the Admin Price Test Failed

The synthetic data was generated with **uniform pass-through applied to ALL categories**, including administered prices. This violated the falsification test assumption that admin prices (utilities, education, healthcare, communications) should NOT respond to FX shocks.

### Expected Behavior with Real Data

With real BNS CPI data:
- **Tradable categories** (food, clothing, electronics): Should show FX pass-through
- **Admin price categories** (utilities, education, health): Should show NO pass-through
- **Falsification test**: Should PASS

---

## Available Analysis Components

### Income Series (Built)

The national income time series has been successfully built:

```
Observations: 1,126
Columns: date, nominal_income, wage_income, transfer_income,
         nominal_income_growth, wage_income_share, transfer_income_share
Date range: 2010-Q1 to 2025-Q3
```

This enables partial analysis of Blocks B-E once CPI data is available for constructing the inflation instrument.

### Exchange Rate Data (Ready)

Monthly USD/KZT exchange rates from NBK:

```
Observations: 192 monthly
Range: 2010-01-01 to 2025-12-01
Source: National Bank of Kazakhstan (NBK)
```

Key dates in the series:
- 2010-01: 148.46 KZT/USD
- 2015-08: 188.38 KZT/USD (pre-float)
- 2015-09: 270.42 KZT/USD (post-float)
- 2025-12: 512.53 KZT/USD

---

## Next Steps

### Option 1: Wait for API Recovery

```bash
# Check API status periodically
curl -I https://stat.gov.kz/en/industries/prices/stat-official-ind-prices/

# Once working, download data
PYTHONPATH=. python scripts/download_all_data.py

# Run full analysis
kzresearch passthrough run-full-chain
```

### Option 2: Manual Download

1. Visit: https://stat.gov.kz/en/industries/prices/stat-official-ind-prices/
2. Download CPI by category data (COICOP divisions)
3. Save as: `data/raw/kazakhstan_bns/cpi_categories.parquet`
4. Run: `kzresearch passthrough run-full-chain`

### Data Format Required

The CPI categories data should have columns:
- `date`: Monthly date
- `category`: COICOP code (01-12)
- `cpi_index`: Index level
- `inflation_mom`: Month-over-month inflation rate

---

## Technical Notes

### Synthetic Data Detection

Signs that data is synthetic (to avoid in future):
1. Uniform standard deviations across all categories
2. All categories show similar pass-through patterns
3. Admin prices respond to FX shocks (should not)

### Real Data Characteristics

Real BNS CPI data should show:
1. **Heterogeneous volatility**: Food more volatile than services
2. **Differential pass-through**: High-import categories (clothing) > low-import (services)
3. **Admin price stability**: Utilities, education show minimal FX response

---

## Verification Commands

Once data is available:

```bash
# 1. Check download status
cat data/metadata/download_status.json | grep -E "status|n_rows"

# 2. Verify no synthetic data
cat data/metadata/download_status.json | grep synthetic
# Should return nothing

# 3. Check CPI volatility heterogeneity
python3 -c "
import pandas as pd
df = pd.read_parquet('data/raw/kazakhstan_bns/cpi_categories.parquet')
print(df.groupby('category')['inflation_mom'].std().sort_values(ascending=False))
"

# 4. Run falsification tests
kzresearch passthrough falsification
# Should show: Admin prices: PASS
```

---

*Last updated: January 31, 2026*
*Status: Awaiting BNS API recovery*
*Kazakhstan Econometric Research Platform*
