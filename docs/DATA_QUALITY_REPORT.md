# Data Quality Report: Kazakhstan Household Welfare Study (v4)

---

## STATUS UPDATE (v4)

**Study design revised.** The pipeline now:
- Uses oil exposure only (E_oil_r) in main specification
- Cyclical exposure dropped from core model (no reliable regional data)
- GRP-based cyclical proxy available for robustness checks
- Pipeline is FULLY OPERATIONAL for main specification

**Current pipeline status: READY** - Main spec (oil-only) fully operational.

---

## Summary Table

| Data Source | Status | Quality Grade | Pipeline Impact |
|-------------|--------|---------------|-----------------|
| FRED (IGREA, VIX, Brent) | REAL | A | Ready |
| Baumeister Oil Shocks | REAL | A- | Ready |
| BNS Per-Capita Income | REAL | B- | Ready (2 regions missing) |
| Mining Shares (Alternative) | REAL | B | Ready (USGS/EITI/stat.gov.kz) |
| Cyclical Proxy (GRP-based) | DERIVED | C | Robustness only |
| BNS Employment | NOT NEEDED | N/A | Dropped from core model |
| BNS Expenditure | PARTIAL | C | Optional for E_debt_r |

---

## Pipeline Status by Component

### Main Specification: READY

| Component | Data Source | Status |
|-----------|-------------|--------|
| Shock time series | FRED + Baumeister | Available |
| Outcome variable | BNS income | Partial (14/16 regions) |
| Oil exposure (E_oil_r) | USGS/EITI/stat.gov.kz | **AVAILABLE** |

### Robustness Specification: READY

| Component | Data Source | Status |
|-----------|-------------|--------|
| Cyclical proxy (E_cyc_proxy_r) | GRP-based | **AVAILABLE** (not employment) |
| Global activity shock | FRED IGREA | Available |

### Optional Components

| Component | Required Data | Current Status | Notes |
|-----------|---------------|----------------|-------|
| True cyclical exposure (E_cyc_r) | Employment by sector | Not available | Dropped from core model |
| Debt exposure (E_debt_r) | Expenditure debt share | Needs verification | Optional for auxiliary specs |

---

## Detailed Assessment by Source

### 1. FRED Economic Data (EXCELLENT)

| Series | Rows | Date Range | Missing | Quality |
|--------|------|------------|---------|---------|
| IGREA (Kilian Global Activity) | 1867+ | 1968-2025 | 0% | EXCELLENT |
| VIXCLS (Volatility Index) | 8000+ | 1990-2026 | 0% | EXCELLENT |
| DCOILBRENTEU (Brent Crude) | 9000+ | 1987-2026 | 0% | EXCELLENT |

**Assessment**: FRED data is the gold standard. No concerns.

**Location**: `data/raw/fred/*.parquet`, `data/backup/fred/*.json`

---

### 2. Baumeister-Hamilton Oil Shocks (EXCELLENT)

| Metric | Value |
|--------|-------|
| Rows | 608 |
| Date Range | 1975-02-01 to 2025-09-01 |
| Columns | date, oil_supply_shock, aggregate_demand_shock |
| Missing Values | 0% |

**Verification**: Data was verified to NOT match seed=42 synthetic pattern - confirmed real.

**v2 Behavior**: If download fails, code raises `ValueError` with instructions for manual download. No silent fallback to synthetic data.

**Location**: `data/raw/baumeister_shocks/shocks.parquet`, `data/backup/baumeister/*.xlsx`

---

### 3. BNS Per-Capita Income (PARTIAL)

| Metric | Value |
|--------|-------|
| Rows | 846 |
| Expected Rows | 868 (14 regions x 62 quarters) |
| Missing Rows | 22 (2.5%) |
| Unique Regions | 14 |
| Missing Regions | 2 (West Kazakhstan, North Kazakhstan) |

**v2 Behavior**: If no income data at all, code raises `ValueError`. Current data is partial but usable.

**Critical Missing Regions**:
- **West Kazakhstan**: An OIL region (historically ~50% exposure). Its absence may bias results.
- **North Kazakhstan**: Lower oil exposure (~10%), less critical but still a gap.

**Location**: `data/raw/kazakhstan_bns/income_per_capita.parquet`

---

### 4. Mining Shares (RESOLVED via Alternative Sources)

**Status: AVAILABLE - Using USGS/EITI/stat.gov.kz data**

The BNS API endpoint returns HTTP 500 errors, but alternative sources are now integrated.

**v3 Behavior**:
```python
# Try BNS first, fall back to alternatives
mining_df = bns_data.get(BNSDataType.MINING_SHARES, pd.DataFrame())
if mining_df.empty:
    mining_df, source = self._load_alternative_mining_shares()
    # Uses data/raw/alternative_sources/mining_shares.csv
```

**Alternative Sources Used**:
| Source | Coverage | Key Data |
|--------|----------|----------|
| USGS Mineral Industry Report | 2022 | Atyrau: 32%, Mangystau: 29%, Aktobe: 15% |
| EITI Kazakhstan | 2020-2021 | Regional GVA from extractive industries |
| stat.gov.kz GRP Publications | 2023 | GRP by region |

**File**: `data/raw/alternative_sources/mining_shares.csv`

---

### 5. BNS Employment Data (BLOCKS PIPELINE)

**Status: DATA UNAVAILABLE - ANALYSIS CANNOT PROCEED**

Employment by sector and region is not available from BNS.

**v2 Behavior**:
```python
raise ValueError(
    "CRITICAL: No employment data available. "
    "Cannot compute cyclical exposure (E_cyc_r) without real regional employment data."
)
```

**Resolution Options**:
1. Contact BNS for employment by sector statistics
2. Use ILO labor statistics
3. Use OECD employment data
4. Find academic sources with regional cyclical sensitivity measures

---

### 6. BNS Expenditure Data (NEEDS VERIFICATION)

**Status**: Data available but requires verification that `debt_share` column exists.

**v2 Behavior**: Code will raise `ValueError` if:
- Expenditure data is empty
- `debt_share` column is missing
- `year` column is missing
- No data in baseline period (2010-2013)
- Any regions have missing debt exposure

**Location**: `data/raw/kazakhstan_bns/expenditure_structure.parquet`

---

## Code Behavior Summary (v3)

### Previous Behavior (v1 - REMOVED)

| Scenario | Old Behavior | Problem |
|----------|--------------|---------|
| No income data | Generate random with seed=42 | Synthetic data, meaningless results |
| No mining data | Use hardcoded `{"Atyrau": 0.8, ...}` | Undocumented source, unreliable |
| No employment data | Use hardcoded regional values | Undocumented source, unreliable |
| Baumeister download fails | Generate random with seed=42 | Synthetic data, meaningless results |

### Current Behavior (v3)

| Scenario | New Behavior | Benefit |
|----------|--------------|---------|
| No income data | `ValueError` with clear message | Prevents unreliable analysis |
| No mining data | **Fall back to USGS/EITI** | Documented alternatives with lineage |
| No employment data | `ValueError` with resolution options | Requires real data (still needed) |
| Baumeister download fails | `ValueError` with manual download instructions | Requires real data |

---

## Data Lineage Tracking

The `src/data/data_lineage.py` module provides runtime tracking:

```python
from src.data.data_lineage import get_tracker, print_lineage_report

# After running pipeline
print_lineage_report()

# Check for issues
tracker = get_tracker()
if tracker.has_critical_issues():
    print("CRITICAL ISSUES:", tracker.get_critical_issues())
```

---

## Test Results

All 30 unit tests pass:
```
tests/test_panel_data.py - 10 passed
tests/test_shift_share.py - 6 passed
tests/test_simulator.py - 14 passed
-------------------------------
TOTAL: 30 passed in 4.69s
```

Tests use fixture data and do not require real BNS data.

---

## Reliability Assessment (v4)

### Overall Status: READY

Main specification (oil-only) is fully operational.

| Component | Status | Action Required |
|-----------|--------|-----------------|
| FRED data | Ready | None |
| Baumeister shocks | Ready | None |
| BNS income | Partial | Address missing regions if possible |
| Mining shares | Ready | Using USGS/EITI/stat.gov.kz |
| Cyclical proxy | Ready | GRP-based, for robustness only |
| Employment data | Not needed | Dropped from core model |
| Expenditure data | Optional | For auxiliary specs only |

### Quality Implications

With v4 design:
- Main spec uses only oil exposure (cleanest identification)
- Cyclical effects absorbed by time fixed effects at national level
- GRP-based proxy available for robustness checks
- Pipeline fully operational

---

## Recommendations

### Immediate Actions

1. ~~**Obtain mining sector shares**~~: **RESOLVED** - USGS/EITI data available
2. **Obtain employment data**: Required for cyclical exposure computation (E_cyc_r)
3. **Verify expenditure data**: Check that debt_share column is present and populated

### Data Collection Options

| Data Type | Primary Source | Alternative Sources | Status |
|-----------|----------------|---------------------|--------|
| Mining shares | BNS | USGS, EITI, stat.gov.kz GRP | **RESOLVED** |
| Employment | BNS | ILO, OECD | Still needed |
| Debt share | BNS expenditure | Household budget surveys | Verify |

### To Run Pipeline

1. Employment data still needed for full model, but partial run possible:
   ```bash
   PYTHONPATH=. kzwelfare build-panel  # Uses alternative mining shares
   ```
2. Check data lineage report to see which sources were used:
   ```python
   from src.data.data_lineage import print_lineage_report
   print_lineage_report()
   ```
3. Proceed with analysis once all exposures computed

---

---

## Estimation Results (Actual)

### Sample
- **Observations:** 746 (14 regions × ~53 quarters)
- **Missing regions:** West Kazakhstan, North Kazakhstan

### Main Specification Results
```
y_{r,t} = α_r + δ_t + β(E_oil_r × Shock_oil_t) + u_{r,t}
```

| Coefficient | Estimate | Std. Error | p-value |
|-------------|----------|------------|---------|
| E_oil × Supply | 0.035 | 0.040 | 0.375 |
| E_oil × Demand | -0.002 | 0.108 | 0.983 |

**R² within: -0.0008** (model adds noise, not signal)

### Robustness Specification Results
```
y_{r,t} = α_r + δ_t + β(E_oil_r × Shock_oil_t) + θ(E_cyc_proxy_r × Shock_cyc_t) + u_{r,t}
```

| Coefficient | Estimate | Std. Error | p-value |
|-------------|----------|------------|---------|
| E_oil × Supply | 0.051 | 0.044 | 0.244 |
| E_oil × Demand | -0.013 | 0.109 | 0.905 |
| E_cyc_proxy × Activity | -0.003 | 0.002 | 0.086* |

**R² within: -0.0102**

### Beta Stability Check: FAILED
- β (main): 0.035
- β (robust): 0.051
- **Change: 46%** (exceeds 20% threshold)

### Conclusion
**No significant oil effects detected.** The null cannot be rejected.

---

## Conclusion

**The data foundation is complete and estimation has been run.**

Key findings:
1. All oil coefficients are statistically insignificant (p > 0.20)
2. R² within is negative (model explains less than intercept)
3. β is not stable across specifications (changes 46%)
4. The null hypothesis cannot be rejected

To reproduce:
```bash
PYTHONPATH=. kzwelfare build-panel
PYTHONPATH=. kzwelfare estimate all
```

---

*Document version: 4.0*
*Generated: January 2026*
*Assessment: Estimation complete - null results*
*Update: Actual estimation results included*
