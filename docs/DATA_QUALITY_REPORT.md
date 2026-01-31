# Data Quality Report: Kazakhstan Household Welfare Study

---

## CRITICAL WARNING

**The data foundation of this study has significant quality issues. Results based on this data should be treated as preliminary and potentially unreliable.**

---

## Summary Table

| Data Source | Status | Quality Grade | Critical Issues |
|-------------|--------|---------------|-----------------|
| FRED (IGREA, VIX, Brent) | REAL | A | None - high quality federal data |
| Baumeister Oil Shocks | REAL* | B+ | Downloaded successfully; verify source periodically |
| BNS Per-Capita Income | REAL | C | Missing 2 regions (West Kaz, North Kaz); 846/868 rows |
| BNS Mining Shares | MISSING | F | API returns 500 errors; **HARDCODED VALUES** |
| BNS Employment | MISSING | F | Not available; **HARDCODED VALUES** |
| BNS Expenditure | PARTIAL | C | Limited data available |
| Oil Exposure (E_oil_r) | CONSTRUCTED | D | **No empirical basis documented** |
| Cyclical Exposure (E_cyc_r) | CONSTRUCTED | D | **No empirical basis documented** |

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

### 2. Baumeister-Hamilton Oil Shocks (GOOD with Caveats)

| Metric | Value |
|--------|-------|
| Rows | 608 |
| Date Range | 1975-02-01 to 2025-09-01 |
| Columns | date, oil_supply_shock, aggregate_demand_shock |
| Missing Values | 0% |

**Verification**: The data does NOT match the seed=42 synthetic pattern:
```python
# Checked: First values differ from np.random.seed(42) output
# Data appears to be genuine Baumeister series
```

**Caveat**: The data was downloaded from Google Drive links. The `baumeister_loader.py` has a silent fallback:

```python
# baumeister_loader.py:146-147
logger.warning("Could not fetch Baumeister data, generating placeholder")
return self._generate_placeholder_data()  # Uses np.random.seed(42)!
```

**Risk**: If the download fails silently in production, ENTIRELY SYNTHETIC DATA (deterministic random with seed=42) will be used without the user knowing.

**Recommendation**:
1. Add explicit data lineage tracking
2. Fail loudly when real data unavailable
3. Never silently substitute synthetic data

**Location**: `data/raw/baumeister_shocks/shocks.parquet`, `data/backup/baumeister/*.xlsx`

---

### 3. BNS Per-Capita Income (PROBLEMATIC)

| Metric | Value |
|--------|-------|
| Rows | 846 |
| Expected Rows | 868 (14 regions x 62 quarters) |
| Missing Rows | 22 (2.5%) |
| Unique Regions | 14 |
| Missing Regions | 2 (West Kazakhstan, North Kazakhstan) |
| Unique Quarters | 62 |

**Problem Details**:

| Missing Data Type | Count | Impact |
|-------------------|-------|--------|
| West Kazakhstan region | All quarters | This is an OIL region (50% exposure). Losing it biases results. |
| North Kazakhstan region | All quarters | Low oil exposure (10%). Less critical but still a gap. |

**Data Lineage**:
- Source: BNS iblock element downloads
- Files: `data/raw/kazakhstan_bns/income_per_capita.parquet`
- Processing: Region names standardized, harmonized to stable geography

**Fallback Behavior** (panel_data.py:232-239):
```python
if income_df.empty:
    logger.warning("No income data available, using placeholder")
    np.random.seed(42)
    panel["income_pc"] = np.exp(10 + np.random.randn(len(panel)) * 0.3)
```

**Risk**: If BNS download fails, SYNTHETIC INCOME DATA will be used.

---

### 4. BNS Mining Shares (CRITICAL FAILURE)

**Status**: DATA NOT AVAILABLE

The BNS API endpoint for mining sector statistics returns HTTP 500 errors. This data is critical for computing oil exposure.

**Current Workaround** (panel_data.py:321-329):
```python
if mining_df.empty:
    logger.warning("No mining data available, using stylized exposure")
    oil_regions = {
        "Atyrau": 0.8,
        "Mangystau": 0.7,
        "West Kazakhstan": 0.5,
        "Kyzylorda": 0.3,
        "Aktobe": 0.25,
    }
    panel["E_oil_r"] = panel["region"].map(oil_regions).fillna(0.05)
```

**Problems with Hardcoded Values**:

| Issue | Description |
|-------|-------------|
| No source citation | Where do these numbers come from? Academic paper? Government report? Expert guess? |
| No time variation | True mining shares likely changed 2010-2024 |
| Binary treatment | Treating all non-listed regions as 5% may not reflect reality |
| Cannot validate | Without real data, we cannot verify these are accurate |

**Impact on Results**: The entire shift-share identification depends on accurate exposure measurement. If exposures are wrong, the interaction terms (E_oil_r x shock) are wrong, and all estimates are biased.

---

### 5. BNS Employment Data (CRITICAL FAILURE)

**Status**: DATA NOT AVAILABLE

Employment data by sector and region is unavailable from BNS.

**Current Workaround** (panel_data.py:397-416):
```python
if employment_df.empty:
    logger.warning("No employment data available, using placeholder")
    cyclical_exposures = {
        "Almaty City": 0.55,
        "Astana": 0.50,
        "Karaganda": 0.40,
        # ... etc
    }
    panel["E_cyc_r"] = panel["region"].map(cyclical_exposures).fillna(0.3)
```

Same problems as mining shares: no source, no validation, hardcoded guesses.

---

## Data Lineage Summary

### What Is Real

| Variable | Source | Verified |
|----------|--------|----------|
| oil_supply_shock | Baumeister Google Drive | Yes (pattern check) |
| aggregate_demand_shock | Baumeister Google Drive | Yes (pattern check) |
| global_activity_shock | FRED IGREA | Yes |
| vix_shock | FRED VIXCLS | Yes |
| brent_shock | FRED DCOILBRENTEU | Yes |
| income_pc (partial) | BNS iblock | Yes (14/16 regions) |

### What Is Hardcoded/Synthetic

| Variable | Source | Evidence |
|----------|--------|----------|
| E_oil_r | Hardcoded in panel_data.py | "stylized exposure" comment |
| E_cyc_r | Hardcoded in panel_data.py | "placeholder" comment |
| E_debt_r | Constant 0.1 | No data available |
| income_pc (fallback) | np.random.seed(42) | If BNS fails |
| shock data (fallback) | np.random.seed(42) | If Baumeister fails |

---

## Missing Data Analysis

### Region-Quarter Coverage

```
Expected panel size: 16 regions x 60 quarters (2010-2024) = 960 cells
Actual income data: 846 cells with real data
Missing: 114 cells (11.9%)
```

### Missing Pattern

| Region | Status | Notes |
|--------|--------|-------|
| Akmola | Complete | 62/62 quarters |
| Aktobe | Complete | 62/62 quarters |
| Almaty City | Complete | 62/62 quarters |
| Almaty Region | Complete | 62/62 quarters |
| Astana | Complete | 62/62 quarters |
| Atyrau | Complete | 62/62 quarters |
| East Kazakhstan | Complete | 62/62 quarters |
| Jambyl | Complete | 62/62 quarters |
| Karaganda | Complete | 62/62 quarters |
| Kostanay | Complete | 62/62 quarters |
| Kyzylorda | Complete | 62/62 quarters |
| Mangystau | Complete | 62/62 quarters |
| Pavlodar | Complete | 62/62 quarters |
| South Kazakhstan | Complete | 62/62 quarters |
| **West Kazakhstan** | **MISSING** | **0/62 quarters - OIL REGION** |
| **North Kazakhstan** | **MISSING** | **0/62 quarters** |

**Critical**: West Kazakhstan (50% oil exposure) is entirely missing. This is one of the most oil-dependent regions and its absence biases results.

---

## Reliability Assessment

### Overall Data Quality Score: D+

| Component | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| Shock data quality | 30% | A- | 0.26 |
| Income data completeness | 25% | C | 0.18 |
| Exposure measurement | 30% | F | 0.00 |
| Employment data | 10% | F | 0.00 |
| Documentation | 5% | C | 0.04 |
| **Total** | 100% | | **0.48 (D+)** |

### Implications for Results

1. **Baseline regressions**: Compromised by hardcoded exposures and missing regions
2. **Local projections**: Same issues
3. **Scenario simulations**: Unreliable multipliers lead to unreliable predictions
4. **Falsification tests**: May pass spuriously if exposures are wrong

---

## Recommendations

### Immediate Actions

1. **Add data lineage tracking**: Create `data_lineage.py` to log which data is real vs. synthetic
2. **Fail loudly**: Remove silent fallbacks to synthetic data
3. **Document exposure sources**: If using hardcoded values, cite the source

### Data Collection

1. **Contact BNS directly**: Request mining sector shares by region
2. **Use alternative sources**: OECD regional statistics, World Bank, academic papers
3. **Web archive**: Check if BNS had working endpoints in the past

### Validation

1. **Cross-validate exposures**: Compare hardcoded values to any available sources
2. **Sensitivity analysis**: Test how results change with different exposure assumptions
3. **Missing data imputation**: Consider multiple imputation for missing income data

---

## Appendix: Code Locations of Fallbacks

| File | Line | Fallback Type |
|------|------|---------------|
| `baumeister_loader.py` | 146-147 | Synthetic shocks with seed=42 |
| `panel_data.py` | 232-239 | Synthetic income with seed=42 |
| `panel_data.py` | 321-329 | Hardcoded oil exposures |
| `panel_data.py` | 362-379 | Constant debt exposure (0.1) |
| `panel_data.py` | 397-416 | Hardcoded cyclical exposures |

---

## Conclusion

**The data foundation is NOT firm.**

While some components (FRED, Baumeister shocks) are high quality, the critical exposure variables are hardcoded without empirical basis, and significant income data is missing. Results from this study should be considered exploratory, not definitive.

Any claims of causal effects or policy-relevant multipliers must be qualified with these data limitations.

---

*Report generated: January 2026*
*Assessment: Brutally honest*
