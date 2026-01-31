# Data Sources Documentation (v4)

## Overview

This document provides comprehensive documentation for all data sources used in the Kazakhstan Household Welfare study.

**v4 Update**: Study design revised. Main spec uses oil exposure only (E_oil_r). Cyclical exposure dropped from core model; GRP-based proxy available for robustness checks.

---

## 1. FRED Economic Data

### 1.1 Global Real Economic Activity Index (IGREA)

| Attribute | Value |
|-----------|-------|
| **Source** | Federal Reserve Economic Data (FRED) |
| **Series ID** | IGREA |
| **Description** | Kilian Global Real Economic Activity Index |
| **Frequency** | Monthly |
| **Date Range** | 1968-01-01 to present |
| **Units** | Index |
| **Access Method** | `fredapi` Python library |
| **API Endpoint** | `https://api.stlouisfed.org/fred/series/observations` |
| **Status** | Available |

**Transformation Applied**:
- AR(1) residual computed as innovation (global_activity_shock)
- Aggregated to quarterly frequency using mean

**Citation**:
> Kilian, L. (2009). "Not All Oil Price Shocks Are Alike: Disentangling Demand and Supply Shocks in the Crude Oil Market." American Economic Review, 99(3), 1053-1069.

### 1.2 CBOE Volatility Index (VIX)

| Attribute | Value |
|-----------|-------|
| **Source** | Federal Reserve Economic Data (FRED) |
| **Series ID** | VIXCLS |
| **Description** | CBOE Volatility Index: VIX |
| **Frequency** | Daily |
| **Date Range** | 1990-01-02 to present |
| **Units** | Index |
| **Status** | Available |

**Transformation Applied**:
- AR(1) residual computed as innovation (vix_shock)
- Aggregated to quarterly frequency using mean

### 1.3 Brent Crude Oil Price

| Attribute | Value |
|-----------|-------|
| **Source** | Federal Reserve Economic Data (FRED) |
| **Series ID** | DCOILBRENTEU |
| **Description** | Crude Oil Prices: Brent - Europe |
| **Frequency** | Daily |
| **Date Range** | 1987-05-20 to present |
| **Units** | Dollars per Barrel |
| **Status** | Available |

**Transformation Applied**:
- Log returns computed (brent_shock)
- Aggregated to quarterly frequency using mean

---

## 2. Baumeister-Hamilton Structural Oil Shocks

### 2.1 Oil Supply Shock

| Attribute | Value |
|-----------|-------|
| **Source** | Christiane Baumeister's Research Page |
| **URL** | https://sites.google.com/site/cjsbaumeister/datasets |
| **Description** | Structural oil supply shock from sign-restricted SVAR |
| **Frequency** | Monthly |
| **Date Range** | 1975-02 to 2025-09 |
| **Units** | Standard deviations |
| **Access Method** | Google Drive direct download |
| **File Format** | Excel (.xlsx) |
| **Status** | Available |

**v2 Behavior**: If download fails, code raises `ValueError` with manual download instructions. NO synthetic fallback.

### 2.2 Aggregate Demand Shock

| Attribute | Value |
|-----------|-------|
| **Source** | Christiane Baumeister's Research Page |
| **URL** | https://sites.google.com/site/cjsbaumeister/datasets |
| **Description** | Global aggregate demand shock from sign-restricted SVAR |
| **Frequency** | Monthly |
| **Date Range** | 1975-02 to 2025-09 |
| **Units** | Standard deviations |
| **Status** | Available |

**Citation**:
> Baumeister, C. and Hamilton, J.D. (2019). "Structural Interpretation of Vector Autoregressions with Incomplete Identification: Revisiting the Role of Oil Supply and Demand Shocks." American Economic Review, 109(5), 1873-1910.

---

## 3. Kazakhstan Bureau of National Statistics (BNS)

### 3.1 Per-Capita Monetary Income

| Attribute | Value |
|-----------|-------|
| **Source** | Kazakhstan Bureau of National Statistics |
| **URL** | https://stat.gov.kz |
| **iBlock ID** | 48953 |
| **Description** | Average per capita nominal monetary income of the population |
| **Frequency** | Quarterly |
| **Date Range** | 2010Q1 to 2025Q2 |
| **Units** | Tenge |
| **Status** | Partial (14/16 regions) |

**v2 Behavior**: If no income data available, code raises `ValueError`.

**Missing Regions**:
- West Kazakhstan (oil region - critical gap)
- North Kazakhstan

**File**: `data/raw/kazakhstan_bns/income_per_capita.parquet`

### 3.2 Mining Sector Shares (BLOCKING)

| Attribute | Value |
|-----------|-------|
| **Source** | Kazakhstan Bureau of National Statistics |
| **Required For** | Computing oil exposure (E_oil_r) |
| **Status** | **UNAVAILABLE - BLOCKS PIPELINE** |
| **API Status** | HTTP 500 errors |

**v2 Behavior**:
```python
raise ValueError(
    "CRITICAL: No mining sector data available. "
    "Cannot compute oil exposure (E_oil_r) without real regional mining share data. "
    "The shift-share identification requires measured exposures, not hardcoded values."
)
```

**Resolution Options**:
1. Contact BNS directly
2. Use OECD regional statistics
3. Find academic sources with documented regional oil dependence

### 3.3 Employment by Sector (BLOCKING)

| Attribute | Value |
|-----------|-------|
| **Source** | Kazakhstan Bureau of National Statistics |
| **Required For** | Computing cyclical exposure (E_cyc_r) |
| **Status** | **UNAVAILABLE - BLOCKS PIPELINE** |

**v2 Behavior**:
```python
raise ValueError(
    "CRITICAL: No employment data available. "
    "Cannot compute cyclical exposure (E_cyc_r) without real regional employment data."
)
```

### 3.4 Expenditure Structure

| Attribute | Value |
|-----------|-------|
| **Source** | Kazakhstan Bureau of National Statistics |
| **iBlock ID** | 469805 |
| **Required For** | Computing debt exposure (E_debt_r) |
| **Status** | Needs verification (debt_share column required) |

**v2 Behavior**: Code raises `ValueError` if debt_share column missing.

---

## 4. Alternative Data Sources (for BNS API Failures)

When BNS API endpoints return HTTP 500 errors, the pipeline falls back to these documented alternative sources.

### 4.1 USGS Mineral Industry Reports

| Attribute | Value |
|-----------|-------|
| **Source** | U.S. Geological Survey |
| **URL** | https://pubs.usgs.gov/myb/vol3/2022/myb3-2022-kazakhstan.pdf |
| **Description** | Annual mineral industry survey including oil/gas sector |
| **Coverage** | 2016-2022 (annual PDFs) |
| **Regional Data** | YES - by province |
| **Mining Employment** | YES - 273,000+ workers in 230+ enterprises |
| **Oil/Gas Regional Shares** | YES - documented percentages |
| **Status** | Available |

**Key Regional Data (USGS 2022):**
| Region | Oilfield Service Share | Notes |
|--------|------------------------|-------|
| Atyrau | 32% | Primary oil region |
| Mangystau | 29% | Primary oil region |
| Aktobe | 15% | Secondary oil region |
| West Kazakhstan | 6% | Minor oil region |
| Kyzylorda | 6% | Minor oil region |

**Citation**:
> USGS (2022). "2022 Minerals Yearbook: Kazakhstan." U.S. Geological Survey.

### 4.2 EITI Kazakhstan (Extractive Industries Transparency Initiative)

| Attribute | Value |
|-----------|-------|
| **Source** | Extractive Industries Transparency Initiative |
| **URL** | https://eiti.org/countries/kazakhstan |
| **Description** | Transparency reports on extractive industry revenues |
| **Coverage** | 2005-2021+ |
| **Regional GVA** | YES - Table 80 in 2020-2021 Report |
| **Mining Revenue** | YES - by region |
| **Format** | CSV export, PDF reports |
| **Status** | Available |

**Citation**:
> EITI (2021). "Extractive Industries Transparency Initiative: Kazakhstan Country Report 2020-2021."

### 4.3 stat.gov.kz GRP Publications

| Attribute | Value |
|-----------|-------|
| **Source** | Kazakhstan Bureau of National Statistics |
| **URL** | https://stat.gov.kz/en/industries/economy/national-accounts/publications/ |
| **Description** | Gross Regional Product publications (works when API fails) |
| **Coverage** | Annual, 2010-2023 |
| **Regional GDP Share** | YES |
| **GRP Per Capita** | YES |
| **Status** | Available (publications work even when API down) |

**Key Regional Data (2023):**
| Region | GDP Share | GRP Per Capita | Oil Sector % |
|--------|-----------|----------------|--------------|
| Atyrau | 12.4% | 21,401K tenge (~$44,000) | 65% |
| Almaty City | 20.6% | 11,310K tenge | 0% |
| Mangystau | 4.8% | 9,500K tenge | 55% |
| Karaganda | 7.2% | 5,890K tenge | 12% (coal) |

### 4.4 Alternative Sources File Locations

| Data Type | File Path | Status |
|-----------|-----------|--------|
| Mining shares | `data/raw/alternative_sources/mining_shares.csv` | Available |
| GRP composition | `data/raw/alternative_sources/grp_composition.csv` | Available |
| Documentation | `data/raw/alternative_sources/README.md` | Available |

### 4.5 Fallback Logic

The pipeline implements this fallback strategy:

```
1. Try BNS API (iblock endpoint)
   ├─ Success → Use BNS data
   └─ Failure (HTTP 500) → Continue to step 2

2. Try alternative sources
   ├─ mining_shares.csv exists → Use USGS/EITI data
   └─ File missing → Raise ValueError with download instructions
```

**Code location**: `src/model/panel_data.py:_compute_oil_exposure()`

---

## 5. Region Crosswalk

### 4.1 Geographic Changes

| Reform | Date | Change |
|--------|------|--------|
| Turkestan/Shymkent | June 2018 | South Kazakhstan Oblast split |
| Abay/Zhetysu/Ulytau | June 2022 | Three new regions carved |

### 4.2 Harmonization Mapping

| New Region | Parent Region |
|------------|---------------|
| Abay | East Kazakhstan |
| Zhetysu | Almaty Region |
| Ulytau | Karaganda |
| Turkestan | South Kazakhstan |
| Shymkent | South Kazakhstan |

**File**: `data/crosswalks/regions.csv`

---

## 6. Data Requirements Summary (v4)

### Required for Pipeline to Run

| Data | Variable | Source | Status |
|------|----------|--------|--------|
| Mining shares | E_oil_r | BNS | **BLOCKING** |
| Employment by sector | E_cyc_r | BNS | **BLOCKING** |
| Expenditure debt share | E_debt_r | BNS | Needs verification |
| Per-capita income | log_income_pc | BNS | Partial |
| Oil supply shock | oil_supply_shock | Baumeister | Available |
| Demand shock | aggregate_demand_shock | Baumeister | Available |
| Global activity | global_activity_shock | FRED | Available |

### Pipeline Will NOT Run Until

1. Mining sector shares by region are provided
2. Employment by sector data is provided
3. Expenditure data with debt_share column is verified

---

## 7. File Locations

| Data Type | Raw Data Location | Status |
|-----------|-------------------|--------|
| FRED series | `data/raw/fred/*.parquet` | Available |
| Baumeister shocks | `data/raw/baumeister_shocks/shocks.parquet` | Available |
| BNS income | `data/raw/kazakhstan_bns/income_per_capita.parquet` | Partial |
| BNS expenditure | `data/raw/kazakhstan_bns/expenditure_structure.parquet` | Needs verification |
| BNS mining | N/A | **MISSING** |
| BNS employment | N/A | **MISSING** |
| Region crosswalk | `data/crosswalks/regions.csv` | Available |
| Final panel | `data/processed/panel.parquet` | Cannot build until data obtained |

---

## 8. Access Instructions

### FRED API

1. Register for free API key at https://fred.stlouisfed.org/docs/api/api_key.html
2. Set environment variable: `export FRED_API_KEY=your_key_here`
3. Or add to `.env` file: `FRED_API_KEY=your_key_here`

### Baumeister Data

Data is publicly available at:
https://sites.google.com/site/cjsbaumeister/datasets

If automatic download fails:
1. Manually download supply and demand shock Excel files
2. Place in `data/backup/baumeister/`
3. Pipeline will attempt to load from backup

### BNS Data

No API key required. Access via iblock element IDs.

**For missing data (mining, employment)**:
1. Contact BNS directly: https://stat.gov.kz/contact
2. Request regional mining sector shares and employment by sector
3. Alternative: Use OECD regional statistics

---

## 9. Update Frequency

| Source | Update Frequency | Lag |
|--------|------------------|-----|
| FRED IGREA | Monthly | 1-2 months |
| FRED VIX | Daily | 1 day |
| FRED Brent | Daily | 1 day |
| Baumeister | Monthly | 2-3 months |
| BNS Income | Quarterly | 2-3 months |

---

*Document version: 4.0*
*Generated: January 2026*
*Update: Revised study design - oil exposure only in main spec, cyclical proxy for robustness*
