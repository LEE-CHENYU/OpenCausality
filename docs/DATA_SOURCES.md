# Data Sources Documentation

## Overview

This document provides comprehensive documentation for all data sources used in the Kazakhstan Household Welfare study.

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
| **Access Method** | `fredapi` Python library |

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
| **Access Method** | `fredapi` Python library |

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

**Methodology**:
- Sign-restricted Structural VAR with informative priors
- Posterior median estimates
- Updated monthly with ~2 month lag

### 2.2 Aggregate Demand Shock

| Attribute | Value |
|-----------|-------|
| **Source** | Christiane Baumeister's Research Page |
| **URL** | https://sites.google.com/site/cjsbaumeister/datasets |
| **Description** | Global aggregate demand shock from sign-restricted SVAR |
| **Frequency** | Monthly |
| **Date Range** | 1975-02 to 2025-09 |
| **Units** | Standard deviations |
| **Access Method** | Google Drive direct download |

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
| **Access Method** | REST API with iblock element downloads |
| **File Format** | CSV (tab-delimited) |

**Data Quality Issues**:
- 2 of 16 regions missing (West Kazakhstan, North Kazakhstan)
- Tab-delimited CSV with space as thousands separator
- Region names in English, require harmonization

**API Endpoint**:
```
GET /api/iblock/element/{ID}/csv/file/en/
```

### 3.2 Expenditure Structure

| Attribute | Value |
|-----------|-------|
| **Source** | Kazakhstan Bureau of National Statistics |
| **iBlock ID** | 469805 |
| **Description** | Household expenditure structure by category |
| **Frequency** | Quarterly |
| **Date Range** | 2010Q1 to 2025Q2 |
| **Units** | Percentage |

### 3.3 Mining Sector Shares (UNAVAILABLE)

| Attribute | Value |
|-----------|-------|
| **Source** | Kazakhstan Bureau of National Statistics |
| **Status** | **API returns HTTP 500 errors** |
| **Description** | Mining sector share of regional GRP |
| **Required For** | Computing oil exposure (E_oil_r) |

**Fallback**: Hardcoded stylized values in `panel_data.py:321-329`

### 3.4 Employment by Sector (UNAVAILABLE)

| Attribute | Value |
|-----------|-------|
| **Source** | Kazakhstan Bureau of National Statistics |
| **Status** | **Data not available** |
| **Description** | Employment by sector and region |
| **Required For** | Computing cyclical exposure (E_cyc_r) |

**Fallback**: Hardcoded stylized values in `panel_data.py:397-416`

---

## 4. Region Crosswalk

### 4.1 Geographic Changes

| Reform | Date | Change |
|--------|------|--------|
| Turkestan/Shymkent | June 2018 | South Kazakhstan Oblast split into Turkestan Region and Shymkent City |
| Abay/Zhetysu/Ulytau | June 2022 | Three new regions carved from existing oblasts |

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

## 5. Data Quality Summary

### Overall Assessment

| Source | Verified | Quality | Issues |
|--------|----------|---------|--------|
| FRED | Yes | Excellent | None |
| Baumeister | Yes | Good | Silent fallback to synthetic data |
| BNS Income | Partial | Fair | 2 regions missing |
| BNS Mining | No | Unusable | API errors; hardcoded fallback |
| BNS Employment | No | Unusable | Unavailable; hardcoded fallback |

### Data Lineage

All data downloads are logged in the data lineage system (`src/data/data_lineage.py`). Check the lineage report before running analysis:

```python
from src.data.data_lineage import print_lineage_report
print_lineage_report()
```

---

## 6. Access Instructions

### FRED API

1. Register for free API key at https://fred.stlouisfed.org/docs/api/api_key.html
2. Set environment variable: `export FRED_API_KEY=your_key_here`
3. Or add to `.env` file: `FRED_API_KEY=your_key_here`

### Baumeister Data

Data is publicly available at:
https://sites.google.com/site/cjsbaumeister/datasets

Direct download links (may change):
- Supply shocks: Google Drive file ID `1OsA8btgm2rmDucUFngiLkwv4uywTDmya`
- Demand shocks: Google Drive file ID `1neFXLrIvGwggebQRwjmtrWK-dfQZ9NH8`

### BNS Data

No API key required. Access via:
1. iblock element IDs (discovered from website)
2. Direct file downloads from stat.gov.kz

---

## 7. File Locations

| Data Type | Raw Data Location | Processed Location |
|-----------|-------------------|-------------------|
| FRED series | `data/raw/fred/*.parquet` | Merged into panel |
| Baumeister shocks | `data/raw/baumeister_shocks/shocks.parquet` | Merged into panel |
| BNS income | `data/raw/kazakhstan_bns/income_per_capita.parquet` | Merged into panel |
| BNS expenditure | `data/raw/kazakhstan_bns/expenditure_structure.parquet` | Merged into panel |
| Region crosswalk | `data/crosswalks/regions.csv` | Used during harmonization |
| Final panel | N/A | `data/processed/panel.parquet` |

---

## 8. Update Frequency

| Source | Update Frequency | Lag |
|--------|------------------|-----|
| FRED IGREA | Monthly | 1-2 months |
| FRED VIX | Daily | 1 day |
| FRED Brent | Daily | 1 day |
| Baumeister | Monthly | 2-3 months |
| BNS Income | Quarterly | 2-3 months |

---

*Document generated: January 2026*
