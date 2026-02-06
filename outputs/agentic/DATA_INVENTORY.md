# KSPI K2 DAG: Data Inventory

**Generated:** 2026-02-05
**Purpose:** Track all available data for the 4-bank panel + KSPI stress testing DAG

---

## Summary Statistics

| Bank | Coverage | Periods | Frequency | Data Quality |
|------|----------|---------|-----------|--------------|
| **Kaspi Bank** | 2011Q4-2024Q3 | 27 | Annual (2011-2019) + Quarterly (2020-2024) | High |
| **Halyk Bank** | 2011Q4-2023Q4 | 13 | Annual | Medium-High |
| **ForteBank** | 2017Q4-2023Q4 | 7 | Annual | Medium |
| **BCC** | 2015Q4-2023Q4 | 9 | Annual | Medium |

---

## 1. KSPI/Kaspi Bank Data

### 1.1 Primary Data Files

| File | Description | Coverage | Observations | Status |
|------|-------------|----------|--------------|--------|
| `data/raw/kspi/kaspi_bank_extended_kpis.json` | Extended Kaspi Bank subsidiary data | 2011Q4-2024Q3 | 27 periods | **Complete** |
| `data/raw/kspi/kspi_historical_kpis.json` | KSPI quarterly data (post-IPO) | 2020Q3-2024Q3 | 17 quarters | Complete |
| `data/raw/kspi/kpis_2024Q3.json` | Latest quarter snapshot | 2024Q3 | 1 | Complete |

### 1.2 Kaspi Bank KPIs Available

| KPI | Unit | Coverage | Source | Notes |
|-----|------|----------|--------|-------|
| `net_loans` | bn KZT | 2011-2024 | IFRS/IR | Bank subsidiary level |
| `deposits` | bn KZT | 2011-2024 | IFRS/IR | Bank subsidiary level |
| `npl_ratio` | % | 2011-2024 | IFRS/IR | 90+ DPD |
| `cor` | % | 2011-2024 | IFRS/IR | Cost of Risk |
| `total_capital` | bn KZT | 2011-2024 | NBK prudential | Regulatory capital |
| `rwa` | bn KZT | 2011-2024 | NBK prudential | Risk-weighted assets |
| `k2_ratio` | % | 2011-2024 | NBK prudential | Total capital adequacy |
| `deposit_cost` | % | 2011-2024 | IFRS/IR | Interest expense / avg deposits |
| `payments_revenue` | bn KZT | 2020Q3-2024Q3 | IR | Payments segment only |
| `ppop` | bn KZT | 2020Q3-2024Q3 | IR | Pre-provision operating profit |
| `net_income` | bn KZT | 2020Q3-2024Q3 | IR | Bank segment net income |

### 1.3 Data Quality Notes

- **2011-2019:** Annual frequency only (year-end snapshots). Source: Kaspi Bank JSC annual reports, KASE filings.
- **2020-2024:** True quarterly data from IR presentations and 20-F filings.
- **Entity boundary:** All data at **Kaspi Bank JSC** (subsidiary) level, NOT Kaspi.kz JSC (group consolidated).
- **No interpolation:** `estimation_eligible=true` only for actual reported observations.

---

## 2. Halyk Bank Data

### 2.1 Data File

| File | Description | Coverage | Observations |
|------|-------------|----------|--------------|
| `data/raw/kz_banks/halyk_bank_quarterly.json` | Halyk Bank KPIs | 2011Q4-2023Q4 | 13 years |

### 2.2 KPIs Available

| KPI | Unit | Coverage | Notes |
|-----|------|----------|-------|
| `net_loans` | bn KZT | 2011-2023 | Consolidated (includes Altyn Bank) |
| `deposits` | bn KZT | 2011-2023 | |
| `npl_ratio` | % | 2011-2023 | 90+ DPD per IFRS 9 |
| `cor` | % | 2011-2023 | |
| `total_capital` | bn KZT | 2011-2023 | NBK regulatory |
| `rwa` | bn KZT | 2011-2023 | NBK standardized |
| `deposit_cost` | % | 2011-2023 | |

### 2.3 Data Gaps

- **Missing 2024:** Need to extract 2024 annual report or Q3 interim data
- **Quarterly granularity:** All data is annual frequency

---

## 3. ForteBank Data

### 3.1 Data File

| File | Description | Coverage | Observations |
|------|-------------|----------|--------------|
| `data/raw/kz_banks/fortebank_quarterly.json` | ForteBank KPIs | 2017Q4-2023Q4 | 7 years |

### 3.2 KPIs Available

Same as Halyk: net_loans, deposits, npl_ratio, cor, total_capital, rwa, deposit_cost

### 3.3 Data Gaps

- **Pre-2017:** Bank formed from merger (Alliance Bank, Temirbank, ForteBank) in 2016-2017. No continuous pre-merger data.
- **Missing 2024:** Need 2024 data extraction
- **Quarterly granularity:** All annual

---

## 4. Bank CenterCredit (BCC) Data

### 4.1 Data File

| File | Description | Coverage | Observations |
|------|-------------|----------|--------------|
| `data/raw/kz_banks/bcc_quarterly.json` | BCC KPIs | 2015Q4-2023Q4 | 9 years |

### 4.2 KPIs Available

Same as Halyk: net_loans, deposits, npl_ratio, cor, total_capital, rwa, deposit_cost

### 4.3 Data Gaps

- **Pre-2015:** Could extend with older annual reports
- **Missing 2024:** Need 2024 data extraction
- **Quarterly granularity:** All annual

---

## 5. Exposure Variables (Shift-Share Identification)

### 5.1 Data File

| File | Description |
|------|-------------|
| `data/raw/kz_banks/bank_exposures.json` | Baseline exposure variables for panel LP |

### 5.2 Exposures by Bank

| Bank | E_consumer | E_unsecured | E_demand_dep | E_retail_dep | E_shortterm | Baseline Year |
|------|------------|-------------|--------------|--------------|-------------|---------------|
| Kaspi | 0.85 | 0.70 | 0.55 | 0.80 | 0.60 | 2015 |
| Halyk | 0.35 | 0.20 | 0.40 | 0.55 | 0.35 | 2015 |
| Forte | 0.45 | 0.30 | 0.35 | 0.45 | 0.40 | 2017 |
| BCC | 0.30 | 0.15 | 0.30 | 0.40 | 0.30 | 2015 |

### 5.3 Source

All exposures from IFRS financial statement notes:
- Loan portfolio by segment note
- Deposits by type/maturity note
- Loan maturity profile note

---

## 6. Macro/External Data

### 6.1 Kazakhstan BNS (Bureau of National Statistics)

| File | Description | Coverage | Frequency |
|------|-------------|----------|-----------|
| `data/raw/kazakhstan_bns/cpi_categories.csv` | CPI by category | 2008-2024 | Monthly |
| `data/raw/kazakhstan_bns/cpi_categories.parquet` | CPI index | 2008-2024 | Monthly |
| `data/raw/kazakhstan_bns/income_per_capita.parquet` | Income per capita | 2010-2023 | Annual |
| `data/raw/kazakhstan_bns/expenditure_structure.parquet` | Expenditure structure | 2010-2023 | Annual |
| `data/raw/kazakhstan_bns/grp_by_region.parquet` | GRP by region | 2010-2023 | Annual |
| `data/raw/kazakhstan_bns/national_income.parquet` | National income | 2010-2023 | Quarterly |

### 6.2 NBK (National Bank of Kazakhstan)

| File | Description | Coverage | Frequency |
|------|-------------|----------|-----------|
| `data/raw/nbk/usd_kzt.parquet` | USD/KZT exchange rate | 2000-2024 | Daily |
| `data/processed/nbk_credit_historical.csv` | System NPL/credit | 2013-2024 | Monthly |
| `data/processed/nbk_credit_monthly_2024.csv` | Current credit data | 2024 | Monthly |
| `data/raw/nbk_credit/*.xlsx` | Raw NBK credit reports | 2018-2024 | Monthly |

### 6.3 External (FRED, IMF, World Bank)

| File | Description | Coverage |
|------|-------------|----------|
| `data/raw/fred/DCOILBRENTEU.parquet` | Brent crude oil price | 1987-2024 |
| `data/raw/fred/VIXCLS.parquet` | VIX volatility index | 1990-2024 |
| `data/raw/fred/IGREA.parquet` | Global economic activity | 1990-2024 |
| `data/raw/baumeister_shocks/shocks.parquet` | Oil supply/demand shocks | 1973-2024 |
| `data/raw/imf_fsi/npl_ratio_KZ.csv` | KZ system NPL (IMF FSI) | 2008-2023 |

---

## 7. Processed Data

| File | Description | Used For |
|------|-------------|----------|
| `data/processed/panel.parquet` | Full estimation panel | All LP regressions |
| `data/processed/credit_panel.parquet` | Credit panel by bank | Sector panel LP |
| `data/processed/credit_panel_annual.parquet` | Annual credit panel | Annual LP |
| `data/processed/fx_passthrough/*.parquet` | FX passthrough data | Block A/F estimation |

---

## 8. Data Gaps to Fill

### 8.1 Critical (Blocks Panel LP)

| Bank | Gap | Action Required | Priority |
|------|-----|-----------------|----------|
| Halyk | Missing 2024 | Extract from 2024 annual report or Q3 interim | High |
| ForteBank | Missing 2024 | Extract from 2024 annual report | High |
| BCC | Missing 2024 | Extract from 2024 annual report | High |
| All 3 banks | Quarterly granularity | Extract Q1-Q3 interim statements if available | Medium |

### 8.2 Nice to Have

| Bank | Gap | Action Required | Priority |
|------|-----|-----------------|----------|
| BCC | Pre-2015 data | Extract from 2011-2014 annual reports | Low |
| Halyk | Pre-2011 data | Extract from 2008-2010 annual reports | Low |

### 8.3 Macro Data

| Series | Gap | Action | Priority |
|--------|-----|--------|----------|
| NBK base rate | Full history | Verify `nbk_policy_rate.py` coverage | Medium |
| CPI tradable index | Constructed | Verify category classification | High |

---

## 9. Data Quality Matrix

| Bank | NPL | CoR | Capital | RWA | Deposit Cost | Overall |
|------|-----|-----|---------|-----|--------------|---------|
| Kaspi | A | A | A | A | A | **A** |
| Halyk | B | B | B | B | B | **B** |
| Forte | B | B | B | B | B | **B** |
| BCC | B | B | B | B | B | **B** |

**Legend:**
- A: Direct quarterly reporting with audit trail
- B: Annual reporting, consistent definitions
- C: Annual reporting, definition changes or gaps

---

## 10. Verification Checklist

- [x] Kaspi Bank extended data loaded (27 periods)
- [x] Halyk Bank data loaded (13 periods)
- [x] ForteBank data loaded (7 periods)
- [x] BCC data loaded (9 periods)
- [x] Exposure variables defined for all 4 banks
- [x] CPI categories data available (monthly, 2008-2024)
- [x] USD/KZT exchange rate available (daily, 2000-2024)
- [x] Oil shocks data available (Baumeister-Hamilton)
- [ ] 2024 data for Halyk, Forte, BCC (PENDING)
- [ ] Quarterly data for non-Kaspi banks (NOT AVAILABLE)

---

*Inventory compiled by data_assembler.py pipeline*
