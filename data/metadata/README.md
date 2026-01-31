# Data Dictionary

This directory contains metadata and documentation for all datasets used in the Kazakhstan Household Welfare study.

## Files

- `datasets.yaml` - Machine-readable metadata for all datasets

---

## Variable Definitions

### Outcome Variable

| Variable | Description | Source | Units |
|----------|-------------|--------|-------|
| `income_pc` | Per-capita quarterly monetary income | BNS | Tenge |
| `log_income_pc` | Natural log of income_pc | Derived | Log tenge |

### Exposure Variables

| Variable | Description | Source | Range |
|----------|-------------|--------|-------|
| `E_oil_r` | Oil/mining sector exposure | **HARDCODED** | [0, 1] |
| `E_cyc_r` | Cyclical employment exposure | **HARDCODED** | [0, 1] |
| `E_debt_r` | Debt repayment exposure | Constant 0.1 | [0, 1] |

**WARNING**: Exposure variables are not computed from real data. See `datasets.yaml` for details.

### Shock Variables

| Variable | Description | Source | Units |
|----------|-------------|--------|-------|
| `oil_supply_shock` | Structural oil supply shock | Baumeister | SD |
| `aggregate_demand_shock` | Global aggregate demand shock | Baumeister | SD |
| `global_activity_shock` | Kilian real economic activity innovation | FRED IGREA | SD |
| `vix_shock` | VIX innovation | FRED VIXCLS | SD |
| `brent_shock` | Brent log return | FRED DCOILBRENTEU | Log return |

### Interaction Terms

| Variable | Components | Description |
|----------|------------|-------------|
| `E_oil_x_supply` | E_oil_r x oil_supply_shock | Oil exposure times supply shock |
| `E_oil_x_demand` | E_oil_r x aggregate_demand_shock | Oil exposure times demand shock |
| `E_cyc_x_activity` | E_cyc_r x global_activity_shock | Cyclical exposure times activity |
| `E_oil_x_vix` | E_oil_r x vix_shock | Oil exposure times VIX shock |

### Identifiers

| Variable | Description | Example |
|----------|-------------|---------|
| `region` | Canonical region name | "Atyrau" |
| `quarter` | Quarter identifier | "2020Q1" |
| `year` | Calendar year | 2020 |
| `q` | Quarter of year (1-4) | 1 |
| `region_id` | Numeric region identifier | 5 |
| `quarter_id` | Numeric quarter identifier | 40 |

---

## Canonical Regions (16)

| Region | Oil Exposure* | Type |
|--------|---------------|------|
| Akmola | 0.05 | Agricultural |
| Aktobe | 0.25 | Mixed/Oil |
| Almaty City | 0.05 | Urban/Service |
| Almaty Region | 0.05 | Mixed |
| Astana | 0.05 | Urban/Capital |
| Atyrau | 0.80 | Oil (Primary) |
| East Kazakhstan | 0.05 | Industrial |
| Jambyl | 0.05 | Agricultural |
| Karaganda | 0.05 | Mining/Industrial |
| Kostanay | 0.05 | Agricultural |
| Kyzylorda | 0.30 | Oil (Secondary) |
| Mangystau | 0.70 | Oil (Primary) |
| North Kazakhstan | 0.05 | Agricultural |
| Pavlodar | 0.05 | Industrial |
| South Kazakhstan | 0.05 | Agricultural |
| West Kazakhstan | 0.50 | Oil (Secondary) |

*Note: Oil exposure values are hardcoded estimates, not measured from data.

---

## Data Quality Flags

When working with the data, check these quality indicators:

### High Quality
- FRED series (IGREA, VIX, Brent)
- Baumeister shocks (verify download succeeded)

### Medium Quality
- BNS income (14/16 regions present)

### Low Quality / Critical Issues
- E_oil_r (hardcoded, undocumented source)
- E_cyc_r (hardcoded, undocumented source)
- E_debt_r (constant placeholder)

---

## File Formats

| Location | Format | Notes |
|----------|--------|-------|
| `data/raw/` | Parquet | Original fetched data |
| `data/processed/` | Parquet | Analysis-ready panel |
| `data/backup/` | JSON/Excel/Parquet | Archived API responses |
| `data/crosswalks/` | CSV | Reference mappings |

---

## Usage

### Loading Panel Data

```python
import pandas as pd

panel = pd.read_parquet("data/processed/panel.parquet")
print(f"Shape: {panel.shape}")
print(f"Columns: {list(panel.columns)}")
```

### Checking Data Lineage

```python
from src.data.data_lineage import print_lineage_report
print_lineage_report()
```

### Validating Quality

```python
from src.data.data_lineage import check_data_quality
if not check_data_quality():
    print("WARNING: Critical data quality issues detected")
```

---

## Known Issues

1. **Missing Regions**: West Kazakhstan and North Kazakhstan are missing from BNS income data
2. **Hardcoded Exposures**: Oil and cyclical exposures use hardcoded values without documented sources
3. **Silent Fallbacks**: Some data loaders silently substitute synthetic data if downloads fail
4. **API Failures**: BNS mining and employment endpoints return HTTP 500 errors

---

*Last updated: January 2026*
