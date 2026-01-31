# Data Dictionary (v4)

This directory contains metadata and documentation for all datasets used in the Kazakhstan Household Welfare study.

**v4 Update**: Study design revised. Main spec uses oil exposure only. Cyclical exposure dropped from core model; GRP-based proxy available for robustness.

## Files

- `datasets.yaml` - Machine-readable metadata for all datasets

---

## Pipeline Status

**Current Status: READY**

Main specification (oil-only) is fully operational:
```bash
PYTHONPATH=. kzwelfare estimate all
```

---

## Variable Definitions

### Outcome Variable

| Variable | Description | Source | Units |
|----------|-------------|--------|-------|
| `income_pc` | Per-capita quarterly monetary income | BNS | Tenge |
| `log_income_pc` | Natural log of income_pc | Derived | Log tenge |

### Exposure Variables

| Variable | Description | Source | Status | Used In |
|----------|-------------|--------|--------|---------|
| `E_oil_r` | Oil/mining sector exposure | USGS/EITI | **AVAILABLE** | Main + Robustness |
| `E_cyc_proxy_r` | Cyclical exposure (GRP-based) | stat.gov.kz | **AVAILABLE** | Robustness only |
| `E_cyc_r` | True cyclical employment | BNS | NOT AVAILABLE | Dropped from model |
| `E_debt_r` | Debt repayment exposure | BNS Expenditure | Optional | Auxiliary only |

**v4 Design**: Main spec uses E_oil_r only. E_cyc_proxy_r is for robustness checks (NOT employment data).

### Shock Variables

| Variable | Description | Source | Status |
|----------|-------------|--------|--------|
| `oil_supply_shock` | Structural oil supply shock | Baumeister | Available |
| `aggregate_demand_shock` | Global aggregate demand shock | Baumeister | Available |
| `global_activity_shock` | Kilian real economic activity innovation | FRED IGREA | Available |
| `vix_shock` | VIX innovation | FRED VIXCLS | Available |
| `brent_shock` | Brent log return | FRED DCOILBRENTEU | Available |

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

| Region | Type | Income Data |
|--------|------|-------------|
| Akmola | Agricultural | Available |
| Aktobe | Mixed/Oil | Available |
| Almaty City | Urban/Service | Available |
| Almaty Region | Mixed | Available |
| Astana | Urban/Capital | Available |
| Atyrau | Oil (Primary) | Available |
| East Kazakhstan | Industrial | Available |
| Jambyl | Agricultural | Available |
| Karaganda | Mining/Industrial | Available |
| Kostanay | Agricultural | Available |
| Kyzylorda | Oil (Secondary) | Available |
| Mangystau | Oil (Primary) | Available |
| North Kazakhstan | Agricultural | **MISSING** |
| Pavlodar | Industrial | Available |
| South Kazakhstan | Agricultural | Available |
| West Kazakhstan | Oil (Secondary) | **MISSING** |

---

## Data Quality Status (v3)

### Available and Ready

| Source | Status | Notes |
|--------|--------|-------|
| FRED IGREA | Ready | High quality |
| FRED VIX | Ready | High quality |
| FRED Brent | Ready | High quality |
| Baumeister shocks | Ready | Verified real (not synthetic) |
| BNS income | Partial | 14/16 regions available |
| Mining shares (alt) | Ready | USGS/EITI/stat.gov.kz data |

### Resolved Issues

| Source | Variable | Status |
|--------|----------|--------|
| Mining shares | E_oil_r | **RESOLVED** - Using alternative sources |

### Still Blocking

| Source | Variable | Status |
|--------|----------|--------|
| BNS Employment | E_cyc_r | **UNAVAILABLE** - Need ILO/OECD data |

### Needs Verification

| Source | Variable | Check |
|--------|----------|-------|
| BNS Expenditure | E_debt_r | Verify debt_share column exists |

---

## Code Behavior (v3)

### Previous Behavior (v1 - REMOVED)
```python
# OLD - Silent fallback to hardcoded values
if mining_df.empty:
    oil_regions = {"Atyrau": 0.8, "Mangystau": 0.7, ...}  # UNDOCUMENTED
    panel["E_oil_r"] = panel["region"].map(oil_regions)
```

### Current Behavior (v3)
```python
# NEW - Fall back to documented alternative sources
mining_df = bns_data.get(BNSDataType.MINING_SHARES, pd.DataFrame())
if mining_df.empty:
    mining_df, source = self._load_alternative_mining_shares()
    # Uses data/raw/alternative_sources/mining_shares.csv
    # Records data lineage showing USGS/EITI as source
```

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

### Checking Pipeline Status

```python
import yaml

with open("data/metadata/datasets.yaml") as f:
    meta = yaml.safe_load(f)

print(f"Pipeline status: {meta['pipeline_status']['overall']}")
print(f"Blocking issues: {meta['pipeline_status']['blocking_issues']}")
```

### Loading Panel Data (Once Available)

```python
import pandas as pd

# This will fail until blocking issues resolved
panel = pd.read_parquet("data/processed/panel.parquet")
```

### Checking Data Lineage

```python
from src.data.data_lineage import print_lineage_report
print_lineage_report()
```

---

## Resolution Steps

Current status and next steps:

1. ~~**Obtain mining sector shares**~~: **RESOLVED**
   - Alternative sources in `data/raw/alternative_sources/mining_shares.csv`
   - Sources: USGS, EITI, stat.gov.kz GRP

2. **Obtain employment data** (still needed):
   - Try ILO labor statistics
   - Try OECD employment data
   - Add to `data/raw/alternative_sources/employment_shares.csv`

3. **Verify expenditure data**:
   - Check that `debt_share` column exists in expenditure file

4. **Run pipeline**:
   ```bash
   PYTHONPATH=. kzwelfare build-panel
   PYTHONPATH=. kzwelfare estimate all
   ```

5. **Check data lineage**:
   ```python
   from src.data.data_lineage import print_lineage_report
   print_lineage_report()
   ```

---

*Document version: 4.0*
*Last updated: January 2026*
*Update: Revised study design - oil exposure only in main spec*
