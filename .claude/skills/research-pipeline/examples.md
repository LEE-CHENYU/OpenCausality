# Research Pipeline Examples

## Example 1: Fresh Setup

Starting from a clean state, download all data and run the FX pass-through analysis.

```bash
# 1. Download all data sources
PYTHONPATH=. python scripts/download_all_data.py

# 2. Verify download status
cat data/metadata/download_status.json | python -m json.tool

# 3. Run the full analysis chain
kzresearch passthrough run-full-chain

# 4. Check results
cat studies/fx_passthrough/outputs/results_summary.json | python -m json.tool
```

Expected output in `results_summary.json`:
```json
{
  "estimation_results": {
    "overall_passthrough": 0.35,
    "ci_lower": 0.28,
    "ci_upper": 0.42
  },
  "diagnostics": {
    "admin_prices_pass": true,
    "pre_trend_pass": true
  }
}
```

## Example 2: Recovering from Synthetic Data

If you discover the data is synthetic:

```bash
# 1. Delete processed files
rm -rf data/processed/fx_passthrough/*.parquet

# 2. Force re-download all raw data
PYTHONPATH=. python scripts/download_all_data.py --force

# 3. Verify no synthetic data flag
grep -c "synthetic" data/metadata/download_status.json
# Should output: 0

# 4. Re-run analysis
kzresearch passthrough run-full-chain

# 5. Verify admin price test passes
cat studies/fx_passthrough/outputs/results_summary.json | grep admin_prices_pass
# Should show: "admin_prices_pass": true
```

## Example 3: Checking Data Quality

Verify CPI data shows heterogeneous volatility (sign of real data):

```python
import pandas as pd

# Load CPI data
df = pd.read_parquet("data/raw/kazakhstan_bns/cpi_categories.parquet")

# Check volatility by category
volatility = df.groupby("category")["inflation_mom"].agg(["mean", "std"])
print("Inflation volatility by category:")
print(volatility.sort_values("std", ascending=False))

# Real data should show:
# - Food (01): Higher volatility (std > 0.01)
# - Admin prices (04, 06, 08, 10): Lower volatility
# - Different means across categories

# Check for uniform patterns (synthetic data warning sign)
std_values = volatility["std"].values
if std_values.std() < 0.001:
    print("WARNING: Suspiciously uniform volatility - may be synthetic data!")
```

## Example 4: Running Falsification Tests

```bash
# Run just the falsification tests
kzresearch passthrough falsification

# Expected output:
# Falsification Tests
# ==================
# Admin prices: PASS (p-value > 0.10 for admin category pass-through)
# Pre-trend: PASS (no significant pre-2014 effect)
```

If admin prices fail:
```
# Admin prices: FAIL (significant pass-through detected)
# This indicates synthetic data - admin prices should NOT respond to FX shocks
```

## Example 5: Diagnosing Download Failures

```bash
# Check status of all sources
PYTHONPATH=. python scripts/download_all_data.py --status

# If NBK fails, check network:
curl -I https://nationalbank.kz/rss/rates_all.xml

# If BNS fails, check API:
curl -I "https://stat.gov.kz/api/iblock/element/49140/csv/file/en/"

# If World Bank times out (optional source):
# This is acceptable - World Bank REER is annual and optional
```

## Example 6: Running Specific Studies

### FX Pass-through (Block A)
```bash
# Full chain
kzresearch passthrough run-full-chain

# Or step by step:
kzresearch passthrough fetch-data    # Download/verify data
kzresearch passthrough build-panel   # Build estimation panel
kzresearch passthrough estimate      # Run local projections
kzresearch passthrough falsification # Run tests
```

### Credit Default Study
```bash
# Run credit default pipeline
PYTHONPATH=. python -c "
from studies.credit_default.src import run_pipeline
run_pipeline()
"

# Or use the CLI if available
kzresearch credit run-analysis
```

## Example 7: Data Inspection

```python
# Quick data inspection script
import pandas as pd
from pathlib import Path

data_dir = Path("data/raw")

# List all downloaded files
for parquet in data_dir.rglob("*.parquet"):
    df = pd.read_parquet(parquet)
    print(f"\n{parquet.relative_to(data_dir)}:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
```

## Common Issues and Solutions

### Issue: "All exchange rate fetch tiers failed"
**Cause:** NBK and World Bank APIs both unreachable
**Solution:**
1. Check network connectivity
2. Wait and retry (APIs may have rate limits)
3. Try downloading during different hours

### Issue: "admin_prices_pass: false"
**Cause:** Using synthetic data with artificial pass-through
**Solution:**
1. Delete `data/processed/fx_passthrough/`
2. Run `scripts/download_all_data.py --force`
3. Re-run analysis

### Issue: "No FRED API key configured"
**Cause:** Missing API key in environment
**Solution:**
1. Get API key from https://fred.stlouisfed.org/docs/api/api_key.html
2. Add to `.env`: `FRED_API_KEY=your_key_here`

### Issue: CPI data shows only "00" category
**Cause:** BNS API returned headline CPI only
**Solution:**
1. Check if BNS API structure changed
2. May need to find new iblock ID for category-level data
3. Consult https://stat.gov.kz/en/industries/prices/
