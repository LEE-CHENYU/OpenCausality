# Research Pipeline Skill

A skill for managing the Kazakhstan econometric research data pipeline.

## Commands

### `/research-pipeline status`
Show the current download status of all data sources.

**Implementation:**
```bash
PYTHONPATH=. python scripts/download_all_data.py --status
```

Or read the status file directly:
```bash
cat data/metadata/download_status.json | python -m json.tool
```

### `/research-pipeline fetch-all`
Download all data sources with status tracking.

**Implementation:**
```bash
PYTHONPATH=. python scripts/download_all_data.py
```

To force re-download of all sources:
```bash
PYTHONPATH=. python scripts/download_all_data.py --force
```

### `/research-pipeline verify`
Verify no synthetic data is present and all sources are valid.

**Implementation:**
```bash
PYTHONPATH=. python scripts/download_all_data.py --verify
```

Then check that:
1. No sources have status "synthetic"
2. All required sources have status "downloaded"
3. CPI data shows heterogeneous volatility across categories

### `/research-pipeline run <study>`
Run the full analysis pipeline for a specific study.

**Studies available:**
- `fx_passthrough` - FX pass-through analysis (Block A)
- `credit_default` - Credit default study

**Implementation for fx_passthrough:**
```bash
kzresearch passthrough run-full-chain
```

**Implementation for credit_default:**
```bash
PYTHONPATH=. python -c "from studies.credit_default.src import run_pipeline; run_pipeline()"
```

## Data Sources

| Source | Description | Required | Path |
|--------|-------------|----------|------|
| fred_igrea | Kilian Global Real Economic Activity Index | Yes | data/raw/fred/IGREA.parquet |
| fred_vix | CBOE Volatility Index | Yes | data/raw/fred/VIXCLS.parquet |
| fred_brent | Brent Crude Oil Price | Yes | data/raw/fred/DCOILBRENTEU.parquet |
| baumeister_shocks | Baumeister-Hamilton Oil Supply Shocks | Yes | data/raw/baumeister_shocks/shocks.parquet |
| nbk_usd_kzt | NBK USD/KZT Exchange Rate | Yes | data/raw/nbk/usd_kzt.parquet |
| bns_cpi_categories | BNS CPI by COICOP Category | Yes | data/raw/kazakhstan_bns/cpi_categories.parquet |
| bns_national_income | BNS National Income by Region | Yes | data/raw/kazakhstan_bns/national_income.parquet |
| worldbank_reer | World Bank REER (annual) | No | data/raw/worldbank/reer_kz.parquet |

## Verification Checklist

After downloading data, verify:

1. **Download status is clean:**
   ```bash
   cat data/metadata/download_status.json | grep -c '"status": "downloaded"'
   # Should be >= 7 (all required sources)
   ```

2. **No synthetic data:**
   ```bash
   cat data/metadata/download_status.json | grep synthetic
   # Should return nothing
   ```

3. **CPI data shows heterogeneous patterns:**
   ```python
   import pandas as pd
   df = pd.read_parquet("data/raw/kazakhstan_bns/cpi_categories.parquet")
   print(df.groupby("category")["inflation_mom"].std())
   # Should show different volatilities for different categories
   # Food should be more volatile than utilities/education
   ```

4. **Admin price test passes:**
   ```bash
   kzresearch passthrough falsification
   # Should show: Admin prices: PASS
   ```

## Troubleshooting

### API Timeouts
If NBK or World Bank APIs timeout:
1. Check network connectivity
2. The fetchers now have 60-90 second timeouts
3. Try running at a different time (API rate limits)

### Synthetic Data Warning
If synthetic data is detected:
1. Delete files in `data/processed/fx_passthrough/`
2. Run `PYTHONPATH=. python scripts/download_all_data.py --force`
3. Re-run the analysis pipeline

### Admin Price Test Fails
If admin prices show significant pass-through:
1. Verify you're using real BNS data, not synthetic
2. Check `data/metadata/download_status.json` for `bns_cpi_categories`
3. Re-download with `--force` if needed
