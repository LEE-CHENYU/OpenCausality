# Kazakhstan Household Welfare Study

## Research Question

**How do global oil shocks affect household welfare in Kazakhstan?**

## Identification Strategy

This study uses a **shift-share design** to identify causal effects:

- **Exposure (E_oil_r)**: Regional oil sector share (mining/extraction GRP)
- **Shock (S_t)**: Global oil supply/demand shocks (Baumeister decomposition)
- **Outcome (Y_{r,t})**: Per-capita household income by region

### Specification

```
log_income_pc_{r,t} = alpha_r + delta_t + beta(E_oil_r x S_t) + u_{r,t}
```

Where:
- `alpha_r`: Region fixed effects
- `delta_t`: Time fixed effects (absorbs national cycle)
- `beta`: Effect of oil exposure x shock interaction

## Data Sources

| Source | Description | Coverage |
|--------|-------------|----------|
| Kazakhstan BNS | Per-capita income, GRP, employment | 2010-2024 |
| USGS/EITI | Mining sector shares | 2010-2022 |
| Baumeister | Oil supply/demand shocks | 1975-2024 |
| FRED | Global activity (IGREA), VIX | 2000-2024 |

## Key Files

```
studies/household_welfare/
├── config/
│   └── model_spec.yaml         # Pre-specified parameters
├── src/
│   ├── panel_data.py           # Panel construction
│   ├── shift_share.py          # Shift-share regression
│   ├── local_projections.py    # Dynamic IRFs
│   ├── simulator.py            # Scenario engine
│   └── data_pipeline.py        # Data orchestration
├── outputs/
│   └── estimation_results.md   # Latest results
└── tests/
```

## Running the Study

```bash
# From econometric-research/

# Fetch data
kzresearch welfare fetch-data all

# Build panel
kzresearch welfare build-panel

# Estimate models
kzresearch welfare estimate all

# Run simulations
kzresearch welfare simulate oil_supply_disruption
```

## Results Summary

See `outputs/estimation_results.md` for current results.

**Key Finding**: No statistically significant evidence that global oil shocks differentially affect high-oil-exposure regions. All oil coefficients p > 0.20.

## Caveats

1. Missing income data for 2 oil regions (West Kazakhstan, North Kazakhstan)
2. Oil exposure from alternative sources (USGS/EITI), not direct BNS measurement
3. Cyclical proxy is GRP-based, not employment

## Version

Study Version: 4.0 (Oil exposure only in main spec)
