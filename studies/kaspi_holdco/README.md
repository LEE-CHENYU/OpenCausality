# Kaspi.kz Holding Company Capital Adequacy Study

## Overview

This study performs holding company capital adequacy sensitivity analysis for Kaspi.kz. Unlike bank-level Basel ratios, it focuses on:

1. **Stand-alone solvency** - Parent liabilities vs assets
2. **Liquidity runway** - Survival under dividend stop
3. **Capital call capacity** - Can parent inject capital if ARDFM requires?
4. **Bank stress passthrough** - K2 shortfall → capital call translation

## Key Findings (FY2024 Baseline)

| Metric | Value | Threshold |
|--------|-------|-----------|
| Bank K2 Ratio | 12.7% | min 12.0% |
| K2 Headroom | ~60 KZT bn | - |
| HoldCo Cash | 325 KZT bn | - |
| HoldCo Standalone Runway | ~187 months | - |
| Max Capital Call (no sh. div) | ~770 bn | - |

## Quick Start

```bash
# Show bank and holdco state
kzkaspi state

# Run bank stress scenarios
kzkaspi bank-stress --scenario moderate
kzkaspi bank-stress-grid

# Run holdco simulations
kzkaspi holdco-simulate --scenario baseline
kzkaspi holdco-sensitivity

# Full bank → holdco passthrough
kzkaspi full-stress
kzkaspi full-stress-grid

# Executive summary
kzkaspi summary
```

## Data Sources

- **FY2024 20-F Filing**: Parent-only and consolidated financials
- **NBK Framework**: K1-2 (10.5% min), K2 (12.0% min) ratios

### Bank-Level (NBK Framework)

| Metric | Value (KZT bn) |
|--------|----------------|
| RWA | 8,059 |
| Tier 1 Capital | 1,016 |
| Total Capital | 1,027 |
| K1-2 Ratio | 12.6% |
| K2 Ratio | 12.7% |

### Holding Company (Parent-Only)

| Metric | Value (KZT bn) |
|--------|----------------|
| Parent Cash | 324.993 |
| Parent G&A | 20.810 |
| Parent Liabilities | 0.155 |
| Dividend from Bank | 285.206 |
| Dividend from Other Subs | 467.500 |
| Dividends to Shareholders | 646.056 |

## Model Architecture

### Bank Stress Engine (`bank_stress.py`)

Joint liquidity + solvency stress simulation:

1. **Liquidity waterfall**: outflows → cash → repo → security sales
2. **Solvency**: retained earnings - credit losses - fire-sale losses - MTM

Key outputs:
- `k2_after`: Stressed K2 ratio (determines dividend restriction)
- `capital_shortfall`: Required capital injection from parent
- `liquidity_gap`: Emergency liquidity support needed

### HoldCo Simulator (`holdco_simulator.py`)

12-month cash flow projection:

```
Ending Cash = Starting Cash
            + Dividends from bank
            + Dividends from other subs
            + New equity/debt raised
            - Annual fixed costs (G&A + interest)
            - Dividends to shareholders
            - Buybacks
            - Capital injection to bank
            - Acquisitions
```

### Bank → HoldCo Integration (`bank_integration.py`)

Parameterized passthrough from bank stress to holdco impact:

```python
# Dividend payout formula (buffer-based)
payout = clip((k2 - k2_min) / (k2_baseline - k2_min), 0, 1) ^ gamma

# Capital call formula
capital_call = shortfall * multiplier
```

## Stress Scenarios

### Bank Scenarios

| Scenario | Profit | Credit Loss | RWA | Retail Run |
|----------|--------|-------------|-----|------------|
| Baseline | 100% | 1% | +0% | 5% |
| Mild | 80% | 2% | +5% | 10% |
| Moderate | 50% | 3% | +10% | 15% |
| Severe | 0% | 5% | +15% | 25% |
| Oil Crisis | 0% | 7% | +25% | 30% |

### HoldCo Scenarios

- `baseline` - FY2024 actuals
- `no_bank_dividend` - Bank retains all earnings
- `no_dividends` - All subs retain earnings
- `dividend_stop_no_payout` - Full freeze
- `moderate_capital_call` - 50% haircut + 100bn injection
- `severe_capital_call` - No dividend + 200bn injection

## Key Constraints

1. **Ring-fencing**: Bank can only lend 10% of capital to parent
2. **Dividend restriction**: K2 must stay above 12.0% for dividends
3. **NBK RWA**: ~45% larger than Basel III (consumer risk weights 150%+)

## Directory Structure

```
studies/kaspi_holdco/
├── __init__.py
├── README.md
├── config/
│   └── model_spec.yaml
├── src/
│   ├── __init__.py
│   ├── cli.py                    # CLI commands (kzkaspi)
│   ├── bank_state.py             # BankState, BankScenario dataclasses
│   ├── bank_stress.py            # Bank K2/K1-2 stress simulation
│   ├── holdco_state.py           # HoldCoState, HoldCoScenario dataclasses
│   ├── holdco_simulator.py       # 12-month cash flow simulation
│   ├── bank_integration.py       # K2 shortfall → capital call translation
│   └── stress_scenarios.py       # Pre-defined scenarios and FY2024 baseline
├── tests/
│   ├── __init__.py
│   ├── test_bank_stress.py
│   └── test_holdco_simulator.py
└── outputs/
    └── .gitkeep
```

## Verification

```bash
# 1. Verify baseline state
kzkaspi state
# Expected: K2=12.7%, Cash=325, Runway ~187 months

# 2. Run bank stress grid
kzkaspi bank-stress-grid
# Identify scenarios where K2 < 12%

# 3. Run holdco sensitivity
kzkaspi holdco-sensitivity
# Find (dividend, capital_call) combos that make cash negative

# 4. Full passthrough
kzkaspi full-stress
# See bank stress → K2 → capital call → holdco ending cash
```
