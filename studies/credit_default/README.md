# Kazakhstan Credit Quality Study

## Overview

This study analyzes the relationship between **external shocks and aggregate credit quality** in Kazakhstan.

**DESIGN REVISION (v2.0):** The original micro design (loan-level diff-in-discontinuities) required internal fintech data that is unavailable. The study was revised to use publicly available aggregate data.

**Current Design:** Reduced-form local projections: External Shocks → NPL Ratio

**Research Question:** How do exogenous external shocks (oil, global risk) affect aggregate credit quality in Kazakhstan?

**NOT estimated:** Causal income → default elasticity (requires micro loan-level data)

## Identification Strategies

### Design 1: Minimum Wage Difference-in-Discontinuities (PRIMARY)

Uses the January 2024 minimum wage increase (70,000 → 85,000 tenge) as a source of exogenous income variation for formal workers.

**Running Variable:** Pre-policy payroll wage (observed from cashflows)
**Cutoff:** 70,000 tenge (old minimum wage)

**Specification:**
```
Y_it = α + β₁(Post_t) + β₂(Below_c_i) + β₃(Post_t × Below_c_i)
     + f(wage_i - c) + g(wage_i - c) × Post_t + X_it'δ + ε_it
```

Key coefficient: β₃ = effect of MW-induced income shock on default

### Design 2: Pension Eligibility Fuzzy RDD (SECONDARY)

Uses pension eligibility age thresholds (63 for men, 61 for women) as an instrument for pension income receipt.

**Running Variable:** Age relative to pension threshold
**Instrument:** 1(Age ≥ Cutoff)
**Endogenous:** Pension inflow (observed from cashflows)

**Specification:**
```
First stage:  PensionInflow_i = γ₀ + γ₁(Age_i ≥ c) + f(Age_i - c) + X_i'δ + ν_i
Second stage: Default_i = α + β(PensionInflow_i_hat) + f(Age_i - c) + X_i'δ + ε_i
```

## Critical Requirements

### Income from Cashflows (NOT Stated Income)

Stated income is too noisy and will attenuate estimates toward zero. Must use actual observed cashflows:
- Payroll-identified wage inflows (for MW design)
- Pension inflows (for pension RDD)

### Policy Confound Avoidance

| Date | Policy | Action |
|------|--------|--------|
| Dec 2023 | DSTI tightening | Restrict originations to before Dec 2023 |
| Jun 2024 | DTI introduction | Truncate outcomes at May 2024 |
| Mar 2023 | Personal bankruptcy law | Include as control if needed |

## Sample Restrictions

| Criterion | Restriction |
|-----------|-------------|
| Origination | Before December 2023 |
| Outcome window | January - May 2024 |
| Income source | Verified payroll (MW) or pension (RDD) |
| Risk set | Loans "alive" at January 2024 |

## Primary Outcome

**DPD30** (30+ days past due) at 3-month horizon

Secondary outcomes: DPD15, DPD60, DPD90

## Module Structure

```
studies/credit_default/
├── config/
│   └── model_spec.yaml          # Pre-specified parameters
├── src/
│   ├── internal_loans.py        # Loan + cashflow data loader
│   ├── credit_bureau.py         # Credit bureau (partnership required)
│   ├── panel_data.py            # Loan-month panel builder
│   ├── sample_construction.py   # Eligibility, treatment assignment
│   ├── confound_checks.py       # Policy confound validation
│   ├── diff_in_discs.py         # MW difference-in-discontinuities
│   ├── fuzzy_rdd.py             # Pension fuzzy RDD
│   ├── elasticity_store.py      # Store estimated elasticities
│   ├── scenario_simulator.py    # Segment-specific simulation
│   └── portfolio_stress.py      # Portfolio stress testing
├── outputs/
└── tests/
```

## Usage

### Run Policy Confound Check

```python
from studies.credit_default.src.confound_checks import check_confounds

results = check_confounds(
    origination_cutoff=date(2023, 12, 1),
    outcome_start=date(2024, 1, 1),
    outcome_end=date(2024, 5, 31),
    treatment_date=date(2024, 1, 1),
)
# All checks must pass before estimation
```

### Estimate MW Effect

```python
from studies.credit_default.src.diff_in_discs import DiffInDiscsEstimator

estimator = DiffInDiscsEstimator(
    old_minimum_wage=70000,
    new_minimum_wage=85000,
)
result = estimator.estimate(data, outcome="dpd30")
print(estimator.summary(result))
```

### Estimate Pension Effect

```python
from studies.credit_default.src.fuzzy_rdd import FuzzyRDDEstimator

estimator = FuzzyRDDEstimator(cutoff_men=63, cutoff_women=61)
result = estimator.estimate(data, outcome="dpd30")
print(estimator.summary(result))
```

### Run Scenario Simulation

```python
from studies.credit_default.src.scenario_simulator import CreditScenarioSimulator

simulator = CreditScenarioSimulator(baseline_default_rate=0.05)
result = simulator.simulate_income_shock(income_change_pct=-0.10)
print(result.summary())
```

### Run Portfolio Stress Test

```python
from studies.credit_default.src.portfolio_stress import (
    PortfolioStressTester,
    create_sample_portfolio,
)

tester = PortfolioStressTester()
portfolio = create_sample_portfolio()
result = tester.stress_test(portfolio, "moderate_recession")
print(result.summary())
```

## Validation Checklist

| Test | Method | Purpose |
|------|--------|---------|
| Pre-trends | Event study + joint F | Parallel trends |
| First-stage F | Report F-stat | Instrument strength (F > 10) |
| Balance test | Covariate RDD at cutoff | Randomization |
| Placebo cutoff | Test at false thresholds | Design validity |
| McCrary density | Histogram + formal test | No manipulation |
| Policy confound | Check against DSTI/DTI dates | Exclusion restriction |

## External Validity Caveats

Estimated elasticities are **local average treatment effects (LATEs)**:

1. **MW design:** Applies to formal workers whose income was raised by MW policy
2. **Pension RDD:** Applies to near-retirees receiving first pension

For portfolio-wide application:
- Reweight/stratify by income source
- Apply primarily to borrowers similar to identification sample
- Never extrapolate to dissimilar segments without explicit caveats

## Dependencies

- Uses shared infrastructure from `shared/data/`, `shared/model/`, `shared/engine/`
- Requires internal loan/cashflow data (fintech/lender)
- Credit bureau data: Partnership required (mock available for development)

## CLI Commands

```bash
# Check for policy confounds
kzresearch credit check-confounds

# Build loan panel
kzresearch credit build-panel --origination-before 2023-12-01 --outcome-window 2024-01-01:2024-05-31

# Estimate effects
kzresearch credit estimate mw-diff-discs --outcome dpd30 --horizon 3
kzresearch credit estimate pension-rdd --outcome dpd30 --bandwidth optimal

# Run diagnostics
kzresearch credit diagnostics

# Simulate scenarios
kzresearch credit simulate --income-change -0.10 --segment formal-payroll

# Run full MVP
kzresearch credit run-mvp
```

## References

- Minimum wage data: stat.gov.kz
- Pension eligibility: gov.kz
- Policy confounds: IMF Article IV Consultation (2024)
- McCrary test: McCrary (2008)
- rdrobust: Calonico, Cattaneo, Titiunik (2014)
