"""
Kazakhstan Consumer Credit Default Sensitivity Study.

Research Question: How sensitive is consumer credit default to changes in disposable income?

Key Parameter: Income-Default Elasticity (beta) - the percentage point change in
default probability for a 1% change in disposable income.

Identification Strategies:
1. Minimum Wage Diff-in-Discontinuities (PRIMARY)
2. Pension Eligibility Fuzzy RDD (SECONDARY)
3. TSA RDD (requires partnership)

Key files:
- src/panel_data.py: Loan-month panel construction
- src/sample_construction.py: Treatment assignment, eligibility
- src/confound_checks.py: Policy confound validation
- src/diff_in_discs.py: Minimum wage design
- src/fuzzy_rdd.py: Pension eligibility RDD
- src/portfolio_stress.py: Stress testing with caveats
"""
