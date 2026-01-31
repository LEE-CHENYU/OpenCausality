"""
Kazakhstan Household Welfare Study.

Research Question: How do global oil shocks affect household welfare in Kazakhstan?

This study uses a shift-share design:
- Exposure: Regional oil sector share (E_oil_r)
- Shock: Global oil supply/demand shocks (Baumeister decomposition)
- Outcome: Per-capita household income

Key files:
- src/panel_data.py: Panel construction with region harmonization
- src/shift_share.py: Shift-share regression model
- src/local_projections.py: Dynamic IRFs via local projections
- src/simulator.py: Scenario simulation engine
"""
