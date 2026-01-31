# %% [markdown]
# # Kazakhstan Household Welfare Model - Data Exploration
#
# This notebook explores the data sources and validates the pipeline.

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path.cwd().parent))

from config.settings import get_settings
from src.data.fred_client import FREDClient, FREDSeries
from src.data.baumeister_loader import BaumeisterLoader
from src.model.panel_data import PanelBuilder, CANONICAL_REGIONS, REGION_CROSSWALK

# %% [markdown]
# ## 1. FRED Data

# %%
# Initialize FRED client
fred = FREDClient()

# Fetch all series
print("Fetching FRED data...")
brent = fred.fetch_brent(start_date="2010-01-01")
vix = fred.fetch_vix(start_date="2010-01-01")
igrea = fred.fetch_global_activity(start_date="2010-01-01")

print(f"Brent: {len(brent)} quarters")
print(f"VIX: {len(vix)} quarters")
print(f"IGREA: {len(igrea)} quarters")

# %%
# Plot Brent oil prices
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Brent
axes[0].plot(brent["date"], brent["value"])
axes[0].set_title("Brent Crude Oil Price (Quarterly Average)")
axes[0].set_ylabel("USD/barrel")

# VIX
axes[1].plot(vix["date"], vix["value"])
axes[1].set_title("VIX Volatility Index (Quarterly Average)")
axes[1].set_ylabel("VIX")

# IGREA
axes[2].plot(igrea["date"], igrea["value"])
axes[2].set_title("Kilian Global Economic Activity Index")
axes[2].set_ylabel("IGREA")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Baumeister Oil Shocks

# %%
# Load Baumeister shocks
baumeister = BaumeisterLoader()
shocks = baumeister.get_shocks_for_panel(start_date="2010-01-01")

print(f"Shock data shape: {shocks.shape}")
print(f"Columns: {list(shocks.columns)}")
shocks.head()

# %% [markdown]
# ## 3. Region Crosswalk

# %%
# Display region crosswalk
print("Region Crosswalk (new -> parent):")
for new, parent in REGION_CROSSWALK.items():
    print(f"  {new} -> {parent}")

print(f"\nCanonical regions ({len(CANONICAL_REGIONS)}):")
for r in CANONICAL_REGIONS:
    print(f"  - {r}")

# %%
# Test harmonization
builder = PanelBuilder()

test_regions = ["Abay", "Turkestan", "Almaty Region", "Atyrau"]
print("\nHarmonization test:")
for region in test_regions:
    harmonized = builder.harmonize_region(region)
    changed = " (mapped)" if harmonized != region else ""
    print(f"  {region} -> {harmonized}{changed}")

# %% [markdown]
# ## 4. Panel Skeleton

# %%
# Create panel skeleton
skeleton = builder._create_skeleton(2015, 2020)
print(f"Panel skeleton shape: {skeleton.shape}")
print(f"Unique regions: {skeleton['region'].nunique()}")
print(f"Unique quarters: {skeleton['quarter'].nunique()}")
skeleton.head(10)
