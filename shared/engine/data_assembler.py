"""
Data Assembler for DAG Edge Estimation.

Maps DAG node IDs to actual pandas Series from cached data files.
Handles frequency alignment (daily -> monthly, monthly -> quarterly)
and transforms (log, diff, innovation extraction).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Base data directory
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Node-to-data mapping
# ---------------------------------------------------------------------------

def _load_parquet(path: Path) -> pd.DataFrame:
    """Load parquet file, raising clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_parquet(path)


def _daily_to_monthly_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """Resample daily series to monthly mean."""
    s = df[col].dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.resample("MS").mean().dropna()


def _monthly_to_quarterly_mean(s: pd.Series) -> pd.Series:
    """Resample monthly series to quarterly mean (period-start)."""
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.resample("QS").mean().dropna()


def _quarterly_label_to_date(label: str) -> pd.Timestamp:
    """Convert '2024Q3' to Timestamp at quarter start."""
    year = int(label[:4])
    q = int(label[-1])
    month = (q - 1) * 3 + 1
    return pd.Timestamp(year=year, month=month, day=1)


def _ar1_innovation(s: pd.Series) -> pd.Series:
    """Extract AR(1) innovation: residual from regressing s_t on s_{t-1}."""
    y = s.iloc[1:].values
    x = s.iloc[:-1].values
    # Simple OLS: y = a + b*x + e
    x_mat = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(x_mat, y, rcond=None)[0]
    resid = y - x_mat @ beta
    return pd.Series(resid, index=s.index[1:], name=s.name)


# ---------------------------------------------------------------------------
# KSPI quarterly data loader (with extended bank-level data support)
# ---------------------------------------------------------------------------

def _load_kspi_quarterly() -> pd.DataFrame:
    """Load KSPI historical KPIs from JSON file.

    Tries kaspi_bank_extended_kpis.json first (extended bank-level data),
    then falls back to kspi_historical_kpis.json.
    """
    # Try extended bank-level data first
    extended_path = RAW_DIR / "kspi" / "kaspi_bank_extended_kpis.json"
    json_path = RAW_DIR / "kspi" / "kspi_historical_kpis.json"

    path_to_load = extended_path if extended_path.exists() else json_path
    if not path_to_load.exists():
        raise FileNotFoundError(f"KSPI historical KPIs not found: {json_path}")

    with open(path_to_load) as f:
        data = json.load(f)

    records = data["quarters"]
    df = pd.DataFrame(records)

    # Convert quarter labels to datetime index
    df["date"] = df["quarter"].apply(_quarterly_label_to_date)
    df = df.set_index("date").sort_index()
    return df


def _load_kspi_quarterly_filtered(freq: str = "quarterly") -> pd.DataFrame:
    """Load KSPI KPIs filtered by frequency for estimation.

    Args:
        freq: "quarterly" for quarterly-only obs, "annual" for annual-only obs,
              "all" for all estimation-eligible obs regardless of frequency.

    Returns:
        DataFrame filtered to estimation-eligible observations at the
        requested frequency.
    """
    df = _load_kspi_quarterly()

    # If frequency/estimation_eligible columns exist, filter
    if "estimation_eligible" in df.columns:
        df = df[df["estimation_eligible"] == True]  # noqa: E712

    if "frequency" in df.columns and freq != "all":
        if freq == "quarterly":
            df = df[df["frequency"].isin(["quarterly", "semiannual"])]
        elif freq == "annual":
            df = df[df["frequency"] == "annual"]

    return df


def get_estimation_eligible(series: pd.Series, source_df: pd.DataFrame) -> pd.Series:
    """Filter a series to only estimation-eligible observations.

    Args:
        series: The KPI time series
        source_df: The full DataFrame with metadata columns

    Returns:
        Filtered series with only estimation-eligible observations
    """
    if "estimation_eligible" in source_df.columns:
        eligible_idx = source_df[source_df["estimation_eligible"] == True].index  # noqa: E712
        return series.reindex(eligible_idx).dropna()
    return series


def compute_share_interpolated(source_df: pd.DataFrame) -> float:
    """Compute fraction of interpolated observations in a dataset.

    Returns value between 0 and 1.
    """
    if "is_interpolated" not in source_df.columns:
        return 0.0
    n_total = len(source_df)
    if n_total == 0:
        return 0.0
    n_interpolated = source_df["is_interpolated"].sum()
    return float(n_interpolated / n_total)


# ---------------------------------------------------------------------------
# NBK policy rate loader
# ---------------------------------------------------------------------------

def _load_nbk_rate_monthly() -> pd.Series:
    """
    Load NBK policy rate as monthly series.

    Uses the exchange rate file's date range to construct a monthly
    policy rate series from known NBK rate decisions.
    """
    # Known NBK base rate history (end-of-month values, %)
    # Source: National Bank of Kazakhstan press releases
    nbk_rates = {
        "2019-01-01": 9.25, "2019-04-01": 9.0, "2019-07-01": 9.0,
        "2019-10-01": 9.25,
        "2020-01-01": 9.25, "2020-03-01": 9.5, "2020-04-01": 9.5,
        "2020-07-01": 9.0, "2020-10-01": 9.0, "2020-12-01": 9.0,
        "2021-01-01": 9.0, "2021-03-01": 9.0, "2021-07-01": 9.25,
        "2021-10-01": 9.75, "2021-12-01": 9.75,
        "2022-01-01": 10.25, "2022-02-01": 13.5, "2022-04-01": 14.0,
        "2022-07-01": 14.5, "2022-08-01": 14.5, "2022-10-01": 14.5,
        "2022-12-01": 16.75,
        "2023-01-01": 16.75, "2023-04-01": 16.75, "2023-06-01": 16.75,
        "2023-08-01": 16.5, "2023-10-01": 15.75, "2023-12-01": 15.75,
        "2024-01-01": 15.25, "2024-03-01": 14.75, "2024-05-01": 14.25,
        "2024-07-01": 14.25, "2024-09-01": 14.25,
    }

    # Build monthly series with forward-fill
    idx = pd.date_range("2019-01-01", "2024-09-30", freq="MS")
    s = pd.Series(dtype=float, index=idx, name="nbk_rate")
    for dt_str, rate in nbk_rates.items():
        dt = pd.Timestamp(dt_str)
        if dt in s.index:
            s[dt] = rate
    s = s.ffill().dropna()
    return s


# ---------------------------------------------------------------------------
# Individual node loaders
# ---------------------------------------------------------------------------

NODE_LOADERS: dict[str, Any] = {}


def _register(node_id: str):
    """Decorator to register a node loader."""
    def wrapper(fn):
        NODE_LOADERS[node_id] = fn
        return fn
    return wrapper


@_register("oil_supply_shock")
def _load_oil_supply() -> pd.Series:
    df = _load_parquet(RAW_DIR / "baumeister_shocks" / "shocks.parquet")
    if "date" in df.columns:
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    col = "oil_supply_shock" if "oil_supply_shock" in df.columns else df.columns[0]
    return df[col].dropna().rename("oil_supply_shock")


@_register("oil_demand_shock")
def _load_oil_demand() -> pd.Series:
    df = _load_parquet(RAW_DIR / "baumeister_shocks" / "shocks.parquet")
    if "date" in df.columns:
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    col = "aggregate_demand_shock" if "aggregate_demand_shock" in df.columns else df.columns[1]
    return df[col].dropna().rename("oil_demand_shock")


@_register("vix_shock")
def _load_vix() -> pd.Series:
    df = _load_parquet(RAW_DIR / "fred" / "VIXCLS.parquet")
    if "date" in df.columns:
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    col = [c for c in df.columns if "vix" in c.lower() or "value" in c.lower() or c == "VIXCLS"]
    col = col[0] if col else df.columns[0]
    monthly = _daily_to_monthly_mean(df, col)
    innov = _ar1_innovation(monthly)
    return innov.rename("vix_shock")


@_register("kzt_usd")
def _load_fx() -> pd.Series:
    df = _load_parquet(RAW_DIR / "nbk" / "usd_kzt.parquet")
    if "date" in df.columns:
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    col = "rate" if "rate" in df.columns else df.columns[0]
    # Monthly end-of-period
    monthly = df[col].resample("MS").last().dropna()
    return monthly.rename("kzt_usd")


@_register("brent_price")
def _load_brent() -> pd.Series:
    df = _load_parquet(RAW_DIR / "fred" / "DCOILBRENTEU.parquet")
    if "date" in df.columns:
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    col = [c for c in df.columns if "brent" in c.lower() or "oil" in c.lower() or "value" in c.lower() or c == "DCOILBRENTEU"]
    col = col[0] if col else df.columns[0]
    monthly = _daily_to_monthly_mean(df, col)
    return monthly.rename("brent_price")


@_register("cpi_headline")
def _load_cpi_headline() -> pd.Series:
    df = _load_parquet(PROCESSED_DIR / "fx_passthrough" / "headline_cpi.parquet")
    if "date" in df.columns:
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    col = "inflation_mom" if "inflation_mom" in df.columns else df.columns[0]
    return df[col].dropna().rename("cpi_headline")


@_register("cpi_tradable")
def _load_cpi_tradable() -> pd.Series:
    df = _load_parquet(PROCESSED_DIR / "fx_passthrough" / "cpi_panel.parquet")
    if "high_import" in df.columns:
        sub = df[df["high_import"] == True]  # noqa: E712
    elif "tradable" in df.columns:
        sub = df[df["tradable"] == True]  # noqa: E712
    else:
        sub = df
    col = "inflation_mom" if "inflation_mom" in sub.columns else sub.columns[-1]
    if "date" in sub.columns:
        grouped = sub.groupby("date")[col].mean()
    else:
        grouped = sub.groupby(sub.index)[col].mean()
    if not isinstance(grouped.index, pd.DatetimeIndex):
        grouped.index = pd.to_datetime(grouped.index)
    return grouped.dropna().rename("cpi_tradable")


@_register("cpi_nontradable")
def _load_cpi_nontradable() -> pd.Series:
    df = _load_parquet(PROCESSED_DIR / "fx_passthrough" / "cpi_panel.parquet")
    if "high_import" in df.columns:
        sub = df[df["high_import"] == False]  # noqa: E712
    elif "tradable" in df.columns:
        sub = df[df["tradable"] == False]  # noqa: E712
    else:
        sub = df
    col = "inflation_mom" if "inflation_mom" in sub.columns else sub.columns[-1]
    if "date" in sub.columns:
        grouped = sub.groupby("date")[col].mean()
    else:
        grouped = sub.groupby(sub.index)[col].mean()
    if not isinstance(grouped.index, pd.DatetimeIndex):
        grouped.index = pd.to_datetime(grouped.index)
    return grouped.dropna().rename("cpi_nontradable")


@_register("nbk_policy_rate")
def _load_nbk_rate() -> pd.Series:
    return _load_nbk_rate_monthly()


@_register("nominal_income")
def _load_nominal_income() -> pd.Series:
    df = _load_parquet(PROCESSED_DIR / "fx_passthrough" / "income_series.parquet")
    if "date" in df.columns:
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    col = "nominal_income_growth" if "nominal_income_growth" in df.columns else df.columns[0]
    return df[col].dropna().rename("nominal_income")


@_register("real_expenditure")
def _load_real_expenditure() -> pd.Series:
    # Try spending_series first, then expenditure_series
    for fname in ["spending_series.parquet", "expenditure_series.parquet"]:
        path = PROCESSED_DIR / "fx_passthrough" / fname
        if path.exists():
            df = _load_parquet(path)
            if "date" in df.columns:
                df = df.set_index("date")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            col = "real_expenditure_growth" if "real_expenditure_growth" in df.columns else df.columns[0]
            return df[col].dropna().rename("real_expenditure")
    raise FileNotFoundError("No expenditure series found")


# KSPI-specific node loaders
@_register("npl_kspi")
def _load_npl_kspi() -> pd.Series:
    df = _load_kspi_quarterly()
    return df["npl_ratio"].dropna().rename("npl_kspi")


@_register("cor_kspi")
def _load_cor_kspi() -> pd.Series:
    df = _load_kspi_quarterly()
    return df["cor"].dropna().rename("cor_kspi")


@_register("deposit_cost_kspi")
def _load_deposit_cost() -> pd.Series:
    df = _load_kspi_quarterly()
    return df["deposit_cost"].dropna().rename("deposit_cost_kspi")


@_register("loan_portfolio_kspi")
def _load_loan_portfolio() -> pd.Series:
    df = _load_kspi_quarterly()
    return df["net_loans"].dropna().rename("loan_portfolio_kspi")


@_register("rwa_kspi")
def _load_rwa() -> pd.Series:
    df = _load_kspi_quarterly()
    return df["rwa"].dropna().rename("rwa_kspi")


@_register("total_capital_kspi")
def _load_total_capital() -> pd.Series:
    df = _load_kspi_quarterly()
    return df["total_capital"].dropna().rename("total_capital_kspi")


@_register("payments_revenue_kspi")
def _load_payments_revenue() -> pd.Series:
    df = _load_kspi_quarterly()
    return df["payments_revenue"].dropna().rename("payments_revenue_kspi")


@_register("portfolio_mix_kspi")
def _load_portfolio_mix() -> pd.Series:
    """Derived: BNPL share proxy = payments_revenue / net_loans."""
    df = _load_kspi_quarterly()
    mix = (df["payments_revenue"] / df["net_loans"]).dropna()
    return mix.rename("portfolio_mix_kspi")


@_register("k2_ratio_kspi")
def _load_k2_ratio() -> pd.Series:
    df = _load_kspi_quarterly()
    return df["k2_ratio"].dropna().rename("k2_ratio_kspi")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_node_series(node_id: str) -> pd.Series:
    """
    Load data for a single DAG node.

    Args:
        node_id: The DAG node identifier

    Returns:
        pd.Series indexed by date
    """
    loader = NODE_LOADERS.get(node_id)
    if loader is None:
        raise KeyError(f"No data loader registered for node: {node_id}")
    return loader()


# Edge-to-node mapping: defines treatment and outcome nodes for each edge
EDGE_NODE_MAP: dict[str, tuple[str, str]] = {
    # Group A: Monthly LP
    "oil_supply_to_brent": ("oil_supply_shock", "brent_price"),
    "oil_supply_to_fx": ("oil_supply_shock", "kzt_usd"),
    "oil_demand_to_fx": ("oil_demand_shock", "kzt_usd"),
    "vix_to_fx": ("vix_shock", "kzt_usd"),
    "cpi_to_nbk_rate": ("cpi_headline", "nbk_policy_rate"),
    "fx_to_nbk_rate": ("kzt_usd", "nbk_policy_rate"),
    # Group B: Immutable
    "fx_to_cpi_tradable": ("kzt_usd", "cpi_tradable"),
    "fx_to_cpi_nontradable": ("kzt_usd", "cpi_nontradable"),
    "cpi_to_nominal_income": ("cpi_headline", "nominal_income"),
    "fx_to_real_expenditure": ("kzt_usd", "real_expenditure"),
    # Group C-Q: Quarterly LP with KSPI data (true quarterly obs only)
    "shock_to_npl_kspi": ("cpi_tradable", "npl_kspi"),
    "shock_to_cor_kspi": ("cpi_tradable", "cor_kspi"),
    "nbk_rate_to_deposit_cost": ("nbk_policy_rate", "deposit_cost_kspi"),
    "nbk_rate_to_cor": ("nbk_policy_rate", "cor_kspi"),
    # Group C-KSPI: KSPI-only, no extension possible
    "expenditure_to_payments_revenue": ("real_expenditure", "payments_revenue_kspi"),
    "portfolio_mix_to_rwa": ("portfolio_mix_kspi", "rwa_kspi"),
    # Group C-Bridge: Accounting bridges (reclassified from LP)
    "loan_portfolio_to_rwa": ("loan_portfolio_kspi", "rwa_kspi"),
    "cor_to_capital": ("cor_kspi", "total_capital_kspi"),
    # Group D: Identity
    "capital_to_k2": ("total_capital_kspi", "k2_ratio_kspi"),
    "rwa_to_k2": ("rwa_kspi", "k2_ratio_kspi"),
}

# Sector panel edge mapping: maps panel edge IDs to their specifications
PANEL_EDGE_SPECS: dict[str, dict[str, Any]] = {
    # shock_to_npl_sector: Imported inflation shock → NPL
    # Banks with more retail/consumer exposure should see larger NPL response
    "shock_to_npl_sector": {
        "outcome": "npl_ratio",
        "shock_node": "cpi_tradable",  # Use imported inflation instrument
        "shock_transform": "innovation",  # AR residual or use as-is if already instrument
        "exposure": "E_consumer",  # Primary: retail_loan_share
        "exposure_alternatives": ["E_unsecured", "capital_buffer"],
        "kspi_companion": "shock_to_npl_kspi",
        "interpretation": "Differential NPL response per unit consumer loan share, per unit imported inflation shock",
    },
    # shock_to_cor_sector: Imported inflation shock → CoR
    # Same exposure logic as NPL
    "shock_to_cor_sector": {
        "outcome": "cor",
        "shock_node": "cpi_tradable",
        "shock_transform": "innovation",
        "exposure": "E_consumer",
        "exposure_alternatives": ["E_unsecured", "baseline_npl_ratio"],
        "kspi_companion": "shock_to_cor_kspi",
        "interpretation": "Differential CoR response per unit consumer loan share, per unit imported inflation shock",
    },
    # nbk_rate_to_deposit_cost_sector: MP shock → deposit cost
    # Banks with more interest-bearing deposits should respond more
    "nbk_rate_to_deposit_cost_sector": {
        "outcome": "deposit_cost",
        "shock_node": "nbk_policy_rate",
        "shock_transform": "mp_innovation",  # Δrate - E[Δrate|info_{t-1}]
        "exposure": "E_demand_dep",  # Primary: interest_bearing_deposit_share
        "exposure_alternatives": ["E_term_dep", "E_retail_dep"],
        "kspi_companion": "nbk_rate_to_deposit_cost",
        "interpretation": "Differential deposit cost pass-through per unit interest-bearing deposit share, per unit MP shock",
    },
    # nbk_rate_to_cor_sector: MP shock → CoR
    # Banks with more rate-sensitive borrowers should see larger CoR response
    "nbk_rate_to_cor_sector": {
        "outcome": "cor",
        "shock_node": "nbk_policy_rate",
        "shock_transform": "mp_innovation",
        "exposure": "E_shortterm",  # Primary: short-term/variable-rate loan share
        "exposure_alternatives": ["E_consumer", "avg_loan_duration_proxy"],
        "kspi_companion": "nbk_rate_to_cor",
        "interpretation": "Differential CoR response per unit short-term loan share, per unit MP shock",
    },
}

# Accounting bridge edges: deterministic sensitivities, not regressions
ACCOUNTING_BRIDGE_EDGES = {"loan_portfolio_to_rwa", "cor_to_capital"}

# Edges that need log-return transforms on treatment or outcome
LOG_RETURN_TREATMENT = {"oil_supply_to_fx", "oil_demand_to_fx", "vix_to_fx",
                        "fx_to_nbk_rate", "fx_to_cpi_tradable",
                        "fx_to_cpi_nontradable", "fx_to_real_expenditure"}
LOG_RETURN_OUTCOME = {"oil_supply_to_brent"}
LOG_TRANSFORM_TREATMENT = {"loan_portfolio_to_rwa"}
LOG_TRANSFORM_OUTCOME = {"loan_portfolio_to_rwa"}


def _align_frequencies(treatment: pd.Series, outcome: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Align treatment and outcome to common frequency.

    Strategy:
    - If both have the same frequency, do nothing
    - If treatment is higher freq, aggregate to outcome's freq
    - If outcome is higher freq, aggregate to treatment's freq
    """
    if treatment.empty or outcome.empty:
        return treatment, outcome

    # Infer frequencies
    t_freq = pd.infer_freq(treatment.index)
    o_freq = pd.infer_freq(outcome.index)

    # Monthly starts
    t_is_monthly = t_freq and t_freq.startswith("M")
    o_is_monthly = o_freq and o_freq.startswith("M")
    t_is_quarterly = t_freq and t_freq.startswith("Q")
    o_is_quarterly = o_freq and o_freq.startswith("Q")

    # Calculate median gap
    t_gap = treatment.index.to_series().diff().median()
    o_gap = outcome.index.to_series().diff().median()

    if t_gap is pd.NaT or o_gap is pd.NaT:
        # Can't determine - just inner join on index
        common = treatment.index.intersection(outcome.index)
        return treatment.reindex(common), outcome.reindex(common)

    # If treatment is monthly and outcome is quarterly -> aggregate treatment
    if t_gap < pd.Timedelta(days=60) and o_gap >= pd.Timedelta(days=60):
        treatment = _monthly_to_quarterly_mean(treatment)
    # If treatment is quarterly and outcome is monthly -> aggregate outcome
    elif t_gap >= pd.Timedelta(days=60) and o_gap < pd.Timedelta(days=60):
        outcome = _monthly_to_quarterly_mean(outcome)

    # Inner join on overlapping dates
    common = treatment.index.intersection(outcome.index)
    if len(common) == 0:
        # Try approximate merge
        combined = pd.merge_asof(
            treatment.to_frame("treatment").sort_index(),
            outcome.to_frame("outcome").sort_index(),
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta(days=15),
        )
        return combined["treatment"].dropna(), combined["outcome"].dropna()

    return treatment.reindex(common).dropna(), outcome.reindex(common).dropna()


def assemble_edge_data(edge_id: str) -> pd.DataFrame:
    """
    Assemble treatment and outcome data for an edge.

    Returns a DataFrame indexed by date with columns:
    - 'treatment': The treatment variable (possibly transformed)
    - 'outcome': The outcome variable (possibly transformed)

    Handles:
    - Frequency alignment (daily -> monthly, monthly -> quarterly)
    - Log/diff transforms as needed
    - AR(1) innovation extraction for VIX
    """
    if edge_id not in EDGE_NODE_MAP:
        raise KeyError(f"Unknown edge: {edge_id}")

    treatment_node, outcome_node = EDGE_NODE_MAP[edge_id]

    # Load raw series
    treatment = load_node_series(treatment_node)
    outcome = load_node_series(outcome_node)

    # Apply transforms: log-return on treatment
    if edge_id in LOG_RETURN_TREATMENT:
        log_level = np.log(treatment.clip(lower=1e-6))
        treatment = log_level.diff().dropna()
        treatment.name = f"dlog_{treatment_node}"

    if edge_id in LOG_RETURN_OUTCOME:
        log_level = np.log(outcome.clip(lower=1e-6))
        outcome = log_level.diff().dropna()
        outcome.name = f"dlog_{outcome_node}"

    if edge_id in LOG_TRANSFORM_TREATMENT:
        treatment = np.log(treatment.clip(lower=1e-6))
        treatment = treatment.diff().dropna()
        treatment.name = f"dlog_{treatment_node}"

    if edge_id in LOG_TRANSFORM_OUTCOME:
        outcome = np.log(outcome.clip(lower=1e-6))
        outcome = outcome.diff().dropna()
        outcome.name = f"dlog_{outcome_node}"

    # Align frequencies
    treatment, outcome = _align_frequencies(treatment, outcome)

    # Build DataFrame
    combined = pd.DataFrame({
        "treatment": treatment,
        "outcome": outcome,
    }).dropna()

    if combined.empty:
        logger.warning(f"Edge {edge_id}: no overlapping data after alignment")

    logger.info(
        f"Edge {edge_id}: assembled {len(combined)} obs, "
        f"date range {combined.index.min()} to {combined.index.max()}"
    )

    return combined


def get_edge_group(edge_id: str) -> str:
    """Classify edge into estimation group.

    Groups:
        MONTHLY_LP:     Monthly LP (Group A, 6 edges)
        IMMUTABLE:      Validated evidence (Group B, 4 edges)
        QUARTERLY_LP:   Quarterly LP, KSPI-only, true quarterly obs (Group C-Q, 4 edges)
        ANNUAL_LP:      Annual LP robustness (Group C-A, same 4 edges at annual freq)
        PANEL_LP:       Sector panel LP, Exposure×Shock (Group C-Panel, 4 edges)
        KSPI_ONLY:      KSPI-only, no extension (Group C-KSPI, 2 edges)
        ACCOUNTING_BRIDGE: Accounting bridges (Group C-Bridge, 2 edges)
        IDENTITY:       Mechanical identity (Group D, 2 edges)
    """
    monthly_lp = {
        "oil_supply_to_brent", "oil_supply_to_fx", "oil_demand_to_fx",
        "vix_to_fx", "cpi_to_nbk_rate", "fx_to_nbk_rate",
    }
    immutable = {
        "fx_to_cpi_tradable", "fx_to_cpi_nontradable",
        "cpi_to_nominal_income", "fx_to_real_expenditure",
    }
    quarterly_lp = {
        "shock_to_npl_kspi", "shock_to_cor_kspi",
        "nbk_rate_to_deposit_cost", "nbk_rate_to_cor",
    }
    kspi_only = {
        "expenditure_to_payments_revenue", "portfolio_mix_to_rwa",
    }
    accounting_bridge = {
        "loan_portfolio_to_rwa", "cor_to_capital",
    }
    panel_lp = set(PANEL_EDGE_SPECS.keys())
    identity = {"capital_to_k2", "rwa_to_k2"}

    if edge_id in monthly_lp:
        return "MONTHLY_LP"
    elif edge_id in immutable:
        return "IMMUTABLE"
    elif edge_id in quarterly_lp:
        return "QUARTERLY_LP"
    elif edge_id in panel_lp:
        return "PANEL_LP"
    elif edge_id in kspi_only:
        return "KSPI_ONLY"
    elif edge_id in accounting_bridge:
        return "ACCOUNTING_BRIDGE"
    elif edge_id in identity:
        return "IDENTITY"
    else:
        return "UNKNOWN"
