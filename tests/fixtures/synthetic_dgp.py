"""
Synthetic Data Generating Processes for Integration Tests.

Provides deterministic DGP functions with fixed seeds for reproducible testing.
Each function returns a DataFrame ready for estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def make_clean_ts(n: int = 100, beta: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Clean time-series with known causal effect.

    y_t = beta * x_t + epsilon_t
    x_t ~ N(0, 1), epsilon_t ~ N(0, 0.5)

    Strong signal-to-noise ratio for reliable estimation.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    eps = rng.normal(0, 0.5, n)
    y = beta * x + eps
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame({"date": dates, "x": x, "y": y})


def make_confounded_ts(
    n: int = 100, beta_true: float = 0.5, beta_confound: float = 0.8, seed: int = 42,
) -> pd.DataFrame:
    """Time-series with omitted variable bias.

    z_t ~ N(0, 1)       (unobserved confounder)
    x_t = 0.6 * z_t + u_t
    y_t = beta_true * x_t + beta_confound * z_t + epsilon_t

    OLS of y on x (omitting z) yields biased estimate.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    u = rng.normal(0, 0.5, n)
    x = 0.6 * z + u
    eps = rng.normal(0, 0.5, n)
    y = beta_true * x + beta_confound * z + eps
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame({"date": dates, "x": x, "y": y, "z": z})


def make_nonstationary_ts(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Non-stationary time-series (random walks).

    x_t = x_{t-1} + u_t (random walk)
    y_t = y_{t-1} + v_t (independent random walk)

    Spurious regression: OLS will show significant effect despite no causation.
    Strong signal for TSGuard to detect.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    y = np.cumsum(rng.standard_normal(n))
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame({"date": dates, "x": x, "y": y})


def make_regime_break_ts(
    n: int = 100,
    beta_pre: float = 0.5,
    beta_post: float = -0.3,
    break_at: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Time-series with structural break (sign flip).

    y_t = beta_pre * x_t + eps  for t < break_at
    y_t = beta_post * x_t + eps for t >= break_at
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    eps = rng.normal(0, 0.3, n)
    beta = np.where(np.arange(n) < break_at, beta_pre, beta_post)
    y = beta * x + eps
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    regime = np.where(np.arange(n) < break_at, "pre", "post")
    return pd.DataFrame({"date": dates, "x": x, "y": y, "regime": regime})


def make_reverse_causal_ts(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Time-series where causation runs from Y to X (reverse causation).

    y_t ~ AR(1) + innovation
    x_t = 0.5 * y_{t-1} + u_t   (X is caused by lagged Y)

    Leads test should detect that future X predicts current Y.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0] = rng.standard_normal()
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + rng.normal(0, 0.5)
    x = np.zeros(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = 0.5 * y[t - 1] + rng.normal(0, 0.5)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame({"date": dates, "x": x, "y": y})


def make_small_sample_ts(n: int = 20, beta: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Small-sample time-series for testing rating caps."""
    return make_clean_ts(n=n, beta=beta, seed=seed)


def make_iv_dgp(
    n: int = 1000,
    beta: float = 0.5,
    gamma: float = 0.8,
    pi: float = 0.6,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Instrumental Variables DGP: Z → X → Y with confounding.

    Structure:
        U ~ N(0, 1)           (unobserved confounder)
        Z ~ N(0, 1)           (instrument, Z ⊥ U)
        X = pi * Z + gamma * U + v   (first stage, v ~ N(0, 0.3))
        Y = beta * X + gamma * U + e  (structural, e ~ N(0, 0.5))

    OLS of Y on X is biased (omitted U). IV using Z recovers beta.

    Args:
        n: Sample size
        beta: True causal effect of X on Y
        gamma: Confounding strength
        pi: First-stage coefficient (instrument relevance)
        seed: Random seed

    Returns:
        (df, truth) where truth = {"beta": beta, "first_stage_pi": pi, ...}
    """
    rng = np.random.default_rng(seed)

    U = rng.standard_normal(n)
    Z = rng.standard_normal(n)
    v = rng.normal(0, 0.3, n)
    e = rng.normal(0, 0.5, n)

    X = pi * Z + gamma * U + v
    Y = beta * X + gamma * U + e

    df = pd.DataFrame({"Y": Y, "X": X, "Z": Z})

    # Compute theoretical first-stage F statistic
    # F ≈ n * pi^2 * Var(Z) / Var(v + gamma*U)
    var_z = 1.0
    var_noise = 0.3**2 + gamma**2  # Var(v) + gamma^2*Var(U)
    approx_first_stage_f = n * (pi**2 * var_z) / var_noise

    truth = {
        "beta": beta,
        "first_stage_pi": pi,
        "confounding_gamma": gamma,
        "approx_first_stage_F": approx_first_stage_f,
    }
    return df, truth


def make_did_dgp(
    n_units: int = 100,
    n_periods: int = 20,
    treat_period: int = 10,
    att: float = 2.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Canonical 2x2 Difference-in-Differences DGP.

    Structure:
        Y_it = alpha_i + delta_t + att * (treated_i × post_t) + epsilon_it

    Half the units are treated (first n_units/2). Treatment turns on at treat_period.
    Parallel trends hold by construction.

    Args:
        n_units: Number of units
        n_periods: Number of time periods
        treat_period: Period when treatment starts (0-indexed)
        att: Average Treatment effect on the Treated
        seed: Random seed

    Returns:
        (df, truth) where truth = {"att": att, ...}
    """
    rng = np.random.default_rng(seed)

    n_treated = n_units // 2
    unit_ids = np.repeat(np.arange(n_units), n_periods)
    time_ids = np.tile(np.arange(n_periods), n_units)

    # Unit fixed effects
    alpha = rng.standard_normal(n_units)
    alpha_expanded = alpha[unit_ids]

    # Time fixed effects (common trend)
    delta = np.cumsum(rng.normal(0.1, 0.05, n_periods))
    delta_expanded = delta[time_ids]

    # Treatment assignment
    treated = (unit_ids < n_treated).astype(float)
    post = (time_ids >= treat_period).astype(float)
    treatment = treated * post

    # Outcome with parallel trends by construction
    epsilon = rng.normal(0, 0.5, len(unit_ids))
    Y = alpha_expanded + delta_expanded + att * treatment + epsilon

    df = pd.DataFrame({
        "unit": unit_ids,
        "time": time_ids,
        "Y": Y,
        "treated": treated,
        "post": post,
        "treatment": treatment,
    })

    truth = {
        "att": att,
        "n_treated": n_treated,
        "n_control": n_units - n_treated,
        "treat_period": treat_period,
        "n_pre_periods": treat_period,
        "n_post_periods": n_periods - treat_period,
    }
    return df, truth


def make_sharp_rdd_dgp(
    n: int = 1000,
    tau: float = 3.0,
    cutoff: float = 0.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Sharp RDD DGP: treatment assigned at running variable cutoff.

    Structure:
        X ~ Uniform(-1, 1)     (running variable)
        D = 1[X >= cutoff]     (sharp treatment)
        Y = 0.5*X + tau*D + eps, eps ~ N(0, 0.5)

    Args:
        n: Sample size
        tau: True treatment effect at cutoff
        cutoff: Cutoff value
        seed: Random seed

    Returns:
        (df, truth) where truth = {"tau": tau, "cutoff": cutoff}
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    D = (X >= cutoff).astype(float)
    eps = rng.normal(0, 0.5, n)
    Y = 0.5 * X + tau * D + eps
    df = pd.DataFrame({"Y": Y, "X": X, "D": D})
    return df, {"tau": tau, "cutoff": cutoff}


def make_fuzzy_rdd_dgp(
    n: int = 1000,
    late: float = 2.5,
    cutoff: float = 0.0,
    compliance_rate: float = 0.7,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Fuzzy RDD DGP: imperfect compliance at cutoff.

    Structure:
        X ~ Uniform(-1, 1)            (running variable)
        Z = 1[X >= cutoff]            (instrument / eligibility)
        D = Z * Bernoulli(compliance_rate) + (1-Z) * Bernoulli(0.1)
        Y = 0.5*X + late*D + eps, eps ~ N(0, 0.5)

    The LATE at the cutoff equals ``late``.

    Args:
        n: Sample size
        late: Local Average Treatment Effect
        cutoff: Cutoff value
        compliance_rate: Pr(D=1 | Z=1)
        seed: Random seed

    Returns:
        (df, truth) where truth = {"late": late, "cutoff": cutoff, ...}
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    Z = (X >= cutoff).astype(float)
    # Fuzzy compliance
    D = np.where(
        Z == 1,
        rng.binomial(1, compliance_rate, n),
        rng.binomial(1, 0.1, n),
    ).astype(float)
    eps = rng.normal(0, 0.5, n)
    Y = 0.5 * X + late * D + eps
    df = pd.DataFrame({"Y": Y, "X": X, "D": D, "Z": Z})
    return df, {
        "late": late,
        "cutoff": cutoff,
        "compliance_rate": compliance_rate,
    }


def make_panel_fe_dgp(
    n_units: int = 20,
    n_periods: int = 30,
    beta: float = 2.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Panel FE DGP: entity + time fixed effects with treatment effect.

    Structure:
        Y_it = alpha_i + delta_t + beta * X_it + epsilon_it

    Args:
        n_units: Number of units (entities)
        n_periods: Number of time periods
        beta: True treatment effect
        seed: Random seed

    Returns:
        (df, truth) where df has columns: unit, time, Y, X
    """
    rng = np.random.default_rng(seed)

    unit_ids = np.repeat(np.arange(n_units), n_periods)
    time_ids = np.tile(np.arange(n_periods), n_units)

    # Entity fixed effects
    alpha = rng.standard_normal(n_units) * 2
    alpha_expanded = alpha[unit_ids]

    # Time fixed effects
    delta = np.cumsum(rng.normal(0.1, 0.05, n_periods))
    delta_expanded = delta[time_ids]

    # Treatment (continuous, varying by unit and time)
    X = rng.standard_normal(len(unit_ids))

    # Outcome
    epsilon = rng.normal(0, 0.5, len(unit_ids))
    Y = alpha_expanded + delta_expanded + beta * X + epsilon

    df = pd.DataFrame({
        "unit": unit_ids,
        "time": time_ids,
        "Y": Y,
        "X": X,
    })
    return df, {"beta": beta, "n_units": n_units, "n_periods": n_periods}


def make_regression_kink_dgp(
    n: int = 1000,
    kink_point: float = 0.0,
    slope_change: float = 1.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Regression Kink DGP: slope changes at kink point.

    Structure:
        X ~ Uniform(-2, 2)     (running variable)
        Y = 0.5*X + slope_change * max(X - kink_point, 0) + eps

    The change in slope at kink_point equals slope_change.

    Args:
        n: Sample size
        kink_point: Location of the kink
        slope_change: True change in slope at kink
        seed: Random seed

    Returns:
        (df, truth) where truth = {"slope_change": slope_change, "kink_point": kink_point}
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, n)
    eps = rng.normal(0, 0.3, n)

    # Piecewise linear: slope = 0.5 below kink, slope = 0.5 + slope_change above kink
    Y = 0.5 * X + slope_change * np.maximum(X - kink_point, 0) + eps

    df = pd.DataFrame({"Y": Y, "X": X})
    return df, {"slope_change": slope_change, "kink_point": kink_point}


def make_minimal_dag_yaml(
    edges: list[dict[str, str]],
    nodes: list[dict[str, str]],
    tmp_path: Path,
    latents: list[dict[str, Any]] | None = None,
) -> Path:
    """Create a minimal DAG YAML file for testing.

    Args:
        edges: List of dicts with keys: id, from, to, and optional edge_type.
        nodes: List of dicts with keys: id, name, and optional description, unit.
        tmp_path: Directory to write the YAML file.
        latents: Optional list of latent confounder dicts.

    Returns:
        Path to the created YAML file.
    """
    dag: dict[str, Any] = {
        "schema_version": 1,
        "metadata": {
            "name": "test_dag",
            "description": "Synthetic DAG for testing",
            "target_node": nodes[-1]["id"] if nodes else "",
        },
        "nodes": [
            {
                "id": n["id"],
                "name": n.get("name", n["id"]),
                "description": n.get("description", ""),
                "unit": n.get("unit", "level"),
                "frequency": n.get("frequency", "monthly"),
                "observed": n.get("observed", True),
                "latent": n.get("latent", False),
            }
            for n in nodes
        ],
        "edges": [
            {
                "id": e["id"],
                "from": e["from"],
                "to": e["to"],
                "edge_type": e.get("edge_type", "causal"),
                "allowed_designs": e.get("allowed_designs", ["LOCAL_PROJECTIONS"]),
            }
            for e in edges
        ],
    }

    if latents:
        dag["latents"] = [
            {
                "id": lat["id"],
                "description": lat.get("description", ""),
                "affects": lat.get("affects", []),
            }
            for lat in latents
        ]

    dag_path = tmp_path / "test_dag.yaml"
    dag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dag_path, "w") as f:
        yaml.dump(dag, f, sort_keys=False)
    return dag_path
