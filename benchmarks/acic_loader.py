"""
ACIC 2023 Benchmark Data Loader.

Downloads and parses ACIC 2023 practice datasets for benchmarking
causal inference estimators against known ground truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/benchmarks/acic2023")


@dataclass
class ACICDataset:
    """A single ACIC benchmark dataset with ground truth."""

    dataset_id: str
    df: pd.DataFrame
    treatment_col: str
    outcome_col: str
    true_effects: dict[str, float]  # e.g. {"att": 2.5, "ate": 1.8}
    covariates: list[str]


def generate_acic_practice_data(
    n_datasets: int = 5,
    n_obs: int = 500,
    seed: int = 42,
) -> list[ACICDataset]:
    """Generate synthetic ACIC-style practice datasets.

    Creates datasets that mimic the ACIC 2023 competition format:
    - Binary treatment with selection on observables
    - Continuous outcome with heterogeneous treatment effects
    - Multiple covariates (confounders)
    - Known ground truth ATT/ATE

    Args:
        n_datasets: Number of datasets to generate
        n_obs: Observations per dataset
        seed: Base random seed

    Returns:
        List of ACICDataset objects with known ground truth
    """
    datasets = []
    rng = np.random.default_rng(seed)

    for i in range(n_datasets):
        ds_seed = seed + i * 1000
        ds_rng = np.random.default_rng(ds_seed)

        # Generate covariates
        x1 = ds_rng.standard_normal(n_obs)
        x2 = ds_rng.standard_normal(n_obs)
        x3 = ds_rng.binomial(1, 0.5, n_obs).astype(float)

        # Treatment assignment (selection on observables)
        propensity = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2 + 0.2 * x3 - 0.3)))
        treatment = ds_rng.binomial(1, propensity).astype(float)

        # True treatment effect (heterogeneous)
        base_effect = 1.0 + 0.5 * i  # Varies across datasets
        tau = base_effect + 0.3 * x1 + 0.2 * x2  # Individual effects

        # Outcome
        y0 = 2.0 + 1.5 * x1 + 0.8 * x2 + 0.5 * x3 + ds_rng.normal(0, 1, n_obs)
        y1 = y0 + tau
        y_obs = treatment * y1 + (1 - treatment) * y0

        # Ground truth
        ate = float(np.mean(tau))
        att = float(np.mean(tau[treatment == 1]))

        df = pd.DataFrame({
            "Y": y_obs,
            "treatment": treatment,
            "X1": x1,
            "X2": x2,
            "X3": x3,
        })

        datasets.append(ACICDataset(
            dataset_id=f"acic_practice_{i:03d}",
            df=df,
            treatment_col="treatment",
            outcome_col="Y",
            true_effects={"ate": ate, "att": att},
            covariates=["X1", "X2", "X3"],
        ))

    return datasets


def load_acic_data(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    n_practice: int = 5,
) -> list[ACICDataset]:
    """Load ACIC benchmark data.

    First checks for cached data on disk. If not found, generates
    synthetic practice data.

    Args:
        cache_dir: Directory to cache downloaded data
        n_practice: Number of practice datasets to generate

    Returns:
        List of ACICDataset objects
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.csv"

    # Check for cached datasets
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        datasets = []
        for _, row in manifest.iterrows():
            data_path = cache_dir / f"{row['dataset_id']}.csv"
            if data_path.exists():
                df = pd.read_csv(data_path)
                datasets.append(ACICDataset(
                    dataset_id=row["dataset_id"],
                    df=df,
                    treatment_col=row["treatment_col"],
                    outcome_col=row["outcome_col"],
                    true_effects={"ate": row["true_ate"], "att": row["true_att"]},
                    covariates=row["covariates"].split(","),
                ))
        if datasets:
            logger.info(f"Loaded {len(datasets)} cached ACIC datasets from {cache_dir}")
            return datasets

    # Generate practice data
    logger.info(f"Generating {n_practice} ACIC practice datasets")
    datasets = generate_acic_practice_data(n_datasets=n_practice)

    # Cache to disk
    manifest_rows = []
    for ds in datasets:
        ds.df.to_csv(cache_dir / f"{ds.dataset_id}.csv", index=False)
        manifest_rows.append({
            "dataset_id": ds.dataset_id,
            "treatment_col": ds.treatment_col,
            "outcome_col": ds.outcome_col,
            "true_ate": ds.true_effects["ate"],
            "true_att": ds.true_effects["att"],
            "covariates": ",".join(ds.covariates),
        })
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    logger.info(f"Cached {len(datasets)} datasets to {cache_dir}")

    return datasets
