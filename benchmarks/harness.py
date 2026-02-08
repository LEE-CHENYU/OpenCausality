"""
Benchmark Harness.

Runs estimator adapters against benchmark datasets and computes
evaluation metrics (bias, RMSE, coverage).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from benchmarks.acic_loader import ACICDataset
from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of running one adapter on one dataset."""

    adapter_name: str
    dataset_id: str
    estimate: float
    true_effect: float
    bias: float
    se: float
    ci_lower: float
    ci_upper: float
    covered: bool  # True if true_effect in [ci_lower, ci_upper]
    runtime_seconds: float
    n_obs: int


@dataclass
class BenchmarkSummary:
    """Aggregated summary across multiple datasets."""

    adapter_name: str
    n_datasets: int
    mean_bias: float
    rmse: float
    coverage_90: float
    mean_runtime: float

    def to_dict(self) -> dict:
        return {
            "adapter": self.adapter_name,
            "n_datasets": self.n_datasets,
            "mean_bias": round(self.mean_bias, 4),
            "rmse": round(self.rmse, 4),
            "coverage_90": round(self.coverage_90, 3),
            "mean_runtime_s": round(self.mean_runtime, 3),
        }


def run_adapter_on_dataset(
    adapter: EstimatorAdapter,
    dataset: ACICDataset,
    target_effect: str = "att",
) -> BenchmarkResult:
    """Run a single adapter on a single dataset.

    Args:
        adapter: The estimator adapter to benchmark
        dataset: ACIC dataset with ground truth
        target_effect: Which effect to evaluate ("att" or "ate")

    Returns:
        BenchmarkResult with metrics
    """
    true_effect = dataset.true_effects[target_effect]

    req = EstimationRequest(
        df=dataset.df,
        outcome=dataset.outcome_col,
        treatment=dataset.treatment_col,
        controls=dataset.covariates,
        edge_id=f"benchmark_{dataset.dataset_id}",
    )

    start = time.perf_counter()
    result = adapter.estimate(req)
    elapsed = time.perf_counter() - start

    bias = result.point - true_effect
    covered = result.ci_lower <= true_effect <= result.ci_upper

    return BenchmarkResult(
        adapter_name=result.method_name,
        dataset_id=dataset.dataset_id,
        estimate=result.point,
        true_effect=true_effect,
        bias=bias,
        se=result.se,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        covered=covered,
        runtime_seconds=elapsed,
        n_obs=result.n_obs,
    )


def run_benchmark(
    adapter: EstimatorAdapter,
    datasets: list[ACICDataset],
    target_effect: str = "att",
) -> list[BenchmarkResult]:
    """Run adapter across all datasets.

    Args:
        adapter: The estimator adapter to benchmark
        datasets: List of ACIC datasets
        target_effect: Which effect to evaluate

    Returns:
        List of BenchmarkResult for each dataset
    """
    results = []
    for ds in datasets:
        try:
            br = run_adapter_on_dataset(adapter, ds, target_effect)
            results.append(br)
        except Exception as e:
            logger.warning(f"Benchmark failed for {ds.dataset_id}: {e}")
    return results


def summarize_benchmark(results: list[BenchmarkResult]) -> BenchmarkSummary:
    """Compute summary statistics from benchmark results.

    Args:
        results: List of individual BenchmarkResult

    Returns:
        BenchmarkSummary with aggregated metrics
    """
    if not results:
        return BenchmarkSummary(
            adapter_name="unknown",
            n_datasets=0,
            mean_bias=float("nan"),
            rmse=float("nan"),
            coverage_90=float("nan"),
            mean_runtime=float("nan"),
        )

    biases = [r.bias for r in results]
    mean_bias = float(np.mean(biases))
    rmse = float(np.sqrt(np.mean(np.array(biases) ** 2)))
    coverage_90 = float(np.mean([r.covered for r in results]))
    mean_runtime = float(np.mean([r.runtime_seconds for r in results]))

    return BenchmarkSummary(
        adapter_name=results[0].adapter_name,
        n_datasets=len(results),
        mean_bias=mean_bias,
        rmse=rmse,
        coverage_90=coverage_90,
        mean_runtime=mean_runtime,
    )


def format_results_table(summaries: list[BenchmarkSummary]) -> str:
    """Format benchmark summaries as a plain-text table.

    Args:
        summaries: List of BenchmarkSummary objects

    Returns:
        Formatted table string
    """
    rows = [s.to_dict() for s in summaries]
    df = pd.DataFrame(rows)
    return df.to_string(index=False)
