"""
DGP-based Benchmark Suite.

Runs all adapters against their corresponding DGP fixtures and
produces a summary table.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from benchmarks.harness import BenchmarkResult, BenchmarkSummary, summarize_benchmark
from shared.engine.adapters.base import EstimationRequest, EstimationResult
from shared.engine.adapters.registry import get_adapter

logger = logging.getLogger(__name__)


def benchmark_lp_adapter() -> list[BenchmarkResult]:
    """Benchmark LPAdapter on clean DGP."""
    from tests.fixtures.synthetic_dgp import make_clean_ts

    adapter = get_adapter("LOCAL_PROJECTIONS")
    results = []

    for seed in range(42, 52):
        df = make_clean_ts(n=200, beta=0.5, seed=seed)
        data = df.rename(columns={"y": "outcome", "x": "treatment"}).set_index("date")

        req = EstimationRequest(
            df=data,
            outcome="outcome",
            treatment="treatment",
            edge_id=f"dgp_lp_seed{seed}",
        )
        result = adapter.estimate(req)
        bias = result.point - 0.5
        covered = result.ci_lower <= 0.5 <= result.ci_upper

        results.append(BenchmarkResult(
            adapter_name="LOCAL_PROJECTIONS",
            dataset_id=f"clean_ts_seed{seed}",
            estimate=result.point,
            true_effect=0.5,
            bias=bias,
            se=result.se,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            covered=covered,
            runtime_seconds=0.0,
            n_obs=result.n_obs,
        ))

    return results


def benchmark_iv_adapter() -> list[BenchmarkResult]:
    """Benchmark IV2SLSAdapter on IV DGP."""
    from tests.fixtures.synthetic_dgp import make_iv_dgp

    adapter = get_adapter("IV_2SLS")
    results = []

    for seed in range(42, 52):
        df, truth = make_iv_dgp(n=1000, beta=0.5, seed=seed)
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="X",
            instruments=["Z"],
            edge_id=f"dgp_iv_seed{seed}",
        )
        result = adapter.estimate(req)
        bias = result.point - truth["beta"]
        covered = result.ci_lower <= truth["beta"] <= result.ci_upper

        results.append(BenchmarkResult(
            adapter_name="IV_2SLS",
            dataset_id=f"iv_dgp_seed{seed}",
            estimate=result.point,
            true_effect=truth["beta"],
            bias=bias,
            se=result.se,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            covered=covered,
            runtime_seconds=0.0,
            n_obs=result.n_obs,
        ))

    return results


def benchmark_did_adapter() -> list[BenchmarkResult]:
    """Benchmark DIDEventStudyAdapter on DID DGP."""
    from tests.fixtures.synthetic_dgp import make_did_dgp

    adapter = get_adapter("DID_EVENT_STUDY")
    results = []

    for seed in range(42, 52):
        df, truth = make_did_dgp(n_units=100, n_periods=20, att=2.0, seed=seed)
        req = EstimationRequest(
            df=df,
            outcome="Y",
            treatment="treatment",
            unit="unit",
            time="time",
            edge_id=f"dgp_did_seed{seed}",
        )
        result = adapter.estimate(req)
        bias = result.point - truth["att"]
        covered = result.ci_lower <= truth["att"] <= result.ci_upper

        results.append(BenchmarkResult(
            adapter_name="DID_EVENT_STUDY",
            dataset_id=f"did_dgp_seed{seed}",
            estimate=result.point,
            true_effect=truth["att"],
            bias=bias,
            se=result.se,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            covered=covered,
            runtime_seconds=0.0,
            n_obs=result.n_obs,
        ))

    return results


def run_all_dgp_benchmarks() -> dict[str, BenchmarkSummary]:
    """Run all DGP benchmarks and return summaries."""
    summaries = {}

    for name, bench_fn in [
        ("LOCAL_PROJECTIONS", benchmark_lp_adapter),
        ("IV_2SLS", benchmark_iv_adapter),
        ("DID_EVENT_STUDY", benchmark_did_adapter),
    ]:
        try:
            results = bench_fn()
            summaries[name] = summarize_benchmark(results)
        except Exception as e:
            logger.error(f"DGP benchmark failed for {name}: {e}")

    return summaries
