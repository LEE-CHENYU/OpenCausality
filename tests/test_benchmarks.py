"""
Benchmark Framework Tests.

Tests the benchmark harness, ACIC loader, and DGP benchmark suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.acic_loader import ACICDataset, generate_acic_practice_data
from benchmarks.harness import (
    BenchmarkResult,
    BenchmarkSummary,
    run_adapter_on_dataset,
    summarize_benchmark,
)
from shared.engine.adapters.base import EstimationRequest, EstimationResult, EstimatorAdapter


class MockAdapter(EstimatorAdapter):
    """Mock adapter that returns a fixed estimate for testing."""

    def __init__(self, point: float = 1.0, se: float = 0.1):
        self._point = point
        self._se = se

    def supported_designs(self) -> list[str]:
        return ["MOCK"]

    def estimate(self, req: EstimationRequest) -> EstimationResult:
        return EstimationResult(
            point=self._point,
            se=self._se,
            ci_lower=self._point - 1.96 * self._se,
            ci_upper=self._point + 1.96 * self._se,
            pvalue=0.01,
            n_obs=len(req.df),
            method_name="MOCK",
            library="test",
            library_version="1.0",
        )


class TestACICLoader:
    """Test ACIC data generation."""

    def test_generate_practice_data(self):
        datasets = generate_acic_practice_data(n_datasets=3, n_obs=100)
        assert len(datasets) == 3
        for ds in datasets:
            assert isinstance(ds, ACICDataset)
            assert len(ds.df) == 100
            assert "Y" in ds.df.columns
            assert "treatment" in ds.df.columns
            assert "ate" in ds.true_effects
            assert "att" in ds.true_effects

    def test_datasets_have_varying_effects(self):
        datasets = generate_acic_practice_data(n_datasets=5, n_obs=200)
        ates = [ds.true_effects["ate"] for ds in datasets]
        # Effects should vary across datasets
        assert len(set(round(a, 1) for a in ates)) > 1

    def test_treatment_is_binary(self):
        datasets = generate_acic_practice_data(n_datasets=1, n_obs=500)
        ds = datasets[0]
        assert set(ds.df["treatment"].unique()) <= {0.0, 1.0}


class TestBenchmarkHarness:
    """Test the benchmark harness with mock adapter."""

    def test_run_adapter_on_dataset(self):
        adapter = MockAdapter(point=2.5, se=0.2)
        ds = ACICDataset(
            dataset_id="test_001",
            df=pd.DataFrame({
                "Y": np.random.randn(100),
                "treatment": np.random.binomial(1, 0.5, 100).astype(float),
                "X1": np.random.randn(100),
            }),
            treatment_col="treatment",
            outcome_col="Y",
            true_effects={"att": 2.5, "ate": 2.0},
            covariates=["X1"],
        )
        result = run_adapter_on_dataset(adapter, ds, target_effect="att")
        assert isinstance(result, BenchmarkResult)
        assert result.adapter_name == "MOCK"
        assert result.dataset_id == "test_001"
        assert result.estimate == 2.5
        assert result.true_effect == 2.5
        assert abs(result.bias) < 1e-10  # Perfect estimate
        assert result.covered  # CI should cover true effect

    def test_summarize_benchmark(self):
        results = [
            BenchmarkResult(
                adapter_name="MOCK",
                dataset_id=f"ds_{i}",
                estimate=2.5 + 0.1 * i,
                true_effect=2.5,
                bias=0.1 * i,
                se=0.2,
                ci_lower=2.3 + 0.1 * i,
                ci_upper=2.7 + 0.1 * i,
                covered=i < 3,  # First 3 covered
                runtime_seconds=0.01,
                n_obs=100,
            )
            for i in range(5)
        ]

        summary = summarize_benchmark(results)
        assert isinstance(summary, BenchmarkSummary)
        assert summary.adapter_name == "MOCK"
        assert summary.n_datasets == 5
        assert summary.coverage_90 == 0.6  # 3/5 covered

    def test_bias_computation(self):
        results = [
            BenchmarkResult(
                adapter_name="MOCK",
                dataset_id="ds_0",
                estimate=1.0,
                true_effect=0.5,
                bias=0.5,
                se=0.1,
                ci_lower=0.8,
                ci_upper=1.2,
                covered=False,
                runtime_seconds=0.01,
                n_obs=100,
            ),
        ]
        summary = summarize_benchmark(results)
        assert abs(summary.mean_bias - 0.5) < 1e-10
        assert abs(summary.rmse - 0.5) < 1e-10

    def test_empty_results(self):
        summary = summarize_benchmark([])
        assert summary.n_datasets == 0
        assert np.isnan(summary.rmse)


class TestDGPBenchmarks:
    """Test that DGP benchmarks run without errors."""

    def test_lp_dgp_benchmark(self):
        from benchmarks.dgp_benchmarks import benchmark_lp_adapter

        results = benchmark_lp_adapter()
        assert len(results) == 10  # 10 seeds
        for r in results:
            assert r.adapter_name == "LOCAL_PROJECTIONS"
            assert abs(r.bias) < 0.3  # Should be close to truth

    def test_iv_dgp_benchmark(self):
        from benchmarks.dgp_benchmarks import benchmark_iv_adapter

        results = benchmark_iv_adapter()
        assert len(results) == 10
        for r in results:
            assert r.adapter_name == "IV_2SLS"

    def test_did_dgp_benchmark(self):
        from benchmarks.dgp_benchmarks import benchmark_did_adapter

        results = benchmark_did_adapter()
        assert len(results) == 10
        for r in results:
            assert r.adapter_name == "DID_EVENT_STUDY"
