"""Benchmarks for the full-axle solve and analysis pipelines."""

from __future__ import annotations

from pathlib import Path

import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.analysis import analyze_sweep
from kinematics.core.sweep import solve_sweep

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def axle_suspension():
    """Build the rocker/ARB axle once for the module."""
    return load_geometry(DATA_DIR / "axle_geometry_rocker.yaml")


@pytest.fixture(scope="module")
def axle_sweep_config(axle_suspension):
    """Load the opposed-wheel-travel axle-articulation sweep."""
    return load_sweep(DATA_DIR / "axle_rocker_sweep.yaml", axle_suspension)


@pytest.mark.benchmark
def test_bench_solve_sweep(benchmark, axle_suspension, axle_sweep_config):
    """Benchmark the constraint solve across the full axle sweep."""
    states, _stats = benchmark(solve_sweep, axle_suspension, axle_sweep_config)
    assert states


@pytest.mark.benchmark
def test_bench_analyze_sweep(benchmark, axle_suspension, axle_sweep_config):
    """Benchmark the full solve and metric-assembly pipeline."""
    analysis = benchmark(analyze_sweep, axle_suspension, axle_sweep_config)
    assert analysis.frames
