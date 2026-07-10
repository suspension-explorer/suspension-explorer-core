"""
Performance benchmarks for the full-axle solve/analyze pipeline.

These are deselected from the default suite via the ``benchmark`` marker (see
``addopts`` in pyproject.toml). Run them explicitly with ``just bench`` or::

    uv run pytest tests/benchmarks -m benchmark --benchmark-only

They establish a baseline for the two hot entry points on a rocker/ARB axle
(two coupled corners): ``solve_sweep`` (the constraint solve across every step)
and ``analyze_sweep`` (solve plus metric assembly for the front end). Both are
driven through the curated public API so the numbers track what consumers see.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kinematics import analyze_sweep, solve_sweep
from kinematics.io import load_geometry, load_sweep

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def axle_suspension():
    """Build the rocker/ARB axle (two coupled corners) once for the module."""
    return load_geometry(DATA_DIR / "axle_geometry_rocker.yaml")


@pytest.fixture(scope="module")
def axle_sweep_config(axle_suspension):
    """
    Expand the combined heave-plus-steer sweep for the axle.

    The suspension is required so each target's ``(point, side)`` resolves to a
    concrete per-corner point key.
    """
    return load_sweep(DATA_DIR / "axle_rocker_sweep.yaml", axle_suspension)


@pytest.mark.benchmark
def test_bench_solve_sweep(benchmark, axle_suspension, axle_sweep_config):
    """Benchmark the constraint solve across the full axle sweep."""
    states, _stats = benchmark(solve_sweep, axle_suspension, axle_sweep_config)

    # Sanity check: every sweep step produced a solved state, so the timing
    # reflects a complete solve rather than an early bail-out.
    assert len(states) > 0


@pytest.mark.benchmark
def test_bench_analyze_sweep(benchmark, axle_suspension, axle_sweep_config):
    """Benchmark the full solve-plus-metric-assembly analysis pipeline."""
    analysis = benchmark(analyze_sweep, axle_suspension, axle_sweep_config)

    # Sanity check: analysis carries one frame per solved sweep step.
    assert len(analysis.frames) > 0
