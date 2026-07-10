#!/usr/bin/env python3
"""
Profile a full-axle solve/analyze sweep and print the hottest functions.

Runs the same rocker/ARB axle sweep the benchmarks use (two coupled corners
driven by combined heave plus steer) under cProfile, then prints the top 30
functions by cumulative time. Use this to find where the solve spends its
time before reaching for any optimisation.

Run
---
    uv run python tools/profile_sweep.py
"""

from __future__ import annotations

import cProfile
import pstats
from pathlib import Path

from kinematics import analyze_sweep, solve_sweep
from kinematics.io import load_geometry, load_sweep

# Reuse the test fixtures so the profile tracks the benchmarked workload.
DATA_DIR = Path(__file__).parent.parent / "tests" / "data"
GEOMETRY_FILE = DATA_DIR / "axle_geometry_rocker.yaml"
SWEEP_FILE = DATA_DIR / "axle_rocker_sweep.yaml"

# Number of functions to report, ranked by cumulative time.
TOP_N = 30


def run_sweep() -> None:
    """
    Load the axle geometry and sweep, then solve and analyze it.

    The suspension is passed to the sweep loader so each target's
    ``(point, side)`` resolves to a concrete per-corner point key.
    """
    suspension = load_geometry(GEOMETRY_FILE)
    sweep_config = load_sweep(SWEEP_FILE, suspension)

    # solve_sweep is the raw constraint solve; analyze_sweep repeats the solve
    # and adds metric assembly. Profiling both mirrors the two benchmarks.
    solve_sweep(suspension, sweep_config)
    analyze_sweep(suspension, sweep_config)


def main() -> None:
    """Profile a full sweep and print the top functions by cumulative time."""
    profiler = cProfile.Profile()
    profiler.enable()
    run_sweep()
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)

    print(f"Top {TOP_N} functions by cumulative time")
    print(f"Geometry: {GEOMETRY_FILE}")
    print(f"Sweep:    {SWEEP_FILE}")
    print("-" * 88)
    stats.print_stats(TOP_N)


if __name__ == "__main__":
    main()
