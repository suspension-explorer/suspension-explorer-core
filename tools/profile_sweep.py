#!/usr/bin/env python3
"""
Profile a full-axle solve/analyze sweep and print the hottest functions.

Runs the same rocker/ARB axle-articulation sweep the benchmarks use (two
coupled corners with opposed wheel travel) under cProfile, then prints the top 30
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

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.analysis import analyze_sweep
from kinematics.core.sweep import solve_sweep

DATA_DIR = Path(__file__).parent.parent / "tests" / "data"
GEOMETRY_FILE = DATA_DIR / "axle_geometry_rocker.yaml"
SWEEP_FILE = DATA_DIR / "axle_rocker_sweep.yaml"
TOP_N = 30


def run_sweep() -> None:
    """Load the axle geometry and sweep, then solve and analyze it."""
    suspension = load_geometry(GEOMETRY_FILE)
    sweep_config = load_sweep(SWEEP_FILE, suspension)

    # Profile the raw solve and, separately, analysis including its own solve.
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
