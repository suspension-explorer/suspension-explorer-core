"""File-to-file sweep command service."""

from dataclasses import dataclass
from pathlib import Path

from kinematics.cli.io.results_writer import SolutionFrame, create_writer_for_path
from kinematics.cli.io.yaml import load_geometry, load_sweep
from kinematics.core import SweepAnalysis, analyze_solved_sweep, solve_sweep
from kinematics.core.export import (
    flat_specs_for_suspension,
    flatten_metric_rows,
    point_key_name,
)
from kinematics.core.types import Suspension, SuspensionState


@dataclass(frozen=True)
class SweepRun:
    """Solved objects retained for terminal reporting and optional rendering."""

    suspension: Suspension
    states: list[SuspensionState]
    analysis: SweepAnalysis


def run_sweep_files(
    geometry_path: Path,
    sweep_path: Path,
    output_path: Path,
) -> SweepRun:
    """Load, solve, analyze, and write one sweep without terminal behavior."""
    suspension = load_geometry(geometry_path)
    sweep_config = load_sweep(sweep_path, suspension)
    states, solver_stats = solve_sweep(suspension, sweep_config)
    analysis = analyze_solved_sweep(
        suspension,
        sweep_config,
        states,
        solver_stats,
    )

    writer = create_writer_for_path(
        output_path,
        geometry_path=str(geometry_path),
        sweep_path=str(sweep_path),
    )
    output_points = suspension.output_points()
    metric_specs = flat_specs_for_suspension(suspension)
    for analyzed_frame in analysis.frames:
        positions = {
            point_key_name(point): position
            for point in output_points
            if (position := analyzed_frame.positions.get(point_key_name(point)))
            is not None
        }
        writer.add_frame(
            analyzed_frame.index,
            SolutionFrame(
                positions=positions,
                solver_info=analyzed_frame.solver,
                metrics=flatten_metric_rows(
                    analyzed_frame.metrics,
                    analyzed_frame.corner_metrics,
                ),
                metric_specs=metric_specs,
            ),
        )
    writer.write()

    return SweepRun(
        suspension=suspension,
        states=states,
        analysis=analysis,
    )
