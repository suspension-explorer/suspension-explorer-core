"""
File-to-file sweep command service.
"""

from dataclasses import dataclass
from pathlib import Path

from kinematics.cli.io.loaders import load_geometry
from kinematics.cli.io.results_writer import SolutionFrame, create_writer_for_path
from kinematics.cli.io.sweep_loader import load_sweep
from kinematics.core.export import flatten_positions
from kinematics.core.metrics.main import AxleMetricRows, MetricRow, flatten_metric_rows
from kinematics.core.metrics.registry import flat_specs_for_suspension
from kinematics.core.suspensions.base import Suspension
from kinematics.core.sweep import EvaluatedSweep, solve_evaluated_sweep


@dataclass(frozen=True)
class SweepRun:
    """
    Solved objects retained for terminal reporting and optional rendering.
    """

    suspension: Suspension
    evaluated: EvaluatedSweep


def flatten_metrics_for_export(
    row: MetricRow | AxleMetricRows,
) -> dict[str, float | None]:
    """
    Flatten corner or axle metric rows at the file-output boundary.
    """
    if isinstance(row, AxleMetricRows):
        return flatten_metric_rows(row.axle, row.corners)
    return dict(row)


def run_sweep_files(
    geometry_path: Path,
    sweep_path: Path,
    output_path: Path,
) -> SweepRun:
    """
    Load, solve, analyze, and write one sweep without terminal behavior.
    """
    suspension = load_geometry(geometry_path)
    sweep_config = load_sweep(sweep_path, suspension)
    evaluated = solve_evaluated_sweep(suspension, sweep_config)

    writer = create_writer_for_path(
        output_path,
        geometry_path=str(geometry_path),
        sweep_path=str(sweep_path),
    )
    output_points = suspension.output_points()
    metric_specs = flat_specs_for_suspension(suspension)
    for index, (state, solver_info, metric_row) in enumerate(
        zip(
            evaluated.states,
            evaluated.solver_stats,
            evaluated.metrics.rows,
        )
    ):
        writer.add_frame(
            index,
            SolutionFrame(
                positions=flatten_positions(state.positions, output_points),
                solver_info=solver_info,
                metrics=flatten_metrics_for_export(metric_row),
                metric_specs=metric_specs,
            ),
        )
    writer.write()

    return SweepRun(
        suspension=suspension,
        evaluated=evaluated,
    )
