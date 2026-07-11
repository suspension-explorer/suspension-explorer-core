"""
Main orchestration functions for suspension kinematics.

This module provides high-level functions to coordinate the solving of suspension
geometries.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import List

from kinematics.core.types import SweepConfig
from kinematics.metrics.main import AxleMetricRows, MetricRow
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.sensitivity import (
    TangentField,
    TangentSolveInfo,
    compute_state_tangents,
)
from kinematics.solver import (
    SolverInfo,
    convert_targets_to_absolute,
    solve_suspension_sweep,
)
from kinematics.state import SuspensionState
from kinematics.suspensions.base import Suspension


def solve_sweep(
    suspension: Suspension,
    sweep_config: SweepConfig,
) -> tuple[List[SuspensionState], List[SolverInfo]]:
    """
    Orchestrates the solving of suspension kinematics for a parametric sweep.

    This function coordinates the complete process of solving suspension kinematics
    by setting up derived point calculations and running the solver across target
    configurations.

    Args:
        suspension: The Suspension instance containing geometry and behavior.
        sweep_config: Configuration for the parametric sweep.

    Returns:
        Tuple containing the list of solved suspension states and corresponding
        solver information for each step in the sweep.
    """
    derived_spec = suspension.derived_spec()
    derived_manager = DerivedPointsManager(derived_spec)

    kinematic_states, solver_stats = solve_suspension_sweep(
        initial_state=suspension.initial_state(),
        constraints=suspension.constraints(),
        sweep_config=sweep_config,
        derived_manager=derived_manager,
    )

    return kinematic_states, solver_stats


@dataclass(frozen=True)
class SweepTangents:
    """Per-state tangent fields and their numerical solve health."""

    per_step: list[list[TangentField]]
    solve_infos: list[TangentSolveInfo]


@dataclass(frozen=True)
class SweepMetricsResult:
    """Metric rows plus visible derivative-computation status."""

    rows: list[MetricRow | AxleMetricRows]
    derivative_error: str | None = None
    tangent_solve_infos: list[TangentSolveInfo] | None = None


def compute_sweep_tangents(
    suspension: Suspension,
    sweep_config: SweepConfig,
    states: List[SuspensionState],
) -> SweepTangents:
    """Compute solution-manifold tangents for every solved sweep state."""
    derived_manager = DerivedPointsManager(suspension.derived_spec())
    constraints = suspension.constraints()
    initial_state = suspension.initial_state()
    tangents_per_step: list[list[TangentField]] = []
    solve_infos: list[TangentSolveInfo] = []

    for step_index, state in enumerate(states):
        step_targets = convert_targets_to_absolute(
            [target_sweep[step_index] for target_sweep in sweep_config.target_sweeps],
            initial_state,
        )
        fields, solve_info = compute_state_tangents(
            state,
            constraints,
            derived_manager,
            step_targets,
        )
        tangents_per_step.append(fields)
        solve_infos.append(solve_info)

    return SweepTangents(per_step=tangents_per_step, solve_infos=solve_infos)


def compute_sweep_metrics(
    suspension: Suspension,
    sweep_config: SweepConfig,
    states: List[SuspensionState],
) -> SweepMetricsResult:
    """Compute all sweep metrics, reporting derivative failures explicitly."""
    if suspension.config is None:
        return SweepMetricsResult(rows=[OrderedDict() for _ in states])

    derivative_error: str | None = None
    try:
        tangents = compute_sweep_tangents(suspension, sweep_config, states)
    except Exception as error:  # noqa: BLE001 - metrics degrade without derivatives
        tangents = None
        derivative_error = f"{type(error).__name__}: {error}"

    rows = [
        suspension.compute_state_metrics(
            state,
            tangents.per_step[index] if tangents is not None else None,
        )
        for index, state in enumerate(states)
    ]
    return SweepMetricsResult(
        rows=rows,
        derivative_error=derivative_error,
        tangent_solve_infos=tangents.solve_infos if tangents is not None else None,
    )
