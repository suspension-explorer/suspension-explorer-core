"""
Main orchestration functions for suspension kinematics.

This module provides high-level functions to coordinate the solving of suspension
geometries.
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, List

from kinematics.core.types import SweepConfig
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.sensitivity import TangentField, compute_state_tangents
from kinematics.solver import (
    SolverInfo,
    convert_targets_to_absolute,
    solve_suspension_sweep,
)
from kinematics.state import SuspensionState
from kinematics.suspensions.base import Suspension

if TYPE_CHECKING:
    from kinematics.metrics.main import MetricRow


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


def compute_sweep_tangents(
    suspension: Suspension,
    sweep_config: SweepConfig,
    states: List[SuspensionState],
) -> list[list[TangentField]]:
    """
    Compute solution-manifold tangents for every solved state of a sweep.

    This is the post-solve analysis companion to :func:`solve_sweep`: for
    each solved state it evaluates the analytical residual Jacobian and
    extracts d(position)/d(target) for every sweep target via the implicit
    function theorem (see kinematics.sensitivity). The result feeds the
    derivative metrics (motion ratios, camber gain, bump steer, ...).

    Args:
        suspension: The suspension the states were solved for.
        sweep_config: The sweep configuration that produced the states.
        states: The solved states, one per sweep step.

    Returns:
        One list of TangentField per state, each ordered like the sweep's
        target dimensions.
    """
    derived_manager = DerivedPointsManager(suspension.derived_spec())
    constraints = suspension.constraints()
    initial_state = suspension.initial_state()

    tangents_per_step: list[list[TangentField]] = []
    for step_index, state in enumerate(states):
        step_targets = convert_targets_to_absolute(
            [sweep[step_index] for sweep in sweep_config.target_sweeps],
            initial_state,
        )
        tangents_per_step.append(
            compute_state_tangents(state, constraints, derived_manager, step_targets)
        )
    return tangents_per_step


def compute_sweep_metrics(
    suspension: Suspension,
    sweep_config: SweepConfig,
    states: List[SuspensionState],
) -> list["MetricRow"]:
    """
    Compute the full metric row for every solved state of a sweep.

    This is the single high-level metrics entry point for sweep consumers
    (CLI, API adapters): it computes the solution-manifold tangents internally
    and feeds them to the per-state metric dispatch, so every consumer gets
    the complete column set -- per-state metrics plus the derivative metrics
    (motion ratios, camber gain, bump steer, modal roll/heave rates) -- without
    knowing anything about tangents or automatic differentiation.

    The tangent computation is best-effort: if it fails (e.g. a pathological
    configuration), the per-state metrics are still returned and only the
    derivative columns are omitted, consistently across all rows.

    Args:
        suspension: The suspension the states were solved for.
        sweep_config: The sweep configuration that produced the states.
        states: The solved states, one per sweep step.

    Returns:
        One ordered metric row per state. Empty rows when the suspension has
        no configuration (metrics need vehicle parameters).
    """
    if suspension.config is None:
        return [OrderedDict() for _ in states]

    tangents: list[list[TangentField]] | None
    try:
        tangents = compute_sweep_tangents(suspension, sweep_config, states)
    except Exception:  # noqa: BLE001 - derivative metrics degrade gracefully
        tangents = None

    return [
        suspension.compute_state_metrics(
            state, tangents[index] if tangents is not None else None
        )
        for index, state in enumerate(states)
    ]
