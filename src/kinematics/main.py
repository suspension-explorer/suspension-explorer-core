"""
Main orchestration functions for suspension kinematics.

This module provides high-level functions to coordinate the solving of suspension
geometries.
"""

from typing import List

from kinematics.core.types import PointTarget, PointTargetDirection, SweepConfig
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.solver import SolverInfo, solve_suspension_sweep
from kinematics.state import SuspensionState
from kinematics.suspensions.base import Suspension
from kinematics.targets import resolve_target


def _direction_key(direction: PointTargetDirection) -> tuple[float, float, float]:
    """
    Build a hashable key identifying a target direction by its unit vector.

    Rounds the resolved unit vector so directions that are equal up to
    floating-point noise compare equal when matching swept vs. held DOFs.
    """
    unit = resolve_target(direction).data
    return (
        round(float(unit[0]), 9),
        round(float(unit[1]), 9),
        round(float(unit[2]), 9),
    )


def apply_default_holds(
    suspension: Suspension,
    sweep_config: SweepConfig,
) -> SweepConfig:
    """
    Inject "hold at initial" targets for control DOFs the sweep does not drive.

    For each target from `suspension.default_hold_targets()` whose
    (point, direction) is not already driven by a sweep dimension, append a
    constant dimension that pins that DOF at every step. This keeps the system
    determinate so an unspecified control DOF (e.g. the steering rack on a pure
    bump sweep) stays put instead of drifting. Sweeps that already drive the DOF
    are returned unchanged, so explicit steering sweeps are unaffected.

    Args:
        suspension: The suspension whose default holds are considered.
        sweep_config: The user-provided sweep configuration.

    Returns:
        The original config if nothing needs holding, otherwise a new config
        with the extra constant dimensions appended.
    """
    holds = suspension.default_hold_targets()
    n_steps = sweep_config.n_steps
    if not holds or n_steps == 0:
        return sweep_config

    # DOFs the sweep already drives, identified by (point, direction). Each
    # dimension shares one point/direction across its steps, so the first step
    # is representative.
    driven = {
        (dimension[0].point_id, _direction_key(dimension[0].direction))
        for dimension in sweep_config.target_sweeps
        if dimension
    }

    # A held DOF is constant across the sweep: the same relative-zero target is
    # repeated for every step so it expands to the initial coordinate throughout.
    extra_dimensions: list[list[PointTarget]] = []
    for hold in holds:
        if (hold.point_id, _direction_key(hold.direction)) in driven:
            continue
        extra_dimensions.append([hold] * n_steps)

    if not extra_dimensions:
        return sweep_config

    return SweepConfig(sweep_config.target_sweeps + extra_dimensions)


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
    # Pin any control DOF (e.g. the steering rack) the sweep does not drive, so
    # the solve stays determinate instead of letting that DOF wander freely.
    sweep_config = apply_default_holds(suspension, sweep_config)

    derived_spec = suspension.derived_spec()
    derived_manager = DerivedPointsManager(derived_spec)

    kinematic_states, solver_stats = solve_suspension_sweep(
        initial_state=suspension.initial_state(),
        constraints=suspension.constraints(),
        sweep_config=sweep_config,
        derived_manager=derived_manager,
    )

    return kinematic_states, solver_stats
