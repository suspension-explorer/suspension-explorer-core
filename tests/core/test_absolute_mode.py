import numpy as np

from kinematics.core.coordinates import CoordinateAxis, PointCoordinate
from kinematics.core.enums import Axis, PointID, Scope, TargetValueMode
from kinematics.core.points.derived.manager import (
    DerivedPointsManager,
    DerivedPointsSpec,
)
from kinematics.core.solver import convert_targets_to_absolute, solve_suspension_sweep
from kinematics.core.state import SuspensionState
from kinematics.core.targeting import SweepConfig


def test_resolve_targets_to_absolute():
    from kinematics.core.primitives.geometry import Point3

    initial_positions = {
        PointID.WHEEL_CENTER: Point3([0.0, 0.0, 150.0]),
    }
    initial_state = SuspensionState(positions=initial_positions, free_points=set())

    # Relative conversion -> absolute
    relative_target = PointCoordinate(
        PointID.WHEEL_CENTER, CoordinateAxis(Axis.Z), scope=Scope.CORNER
    ).target(50.0, TargetValueMode.RELATIVE)

    resolved = convert_targets_to_absolute([relative_target], initial_state)

    assert resolved[0].mode == TargetValueMode.ABSOLUTE
    assert resolved[0].value == 200.0  # 150 + 50

    # Absolute passthrough
    absolute_target = PointCoordinate(
        PointID.WHEEL_CENTER, CoordinateAxis(Axis.Z), scope=Scope.CORNER
    ).target(400.0, TargetValueMode.ABSOLUTE)

    resolved2 = convert_targets_to_absolute([absolute_target], initial_state)

    assert resolved2[0].mode == TargetValueMode.ABSOLUTE
    assert resolved2[0].value == 400.0


def test_default_relative_mode():
    target = PointCoordinate(
        PointID.WHEEL_CENTER, CoordinateAxis(Axis.Z), scope=Scope.CORNER
    ).target(50)
    assert target.mode == TargetValueMode.RELATIVE


def test_absolute_mode_solve():
    from kinematics.core.primitives.geometry import Point3

    # No derived points
    derived_manager = DerivedPointsManager(DerivedPointsSpec({}, {}))

    positions = {PointID.LOWER_WISHBONE_OUTBOARD: Point3([0.0, 0.0, 0.0])}
    free = {PointID.LOWER_WISHBONE_OUTBOARD}
    initial_state = SuspensionState(positions=positions, free_points=free)

    # Fully determined with 3 absolute targets (X, Y, Z) applied simultaneously.
    x_sweep = [
        PointCoordinate(
            PointID.LOWER_WISHBONE_OUTBOARD, CoordinateAxis(Axis.X), scope=Scope.CORNER
        ).target(10.0, TargetValueMode.ABSOLUTE)
    ]
    y_sweep = [
        PointCoordinate(
            PointID.LOWER_WISHBONE_OUTBOARD, CoordinateAxis(Axis.Y), scope=Scope.CORNER
        ).target(-5.0, TargetValueMode.ABSOLUTE)
    ]
    z_sweep = [
        PointCoordinate(
            PointID.LOWER_WISHBONE_OUTBOARD, CoordinateAxis(Axis.Z), scope=Scope.CORNER
        ).target(100.0, TargetValueMode.ABSOLUTE)
    ]

    states, solver_stats = solve_suspension_sweep(
        initial_state=initial_state,
        constraints=[],
        sweep_config=SweepConfig([x_sweep, y_sweep, z_sweep]),
        derived_manager=derived_manager,
        finalize_state=lambda positions: None,
    )

    assert len(states) == 1
    assert len(solver_stats) == 1
    assert np.allclose(
        states[0].positions[PointID.LOWER_WISHBONE_OUTBOARD].data,
        np.array([10.0, -5.0, 100.0]),
    )
