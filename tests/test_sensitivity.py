"""Focused tests for solution-manifold tangent computation."""

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.points.derived.manager import DerivedPointsManager
from kinematics.core.primitives.enums import Axis, PointID, TargetPositionMode
from kinematics.core.primitives.geometry import extract_array
from kinematics.core.sensitivity import (
    TangentField,
    combine_tangents,
    compute_state_tangents,
)
from kinematics.core.sweep import solve_sweep
from kinematics.core.targeting import PointTarget, PointTargetAxis, SweepConfig

FD_STEP = 0.25


def _bump_target(value: float) -> PointTarget:
    return PointTarget(
        point_id=PointID.WHEEL_CENTER,
        direction=PointTargetAxis(axis=Axis.Z),
        value=value,
        mode=TargetPositionMode.ABSOLUTE,
    )


def _trackrod_inboard_target(value: float) -> PointTarget:
    return PointTarget(
        point_id=PointID.TRACKROD_INBOARD,
        direction=PointTargetAxis(axis=Axis.Y),
        value=value,
        mode=TargetPositionMode.ABSOLUTE,
    )


def test_corner_tangent_matches_finite_difference(
    double_wishbone_geometry_file,
) -> None:
    corner = load_geometry(double_wishbone_geometry_file)
    initial = corner.initial_state()
    design_z = float(initial.positions[PointID.WHEEL_CENTER][Axis.Z])
    trackrod_inboard_y = float(initial.positions[PointID.TRACKROD_INBOARD][Axis.Y])
    target_z = design_z + 10.0
    targets = [_bump_target(target_z), _trackrod_inboard_target(trackrod_inboard_y)]

    state = solve_sweep(corner, SweepConfig([[targets[0]], [targets[1]]]))[0][0]
    fields, solve_info = compute_state_tangents(
        state,
        corner.constraints(),
        DerivedPointsManager(corner.derived_spec()),
        targets,
    )

    states = solve_sweep(
        corner,
        SweepConfig(
            [
                [_bump_target(target_z - FD_STEP), _bump_target(target_z + FD_STEP)],
                [
                    _trackrod_inboard_target(trackrod_inboard_y),
                    _trackrod_inboard_target(trackrod_inboard_y),
                ],
            ]
        ),
    )[0]
    field = fields[0]
    for point_id in state.positions:
        finite_difference = (
            extract_array(states[1].positions[point_id])
            - extract_array(states[0].positions[point_id])
        ) / (2.0 * FD_STEP)
        np.testing.assert_allclose(
            field.velocity(point_id),
            finite_difference,
            rtol=1e-3,
            atol=1e-5,
        )

    assert not solve_info.rank_deficient
    assert solve_info.rank == solve_info.n_variables
    assert solve_info.smallest_singular_value > 0.0
    assert np.isfinite(solve_info.condition_number)
    assert field.velocity(PointID.WHEEL_CENTER)[Axis.Z] == pytest.approx(1.0)


def test_combine_tangents_is_linear() -> None:
    target = _bump_target(0.0)
    field_a = TangentField(
        target_index=0,
        target=target,
        velocities={PointID.WHEEL_CENTER: np.array([1.0, 0.0, 0.0])},
    )
    field_b = TangentField(
        target_index=1,
        target=target,
        velocities={PointID.WHEEL_CENTER: np.array([0.0, 2.0, 0.0])},
    )

    combined = combine_tangents([field_a, field_b], [2.0, -1.0])

    np.testing.assert_allclose(combined[PointID.WHEEL_CENTER], [2.0, -2.0, 0.0])
