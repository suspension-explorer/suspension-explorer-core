"""Focused tests for solution-manifold tangent computation."""

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.constraints import Constraint, FixedAxisConstraint
from kinematics.core.enums import Axis, PointID, TargetValueMode
from kinematics.core.points.derived.manager import (
    DerivedPointsManager,
    DerivedPointsSpec,
)
from kinematics.core.primitives.geometry import Direction3, Point3, extract_array
from kinematics.core.sensitivity import (
    TangentField,
    combine_tangents,
    compute_state_tangents,
)
from kinematics.core.state import SuspensionState
from kinematics.core.sweep import solve_sweep
from kinematics.core.targeting import (
    PointTarget,
    PointTargetAxis,
    PointTargetVector,
    SweepConfig,
)

FD_STEP = 0.25


def _bump_target(value: float) -> PointTarget:
    return PointTarget(
        point_id=PointID.WHEEL_CENTER,
        direction=PointTargetAxis(axis=Axis.Z),
        value=value,
        mode=TargetValueMode.ABSOLUTE,
    )


def test_corner_tangent_matches_finite_difference(
    double_wishbone_geometry_file,
) -> None:
    corner = load_geometry(double_wishbone_geometry_file)
    initial = corner.initial_state()
    design_z = float(initial.positions[PointID.WHEEL_CENTER][Axis.Z])
    trackrod_inboard_y = float(initial.positions[PointID.TRACKROD_INBOARD][Axis.Y])
    target_z = design_z + 10.0
    rack_coordinate = next(
        coordinate
        for coordinate in corner.drive_coordinates()
        if coordinate.id == "rack"
    )
    rack_target = rack_coordinate.position_target(
        PointTargetAxis(axis=Axis.Y),
        trackrod_inboard_y,
        TargetValueMode.ABSOLUTE,
    )
    targets = [_bump_target(target_z), rack_target]

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
                    rack_target,
                    rack_target,
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
    assert solve_info.full_column_rank
    assert solve_info.nullity == 0
    assert solve_info.rate_consistent
    assert solve_info.response_for_target(0).unique
    assert solve_info.response_for_target(1).unique
    assert field.velocity(PointID.WHEEL_CENTER)[Axis.Z] == pytest.approx(1.0)


def test_full_rank_inconsistent_tangent_is_not_reported_as_unique() -> None:
    point = PointID.WHEEL_CENTER
    state = SuspensionState(
        positions={point: Point3([1.0, 2.0, 3.0])},
        free_points={point},
    )
    constraints: list[Constraint] = [
        FixedAxisConstraint(point, axis, float(state.positions[point][axis]))
        for axis in Axis
    ]
    target = PointTarget(
        point_id=point,
        direction=PointTargetAxis(axis=Axis.X),
        value=1.0,
        mode=TargetValueMode.ABSOLUTE,
    )

    _fields, info = compute_state_tangents(
        state,
        constraints,
        DerivedPointsManager(DerivedPointsSpec({}, {})),
        [target],
    )

    response = info.response_for_target(0)
    assert info.full_column_rank
    assert info.nullity == 0
    assert info.mobility == 0
    assert info.target_rank == 0
    assert not info.rate_consistent
    # The reduced solve never compromises permanent constraints to partially
    # satisfy an impossible target request.
    assert response.max_constraint_rate_residual == pytest.approx(0.0)
    assert response.selected_target_rate_residual == pytest.approx(1.0)
    assert response.max_other_target_rate_residual == 0.0
    assert response.max_rate_residual > response.consistency_tolerance
    assert not response.rate_consistent
    assert not response.unique


def test_consistent_but_underconstrained_tangent_is_not_unique() -> None:
    point = PointID.WHEEL_CENTER
    state = SuspensionState(
        positions={point: Point3([1.0, 2.0, 3.0])},
        free_points={point},
    )
    target = PointTarget(
        point_id=point,
        direction=PointTargetAxis(axis=Axis.X),
        value=1.0,
        mode=TargetValueMode.ABSOLUTE,
    )

    _fields, info = compute_state_tangents(
        state,
        [],
        DerivedPointsManager(DerivedPointsSpec({}, {})),
        [target],
    )

    response = info.response_for_target(0)
    assert info.rank == 1
    assert info.nullity == 2
    assert not info.full_column_rank
    assert info.rank_deficient
    assert response.rate_consistent
    assert not response.unique


def test_one_dof_mechanism_needs_no_hold_beyond_its_driver() -> None:
    point = PointID.WHEEL_CENTER
    state = SuspensionState(
        positions={point: Point3([1.0, 2.0, 3.0])},
        free_points={point},
    )
    constraints: list[Constraint] = [
        FixedAxisConstraint(point, Axis.Y, 2.0),
        FixedAxisConstraint(point, Axis.Z, 3.0),
    ]
    target = PointTarget(
        point_id=point,
        direction=PointTargetAxis(axis=Axis.X),
        value=1.0,
        mode=TargetValueMode.ABSOLUTE,
    )

    fields, info = compute_state_tangents(
        state,
        constraints,
        DerivedPointsManager(DerivedPointsSpec({}, {})),
        [target],
    )

    assert info.constraint_rank == 2
    assert info.mobility == 1
    assert info.target_rank == 1
    assert info.nullity == 0
    assert info.response_for_target(0).unique
    assert fields[0].velocity(point) == pytest.approx([1.0, 0.0, 0.0])


def test_ill_conditioning_cannot_hide_an_inconsistent_target_basis() -> None:
    point = PointID.WHEEL_CENTER
    state = SuspensionState(
        positions={point: Point3([1.0, 2.0, 3.0])},
        free_points={point},
    )
    constraints: list[Constraint] = [FixedAxisConstraint(point, Axis.Z, 3.0)]
    targets = [
        PointTarget(
            point_id=point,
            direction=PointTargetVector(Direction3([1.0, y_component, 0.0])),
            value=1.0,
            mode=TargetValueMode.ABSOLUTE,
        )
        for y_component in (0.0, 1e-12, 2e-12)
    ]

    _fields, info = compute_state_tangents(
        state,
        constraints,
        DerivedPointsManager(DerivedPointsSpec({}, {})),
        targets,
    )

    response = info.response_for_target(0)
    assert info.full_column_rank
    assert info.condition_number > 1e11
    assert response.max_rate_residual > 0.3
    assert response.max_rate_residual > response.consistency_tolerance
    assert not response.rate_consistent
    assert not response.unique


def test_target_rate_diagnostics_cover_selected_and_held_targets() -> None:
    point = PointID.WHEEL_CENTER
    state = SuspensionState(
        positions={point: Point3([1.0, 2.0, 3.0])},
        free_points={point},
    )
    targets = [
        PointTarget(
            point_id=point,
            direction=PointTargetAxis(axis=axis),
            value=float(state.positions[point][axis]),
            mode=TargetValueMode.ABSOLUTE,
        )
        for axis in Axis
    ]

    _fields, info = compute_state_tangents(
        state,
        [],
        DerivedPointsManager(DerivedPointsSpec({}, {})),
        targets,
    )

    for target_index in range(3):
        response = info.response_for_target(target_index)
        assert response.target_rate_residuals == pytest.approx((0.0, 0.0, 0.0))
        assert response.selected_target_rate_residual == pytest.approx(0.0)
        assert response.max_other_target_rate_residual == pytest.approx(0.0)
        assert response.unique

    with pytest.raises(KeyError, match="target index 3"):
        info.response_for_target(3)


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
