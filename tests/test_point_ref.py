import numpy as np
import pytest

from kinematics.constraints import (
    AngleConstraint,
    CoplanarPointsConstraint,
    DistanceConstraint,
    EqualDistanceConstraint,
    FixedAxisConstraint,
    PointOnLineConstraint,
    PointOnPlaneConstraint,
    SphericalJointConstraint,
    ThreePointAngleConstraint,
    VectorsParallelConstraint,
    VectorsPerpendicularConstraint,
)
from kinematics.core.enums import Axis, PointID
from kinematics.core.geometry import Direction3, Point3
from kinematics.core.point_ref import PointKey, PointRef, Side, point_key_name
from kinematics.core.types import PointTarget, PointTargetAxis, SweepConfig
from kinematics.points.derived.manager import DerivedPointsManager, DerivedPointsSpec
from kinematics.solver import ResidualComputer, solve_least_squares_problem
from kinematics.state import SuspensionState

# ----------------------------------------------------------------------------
# Side / PointRef basics
# ----------------------------------------------------------------------------


def test_side_values():
    assert int(Side.LEFT) == 0
    assert int(Side.RIGHT) == 1
    assert int(Side.CENTER) == 2
    assert Side.LEFT.lateral_sign == 1.0
    assert Side.RIGHT.lateral_sign == -1.0
    with pytest.raises(ValueError, match="CENTER does not have a lateral sign"):
        _ = Side.CENTER.lateral_sign


def test_point_ref_is_tuple():
    ref = PointRef(Side.LEFT, PointID.LOWER_WISHBONE_OUTBOARD)
    assert ref.side == Side.LEFT
    assert ref.point == PointID.LOWER_WISHBONE_OUTBOARD
    assert tuple(ref) == (Side.LEFT, PointID.LOWER_WISHBONE_OUTBOARD)


def test_point_ref_name_formatting():
    ref = PointRef(Side.LEFT, PointID.LOWER_WISHBONE_OUTBOARD)
    assert ref.name == "LEFT_LOWER_WISHBONE_OUTBOARD"

    ref2 = PointRef(Side.RIGHT, PointID.TRACKROD_INBOARD)
    assert ref2.name == "RIGHT_TRACKROD_INBOARD"

    ref3 = PointRef(Side.CENTER, PointID.AXLE_MIDPOINT)
    assert ref3.name == "CENTER_AXLE_MIDPOINT"


def test_public_point_names_are_lowercase_snake_case():
    assert point_key_name(PointID.AXLE_OUTBOARD) == "axle_outboard"
    assert (
        point_key_name(PointRef(Side.LEFT, PointID.AXLE_OUTBOARD))
        == "left_axle_outboard"
    )


def test_point_ref_equality_and_hashing_behave_as_tuples():
    a = PointRef(Side.LEFT, PointID.LOWER_WISHBONE_OUTBOARD)
    b = PointRef(Side.LEFT, PointID.LOWER_WISHBONE_OUTBOARD)
    c = PointRef(Side.RIGHT, PointID.LOWER_WISHBONE_OUTBOARD)

    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert hash(a) == hash((Side.LEFT, PointID.LOWER_WISHBONE_OUTBOARD))

    # Usable as dict keys / set members like plain tuples.
    d = {a: 1}
    d[b] = 2
    assert d[a] == 2
    assert len({a, b, c}) == 2


def test_point_ref_sorting_is_deterministic_and_grouped_by_side_then_point():
    refs = [
        PointRef(Side.RIGHT, PointID.UPPER_WISHBONE_OUTBOARD),
        PointRef(Side.LEFT, PointID.UPPER_WISHBONE_OUTBOARD),
        PointRef(Side.RIGHT, PointID.LOWER_WISHBONE_OUTBOARD),
        PointRef(Side.LEFT, PointID.LOWER_WISHBONE_OUTBOARD),
        PointRef(Side.CENTER, PointID.TRACKROD_INBOARD),
    ]
    expected = [
        PointRef(Side.LEFT, PointID.LOWER_WISHBONE_OUTBOARD),
        PointRef(Side.LEFT, PointID.UPPER_WISHBONE_OUTBOARD),
        PointRef(Side.RIGHT, PointID.LOWER_WISHBONE_OUTBOARD),
        PointRef(Side.RIGHT, PointID.UPPER_WISHBONE_OUTBOARD),
        PointRef(Side.CENTER, PointID.TRACKROD_INBOARD),
    ]
    assert sorted(refs) == expected
    # Deterministic across repeated sorts.
    assert sorted(refs) == sorted(list(reversed(refs)))


# ----------------------------------------------------------------------------
# Constraint.remap round-trips
# ----------------------------------------------------------------------------


def _left(pid: PointKey) -> PointKey:
    """Map a PointID into the LEFT side namespace."""
    assert isinstance(pid, PointID)
    return PointRef(Side.LEFT, pid)


def _positions_for(points, keyfn=lambda p: p):
    """Build a positions dict with a distinct coordinate per point."""
    positions = {}
    for i, pid in enumerate(points):
        positions[keyfn(pid)] = Point3([float(i) + 1.0, float(i) * 0.5, -float(i)])
    return positions


# Each entry: (constructor callable, point-attr names, non-point-attr names).
def _build_constraints():
    p = list(PointID)[1:]  # skip NOT_ASSIGNED (index 0)
    line_point = Point3([1.0, 2.0, 3.0])
    line_dir = Direction3([1.0, 0.0, 0.0])
    plane_point = Point3([0.0, 0.0, 1.0])
    plane_normal = Direction3([0.0, 0.0, 1.0])

    return [
        (
            DistanceConstraint(p[0], p[1], 2.5),
            ("p1", "p2"),
            {"target_distance": 2.5},
        ),
        (
            SphericalJointConstraint(p[0], p[1]),
            ("p1", "p2"),
            {},
        ),
        (
            AngleConstraint(p[0], p[1], p[2], p[3], 0.75),
            ("v1_start", "v1_end", "v2_start", "v2_end"),
            {"target_angle": 0.75},
        ),
        (
            ThreePointAngleConstraint(p[0], p[1], p[2], 1.1),
            ("p1", "p2", "p3"),
            {"target_angle": 1.1},
        ),
        (
            VectorsParallelConstraint(p[0], p[1], p[2], p[3]),
            ("v1_start", "v1_end", "v2_start", "v2_end"),
            {},
        ),
        (
            VectorsPerpendicularConstraint(p[0], p[1], p[2], p[3]),
            ("v1_start", "v1_end", "v2_start", "v2_end"),
            {},
        ),
        (
            EqualDistanceConstraint(p[0], p[1], p[2], p[3]),
            ("p1", "p2", "p3", "p4"),
            {},
        ),
        (
            FixedAxisConstraint(p[0], Axis.Y, 4.0),
            ("point_id",),
            {"axis": Axis.Y, "value": 4.0},
        ),
        (
            PointOnLineConstraint(p[0], line_point, line_dir),
            ("point_id",),
            {"line_point": line_point, "line_direction": line_dir},
        ),
        (
            PointOnPlaneConstraint(p[0], plane_point, plane_normal),
            ("point_id",),
            {"plane_point": plane_point, "plane_normal": plane_normal},
        ),
        (
            CoplanarPointsConstraint(p[0], p[1], p[2], p[3]),
            ("p1", "p2", "p3", "p4"),
            {},
        ),
    ]


@pytest.mark.parametrize(
    "constraint, point_attrs, nonpoint_attrs", _build_constraints()
)
def test_remap_round_trip(constraint, point_attrs, nonpoint_attrs):
    original_point_values = {a: getattr(constraint, a) for a in point_attrs}
    original_involved = set(constraint.involved_points)

    remapped = constraint.remap(_left)

    # Same type, distinct instance.
    assert type(remapped) is type(constraint)
    assert remapped is not constraint

    # Point attributes are mapped into the LEFT namespace.
    for attr in point_attrs:
        expected = PointRef(Side.LEFT, original_point_values[attr])
        assert getattr(remapped, attr) == expected

    # involved_points is now the side-qualified set.
    assert remapped.involved_points == {
        PointRef(Side.LEFT, pid) for pid in original_involved
    }

    # Non-point parameters preserved (identity for object members).
    for attr, value in nonpoint_attrs.items():
        assert getattr(remapped, attr) == value

    # Original is unmodified.
    for attr, value in original_point_values.items():
        assert getattr(constraint, attr) == value
    assert set(constraint.involved_points) == original_involved


def test_remap_shares_line_point_with_original():
    # Documented behavior: remapped copy shares the (immutable) line_point.
    c = PointOnLineConstraint(
        PointID.TRACKROD_INBOARD,
        Point3([1.0, 2.0, 3.0]),
        Direction3([0.0, 1.0, 0.0]),
    )
    remapped = c.remap(_left)
    assert isinstance(remapped, PointOnLineConstraint)
    assert remapped.line_point is c.line_point
    assert remapped.line_direction is c.line_direction


@pytest.mark.parametrize(
    "constraint, point_attrs",
    [
        (
            DistanceConstraint(
                PointID.LOWER_WISHBONE_OUTBOARD, PointID.UPPER_WISHBONE_OUTBOARD, 2.0
            ),
            ("p1", "p2"),
        ),
        (
            AngleConstraint(
                PointID.LOWER_WISHBONE_OUTBOARD,
                PointID.UPPER_WISHBONE_OUTBOARD,
                PointID.TRACKROD_INBOARD,
                PointID.TRACKROD_OUTBOARD,
                0.5,
            ),
            ("v1_start", "v1_end", "v2_start", "v2_end"),
        ),
        (
            PointOnLineConstraint(
                PointID.TRACKROD_INBOARD,
                Point3([0.0, 0.0, 0.0]),
                Direction3([0.0, 1.0, 0.0]),
            ),
            ("point_id",),
        ),
    ],
)
def test_remap_residual_equivalence(constraint, point_attrs):
    points = [getattr(constraint, a) for a in point_attrs]
    positions_pid = _positions_for(points)
    positions_ref = _positions_for(points, keyfn=lambda pid: PointRef(Side.LEFT, pid))

    remapped = constraint.remap(_left)

    r_original = constraint.residual(positions_pid)
    r_remapped = remapped.residual(positions_ref)
    assert r_remapped == pytest.approx(r_original)


# ----------------------------------------------------------------------------
# End-to-end: the machinery is key-agnostic at runtime with PointRef keys
# ----------------------------------------------------------------------------


def test_state_round_trip_with_point_ref_keys():
    a = PointRef(Side.LEFT, PointID.LOWER_WISHBONE_OUTBOARD)
    b = PointRef(Side.RIGHT, PointID.LOWER_WISHBONE_OUTBOARD)
    positions = {a: Point3([1.0, 2.0, 3.0]), b: Point3([4.0, 5.0, 6.0])}
    state = SuspensionState(positions=positions, free_points={a, b})

    # Sorted deterministically: LEFT before RIGHT.
    assert state.free_points_order == [a, b]

    arr = state.get_free_array()
    np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    state.update_from_array(np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
    np.testing.assert_array_equal(state.positions[a].data, np.array([7.0, 8.0, 9.0]))
    np.testing.assert_array_equal(state.positions[b].data, np.array([10.0, 11.0, 12.0]))


def test_solve_with_point_ref_keys():
    # Two free points keyed by PointRef; one distance constraint plus a target
    # that pins the free DOF. Proves ResidualComputer/solver are key-agnostic.
    a = PointRef(Side.LEFT, PointID.LOWER_WISHBONE_OUTBOARD)
    b = PointRef(Side.LEFT, PointID.UPPER_WISHBONE_OUTBOARD)

    positions = {a: Point3([0.0, 0.0, 0.0]), b: Point3([3.0, 0.0, 0.0])}
    state = SuspensionState(positions=positions, free_points={a, b})

    constraint = DistanceConstraint(
        PointID.LOWER_WISHBONE_OUTBOARD, PointID.UPPER_WISHBONE_OUTBOARD, 3.0
    ).remap(_left)
    assert constraint.involved_points == {a, b}

    empty_derived = DerivedPointsManager(
        DerivedPointsSpec(functions={}, dependencies={})
    )

    # Pin all coordinates so the system is well-determined: 6 vars.
    # 1 distance constraint + 5 targets = 6 residuals.
    targets = [
        PointTarget(a, PointTargetAxis(Axis.X), 0.0),
        PointTarget(a, PointTargetAxis(Axis.Y), 0.0),
        PointTarget(a, PointTargetAxis(Axis.Z), 0.0),
        PointTarget(b, PointTargetAxis(Axis.Y), 0.0),
        PointTarget(b, PointTargetAxis(Axis.Z), 0.0),
    ]
    from kinematics.core.enums import TargetPositionMode

    targets = [t._replace(mode=TargetPositionMode.ABSOLUTE) for t in targets]

    computer = ResidualComputer(
        constraints=[constraint],
        derived_manager=empty_derived,
        state_buffer=state,
        n_target_variables=len(targets),
    )

    # Residual evaluates with PointRef keys.
    residuals = computer.compute(state.get_free_array(), targets)
    assert residuals.shape == (1 + len(targets),)

    # SweepConfig is only used here to document intent; solve directly.
    _ = SweepConfig([[t] for t in targets])

    result = solve_least_squares_problem(
        residual_function=computer.compute,
        x_0=state.get_free_array(),
        args=(targets,),
        n_residuals=computer.n_residuals,
        jacobian_function=computer.compute_jacobian,
    )
    assert result.success

    state.update_from_array(result.x)
    # Point a pinned at origin; b keeps distance 3 along +X.
    np.testing.assert_allclose(state.positions[a].data, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(state.positions[b].data, [3.0, 0.0, 0.0], atol=1e-6)
