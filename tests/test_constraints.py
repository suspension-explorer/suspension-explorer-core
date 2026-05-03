import math

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
from kinematics.core.constants import TEST_TOLERANCE
from kinematics.core.enums import Axis, PointID
from kinematics.core.geometry import Direction3, Point3


def simple_positions():
    """
    Returns a dictionary of simple coordinate positions for testing.
    """
    return {
        PointID.LOWER_WISHBONE_INBOARD_FRONT: Point3([0.0, 0.0, 0.0]),  # origin
        PointID.LOWER_WISHBONE_INBOARD_REAR: Point3([1.0, 0.0, 0.0]),  # x_unit
        PointID.LOWER_WISHBONE_OUTBOARD: Point3([0.0, 1.0, 0.0]),  # y_unit
        PointID.UPPER_WISHBONE_INBOARD_FRONT: Point3([0.0, 0.0, 1.0]),  # z_unit
        PointID.UPPER_WISHBONE_INBOARD_REAR: Point3([1.0, 1.0, 0.0]),  # diagonal_xy
        PointID.UPPER_WISHBONE_OUTBOARD: Point3([1.0, 1.0, 1.0]),  # diagonal_xyz
        PointID.PUSHROD_INBOARD: Point3([2.0, 0.0, 0.0]),  # 2x along x
        PointID.PUSHROD_OUTBOARD: Point3([0.0, 2.0, 0.0]),  # 2x along y
    }


def test_distance_constraint_satisfied():
    """
    Test distance constraint when target distance matches actual distance.
    """
    positions = simple_positions()

    # Distance from origin to x_unit is exactly 1
    constraint = DistanceConstraint(
        PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        1.0,
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_distance_constraint_too_far():
    """
    Test distance constraint when points are too far apart.
    """
    positions = simple_positions()

    # Distance from origin to x_unit is 1, but we want 0.5
    constraint = DistanceConstraint(
        PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        0.5,
    )

    residual = constraint.residual(positions)
    # Residual should be 1.0 - 0.5 = 0.5 (positive means too far)
    assert math.isclose(residual, 0.5, abs_tol=TEST_TOLERANCE)


def test_distance_constraint_too_close():
    """
    Test distance constraint when points are too close.
    """
    positions = simple_positions()

    # Distance from origin to x_unit is 1, but we want 2.0
    constraint = DistanceConstraint(
        PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        2.0,
    )

    residual = constraint.residual(positions)
    # Residual should be 1.0 - 2.0 = -1.0 (negative means too close)
    assert math.isclose(residual, -1.0, abs_tol=TEST_TOLERANCE)


def test_distance_constraint_diagonal():
    """
    Test distance constraint with diagonal vector.
    """
    positions = simple_positions()

    # Distance from origin to (1,1,1) is sqrt(3)
    constraint = DistanceConstraint(
        PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        PointID.UPPER_WISHBONE_OUTBOARD,  # (1,1,1)
        math.sqrt(3),
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_distance_constraint_negative_target():
    """
    Test distance constraint raises error for negative target distance.
    """
    with pytest.raises(ValueError):
        DistanceConstraint(
            PointID.LOWER_WISHBONE_INBOARD_FRONT,
            PointID.LOWER_WISHBONE_INBOARD_REAR,
            -1.0,
        )


def test_spherical_joint_constraint_coincident():
    """
    Test spherical joint constraint when points are coincident.
    """
    positions = simple_positions()
    # Modify to make two points coincident
    positions[PointID.LOWER_WISHBONE_INBOARD_REAR] = positions[
        PointID.LOWER_WISHBONE_INBOARD_FRONT
    ].copy()

    constraint = SphericalJointConstraint(
        PointID.LOWER_WISHBONE_INBOARD_FRONT, PointID.LOWER_WISHBONE_INBOARD_REAR
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_spherical_joint_constraint_separated():
    """
    Test spherical joint constraint when points are separated.
    """
    positions = simple_positions()

    constraint = SphericalJointConstraint(
        PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit (distance 1)
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 1.0, abs_tol=TEST_TOLERANCE)


def test_angle_constraint_perpendicular():
    """
    Test angle constraint for perpendicular vectors.
    """
    positions = simple_positions()

    # Vectors from origin to x_unit and from origin to y_unit are perpendicular
    constraint = AngleConstraint(
        v1_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        v1_end=PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        v2_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        v2_end=PointID.LOWER_WISHBONE_OUTBOARD,  # y_unit
        target_angle=math.pi / 2,  # 90 degrees
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_angle_constraint_parallel():
    """
    Test angle constraint for parallel vectors.
    """
    positions = simple_positions()

    # Vectors along x-axis (from origin to x_unit and from origin to 2x)
    constraint = AngleConstraint(
        v1_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        v1_end=PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        v2_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        v2_end=PointID.PUSHROD_INBOARD,  # 2x along x
        target_angle=0.0,  # parallel
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_angle_constraint_45_degrees():
    """
    Test angle constraint for 45-degree angle.
    """
    positions = simple_positions()

    # Vector from origin to x_unit and from origin to diagonal_xy (45 degrees)
    constraint = AngleConstraint(
        v1_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        v1_end=PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        v2_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        v2_end=PointID.UPPER_WISHBONE_INBOARD_REAR,  # diagonal_xy
        target_angle=math.pi / 4,  # 45 degrees
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_angle_constraint_invalid_target():
    """
    Test angle constraint raises error for invalid target angle.
    """
    with pytest.raises(ValueError):
        AngleConstraint(
            PointID.LOWER_WISHBONE_INBOARD_FRONT,
            PointID.LOWER_WISHBONE_INBOARD_REAR,
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.UPPER_WISHBONE_INBOARD_FRONT,
            -0.1,  # Invalid: negative angle
        )

    with pytest.raises(ValueError):
        AngleConstraint(
            PointID.LOWER_WISHBONE_INBOARD_FRONT,
            PointID.LOWER_WISHBONE_INBOARD_REAR,
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.UPPER_WISHBONE_INBOARD_FRONT,
            math.pi + 0.1,  # Invalid: > π
        )


def test_three_point_angle_constraint_45_degrees():
    """
    Test three-point angle constraint for 45-degree angle at vertex.
    """
    positions = simple_positions()

    # Angle at diagonal_xy vertex between origin and y_unit is 45 degrees
    # Vector from (1,1,0) to (0,0,0) and from (1,1,0) to (0,1,0)
    constraint = ThreePointAngleConstraint(
        p1=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        p2=PointID.UPPER_WISHBONE_INBOARD_REAR,  # diagonal_xy (vertex)
        p3=PointID.LOWER_WISHBONE_OUTBOARD,  # y_unit
        target_angle=math.pi / 4,  # 45 degrees
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_three_point_angle_constraint_90_degrees():
    """
    Test three-point angle constraint for 90-degree angle at vertex.
    """
    positions = simple_positions()

    # Angle at origin between x_unit and y_unit is 90 degrees
    constraint = ThreePointAngleConstraint(
        p1=PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        p2=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin (vertex)
        p3=PointID.LOWER_WISHBONE_OUTBOARD,  # y_unit
        target_angle=math.pi / 2,  # 90 degrees
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_vectors_parallel_constraint_satisfied():
    """
    Test parallel vectors constraint when vectors are parallel.
    """
    positions = simple_positions()

    # Vectors from origin to x_unit and from y_unit to diagonal_xy are both along x
    constraint = VectorsParallelConstraint(
        v1_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        v1_end=PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        v2_start=PointID.LOWER_WISHBONE_OUTBOARD,  # y_unit
        v2_end=PointID.UPPER_WISHBONE_INBOARD_REAR,  # diagonal_xy
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_vectors_parallel_constraint_violated():
    """
    Test parallel vectors constraint when vectors are perpendicular.
    """
    positions = simple_positions()

    # Vectors along x and y axes are perpendicular, not parallel
    constraint = VectorsParallelConstraint(
        v1_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        v1_end=PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        v2_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        v2_end=PointID.LOWER_WISHBONE_OUTBOARD,  # y_unit
    )

    residual = constraint.residual(positions)
    # For perpendicular vectors, cross product magnitude is 1
    assert math.isclose(residual, 1.0, abs_tol=TEST_TOLERANCE)


def test_vectors_perpendicular_constraint_satisfied():
    """
    Test perpendicular vectors constraint when vectors are perpendicular.
    """
    positions = simple_positions()

    # Vectors along x and y axes are perpendicular
    constraint = VectorsPerpendicularConstraint(
        v1_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        v1_end=PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        v2_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        v2_end=PointID.LOWER_WISHBONE_OUTBOARD,  # y_unit
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_vectors_perpendicular_constraint_violated():
    """
    Test perpendicular vectors constraint when vectors are parallel.
    """
    positions = simple_positions()

    # Two parallel vectors along x-axis
    constraint = VectorsPerpendicularConstraint(
        v1_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        v1_end=PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        v2_start=PointID.LOWER_WISHBONE_OUTBOARD,  # y_unit
        v2_end=PointID.UPPER_WISHBONE_INBOARD_REAR,  # diagonal_xy
    )

    residual = constraint.residual(positions)
    # For parallel vectors, dot product is 1
    assert math.isclose(residual, 1.0, abs_tol=TEST_TOLERANCE)


def test_equal_distance_constraint_satisfied():
    """
    Test equal distance constraint when distances are equal.
    """
    positions = simple_positions()

    # Distance from origin to x_unit equals distance from origin to y_unit (both 1)
    constraint = EqualDistanceConstraint(
        p1=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        p2=PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        p3=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        p4=PointID.LOWER_WISHBONE_OUTBOARD,  # y_unit
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_equal_distance_constraint_violated():
    """
    Test equal distance constraint when distances are unequal.
    """
    positions = simple_positions()

    # Distance from origin to x_unit (1) vs distance from origin to 2x (2)
    constraint = EqualDistanceConstraint(
        p1=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        p2=PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit (dist 1)
        p3=PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        p4=PointID.PUSHROD_INBOARD,  # 2x (dist 2)
    )

    residual = constraint.residual(positions)
    # Residual should be 1 - 2 = -1
    assert math.isclose(residual, -1.0, abs_tol=TEST_TOLERANCE)


def test_fixed_axis_constraint_satisfied_x():
    """
    Test fixed axis constraint when X coordinate matches target.
    """
    positions = simple_positions()

    # x_unit has X coordinate = 1.0
    constraint = FixedAxisConstraint(
        PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        Axis.X,
        1.0,
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_fixed_axis_constraint_violated_y():
    """
    Test fixed axis constraint when Y coordinate doesn't match target.
    """
    positions = simple_positions()

    # x_unit has Y coordinate = 0.0, but we want 0.5
    constraint = FixedAxisConstraint(
        PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        Axis.Y,
        0.5,
    )

    residual = constraint.residual(positions)
    # Residual should be 0.0 - 0.5 = -0.5
    assert math.isclose(residual, -0.5, abs_tol=TEST_TOLERANCE)


def test_fixed_axis_constraint_satisfied_z():
    """
    Test fixed axis constraint when Z coordinate matches target.
    """
    positions = simple_positions()

    # z_unit has Z coordinate = 1.0
    constraint = FixedAxisConstraint(
        PointID.UPPER_WISHBONE_INBOARD_FRONT,  # z_unit
        Axis.Z,
        1.0,
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_point_on_line_constraint_on_line():
    """
    Test point-on-line constraint when point is on the line.
    """
    positions = simple_positions()

    # Point y_unit (0,1,0) is on Y axis through origin
    line_point = Point3([0.0, 0.0, 0.0])
    line_direction = Direction3([0.0, 1.0, 0.0])

    constraint = PointOnLineConstraint(
        PointID.LOWER_WISHBONE_OUTBOARD,  # y_unit
        line_point,
        line_direction,
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_point_on_line_constraint_off_line():
    """
    Test point-on-line constraint when point is off the line.
    """
    positions = simple_positions()

    # Point x_unit (1,0,0) is distance 1 from Y axis
    line_point = Point3([0.0, 0.0, 0.0])
    line_direction = Direction3([0.0, 1.0, 0.0])

    constraint = PointOnLineConstraint(
        PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        line_point,
        line_direction,
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 1.0, abs_tol=TEST_TOLERANCE)


def test_point_on_line_constraint_zero_direction():
    """
    Test point-on-line constraint raises error for zero direction vector.
    """
    with pytest.raises(ValueError):
        PointOnLineConstraint(
            PointID.LOWER_WISHBONE_INBOARD_FRONT,
            Point3([0.0, 0.0, 0.0]),
            Direction3([0.0, 0.0, 0.0]),  # Zero direction
        )


def test_point_on_plane_constraint_on_plane():
    """
    Test point-on-plane constraint when point is on the plane.
    """
    positions = simple_positions()

    # Point diagonal_xy (1,1,0) is on Z=0 plane
    plane_point = Point3([0.0, 0.0, 0.0])
    plane_normal = Direction3([0.0, 0.0, 1.0])

    constraint = PointOnPlaneConstraint(
        PointID.UPPER_WISHBONE_INBOARD_REAR,  # diagonal_xy
        plane_point,
        plane_normal,
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_point_on_plane_constraint_above_plane():
    """
    Test point-on-plane constraint when point is above the plane.
    """
    positions = simple_positions()

    # Point z_unit (0,0,1) is 1 unit above Z=0 plane
    plane_point = Point3([0.0, 0.0, 0.0])
    plane_normal = Direction3([0.0, 0.0, 1.0])

    constraint = PointOnPlaneConstraint(
        PointID.UPPER_WISHBONE_INBOARD_FRONT,  # z_unit
        plane_point,
        plane_normal,
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 1.0, abs_tol=TEST_TOLERANCE)


def test_point_on_plane_constraint_below_plane():
    """
    Test point-on-plane constraint when point is below the plane.
    """
    positions = simple_positions()
    # Modify z_unit to be below Z=0 plane
    positions[PointID.UPPER_WISHBONE_INBOARD_FRONT] = Point3([0.0, 0.0, -2.0])

    plane_point = Point3([0.0, 0.0, 0.0])
    plane_normal = Direction3([0.0, 0.0, 1.0])

    constraint = PointOnPlaneConstraint(
        PointID.UPPER_WISHBONE_INBOARD_FRONT, plane_point, plane_normal
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, -2.0, abs_tol=TEST_TOLERANCE)


def test_point_on_plane_constraint_zero_normal():
    """
    Test point-on-plane constraint raises error for zero normal vector.
    """
    with pytest.raises(ValueError):
        PointOnPlaneConstraint(
            PointID.LOWER_WISHBONE_INBOARD_FRONT,
            Point3([0.0, 0.0, 0.0]),
            Direction3([0.0, 0.0, 0.0]),  # Zero normal
        )


def test_coplanar_points_constraint_coplanar():
    """
    Test coplanar points constraint when points are coplanar.
    """
    positions = simple_positions()

    # Origin, x_unit, y_unit, and diagonal_xy are all in Z=0 plane
    constraint = CoplanarPointsConstraint(
        PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        PointID.LOWER_WISHBONE_OUTBOARD,  # y_unit
        PointID.UPPER_WISHBONE_INBOARD_REAR,  # diagonal_xy
    )

    residual = constraint.residual(positions)
    assert math.isclose(residual, 0.0, abs_tol=TEST_TOLERANCE)


def test_coplanar_points_constraint_non_coplanar():
    """
    Test coplanar points constraint when points are not coplanar.
    """
    positions = simple_positions()

    # Origin, x_unit, y_unit, and z_unit are not coplanar
    constraint = CoplanarPointsConstraint(
        PointID.LOWER_WISHBONE_INBOARD_FRONT,  # origin
        PointID.LOWER_WISHBONE_INBOARD_REAR,  # x_unit
        PointID.LOWER_WISHBONE_OUTBOARD,  # y_unit
        PointID.UPPER_WISHBONE_INBOARD_FRONT,  # z_unit
    )

    residual = constraint.residual(positions)
    # Scalar triple product of x, y, z unit vectors is 1 (non-coplanar)
    assert math.isclose(residual, 1.0, abs_tol=TEST_TOLERANCE)


def test_constraint_involved_points():
    """
    Test that constraints report correct involved points.
    """
    # Test a few different constraint types
    dist_constraint = DistanceConstraint(
        PointID.LOWER_WISHBONE_INBOARD_FRONT, PointID.LOWER_WISHBONE_INBOARD_REAR, 1.0
    )
    assert dist_constraint.involved_points == {
        PointID.LOWER_WISHBONE_INBOARD_FRONT,
        PointID.LOWER_WISHBONE_INBOARD_REAR,
    }

    angle_constraint = AngleConstraint(
        PointID.LOWER_WISHBONE_INBOARD_FRONT,
        PointID.LOWER_WISHBONE_INBOARD_REAR,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.UPPER_WISHBONE_INBOARD_FRONT,
        math.pi / 2,
    )
    assert angle_constraint.involved_points == {
        PointID.LOWER_WISHBONE_INBOARD_FRONT,
        PointID.LOWER_WISHBONE_INBOARD_REAR,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.UPPER_WISHBONE_INBOARD_FRONT,
    }

    fixed_constraint = FixedAxisConstraint(
        PointID.LOWER_WISHBONE_INBOARD_FRONT, Axis.X, 0.0
    )
    assert fixed_constraint.involved_points == {PointID.LOWER_WISHBONE_INBOARD_FRONT}
