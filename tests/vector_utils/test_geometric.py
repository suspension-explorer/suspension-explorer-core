import math

import numpy as np
import pytest

from kinematics.core.constants import TEST_TOLERANCE
from kinematics.core.geometry import Direction3, Point3, Vector3
from kinematics.core.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_point_midpoint,
    compute_point_to_line_distance,
    compute_point_to_plane_distance,
    compute_scalar_triple_product,
    compute_vector_vector_angle,
    compute_vectors_cross_product_magnitude,
    compute_vectors_dot_product,
    intersect_line_with_vertical_plane,
    intersect_two_planes,
    plane_from_three_points,
)
from kinematics.core.vector_utils.visualization import (
    plot_line_plane_intersection,
    plot_plane_from_points,
    plot_plane_intersection,
)


def simple_points():
    """
    Returns a dictionary of simple coordinate positions (Point3) for testing.
    """
    return {
        "origin": Point3([0.0, 0.0, 0.0]),
        "x_unit": Point3([1.0, 0.0, 0.0]),
        "y_unit": Point3([0.0, 1.0, 0.0]),
        "z_unit": Point3([0.0, 0.0, 1.0]),
        "diagonal_xy": Point3([1.0, 1.0, 0.0]),
        "diagonal_xyz": Point3([1.0, 1.0, 1.0]),
    }


def simple_vectors():
    """
    Returns a dictionary of simple coordinate vectors (Vector3) for testing.
    """
    return {
        "origin": Vector3([0.0, 0.0, 0.0]),
        "x_unit": Vector3([1.0, 0.0, 0.0]),
        "y_unit": Vector3([0.0, 1.0, 0.0]),
        "z_unit": Vector3([0.0, 0.0, 1.0]),
        "diagonal_xy": Vector3([1.0, 1.0, 0.0]),
        "diagonal_xyz": Vector3([1.0, 1.0, 1.0]),
    }


def test_point_distance_unit_vectors():
    """
    Test distances between points on coordinate axes.
    """
    pos = simple_points()

    # Distance from origin to unit vector offset positions should be 1.
    assert math.isclose(
        compute_point_point_distance(pos["origin"], pos["x_unit"]),
        1.0,
        abs_tol=TEST_TOLERANCE,
    )
    assert math.isclose(
        compute_point_point_distance(pos["origin"], pos["y_unit"]),
        1.0,
        abs_tol=TEST_TOLERANCE,
    )
    assert math.isclose(
        compute_point_point_distance(pos["origin"], pos["z_unit"]),
        1.0,
        abs_tol=TEST_TOLERANCE,
    )

    # Distance between orthogonal unit vectors should be sqrt(2).
    assert math.isclose(
        compute_point_point_distance(pos["x_unit"], pos["y_unit"]),
        math.sqrt(2),
        abs_tol=TEST_TOLERANCE,
    )

    # Distance from origin to (1,1,1) should be sqrt(3).
    assert math.isclose(
        compute_point_point_distance(pos["origin"], pos["diagonal_xyz"]),
        math.sqrt(3),
        abs_tol=TEST_TOLERANCE,
    )


def test_point_distance_zero():
    """
    Test distance from a point to itself is zero.
    """
    pos = simple_points()
    assert math.isclose(
        compute_point_point_distance(pos["origin"], pos["origin"]),
        0.0,
        abs_tol=TEST_TOLERANCE,
    )


def test_compute_midpoint_coordinate_axes():
    """
    Test midpoints using simple coordinate positions.
    """
    pos = simple_points()

    # Midpoint between origin and x_unit should be (0.5, 0, 0).
    mid = compute_point_point_midpoint(pos["origin"], pos["x_unit"])
    expected = np.array([0.5, 0.0, 0.0])
    np.testing.assert_allclose(mid.data, expected, atol=TEST_TOLERANCE)

    # Midpoint between x_unit and y_unit should be (0.5, 0.5, 0).
    mid = compute_point_point_midpoint(pos["x_unit"], pos["y_unit"])
    expected = np.array([0.5, 0.5, 0.0])
    np.testing.assert_allclose(mid.data, expected, atol=TEST_TOLERANCE)


def test_compute_midpoint_diagonal():
    """
    Test midpoint calculation with diagonal vectors.
    """
    pos = simple_points()

    # Midpoint between origin and (1,1,1) should be (0.5, 0.5, 0.5).
    mid = compute_point_point_midpoint(pos["origin"], pos["diagonal_xyz"])
    expected = np.array([0.5, 0.5, 0.5])
    np.testing.assert_allclose(mid.data, expected, atol=TEST_TOLERANCE)


def test_compute_vector_angle_perpendicular():
    """
    Test angle between perpendicular vectors is pi/2.
    """
    pos = simple_vectors()

    # X and Y axes are perpendicular.
    angle = compute_vector_vector_angle(pos["x_unit"], pos["y_unit"])
    assert math.isclose(angle, math.pi / 2, abs_tol=TEST_TOLERANCE)

    # Y and Z axes are perpendicular.
    angle = compute_vector_vector_angle(pos["y_unit"], pos["z_unit"])
    assert math.isclose(angle, math.pi / 2, abs_tol=TEST_TOLERANCE)


def test_compute_vector_angle_parallel():
    """
    Test angle between parallel vectors is 0.
    """
    pos = simple_vectors()

    # Vector with itself.
    angle = compute_vector_vector_angle(pos["x_unit"], pos["x_unit"])
    assert math.isclose(angle, 0.0, abs_tol=TEST_TOLERANCE)

    # Parallel vectors (same direction).
    v1 = Vector3([2.0, 0.0, 0.0])  # 2x along X.
    angle = compute_vector_vector_angle(pos["x_unit"], v1)
    assert math.isclose(angle, 0.0, abs_tol=TEST_TOLERANCE)


def test_compute_vector_angle_antiparallel():
    """
    Test angle between anti-parallel vectors is pi.
    """
    pos = simple_vectors()

    # Opposite directions along X axis.
    v_neg_x = Vector3([-1.0, 0.0, 0.0])
    angle = compute_vector_vector_angle(pos["x_unit"], v_neg_x)
    assert math.isclose(angle, math.pi, abs_tol=TEST_TOLERANCE)


def test_compute_vector_angle_45_degrees():
    """
    Test angle for 45-degree case.
    """
    pos = simple_vectors()

    # X unit vector and diagonal (1,1,0) should form 45 degrees.
    angle = compute_vector_vector_angle(pos["x_unit"], pos["diagonal_xy"])
    assert math.isclose(angle, math.pi / 4, abs_tol=TEST_TOLERANCE)


def test_compute_vector_angle_zero_vector():
    """
    Test that zero vectors raise ValueError.
    """
    pos = simple_vectors()
    zero_vec = Vector3([0.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        compute_vector_vector_angle(pos["x_unit"], zero_vec)

    with pytest.raises(ValueError):
        compute_vector_vector_angle(zero_vec, pos["y_unit"])


def test_cross_product_magnitude_perpendicular():
    """
    Test cross product magnitude for perpendicular vectors is 1.
    """
    pos = simple_vectors()

    # Perpendicular unit vectors have cross product magnitude 1.
    cross_mag = compute_vectors_cross_product_magnitude(pos["x_unit"], pos["y_unit"])
    assert math.isclose(cross_mag, 1.0, abs_tol=TEST_TOLERANCE)


def test_cross_product_magnitude_parallel():
    """
    Test cross product magnitude for parallel vectors is 0.
    """
    pos = simple_vectors()

    # Parallel vectors have cross product magnitude 0.
    v_parallel = Vector3([2.0, 0.0, 0.0])  # Parallel to x_unit
    cross_mag = compute_vectors_cross_product_magnitude(pos["x_unit"], v_parallel)
    assert math.isclose(cross_mag, 0.0, abs_tol=TEST_TOLERANCE)


def test_cross_product_magnitude_45_degrees():
    """
    Test cross product magnitude for 45-degree angle.
    """
    pos = simple_vectors()

    # 45 degree angle should give cross product magnitude sin(pi/4) = sqrt(2)/2
    cross_mag = compute_vectors_cross_product_magnitude(
        pos["x_unit"], pos["diagonal_xy"]
    )
    expected = math.sin(math.pi / 4)  # sqrt(2)/2 ~= 0.707
    assert math.isclose(cross_mag, expected, abs_tol=TEST_TOLERANCE)


def test_dot_product_perpendicular():
    """
    Test dot product for perpendicular vectors is 0.
    """
    pos = simple_vectors()

    # Perpendicular unit vectors have dot product 0.
    dot = compute_vectors_dot_product(pos["x_unit"], pos["y_unit"])
    assert math.isclose(dot, 0.0, abs_tol=TEST_TOLERANCE)


def test_dot_product_parallel():
    """
    Test dot product for parallel vectors is 1.
    """
    pos = simple_vectors()

    # Parallel unit vectors have dot product 1.
    v_parallel = Vector3([2.0, 0.0, 0.0])  # Parallel to x_unit
    dot = compute_vectors_dot_product(pos["x_unit"], v_parallel)
    assert math.isclose(dot, 1.0, abs_tol=TEST_TOLERANCE)


def test_dot_product_antiparallel():
    """
    Test dot product for anti-parallel vectors is -1.
    """
    pos = simple_vectors()

    # Anti-parallel unit vectors have dot product -1.
    v_antiparallel = Vector3([-1.0, 0.0, 0.0])
    dot = compute_vectors_dot_product(pos["x_unit"], v_antiparallel)
    assert math.isclose(dot, -1.0, abs_tol=TEST_TOLERANCE)


def test_dot_product_45_degrees():
    """
    Test dot product for 45-degree angle.
    """
    pos = simple_vectors()

    # 45 degree angle should give dot product cos(pi/4) = sqrt(2)/2
    dot = compute_vectors_dot_product(pos["x_unit"], pos["diagonal_xy"])
    expected = math.cos(math.pi / 4)  # sqrt(2)/2 ~= 0.707
    assert math.isclose(dot, expected, abs_tol=TEST_TOLERANCE)


def test_point_to_line_distance_on_line():
    """
    Test distance from point on line to line is 0.
    """
    pts = simple_points()

    # Point on the line (Y axis) should have distance 0.
    line_point = pts["origin"]
    line_direction = Direction3([0.0, 1.0, 0.0])
    test_point = Point3([0.0, 2.0, 0.0])  # On Y axis

    distance = compute_point_to_line_distance(test_point, line_point, line_direction)
    assert math.isclose(distance, 0.0, abs_tol=TEST_TOLERANCE)


def test_point_to_line_distance_perpendicular():
    """
    Test distance from point perpendicular to line.
    """
    pts = simple_points()

    # Point (1,0,0) to Y axis should have distance 1.
    line_point = pts["origin"]
    line_direction = Direction3([0.0, 1.0, 0.0])
    test_point = pts["x_unit"]

    distance = compute_point_to_line_distance(test_point, line_point, line_direction)
    assert math.isclose(distance, 1.0, abs_tol=TEST_TOLERANCE)


def test_point_to_line_distance_diagonal():
    """
    Test distance calculation with diagonal geometry.
    """
    pts = simple_points()

    # Point (1,1,0) to X axis should have distance 1 (Y component).
    line_point = pts["origin"]
    line_direction = Direction3([1.0, 0.0, 0.0])
    test_point = pts["diagonal_xy"]

    distance = compute_point_to_line_distance(test_point, line_point, line_direction)
    assert math.isclose(distance, 1.0, abs_tol=TEST_TOLERANCE)


def test_point_to_plane_distance_on_plane():
    """
    Test distance from point on plane to plane is 0.
    """
    pts = simple_points()

    # Point on Z=0 plane should have distance 0.
    plane_point = pts["origin"]
    plane_normal = Direction3([0.0, 0.0, 1.0])
    test_point = pts["diagonal_xy"]  # (1,1,0) - on Z=0 plane

    distance = compute_point_to_plane_distance(test_point, plane_point, plane_normal)
    assert math.isclose(distance, 0.0, abs_tol=TEST_TOLERANCE)


def test_point_to_plane_distance_positive():
    """
    Test positive signed distance to plane.
    """
    pts = simple_points()

    # Point above Z=0 plane should have positive distance.
    plane_point = pts["origin"]
    plane_normal = Direction3([0.0, 0.0, 1.0])
    test_point = Point3([0.0, 0.0, 2.0])  # 2 units above Z=0.

    distance = compute_point_to_plane_distance(test_point, plane_point, plane_normal)
    assert math.isclose(distance, 2.0, abs_tol=TEST_TOLERANCE)


def test_point_to_plane_distance_negative():
    """
    Test negative signed distance to plane.
    """
    pts = simple_points()

    # Point below Z=0 plane should have negative distance.
    plane_point = pts["origin"]
    plane_normal = Direction3([0.0, 0.0, 1.0])
    test_point = Point3([0.0, 0.0, -1.5])  # 1.5 units below Z=0

    distance = compute_point_to_plane_distance(test_point, plane_point, plane_normal)
    assert math.isclose(distance, -1.5, abs_tol=TEST_TOLERANCE)


def test_scalar_triple_product_right_handed_basis():
    """
    Test scalar triple product for right-handed unit basis is 1.
    """
    pos = simple_vectors()

    # X dot (Y cross Z) = 1 for right-handed coordinate system.
    triple_product = compute_scalar_triple_product(
        pos["x_unit"],
        pos["y_unit"],
        pos["z_unit"],
    )
    assert math.isclose(triple_product, 1.0, abs_tol=TEST_TOLERANCE)


def test_scalar_triple_product_left_handed_basis():
    """
    Test scalar triple product for left-handed basis is -1.
    """
    pos = simple_vectors()

    # X dot (Z cross Y) = -1 (reversed order).
    triple_product = compute_scalar_triple_product(
        pos["x_unit"],
        pos["z_unit"],
        pos["y_unit"],
    )
    assert math.isclose(triple_product, -1.0, abs_tol=TEST_TOLERANCE)


def test_scalar_triple_product_coplanar():
    """
    Test scalar triple product for coplanar vectors is 0.
    """
    pos = simple_vectors()

    # Three vectors in XY plane should have triple product 0.
    v1 = pos["x_unit"]
    v2 = pos["y_unit"]
    v3 = pos["diagonal_xy"]  # Also in XY plane

    triple_product = compute_scalar_triple_product(v1, v2, v3)
    assert math.isclose(triple_product, 0.0, abs_tol=TEST_TOLERANCE)


def test_zero_vector_errors():
    """
    Test that functions properly raise errors for zero-length vectors.
    """
    zero_vec = Vector3([0.0, 0.0, 0.0])
    unit_vec = Vector3([1.0, 0.0, 0.0])

    # These functions should raise ValueError for zero vectors.
    with pytest.raises(ValueError):
        compute_vectors_cross_product_magnitude(zero_vec, unit_vec)

    with pytest.raises(ValueError):
        compute_vectors_dot_product(unit_vec, zero_vec)

    # Direction3 rejects zero-length vectors at construction time.
    with pytest.raises(ValueError):
        Direction3([0.0, 0.0, 0.0])


# Tests for plane_from_three_points


def test_plane_from_three_points_xy_plane():
    """
    Test plane construction from three points in XY plane.
    """
    # Three points in XY plane (Z=0)
    a = Point3([0.0, 0.0, 0.0])
    b = Point3([1.0, 0.0, 0.0])
    c = Point3([0.0, 1.0, 0.0])

    result = plane_from_three_points(a, b, c)
    assert result is not None

    normal, d = result

    # Visualize the result if plotting is enabled
    plot_plane_from_points(a, b, c, normal, d, "XY Plane from Three Points")

    # Normal should be pointing in +Z direction (or -Z, but consistent)
    assert math.isclose(abs(normal[2]), 1.0, abs_tol=TEST_TOLERANCE)
    assert math.isclose(abs(normal[0]), 0.0, abs_tol=TEST_TOLERANCE)
    assert math.isclose(abs(normal[1]), 0.0, abs_tol=TEST_TOLERANCE)

    # Distance d should be 0 since plane passes through origin
    assert math.isclose(d, 0.0, abs_tol=TEST_TOLERANCE)


def test_plane_from_three_points_arbitrary():
    """
    Test plane construction from three arbitrary non-collinear points.
    """
    # Three points forming a triangle
    a = Point3([1.0, 0.0, 0.0])
    b = Point3([0.0, 1.0, 0.0])
    c = Point3([0.0, 0.0, 1.0])

    result = plane_from_three_points(a, b, c)
    assert result is not None

    normal, d = result

    # Visualize the result if plotting is enabled
    plot_plane_from_points(a, b, c, normal, d, "Arbitrary Triangle Plane")

    # Normal should be unit length
    assert math.isclose(np.linalg.norm(normal.data), 1.0, abs_tol=TEST_TOLERANCE)

    # All three points should satisfy the plane equation n dot x + d = 0
    assert math.isclose(np.dot(normal.data, a.data) + d, 0.0, abs_tol=TEST_TOLERANCE)
    assert math.isclose(np.dot(normal.data, b.data) + d, 0.0, abs_tol=TEST_TOLERANCE)
    assert math.isclose(np.dot(normal.data, c.data) + d, 0.0, abs_tol=TEST_TOLERANCE)


def test_plane_from_three_points_collinear():
    """
    Test that collinear points return None (degenerate case).
    """
    # Three collinear points
    a = Point3([0.0, 0.0, 0.0])
    b = Point3([1.0, 1.0, 1.0])
    c = Point3([2.0, 2.0, 2.0])

    result = plane_from_three_points(a, b, c)
    assert result is None


def test_plane_from_three_points_duplicate():
    """
    Test that duplicate points return None (degenerate case).
    """
    # Two identical points
    a = Point3([1.0, 2.0, 3.0])
    b = Point3([1.0, 2.0, 3.0])  # Same as a
    c = Point3([4.0, 5.0, 6.0])

    result = plane_from_three_points(a, b, c)
    assert result is None


def test_plane_from_three_points_offset_plane():
    """
    Test plane construction for a plane not passing through origin.
    """
    # Three points in plane Z = 5
    a = Point3([0.0, 0.0, 5.0])
    b = Point3([1.0, 0.0, 5.0])
    c = Point3([0.0, 1.0, 5.0])

    result = plane_from_three_points(a, b, c)
    assert result is not None

    normal, d = result

    # Normal should be pointing in Z direction
    assert math.isclose(abs(normal[2]), 1.0, abs_tol=TEST_TOLERANCE)

    # Distance should be -5 (since n dot x + d = 0 and points have Z = 5)
    expected_d = -5.0 if normal[2] > 0 else 5.0
    assert math.isclose(d, expected_d, abs_tol=TEST_TOLERANCE)


# Tests for intersect_two_planes


def test_intersect_two_planes_xy_xz():
    """
    Test intersection of XY plane and XZ plane (should give X axis).
    """
    n1 = Direction3([0.0, 0.0, 1.0])
    d1 = 0.0
    n2 = Direction3([0.0, 1.0, 0.0])
    d2 = 0.0

    result = intersect_two_planes(n1, d1, n2, d2)
    assert result is not None

    point, direction = result

    # Visualize the result if plotting is enabled
    plot_plane_intersection(
        n1, d1, n2, d2, point, direction, "XY and XZ Plane Intersection"
    )

    # Direction should be along X axis (Direction3 is already unit length)
    assert math.isclose(abs(direction[0]), 1.0, abs_tol=TEST_TOLERANCE)
    assert math.isclose(abs(direction[1]), 0.0, abs_tol=TEST_TOLERANCE)
    assert math.isclose(abs(direction[2]), 0.0, abs_tol=TEST_TOLERANCE)

    # Point should be on X axis (Y = 0, Z = 0)
    assert math.isclose(point[1], 0.0, abs_tol=TEST_TOLERANCE)
    assert math.isclose(point[2], 0.0, abs_tol=TEST_TOLERANCE)


def test_intersect_two_planes_parallel():
    """
    Test that parallel planes return None.
    """
    n1 = Direction3([0.0, 0.0, 1.0])
    d1 = 0.0
    n2 = Direction3([0.0, 0.0, 1.0])
    d2 = 5.0

    result = intersect_two_planes(n1, d1, n2, d2)
    assert result is None


def test_intersect_two_planes_antiparallel():
    """
    Test that anti-parallel planes return None.
    """
    n1 = Direction3([1.0, 0.0, 0.0])
    d1 = 0.0
    n2 = Direction3([-1.0, 0.0, 0.0])
    d2 = 5.0

    result = intersect_two_planes(n1, d1, n2, d2)
    assert result is None


def test_intersect_two_planes_arbitrary():
    """
    Test intersection of two arbitrary planes.
    """
    n1 = Direction3([1.0, 1.0, 0.0])
    d1 = 0.0
    n2 = Direction3([0.0, 0.0, 1.0])
    d2 = -1.0

    result = intersect_two_planes(n1, d1, n2, d2)
    assert result is not None

    point, direction = result

    # The intersection line should be at Z = 1
    assert math.isclose(point[2], 1.0, abs_tol=TEST_TOLERANCE)

    # Direction should be perpendicular to both normals
    assert math.isclose(direction.dot(n1), 0.0, abs_tol=TEST_TOLERANCE)
    assert math.isclose(direction.dot(n2), 0.0, abs_tol=TEST_TOLERANCE)


def test_intersect_two_planes_validation():
    """
    Test that intersection result satisfies both plane equations.
    """
    n1 = Direction3([1.0, 0.0, 0.0])
    d1 = -2.0
    n2 = Direction3([0.0, 1.0, 1.0])
    d2 = -3.0

    result = intersect_two_planes(n1, d1, n2, d2)
    assert result is not None

    point, direction = result

    # Point should satisfy both plane equations: n dot p + d = 0
    assert math.isclose(np.dot(n1.data, point.data) + d1, 0.0, abs_tol=TEST_TOLERANCE)
    assert math.isclose(np.dot(n2.data, point.data) + d2, 0.0, abs_tol=TEST_TOLERANCE)

    # Another point on the line should also satisfy both equations
    test_point = point + 5.0 * direction
    assert math.isclose(
        np.dot(n1.data, test_point.data) + d1, 0.0, abs_tol=TEST_TOLERANCE
    )
    assert math.isclose(
        np.dot(n2.data, test_point.data) + d2, 0.0, abs_tol=TEST_TOLERANCE
    )


# Tests for intersect_line_with_vertical_plane


def test_intersect_line_with_vertical_plane_simple():
    """
    Test line intersection with vertical plane Y = constant.
    """
    line_point = Point3([0.0, 0.0, 0.0])
    line_direction = Direction3([1.0, 1.0, 1.0])
    plane_y = 5.0

    intersection = intersect_line_with_vertical_plane(
        line_point, line_direction, plane_y
    )
    assert intersection is not None

    # Visualize the result if plotting is enabled
    plot_line_plane_intersection(
        line_point,
        line_direction,
        plane_y,
        intersection,
        "Simple Line-Plane Intersection",
    )

    # Intersection should be at (5, 5, 5)
    expected = np.array([5.0, 5.0, 5.0])
    np.testing.assert_allclose(intersection.data, expected, atol=TEST_TOLERANCE)


def test_intersect_line_with_vertical_plane_parallel():
    """
    Test that line parallel to vertical plane returns None.
    """
    line_point = Point3([0.0, 2.0, 0.0])
    line_direction = Direction3([1.0, 0.0, 0.0])  # No Y direction
    plane_y = 5.0

    intersection = intersect_line_with_vertical_plane(
        line_point, line_direction, plane_y
    )
    assert intersection is None


def test_intersect_line_with_vertical_plane_negative_direction():
    """
    Test line intersection with negative Y direction.
    """
    line_point = Point3([1.0, 10.0, 2.0])
    line_direction = Direction3([0.0, -2.0, 1.0])
    plane_y = 4.0

    intersection = intersect_line_with_vertical_plane(
        line_point, line_direction, plane_y
    )
    assert intersection is not None

    # Should intersect when Y decreases from 10 to 4 (difference of 6)
    # Direction is normalized, so t scales accordingly.
    expected = np.array([1.0, 4.0, 5.0])
    np.testing.assert_allclose(intersection.data, expected, atol=TEST_TOLERANCE)


def test_intersect_line_with_vertical_plane_already_on_plane():
    """
    Test line that starts on the vertical plane.
    """
    line_point = Point3([0.0, 3.0, 0.0])
    line_direction = Direction3([1.0, 2.0, 1.0])
    plane_y = 3.0

    intersection = intersect_line_with_vertical_plane(
        line_point, line_direction, plane_y
    )
    assert intersection is not None

    # Should return the starting point since t = 0
    np.testing.assert_allclose(intersection.data, line_point.data, atol=TEST_TOLERANCE)


def test_intersect_line_with_vertical_plane_arbitrary():
    """
    Test intersection with arbitrary line and plane values.
    """
    line_point = Point3([2.0, -1.0, 3.0])
    line_direction = Direction3([1.0, 3.0, -2.0])
    plane_y = 8.0

    intersection = intersect_line_with_vertical_plane(
        line_point, line_direction, plane_y
    )
    assert intersection is not None

    # Y changes from -1 to 8 (difference of 9)
    # Direction is normalized, so t scales accordingly.
    expected = np.array([5.0, 8.0, -3.0])
    np.testing.assert_allclose(intersection.data, expected, atol=TEST_TOLERANCE)


def test_intersect_line_with_vertical_plane_small_direction():
    """
    Test line with very small Y direction component (near parallel).
    """
    line_point = Point3([0.0, 0.0, 0.0])
    line_direction = Direction3([1.0, 1e-5, 1.0])
    plane_y = 1.0

    intersection = intersect_line_with_vertical_plane(
        line_point, line_direction, plane_y
    )
    assert intersection is not None

    # Should still compute intersection, just with very large t value
    assert math.isclose(intersection[1], 1.0, abs_tol=TEST_TOLERANCE)
