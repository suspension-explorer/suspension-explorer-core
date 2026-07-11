"""
Geometric computation utilities.

This module provides functions for computing geometric relationships between points and
vectors, including distances, midpoints, and angles. These utilities are fundamental to
kinematic analysis and constraint evaluation.
"""

import numpy as np

from kinematics.core.constants import EPS_GEOMETRIC, EPS_NUMERICAL
from kinematics.core.enums import Axis
from kinematics.core.geometry import Direction3, Point3, Vector3, midpoint
from kinematics.core.vector_utils.generic import normalize_vector


def compute_point_point_distance(p1: Point3, p2: Point3) -> float:
    """
    Compute the Euclidean distance between two points.

    Args:
        p1: First point in 3D space.
        p2: Second point in 3D space.

    Returns:
        The Euclidean distance between the two points (always non-negative).
    """
    return (p2 - p1).norm()


def signed_angle_about_axis(
    design: Point3,
    current: Point3,
    axis_point: Point3,
    axis_direction: Direction3,
) -> float:
    """Return the signed rotation from design to current about a fixed axis."""
    direction = axis_direction.data
    design_radius = design - axis_point
    current_radius = current - axis_point
    design_perpendicular = design_radius - direction * design_radius.dot(axis_direction)
    current_perpendicular = current_radius - direction * current_radius.dot(
        axis_direction
    )
    sine = axis_direction.dot(design_radius.cross(current_radius))
    cosine = design_perpendicular.dot(current_perpendicular)
    if (
        design_perpendicular.norm() < EPS_GEOMETRIC
        or current_perpendicular.norm() < EPS_GEOMETRIC
    ):
        raise ValueError("Rotation is undefined for a point on the rotation axis")
    return float(np.arctan2(sine, cosine))


def compute_point_point_midpoint(p1: Point3, p2: Point3) -> Point3:
    """
    Compute the midpoint between two points.

    Uses the affine-correct formulation: a + (b - a) / 2.

    Args:
        p1: First point in 3D space.
        p2: Second point in 3D space.

    Returns:
        The midpoint between the two points.
    """
    return midpoint(p1, p2)


def compute_vector_vector_angle(v1: Vector3, v2: Vector3) -> float:
    """
    Compute the angle between two vectors in radians using atan2 for robustness.

    The vectors are automatically normalized before computing the angle,
    so input vectors do not need to be unit length. Uses atan2 of the cross
    product magnitude and dot product to avoid numerical instability near
    parallel/anti-parallel vectors.

    Args:
        v1: First vector in 3D space.
        v2: Second vector in 3D space.

    Returns:
        The angle between the vectors in radians (0 to pi).

    Raises:
        ValueError: If either input vector has zero length (magnitude < EPS_GEOMETRIC).

    References:
        Ericson, C. "Real-Time Collision Detection" (2004), Section 3.3.2, p. 39
    """
    # normalize_vector will raise ValueError if either vector is zero-length.
    v1_dir = normalize_vector(v1)
    v2_dir = normalize_vector(v2)

    dot = v1_dir.dot(v2_dir)
    cross = v1_dir.cross(v2_dir)
    cross_magnitude = cross.norm()

    # atan2 handles all edge cases gracefully (parallel, anti-parallel, perpendicular).
    theta = np.arctan2(cross_magnitude, dot)

    return float(theta)


def compute_vectors_cross_product_magnitude(v1: Vector3, v2: Vector3) -> float:
    """
    Compute the magnitude of the cross product between two normalized vectors.

    This is useful for measuring how "non-parallel" two vectors are. Returns 0
    when vectors are parallel or anti-parallel, and 1 when perpendicular.

    Args:
        v1: First vector in 3D space.
        v2: Second vector in 3D space.

    Returns:
        Magnitude of the cross product between normalized vectors (0 to 1).

    Raises:
        ValueError: If either input vector has zero length (magnitude < EPS_GEOMETRIC).
    """
    v1_dir = normalize_vector(v1)
    v2_dir = normalize_vector(v2)

    return v1_dir.cross(v2_dir).norm()


def compute_vectors_dot_product(v1: Vector3, v2: Vector3) -> float:
    """
    Compute the dot product between two normalized vectors.

    This is useful for measuring how "parallel" two vectors are. Returns 1 when
    vectors are parallel, -1 when anti-parallel, and 0 when perpendicular.

    Args:
        v1: First vector in 3D space.
        v2: Second vector in 3D space.

    Returns:
        Dot product between normalized vectors (-1 to 1).

    Raises:
        ValueError: If either input vector has zero length (magnitude < EPS_GEOMETRIC).
    """
    v1_dir = normalize_vector(v1)
    v2_dir = normalize_vector(v2)

    return v1_dir.dot(v2_dir)


def compute_point_to_line_distance(
    point: Point3, line_point: Point3, line_direction: Direction3
) -> float:
    """
    Compute the perpendicular distance from a point to a line.

    The line is defined by a point on the line and a direction.

    Args:
        point: The point to measure distance from.
        line_point: A point on the line.
        line_direction: Direction of the line (unit vector).

    Returns:
        Perpendicular distance from the point to the line (always non-negative).
    """
    point_to_line = point - line_point

    return point_to_line.cross(line_direction).norm()


def compute_point_to_plane_distance(
    point: Point3, plane_point: Point3, plane_normal: Direction3
) -> float:
    """
    Compute the signed distance from a point to a plane.

    The plane is defined by a point on the plane and a normal direction.
    Positive distance indicates the point is on the side of the plane
    in the direction of the normal.

    Args:
        point: The point to measure distance from.
        plane_point: A point on the plane.
        plane_normal: Normal direction of the plane (unit vector).

    Returns:
        Signed distance from the point to the plane.
    """
    point_to_plane = point - plane_point

    return point_to_plane.dot(plane_normal)


def compute_scalar_triple_product(v1: Vector3, v2: Vector3, v3: Vector3) -> float:
    """
    Compute the scalar triple product v1 dot (v2 cross v3).

    The scalar triple product represents the signed volume of the parallelepiped
    defined by the three vectors. It is zero when the vectors are coplanar.

    Args:
        v1: First vector.
        v2: Second vector.
        v3: Third vector.

    Returns:
        The scalar triple product (can be positive, negative, or zero).
    """
    cross = v2.cross(v3)
    return v1.dot(cross)


def plane_from_three_points(
    a: Point3, b: Point3, c: Point3
) -> tuple[Direction3, float] | None:
    """
    Construct a plane from three 3D points.

    The plane is represented in the form n.x + d = 0, where n is the unit normal
    direction and d is the distance offset.

    Args:
        a: First point defining the plane.
        b: Second point defining the plane.
        c: Third point defining the plane.

    Returns:
        A tuple containing the normal direction and distance `d`, or None if the points
        are degenerate and do not form a unique plane.
    """
    # Compute two edge vectors on the plane.
    v1 = b - a
    v2 = c - a

    # The normal is perpendicular to both edge vectors.
    normal = v1.cross(v2)
    normal_magnitude = normal.norm()

    # If the magnitude of the normal is near zero, the points are collinear.
    if normal_magnitude < EPS_GEOMETRIC:
        return None

    # Normalize the normal vector for a consistent plane representation.
    normal_dir = normal.normalize()

    # Compute d for the plane equation n.x + d = 0 using point a.
    d = -float(np.dot(normal_dir.data, a.data))

    return normal_dir, d


def intersect_two_planes(
    n1: Direction3, d1: float, n2: Direction3, d2: float
) -> tuple[Point3, Direction3] | None:
    """
    Find the 3D line of intersection between two planes.

    The line is represented by a point on the line and a direction. This
    function handles the case where planes may be parallel.

    Args:
        n1: The normal direction of the first plane.
        d1: The distance offset of the first plane.
        n2: The normal direction of the second plane.
        d2: The distance offset of the second plane.

    Returns:
        A tuple containing a point on the line and the line's direction,
        or None if the planes are parallel and have no unique intersection.
    """
    # The direction of the intersection line is perpendicular to both plane normals.
    direction = n1.cross(n2)
    direction_magnitude_squared = direction.squared_norm()

    # If the magnitude is near zero, the normals are parallel, so the planes are.
    if direction_magnitude_squared < EPS_GEOMETRIC * EPS_GEOMETRIC:
        return None

    # Find a specific point on the intersection line. This formula derives from
    # solving the system of linear equations for the two planes.
    point_data = (
        np.cross(d2 * n1.data - d1 * n2.data, direction.data)
        / direction_magnitude_squared
    )
    line_direction = direction.normalize()

    return Point3(point_data), line_direction


def intersect_line_with_vertical_plane(
    line_point: Point3,
    line_direction: Direction3,
    plane_y: float,
) -> Point3 | None:
    """
    Find where a 3D line intersects a vertical plane defined by y = constant.

    This represents the intersection with a side-view plane in vehicle coordinates.

    Args:
        line_point: A point on the 3D line.
        line_direction: The direction of the 3D line.
        plane_y: The Y-coordinate that defines the vertical plane.

    Returns:
        The 3D intersection point, or None if the line is parallel to the plane.
    """
    return intersect_line_with_axis_aligned_plane(
        line_point, line_direction, Axis.Y, plane_y
    )


def intersect_line_with_axis_aligned_plane(
    line_point: Point3,
    line_direction: Direction3,
    axis: Axis,
    coordinate: float,
) -> Point3 | None:
    """
    Find where a 3D line intersects an axis-aligned plane.

    The plane is defined by fixing one coordinate axis to a constant value.
    For example, axis=Axis.Y with coordinate=100 gives the plane y=100
    (side-view plane), while axis=Axis.X with coordinate=0 gives the plane
    x=0 (front-view plane).

    Args:
        line_point: A point on the 3D line.
        line_direction: The direction of the 3D line.
        axis: Which coordinate axis the plane is aligned to (X, Y, or Z).
        coordinate: The constant value along that axis.

    Returns:
        The 3D intersection point, or None if the line is parallel to the plane.
    """
    direction_component = line_direction[axis]

    # If the line's direction along the chosen axis is near zero, it is parallel.
    if abs(direction_component) < EPS_GEOMETRIC:
        return None

    # Solve for t where line_point[axis] + t * direction[axis] = coordinate.
    t = (coordinate - line_point[axis]) / direction_component

    return Point3(line_point.data + t * line_direction.data)


def rotate_vector_rodrigues(v: Vector3, rotvec: Vector3) -> Vector3:
    """
    Rotate a vector by a rotation vector using Rodrigues' formula.

    The rotation vector encodes both axis and angle: its direction is the rotation
    axis and its magnitude is the rotation angle in radians.

    Args:
        v: The 3D vector to rotate.
        rotvec: Rotation vector (axis * angle).

    Returns:
        The rotated vector. Returns a copy of v if the rotation angle is near zero.
    """
    angle = rotvec.norm()
    if angle < EPS_NUMERICAL:
        return v.copy()

    k = rotvec / angle
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Rodrigues' formula: v*cos(a) + (k x v)*sin(a) + k*(k.v)*(1-cos(a))
    return v * cos_a + k.cross(v) * sin_a + k * (k.dot(v) * (1.0 - cos_a))


def rotate_point_about_axis(
    point: Point3, pivot: Point3, axis: Direction3, angle_rad: float
) -> Point3:
    """
    Rotate a point about an arbitrary axis using Rodrigues' rotation formula.

    Args:
        point: Point to rotate.
        pivot: Point on the rotation axis.
        axis: Direction of the rotation axis (unit vector).
        angle_rad: Rotation angle in radians (right-hand rule about axis).

    Returns:
        Rotated point coordinates.
    """
    k = axis
    v = point - pivot

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Rodrigues' formula: v*cos(a) + (k x v)*sin(a) + k*(k.v)*(1-cos(a))
    rotated = v * cos_a + k.cross(v) * sin_a + k * (k.dot(v) * (1.0 - cos_a))
    return pivot + rotated
