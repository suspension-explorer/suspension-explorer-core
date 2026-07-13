"""
Generic vector utility functions.

This module provides fundamental vector operations used throughout the kinematics
system. These operations do not use any types specific to this project, so can be used
in utility contexts without introducing circular dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from kinematics.core.primitives.constants import EPS_GEOMETRIC
from kinematics.core.primitives.geometry import Direction3, Point3, Vector3

if TYPE_CHECKING:
    from kinematics.core.primitives.dual import DualVec3

FloatingT = TypeVar("FloatingT", bound=np.floating)


class LineIntersectionResult(NamedTuple):
    """
    Result of a line-line intersection calculation.

    Attributes:
        point: The intersection point coordinates.
        t1: Parameter t for first line (0 <= t <= 1 means point is on line segment).
        t2: Parameter t for second line (0 <= t <= 1 means point is on line segment).
    """

    point: NDArray[np.float64]
    t1: float
    t2: float


def compute_2d_vector_vector_intersection(
    line1_start: NDArray[np.float64],
    line1_end: NDArray[np.float64],
    line2_start: NDArray[np.float64],
    line2_end: NDArray[np.float64],
    *,
    segments_only: bool = True,
) -> Optional[LineIntersectionResult]:
    """
    Compute the intersection of two 2D line segments or their infinite extensions.

    Returns:
        LineIntersectionResult(point, t1, t2) or None if:
        - Lines are parallel/colinear (no unique intersection)
        - Any line is degenerate (zero length)
        - segments_only=True and the intersection lies outside either segment
    Notes:
        - Parallel test is scale-aware: |den| < EPS_GEOMETRIC * |d1| * |d2|
        - Endpoints are accepted with a small tolerance when segments_only=True
    """
    # Ensure dtype and finiteness.
    p1 = np.asarray(line1_start, dtype=np.float64)
    p2 = np.asarray(line1_end, dtype=np.float64)
    p3 = np.asarray(line2_start, dtype=np.float64)
    p4 = np.asarray(line2_end, dtype=np.float64)

    if not (
        np.isfinite(p1).all()
        and np.isfinite(p2).all()
        and np.isfinite(p3).all()
        and np.isfinite(p4).all()
    ):
        return None

    # Extract coordinates.
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]

    # Direction vectors and lengths.
    d1x = x2 - x1
    d1y = y2 - y1
    d2x = x4 - x3
    d2y = y4 - y3
    len1 = np.hypot(d1x, d1y)
    len2 = np.hypot(d2x, d2y)
    if len1 < EPS_GEOMETRIC or len2 < EPS_GEOMETRIC:
        # Degenerate line segment.
        return None

    # Denominator (2D cross of directions) with scale-aware parallel test.
    denominator = d1x * d2y - d1y * d2x
    if abs(denominator) < (EPS_GEOMETRIC * len1 * len2):
        # Parallel or colinear: no unique intersection point.
        return None

    # Relative position from line1 start to line2 start.
    dx = x3 - x1
    dy = y3 - y1

    # Solve for parameters using cross products.
    # t1 gives intersection along line1; t2 along line2.
    t1 = (dx * d2y - dy * d2x) / denominator
    t2 = (dx * d1y - dy * d1x) / denominator

    # Segment check with endpoint tolerance.
    if segments_only:
        # Tolerance scaled by segment size to catch endpoint hits.
        tol = EPS_GEOMETRIC / max(max(len1, len2), 1.0)
        t1c = min(max(t1, 0.0), 1.0)
        t2c = min(max(t2, 0.0), 1.0)
        if abs(t1 - t1c) > tol or abs(t2 - t2c) > tol:
            return None

    # Intersection point.
    point_x = x1 + t1 * d1x
    point_y = y1 + t1 * d1y
    point = np.array([point_x, point_y], dtype=np.float64)

    return LineIntersectionResult(point=point, t1=float(t1), t2=float(t2))


@overload
def normalize_vector(v: DualVec3) -> DualVec3: ...


@overload
def normalize_vector(v: Vector3) -> Direction3: ...


@overload
def normalize_vector(v: NDArray[FloatingT]) -> NDArray[FloatingT]: ...


def normalize_vector(
    v: NDArray[FloatingT] | DualVec3 | Vector3,
) -> NDArray[FloatingT] | DualVec3 | Direction3:
    """
    Normalize a vector to a unit vector.

    For Vector3 input, returns Direction3.
    For DualVec3 input, returns DualVec3 (preserves derivative tracking).
    For raw ndarray input, returns ndarray.

    Args:
        v: Input vector (Vector3, ndarray, or DualVec3).

    Returns:
        Unit vector in the same direction as the input.

    Raises:
        ValueError: If the input vector has zero length (magnitude < EPS_GEOMETRIC).
    """
    # Import here to avoid circular dependency at module level.
    from kinematics.core.primitives.dual import DualVec3
    from kinematics.core.primitives.dual import norm as dual_norm

    if isinstance(v, DualVec3):
        # Use dual-aware norm to propagate derivatives through the quotient rule.
        n = dual_norm(v)
        if n.val < EPS_GEOMETRIC:
            raise ValueError("Cannot normalize zero-length vector")
        return v / n

    if isinstance(v, Vector3):
        return Direction3(v)

    norm = np.linalg.norm(v)
    if norm < EPS_GEOMETRIC:
        raise ValueError("Cannot normalize zero-length vector")
    return (v / norm).astype(v.dtype)


def project_coordinate(position: Point3, direction: Direction3) -> float:
    """
    Computes the scalar coordinate of a position along a unit direction. This
    represents the signed distance of the position along the given direction.

    Args:
        position: The 3D position (Point3).
        direction: The unit direction (Direction3).

    Returns:
        The scalar projection value.
    """
    return float(np.dot(position.data, direction.data))


def rotate_2d_vector(vector: np.ndarray, angle_radians: float) -> np.ndarray:
    """
    Rotate a 2D vector by the specified angle.

    Args:
        vector: 2D vector [x, y] to rotate
        angle_radians: Rotation angle in radians (positive = counter-clockwise)

    Returns:
        Rotated 2D vector
    """
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)

    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)

    return rotation_matrix @ vector


def perpendicular_2d(vector: np.ndarray, clockwise: bool = False) -> np.ndarray:
    """
    Compute the perpendicular to a 2D vector.

    Args:
        vector: 2D vector [x, y]
        clockwise: If True, rotate clockwise (right-hand perpendicular),
                   if False, rotate anti-clockwise (left-hand perpendicular)

    Returns:
        Perpendicular 2D vector with same magnitude as input.
    """
    if clockwise:
        # 90 degrees clockwise: [x, y] -> [y, -x].
        return np.array([vector[1], -vector[0]], dtype=np.float64)
    else:
        # 90 degrees anti-clockwise: [x, y] -> [-y, x].
        return np.array([-vector[1], vector[0]], dtype=np.float64)
