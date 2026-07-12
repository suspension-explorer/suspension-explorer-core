"""
Common derived point calculation functions.

These functions calculate positions of derived points based on the positions of other
points in the suspension system. They are shared across different suspension types to
avoid code duplication.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kinematics.core.enums import PointID
from kinematics.core.point_ref import PointKey
from kinematics.core.types import WorldAxisSystem
from kinematics.core.vector_utils.generic import normalize_vector


def get_wheel_plane_down_vector(positions: dict[PointKey, Any]) -> Any:
    """
    Calculates the 'down' direction vector in the wheel's plane of rotation.

    This vector is always perpendicular to the axle's direction and is calculated
    by finding the component of the global down vector that is orthogonal to
    the axle vector (using Gram-Schmidt orthogonalization).

    Args:
        positions: Dictionary of point coordinates. Must contain AXLE_INBOARD
                   and AXLE_OUTBOARD.

    Returns:
        A normalized 3D vector representing the 'down' direction in the
        wheel's plane.

    Raises:
        ValueError: If the axle has zero length or the resulting projected
                    down vector is a zero vector (i.e., axle is vertical).
    """
    axle_inboard = positions[PointID.AXLE_INBOARD]
    axle_outboard = positions[PointID.AXLE_OUTBOARD]

    # Compute the normalized axle direction (wheel's spin axis).
    axle_vector = axle_outboard - axle_inboard
    axle_direction = normalize_vector(axle_vector)

    # Find the 'down' direction within the wheel plane (perpendicular to the axle).
    global_down = -1 * WorldAxisSystem.Z

    # Project global down onto the plane perpendicular to the axle. This removes
    # the component of 'down' that is parallel to the axle.
    down_parallel_to_axle = np.dot(global_down, axle_direction) * axle_direction
    wheel_down = global_down - down_parallel_to_axle

    # Normalize to get the final unit vector. This will raise a ValueError if
    # the axle is vertical, which is the correct fail-fast behavior.
    return normalize_vector(wheel_down)


def get_axle_midpoint(positions: dict[PointKey, Any]) -> Any:
    """
    Computes the center point between the inboard and outboard axle positions.

    Args:
        positions: Dictionary mapping point IDs to their 3D coordinates.
                  Must contain AXLE_INBOARD and AXLE_OUTBOARD entries.

    Returns:
        A numpy array representing the 3D coordinates of the axle midpoint.
    """
    p1 = positions[PointID.AXLE_INBOARD]
    p2 = positions[PointID.AXLE_OUTBOARD]
    return p1 + (p2 - p1) / 2


def get_wheel_center(positions: dict[PointKey, Any], wheel_offset: float) -> Any:
    """
    Determine wheel center from hub face using ISO/SAE wheel-offset convention.

    Starting at `AXLE_OUTBOARD` (hub mounting face), this moves along the axle
    axis by `wheel_offset` toward axle inboard for positive values.

    Args:
        positions: Dictionary mapping point IDs to their 3D coordinates.
                Must contain AXLE_INBOARD and AXLE_OUTBOARD entries.
        wheel_offset: Wheel offset (ET) from hub mounting face to wheel center
                  plane in mm. Positive values place the wheel centerline
                  inboard of the hub face; negative values place it outboard.

    Returns:
        A numpy array representing the 3D coordinates of the wheel center.
    """
    p1 = positions[PointID.AXLE_OUTBOARD]  # Hub face.
    p2 = positions[PointID.AXLE_INBOARD]  # Axle inboard point.
    v = p1 - p2  # Points outboard; from inboard to axle outboard (hub face).
    v = normalize_vector(v)

    # ISO/SAE wheel offset convention: positive offset places centerline inboard.
    return p1 - v * wheel_offset


def get_wheel_inboard(positions: dict[PointKey, Any], wheel_width: float) -> Any:
    """
    Determines the inboard edge position of the wheel by moving inward from the wheel
    center by half the wheel width along the axle axis.

    Args:
        positions: Dictionary mapping point IDs to their 3D coordinates.
                Must contain AXLE_INBOARD and WHEEL_CENTER entries.
        wheel_width: Total width of the wheel across its axial dimension.

    Returns:
        A numpy array representing the 3D coordinates of the wheel's inboard lip/edge.
    """
    p1 = positions[PointID.AXLE_INBOARD]
    p2 = positions[PointID.WHEEL_CENTER]
    v = p2 - p1  # Points outboard; from inboard to wheel center.
    v = normalize_vector(v)
    return p2 - v * (wheel_width / 2)


def get_wheel_outboard(positions: dict[PointKey, Any], wheel_width: float) -> Any:
    """
    Determines the outboard edge position of the wheel by moving outward from the wheel
    center by half the wheel width along the axle axis.

    Args:
        positions: Dictionary mapping point IDs to their 3D coordinates.
                Must contain WHEEL_CENTER and AXLE_INBOARD entries.
        wheel_width: Total width of the wheel across its axial dimension.

    Returns:
        A numpy array representing the 3D coordinates of the wheel's outboard lip/edge.
    """
    p1 = positions[PointID.WHEEL_CENTER]
    p2 = positions[PointID.AXLE_INBOARD]
    v = p1 - p2  # Points outboard; from axle inboard to wheel center.
    v = normalize_vector(v)
    return p1 + v * (wheel_width / 2)


def get_contact_patch_center(positions: dict[PointKey, Any], tire_radius: float) -> Any:
    """
    Computes the position of the geometric contact patch center.

    This is the lowest point on an ideal tire circle in the wheel's center
    plane. It is found by moving from the wheel center in the wheel-plane
    'down' direction by a distance equal to the tire radius. Its Z-coordinate
    is not fixed and will move with the suspension.

    Args:
        positions: Dictionary of point coordinates.
        tire_radius: The radius of the tire in mm.

    Returns:
        The 3D coordinates of the geometric contact point.
    """
    wheel_center = positions[PointID.WHEEL_CENTER]
    wheel_down_normalized = get_wheel_plane_down_vector(positions)

    # Calculate the contact point by moving from the wheel center by the radius.
    contact_point = wheel_center + wheel_down_normalized * tire_radius

    return contact_point
