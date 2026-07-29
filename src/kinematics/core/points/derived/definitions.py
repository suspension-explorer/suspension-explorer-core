"""Common non-ground derived point calculation functions.

These functions calculate positions of derived points based on other suspension
points. Coupled axle ground-tangency geometry lives in :mod:`.ground`.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from kinematics.core.enums import PointID
from kinematics.core.points.derived.ground import get_wheel_contact_centre
from kinematics.core.points.derived.manager import DerivedPointsSpec
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.primitives.vector_utils.generic import normalize_vector
from kinematics.core.schema.config import WheelConfig


def get_point_along_line(
    positions: dict[PointKey, Any],
    start_point: PointKey,
    end_point: PointKey,
    distance_from_start: float,
) -> Any:
    """Place a point at a fixed distance along a line between two other points."""
    start = positions[start_point]
    line_direction = normalize_vector(positions[end_point] - start)
    return start + line_direction * distance_from_start


def get_axle_midpoint(positions: dict[PointKey, Any]) -> Any:
    """
    Compute the centre point between the inboard and outboard axle positions.

    Args:
        positions: Dictionary containing AXLE_INBOARD and AXLE_OUTBOARD.

    Returns:
        The axle midpoint position.
    """
    p1 = positions[PointID.AXLE_INBOARD]
    p2 = positions[PointID.AXLE_OUTBOARD]
    return p1 + (p2 - p1) / 2


def get_wheel_center(positions: dict[PointKey, Any], wheel_offset: float) -> Any:
    """
    Determine wheel centre from hub face using the ISO/SAE wheel-offset convention.

    Starting at AXLE_OUTBOARD (the hub mounting face), this moves along the
    axle axis toward inboard for positive wheel offset.

    Args:
        positions: Dictionary containing AXLE_INBOARD and AXLE_OUTBOARD.
        wheel_offset: Offset from hub mounting face to wheel centre plane in mm.

    Returns:
        The wheel-centre position.
    """
    p1 = positions[PointID.AXLE_OUTBOARD]  # Hub face.
    p2 = positions[PointID.AXLE_INBOARD]  # Axle inboard point.
    v = normalize_vector(p1 - p2)  # Points outboard from axle inboard to hub face.
    # Positive ISO/SAE offset places the centreline inboard.
    return p1 - v * wheel_offset


def get_wheel_inboard(positions: dict[PointKey, Any], wheel_width: float) -> Any:
    """
    Determine the inboard wheel edge from the centre and total wheel width.

    Args:
        positions: Dictionary containing AXLE_INBOARD and WHEEL_CENTER.
        wheel_width: Total wheel width across its axial dimension.

    Returns:
        The inboard wheel-face position.
    """
    p1 = positions[PointID.AXLE_INBOARD]
    p2 = positions[PointID.WHEEL_CENTER]
    v = normalize_vector(p2 - p1)  # Points outboard from axle inboard.
    return p2 - v * (wheel_width / 2)


def get_wheel_outboard(positions: dict[PointKey, Any], wheel_width: float) -> Any:
    """
    Determine the outboard wheel edge from the centre and total wheel width.

    Args:
        positions: Dictionary containing AXLE_INBOARD and WHEEL_CENTER.
        wheel_width: Total wheel width across its axial dimension.

    Returns:
        The outboard wheel-face position.
    """
    p1 = positions[PointID.WHEEL_CENTER]
    p2 = positions[PointID.AXLE_INBOARD]
    v = normalize_vector(p1 - p2)  # Points outboard from axle inboard.
    return p1 + v * (wheel_width / 2)


def build_wheel_derived_spec(wheel: "WheelConfig") -> "DerivedPointsSpec":
    """
    Build the standard wheel derived-point specification.

    Every corner whose wheel spin axis is AXLE_INBOARD -> AXLE_OUTBOARD derives
    the wheel centre, rim faces, and flat-ground wheel contact centre the same way.
    When both corners are composed, AxleSuspension removes the WHEEL_CONTACT_CENTRE
    entries from the composed derived-point graph entirely and writes both
    contact centres from its post-solve ground closure instead, so no
    per-corner flat-ground result can reach an axle state.
    """
    tire_radius = wheel.tire.nominal_radius
    functions = {
        PointID.AXLE_MIDPOINT: get_axle_midpoint,
        PointID.WHEEL_CENTER: partial(get_wheel_center, wheel_offset=wheel.offset),
        PointID.WHEEL_INBOARD: partial(
            get_wheel_inboard, wheel_width=wheel.tire.section_width
        ),
        PointID.WHEEL_OUTBOARD: partial(
            get_wheel_outboard, wheel_width=wheel.tire.section_width
        ),
        PointID.WHEEL_CONTACT_CENTRE: partial(
            get_wheel_contact_centre, tire_radius=tire_radius
        ),
    }
    dependencies = {
        PointID.AXLE_MIDPOINT: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
        PointID.WHEEL_CENTER: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
        PointID.WHEEL_INBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
        PointID.WHEEL_OUTBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
        PointID.WHEEL_CONTACT_CENTRE: {
            PointID.WHEEL_CENTER,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        },
    }
    return DerivedPointsSpec(functions=functions, dependencies=dependencies)
